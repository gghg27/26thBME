# -*- coding: utf-8 -*-
"""Build dual-stream train data and window index from raw train_data/*.mat."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.io import savemat

try:
    import mne
    from autoreject import Ransac
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "MNE, autoreject, and mne-icalabel are required for preprocessing. "
        "Install the same preprocessing dependencies used by preprcocess/com_preprocess.py."
    ) from exc

from common import (
    N_CHANNELS,
    SFREQ,
    STEP,
    TRAIN_TRIAL_LEN,
    WIN_LEN,
    diagnosis_label_from_name,
    extract_de_sequence,
    find_train_eeg_arrays,
    label4_from_diag_emotion,
    load_channel_names,
    load_mat_auto,
    natural_key,
    parse_train_subject_info,
    subject_wise_zscore,
    window_starts,
)


def preprocess_train_subject(
    pos_data: np.ndarray,
    neu_data: np.ndarray,
    *,
    sfreq: int = SFREQ,
    ch_name_path: str | Path | None = None,
    ransac_duration: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Clean one subject once, then split abs/rel signals back to pos/neu."""

    pos_data = np.asarray(pos_data, dtype=np.float64)
    neu_data = np.asarray(neu_data, dtype=np.float64)
    if pos_data.ndim != 2 or neu_data.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got pos={pos_data.shape}, neu={neu_data.shape}")
    if pos_data.shape[0] != N_CHANNELS or neu_data.shape[0] != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} channels, got pos={pos_data.shape}, neu={neu_data.shape}")

    neu_len = neu_data.shape[1]
    pos_len = pos_data.shape[1]
    all_data = np.concatenate([neu_data, pos_data], axis=1)

    ch_names = load_channel_names(ch_name_path)
    if len(ch_names) != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} channel names, got {len(ch_names)}")

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    raw = mne.io.RawArray(all_data, info, verbose=False)
    raw.set_montage(montage, on_missing="ignore")

    raw_main = raw.copy().filter(l_freq=0.1, h_freq=45, verbose=False)
    raw_ica = raw.copy().filter(l_freq=1.0, h_freq=45, verbose=False)

    epochs = mne.make_fixed_length_epochs(
        raw_main,
        duration=ransac_duration,
        overlap=0.0,
        preload=True,
        verbose=False,
    )
    ransac = Ransac(n_resample=100, min_channels=0.5, min_corr=0.75, verbose=False)
    try:
        ransac.fit(epochs)
        bads = list(ransac.bad_chs_)
    except Exception as exc:
        print(f"[preprocess] RANSAC failed, continue without bad-channel interpolation: {repr(exc)}")
        bads = []
    print(f"[preprocess] RANSAC bad channels: {bads}")

    raw_main.info["bads"] = bads
    raw_ica.info["bads"] = bads
    if bads:
        raw_main.interpolate_bads(reset_bads=True, verbose=False)
        raw_ica.interpolate_bads(reset_bads=True, verbose=False)

    labels: list[str] = []
    exclude_idx: list[int] = []
    try:
        ica = ICA(n_components=None, random_state=97, method="infomax", verbose=False)
        ica.fit(raw_ica, verbose=False)
        ic_labels = label_components(raw_ica, ica, method="iclabel")
        labels = [str(label) for label in ic_labels["labels"]]
        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        print(f"[preprocess] IC labels: {labels}")
        print(f"[preprocess] Excluding ICA components: {exclude_idx}")
        ica.exclude = exclude_idx
        raw_clean = ica.apply(raw_main.copy(), verbose=False)
        clean_abs_all = raw_clean.get_data().astype(np.float32)
    except Exception as exc:
        print(f"[preprocess] ICA/ICLabel failed, using filtered/interpolated data: {repr(exc)}")
        clean_abs_all = raw_main.get_data().astype(np.float32)

    clean_rel_all, subj_mean, subj_std = subject_wise_zscore(clean_abs_all)
    neu_abs = clean_abs_all[:, :neu_len]
    pos_abs = clean_abs_all[:, neu_len : neu_len + pos_len]
    neu_rel = clean_rel_all[:, :neu_len]
    pos_rel = clean_rel_all[:, neu_len : neu_len + pos_len]

    info_dict = {
        "bad_channels": bads,
        "ica_labels": labels,
        "ica_exclude_idx": [int(idx) for idx in exclude_idx],
        "subject_mean_shape": list(subj_mean.shape),
        "subject_std_shape": list(subj_std.shape),
    }
    return pos_abs, neu_abs, pos_rel, neu_rel, subj_mean, subj_std, info_dict


def load_train_clean_cache(
    clean_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load cached clean abs EEG and rebuild subject-wise rel EEG cheaply."""

    mat = load_mat_auto(clean_path)
    required = ["EEG_data_pos_abs", "EEG_data_neu_abs"]
    missing = [key for key in required if key not in mat]
    if missing:
        raise KeyError(f"{clean_path} is missing cached clean keys: {missing}")

    pos_abs = np.asarray(mat["EEG_data_pos_abs"], dtype=np.float32)
    neu_abs = np.asarray(mat["EEG_data_neu_abs"], dtype=np.float32)
    if pos_abs.shape[0] != N_CHANNELS or neu_abs.shape[0] != N_CHANNELS:
        raise ValueError(f"{clean_path}: bad cached shapes pos={pos_abs.shape}, neu={neu_abs.shape}")

    clean_all = np.concatenate([neu_abs, pos_abs], axis=1)
    clean_rel_all, subj_mean, subj_std = subject_wise_zscore(clean_all)
    neu_len = neu_abs.shape[1]
    pos_len = pos_abs.shape[1]
    neu_rel = clean_rel_all[:, :neu_len]
    pos_rel = clean_rel_all[:, neu_len : neu_len + pos_len]
    info_dict = {
        "loaded_from_clean_cache": True,
        "cache_path": str(clean_path),
        "subject_mean_shape": list(subj_mean.shape),
        "subject_std_shape": list(subj_std.shape),
    }
    return pos_abs, neu_abs, pos_rel, neu_rel, subj_mean, subj_std, info_dict


def _save_trial_pair(
    *,
    records: list[dict],
    trial_abs: np.ndarray,
    trial_rel: np.ndarray,
    diagnosis: str,
    subject_id: int,
    file_name: str,
    emotion: str,
    emotion_label: int,
    trial_id: int,
    trial_offset_start: int,
    trial_abs_root: Path,
    trial_rel_root: Path,
    de_abs_root: Path,
    smooth_kernel: int,
) -> None:
    diag_label = diagnosis_label_from_name(diagnosis)
    label4 = label4_from_diag_emotion(diagnosis, emotion_label)
    global_subject_id = subject_id if diagnosis.upper() == "DEP" else 1000 + subject_id
    stem = f"{diagnosis}{subject_id}_trial{trial_id:02d}_{emotion}"
    trial_path_abs = trial_abs_root / f"{stem}_abs.npy"
    trial_path_rel = trial_rel_root / f"{stem}_rel.npy"
    de_path_abs = de_abs_root / f"{stem}_de_abs.npy"

    np.save(trial_path_abs, trial_abs.astype(np.float32))
    np.save(trial_path_rel, trial_rel.astype(np.float32))
    de_seq = extract_de_sequence(
        trial_abs,
        sfreq=SFREQ,
        win_len=WIN_LEN,
        step=STEP,
        smooth_kernel=smooth_kernel,
    )
    np.save(de_path_abs, de_seq.astype(np.float32))

    starts = window_starts(trial_abs.shape[-1], WIN_LEN, STEP)
    if len(starts) != de_seq.shape[0]:
        raise ValueError(f"{stem}: windows={len(starts)} but de_seq={de_seq.shape}")

    for win_id, start in enumerate(starts):
        end = start + WIN_LEN
        records.append(
            {
                "subject_id": int(subject_id),
                "subject_number": int(subject_id),
                "global_subject_id": int(global_subject_id),
                "file_name": file_name,
                "diagnosis": diagnosis,
                "diagnosis_label": int(diag_label),
                "emotion": emotion,
                "emotion_label": int(emotion_label),
                "label4": int(label4),
                "trial_id": int(trial_id),
                "trial_path_abs": str(trial_path_abs),
                "trial_path_rel": str(trial_path_rel),
                "trial_path": str(trial_path_abs),
                "start": int(start),
                "end": int(end),
                "trial_start": int(trial_offset_start),
                "trial_end": int(trial_offset_start + trial_abs.shape[-1]),
                "de_path_abs": str(de_path_abs),
                "de_path": str(de_path_abs),
                "de_win_id": int(win_id),
            }
        )

    print(
        f"[trial] {stem}: abs={trial_path_abs.exists()} rel={trial_path_rel.exists()} "
        f"trial_shape={trial_abs.shape} de_shape={de_seq.shape}"
    )


def build_dual_stream_train_index(
    mat_root: str | Path = "train_data",
    save_clean_root: str | Path = "data/com_dual_stream_clean_10s",
    save_trial_abs_root: str | Path = "data/com_dual_stream_split_abs_10s",
    save_trial_rel_root: str | Path = "data/com_dual_stream_split_rel_10s",
    save_de_abs_root: str | Path = "data/com_dual_stream_de_abs_10s",
    out_csv: str | Path = "data/com_dual_stream_window_index_10s.csv",
    ch_name_path: str | Path | None = "ch_name.mat",
    smooth_kernel: int = 3,
    force_clean: bool = False,
) -> pd.DataFrame:
    mat_root = Path(mat_root)
    save_clean_root = Path(save_clean_root)
    trial_abs_root = Path(save_trial_abs_root)
    trial_rel_root = Path(save_trial_rel_root)
    de_abs_root = Path(save_de_abs_root)
    out_csv = Path(out_csv)

    for path in [save_clean_root, trial_abs_root, trial_rel_root, de_abs_root, out_csv.parent]:
        path.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(mat_root.glob("*.mat"), key=natural_key)
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {mat_root}")
    print(f"[train preprocess] found {len(mat_files)} train mat files in {mat_root}")

    records: list[dict] = []
    preprocess_infos: dict[str, dict] = {}

    for mat_path in mat_files:
        print("\n" + "=" * 80)
        print(f"[train preprocess] processing {mat_path}")
        diagnosis, subject_id = parse_train_subject_info(mat_path)
        clean_path = save_clean_root / f"{diagnosis}_{subject_id}_dual_stream_clean.mat"
        if clean_path.exists() and not force_clean:
            print(f"[clean cache] hit {clean_path}; skip filter/RANSAC/ICA and rebuild rel/index.")
            pos_abs, neu_abs, pos_rel, neu_rel, subj_mean, subj_std, info_dict = load_train_clean_cache(clean_path)
        else:
            if clean_path.exists() and force_clean:
                print(f"[clean cache] force_clean=True; rebuild {clean_path}")
            mat = load_mat_auto(mat_path)
            pos_raw, neu_raw = find_train_eeg_arrays(mat)
            print(
                f"[raw] diagnosis={diagnosis} subject_id={subject_id} "
                f"pos={pos_raw.shape} neu={neu_raw.shape}"
            )
            if pos_raw.shape[-1] != 4 * TRAIN_TRIAL_LEN or neu_raw.shape[-1] != 4 * TRAIN_TRIAL_LEN:
                raise ValueError(
                    f"{mat_path.name}: expected pos/neu length {4 * TRAIN_TRIAL_LEN}, "
                    f"got pos={pos_raw.shape}, neu={neu_raw.shape}"
                )
            pos_abs, neu_abs, pos_rel, neu_rel, subj_mean, subj_std, info_dict = preprocess_train_subject(
                pos_raw,
                neu_raw,
                sfreq=SFREQ,
                ch_name_path=ch_name_path,
            )
        preprocess_infos[f"{diagnosis}_{subject_id}"] = info_dict

        savemat(
            clean_path,
            {
                "EEG_data_pos_abs": pos_abs,
                "EEG_data_neu_abs": neu_abs,
                "EEG_data_pos_rel": pos_rel,
                "EEG_data_neu_rel": neu_rel,
                "subject_mean": subj_mean,
                "subject_std": subj_std,
            },
        )
        print(f"[clean] saved {clean_path}")

        for idx in range(4):
            start = idx * TRAIN_TRIAL_LEN
            end = (idx + 1) * TRAIN_TRIAL_LEN
            _save_trial_pair(
                records=records,
                trial_abs=neu_abs[:, start:end],
                trial_rel=neu_rel[:, start:end],
                diagnosis=diagnosis,
                subject_id=subject_id,
                file_name=mat_path.name,
                emotion="neu",
                emotion_label=0,
                trial_id=idx + 1,
                trial_offset_start=start,
                trial_abs_root=trial_abs_root,
                trial_rel_root=trial_rel_root,
                de_abs_root=de_abs_root,
                smooth_kernel=smooth_kernel,
            )

        for idx in range(4):
            start = idx * TRAIN_TRIAL_LEN
            end = (idx + 1) * TRAIN_TRIAL_LEN
            _save_trial_pair(
                records=records,
                trial_abs=pos_abs[:, start:end],
                trial_rel=pos_rel[:, start:end],
                diagnosis=diagnosis,
                subject_id=subject_id,
                file_name=mat_path.name,
                emotion="pos",
                emotion_label=1,
                trial_id=idx + 5,
                trial_offset_start=start,
                trial_abs_root=trial_abs_root,
                trial_rel_root=trial_rel_root,
                de_abs_root=de_abs_root,
                smooth_kernel=smooth_kernel,
            )

    df = pd.DataFrame(records)
    df["subject_id"] = df["global_subject_id"].astype(int)
    df = df.sort_values(["subject_id", "trial_id", "de_win_id"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    info_path = save_clean_root / "dual_stream_train_preprocess_info.json"
    info_path.write_text(json.dumps(preprocess_infos, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"[train preprocess] window index saved: {out_csv}")
    print(f"[train preprocess] preprocess info saved: {info_path}")
    print(f"[train preprocess] subjects={df['subject_id'].nunique()} trials={df[['subject_id', 'trial_id']].drop_duplicates().shape[0]} windows={len(df)}")
    print("[train preprocess] diagnosis label counts (0=DEP, 1=HC):")
    print(df.drop_duplicates("subject_id")["diagnosis_label"].value_counts().sort_index())
    print("[train preprocess] emotion label counts:")
    print(df[["subject_id", "trial_id", "emotion_label"]].drop_duplicates()["emotion_label"].value_counts().sort_index())
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V6 dual-stream train data/index.")
    parser.add_argument("--mat_root", type=str, default="train_data")
    parser.add_argument("--save_clean_root", type=str, default="data/com_dual_stream_clean_10s")
    parser.add_argument("--save_trial_abs_root", type=str, default="data/com_dual_stream_split_abs_10s")
    parser.add_argument("--save_trial_rel_root", type=str, default="data/com_dual_stream_split_rel_10s")
    parser.add_argument("--save_de_abs_root", type=str, default="data/com_dual_stream_de_abs_10s")
    parser.add_argument("--out_csv", type=str, default="data/com_dual_stream_window_index_10s.csv")
    parser.add_argument("--ch_name_path", type=str, default="ch_name.mat")
    parser.add_argument("--smooth_kernel", type=int, default=3)
    parser.add_argument(
        "--force_clean",
        action="store_true",
        help="Re-run filter/RANSAC/ICA even if cached clean abs files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dual_stream_train_index(**vars(args))


if __name__ == "__main__":
    main()

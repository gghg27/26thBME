# -*- coding: utf-8 -*-
"""Build dual-stream test data and window index from raw test_data/*.mat."""

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
        "Install the same preprocessing dependencies used by preprcocess/preprocess_test.py."
    ) from exc

from common import (
    N_CHANNELS,
    N_TRIALS,
    SFREQ,
    STEP,
    TEST_TRIAL_LEN,
    WIN_LEN,
    extract_de_sequence,
    find_test_eeg_array,
    load_channel_names,
    load_mat_auto,
    natural_key,
    parse_test_subject_number,
    parse_test_user_id,
    subject_wise_zscore,
    window_starts,
)


def preprocess_test_subject(
    test_data: np.ndarray,
    *,
    sfreq: int = SFREQ,
    ch_name_path: str | Path | None = None,
    ransac_duration: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Clean a full test subject once and create abs/rel signals."""

    test_data = np.asarray(test_data, dtype=np.float64)
    if test_data.ndim != 2 or test_data.shape[0] != N_CHANNELS:
        raise ValueError(f"Expected test_data [{N_CHANNELS}, T], got {test_data.shape}")

    ch_names = load_channel_names(ch_name_path)
    if len(ch_names) != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} channel names, got {len(ch_names)}")

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    raw = mne.io.RawArray(test_data, info, verbose=False)
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
        clean_abs = raw_clean.get_data().astype(np.float32)
    except Exception as exc:
        print(f"[preprocess] ICA/ICLabel failed, using filtered/interpolated data: {repr(exc)}")
        clean_abs = raw_main.get_data().astype(np.float32)

    clean_rel, subj_mean, subj_std = subject_wise_zscore(clean_abs)
    info_dict = {
        "bad_channels": bads,
        "ica_labels": labels,
        "ica_exclude_idx": [int(idx) for idx in exclude_idx],
        "subject_mean_shape": list(subj_mean.shape),
        "subject_std_shape": list(subj_std.shape),
    }
    return clean_abs, clean_rel, subj_mean, subj_std, info_dict


def load_test_clean_cache(clean_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load cached clean abs EEG and rebuild subject-wise rel EEG cheaply."""

    mat = load_mat_auto(clean_path)
    if "test_eeg_c_abs" not in mat:
        raise KeyError(f"{clean_path} is missing cached key test_eeg_c_abs")
    clean_abs = np.asarray(mat["test_eeg_c_abs"], dtype=np.float32)
    if clean_abs.shape[0] != N_CHANNELS:
        raise ValueError(f"{clean_path}: bad cached shape clean_abs={clean_abs.shape}")
    clean_rel, subj_mean, subj_std = subject_wise_zscore(clean_abs)
    info_dict = {
        "loaded_from_clean_cache": True,
        "cache_path": str(clean_path),
        "subject_mean_shape": list(subj_mean.shape),
        "subject_std_shape": list(subj_std.shape),
    }
    return clean_abs, clean_rel, subj_mean, subj_std, info_dict


def build_dual_stream_test_index(
    test_mat_root: str | Path = "test_data",
    save_clean_root: str | Path = "data/com_test_dual_stream_clean_10s",
    save_trial_abs_root: str | Path = "data/com_test_dual_stream_split_abs_10s",
    save_trial_rel_root: str | Path = "data/com_test_dual_stream_split_rel_10s",
    save_de_abs_root: str | Path = "data/com_test_dual_stream_de_abs_10s",
    out_csv: str | Path = "data/com_test_dual_stream_window_index_10s.csv",
    ch_name_path: str | Path | None = "ch_name.mat",
    smooth_kernel: int = 3,
    force_clean: bool = False,
) -> pd.DataFrame:
    test_mat_root = Path(test_mat_root)
    save_clean_root = Path(save_clean_root)
    trial_abs_root = Path(save_trial_abs_root)
    trial_rel_root = Path(save_trial_rel_root)
    de_abs_root = Path(save_de_abs_root)
    out_csv = Path(out_csv)

    for path in [save_clean_root, trial_abs_root, trial_rel_root, de_abs_root, out_csv.parent]:
        path.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(test_mat_root.glob("*.mat"), key=natural_key)
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {test_mat_root}")
    print(f"[test preprocess] found {len(mat_files)} test mat files in {test_mat_root}")

    records: list[dict] = []
    preprocess_infos: dict[str, dict] = {}
    expected_len = N_TRIALS * TEST_TRIAL_LEN

    for mat_path in mat_files:
        print("\n" + "=" * 80)
        print(f"[test preprocess] processing {mat_path}")
        user_id = parse_test_user_id(mat_path)
        subject_number = parse_test_subject_number(user_id)
        clean_path = save_clean_root / f"{user_id}_dual_stream_clean.mat"
        if clean_path.exists() and not force_clean:
            print(f"[clean cache] hit {clean_path}; skip filter/RANSAC/ICA and rebuild rel/index.")
            clean_abs, clean_rel, subj_mean, subj_std, info_dict = load_test_clean_cache(clean_path)
        else:
            if clean_path.exists() and force_clean:
                print(f"[clean cache] force_clean=True; rebuild {clean_path}")
            mat = load_mat_auto(mat_path)
            eeg = find_test_eeg_array(mat)
            print(f"[raw] user_id={user_id} subject_number={subject_number} eeg={eeg.shape}")
            if eeg.shape[-1] != expected_len:
                raise ValueError(f"{user_id}: expected length {expected_len}, got shape={eeg.shape}")
            clean_abs, clean_rel, subj_mean, subj_std, info_dict = preprocess_test_subject(
                eeg,
                sfreq=SFREQ,
                ch_name_path=ch_name_path,
            )
        preprocess_infos[user_id] = info_dict

        savemat(
            clean_path,
            {
                "test_eeg_c_abs": clean_abs,
                "test_eeg_c_rel": clean_rel,
                "subject_mean": subj_mean,
                "subject_std": subj_std,
            },
        )
        print(f"[clean] saved {clean_path}")

        for idx in range(N_TRIALS):
            trial_id = idx + 1
            start_all = idx * TEST_TRIAL_LEN
            end_all = (idx + 1) * TEST_TRIAL_LEN
            trial_abs = clean_abs[:, start_all:end_all]
            trial_rel = clean_rel[:, start_all:end_all]
            stem = f"{user_id}_trial{trial_id:02d}"
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
            starts = window_starts(TEST_TRIAL_LEN, WIN_LEN, STEP)
            if len(starts) != de_seq.shape[0]:
                raise ValueError(f"{stem}: windows={len(starts)} but de_seq={de_seq.shape}")

            for win_id, win_start in enumerate(starts):
                records.append(
                    {
                        "user_id": user_id,
                        "subject_number": int(subject_number),
                        "file_name": mat_path.name,
                        "trial_id": int(trial_id),
                        "trial_path_abs": str(trial_path_abs),
                        "trial_path_rel": str(trial_path_rel),
                        "trial_path": str(trial_path_abs),
                        "start": int(win_start),
                        "end": int(win_start + WIN_LEN),
                        "trial_start": int(start_all),
                        "trial_end": int(end_all),
                        "de_path_abs": str(de_path_abs),
                        "de_path": str(de_path_abs),
                        "de_win_id": int(win_id),
                        "n_windows": len(starts),
                        "diagnosis": "TEST",
                        "diagnosis_label": -1,
                        "emotion": "unknown",
                        "emotion_label": -1,
                        "label4": -1,
                    }
                )

            print(
                f"[trial] {stem}: abs={trial_path_abs.exists()} rel={trial_path_rel.exists()} "
                f"trial_shape={trial_abs.shape} de_shape={de_seq.shape}"
            )

    df = pd.DataFrame(records).sort_values(["subject_number", "trial_id", "de_win_id"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    info_path = save_clean_root / "dual_stream_test_preprocess_info.json"
    info_path.write_text(json.dumps(preprocess_infos, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"[test preprocess] window index saved: {out_csv}")
    print(f"[test preprocess] preprocess info saved: {info_path}")
    print(f"[test preprocess] users={df['user_id'].nunique()} trials={df[['user_id', 'trial_id']].drop_duplicates().shape[0]} windows={len(df)}")
    print("[test preprocess] windows per trial:")
    print(df.groupby(["user_id", "trial_id"]).size().value_counts().sort_index())
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V6 dual-stream test data/index.")
    parser.add_argument("--test_mat_root", type=str, default="test_data")
    parser.add_argument("--save_clean_root", type=str, default="data/com_test_dual_stream_clean_10s")
    parser.add_argument("--save_trial_abs_root", type=str, default="data/com_test_dual_stream_split_abs_10s")
    parser.add_argument("--save_trial_rel_root", type=str, default="data/com_test_dual_stream_split_rel_10s")
    parser.add_argument("--save_de_abs_root", type=str, default="data/com_test_dual_stream_de_abs_10s")
    parser.add_argument("--out_csv", type=str, default="data/com_test_dual_stream_window_index_10s.csv")
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
    build_dual_stream_test_index(**vars(args))


if __name__ == "__main__":
    main()

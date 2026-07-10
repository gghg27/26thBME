from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from autoreject import Ransac
from mne.preprocessing import ICA
from mne_icalabel import label_components
from scipy.io import savemat

from preprcocess.com_preprocess import (
    extract_trial_de_sequence,
    load_channel_names,
    load_mat_auto,
)
from preprcocess.preprocess_test import find_test_eeg_array, parse_test_subject_number


SFREQ = 250
WIN_LEN = 500
STEP = 250
SMOOTH_KERNEL = 3
N_TRIALS = 8
TEST_TRIAL_LEN = 2500


def _clean_eeg_once(data: np.ndarray, *, ch_name_path: str, sfreq: int = SFREQ):
    """The shared filter -> RANSAC -> interpolation -> ICA/ICLabel pipeline."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != 30:
        raise ValueError(f"Expected [30,T] EEG, got {data.shape}")
    ch_names = load_channel_names(ch_name_path)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    raw_main = raw.copy().filter(l_freq=0.1, h_freq=45, verbose=False)
    raw_ica = raw.copy().filter(l_freq=1.0, h_freq=45, verbose=False)
    epochs = mne.make_fixed_length_epochs(
        raw_main, duration=2.0, overlap=0.0, preload=True, verbose=False
    )
    ransac = Ransac(n_resample=100, min_channels=0.5, min_corr=0.75, verbose=False)
    ransac.fit(epochs)
    bads = list(ransac.bad_chs_)
    raw_main.info["bads"] = bads
    raw_ica.info["bads"] = bads
    raw_main.interpolate_bads(reset_bads=True, verbose=False)
    raw_ica.interpolate_bads(reset_bads=True, verbose=False)
    ica = ICA(n_components=None, random_state=97, method="infomax", verbose=False)
    ica.fit(raw_ica, verbose=False)
    labels = list(label_components(raw_ica, ica, method="iclabel")["labels"])
    exclude = [idx for idx, label in enumerate(labels) if label not in ("brain", "other")]
    ica.exclude = exclude
    clean_all = ica.apply(raw_main.copy(), verbose=False).get_data()
    return clean_all.astype(np.float32), {
        "bad_channels": bads,
        "ica_labels": labels,
        "ica_exclude_idx": exclude,
    }


def make_dual_views(clean_all: np.ndarray, eps: float = 1e-6):
    clean_abs = np.asarray(clean_all, dtype=np.float32)
    subject_mean = clean_abs.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    subject_std = clean_abs.std(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    safe_std = np.maximum(subject_std, np.float32(eps))
    clean_rel = ((clean_abs - subject_mean) / safe_std).astype(np.float32)
    stats = {
        "subject_mean": subject_mean,
        "subject_std": subject_std,
        "channel_std_min": np.float32(subject_std.min()),
        "channel_std_max": np.float32(subject_std.max()),
    }
    if not np.isfinite(clean_abs).all() or not np.isfinite(clean_rel).all():
        raise FloatingPointError("Dual views contain NaN/Inf")
    return clean_abs, clean_rel, stats


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _save_trial_pair(
    trial_abs: np.ndarray,
    trial_rel: np.ndarray,
    name: str,
    roots: dict[str, Path],
    project_root: Path,
    win_len: int,
    step: int,
):
    trial_abs_path = roots["trial_abs"] / name
    trial_rel_path = roots["trial_rel"] / name
    de_name = name.replace(".npy", "_de.npy")
    de_abs_path = roots["de_abs"] / de_name
    de_rel_path = roots["de_rel"] / de_name
    np.save(trial_abs_path, trial_abs.astype(np.float32))
    np.save(trial_rel_path, trial_rel.astype(np.float32))
    de_abs = extract_trial_de_sequence(
        trial_abs, sfreq=SFREQ, win_len=win_len, step=step, smooth_kernel=SMOOTH_KERNEL
    )
    de_rel = extract_trial_de_sequence(
        trial_rel, sfreq=SFREQ, win_len=win_len, step=step, smooth_kernel=SMOOTH_KERNEL
    )
    if de_abs.shape != de_rel.shape:
        raise AssertionError(f"DE shape mismatch for {name}: {de_abs.shape} vs {de_rel.shape}")
    np.save(de_abs_path, de_abs.astype(np.float32))
    np.save(de_rel_path, de_rel.astype(np.float32))
    return {
        "trial_path_abs": _relative_path(trial_abs_path, project_root),
        "trial_path_rel": _relative_path(trial_rel_path, project_root),
        "de_path_abs": _relative_path(de_abs_path, project_root),
        "de_path_rel": _relative_path(de_rel_path, project_root),
        "n_windows": int(de_abs.shape[0]),
        "de_shape": list(de_abs.shape),
    }


def _make_roots(data_root: Path, test: bool = False) -> dict[str, Path]:
    prefix = "com_test_" if test else "com_"
    roots = {
        "trial_abs": data_root / f"{prefix}split_data_subject_abs_2s",
        "trial_rel": data_root / f"{prefix}split_data_subject_rel_2s",
        "de_abs": data_root / f"{prefix}de_features_abs_2s",
        "de_rel": data_root / f"{prefix}de_features_rel_2s",
        "stats": data_root / f"{prefix}subject_stats_dual",
        "clean": data_root / f"{prefix}clean_dual",
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    return roots


def _train_identity(path: Path) -> tuple[str, int]:
    match = re.match(r"^(DEP|HC)(\d+)timedata\.mat$", path.name, re.IGNORECASE)
    if match is None:
        match = re.match(r"^(DEP|HC)_(\d+)\.mat$", path.name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot parse train subject identity: {path.name}")
    return match.group(1).upper(), int(match.group(2))


def build_train_dual(
    raw_root: Path,
    data_root: Path,
    out_csv: Path,
    ch_name_path: str,
    project_root: Path,
    eps: float,
    win_len: int,
    step: int,
):
    roots = _make_roots(data_root, test=False)
    records = []
    for mat_path_text in sorted(glob.glob(str(raw_root / "*.mat"))):
        mat_path = Path(mat_path_text)
        diagnosis, subject_id = _train_identity(mat_path)
        mat = load_mat_auto(str(mat_path))
        pos = np.asarray(mat["EEG_data_pos"], dtype=np.float32)
        neu = np.asarray(mat["EEG_data_neu"], dtype=np.float32)
        pos_len = pos.shape[1]
        clean_all, clean_info = _clean_eeg_once(
            np.concatenate([pos, neu], axis=1), ch_name_path=ch_name_path
        )
        clean_abs, clean_rel, stats = make_dual_views(clean_all, eps=eps)
        pos_abs, neu_abs = clean_abs[:, :pos_len], clean_abs[:, pos_len:]
        pos_rel, neu_rel = clean_rel[:, :pos_len], clean_rel[:, pos_len:]
        stats_path = roots["stats"] / f"{diagnosis}{subject_id}_stats.npz"
        np.savez(stats_path, **stats)
        savemat(
            roots["clean"] / f"{diagnosis}{subject_id}_dual.mat",
            {"EEG_data_pos_abs": pos_abs, "EEG_data_neu_abs": neu_abs,
             "EEG_data_pos_rel": pos_rel, "EEG_data_neu_rel": neu_rel,
             "subject_mean": stats["subject_mean"], "subject_std": stats["subject_std"]},
        )
        trial_len = neu.shape[1] // 4
        for trial_id in range(1, 9):
            emotion = "neu" if trial_id <= 4 else "pos"
            source_id = trial_id - 1 if emotion == "neu" else trial_id - 5
            start_trial, end_trial = source_id * trial_len, (source_id + 1) * trial_len
            source_abs = neu_abs if emotion == "neu" else pos_abs
            source_rel = neu_rel if emotion == "neu" else pos_rel
            name = f"{diagnosis}{subject_id}_trial{trial_id:02d}_{emotion}.npy"
            paths = _save_trial_pair(
                source_abs[:, start_trial:end_trial], source_rel[:, start_trial:end_trial],
                name, roots, project_root, win_len, step,
            )
            diagnosis_label = 0 if diagnosis == "DEP" else 1
            emotion_label = 0 if emotion == "neu" else 1
            label4 = diagnosis_label * 2 + emotion_label
            for win_id in range(paths["n_windows"]):
                start = win_id * step
                records.append({
                    "subject_id": subject_id, "file_name": mat_path.name,
                    "diagnosis": diagnosis, "diagnosis_label": diagnosis_label,
                    "emotion": emotion, "emotion_label": emotion_label,
                    "label4": label4, "trial_id": trial_id, **{k: paths[k] for k in (
                        "trial_path_abs", "trial_path_rel", "de_path_abs", "de_path_rel")},
                    "start": start, "end": start + win_len, "de_win_id": win_id,
                })
        print(
            f"[train dual] {mat_path.name}: mean={stats['subject_mean'].shape}, "
            f"std_min/max={stats['channel_std_min']:.6g}/{stats['channel_std_max']:.6g}, "
            f"bads={clean_info['bad_channels']}"
        )
    df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[train dual] index={out_csv}, rows={len(df)}, window={win_len}, step={step}")


def build_test_dual(
    raw_root: Path,
    data_root: Path,
    out_csv: Path,
    ch_name_path: str,
    project_root: Path,
    eps: float,
    win_len: int,
    step: int,
):
    roots = _make_roots(data_root, test=True)
    records, infos = [], {}
    for mat_path_text in sorted(glob.glob(str(raw_root / "P_test*.mat"))):
        mat_path = Path(mat_path_text)
        user_id = mat_path.stem
        subject_number = parse_test_subject_number(user_id)
        eeg = find_test_eeg_array(load_mat_auto(str(mat_path)))
        if eeg.shape[1] != N_TRIALS * TEST_TRIAL_LEN:
            raise ValueError(f"{user_id}: expected 20000 samples, got {eeg.shape[1]}")
        clean, clean_info = _clean_eeg_once(eeg, ch_name_path=ch_name_path)
        clean_abs, clean_rel, stats = make_dual_views(clean, eps=eps)
        np.savez(roots["stats"] / f"{user_id}_stats.npz", **stats)
        infos[user_id] = clean_info
        for trial_id in range(1, 9):
            s, e = (trial_id - 1) * TEST_TRIAL_LEN, trial_id * TEST_TRIAL_LEN
            name = f"{user_id}_trial{trial_id:02d}.npy"
            paths = _save_trial_pair(
                clean_abs[:, s:e], clean_rel[:, s:e], name, roots, project_root, win_len, step
            )
            for win_id in range(paths["n_windows"]):
                start = win_id * step
                records.append({
                    "user_id": user_id, "subject_id": subject_number,
                    "subject_number": subject_number, "file_name": mat_path.name,
                    "diagnosis": "TEST", "diagnosis_label": -1,
                    "emotion": "unknown", "emotion_label": -1, "label4": -1,
                    "trial_id": trial_id, **{k: paths[k] for k in (
                        "trial_path_abs", "trial_path_rel", "de_path_abs", "de_path_rel")},
                    "start": start, "end": start + win_len, "de_win_id": win_id,
                })
        print(f"[test dual] {user_id}: 8 trials, subject stats over all trials")
    df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    (roots["stats"] / "cleaning_info.json").write_text(
        json.dumps(infos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[test dual] index={out_csv}, rows={len(df)}, window={win_len}, step={step}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "test"))
    parser.add_argument("--raw_root", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--out_csv", type=Path)
    parser.add_argument("--ch_name_path", default="ch_name.mat")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--win_len", type=int, default=WIN_LEN)
    parser.add_argument("--step", type=int, default=STEP)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    out_csv = args.out_csv or args.data_root / (
        "com_index_dual_2s.csv" if args.mode == "train" else "com_test_index_dual_2s.csv"
    )
    builder = build_train_dual if args.mode == "train" else build_test_dual
    builder(
        args.raw_root, args.data_root, out_csv, args.ch_name_path,
        project_root, args.eps, args.win_len, args.step,
    )


if __name__ == "__main__":
    main()

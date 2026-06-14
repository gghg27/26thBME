# -*- coding: utf-8 -*-
"""Shared utilities for the first dual-stream subject-level pipeline."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]

SFREQ = 250
WIN_LEN = 2500
STEP = 2500
TRAIN_TRIAL_LEN = 12500
TEST_TRIAL_LEN = 2500
N_TRIALS = 8
N_CHANNELS = 30

DE_BANDS = [
    (1.0, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 45.0),
]


def natural_key(value: Any) -> list[Any]:
    parts = re.split(r"(\d+)", str(value))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def resolve_path(path_value: str | Path, root: str | Path = ROOT) -> Path:
    path_text = str(path_value).replace("\\", "/")
    path = Path(path_text)
    if not path.is_absolute():
        path = Path(root) / path
    candidates = [path]
    if not path.exists():
        cleaned = path_text
        while cleaned.startswith("../"):
            cleaned = cleaned[3:]
        candidates.append(Path(root) / cleaned)
        if cleaned.startswith("data/"):
            candidates.append(Path(root) / cleaned[len("data/") :])
        candidates.append(Path(root) / "data" / Path(cleaned).name)
        candidates.append(Path(root) / "test_data" / Path(cleaned).name)
        candidates.append(Path(root) / "train_data" / Path(cleaned).name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "File not found. Tried: " + ", ".join(str(candidate) for candidate in candidates)
    )


def load_mat_auto(mat_path: str | Path) -> dict:
    mat_path = Path(mat_path)
    try:
        return loadmat(mat_path)
    except NotImplementedError:
        data: dict[str, np.ndarray] = {}
        with h5py.File(mat_path, "r") as f:
            for key in f.keys():
                arr = np.array(f[key])
                if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                    arr = arr.T
                data[key] = arr
        return data


def _as_channel_first(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    if arr.shape[0] == N_CHANNELS:
        return arr.astype(np.float32, copy=False)
    if arr.shape[1] == N_CHANNELS:
        return arr.T.astype(np.float32, copy=False)
    raise ValueError(f"{name} must contain {N_CHANNELS} channels, got shape={arr.shape}")


def find_train_eeg_arrays(mat: dict) -> tuple[np.ndarray, np.ndarray]:
    if "EEG_data_pos" not in mat or "EEG_data_neu" not in mat:
        keys = [key for key in mat.keys() if not str(key).startswith("__")]
        raise KeyError(f"Expected EEG_data_pos and EEG_data_neu in train mat, got keys={keys}")
    pos = _as_channel_first(mat["EEG_data_pos"], "EEG_data_pos")
    neu = _as_channel_first(mat["EEG_data_neu"], "EEG_data_neu")
    return pos, neu


def find_test_eeg_array(mat: dict) -> np.ndarray:
    if "test_eeg_c" in mat:
        return _as_channel_first(mat["test_eeg_c"], "test_eeg_c")

    candidates: list[tuple[str, np.ndarray]] = []
    for key, value in mat.items():
        if str(key).startswith("__"):
            continue
        arr = np.asarray(value)
        if arr.ndim == 2 and (arr.shape[0] == N_CHANNELS or arr.shape[1] == N_CHANNELS):
            candidates.append((str(key), arr))
    if not candidates:
        keys = [key for key in mat.keys() if not str(key).startswith("__")]
        raise KeyError(f"No 2D EEG array with {N_CHANNELS} channels found, keys={keys}")
    if len(candidates) > 1:
        print(f"[test mat] multiple EEG candidates found: {[name for name, _ in candidates]}; using {candidates[0][0]}")
    name, arr = candidates[0]
    return _as_channel_first(arr, name)


def parse_train_subject_info(mat_path: str | Path) -> tuple[str, int]:
    base = Path(mat_path).name
    patterns = [
        r"^(DEP|HC)_(\d+)\.mat$",
        r"^(DEP|HC)(\d+)timedata\.mat$",
        r"^(DEP|HC)(\d+)\.mat$",
        r"^(DEP|HC)[_\-\s]*(\d+).*\.mat$",
        r".*?(DEP|HC)[_\-\s]*(\d+).*\.mat$",
    ]
    for pattern in patterns:
        match = re.match(pattern, base, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper(), int(match.group(2))
    raise ValueError(f"Cannot parse diagnosis/subject id from file name: {base}")


def parse_test_user_id(mat_path: str | Path) -> str:
    return Path(mat_path).stem


def parse_test_subject_number(user_id: str) -> int:
    match = re.search(r"(\d+)$", str(user_id))
    return int(match.group(1)) if match else -1


def diagnosis_label_from_name(diagnosis: str) -> int:
    diag = str(diagnosis).upper()
    if diag == "DEP":
        return 0
    if diag == "HC":
        return 1
    raise ValueError(f"Unknown diagnosis={diagnosis}; expected DEP or HC")


def label4_from_diag_emotion(diagnosis: str, emotion_label: int) -> int:
    return diagnosis_label_from_name(diagnosis) * 2 + int(emotion_label)


def default_channel_names() -> list[str]:
    return [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ",
        "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ",
        "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
    ]


def standardize_channel_names(ch_names: list[str]) -> list[str]:
    name_map = {
        "FP1": "Fp1",
        "FP2": "Fp2",
        "FZ": "Fz",
        "FCZ": "FCz",
        "CZ": "Cz",
        "CPZ": "CPz",
        "PZ": "Pz",
        "OZ": "Oz",
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
    }
    return [name_map.get(str(ch), str(ch)) for ch in ch_names]


def load_channel_names(ch_name_path: str | Path | None = None) -> list[str]:
    if ch_name_path is not None:
        path = Path(ch_name_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            mat = loadmat(path)
            labels = mat["labels"].reshape(-1).tolist()
            names = [item.item() if hasattr(item, "item") else str(item) for item in labels]
            return standardize_channel_names(names)
    print("[channels] ch_name.mat not found; using default 30-channel order.")
    return standardize_channel_names(default_channel_names())


def subject_wise_zscore(clean_all: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_all = np.asarray(clean_all, dtype=np.float32)
    mean = clean_all.mean(axis=1, keepdims=True)
    std = clean_all.std(axis=1, keepdims=True)
    std = np.where(std < eps, eps, std)
    clean_rel = (clean_all - mean) / std
    return clean_rel.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def compute_de_one_window(
    x: np.ndarray,
    sfreq: int = SFREQ,
    bands: list[tuple[float, float]] | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    if bands is None:
        bands = DE_BANDS
    x = np.asarray(x, dtype=np.float32)
    _, time_len = x.shape
    freqs = np.fft.rfftfreq(time_len, d=1.0 / sfreq)
    fft_vals = np.fft.rfft(x, axis=-1)
    power = (fft_vals.real ** 2 + fft_vals.imag ** 2) / max(time_len, 1)
    values = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if mask.sum() == 0:
            band_var = np.zeros((x.shape[0],), dtype=np.float32)
        else:
            band_var = power[:, mask].mean(axis=-1)
        de = 0.5 * np.log(2.0 * math.pi * math.e * band_var + eps)
        values.append(de)
    return np.stack(values, axis=-1).astype(np.float32)


def smooth_de_sequence(de_seq: np.ndarray, smooth_kernel: int = 3) -> np.ndarray:
    de_seq = np.asarray(de_seq, dtype=np.float32)
    if smooth_kernel is None or int(smooth_kernel) <= 1:
        return de_seq
    kernel = int(smooth_kernel)
    pad = kernel // 2
    padded = np.pad(de_seq, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(de_seq, dtype=np.float32)
    for idx in range(de_seq.shape[0]):
        out[idx] = padded[idx : idx + kernel].mean(axis=0)
    return out.astype(np.float32)


def extract_de_sequence(
    trial: np.ndarray,
    sfreq: int = SFREQ,
    win_len: int = WIN_LEN,
    step: int = STEP,
    smooth_kernel: int = 3,
) -> np.ndarray:
    trial = np.asarray(trial, dtype=np.float32)
    de_list = []
    for start in range(0, trial.shape[-1] - win_len + 1, step):
        de_list.append(compute_de_one_window(trial[:, start : start + win_len], sfreq=sfreq))
    if not de_list:
        raise ValueError(f"Trial is too short for DE extraction: shape={trial.shape}, win_len={win_len}")
    return smooth_de_sequence(np.stack(de_list, axis=0), smooth_kernel=smooth_kernel)


def window_starts(total_len: int, win_len: int = WIN_LEN, step: int = STEP) -> list[int]:
    return list(range(0, int(total_len) - int(win_len) + 1, int(step)))

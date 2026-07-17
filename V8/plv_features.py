"""Load, validate, stabilize, or lazily compute five-band PLV features."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

from utils.data import resolve_data_path


DEFAULT_FREQUENCY_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0), ("gamma", 30.0, 45.0),
)


def _context(row: pd.Series, path: Path | str) -> str:
    subject = row.get("user_id", row.get("subject_id", "?"))
    return f"path={path}, subject/user={subject}, trial={row.get('trial_id', '?')}"


def compute_band_plv(eeg_window: np.ndarray, sampling_rate: float = 250.0,
                     bands: Sequence[Sequence[object]] = DEFAULT_FREQUENCY_BANDS) -> np.ndarray:
    """Compute PLV as [bands,channels,channels] from one cleaned EEG window."""
    eeg = np.asarray(eeg_window, dtype=np.float64)
    if eeg.ndim != 2:
        raise ValueError(f"EEG window must be [channels,time], got {eeg.shape}")
    channels, samples = eeg.shape
    if samples < 16:
        raise ValueError(f"EEG window is too short for PLV: {samples} samples")
    nyquist = float(sampling_rate) / 2.0
    result = np.empty((len(bands), channels, channels), dtype=np.float32)
    for band_id, band in enumerate(bands):
        if len(band) != 3:
            raise ValueError(f"frequency band must be (name,low,high), got {band!r}")
        _, low, high = band
        low, high = float(low), float(high)
        if not 0.0 < low < high < nyquist:
            raise ValueError(f"invalid frequency band {band!r} for sampling_rate={sampling_rate}")
        sos = butter(4, (low, high), btype="bandpass", fs=sampling_rate, output="sos")
        filtered = sosfiltfilt(sos, eeg, axis=-1)
        unit_phase = np.exp(1j * np.angle(hilbert(filtered, axis=-1)))
        result[band_id] = np.abs(unit_phase @ unit_phase.conj().T / samples).astype(np.float32)
    return result


def stabilize_plv(plv: np.ndarray, *, num_bands: int = 5, num_nodes: int = 30,
                  context: str = "PLV") -> np.ndarray:
    plv = np.asarray(plv, dtype=np.float32)
    expected = (num_bands, num_nodes, num_nodes)
    if plv.ndim != 3 or plv.shape != expected:
        raise ValueError(f"{context}: PLV shape {plv.shape}, expected {expected}")
    plv = np.nan_to_num(plv, nan=0.0, posinf=0.0, neginf=0.0)
    plv = np.maximum(plv, 0.0)
    plv = 0.5 * (plv + np.swapaxes(plv, -1, -2))
    diagonal = np.arange(num_nodes)
    plv[:, diagonal, diagonal] = 0.0
    if not np.isfinite(plv).all():
        raise ValueError(f"{context}: PLV remains non-finite after stabilization")
    return np.ascontiguousarray(plv, dtype=np.float32)


def _select_saved_plv(row: pd.Series, path: Path, root: Path, num_bands: int, num_nodes: int) -> np.ndarray:
    saved = np.load(path, mmap_mode="r")
    context = _context(row, path)
    de_path = resolve_data_path(row["de_path"], root)
    de_saved = np.load(de_path, mmap_mode="r")
    de_windows = int(de_saved.shape[0]) if de_saved.ndim >= 3 else 1
    de_index = int(row.get("de_win_id", 0))
    if de_index < 0 or de_index >= de_windows:
        raise IndexError(f"{context}: de_win_id={de_index} invalid for DE shape {de_saved.shape} at {de_path}")
    if saved.ndim == 4:
        expected_tail = (num_bands, num_nodes, num_nodes)
        if tuple(saved.shape[1:]) != expected_tail:
            raise ValueError(f"{context}: multi-window PLV shape {saved.shape}, expected [W,{expected_tail}]")
        plv_index = int(row["plv_win_id"]) if "plv_win_id" in row.index and not pd.isna(row["plv_win_id"]) else de_index
        if plv_index < 0 or plv_index >= saved.shape[0]:
            raise IndexError(f"{context}: plv_win_id={plv_index} invalid for PLV shape {saved.shape}")
        if saved.shape[0] != de_windows:
            raise ValueError(
                f"{context}: DE/PLV window counts differ, DE={de_saved.shape} at {de_path}, PLV={saved.shape}"
            )
        return np.asarray(saved[plv_index], dtype=np.float32)
    if saved.ndim == 3:
        plv_index = int(row.get("plv_win_id", 0)) if not pd.isna(row.get("plv_win_id", 0)) else 0
        if plv_index != 0:
            raise IndexError(f"{context}: single-window PLV cannot use plv_win_id={plv_index}")
        return np.asarray(saved, dtype=np.float32)
    raise ValueError(f"{context}: PLV must be [W,5,30,30] or [5,30,30], got {saved.shape}")


def load_or_compute_plv(row: pd.Series, eeg_window: np.ndarray | None = None, *, root: Path,
                        cache_dir: Path | None, sampling_rate: float = 250.0,
                        num_bands: int = 5, num_nodes: int = 30) -> np.ndarray:
    """Prefer indexed PLV, otherwise lazily compute/cache it from the same EEG window."""
    if "plv_path" in row.index and not pd.isna(row["plv_path"]) and str(row["plv_path"]).strip():
        path = resolve_data_path(row["plv_path"], root)
        raw = _select_saved_plv(row, path, root, num_bands, num_nodes)
        return stabilize_plv(raw, num_bands=num_bands, num_nodes=num_nodes, context=_context(row, path))

    trial_path = resolve_data_path(row["trial_path"], root)
    start = int(row.get("start", 0))
    raw_end = row.get("end", None)
    end = int(raw_end) if raw_end is not None and not pd.isna(raw_end) else -1
    token = f"{trial_path.resolve()}|{start}|{end}|{sampling_rate}|{DEFAULT_FREQUENCY_BANDS}"
    cache_path = None if cache_dir is None else cache_dir / f"{hashlib.sha1(token.encode('utf-8')).hexdigest()}.npy"
    if cache_path is not None and cache_path.exists():
        raw = np.load(cache_path, mmap_mode="r")
    else:
        if eeg_window is None:
            trial = np.load(trial_path, mmap_mode="r")
            actual_end = int(trial.shape[-1]) if end < 0 else end
            eeg_window = np.asarray(trial[:, start:actual_end], dtype=np.float32)
        if end < 0:
            end = start + int(eeg_window.shape[-1])
        raw = compute_band_plv(eeg_window, sampling_rate, DEFAULT_FREQUENCY_BANDS)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(f".{os.getpid()}.tmp.npy")
            np.save(temporary, raw)
            try:
                temporary.replace(cache_path)
            except FileExistsError:
                temporary.unlink(missing_ok=True)
    context = _context(row, cache_path or trial_path)
    return stabilize_plv(raw, num_bands=num_bands, num_nodes=num_nodes, context=context)

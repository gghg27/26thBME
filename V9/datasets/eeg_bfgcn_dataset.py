"""Dataset adapter for existing DE files plus lazily computed five-band PLV."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, hilbert, sosfiltfilt
from torch.utils.data import Dataset

from V9.configs.bfgcn_config import BFGCNConfig, DEFAULT_CONFIG, PROJECT_ROOT


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def compute_plv_features(
    eeg_window: np.ndarray,
    sampling_rate: float,
    bands: Sequence[Sequence[object]],
) -> np.ndarray:
    """Compute band-wise phase-locking values as ``[channels, channels, bands]``.

    PLV(i,j) = abs(mean(exp(1j * (phase_i - phase_j)))). The input is an
    already-cleaned EEG window; only the missing BF-GCN connectivity feature is
    extracted here.
    """
    eeg = np.asarray(eeg_window, dtype=np.float64)
    if eeg.ndim != 2:
        raise ValueError(f"eeg_window must be [channels,time], got {eeg.shape}")
    channels, samples = eeg.shape
    if samples < 16:
        raise ValueError(f"EEG window is too short for PLV filtering: {samples} samples")
    result = np.empty((channels, channels, len(bands)), dtype=np.float32)
    nyquist = sampling_rate / 2.0
    for band_index, band in enumerate(bands):
        if len(band) != 3:
            raise ValueError(f"Band must be (name,low,high), got {band!r}")
        _, low, high = band
        low_f, high_f = float(low), float(high)
        if not (0.0 < low_f < high_f < nyquist):
            raise ValueError(f"Invalid band {band!r} for sampling_rate={sampling_rate}")
        sos = butter(4, (low_f, high_f), btype="bandpass", fs=sampling_rate, output="sos")
        filtered = sosfiltfilt(sos, eeg, axis=-1)
        unit_phase = np.exp(1j * np.angle(hilbert(filtered, axis=-1)))
        # [C,T] @ [T,C] -> [C,C], equivalent to pairwise phase-difference means.
        plv = np.abs(unit_phase @ unit_phase.conj().T / samples)
        result[:, :, band_index] = plv.astype(np.float32)
    return result


class EEGBFGCNDataset(Dataset):
    """Window-level BF-GCN dataset backed by the project's existing index CSV.

    A 50-second training trial is represented as five independent 10-second
    pseudo-trials. With a 2-second window and 1-second step, each pseudo-trial
    contains nine windows. Windows crossing pseudo-trial boundaries are excluded.
    Existing 10-second test trials are kept intact.
    """

    def __init__(
        self,
        index_csv: str | Path,
        subject_ids: Iterable[object] | None = None,
        config: BFGCNConfig | None = None,
        cache_plv: bool = True,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.index_csv = _resolve_project_path(index_csv)
        if not self.index_csv.exists():
            raise FileNotFoundError(f"Index CSV does not exist: {self.index_csv}")
        frame = pd.read_csv(self.index_csv)
        subject_field = "subject_id" if "subject_id" in frame.columns else "user_id"
        required = {subject_field, "trial_id", "trial_path", "de_path"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{self.index_csv} is missing columns: {missing}")
        if subject_ids is not None:
            allowed = {str(item) for item in subject_ids}
            frame = frame[frame[subject_field].astype(str).isin(allowed)].copy()
        if frame.empty:
            raise ValueError(f"No data remains after filtering {self.index_csv}")

        self.subject_field = subject_field
        self.cache_plv = bool(cache_plv)
        self.cache_dir = Path(self.config.plv_cache_dir)
        if self.cache_plv:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = self._build_metadata(frame)

    def _build_metadata(self, frame: pd.DataFrame) -> pd.DataFrame:
        consistency_fields = [
            field for field in ("trial_path", "de_path", "emotion_label", "diagnosis_label")
            if field in frame.columns
        ]
        grouped = frame.groupby([self.subject_field, "trial_id"], dropna=False)
        for field in consistency_fields:
            inconsistent = grouped[field].nunique(dropna=False)
            if (inconsistent != 1).any():
                bad = inconsistent[inconsistent != 1].index.tolist()[:5]
                raise ValueError(f"Field {field!r} is inconsistent within subject/trial groups: {bad}")
        trial_rows = frame.drop_duplicates([self.subject_field, "trial_id", "trial_path"]).copy()
        records: list[dict[str, object]] = []
        cfg = self.config
        expected_windows = (cfg.target_trial_samples - cfg.window_length_samples) // cfg.window_step_samples + 1
        for _, row in trial_rows.iterrows():
            trial_path = _resolve_project_path(row["trial_path"])
            de_path = _resolve_project_path(row["de_path"])
            if not trial_path.exists() or not de_path.exists():
                raise FileNotFoundError(f"Missing trial/de pair: {trial_path}, {de_path}")
            trial_shape = np.load(trial_path, mmap_mode="r").shape
            de_shape = np.load(de_path, mmap_mode="r").shape
            if len(trial_shape) != 2 or trial_shape[0] != cfg.num_channels:
                raise ValueError(f"Expected trial [C,T] with C={cfg.num_channels}, got {trial_shape}: {trial_path}")
            if len(de_shape) != 3 or de_shape[1:] != (cfg.num_channels, len(cfg.frequency_bands)):
                raise ValueError(
                    f"Expected DE [W,{cfg.num_channels},{len(cfg.frequency_bands)}], got {de_shape}: {de_path}"
                )
            total_samples = int(trial_shape[1])
            if total_samples % cfg.target_trial_samples:
                raise ValueError(
                    f"Trial length {total_samples} is not divisible by target 10-s length "
                    f"{cfg.target_trial_samples}: {trial_path}"
                )
            pseudo_count = total_samples // cfg.target_trial_samples
            original_trial_id = int(row["trial_id"])
            subject_id = str(row[self.subject_field])
            user_id = str(row["user_id"]) if "user_id" in row.index else subject_id
            emotion = int(row.get("emotion_label", -1))
            diagnosis = int(row.get("diagnosis_label", -1))
            for pseudo_id in range(pseudo_count):
                for window_id in range(expected_windows):
                    raw_start = pseudo_id * cfg.target_trial_samples + window_id * cfg.window_step_samples
                    de_index = raw_start // cfg.window_step_samples
                    if de_index >= de_shape[0]:
                        raise ValueError(f"DE index {de_index} exceeds {de_shape} for {de_path}")
                    records.append(
                        {
                            "subject_id": subject_id,
                            "user_id": user_id,
                            "emotion_label": emotion,
                            "diagnosis_label": diagnosis,
                            "trial_id": (original_trial_id - 1) * pseudo_count + pseudo_id + 1,
                            "original_trial_id": original_trial_id,
                            "pseudo_trial_id": pseudo_id + 1 if pseudo_count > 1 else 0,
                            "window_id": window_id,
                            "raw_start": raw_start,
                            "de_index": de_index,
                            "trial_path": str(trial_path),
                            "de_path": str(de_path),
                        }
                    )
        metadata = pd.DataFrame.from_records(records)
        counts = metadata.groupby(["subject_id", "original_trial_id", "pseudo_trial_id"]).size()
        if not (counts == expected_windows).all():
            raise AssertionError(f"Inconsistent pseudo-trial window counts: {counts.value_counts().to_dict()}")
        return metadata.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def _plv_cache_path(self, row: pd.Series) -> Path:
        token = f"{row['trial_path']}|{row['raw_start']}|{self.config.sampling_rate}|{self.config.frequency_bands}"
        return self.cache_dir / f"{hashlib.sha1(token.encode('utf-8')).hexdigest()}.npy"

    def _load_plv(self, row: pd.Series, eeg_window: np.ndarray) -> np.ndarray:
        cache_path = self._plv_cache_path(row)
        if self.cache_plv and cache_path.exists():
            plv = np.load(cache_path)
        else:
            plv = compute_plv_features(eeg_window, self.config.sampling_rate, self.config.frequency_bands)
            if self.cache_plv:
                # num_workers=0 is the Windows-safe default. A unique temporary
                # suffix also prevents collisions when users opt into workers.
                temp_path = cache_path.with_suffix(f".{os.getpid()}.{id(self)}.tmp.npy")
                np.save(temp_path, plv)
                try:
                    temp_path.replace(cache_path)
                except FileExistsError:
                    temp_path.unlink(missing_ok=True)
        expected = (self.config.num_channels, self.config.num_channels, len(self.config.frequency_bands))
        if plv.shape != expected:
            raise ValueError(f"Cached/computed PLV has shape {plv.shape}, expected {expected}: {cache_path}")
        return plv.astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[int(index)]
        de_seq = np.load(row["de_path"], mmap_mode="r")
        de = np.asarray(de_seq[int(row["de_index"])], dtype=np.float32)
        trial = np.load(row["trial_path"], mmap_mode="r")
        start = int(row["raw_start"])
        end = start + self.config.window_length_samples
        eeg_window = np.asarray(trial[:, start:end], dtype=np.float32)
        if eeg_window.shape != (self.config.num_channels, self.config.window_length_samples):
            raise ValueError(f"Window slice [{start}:{end}] has shape {eeg_window.shape}: {row['trial_path']}")
        if self.config.use_functional_graph:
            plv = self._load_plv(row, eeg_window)
        else:
            # The learnable-graph-only ablation must not spend time extracting
            # or use information from PLV, while preserving a stable batch API.
            plv = np.zeros(
                (self.config.num_channels, self.config.num_channels, len(self.config.frequency_bands)),
                dtype=np.float32,
            )
        return {
            "de": torch.from_numpy(de.copy()),
            "plv": torch.from_numpy(plv.copy()),
            "emotion_label": torch.tensor(int(row["emotion_label"]), dtype=torch.long),
            "diagnosis_label": torch.tensor(int(row["diagnosis_label"]), dtype=torch.long),
            "subject_id": str(row["subject_id"]),
            "user_id": str(row["user_id"]),
            "trial_id": torch.tensor(int(row["trial_id"]), dtype=torch.long),
            "original_trial_id": torch.tensor(int(row["original_trial_id"]), dtype=torch.long),
            "pseudo_trial_id": torch.tensor(int(row["pseudo_trial_id"]), dtype=torch.long),
            "window_id": torch.tensor(int(row["window_id"]), dtype=torch.long),
        }

    @property
    def subject_diagnoses(self) -> pd.DataFrame:
        """Return one diagnosis row per subject for grouped CV splitting."""
        result = self.metadata[["subject_id", "diagnosis_label"]].drop_duplicates()
        inconsistent = result.groupby("subject_id").size()
        if (inconsistent != 1).any():
            raise ValueError("At least one subject has inconsistent diagnosis labels")
        return result.reset_index(drop=True)

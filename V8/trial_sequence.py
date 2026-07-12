"""Ordered full-trial datasets, padding collate, and valid-window utilities."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SCALAR_KEYS = {"label4", "emotion_label", "diagnosis_label", "subject_id", "domain_id", "trial_id"}


def _value_key(value: Any) -> str:
    if torch.is_tensor(value):
        value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    return str(value)


class TrialSequenceDataset(Dataset):
    """Wrap a window dataset as explicitly time-ordered, complete trial samples."""

    def __init__(self, window_dataset: Dataset, trial_num_windows: int = 0, name: str = "dataset") -> None:
        if not hasattr(window_dataset, "df"):
            raise AttributeError("TrialSequenceDataset expects a window dataset with a pandas df attribute")
        self.window_dataset = window_dataset
        self.trial_num_windows = int(trial_num_windows)
        self.name = str(name)
        self.df = window_dataset.df.reset_index(drop=False).rename(columns={"index": "__original_index__"})
        self.subject_field = self._choose_field(("subject_id", "subject", "user_id", "user", "target_key"))
        if "trial_id" not in self.df.columns:
            raise KeyError("window dataset df must contain trial_id")
        self.time_field = self._choose_time_field()
        self.trial_groups = self._build_groups()
        counts = [len(g["window_indices"]) for g in self.trial_groups]
        self.window_count_distribution = dict(sorted(Counter(counts).items()))
        print(
            f"[TrialSequenceDataset:{self.name}] subject={self.subject_field}, "
            f"time_sort={self.time_field}, trials={len(self.trial_groups)}, "
            f"window_counts={self.window_count_distribution}"
        )

    def _choose_field(self, candidates: Iterable[str]) -> str:
        for field in candidates:
            if field in self.df.columns:
                return field
        raise KeyError(f"none of the required grouping fields exist: {tuple(candidates)}")

    def _choose_time_field(self) -> str:
        for field in ("start", "de_win_id"):
            if field in self.df.columns and self.df[field].notna().any():
                return field
        return "__original_index__"

    @staticmethod
    def _consistent(frame: pd.DataFrame, field: str) -> Any:
        if field not in frame.columns:
            return None
        values = frame[field].dropna().unique()
        if len(values) > 1:
            raise ValueError(f"inconsistent {field} inside trial: {values.tolist()}")
        return values[0] if len(values) else None

    def _build_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for (subject, trial_id), frame in self.df.groupby([self.subject_field, "trial_id"], sort=False, dropna=False):
            frame = frame.copy()
            numeric_time = pd.to_numeric(frame[self.time_field], errors="coerce")
            frame["__time_sort__"] = numeric_time if numeric_time.notna().all() else frame[self.time_field]
            ordered = frame.sort_values(["__time_sort__", "__original_index__"], kind="mergesort")
            time_values = ordered[self.time_field].to_numpy()
            if len(time_values) > 1:
                try:
                    if np.any(time_values[1:] < time_values[:-1]):
                        raise AssertionError("trial time order is not monotonic")
                except TypeError:
                    pass
            group = {
                "subject_key": subject, "trial_id": trial_id,
                "window_indices": ordered.index.astype(int).tolist(),
                "window_start": ordered[self.time_field].tolist(),
            }
            for field in ("label4", "emotion_label", "diagnosis_label", "subject_id", "domain_id", "user_id", "target_key"):
                group[field] = self._consistent(ordered, field)
            groups.append(group)
        if not groups:
            raise ValueError("no trials were built")
        return groups

    def __len__(self) -> int:
        return len(self.trial_groups)

    def _selected_positions(self, length: int) -> np.ndarray:
        if self.trial_num_windows <= 0 or self.trial_num_windows >= length:
            return np.arange(length, dtype=np.int64)
        return np.unique(np.linspace(0, length - 1, self.trial_num_windows).round().astype(np.int64))

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = self.trial_groups[int(index)]
        pos = self._selected_positions(len(group["window_indices"]))
        indices = [group["window_indices"][int(i)] for i in pos]
        items = [self.window_dataset[i] for i in indices]
        out: dict[str, Any] = {}
        for key in items[0]:
            values = [item[key] for item in items]
            if key in SCALAR_KEYS:
                out[key] = values[0]
            elif torch.is_tensor(values[0]):
                out[key] = torch.stack(values)
            else:
                out[key] = values[0]
        out["window_mask"] = torch.ones(len(indices), dtype=torch.bool)
        out["window_indices"] = torch.tensor(indices, dtype=torch.long)
        starts = [group["window_start"][int(i)] for i in pos]
        try:
            out["window_start"] = torch.as_tensor(starts)
        except (TypeError, ValueError):
            out["window_start"] = starts
        out.setdefault("trial_id", torch.as_tensor(group["trial_id"]))
        out.setdefault("subject_id", torch.as_tensor(group.get("subject_id") or 0))
        out.setdefault("user_id", group.get("user_id") or group["subject_key"])
        out.setdefault("target_key", group.get("target_key") or _value_key(group["subject_key"]))
        return out


def trial_sequence_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Right-pad time-varying tensors and emit a boolean ``window_mask``."""
    if not samples:
        return {}
    lengths = [int(sample["window_mask"].numel()) for sample in samples]
    max_t = max(lengths)
    out: dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        first = values[0]
        is_temporal = key in {"window_mask", "window_indices", "window_start"} or (
            torch.is_tensor(first) and first.ndim > 0 and first.shape[0] == lengths[0] and key not in SCALAR_KEYS
        )
        if torch.is_tensor(first) and is_temporal:
            shape = (len(samples), max_t, *first.shape[1:])
            fill = False if first.dtype == torch.bool else (-1 if key == "window_indices" else 0)
            padded = torch.full(shape, fill, dtype=first.dtype)
            for i, value in enumerate(values):
                padded[i, : lengths[i]] = value
            out[key] = padded
        elif torch.is_tensor(first):
            out[key] = torch.stack(values)
        elif isinstance(first, (int, float, np.integer, np.floating)):
            out[key] = torch.as_tensor(values)
        else:
            out[key] = values
    out["window_mask"] = torch.arange(max_t).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    return out


def valid_window_batch(batch: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    """Gather only valid ``[B,T,...]`` values and expand per-trial baselines to windows."""
    mask = batch["window_mask"].bool()
    bsz, steps = mask.shape
    flat: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[:2] == (bsz, steps):
            flat[key] = value[mask]
        elif torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == bsz:
            expanded = value.unsqueeze(1).expand(bsz, steps, *value.shape[1:])
            flat[key] = expanded[mask]
        else:
            flat[key] = value
    return flat, mask


def scatter_valid_features(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = features.new_zeros((*mask.shape, features.shape[-1]))
    out[mask] = features
    return out

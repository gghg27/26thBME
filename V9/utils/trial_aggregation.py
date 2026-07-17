"""Order-invariant probability aggregation from windows to 10-second trials."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch


DEFAULT_TRIAL_KEYS = ("subject_id", "original_trial_id", "pseudo_trial_id")


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def aggregate_window_probabilities(
    window_probabilities: np.ndarray | torch.Tensor,
    metadata: pd.DataFrame | Mapping[str, Sequence[Any]],
    key_fields: Sequence[str] = DEFAULT_TRIAL_KEYS,
    label_fields: Sequence[str] = ("emotion_label", "diagnosis_label", "trial_id", "user_id"),
) -> pd.DataFrame:
    """Mean probabilities within each trial, independent of input window order."""
    probabilities = _as_numpy(window_probabilities)
    if probabilities.ndim != 2:
        raise ValueError(f"window_probabilities must be [samples,classes], got {probabilities.shape}")
    frame = metadata.copy() if isinstance(metadata, pd.DataFrame) else pd.DataFrame(metadata)
    if len(frame) != len(probabilities):
        raise ValueError(f"Metadata rows {len(frame)} != probabilities {len(probabilities)}")
    missing = [field for field in key_fields if field not in frame.columns]
    if missing:
        raise KeyError(f"Aggregation metadata is missing key fields: {missing}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Window probabilities contain NaN or Inf")
    probability_fields = [f"prob_{index}" for index in range(probabilities.shape[1])]
    for index, field in enumerate(probability_fields):
        frame[field] = probabilities[:, index]

    records: list[dict[str, Any]] = []
    group_key: str | list[str] = key_fields[0] if len(key_fields) == 1 else list(key_fields)
    for keys, group in frame.groupby(group_key, sort=False, dropna=False):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(key_fields, keys_tuple))
        for field in label_fields:
            if field not in group.columns:
                continue
            values = group[field].drop_duplicates()
            if len(values) != 1:
                raise ValueError(f"Field {field!r} is inconsistent within trial {row}: {values.tolist()}")
            row[field] = values.iloc[0]
        averaged = group[probability_fields].mean(axis=0).to_numpy(dtype=float)
        row.update({field: float(averaged[index]) for index, field in enumerate(probability_fields)})
        row["prediction"] = int(np.argmax(averaged))
        row["n_windows"] = int(len(group))
        records.append(row)
    return pd.DataFrame.from_records(records)


def natural_sort_key(value: object) -> tuple[object, ...]:
    """Sort P_test2 before P_test10 while retaining arbitrary text identifiers."""
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value)))


def natural_sort_trials(frame: pd.DataFrame, user_field: str = "user_id") -> pd.DataFrame:
    """Naturally order users and then numerically order trial IDs."""
    result = frame.copy()
    result["__user_sort"] = result[user_field].map(natural_sort_key)
    trial_field = "original_trial_id" if "original_trial_id" in result.columns else "trial_id"
    result = result.sort_values(["__user_sort", trial_field], kind="stable").drop(columns="__user_sort")
    return result.reset_index(drop=True)


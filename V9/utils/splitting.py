"""Deterministic diagnosis-stratified subject-level cross-validation."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def _fallback_splits(subjects: pd.DataFrame, n_splits: int, seed: int) -> list[tuple[list[str], list[str]]]:
    rng = np.random.default_rng(seed)
    per_group: dict[int, list[np.ndarray]] = {}
    for diagnosis in (0, 1):
        values = subjects.loc[subjects["diagnosis_label"] == diagnosis, "subject_id"].astype(str).to_numpy()
        rng.shuffle(values)
        per_group[diagnosis] = list(np.array_split(values, n_splits))
    all_subjects = set(subjects["subject_id"].astype(str))
    result = []
    for fold in range(n_splits):
        validation = sorted(
            [str(value) for diagnosis in (0, 1) for value in per_group[diagnosis][fold]], key=str
        )
        result.append((sorted(all_subjects.difference(validation), key=str), validation))
    return result


def stratified_subject_splits(
    subject_diagnoses: pd.DataFrame, n_splits: int = 10, seed: int = 42
) -> list[tuple[list[str], list[str]]]:
    """Return ``(train_subjects,val_subjects)`` with no possible subject leakage."""
    required = {"subject_id", "diagnosis_label"}
    if not required.issubset(subject_diagnoses.columns):
        raise KeyError(f"subject_diagnoses needs columns {sorted(required)}")
    subjects = subject_diagnoses[list(required)].drop_duplicates().copy()
    if subjects["subject_id"].duplicated().any():
        raise ValueError("A subject has multiple diagnosis labels")
    counts = subjects.groupby("diagnosis_label").size().to_dict()
    if set(counts) != {0, 1} or min(counts.values()) < n_splits:
        raise ValueError(f"Need at least n_splits subjects in HC and DEP groups, got {counts}")
    subjects["subject_id"] = subjects["subject_id"].astype(str)
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        result = []
        for train_index, val_index in splitter.split(
            subjects, subjects["diagnosis_label"], groups=subjects["subject_id"]
        ):
            train = subjects.iloc[train_index]["subject_id"].tolist()
            val = subjects.iloc[val_index]["subject_id"].tolist()
            result.append((train, val))
        # StratifiedGroupKFold is heuristic. With one row per group it can still
        # yield 3/3 or 5/1 here even though exact 4 HC + 2 DEP folds are possible.
        diagnosis_lookup = dict(zip(subjects["subject_id"], subjects["diagnosis_label"]))
        expected_parts = {
            diagnosis: sorted(len(values) for values in np.array_split(np.empty(count), n_splits))
            for diagnosis, count in counts.items()
        }
        actual_parts = {
            diagnosis: sorted(sum(diagnosis_lookup[item] == diagnosis for item in val) for _, val in result)
            for diagnosis in counts
        }
        if actual_parts != expected_parts:
            result = _fallback_splits(subjects, n_splits, seed)
    except (ImportError, ValueError):
        result = _fallback_splits(subjects, n_splits, seed)
    for train, val in result:
        overlap = set(train).intersection(val)
        if overlap:
            raise AssertionError(f"Subject leakage detected: {sorted(overlap)}")
    return result

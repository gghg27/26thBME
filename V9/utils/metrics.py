"""Window, trial, and diagnosis-subgroup classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Return binary accuracy, macro-F1, per-class recall, and confusion matrix."""
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError(f"labels/predictions must be matching vectors, got {labels.shape}/{predictions.shape}")
    if labels.size == 0:
        return {
            "accuracy": None,
            "macro_f1": None,
            "neutral_recall": None,
            "positive_recall": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "count": 0,
        }
    recalls = recall_score(labels, predictions, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "neutral_recall": float(recalls[0]),
        "positive_recall": float(recalls[1]),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).astype(int).tolist(),
        "count": int(labels.size),
    }


def trial_and_subgroup_metrics(trial_frame) -> dict[str, Any]:
    """Evaluate overall, HC (0), and DEP (1) trial predictions."""
    if "emotion_label" not in trial_frame or "diagnosis_label" not in trial_frame:
        raise KeyError("trial_frame needs emotion_label and diagnosis_label")
    labels = trial_frame["emotion_label"].to_numpy(dtype=int)
    predictions = trial_frame["prediction"].to_numpy(dtype=int)
    result: dict[str, Any] = {"overall": binary_metrics(labels, predictions)}
    diagnosis = trial_frame["diagnosis_label"].to_numpy(dtype=int)
    for value, name in ((0, "HC"), (1, "DEP")):
        mask = diagnosis == value
        result[name] = binary_metrics(labels[mask], predictions[mask])
    return result


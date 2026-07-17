"""Validate DE/PLV shapes, values, identities, balance, and optional split leakage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from V9.configs.bfgcn_config import BFGCNConfig
from V9.datasets.eeg_bfgcn_dataset import EEGBFGCNDataset


def _parse_subjects(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=None)
    parser.add_argument("--max-plv-samples", type=int, default=0, help="0 checks every PLV window")
    parser.add_argument("--de-abs-limit", type=float, default=100.0)
    parser.add_argument("--symmetry-tolerance", type=float, default=1e-4)
    parser.add_argument("--train-subjects", default="")
    parser.add_argument("--val-subjects", default="")
    parser.add_argument("--no-plv-cache", action="store_true")
    args = parser.parse_args()

    config = BFGCNConfig()
    index_path = args.index or config.train_index_csv
    dataset = EEGBFGCNDataset(index_path, config=config, cache_plv=not args.no_plv_cache)
    metadata = dataset.metadata
    errors: list[str] = []
    expected_de = (config.num_channels, len(config.frequency_bands))
    expected_plv = (config.num_channels, config.num_channels, len(config.frequency_bands))

    de_min, de_max = np.inf, -np.inf
    for path_text in metadata["de_path"].drop_duplicates():
        array = np.load(path_text, mmap_mode="r")
        if array.ndim != 3 or tuple(array.shape[1:]) != expected_de:
            errors.append(f"DE shape {array.shape}, expected [N,{expected_de[0]},{expected_de[1]}]: {path_text}")
            continue
        if not np.isfinite(array).all():
            errors.append(f"DE contains NaN/Inf: {path_text}")
        de_min = min(de_min, float(np.nanmin(array)))
        de_max = max(de_max, float(np.nanmax(array)))
    if max(abs(de_min), abs(de_max)) > args.de_abs_limit:
        errors.append(f"DE extreme value exceeds {args.de_abs_limit}: min={de_min}, max={de_max}")

    total = len(dataset) if args.max_plv_samples <= 0 else min(len(dataset), args.max_plv_samples)
    indices = np.linspace(0, len(dataset) - 1, total, dtype=int) if total else np.asarray([], dtype=int)
    max_asymmetry = 0.0
    diagonal_min, diagonal_max = np.inf, -np.inf
    for count, index in enumerate(indices, start=1):
        item = dataset[int(index)]
        de = item["de"].numpy()
        plv = item["plv"].numpy()
        if de.shape != expected_de:
            errors.append(f"sample {index} DE shape {de.shape}, expected {expected_de}")
        if plv.shape != expected_plv:
            errors.append(f"sample {index} PLV shape {plv.shape}, expected {expected_plv}")
            continue
        if not np.isfinite(plv).all():
            errors.append(f"sample {index} PLV contains NaN/Inf")
        max_asymmetry = max(max_asymmetry, float(np.max(np.abs(plv - plv.transpose(1, 0, 2)))))
        diagonals = np.diagonal(plv, axis1=0, axis2=1)
        diagonal_min = min(diagonal_min, float(diagonals.min()))
        diagonal_max = max(diagonal_max, float(diagonals.max()))
    if max_asymmetry > args.symmetry_tolerance:
        errors.append(f"PLV asymmetry {max_asymmetry} exceeds tolerance {args.symmetry_tolerance}")
    if total and (diagonal_min < 0.95 or diagonal_max > 1.0001):
        errors.append(f"PLV diagonal outside expected near-one range: [{diagonal_min}, {diagonal_max}]")

    trial_keys = ["subject_id", "original_trial_id", "pseudo_trial_id"]
    counts = metadata.groupby(trial_keys).size()
    if counts.nunique() != 1:
        errors.append(f"Inconsistent windows per trial: {counts.value_counts().to_dict()}")
    labeled = metadata[metadata["emotion_label"].isin([0, 1])]
    balance = labeled.groupby(["subject_id", "emotion_label"]).size().unstack(fill_value=0)
    if not balance.empty and (balance.min(axis=1) == 0).any():
        errors.append("At least one labeled subject is missing neutral or positive samples")
    for field in ("emotion_label", "diagnosis_label", "trial_id", "window_id"):
        if metadata[field].isna().any():
            errors.append(f"Metadata field {field} contains missing values")
    train_subjects = _parse_subjects(args.train_subjects)
    val_subjects = _parse_subjects(args.val_subjects)
    leakage = sorted(train_subjects.intersection(val_subjects))
    if leakage:
        errors.append(f"Train/validation subject leakage: {leakage}")

    report = {
        "index": str(index_path),
        "samples": len(dataset),
        "subjects": int(metadata["subject_id"].nunique()),
        "original_trials": int(metadata.groupby(["subject_id", "original_trial_id"]).ngroups),
        "ten_second_trials": int(counts.size),
        "windows_per_trial": counts.value_counts().sort_index().to_dict(),
        "emotion_sample_counts": metadata["emotion_label"].value_counts().sort_index().to_dict(),
        "diagnosis_subject_counts": dataset.subject_diagnoses["diagnosis_label"].value_counts().sort_index().to_dict(),
        "de_shape": list(expected_de),
        "de_min": de_min,
        "de_max": de_max,
        "plv_shape": list(expected_plv),
        "plv_samples_checked": total,
        "plv_max_asymmetry": max_asymmetry,
        "plv_diagonal_range": [diagonal_min, diagonal_max] if total else None,
        "leakage": leakage,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=int))
    if errors:
        raise RuntimeError(f"BF-GCN data inspection failed with {len(errors)} serious issue(s)")


if __name__ == "__main__":
    main()


"""Unified finals inference and voting for V7, V8, and V9 checkpoints."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FRAMEWORK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FRAMEWORK_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_VERSIONS = {"v7", "v8", "v9"}
TRIAL_KEYS = ["user_id", "trial_id"]


class ConfigError(ValueError):
    """Raised when a finals inference configuration is invalid."""


@dataclass(frozen=True)
class CheckpointEntry:
    path: Path
    fold: int | None
    weight: float
    threshold_path: Path | None = None


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _glob_paths(pattern: str) -> list[Path]:
    candidate = Path(pattern)
    resolved = str(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate)
    return sorted((Path(item).resolve() for item in glob.glob(resolved, recursive=True)), key=_natural_key)


def _natural_key(value: object) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value)))


def _safe_name(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
    return result or "model"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _fold_from_path(path: Path) -> int | None:
    matches = re.findall(r"fold[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    return int(matches[-1]) if matches else None


def _nested_get(data: dict[str, Any], dotted_key: str) -> Any:
    value: Any = data
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be a JSON object")
    if not isinstance(config.get("models"), list) or not config["models"]:
        raise ConfigError("Configuration must contain a non-empty models list")
    return config


def _select_models(config: dict[str, Any], requested: set[str] | None) -> list[dict[str, Any]]:
    models = config["models"]
    names = [str(item.get("name", "")) for item in models]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ConfigError("Every model needs a unique non-empty name")
    if requested is not None:
        missing = sorted(requested.difference(names))
        if missing:
            raise ConfigError(f"Unknown model names: {missing}; available={names}")
        selected = [item for item in models if item["name"] in requested]
    else:
        selected = [item for item in models if item.get("enabled", True)]
    if not selected:
        raise ConfigError("No models selected")
    return selected


def _checkpoint_weight(spec: dict[str, Any], path: Path, fold: int | None) -> float:
    weights = spec.get("checkpoint_weights", {})
    if isinstance(weights, list):
        raise ConfigError(f"{spec['name']}: checkpoint_weights must be an object keyed by fold/path")
    candidates = [str(path), path.name]
    if fold is not None:
        candidates = [str(fold), f"fold_{fold}", f"fold{fold}"] + candidates
    value = next((weights[key] for key in candidates if key in weights), 1.0)
    weight = float(value)
    if weight <= 0:
        raise ConfigError(f"{spec['name']}: checkpoint weight must be positive for {path}")
    return weight


def _threshold_path(spec: dict[str, Any], checkpoint: Path, fold: int | None) -> Path | None:
    if spec.get("probability", "raw") != "adaptive":
        return None
    mapping = spec.get("threshold_paths", {})
    keys = [str(fold), f"fold_{fold}", f"fold{fold}"] if fold is not None else []
    explicit = next((mapping[key] for key in keys if key in mapping), None)
    if explicit:
        return _project_path(explicit)
    filename = str(spec.get("threshold_filename", "adaptive_threshold_best.pt"))
    return (checkpoint.parent / filename).resolve()


def discover_checkpoints(spec: dict[str, Any]) -> list[CheckpointEntry]:
    version = str(spec.get("version", "")).lower()
    if version not in SUPPORTED_VERSIONS:
        raise ConfigError(f"{spec['name']}: unsupported version {version!r}")
    explicit = spec.get("checkpoints")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ConfigError(f"{spec['name']}: checkpoints must be a non-empty list")
        paths = [_project_path(item) for item in explicit]
    else:
        pattern = spec.get("checkpoint_glob")
        if not pattern:
            raise ConfigError(f"{spec['name']}: set checkpoints or checkpoint_glob")
        paths = _glob_paths(str(pattern))
    paths = [path for path in paths if path.is_file()]
    folds = spec.get("folds")
    allowed = {int(item) for item in folds} if folds is not None else None
    entries: list[CheckpointEntry] = []
    for path in paths:
        fold = _fold_from_path(path)
        if allowed is not None and fold not in allowed:
            continue
        entries.append(
            CheckpointEntry(
                path=path,
                fold=fold,
                weight=_checkpoint_weight(spec, path, fold),
                threshold_path=_threshold_path(spec, path, fold),
            )
        )
    if not entries:
        raise ConfigError(f"{spec['name']}: no checkpoint files matched")
    for entry in entries:
        if entry.threshold_path is not None and not entry.threshold_path.is_file():
            raise ConfigError(
                f"{spec['name']}: adaptive threshold is missing for {entry.path}: {entry.threshold_path}"
            )
    return entries


def inspect_test_index(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise ConfigError(f"Test index does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "trial_id", "trial_path", "de_path"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ConfigError(f"Test index is missing columns: {missing}")
        rows = list(reader)
    users = {row["user_id"] for row in rows}
    trials = {(row["user_id"], row["trial_id"]) for row in rows}
    return {"rows": len(rows), "users": len(users), "trials": len(trials)}


def _ensure_mapping_users(mapping: dict[str, Any], test_index: Path, model_name: str) -> None:
    with test_index.open("r", encoding="utf-8-sig", newline="") as handle:
        users = {row["user_id"] for row in csv.DictReader(handle)}
    lookup = mapping.get("key_to_domain", {})
    missing = sorted(
        [user for user in users if f"test:{user}" not in lookup], key=_natural_key
    )
    if missing:
        raise ConfigError(
            f"{model_name}: checkpoint domain_mapping does not contain finals users {missing}. "
            "Use checkpoints trained with the same test user IDs or add a compatible inference mapping."
        )


def _predict_experiment_a(
    spec: dict[str, Any],
    entry: CheckpointEntry,
    test_index: Path,
    device: Any,
    batch_size: int,
    num_workers: int,
) -> Any:
    import importlib
    import torch
    from torch.utils.data import DataLoader

    version = spec["version"].lower()
    package = version.upper()
    module = importlib.import_module(f"{package}.train_experiment_a")
    threshold_module = importlib.import_module(f"{package}.adaptive_threshold")
    checkpoint = torch.load(entry.path, map_location=device, weights_only=False)
    model = module.rebuild_stage2(checkpoint, device)
    model.eval()
    mapping = checkpoint["domain_mapping"]
    _ensure_mapping_users(mapping, test_index, spec["name"])
    normalize = bool(spec.get("normalize", True))
    relative_eps = float(spec.get("relative_eps", 1e-6))

    if version == "v7":
        window_data = module.UnlabeledTargetDataset(test_index, mapping, normalize)
    else:
        model_config = checkpoint["model_config"]
        cache_dir = _project_path(spec.get("plv_cache_dir", "finals_prediction/cache/v8_plv"))
        window_data = module.UnlabeledTargetDataset(
            test_index,
            mapping,
            normalize,
            plv_cache_dir=cache_dir,
            sampling_rate=float(spec.get("sampling_rate", 250.0)),
            num_nodes=int(model_config["num_nodes"]),
            num_bands=int(model_config["de_num_bands"]),
        )
    trial_data = module.TrialSequenceDataset(
        window_data, int(checkpoint["trial_num_windows"]), f"finals-{spec['name']}-{entry.fold}"
    )
    loader = DataLoader(
        trial_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=module.trial_sequence_collate,
    )
    de_mean = de_std = None
    if checkpoint["model_config"].get("use_subject_relative_de"):
        de_mean, de_std = module.compute_de_baseline(window_data, relative_eps)
    if version == "v7":
        bio_mean = bio_std = None
        if checkpoint["model_config"].get("use_subject_relative_bio"):
            bio_mean, bio_std = module.compute_bio_baseline(
                model, window_data, device, de_mean, de_std, batch_size, num_workers, relative_eps
            )
        baselines = (de_mean, de_std, bio_mean, bio_std)
    else:
        baselines = (de_mean, de_std)
    frame = module.predict_trials(model, loader, device, baselines, labeled=False, max_batches=0)
    probability_column = "prob_pos"
    if spec.get("probability", "raw") == "adaptive":
        if entry.threshold_path is None:
            raise ConfigError(f"{spec['name']}: adaptive probability requires a threshold checkpoint")
        threshold, _ = threshold_module.load_threshold(entry.threshold_path, device)
        frame, _ = threshold_module.apply_threshold(frame, threshold, device, labeled=False)
        probability_column = "adaptive_probability"
    result = frame[TRIAL_KEYS].copy()
    result["prob_positive"] = frame[probability_column].astype(float)
    del model, checkpoint, loader, trial_data, window_data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _predict_v9(
    spec: dict[str, Any],
    entry: CheckpointEntry,
    test_index: Path,
    device: Any,
    batch_size: int,
    num_workers: int,
) -> Any:
    from V9.configs.bfgcn_config import BFGCNConfig
    from V9.datasets.eeg_bfgcn_dataset import EEGBFGCNDataset
    from V9.predict_bfgcn_test import predict_one_fold
    from V9.utils.checkpoint import load_checkpoint

    state = load_checkpoint(entry.path, "cpu")
    data_config = BFGCNConfig.from_dict(state["config"])
    data_config.test_index_csv = str(test_index)
    data_config.cache_plv = True
    data_config.plv_cache_dir = str(
        _project_path(spec.get("plv_cache_dir", "finals_prediction/cache/v9_plv"))
    )
    dataset = EEGBFGCNDataset(test_index, config=data_config, cache_plv=True)
    trials = predict_one_fold(entry.path, dataset, batch_size, device, num_workers)
    result = trials[TRIAL_KEYS].copy()
    result["prob_positive"] = trials["prob_1"].astype(float)
    return result


def _validate_frame(frame: Any, context: str) -> Any:
    import numpy as np

    missing = [column for column in TRIAL_KEYS + ["prob_positive"] if column not in frame]
    if missing:
        raise RuntimeError(f"{context}: prediction frame missing columns {missing}")
    result = frame[TRIAL_KEYS + ["prob_positive"]].copy()
    result["user_id"] = result["user_id"].astype(str)
    result["trial_id"] = result["trial_id"].astype(int)
    result["prob_positive"] = result["prob_positive"].astype(float)
    if result.duplicated(TRIAL_KEYS).any():
        raise RuntimeError(f"{context}: duplicate user_id/trial_id predictions")
    values = result["prob_positive"].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise RuntimeError(f"{context}: probabilities must be finite and within [0,1]")
    users = sorted(result["user_id"].unique(), key=_natural_key)
    user_order = {user: index for index, user in enumerate(users)}
    result["__user_order"] = result["user_id"].map(user_order)
    return result.sort_values(["__user_order", "trial_id"]).drop(columns="__user_order").reset_index(drop=True)


def _align_probability(reference: Any, frame: Any, context: str) -> Any:
    merged = reference[TRIAL_KEYS].merge(frame, on=TRIAL_KEYS, how="left", validate="one_to_one")
    if merged["prob_positive"].isna().any() or len(merged) != len(frame):
        raise RuntimeError(f"{context}: prediction trial identities do not match other models")
    return merged["prob_positive"].to_numpy(float)


def run_inference(
    config: dict[str, Any],
    models: list[dict[str, Any]],
    entries_by_model: dict[str, list[CheckpointEntry]],
    test_index: Path,
    output_dir: Path,
    device_text: str,
) -> None:
    import numpy as np
    import pandas as pd
    import torch

    if device_text == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_text)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    runtime = config.get("runtime", {})
    batch_size = int(runtime.get("batch_size", 16))
    num_workers = int(runtime.get("num_workers", 0))
    if batch_size < 1 or num_workers < 0:
        raise ConfigError("runtime.batch_size must be positive and num_workers nonnegative")
    print(f"[runtime] device={device} batch_size={batch_size} num_workers={num_workers}")

    long_frames: list[Any] = []
    model_frames: list[Any] = []
    reference = None
    for spec in models:
        name = spec["name"]
        version = spec["version"].lower()
        checkpoint_frames: list[Any] = []
        checkpoint_weights: list[float] = []
        for index, entry in enumerate(entries_by_model[name], start=1):
            print(f"[{name}] {index}/{len(entries_by_model[name])}: {entry.path}")
            if version in {"v7", "v8"}:
                frame = _predict_experiment_a(spec, entry, test_index, device, batch_size, num_workers)
            else:
                frame = _predict_v9(spec, entry, test_index, device, batch_size, num_workers)
            frame = _validate_frame(frame, f"{name}:{entry.path.name}")
            if reference is None:
                reference = frame[TRIAL_KEYS].copy()
            probability = _align_probability(reference, frame, f"{name}:{entry.path.name}")
            checkpoint_frames.append(probability)
            checkpoint_weights.append(entry.weight)
            detail = reference.copy()
            detail["model"] = name
            detail["version"] = version
            detail["fold"] = entry.fold
            detail["checkpoint"] = _display_path(entry.path)
            detail["checkpoint_weight"] = entry.weight
            detail["prob_positive"] = probability
            long_frames.append(detail)
        stacked = np.stack(checkpoint_frames, axis=0)
        model_probability = np.average(stacked, axis=0, weights=np.asarray(checkpoint_weights))
        model_frame = reference.copy()
        model_frame["model"] = name
        model_frame["version"] = version
        model_frame["model_weight"] = float(spec.get("weight", 1.0))
        model_frame["num_checkpoints"] = len(checkpoint_frames)
        model_frame["prob_positive"] = model_probability
        model_frames.append(model_frame)
        print(
            f"[{name}] checkpoints={len(checkpoint_frames)} "
            f"p_mean={model_probability.mean():.4f} positives@0.5={(model_probability >= 0.5).sum()}"
        )

    assert reference is not None
    names = [spec["name"] for spec in models]
    matrix = np.stack(
        [frame["prob_positive"].to_numpy(float) for frame in model_frames], axis=0
    )
    model_weights = np.asarray([float(spec.get("weight", 1.0)) for spec in models], dtype=float)
    if (model_weights <= 0).any():
        raise ConfigError("Every model weight must be positive")
    soft_probability = np.average(matrix, axis=0, weights=model_weights)
    vote = config.get("vote", {})
    method = str(vote.get("method", "soft")).lower()
    threshold = float(vote.get("threshold", 0.5))
    if not 0 <= threshold <= 1:
        raise ConfigError("vote.threshold must be within [0,1]")
    if method == "soft":
        vote_score = soft_probability
    elif method == "hard":
        decision_thresholds = np.asarray(
            [float(spec.get("decision_threshold", 0.5)) for spec in models], dtype=float
        )
        vote_score = np.average(matrix >= decision_thresholds[:, None], axis=0, weights=model_weights)
    else:
        raise ConfigError("vote.method must be 'soft' or 'hard'")

    final = reference.copy()
    for index, name in enumerate(names):
        final[f"prob_{_safe_name(name)}"] = matrix[index]
    final["ensemble_probability_soft"] = soft_probability
    final["ensemble_vote_score"] = vote_score
    final["Emotion_label"] = (vote_score >= threshold).astype(int)
    final["vote_method"] = method
    final["vote_threshold"] = threshold
    final["num_models"] = len(models)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(long_frames, ignore_index=True).to_csv(
        output_dir / "checkpoint_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(model_frames, ignore_index=True).to_csv(
        output_dir / "model_probabilities.csv", index=False, encoding="utf-8-sig"
    )
    final.to_csv(output_dir / "final_vote_probabilities.csv", index=False, encoding="utf-8-sig")
    submission = final[TRIAL_KEYS + ["Emotion_label"]]
    submission.to_csv(output_dir / "submission.csv", index=False, encoding="utf-8-sig")
    try:
        submission.to_excel(output_dir / "submission.xlsx", index=False)
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"[warning] submission.xlsx was not written: {exc}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_index": str(test_index),
        "device": str(device),
        "models": [
            {
                "name": spec["name"],
                "version": spec["version"],
                "weight": float(spec.get("weight", 1.0)),
                "checkpoints": [str(entry.path) for entry in entries_by_model[spec["name"]]],
            }
            for spec in models
        ],
        "vote": {"method": method, "threshold": threshold},
        "trials": len(final),
        "positive_predictions": int(final["Emotion_label"].sum()),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] trials={len(final)} positives={int(final['Emotion_label'].sum())}")
    print(f"[done] output={output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(FRAMEWORK_ROOT / "configs" / "current_models.json"),
        help="JSON configuration path",
    )
    parser.add_argument("--models", default="", help="Comma-separated model names; overrides enabled flags")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/config without loading PyTorch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    requested = {item.strip() for item in args.models.split(",") if item.strip()} or None
    if args.list_models:
        for spec in config["models"]:
            print(
                f"{spec['name']}: version={spec.get('version')} "
                f"enabled={spec.get('enabled', True)} weight={spec.get('weight', 1.0)}"
            )
        return
    models = _select_models(config, requested)
    test_index = _project_path(config.get("test_index", "com_juesai_window_index_2s.csv"))
    index_info = inspect_test_index(test_index)
    print(f"[index] {test_index} {index_info}")
    entries_by_model: dict[str, list[CheckpointEntry]] = {}
    for spec in models:
        entries = discover_checkpoints(spec)
        entries_by_model[spec["name"]] = entries
        print(
            f"[model] {spec['name']} version={spec['version']} checkpoints={len(entries)} "
            f"folds={[entry.fold for entry in entries]}"
        )
    if args.dry_run:
        print("[dry-run] configuration, test index, checkpoints, and thresholds are valid")
        return
    output_dir = _project_path(
        args.output_dir or config.get("output_dir", "finals_prediction/outputs/final_vote")
    )
    device = args.device or config.get("runtime", {}).get("device", "auto")
    run_inference(config, models, entries_by_model, test_index, output_dir, str(device))


if __name__ == "__main__":
    main()

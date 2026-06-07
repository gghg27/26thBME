from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any


TASK_SPECS = {
    "diagnosis": {
        "file": "diag_best.pt",
        "meta": "diag_meta.json",
        "module": "models.diagnosis_model",
        "class": "EmotionPretrainModel",
        "input_window": "10s",
        "index_csv": "com_index_sub_10s.csv",
        "label_map": {"HC": 0, "DEP": 1},
        "model_init_args": {
            "sfreq": 250.0,
            "topk": 8,
            "dropout": 0.2,
            "num_subjects": 60,
            "nclass": 2,
        },
    },
    "hc_emotion": {
        "file": "hc_best.pt",
        "meta": "hc_meta.json",
        "module": "models.hc_contrast_bio",
        "class": "EmotionPretrainModel",
        "input_window": "2s",
        "index_csv": "com_index_sub_2s.csv",
        "label_map": {"neutral": 0, "positive": 1},
        "model_init_args": {
            "sfreq": 250.0,
            "topk": 6,
            "dropout": 0.2,
            "num_subjects": 60,
            "emotion_nclass": 2,
            "diagnosis_nclass": 2,
            "contrast_dim": 64,
            "contrast_hidden_dim": 128,
        },
    },
    "dep_emotion": {
        "file": "dep_best.pt",
        "meta": "dep_meta.json",
        "module": "models.dep_contrast_bio",
        "class": "EmotionPretrainModel",
        "input_window": "2s",
        "index_csv": "com_index_sub_2s.csv",
        "label_map": {"neutral": 0, "positive": 1},
        "model_init_args": {
            "sfreq": 250.0,
            "topk": 6,
            "dropout": 0.2,
            "num_subjects": 60,
            "emotion_nclass": 2,
            "diagnosis_nclass": 2,
            "contrast_dim": 64,
            "contrast_hidden_dim": 128,
        },
    },
}


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def torch_load(path: Path, map_location: str = "cpu") -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load checkpoints. Install torch or provide meta json "
            "files so combine can run without inspecting checkpoint contents."
        ) from exc
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def save_checkpoint(
    path: Path,
    *,
    task: str,
    repeat: int,
    fold: int,
    epoch: int,
    best_metric_name: str,
    best_metric_value: float,
    model: Any,
    optimizer: Any | None = None,
    config: dict[str, Any] | None = None,
    label_map: dict[str, int] | None = None,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "task": task,
        "repeat": repeat,
        "fold": fold,
        "epoch": epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": float(best_metric_value),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config or {},
        "label_map": label_map or TASK_SPECS[task]["label_map"],
    }
    torch.save(checkpoint, path)


def build_meta(
    *,
    task: str,
    repeat: int,
    fold: int,
    model_file: str,
    model_class: str | None = None,
    model_module: str | None = None,
    best_epoch: int | None = None,
    best_metric_name: str | None = None,
    best_metric_value: float | None = None,
    train_subjects: list | None = None,
    val_subjects: list | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = TASK_SPECS[task]
    meta_config = dict(config or {})
    meta_config.setdefault("model_init_args", spec.get("model_init_args", {}))
    return {
        "task": task,
        "repeat": repeat,
        "fold": fold,
        "model_file": model_file,
        "model_class": model_class or spec["class"],
        "model_module": model_module or spec["module"],
        "input_window": spec["input_window"],
        "index_csv": spec["index_csv"],
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": None if best_metric_value is None else float(best_metric_value),
        "train_subjects": train_subjects or [],
        "val_subjects": val_subjects or [],
        "label_map": spec["label_map"],
        "config": meta_config,
    }


def load_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Meta file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_meta(meta: dict[str, Any], *, task: str, repeat: int, fold: int) -> None:
    if meta.get("task") != task:
        raise ValueError(f"Expected task {task}, got {meta.get('task')}")
    if int(meta.get("repeat", -1)) != repeat or int(meta.get("fold", -1)) != fold:
        raise ValueError(
            f"Meta repeat/fold mismatch for {task}: "
            f"{meta.get('repeat')}/{meta.get('fold')} != {repeat}/{fold}"
        )
    for key in ("model_file", "model_class", "model_module"):
        if not meta.get(key):
            raise ValueError(f"Meta for {task} is missing '{key}'")


def validate_checkpoint_meta(
    checkpoint: Any, *, task: str, repeat: int, fold: int, path: Path
) -> None:
    if not isinstance(checkpoint, dict):
        return
    if "task" in checkpoint and checkpoint["task"] != task:
        raise ValueError(f"{path} task mismatch: {checkpoint['task']} != {task}")
    if "repeat" in checkpoint and int(checkpoint["repeat"]) != repeat:
        raise ValueError(f"{path} repeat mismatch: {checkpoint['repeat']} != {repeat}")
    if "fold" in checkpoint and int(checkpoint["fold"]) != fold:
        raise ValueError(f"{path} fold mismatch: {checkpoint['fold']} != {fold}")


def instantiate_model(meta: dict[str, Any]):
    module = importlib.import_module(meta["model_module"])
    model_class = getattr(module, meta["model_class"])
    config = meta.get("config") or {}
    init_args = config.get("model_init_args") or config.get("init_args") or {}
    try:
        return model_class(**init_args)
    except TypeError:
        return model_class()

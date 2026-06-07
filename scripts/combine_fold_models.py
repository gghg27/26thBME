from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.checkpoint import (
    TASK_SPECS,
    build_meta,
    extract_state_dict,
    infer_init_args_from_state_dict,
    load_meta,
    save_json,
    torch_load,
    validate_checkpoint_meta,
    validate_meta,
)
from utils.folds import combined_model_param_dir, fold_name, model_param_dir, task_model_param_dir


TASK_ORDER = ("diagnosis", "hc_emotion", "dep_emotion")
TASK_PREFIX = {
    "diagnosis": "diag",
    "hc_emotion": "hc",
    "dep_emotion": "dep",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine three fold checkpoints into meta.")
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model_params_dir", type=Path, default=ROOT / "model_params")
    parser.add_argument("--diag_model_class", default="EmotionPretrainModel")
    parser.add_argument("--hc_model_class", default="EmotionPretrainModel")
    parser.add_argument("--dep_model_class", default="EmotionPretrainModel")
    parser.add_argument("--diag_model_module", default="models.diagnosis_model")
    parser.add_argument("--hc_model_module", default="models.hc_contrast_bio")
    parser.add_argument("--dep_model_module", default="models.dep_contrast_bio")
    return parser.parse_args()


def cli_model_info(args: argparse.Namespace, task: str) -> tuple[str, str]:
    if task == "diagnosis":
        return args.diag_model_class, args.diag_model_module
    if task == "hc_emotion":
        return args.hc_model_class, args.hc_model_module
    return args.dep_model_class, args.dep_model_module


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def synthesize_meta_from_checkpoint(
    *,
    checkpoint: Any,
    task: str,
    repeat: int,
    fold: int,
    model_file: str,
    model_class: str,
    model_module: str,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if isinstance(checkpoint, dict):
        task = checkpoint.get("task", task)
        config = dict(checkpoint.get("config") or {})
        best_epoch = checkpoint.get("epoch")
        metric_name = checkpoint.get("best_metric_name")
        metric_value = checkpoint.get("best_metric_value")
        if "label_map" in checkpoint:
            config.setdefault("checkpoint_label_map", checkpoint["label_map"])

        # ★ 从实际权重形状推断 model_init_args，覆盖 TASK_SPECS 默认值
        state_dict = extract_state_dict(checkpoint)
        if isinstance(state_dict, dict):
            inferred = infer_init_args_from_state_dict(state_dict)
            merged_init = dict(
                TASK_SPECS[task].get("model_init_args", {})
            )
            merged_init.update(inferred)  # 实际权重优先
            merged_init.update(config.get("model_init_args") or {})  # 训练时显式配置最高优先
            config["model_init_args"] = merged_init
    else:
        best_epoch = None
        metric_name = None
        metric_value = None

    return build_meta(
        task=task,
        repeat=repeat,
        fold=fold,
        model_file=model_file,
        model_class=model_class,
        model_module=model_module,
        best_epoch=best_epoch,
        best_metric_name=metric_name,
        best_metric_value=metric_value,
        config=config,
    )


def load_or_create_meta(
    *,
    args: argparse.Namespace,
    task: str,
    fold_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    spec = TASK_SPECS[task]
    ckpt_path = fold_dir / spec["file"]
    meta_path = fold_dir / spec["meta"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {task}: {ckpt_path}")

    model_class, model_module = cli_model_info(args, task)
    checkpoint = torch_load(ckpt_path)
    validate_checkpoint_meta(
        checkpoint, task=task, repeat=args.repeat, fold=args.fold, path=ckpt_path
    )

    # 从实际权重推断正确的 model_init_args
    state_dict = extract_state_dict(checkpoint)
    inferred = infer_init_args_from_state_dict(state_dict) if isinstance(state_dict, dict) else {}

    if meta_path.exists():
        meta = load_meta(meta_path)
        validate_meta(meta, task=task, repeat=args.repeat, fold=args.fold)
        # 检查已有 meta 的 model_init_args 与实际权重是否一致
        existing_init = (meta.get("config") or {}).get("model_init_args") or {}
        mismatches = {}
        for key, expected in inferred.items():
            actual = existing_init.get(key)
            if actual is not None and isinstance(expected, int) and int(actual) != expected:
                mismatches[key] = f"meta={actual} checkpoint={expected}"
        if mismatches:
            print(f"[combine] meta 与实际权重不一致，重新生成: {mismatches}")
            meta = synthesize_meta_from_checkpoint(
                checkpoint=checkpoint, task=task, repeat=args.repeat,
                fold=args.fold, model_file=spec["file"],
                model_class=model_class, model_module=model_module,
            )
            validate_meta(meta, task=task, repeat=args.repeat, fold=args.fold)
            save_json(meta, meta_path)
        return ckpt_path, meta_path, meta

    meta = synthesize_meta_from_checkpoint(
        checkpoint=checkpoint,
        task=task,
        repeat=args.repeat,
        fold=args.fold,
        model_file=spec["file"],
        model_class=model_class,
        model_module=model_module,
    )
    validate_meta(meta, task=task, repeat=args.repeat, fold=args.fold)
    save_json(meta, meta_path)
    print(f"[combine] synthesized missing meta: {meta_path}")
    return ckpt_path, meta_path, meta


def resolve_task_fold_dir(args: argparse.Namespace, task: str) -> Path:
    spec = TASK_SPECS[task]
    task_dir = task_model_param_dir(
        TASK_PREFIX[task], args.repeat, args.fold, args.model_params_dir
    )
    if (task_dir / spec["file"]).exists():
        return task_dir

    legacy_dir = model_param_dir(args.repeat, args.fold, args.model_params_dir)
    if (legacy_dir / spec["file"]).exists():
        print(f"[combine] using legacy directory for {task}: {legacy_dir}")
        return legacy_dir

    raise FileNotFoundError(
        f"Missing checkpoint for {task}. Tried: "
        f"{task_dir / spec['file']} and {legacy_dir / spec['file']}"
    )


def main() -> None:
    args = parse_args()

    model_entries = {}
    for task in TASK_ORDER:
        fold_dir = resolve_task_fold_dir(args, task)
        ckpt_path, meta_path, _ = load_or_create_meta(
            args=args, task=task, fold_dir=fold_dir
        )
        model_entries[task] = {"path": rel(ckpt_path), "meta": rel(meta_path)}
        print(f"[combine] ok {task}: {ckpt_path.name}")

    combined = {
        "repeat": args.repeat,
        "fold": args.fold,
        "status": "ready",
        "models": model_entries,
        "router": {
            "type": "soft",
            "formula": "p_final = (1 - p_dep_subject) * p_hc + p_dep_subject * p_dep",
        },
    }
    combined_dir = combined_model_param_dir(args.repeat, args.fold, args.model_params_dir)
    out_path = combined_dir / "combined_meta.json"
    save_json(combined, out_path)
    print(f"[combine] wrote {out_path} for {fold_name(args.repeat, args.fold)}")


if __name__ == "__main__":
    main()

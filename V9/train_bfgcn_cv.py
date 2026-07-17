"""Train V9 BF-GCN with diagnosis-stratified, subject-grouped 10-fold CV."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from V9.configs.bfgcn_config import BFGCNConfig
from V9.datasets.eeg_bfgcn_dataset import EEGBFGCNDataset
from V9.models.bfgcn import BFGCN
from V9.utils.checkpoint import load_checkpoint, save_checkpoint
from V9.utils.metrics import trial_and_subgroup_metrics
from V9.utils.seed import seed_worker, set_global_seed
from V9.utils.splitting import stratified_subject_splits
from V9.utils.trial_aggregation import aggregate_window_probabilities


def _batch_metadata(batch: dict[str, Any]) -> pd.DataFrame:
    size = int(batch["de"].shape[0])
    result: dict[str, list[Any]] = {}
    for key in (
        "subject_id", "user_id", "trial_id", "original_trial_id", "pseudo_trial_id",
        "window_id", "emotion_label", "diagnosis_label",
    ):
        value = batch[key]
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().tolist()
        else:
            result[key] = list(value)
        if len(result[key]) != size:
            raise ValueError(f"Batch metadata {key} length mismatch")
    return pd.DataFrame(result)


def _loader(
    dataset: EEGBFGCNDataset,
    batch_size: int,
    shuffle: bool,
    config: BFGCNConfig,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _class_weights(dataset: EEGBFGCNDataset, config: BFGCNConfig, device: torch.device) -> torch.Tensor | None:
    if config.class_weight is not None:
        if len(config.class_weight) != 2:
            raise ValueError("class_weight must have exactly two values")
        return torch.tensor(config.class_weight, dtype=torch.float32, device=device)
    if not config.auto_class_weight:
        return None
    counts = dataset.metadata["emotion_label"].value_counts().reindex([0, 1], fill_value=0).to_numpy(float)
    if np.any(counts == 0):
        raise ValueError(f"Cannot compute class weights with training counts {counts.tolist()}")
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _run_epoch(
    model: BFGCN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[float, dict[str, float], pd.DataFrame, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    all_probabilities: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    for batch in loader:
        de = batch["de"].to(device, non_blocking=True)
        plv = batch["plv"].to(device, non_blocking=True)
        labels = batch["emotion_label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(de, plv)["emotion_logits"]
                loss = criterion(logits, labels)
            if training:
                assert optimizer is not None and scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
        probabilities = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
        all_probabilities.append(probabilities)
        metadata_parts.append(_batch_metadata(batch))
        total_loss += float(loss.detach()) * len(labels)
        total_samples += len(labels)
    if not total_samples:
        raise ValueError("DataLoader produced no samples")
    probabilities = np.concatenate(all_probabilities)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    labels_np = metadata["emotion_label"].to_numpy(dtype=int)
    predictions = probabilities.argmax(axis=1)
    window_metrics = {
        "accuracy": float(accuracy_score(labels_np, predictions)),
        "macro_f1": float(f1_score(labels_np, predictions, labels=[0, 1], average="macro", zero_division=0)),
    }
    trials = aggregate_window_probabilities(probabilities, metadata)
    trial_metrics = trial_and_subgroup_metrics(trials)
    return total_loss / total_samples, window_metrics, trials, trial_metrics


def run_smoke_test(config: BFGCNConfig, dataset: EEGBFGCNDataset, device: torch.device) -> dict[str, Any]:
    """Run real-data forward/loss/backward/step and trial aggregation before CV."""
    indices = list(range(min(2, len(dataset))))
    if len(indices) < 2:
        raise ValueError("Smoke test requires at least two windows")
    batch = next(iter(DataLoader(Subset(dataset, indices), batch_size=len(indices), num_workers=0)))
    model = BFGCN(deepcopy(config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    model.train()
    outputs = model(batch["de"].to(device), batch["plv"].to(device))
    logits = outputs["emotion_logits"]
    loss = criterion(logits, batch["emotion_label"].to(device))
    loss.backward()
    optimizer.step()
    probabilities = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
    trials = aggregate_window_probabilities(probabilities, _batch_metadata(batch))
    branch_count = int(config.use_learnable_graph) + int(config.use_functional_graph) + int(config.use_common_branch)
    expected = {
        "emotion_logits": (len(indices), 2),
        "functional_adj": (len(indices), config.num_channels, config.num_channels),
        "band_attention": (len(indices), len(config.frequency_bands)),
        "branch_attention": (len(indices), branch_count, config.num_channels),
    }
    if tuple(logits.shape) != expected["emotion_logits"]:
        raise AssertionError(f"Smoke logits {tuple(logits.shape)} != {expected['emotion_logits']}")
    if config.use_learnable_graph and tuple(outputs["learnable_adj"].shape) != (
        config.num_channels, config.num_channels
    ):
        raise AssertionError("Smoke learnable adjacency shape mismatch")
    if config.use_functional_graph:
        for key in ("functional_adj", "band_attention"):
            if tuple(outputs[key].shape) != expected[key]:
                raise AssertionError(f"Smoke {key} {tuple(outputs[key].shape)} != {expected[key]}")
    if tuple(outputs["branch_attention"].shape) != expected["branch_attention"]:
        raise AssertionError("Smoke branch attention does not match enabled branches")
    result = {
        "loss": float(loss.detach()),
        "emotion_logits_shape": list(logits.shape),
        "learnable_adj_shape": None if outputs["learnable_adj"] is None else list(outputs["learnable_adj"].shape),
        "functional_adj_shape": None if outputs["functional_adj"] is None else list(outputs["functional_adj"].shape),
        "band_attention_shape": None if outputs["band_attention"] is None else list(outputs["band_attention"].shape),
        "branch_attention_shape": list(outputs["branch_attention"].shape),
        "aggregated_trials": len(trials),
        "status": "passed",
    }
    print("SMOKE_TEST " + json.dumps(result, ensure_ascii=False))
    return result


def _checkpoint_root(config: BFGCNConfig, experiment: str) -> Path:
    root = Path(config.checkpoint_dir)
    return root if experiment == "full_bfgcn" else root / experiment


def _result_root(config: BFGCNConfig, experiment: str) -> Path:
    root = Path(config.result_dir)
    return root if experiment == "full_bfgcn" else root / experiment


def train_fold(
    fold_index: int,
    train_subjects: list[str],
    val_subjects: list[str],
    config: BFGCNConfig,
    experiment: str,
    device: torch.device,
) -> dict[str, Any]:
    run_seed = config.seed + fold_index
    set_global_seed(run_seed)
    overlap = set(train_subjects).intersection(val_subjects)
    if overlap:
        raise AssertionError(f"Subject leakage before fold {fold_index + 1}: {sorted(overlap)}")
    train_data = EEGBFGCNDataset(config.train_index_csv, train_subjects, config, config.cache_plv)
    val_data = EEGBFGCNDataset(config.train_index_csv, val_subjects, config, config.cache_plv)
    train_loader = _loader(train_data, config.batch_size, True, config, run_seed)
    val_loader = _loader(val_data, config.batch_size, False, config, run_seed)

    model = BFGCN(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.scheduler == "cosine"
        else None
    )
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(train_data, config, device), label_smoothing=config.label_smoothing
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")
    best_score = (-1.0, -1.0)
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    best_trials: pd.DataFrame | None = None
    patience = 0
    checkpoint_path = _checkpoint_root(config, experiment) / f"fold_{fold_index + 1:02d}" / "best_model.pt"
    for epoch in range(1, config.epochs + 1):
        train_loss, train_window, _, _ = _run_epoch(
            model, train_loader, criterion, device, optimizer, config.gradient_clip, config.amp, scaler
        )
        with torch.no_grad():
            val_loss, val_window, val_trials, val_metrics = _run_epoch(
                model, val_loader, criterion, device, None, config.gradient_clip, config.amp, None
            )
        if scheduler is not None:
            scheduler.step()
        overall = val_metrics["overall"]
        score = (float(overall["macro_f1"]), float(overall["accuracy"]))
        log = {
            "fold": fold_index + 1,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_window_accuracy": train_window["accuracy"],
            "train_window_macro_f1": train_window["macro_f1"],
            "val_loss": val_loss,
            "window_accuracy": val_window["accuracy"],
            "window_macro_f1": val_window["macro_f1"],
            "trial_accuracy": overall["accuracy"],
            "trial_macro_f1": overall["macro_f1"],
            "HC_trial_accuracy": val_metrics["HC"]["accuracy"],
            "HC_trial_macro_f1": val_metrics["HC"]["macro_f1"],
            "DEP_trial_accuracy": val_metrics["DEP"]["accuracy"],
            "DEP_trial_macro_f1": val_metrics["DEP"]["macro_f1"],
            "overall_confusion_matrix": overall["confusion_matrix"],
            "HC_neutral_recall": val_metrics["HC"]["neutral_recall"],
            "HC_positive_recall": val_metrics["HC"]["positive_recall"],
            "HC_confusion_matrix": val_metrics["HC"]["confusion_matrix"],
            "DEP_neutral_recall": val_metrics["DEP"]["neutral_recall"],
            "DEP_positive_recall": val_metrics["DEP"]["positive_recall"],
            "DEP_confusion_matrix": val_metrics["DEP"]["confusion_matrix"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(log, ensure_ascii=False))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = {"window": val_window, "trial": val_metrics, "val_loss": val_loss}
            best_trials = val_trials.copy()
            patience = 0
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": config.to_dict(),
                    "experiment": experiment,
                    "best_trial_macro_f1": score[0],
                    "best_trial_accuracy": score[1],
                    "train_subjects": train_subjects,
                    "val_subjects": val_subjects,
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print(f"fold={fold_index + 1} early stopping at epoch={epoch}")
                break
    if best_metrics is None or best_trials is None:
        raise RuntimeError("No best validation state was selected")
    result_root = _result_root(config, experiment)
    result_root.mkdir(parents=True, exist_ok=True)
    best_trials.to_csv(result_root / f"fold_{fold_index + 1:02d}_predictions.csv", index=False)
    result = {
        "fold": fold_index + 1,
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        **best_metrics,
    }
    (result_root / f"fold_{fold_index + 1:02d}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def summarize_cv(results: list[dict[str, Any]], config: BFGCNConfig, experiment: str) -> None:
    metrics = {
        "trial_accuracy": [item["trial"]["overall"]["accuracy"] for item in results],
        "trial_macro_f1": [item["trial"]["overall"]["macro_f1"] for item in results],
        "HC_trial_accuracy": [item["trial"]["HC"]["accuracy"] for item in results],
        "HC_trial_macro_f1": [item["trial"]["HC"]["macro_f1"] for item in results],
        "DEP_trial_accuracy": [item["trial"]["DEP"]["accuracy"] for item in results],
        "DEP_trial_macro_f1": [item["trial"]["DEP"]["macro_f1"] for item in results],
    }
    summary = {
        name: {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
        for name, values in metrics.items()
    }
    root = _result_root(config, experiment)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"metric": key, **value} for key, value in summary.items()]).to_csv(
        root / "cv_summary.csv", index=False
    )
    (root / "cv_summary.json").write_text(
        json.dumps({"summary": summary, "folds": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _bool(value: str | int) -> bool:
    return bool(int(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="full_bfgcn")
    parser.add_argument("--train-index", default=None)
    parser.add_argument("--fold", type=int, default=None, help="1-based fold; omit for all ten folds")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-plv-cache", action="store_true")
    parser.add_argument("--use-learnable-graph", "--use_learnable_graph", type=_bool, default=True)
    parser.add_argument("--use-functional-graph", "--use_functional_graph", type=_bool, default=True)
    parser.add_argument("--use-common-branch", "--use_common_branch", type=_bool, default=True)
    parser.add_argument("--use-band-attention", type=_bool, default=True)
    parser.add_argument("--use-branch-attention", type=_bool, default=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", type=_bool, default=True)
    parser.add_argument("--auto-class-weight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BFGCNConfig(
        use_learnable_graph=args.use_learnable_graph,
        use_functional_graph=args.use_functional_graph,
        use_common_branch=args.use_common_branch,
        use_band_attention=args.use_band_attention,
        use_branch_attention=args.use_branch_attention,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        early_stopping_patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=args.amp,
        auto_class_weight=args.auto_class_weight,
        cache_plv=not args.no_plv_cache,
    )
    if args.train_index:
        config.train_index_csv = str(Path(args.train_index).resolve())
    set_global_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}")
    full_data = EEGBFGCNDataset(config.train_index_csv, config=config, cache_plv=config.cache_plv)
    smoke = run_smoke_test(config, full_data, device)
    smoke_path = _result_root(config, args.experiment) / "smoke_test.json"
    smoke_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_path.write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.smoke_only:
        return
    splits = stratified_subject_splits(full_data.subject_diagnoses, config.n_splits, config.seed)
    selected = range(config.n_splits) if args.fold is None else [args.fold - 1]
    if any(index < 0 or index >= config.n_splits for index in selected):
        raise ValueError(f"--fold must be between 1 and {config.n_splits}")
    results = [
        train_fold(index, splits[index][0], splits[index][1], config, args.experiment, device)
        for index in selected
    ]
    summarize_cv(results, config, args.experiment)


if __name__ == "__main__":
    main()

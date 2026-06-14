# -*- coding: utf-8 -*-
"""Train V6 dual-stream subject-level diagnosis + trial-level emotion model."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from common import natural_key
from dual_stream_dataset import DualStreamSubjectSetDataset, dict_collate
from dual_stream_subject_model import (
    DualStreamSubjectEmotionModel,
    hard_expert_emotion_loss,
    mixture_emotion_nll_loss,
    subject_ranking_loss,
)
from utils.folds import get_unified_subject_split


VERSION_NAME = "V6_DualStream_SubjectRoute"
BEST_CRITERIA = [
    ("topk_trial_macro_f1", "max"),
    ("topk_trial_acc", "max"),
    ("trial_macro_f1", "max"),
    ("trial_acc", "max"),
    ("loss", "min"),
]


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    return obj


def save_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def flatten_metrics_for_csv(prefix: str, metrics: dict) -> dict:
    row: dict[str, float] = {}
    for key, value in metrics.items():
        if key in {"trial_records", "topk_records"}:
            continue
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, (int, float, np.integer, np.floating)):
            row[name] = float(value)
        elif isinstance(value, np.ndarray) and value.ndim == 2 and "confusion_matrix" in key:
            for i in range(value.shape[0]):
                for j in range(value.shape[1]):
                    row[f"{name}_{i}_{j}"] = float(value[i, j])
    return row


def make_compare_key(metrics: dict, criteria: list[tuple[str, str]]) -> tuple[float, ...]:
    values = []
    for metric_name, mode in criteria:
        value = metrics.get(metric_name)
        if value is None:
            value = -1e18 if mode == "max" else 1e18
        value = float(value)
        values.append(value if mode == "max" else -value)
    return tuple(values)


def is_better(current: dict, best: dict | None, criteria: list[tuple[str, str]] = BEST_CRITERIA, eps: float = 1e-12) -> bool:
    if best is None:
        return True
    cur_key = make_compare_key(current, criteria)
    best_key = make_compare_key(best, criteria)
    for cur_value, best_value in zip(cur_key, best_key):
        if cur_value > best_value + eps:
            return True
        if cur_value < best_value - eps:
            return False
    return False


def format_criteria(criteria: list[tuple[str, str]] = BEST_CRITERIA) -> str:
    return " -> ".join(f"{name}({mode})" for name, mode in criteria)


def apply_subject_topk(trial_records: list[dict], k_pos: int = 4, verbose: bool = False) -> list[dict]:
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for record in trial_records:
        by_subject[str(record["user_id"])].append(dict(record))

    out: list[dict] = []
    for user_id, records in by_subject.items():
        records = sorted(records, key=lambda row: float(row.get("score_pos", 0.0)), reverse=True)
        cur_k = int(k_pos) if len(records) == 8 else max(1, int(round(len(records) * 0.5)))
        cur_k = max(0, min(cur_k, len(records)))
        positive = {(row["user_id"], int(row["trial_id"])) for row in records[:cur_k]}
        pos_count = 0
        for row in records:
            row["Emotion_label"] = 1 if (row["user_id"], int(row["trial_id"])) in positive else 0
            pos_count += int(row["Emotion_label"])
            out.append(row)
        if verbose:
            print(f"[topk] user={user_id} trials={len(records)} positive={pos_count} expected={cur_k} ok={pos_count == cur_k}")

    return sorted(out, key=lambda row: (natural_key(row["user_id"]), int(row["trial_id"])))


def compute_record_metrics(
    trial_records: list[dict],
    diag_records: list[dict],
    loss_totals: dict,
    total_subjects: int,
    k_pos: int,
    verbose_topk: bool = False,
) -> dict:
    y_true = [int(row["label_emo"]) for row in trial_records if int(row["label_emo"]) >= 0]
    y_pred = [int(row["pred_emo"]) for row in trial_records if int(row["label_emo"]) >= 0]
    diag_true = [int(row["label_diag"]) for row in diag_records if int(row["label_diag"]) >= 0]
    diag_pred = [int(row["pred_diag"]) for row in diag_records if int(row["label_diag"]) >= 0]

    topk_records = apply_subject_topk(trial_records, k_pos=k_pos, verbose=verbose_topk)
    topk_true = [int(row["label_emo"]) for row in topk_records if int(row["label_emo"]) >= 0]
    topk_pred = [int(row["Emotion_label"]) for row in topk_records if int(row["label_emo"]) >= 0]

    metrics = {
        "loss": loss_totals["loss"] / max(total_subjects, 1),
        "loss_emo": loss_totals["loss_emo"] / max(total_subjects, 1),
        "loss_diag": loss_totals["loss_diag"] / max(total_subjects, 1),
        "loss_expert": loss_totals["loss_expert"] / max(total_subjects, 1),
        "loss_rank": loss_totals["loss_rank"] / max(total_subjects, 1),
        "trial_acc": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "trial_macro_f1": f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0) if y_true else 0.0,
        "trial_confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]) if y_true else np.zeros((2, 2), dtype=int),
        "topk_trial_acc": accuracy_score(topk_true, topk_pred) if topk_true else 0.0,
        "topk_trial_macro_f1": f1_score(topk_true, topk_pred, average="macro", labels=[0, 1], zero_division=0) if topk_true else 0.0,
        "topk_trial_confusion_matrix": confusion_matrix(topk_true, topk_pred, labels=[0, 1]) if topk_true else np.zeros((2, 2), dtype=int),
        "diag_acc": accuracy_score(diag_true, diag_pred) if diag_true else 0.0,
        "diag_macro_f1": f1_score(diag_true, diag_pred, average="macro", labels=[0, 1], zero_division=0) if diag_true else 0.0,
        "diag_confusion_matrix": confusion_matrix(diag_true, diag_pred, labels=[0, 1]) if diag_true else np.zeros((2, 2), dtype=int),
        "trial_records": trial_records,
        "topk_records": topk_records,
    }
    return metrics


def make_trial_records(batch: dict, out: dict) -> tuple[list[dict], list[dict]]:
    mix_prob = out["mix_prob"].clamp_min(1e-8)
    prob_pos = mix_prob[..., 1]
    score_pos = torch.log(mix_prob[..., 1]) - torch.log(mix_prob[..., 0])
    pred_emo = mix_prob.argmax(dim=-1)
    pred_diag = out["diag_logits"].argmax(dim=-1)

    trial_records: list[dict] = []
    diag_records: list[dict] = []
    batch_size, n_trials = pred_emo.shape
    for i in range(batch_size):
        user_id = str(batch["user_id"][i])
        label_diag = int(batch["diagnosis_label"][i].detach().cpu())
        pred_diag_i = int(pred_diag[i].detach().cpu())
        diag_records.append(
            {
                "user_id": user_id,
                "subject_id": int(batch["subject_id"][i].detach().cpu()),
                "label_diag": label_diag,
                "pred_diag": pred_diag_i,
                "diag_prob_dep": float(out["diag_prob"][i, 0].detach().cpu()),
                "diag_prob_hc": float(out["diag_prob"][i, 1].detach().cpu()),
            }
        )
        for j in range(n_trials):
            trial_records.append(
                {
                    "user_id": user_id,
                    "subject_id": int(batch["subject_id"][i].detach().cpu()),
                    "trial_id": int(batch["trial_id"][i, j].detach().cpu()),
                    "prob_neu": float(mix_prob[i, j, 0].detach().cpu()),
                    "prob_pos": float(prob_pos[i, j].detach().cpu()),
                    "score_pos": float(score_pos[i, j].detach().cpu()),
                    "pred_emo": int(pred_emo[i, j].detach().cpu()),
                    "label_emo": int(batch["emotion_label"][i, j].detach().cpu()),
                    "pred_diag": pred_diag_i,
                    "label_diag": label_diag,
                }
            )
    return trial_records, diag_records


def forward_model(model: torch.nn.Module, batch: dict) -> dict:
    return model(
        x_abs=batch["x_abs"],
        x_rel=batch["x_rel"],
        de_abs=batch["de_abs"],
        de_z=batch["de_z"],
        win_mask=batch["win_mask"],
    )


def autocast_context(device: torch.device, amp_enabled: bool):
    if amp_enabled and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def compute_loss(out: dict, batch: dict, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    y_emo = batch["emotion_label"].long()
    y_diag = batch["diagnosis_label"].long()
    loss_emo = mixture_emotion_nll_loss(out["mix_prob"], y_emo)
    loss_diag = F.cross_entropy(out["diag_logits"], y_diag)
    loss_rank = subject_ranking_loss(out["mix_prob"], y_emo, margin=args.rank_margin)
    if float(args.lambda_expert) > 0:
        loss_expert = hard_expert_emotion_loss(out["hc_logits"], out["dep_logits"], y_emo, y_diag)
    else:
        loss_expert = out["mix_prob"].new_tensor(0.0)
    loss = (
        loss_emo
        + float(args.lambda_expert) * loss_expert
        + float(args.lambda_diag) * loss_diag
        + float(args.lambda_rank) * loss_rank
    )
    return loss, {
        "loss_emo": loss_emo,
        "loss_diag": loss_diag,
        "loss_expert": loss_expert,
        "loss_rank": loss_rank,
    }


def train_one_epoch(model, loader, optimizer, device, args, scaler=None) -> dict:
    model.train()
    totals = defaultdict(float)
    total_subjects = 0
    trial_records: list[dict] = []
    diag_records: list[dict] = []

    pbar = tqdm(loader, desc="V6-dual-train", leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        amp_enabled = bool(args.amp and device.type == "cuda")
        with autocast_context(device, amp_enabled):
            out = forward_model(model, batch)
            loss, loss_parts = compute_loss(out, batch, args)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()

        batch_subjects = int(batch["x_abs"].size(0))
        total_subjects += batch_subjects
        totals["loss"] += float(loss.item()) * batch_subjects
        for key, value in loss_parts.items():
            totals[key] += float(value.item()) * batch_subjects

        with torch.no_grad():
            records, diag = make_trial_records(batch, out)
        trial_records.extend(records)
        diag_records.extend(diag)
        pbar.set_postfix({"loss": f"{totals['loss'] / max(total_subjects, 1):.4f}"})

    return compute_record_metrics(
        trial_records,
        diag_records,
        totals,
        total_subjects,
        k_pos=args.k_pos,
        verbose_topk=False,
    )


@torch.no_grad()
def validate(model, loader, device, args, save_dir: Path | None = None) -> dict:
    model.eval()
    totals = defaultdict(float)
    total_subjects = 0
    trial_records: list[dict] = []
    diag_records: list[dict] = []

    for batch in tqdm(loader, desc="V6-dual-val", leave=False):
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, bool(args.amp and device.type == "cuda")):
            out = forward_model(model, batch)
            loss, loss_parts = compute_loss(out, batch, args)
        batch_subjects = int(batch["x_abs"].size(0))
        total_subjects += batch_subjects
        totals["loss"] += float(loss.item()) * batch_subjects
        for key, value in loss_parts.items():
            totals[key] += float(value.item()) * batch_subjects
        records, diag = make_trial_records(batch, out)
        trial_records.extend(records)
        diag_records.extend(diag)

    metrics = compute_record_metrics(
        trial_records,
        diag_records,
        totals,
        total_subjects,
        k_pos=args.k_pos,
        verbose_topk=True,
    )
    if save_dir is not None:
        records_df = pd.DataFrame(metrics["trial_records"])
        topk_df = pd.DataFrame(metrics["topk_records"])[["user_id", "trial_id", "Emotion_label"]]
        topk_df.rename(columns={"Emotion_label": "pred_topk"}, inplace=True)
        records_df = records_df.merge(topk_df, on=["user_id", "trial_id"], how="left")
        records_df.to_csv(save_dir / "val_trial_records.csv", index=False, encoding="utf-8-sig")
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    *,
    epoch: int,
    fold: int,
    train_metrics: dict,
    val_metrics: dict,
    best_metrics: dict,
    config_dict: dict,
    train_subjects: list[str],
    val_subjects: list[str],
) -> None:
    ckpt = {
        "version": VERSION_NAME,
        "epoch": int(epoch),
        "fold": int(fold),
        "criteria": BEST_CRITERIA,
        "criteria_readable": format_criteria(BEST_CRITERIA),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": to_jsonable(config_dict),
        "train_subjects": to_jsonable(train_subjects),
        "val_subjects": to_jsonable(val_subjects),
        "train_metrics": to_jsonable(train_metrics),
        "val_metrics": to_jsonable(val_metrics),
        "best_metrics": to_jsonable(best_metrics),
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(args_or_config: argparse.Namespace | dict) -> DualStreamSubjectEmotionModel:
    cfg = vars(args_or_config) if isinstance(args_or_config, argparse.Namespace) else dict(args_or_config)
    return DualStreamSubjectEmotionModel(
        sfreq=float(cfg.get("sfreq", 250.0)),
        topk=int(cfg.get("topk", 8)),
        dropout=float(cfg.get("dropout", 0.35)),
        use_biomarkers=not bool(cfg.get("no_biomarkers", False)),
        biomarker_dim=int(cfg.get("biomarker_dim", 57)),
        de_num_bands=int(cfg.get("de_num_bands", 5)),
        shared_mix_alpha=float(cfg.get("shared_mix_alpha", 0.5)),
        share_abs_rel_encoder=bool(cfg.get("share_abs_rel_encoder", False)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        relative_eps=float(cfg.get("relative_eps", 1e-6)),
        encode_chunk_size=int(cfg.get("encode_chunk_size", 0)),
    )


def build_loaders(args, split: dict, run_seed: int) -> tuple[DualStreamSubjectSetDataset, DualStreamSubjectSetDataset, DataLoader, DataLoader]:
    train_dataset = DualStreamSubjectSetDataset(
        args.index_csv,
        subject_ids=split["train_all"],
        mode="train",
        root=ROOT,
        eps=args.relative_eps,
        check_paths=True,
        debug=True,
    )
    val_dataset = DualStreamSubjectSetDataset(
        args.index_csv,
        subject_ids=split["val_all"],
        mode="train",
        root=ROOT,
        max_windows=train_dataset.max_windows,
        eps=args.relative_eps,
        check_paths=True,
        debug=True,
    )
    generator = torch.Generator()
    generator.manual_seed(run_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dict_collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dict_collate,
        worker_init_fn=seed_worker,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def run_one_fold(args: argparse.Namespace, fold: int, base_seed: int) -> dict:
    run_seed = int(getattr(config, "make_run_seed", lambda seed, f: seed * 1000 + f)(base_seed, fold))
    set_global_seed(run_seed, deterministic=args.deterministic)
    requested = torch.device(args.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    if requested.type == "cuda" and device.type == "cpu":
        print(f"[device] requested {args.device}, but CUDA is unavailable; using CPU.")

    save_dir = Path(args.save_root) / f"fold{fold}"
    save_dir.mkdir(parents=True, exist_ok=True)
    split = get_unified_subject_split(args.index_csv, fold=fold, n_splits=args.n_splits, seed=base_seed)
    print(
        f"[fold] fold={fold} base_seed={base_seed} run_seed={run_seed} "
        f"train_subjects={len(split['train_all'])} val_subjects={len(split['val_all'])}"
    )
    print(f"[fold] train subjects head={split['train_all'][:8]} val subjects head={split['val_all'][:8]}")

    config_dict = vars(args).copy()
    config_dict.update({"version": VERSION_NAME, "fold": fold, "base_seed": base_seed, "run_seed": run_seed})
    save_json(save_dir / "config.json", config_dict)

    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args, split, run_seed)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    print(
        f"[model] encoder_dim={model.encoder_dim} share_abs_rel_encoder={model.share_abs_rel_encoder} "
        f"shared_mix_alpha={model.shared_mix_alpha} encode_chunk_size={model.encode_chunk_size} "
        f"amp={bool(args.amp and device.type == 'cuda')}"
    )
    print(f"[best] criteria={format_criteria(BEST_CRITERIA)}")

    history: list[dict] = []
    best_metrics: dict | None = None
    best_epoch = 0
    best_path = save_dir / "best_checkpoint.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args, scaler=scaler)
        val_metrics = validate(model, val_loader, device, args, save_dir=save_dir)
        scheduler.step()

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update(flatten_metrics_for_csv("train", train_metrics))
        row.update(flatten_metrics_for_csv("val", val_metrics))
        history.append(row)
        pd.DataFrame(history).to_csv(save_dir / "train_history.csv", index=False, encoding="utf-8-sig")

        print(
            f"[epoch {epoch}] loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_trial_acc={val_metrics['trial_acc']:.4f} "
            f"val_trial_f1={val_metrics['trial_macro_f1']:.4f} "
            f"val_topk_acc={val_metrics['topk_trial_acc']:.4f} "
            f"val_topk_f1={val_metrics['topk_trial_macro_f1']:.4f} "
            f"val_diag_acc={val_metrics['diag_acc']:.4f} "
            f"val_diag_f1={val_metrics['diag_macro_f1']:.4f}"
        )

        if is_better(val_metrics, best_metrics):
            best_metrics = copy.deepcopy(val_metrics)
            best_epoch = epoch
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                fold=fold,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_metrics=best_metrics,
                config_dict=config_dict,
                train_subjects=split["train_all"],
                val_subjects=split["val_all"],
            )
            save_json(save_dir / "best_metrics.json", {"best_epoch": best_epoch, "best_metrics": best_metrics})
            print(f"[best] updated epoch={epoch}, path={best_path}")

    if not best_path.exists() and args.epochs == 0:
        raise RuntimeError("No training epochs ran and no best checkpoint exists.")

    result = {
        "fold": fold,
        "base_seed": base_seed,
        "run_seed": run_seed,
        "save_dir": str(save_dir),
        "best_epoch": best_epoch,
        "best_path": str(best_path) if best_path.exists() else "",
        "best_metrics": best_metrics,
        "train_subjects": split["train_all"],
        "val_subjects": split["val_all"],
    }

    if args.predict_test and args.test_csv:
        ckpt_path = Path(args.test_ckpt) if args.test_ckpt else best_path
        from predict_dual_stream_subject import predict_from_checkpoint

        pred_dir = save_dir / "test_predict"
        predict_from_checkpoint(
            ckpt_path=ckpt_path,
            test_csv=args.test_csv,
            save_dir=pred_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold=args.threshold,
            k_pos=args.k_pos,
        )
        result["test_predict_dir"] = str(pred_dir)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=VERSION_NAME)
    parser.add_argument("--index_csv", type=str, default="data/com_dual_stream_window_index_10s.csv")
    parser.add_argument("--test_csv", type=str, default="data/com_test_dual_stream_window_index_10s.csv")
    parser.add_argument("--save_root", type=str, default="outputs/V6_dual_stream")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--biomarker_dim", type=int, default=57)
    parser.add_argument("--de_num_bands", type=int, default=5)
    parser.add_argument("--no_biomarkers", action="store_true")
    parser.add_argument("--share_abs_rel_encoder", action="store_true")
    parser.add_argument(
        "--encode_chunk_size",
        type=int,
        default=0,
        help="Forward encoder windows in chunks. 0 means all windows at once; use 1/2/4 to reduce CUDA memory.",
    )
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision to reduce memory.")
    parser.add_argument("--relative_eps", type=float, default=1e-6)
    parser.add_argument("--lambda_diag", type=float, default=0.2)
    parser.add_argument("--lambda_expert", type=float, default=0.5)
    parser.add_argument("--lambda_rank", type=float, default=0.1)
    parser.add_argument("--rank_margin", type=float, default=0.2)
    parser.add_argument("--shared_mix_alpha", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--k_pos", type=int, default=4)
    parser.add_argument("--predict_test", action="store_true")
    parser.add_argument("--test_ckpt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = range(args.n_splits) if args.all_folds else [args.fold]
    results = [run_one_fold(args, fold=fold, base_seed=args.seed) for fold in folds]
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        row = {
            "fold": result["fold"],
            "base_seed": result["base_seed"],
            "run_seed": result["run_seed"],
            "save_dir": result["save_dir"],
            "best_epoch": result["best_epoch"],
            "best_path": result["best_path"],
        }
        if result.get("best_metrics"):
            row.update(flatten_metrics_for_csv("best", result["best_metrics"]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(save_root / "all_fold_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[done] summary saved: {save_root / 'all_fold_summary.csv'}")


if __name__ == "__main__":
    main()

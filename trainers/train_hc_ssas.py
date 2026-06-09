# -*- coding: utf-8 -*-
"""
SSAS 框架 HC 情绪二分类训练脚本。

Stage 1: 训练 SourceSelectionModel，计算每个 HC 训练被试的 source_weight
Stage 2: 用 source_weight 加权 + GRL 域对抗 + trial compactness 训练 SSASEmotionModel
Stage 3 (可选): 外加 MMD loss 显式拉近源-目标特征分布

对应模型: models/hc_ssas.py
"""

import os
import json
import copy
import random
import sys
from collections import defaultdict
from itertools import cycle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from dataloader import (
    EEGWindowDataset,
    seed_collate_fn,
    MultiSubjectBatchSampler,
    Competition4ClassDataset,
    comp4_collate_fn,
)
from models.hc_ssas import (
    SourceSelectionModel,
    SSASEmotionModel,
    mmd_loss,
)
from utils.folds import get_unified_subject_split


# =========================================================
# Random seed utilities
# =========================================================

def set_global_seed(seed: int, deterministic: bool = True):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================================================
# Label utilities
# =========================================================

def normalize_train_group(train_group: str) -> str:
    group = str(train_group).lower().strip()
    if group in ["hc", "normal", "healthy"]:
        return "hc"
    if group in ["dep", "depression", "mdd"]:
        return "dep"
    if group in ["all", "both", "mix"]:
        return "all"
    raise ValueError(f"Unknown train_group: {train_group}")


def get_emotion_label(batch: dict, device: torch.device) -> torch.Tensor:
    """从 batch 中取情绪二分类标签: 0=中性, 1=正性。"""
    if "emotion_label" in batch:
        return batch["emotion_label"].to(device).long()
    return (batch["label4"].to(device).long() % 2).long()


# =========================================================
# Trial compactness loss
# =========================================================

def make_subject_trial_group_ids(batch: dict, device: torch.device) -> torch.Tensor:
    subject_ids = batch["subject_id"].to(device).long()
    trial_ids = batch["trial_id"].to(device).long()
    return subject_ids * 100000 + trial_ids


def trial_compactness_loss(
    features: torch.Tensor,
    trial_group_ids: torch.Tensor,
) -> torch.Tensor:
    """Pull windows from the same subject-trial close to their trial center."""
    if features.ndim != 2:
        raise ValueError(f"features should be [B, D], got {tuple(features.shape)}")
    if features.size(0) <= 1:
        return features.new_tensor(0.0)

    features = F.normalize(features, p=2, dim=1)
    trial_group_ids = trial_group_ids.view(-1)

    losses = []
    for gid in trial_group_ids.unique():
        mask = trial_group_ids == gid
        if int(mask.sum().item()) < 2:
            continue
        trial_feat = features[mask]
        center = F.normalize(trial_feat.mean(dim=0, keepdim=True), p=2, dim=1)
        cosine_to_center = (trial_feat * center).sum(dim=1)
        losses.append((1.0 - cosine_to_center).mean())

    if len(losses) == 0:
        return features.new_tensor(0.0)
    return torch.stack(losses).mean()


# =========================================================
# Class weights
# =========================================================

def build_emotion_class_weights(index_csv: str, subject_ids):
    df = pd.read_csv(index_csv)
    df = df[df["subject_id"].isin(subject_ids)].reset_index(drop=True)
    if "emotion_label" in df.columns:
        emo = df["emotion_label"].astype(int)
    else:
        emo = (df["label4"].astype(int) % 2)
    counts = emo.value_counts().sort_index().reindex([0, 1], fill_value=1)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights.values, dtype=torch.float32)


# =========================================================
# Metrics utilities
# =========================================================

def compute_trial_level_metrics(segment_records, threshold=0.5, vote_method="hard"):
    trial_dict = defaultdict(list)
    for r in segment_records:
        key = (int(r["subject_id"]), int(r["trial_id"]))
        trial_dict[key].append(r)

    trial_probs, trial_preds, trial_labels, trial_keys = [], [], [], []
    for key, records in trial_dict.items():
        probs = [float(r["prob_pos"]) for r in records]
        preds = [int(r["pred_emo"]) for r in records]
        labels = [int(r["label_emo"]) for r in records]

        label = int(round(np.mean(labels)))
        mean_prob = float(np.mean(probs))

        if vote_method == "prob":
            pred = 1 if mean_prob >= threshold else 0
        elif vote_method == "hard":
            num_pos = sum(preds)
            if num_pos > len(preds) - num_pos:
                pred = 1
            else:
                pred = 0
        else:
            raise ValueError(f"Unknown vote_method: {vote_method}")

        trial_keys.append(key)
        trial_probs.append(mean_prob)
        trial_preds.append(pred)
        trial_labels.append(label)

    trial_acc = accuracy_score(trial_labels, trial_preds)
    trial_macro_f1 = f1_score(trial_labels, trial_preds, average="macro", labels=[0, 1], zero_division=0)
    trial_cm = confusion_matrix(trial_labels, trial_preds, labels=[0, 1])
    return {
        "trial_acc": trial_acc, "trial_macro_f1": trial_macro_f1,
        "trial_confusion_matrix": trial_cm, "trial_keys": trial_keys,
        "trial_probs": trial_probs, "trial_preds": trial_preds, "trial_labels": trial_labels,
    }


def to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, list):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    return obj


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(obj), f, ensure_ascii=False, indent=2)


def flatten_metrics_for_csv(prefix: str, metrics: dict) -> dict:
    row = {}
    for k, v in metrics.items():
        name = f"{prefix}_{k}" if prefix else k
        if isinstance(v, (int, float, np.integer, np.floating)):
            row[name] = float(v)
        elif isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
            if arr.ndim == 0:
                row[name] = float(arr)
            elif arr.ndim == 2 and "confusion_matrix" in k:
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        row[f"{name}_{i}_{j}"] = float(arr[i, j])
        elif isinstance(v, np.ndarray):
            if v.ndim == 0:
                row[name] = float(v)
            elif v.ndim == 2 and "confusion_matrix" in k:
                for i in range(v.shape[0]):
                    for j in range(v.shape[1]):
                        row[f"{name}_{i}_{j}"] = float(v[i, j])
        elif isinstance(v, dict) and ("per_class" in k):
            for cls_id, d in v.items():
                for kk, vv in d.items():
                    if isinstance(vv, (int, float, np.integer, np.floating)):
                        row[f"{name}_class{cls_id}_{kk}"] = float(vv)
    return row


def metric_value(metrics: dict, key: str, default=None):
    v = metrics.get(key, default)
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().item()
    if isinstance(v, np.ndarray):
        v = float(v.item()) if v.ndim == 0 else default
    if v is None:
        return default
    return float(v)


def make_compare_key(metrics: dict, criteria):
    key = []
    for metric_name, mode in criteria:
        v = metric_value(metrics, metric_name, default=None)
        if v is None:
            v = -1e18 if mode == "max" else 1e18
        key.append(v if mode == "max" else -v)
    return tuple(key)


def is_better_by_criteria(current_metrics: dict, best_metrics: dict, criteria, eps: float = 1e-12) -> bool:
    if best_metrics is None:
        return True
    cur_key = make_compare_key(current_metrics, criteria)
    best_key = make_compare_key(best_metrics, criteria)
    for cur_v, best_v in zip(cur_key, best_key):
        if cur_v > best_v + eps:
            return True
        if cur_v < best_v - eps:
            return False
    return False


def format_criteria(criteria) -> str:
    return " -> ".join([f"{name}({mode})" for name, mode in criteria])


class CriteriaEarlyStopping:
    def __init__(self, criteria, patience: int = 20, warmup: int = 10, min_delta: float = 1e-6):
        self.criteria = criteria
        self.patience = int(patience)
        self.warmup = int(warmup)
        self.min_delta = float(min_delta)
        self.best_metrics = None
        self.best_epoch = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, metrics: dict, epoch: int):
        improved = is_better_by_criteria(metrics, self.best_metrics, self.criteria, eps=self.min_delta)
        if improved:
            self.best_metrics = copy.deepcopy(metrics)
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            self.should_stop = False
        else:
            if epoch >= self.warmup:
                self.num_bad_epochs += 1
            if epoch >= self.warmup and self.num_bad_epochs >= self.patience:
                self.should_stop = True
        return improved, self.should_stop

    def state_dict(self):
        return {
            "criteria": self.criteria, "patience": self.patience, "warmup": self.warmup,
            "min_delta": self.min_delta, "best_epoch": self.best_epoch,
            "num_bad_epochs": self.num_bad_epochs, "should_stop": self.should_stop,
            "best_metrics": self.best_metrics,
        }


# =========================================================
# Checkpoint
# =========================================================

def save_best_checkpoint(
    save_dir: str, fold: int, best_name: str, model, optimizer, epoch: int,
    train_metrics: dict, val_metrics: dict, train_subjects, val_subjects, criteria, config: dict,
):
    os.makedirs(save_dir, exist_ok=True)
    model_file = "hc_best.pt" if best_name == "combined" else f"best_{best_name}_fold{fold}.pt"
    ckpt_path = os.path.join(save_dir, model_file)
    json_path = os.path.join(save_dir, f"best_{best_name}_fold{fold}_metrics.json")
    csv_path = os.path.join(save_dir, f"best_{best_name}_fold{fold}_summary.csv")

    ckpt = {
        "best_name": best_name, "epoch": epoch, "criteria": criteria,
        "criteria_readable": format_criteria(criteria),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": to_serializable(train_metrics),
        "val_metrics": to_serializable(val_metrics),
        "train_subjects": to_serializable(train_subjects),
        "val_subjects": to_serializable(val_subjects),
        "config": to_serializable(config),
    }
    torch.save(ckpt, ckpt_path)
    save_json(json_path, {**ckpt, "checkpoint_path": ckpt_path})
    row = {"best_name": best_name, "epoch": epoch, "criteria": format_criteria(criteria), "checkpoint_path": ckpt_path}
    row.update(flatten_metrics_for_csv("train", train_metrics))
    row.update(flatten_metrics_for_csv("val", val_metrics))
    pd.DataFrame([row]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return ckpt_path, json_path, csv_path


# =========================================================
# Data loading helpers
# =========================================================

def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def _subject_weight_tensor(subject_ids: torch.Tensor, source_weights: dict[str, float], device) -> torch.Tensor:
    values = [float(source_weights.get(str(int(s)), 1.0)) for s in subject_ids.detach().cpu()]
    return torch.tensor(values, dtype=torch.float32, device=device)


def build_target_train_loader(target_dataset, batch_size: int, num_workers: int, collate_fn, generator):
    return DataLoader(
        target_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
        worker_init_fn=seed_worker, generator=generator,
    )


# =========================================================
# Stage 1: Source Selection
# =========================================================

def train_stage1_source_selection(
    *,
    index_csv: str,
    source_dataset,
    source_loader,
    target_loader,
    train_subjects,
    repeat: int,
    fold: int,
    rand: int,
    save_dir: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    stage1_epochs: int,
    lr: float,
    weight_decay: float,
    topk: int,
    dropout: float,
    lambda_dss: float,
    source_weight_temperature: float,
    use_biomarkers: bool = True,
    biomarker_dim: int = 64,
):
    model = SourceSelectionModel(
        num_source_domains=len(source_dataset.subject_to_domain),
        topk=topk,
        dropout=dropout,
        emotion_classes=2,
        use_biomarkers=use_biomarkers,
        biomarker_dim=biomarker_dim,
    ).to(device)

    class_weights = build_emotion_class_weights(index_csv, train_subjects).to(device)
    criterion_cls = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    criterion_domain = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(stage1_epochs, 1))

    target_iter = cycle(target_loader)
    history = []
    for epoch in range(1, stage1_epochs + 1):
        model.train()
        total = 0
        total_cls = 0.0
        total_dom = 0.0
        correct = 0
        pbar = tqdm(source_loader, desc=f"SSAS-Stage1 fold{fold} epoch{epoch}", leave=False)
        for source_batch in pbar:
            target_batch = next(target_iter)
            source_batch = _move_batch_to_device(source_batch, device)
            target_batch = _move_batch_to_device(target_batch, device)

            optimizer.zero_grad(set_to_none=True)
            source_out = model(source_batch["x"], source_batch["de_feat"], lambda_domain=0.0)
            target_out = model(target_batch["x"], target_batch["de_feat"], lambda_domain=0.0)

            y = get_emotion_label(source_batch, device)
            loss_cls = criterion_cls(source_out["emo_logits"], y)

            source_domain = source_batch["domain_id"].long()
            target_domain = torch.full(
                (target_batch["x"].size(0),), fill_value=model.target_domain_id, dtype=torch.long, device=device,
            )
            domain_logits = torch.cat([source_out["domain_logits"], target_out["domain_logits"]], dim=0)
            domain_labels = torch.cat([source_domain, target_domain], dim=0)
            loss_domain = criterion_domain(domain_logits, domain_labels)
            loss = loss_cls + lambda_dss * loss_domain
            loss.backward()
            optimizer.step()

            bsz = y.size(0)
            total += bsz
            total_cls += float(loss_cls.item()) * bsz
            total_dom += float(loss_domain.item()) * bsz
            correct += int((source_out["emo_logits"].argmax(dim=1) == y).sum().item())
            pbar.set_postfix({"cls": f"{total_cls / max(total, 1):.4f}", "dom": f"{total_dom / max(total, 1):.4f}"})

        scheduler.step()
        row = {"epoch": epoch, "cls_loss": total_cls / max(total, 1), "domain_loss": total_dom / max(total, 1),
               "source_acc": correct / max(total, 1), "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"[SSAS Stage1] fold={fold} epoch={epoch}: {row}")

    stage1_path = os.path.join(save_dir, "stage1_source_selection.pt")
    torch.save({
        "stage": "stage1_source_selection", "repeat": repeat, "fold": fold, "rand": rand,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "train_subjects": train_subjects,
        "subject_to_domain": {str(k): int(v) for k, v in source_dataset.subject_to_domain.items()},
        "config": {"stage1_epochs": stage1_epochs, "lambda_dss": lambda_dss, "source_weight_temperature": source_weight_temperature},
    }, stage1_path)
    pd.DataFrame(history).to_csv(os.path.join(save_dir, "history_stage1.csv"), index=False, encoding="utf-8-sig")

    source_weights = compute_source_weights(
        model=model, source_dataset=source_dataset, batch_size=batch_size,
        num_workers=num_workers, device=device, temperature=source_weight_temperature,
    )
    source_weights_path = os.path.join(save_dir, "source_weights.json")
    save_json(source_weights_path, source_weights)
    print(f"[SSAS Stage1] saved source weights: {source_weights_path}")
    return source_weights["weights_mean_one"]


@torch.no_grad()
def compute_source_weights(
    *, model, source_dataset, batch_size: int, num_workers: int, device: torch.device, temperature: float,
) -> dict:
    model.eval()
    loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, collate_fn=comp4_collate_fn)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        out = model(batch["x"], batch["de_feat"], lambda_domain=0.0)
        target_probs = torch.softmax(out["domain_logits"], dim=1)[:, model.target_domain_id]
        for sid, prob in zip(batch["subject_id"].detach().cpu().tolist(), target_probs.detach().cpu().tolist()):
            key = str(int(sid))
            sums[key] = sums.get(key, 0.0) + float(prob)
            counts[key] = counts.get(key, 0) + 1

    subjects = sorted(sums.keys(), key=lambda x: int(x))
    raw_scores = np.array([sums[s] / max(counts[s], 1) for s in subjects], dtype=np.float64)
    tau = max(float(temperature), 1e-6)
    logits = raw_scores / tau
    logits = logits - logits.max()
    weights = np.exp(logits)
    weights = weights / max(weights.sum(), 1e-12)
    weights_mean_one = {s: float(w * len(subjects)) for s, w in zip(subjects, weights)}
    return {
        "type": "subject_level_target_probability_softmax", "temperature": float(temperature),
        "raw_target_scores": {s: float(v) for s, v in zip(subjects, raw_scores)},
        "weights_mean_one": weights_mean_one,
    }


# =========================================================
# Stage 2/3: SSAS training
# =========================================================

def train_one_epoch_ssas(
    model, source_loader, target_loader, optimizer,
    criterion_emo, domain_criterion, device,
    source_weights: dict[str, float],
    lambda_adv: float = 0.1,
    grl_domain: float = 0.1,
    lambda_mmd: float = 0.0,
    lambda_con: float = 0.0,
):
    """Stage 2/3: weighted source CE + GRL adversarial + trial compactness (+ optional MMD)."""
    model.train()
    target_iter = cycle(target_loader)
    total = 0
    sums = {"loss": 0.0, "emo_loss": 0.0, "dom_loss": 0.0, "mmd_loss": 0.0, "con_loss": 0.0, "acc": 0.0}
    pbar = tqdm(source_loader, desc="Train-SSAS", leave=False)

    for source_batch in pbar:
        target_batch = next(target_iter)
        source_batch = _move_batch_to_device(source_batch, device)
        target_batch = _move_batch_to_device(target_batch, device)

        optimizer.zero_grad(set_to_none=True)
        source_out = model(source_batch["x"], source_batch["de_feat"], lambda_domain=grl_domain)
        target_out = model(target_batch["x"], target_batch["de_feat"], lambda_domain=grl_domain)

        y = get_emotion_label(source_batch, device)
        sample_ce = criterion_emo(source_out["emo_logits"], y)
        sample_weights = _subject_weight_tensor(source_batch["subject_id"], source_weights, device)
        loss_emo = (sample_ce * sample_weights).mean()

        source_domain = torch.zeros(source_batch["x"].size(0), dtype=torch.long, device=device)
        target_domain = torch.ones(target_batch["x"].size(0), dtype=torch.long, device=device)
        domain_logits = torch.cat([source_out["domain_logits"], target_out["domain_logits"]], dim=0)
        domain_labels = torch.cat([source_domain, target_domain], dim=0)
        loss_dom = domain_criterion(domain_logits, domain_labels)

        loss_mmd = mmd_loss(source_out["z"], target_out["z"]) if lambda_mmd > 0 else source_out["z"].new_tensor(0.0)

        if lambda_con > 0:
            trial_ids = make_subject_trial_group_ids(source_batch, device)
            loss_con = trial_compactness_loss(source_out["contrast_feat"], trial_ids)
        else:
            loss_con = source_out["z"].new_tensor(0.0)

        loss = loss_emo + lambda_adv * loss_dom + lambda_mmd * loss_mmd + lambda_con * loss_con
        loss.backward()
        optimizer.step()

        bsz = y.size(0)
        total += bsz
        sums["loss"] += float(loss.item()) * bsz
        sums["emo_loss"] += float(loss_emo.item()) * bsz
        sums["dom_loss"] += float(loss_dom.item()) * bsz
        sums["mmd_loss"] += float(loss_mmd.item()) * bsz
        sums["con_loss"] += float(loss_con.item()) * bsz
        sums["acc"] += int((source_out["emo_logits"].argmax(dim=1) == y).sum().item())
        pbar.set_postfix({"loss": f"{sums['loss'] / max(total, 1):.4f}", "con": f"{sums['con_loss'] / max(total, 1):.4f}"})

    return {key: value / max(total, 1) for key, value in sums.items()}


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def validate_one_epoch_ssas(
    model, loader, criterion_emo, device, lambda_con: float = 0.0,
):
    model.eval()
    total_loss = 0.0
    total_emo_loss = 0.0
    total_con_loss = 0.0
    total_correct = 0
    total_num = 0
    all_preds, all_labels = [], []
    segment_records = []

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        x = batch["x"].to(device)
        de_feat = batch["de_feat"].to(device)
        y = get_emotion_label(batch, device)
        trial_group_ids = make_subject_trial_group_ids(batch, device)

        out = model(x, de_feat, lambda_domain=0.0)
        logits = out["emo_logits"]

        loss_emo = criterion_emo(logits, y)

        if lambda_con > 0:
            loss_con = trial_compactness_loss(out["contrast_feat"], trial_group_ids)
        else:
            loss_con = logits.new_tensor(0.0)

        loss = loss_emo + lambda_con * loss_con

        pred = logits.argmax(dim=1)
        prob = torch.softmax(logits, dim=1)
        prob_pos = prob[:, 1]

        subject_ids_cpu = batch["subject_id"]
        trial_ids_cpu = batch["trial_id"]

        for i in range(x.size(0)):
            segment_records.append({
                "subject_id": int(subject_ids_cpu[i]),
                "trial_id": int(trial_ids_cpu[i]),
                "prob_pos": float(prob_pos[i].detach().cpu()),
                "pred_emo": int(pred[i].detach().cpu()),
                "label_emo": int(y[i].detach().cpu()),
            })

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_emo_loss += loss_emo.item() * bsz
        total_con_loss += loss_con.item() * bsz
        total_correct += (pred == y).sum().item()
        total_num += bsz

        all_preds.extend(pred.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

        pbar.set_postfix({"loss": f"{total_loss / max(total_num, 1):.4f}", "acc": f"{total_correct / max(total_num, 1):.4f}"})

    if total_num == 0:
        raise RuntimeError("验证集为空。")

    emotion_acc = accuracy_score(all_labels, all_preds)
    emotion_macro_f1 = f1_score(all_labels, all_preds, average="macro", labels=[0, 1], zero_division=0)
    emotion_cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    pe, re, f1e, se = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1], zero_division=0)
    per_class_emo = {int(c): {"precision": float(pe[i]), "recall": float(re[i]), "f1": float(f1e[i]), "support": int(se[i])} for i, c in enumerate([0, 1])}

    trial_metrics = compute_trial_level_metrics(segment_records, threshold=0.5, vote_method="hard")

    pt, rt, f1t, st = precision_recall_fscore_support(
        trial_metrics["trial_labels"], trial_metrics["trial_preds"], labels=[0, 1], zero_division=0,
    )
    trial_per_class = {int(c): {"precision": float(pt[i]), "recall": float(rt[i]), "f1": float(f1t[i]), "support": int(st[i])} for i, c in enumerate([0, 1])}

    return {
        "loss": total_loss / total_num, "emo_loss": total_emo_loss / total_num,
        "con_loss": total_con_loss / total_num,

        "emotion_acc": emotion_acc, "emotion_macro_f1": emotion_macro_f1,
        "emotion_confusion_matrix": emotion_cm, "per_class_emo": per_class_emo,
        "segment_preds_emo": all_preds, "segment_labels_emo": all_labels,

        "trial_acc": trial_metrics["trial_acc"], "trial_macro_f1": trial_metrics["trial_macro_f1"],
        "trial_confusion_matrix": trial_metrics["trial_confusion_matrix"],
        "trial_per_class_emo": trial_per_class,
        "trial_keys": trial_metrics["trial_keys"], "trial_probs": trial_metrics["trial_probs"],
        "trial_preds": trial_metrics["trial_preds"], "trial_labels": trial_metrics["trial_labels"],

        "segment_records": segment_records,
    }


# =========================================================
# Cross-subject training orchestrator
# =========================================================

def train_competition_cross_subject(
    model,
    index_csv: str,
    fold: int,
    save_dir: str,
    dataset: str,
    rand: int,
    train_group: str = "hc",
    n_splits: int = 5,
    epochs: int = 100,
    batch_size: int = 256,
    subjects_per_batch: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    ssas_stage1_epochs: int = 20,
    ssas_lambda_dss: float = 0.2,
    ssas_lambda_adv: float = 0.1,
    ssas_grl_domain: float = 0.1,
    ssas_lambda_mmd: float = 0.0,
    ssas_source_weight_temperature: float = 0.05,
    lambda_con: float = 0.1,
    device: str = "cuda",
    run_seed: int = None,
    deterministic: bool = True,
    early_stop_patience: int = 25,
    early_stop_warmup: int = 15,
    early_stop_min_delta: float = 1e-6,
    early_stop_track: str = "combined",
    use_biomarkers: bool = True,
    biomarker_dim: int = 64,
):
    os.makedirs(save_dir, exist_ok=True)
    train_group = normalize_train_group(train_group)

    if run_seed is None:
        run_seed = config.make_run_seed(rand, fold)
    set_global_seed(run_seed, deterministic=deterministic)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(run_seed)

    print(f"[Seed] rand={rand}, fold={fold}, run_seed={run_seed}, deterministic={deterministic}")
    print(f"[TrainGroup] {train_group}")

    # ── 被试划分 ──
    split = get_unified_subject_split(index_csv=index_csv, fold=fold, n_splits=n_splits, seed=rand)
    if train_group == "hc":
        train_subjects = split["train_hc"]
        val_subjects = split["val_hc"]
    elif train_group == "dep":
        train_subjects = split["train_dep"]
        val_subjects = split["val_dep"]
    else:
        train_subjects = split["train_all"]
        val_subjects = split["val_all"]

    print(f"Fold {fold}: train={len(train_subjects)}, val={len(val_subjects)}")

    if dataset == "com":
        train_dataset = Competition4ClassDataset(index_csv=index_csv, subject_ids=train_subjects, normalize=False)
        val_dataset = Competition4ClassDataset(index_csv=index_csv, subject_ids=val_subjects, normalize=False)
        collate_fn = comp4_collate_fn
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # ── DataLoader ──
    eff_sbj = min(int(subjects_per_batch), max(len(train_subjects), 1))
    train_batch_sampler = MultiSubjectBatchSampler(
        sample_subject_ids=train_dataset.sample_subject_ids, batch_size=batch_size,
        subjects_per_batch=eff_sbj, drop_last=True, shuffle=True,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn,
                              num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=loader_generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=False, collate_fn=collate_fn,
                            worker_init_fn=seed_worker, generator=loader_generator)

    target_train_loader = build_target_train_loader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                                                     collate_fn=collate_fn, generator=loader_generator)

    # ── Stage 1: Source Selection ──
    print("\n" + "=" * 60)
    print("[SSAS] Stage 1: source selection")
    print("=" * 60)
    source_weights = train_stage1_source_selection(
        index_csv=index_csv, source_dataset=train_dataset, source_loader=train_loader,
        target_loader=target_train_loader, train_subjects=train_subjects,
        repeat=-1, fold=fold, rand=rand, save_dir=save_dir, device=device,
        batch_size=batch_size, num_workers=num_workers, stage1_epochs=ssas_stage1_epochs,
        lr=lr, weight_decay=weight_decay, topk=6, dropout=0.2, lambda_dss=ssas_lambda_dss,
        source_weight_temperature=ssas_source_weight_temperature,
        use_biomarkers=use_biomarkers, biomarker_dim=biomarker_dim,
    )
    print("[SSAS] source_weights preview:", list(source_weights.items())[:8])

    # ── Stage 2/3: Loss & Optimizer ──
    class_weights = build_emotion_class_weights(index_csv, train_subjects).to(device)
    print(f"[Class weights] {class_weights.detach().cpu().numpy().tolist()}")
    criterion_emo = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    criterion_emo_train = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05, reduction="none")
    dom_criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    config_dict = {
        "index_csv": index_csv, "fold": fold, "rand": rand, "run_seed": run_seed,
        "deterministic": deterministic, "train_group": train_group,
        "n_splits": n_splits, "epochs": epochs, "batch_size": batch_size,
        "subjects_per_batch": subjects_per_batch, "lr": lr, "weight_decay": weight_decay,
        "ssas_stage1_epochs": ssas_stage1_epochs, "ssas_lambda_dss": ssas_lambda_dss,
        "ssas_lambda_adv": ssas_lambda_adv, "ssas_grl_domain": ssas_grl_domain,
        "ssas_lambda_mmd": ssas_lambda_mmd, "ssas_source_weight_temperature": ssas_source_weight_temperature,
        "lambda_con": lambda_con, "dataset": dataset,
        "early_stop_patience": early_stop_patience, "early_stop_warmup": early_stop_warmup,
        "early_stop_min_delta": early_stop_min_delta, "early_stop_track": early_stop_track,
        "use_biomarkers": use_biomarkers, "biomarker_dim": biomarker_dim,
    }

    best_trackers = {
        "combined": {
            "criteria": [("trial_macro_f1", "max"), ("trial_acc", "max"), ("emotion_macro_f1", "max"), ("emotion_acc", "max"), ("loss", "min")],
            "best_metrics": None, "best_epoch": None, "checkpoint_path": None,
        },
        "trial_f1": {
            "criteria": [("trial_macro_f1", "max"), ("trial_acc", "max"), ("loss", "min")],
            "best_metrics": None, "best_epoch": None, "checkpoint_path": None,
        },
        "trial_acc": {
            "criteria": [("trial_acc", "max"), ("trial_macro_f1", "max"), ("loss", "min")],
            "best_metrics": None, "best_epoch": None, "checkpoint_path": None,
        },
        "segment_emo_f1": {
            "criteria": [("emotion_macro_f1", "max"), ("emotion_acc", "max"), ("trial_macro_f1", "max"), ("loss", "min")],
            "best_metrics": None, "best_epoch": None, "checkpoint_path": None,
        },
    }

    early_stopper = CriteriaEarlyStopping(
        criteria=best_trackers[early_stop_track]["criteria"],
        patience=early_stop_patience, warmup=early_stop_warmup, min_delta=early_stop_min_delta,
    )

    print(f"[EarlyStopping] track={early_stop_track}, patience={early_stop_patience}, warmup={early_stop_warmup}")

    # ── Training Loop ──
    history = []
    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")

        train_metrics = train_one_epoch_ssas(
            model=model, source_loader=train_loader, target_loader=target_train_loader,
            optimizer=optimizer, criterion_emo=criterion_emo_train, domain_criterion=dom_criterion,
            device=device, source_weights=source_weights,
            lambda_adv=ssas_lambda_adv, grl_domain=ssas_grl_domain,
            lambda_mmd=ssas_lambda_mmd, lambda_con=lambda_con,
        )

        val_metrics = validate_one_epoch_ssas(
            model=model, loader=val_loader, criterion_emo=criterion_emo,
            device=device, lambda_con=lambda_con,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"[Train] loss={train_metrics['loss']:.4f} emo={train_metrics['emo_loss']:.4f} "
              f"dom={train_metrics['dom_loss']:.4f} mmd={train_metrics['mmd_loss']:.4f} "
              f"con={train_metrics['con_loss']:.4f} acc={train_metrics['acc']:.4f}")
        print(f"[Val]   loss={val_metrics['loss']:.4f} emo_loss={val_metrics['emo_loss']:.4f} "
              f"con_loss={val_metrics['con_loss']:.4f} "
              f"seg_acc={val_metrics['emotion_acc']:.4f} seg_f1={val_metrics['emotion_macro_f1']:.4f} "
              f"trial_acc={val_metrics['trial_acc']:.4f} trial_f1={val_metrics['trial_macro_f1']:.4f}")

        hist_row = {"epoch": epoch, "lr": current_lr}
        hist_row.update(flatten_metrics_for_csv("train", train_metrics))
        hist_row.update(flatten_metrics_for_csv("val", val_metrics))
        history.append(hist_row)

        epoch_metrics_json = os.path.join(save_dir, f"epoch_{epoch:03d}_metrics.json")
        save_json(epoch_metrics_json, {"epoch": epoch, "lr": current_lr, "train_metrics": train_metrics, "val_metrics": val_metrics})

        for best_name, tracker in best_trackers.items():
            criteria = tracker["criteria"]
            if is_better_by_criteria(val_metrics, tracker["best_metrics"], criteria):
                ckpt_path, json_path, csv_path = save_best_checkpoint(
                    save_dir=save_dir, fold=fold, best_name=best_name, model=model, optimizer=optimizer,
                    epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics,
                    train_subjects=train_subjects, val_subjects=val_subjects, criteria=criteria, config=config_dict,
                )
                tracker["best_metrics"] = copy.deepcopy(val_metrics)
                tracker["best_train_metrics"] = copy.deepcopy(train_metrics)
                tracker["best_epoch"] = epoch
                tracker["checkpoint_path"] = ckpt_path
                tracker["metrics_json_path"] = json_path
                tracker["summary_csv_path"] = csv_path
                print(f"保存 best_{best_name}: epoch={epoch}, criteria={format_criteria(criteria)}, path={ckpt_path}")

        early_improved, should_stop = early_stopper.step(val_metrics, epoch)
        print(f"[EarlyStopping] improved={early_improved}, best_epoch={early_stopper.best_epoch}, bad={early_stopper.num_bad_epochs}/{early_stop_patience}")
        if should_stop:
            print(f"\n触发早停: epoch={epoch}")
            early_stop_json = os.path.join(save_dir, f"early_stop_fold{fold}.json")
            save_json(early_stop_json, {"stopped": True, "stop_epoch": epoch, "early_stop_track": early_stop_track, "early_stopper": early_stopper.state_dict(), "config": config_dict})
            break

    # ── 保存汇总 ──
    early_stop_state_json = os.path.join(save_dir, f"early_stop_state_fold{fold}.json")
    save_json(early_stop_state_json, {"early_stop_track": early_stop_track, "early_stopper": early_stopper.state_dict(), "config": config_dict})

    history_df = pd.DataFrame(history)
    history_csv = os.path.join(save_dir, f"history_fold{fold}.csv")
    history_df.to_csv(history_csv, index=False, encoding="utf-8-sig")

    best_rows = []
    for best_name, tracker in best_trackers.items():
        row = {"best_name": best_name, "best_epoch": tracker["best_epoch"], "criteria": format_criteria(tracker["criteria"]), "checkpoint_path": tracker["checkpoint_path"]}
        if tracker["best_metrics"] is not None:
            row.update(flatten_metrics_for_csv("val", tracker["best_metrics"]))
        best_rows.append(row)

    best_summary_csv = os.path.join(save_dir, f"best_summary_fold{fold}.csv")
    pd.DataFrame(best_rows).to_csv(best_summary_csv, index=False, encoding="utf-8-sig")
    print(f"Best summary: {best_summary_csv}")

    return {
        "fold": fold, "rand": rand, "run_seed": run_seed, "train_group": train_group,
        "train_subjects": train_subjects, "val_subjects": val_subjects,
        "history_csv": history_csv, "best_summary_csv": best_summary_csv,
        "best_models": best_trackers, "early_stop": early_stopper.state_dict(),
    }


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    device = "cuda:0"
    train_group = "hc"
    version = f"{train_group}_ssas"
    model_params_root = "model_params"
    os.makedirs(version, exist_ok=True)
    os.makedirs(model_params_root, exist_ok=True)

    all_fold_rows = []
    all_best_rows_long = []
    all_ckpt_paths = []
    combined_emotion_confusions = []
    combined_trial_confusions = []

    for repeat, rand in enumerate(config.HC_REPEAT_SEEDS):
        print(f"\n{'#' * 70}")
        print(f"SSAS HC 情绪训练: seed={rand}, group={train_group}")
        print(f"{'#' * 70}")

        seed_combined_rows = []
        n_splits = 5

        for fold in range(n_splits):
            print(f"\n{'=' * 50}")
            print(f"Fold {fold + 1}/{n_splits} | seed={rand} | group={train_group}")
            print(f"{'=' * 50}")

            run_seed = config.make_run_seed(rand, fold)
            set_global_seed(run_seed, deterministic=True)
            print(f"[Main Seed] rand={rand}, fold={fold}, run_seed={run_seed}")

            model = SSASEmotionModel(
                sfreq=250.0, topk=6, dropout=0.2,
                emotion_classes=2, contrast_dim=64, contrast_hidden_dim=128,
                domain_hidden_dim=128, use_biomarkers=True, biomarker_dim=64,
            )

            result = train_competition_cross_subject(
                model=model,
                index_csv="com_index_sub_2s.csv",
                fold=fold,
                save_dir=os.path.join(model_params_root, f"hc_ssas_reapt{repeat}_fold{fold}"),
                dataset="com",
                train_group=train_group,
                n_splits=n_splits,
                epochs=100,
                batch_size=128,
                subjects_per_batch=4,
                lr=1e-4,
                rand=rand,
                weight_decay=1e-4,
                num_workers=4,
                ssas_stage1_epochs=20,
                ssas_lambda_dss=0.2,
                ssas_lambda_adv=0.1,
                ssas_grl_domain=0.1,
                ssas_lambda_mmd=0.0,
                ssas_source_weight_temperature=0.05,
                lambda_con=0.1,
                device=device,
                run_seed=run_seed,
                deterministic=True,
                early_stop_track="combined",
                early_stop_patience=25,
                early_stop_warmup=15,
                early_stop_min_delta=1e-6,
                use_biomarkers=True,
                biomarker_dim=64,
            )

            combined_tracker = result["best_models"]["combined"]
            combined_metrics = combined_tracker["best_metrics"]

            fold_row = {
                "rand": rand, "fold": fold, "run_seed": run_seed, "train_group": train_group,
                "best_name": "combined", "best_epoch": combined_tracker["best_epoch"],
                "checkpoint_path": combined_tracker["checkpoint_path"],
            }
            fold_row.update(flatten_metrics_for_csv("val", combined_metrics))
            all_fold_rows.append(fold_row)
            seed_combined_rows.append(fold_row)

            if combined_tracker["checkpoint_path"]:
                all_ckpt_paths.append(Path(combined_tracker["checkpoint_path"]))

            for best_name, tracker in result["best_models"].items():
                row = {"rand": rand, "fold": fold, "run_seed": run_seed, "train_group": train_group,
                       "best_name": best_name, "best_epoch": tracker["best_epoch"],
                       "criteria": format_criteria(tracker["criteria"]), "checkpoint_path": tracker["checkpoint_path"]}
                if tracker["best_metrics"] is not None:
                    row.update(flatten_metrics_for_csv("val", tracker["best_metrics"]))
                all_best_rows_long.append(row)

            if combined_metrics is not None:
                combined_emotion_confusions.append(combined_metrics["emotion_confusion_matrix"])
                combined_trial_confusions.append(combined_metrics["trial_confusion_matrix"])

            print(f"Fold {fold + 1} combined best:")
            print(f"  epoch={combined_tracker['best_epoch']}")
            print(f"  seg_emo_acc={combined_metrics['emotion_acc']:.4f}  seg_emo_f1={combined_metrics['emotion_macro_f1']:.4f}")
            print(f"  trial_acc={combined_metrics['trial_acc']:.4f}  trial_f1={combined_metrics['trial_macro_f1']:.4f}")

        seed_df = pd.DataFrame(seed_combined_rows)
        seed_csv = os.path.join(version, f"seed{rand}_combined_5fold_results.csv")
        seed_df.to_csv(seed_csv, index=False, encoding="utf-8-sig")

        metric_cols = ["val_emotion_acc", "val_emotion_macro_f1", "val_trial_acc", "val_trial_macro_f1", "val_loss"]
        print(f"\n{'=' * 50}")
        print(f"Seed {rand} 5-fold average | group={train_group}")
        print(f"{'=' * 50}")
        summary_lines = []
        for col in metric_cols:
            if col in seed_df.columns:
                m = seed_df[col].mean()
                s = seed_df[col].std(ddof=0)
                print(f"  {col}: {m:.4f} ± {s:.4f}")
                summary_lines.append(f"{col}: {m:.4f} ± {s:.4f}\n")
        with open(os.path.join(version, f"seed{rand}_combined_5fold_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"SSAS HC emotion seed={rand} 5-fold results | group={train_group}\n")
            f.writelines(summary_lines)
            f.write(f"\nCSV: {seed_csv}\n")

    # ── Overall summary ──
    all_fold_df = pd.DataFrame(all_fold_rows)
    all_fold_csv = os.path.join(version, "all_combined_best_metrics.csv")
    all_fold_df.to_csv(all_fold_csv, index=False, encoding="utf-8-sig")
    print(f"\nAll combined best metrics: {all_fold_csv}")

    all_best_df = pd.DataFrame(all_best_rows_long)
    all_best_csv = os.path.join(version, "all_parallel_best_metrics_long.csv")
    all_best_df.to_csv(all_best_csv, index=False, encoding="utf-8-sig")

    np.save(os.path.join(version, "combined_emotion_confusions.npy"), np.array(combined_emotion_confusions, dtype=object))
    np.save(os.path.join(version, "combined_trial_confusions.npy"), np.array(combined_trial_confusions, dtype=object))

    if len(all_fold_df) > 0:
        metric_cols = ["val_emotion_acc", "val_emotion_macro_f1", "val_trial_acc", "val_trial_macro_f1", "val_loss"]
        print(f"\n{'=' * 50}")
        print(f"Overall {len(all_fold_df)} folds | group={train_group}")
        print(f"{'=' * 50}")
        overall_summary = []
        for col in metric_cols:
            if col in all_fold_df.columns:
                m = all_fold_df[col].mean()
                s = all_fold_df[col].std(ddof=0)
                print(f"  {col}: {m:.4f} ± {s:.4f}")
                overall_summary.append({"metric": col, "mean": m, "std": s})
        overall_summary_csv = os.path.join(version, "overall_combined_summary.csv")
        pd.DataFrame(overall_summary).to_csv(overall_summary_csv, index=False, encoding="utf-8-sig")

        # Average trial confusion matrix
        valid_trial = [np.array(c, dtype=np.float32) for c in combined_trial_confusions if c is not None]
        if len(valid_trial) > 0:
            avg = np.stack(valid_trial, axis=0).mean(axis=0)
            avg_norm = avg / np.maximum(avg.sum(axis=1, keepdims=True), 1e-12)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(avg_norm, cmap="Blues")
            labels = ["neutral", "positive"]
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(labels); ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"Avg Trial Emotion CM ({train_group})")
            thresh = avg_norm.max() / 2 if avg_norm.max() != 0 else 0.5
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{avg_norm[i,j]:.2f}", ha="center", va="center", color="white" if avg_norm[i,j] > thresh else "black")
            fig.colorbar(im, ax=ax); fig.tight_layout()
            fig.savefig(os.path.join(version, "avg_trial_emotion_confusion.png"), dpi=200)
            plt.close(fig)
            print(f"Avg trial confusion matrix saved.")
    else:
        print("No fold results collected.")

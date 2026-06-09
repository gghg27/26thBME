# -*- coding: utf-8 -*-
"""
二分支 EEG 情绪识别训练脚本：情绪主任务 + 监督对比学习 + 诊断 GRL + 被试 GRL

对应模型文件：
    two_branch_emotion_grl_contrast_model.py

模型输出：
    emo_logits       [B, 2]  情绪分类头，不带 GRL，0=neutral, 1=positive
    diagnosis_logits [B, 2]  诊断分类头，带 GRL，0=DEP, 1=HC
    subject_logits   [B, S]  被试分类头，带 GRL

核心训练目标：
    loss = lambda_emo * CE(emo_logits, emotion_label)
         + lambda_con * SupCon(contrast_feat, emotion_label)
         + lambda_diag * CE(diagnosis_logits, diagnosis_label)   # 可选，对 train_group='all' 更合理
         + lambda_subject * CE(subject_logits, subject_id/domain_id)
         + lambda_graph * intra_class_graph_loss(adj_dense, emotion_label)

重要说明：
    1) train_group='hc'：只用正常被试训练情绪二分类，此时 diagnosis_label 恒为 HC，建议 lambda_diag=0。
    2) train_group='dep'：只用抑郁被试训练情绪二分类，此时 diagnosis_label 恒为 DEP，建议 lambda_diag=0。
    3) train_group='all'：正常和抑郁一起训练，此时可以尝试开启 diagnosis GRL。

建议初始参数：
    train_group='hc' 或 'dep'
    lambda_emo=1.0
    lambda_diag=0.0
    grl_diag=0.0
    lambda_con=0.03 ~ 0.10
    con_temperature=0.1 ~ 0.2
    lambda_subject=0.001 ~ 0.01
    grl_subject=0.001 ~ 0.01
    lambda_graph=0.0
"""

import os
import json
import copy
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold, KFold
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

from models.hc_contrast_bio import EmotionPretrainModel
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
# Subject-relative baseline utilities
# =========================================================

def _to_subject_int(subject_id):
    if isinstance(subject_id, torch.Tensor):
        return int(subject_id.detach().cpu().item())
    return int(subject_id)


def compute_subject_de_baselines(dataset, eps=1e-6):
    """
    遍历 dataset，按 subject_id 聚合 de_feat。

    返回：
        subject_de_mu: dict[int, torch.Tensor], shape [C, K]
        subject_de_std: dict[int, torch.Tensor], shape [C, K]
    """
    sums = {}
    sq_sums = {}
    counts = defaultdict(int)

    for idx in tqdm(range(len(dataset)), desc="DE baseline", leave=False):
        item = dataset[idx]
        sid = _to_subject_int(item["subject_id"])
        de_feat = torch.as_tensor(item["de_feat"], dtype=torch.float32)

        if sid not in sums:
            sums[sid] = torch.zeros_like(de_feat)
            sq_sums[sid] = torch.zeros_like(de_feat)
        sums[sid] += de_feat
        sq_sums[sid] += de_feat * de_feat
        counts[sid] += 1

    subject_de_mu = {}
    subject_de_std = {}
    for sid, total in sums.items():
        count = max(int(counts[sid]), 1)
        mu = total / count
        var = sq_sums[sid] / count - mu * mu
        subject_de_mu[int(sid)] = mu
        subject_de_std[int(sid)] = torch.sqrt(torch.clamp(var, min=0.0) + eps)

    return subject_de_mu, subject_de_std


def gather_subject_baseline(subject_ids, baseline_dict, device, dtype=None):
    """
    subject_ids: batch["subject_id"]
    baseline_dict: dict[int, Tensor/ndarray]
    return: [B, ...]

    如果 baseline 缺失，返回 None，让模型自动退化为原始特征。
    """
    if baseline_dict is None:
        return None

    if isinstance(subject_ids, torch.Tensor):
        ids = subject_ids.detach().cpu().view(-1).tolist()
    else:
        ids = list(subject_ids)

    values = []
    for sid in ids:
        sid = int(sid)
        if sid not in baseline_dict:
            return None
        values.append(torch.as_tensor(baseline_dict[sid]))

    out = torch.stack(values, dim=0).to(device=device)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


@torch.no_grad()
def compute_subject_bio_baselines(
    model,
    loader,
    device,
    subject_de_mu=None,
    subject_de_std=None,
    eps=1e-6,
):
    """
    用无梯度前向离线提取 out["bio_raw"]，按 subject_id 聚合 bio baseline。

    这里不传 subject_bio_mu/std，因此即使模型开启 bio 相对化，也会先走原始
    bio_raw 路径，避免 baseline 递归依赖。标签不会参与计算。
    """
    model.eval()
    sums = {}
    sq_sums = {}
    counts = defaultdict(int)

    for batch in tqdm(loader, desc="Bio baseline", leave=False):
        x = batch["x"].to(device)
        de_feat = batch["de_feat"].to(device)
        subject_ids = batch["subject_id"]
        de_mu_batch = gather_subject_baseline(subject_ids, subject_de_mu, device, dtype=de_feat.dtype)
        de_std_batch = gather_subject_baseline(subject_ids, subject_de_std, device, dtype=de_feat.dtype)

        out = model(
            x,
            de_feat=de_feat,
            lambda_diag=0.0,
            lambda_subject=0.0,
            subject_de_mu=de_mu_batch,
            subject_de_std=de_std_batch,
            subject_bio_mu=None,
            subject_bio_std=None,
        )
        bio_raw = out.get("bio_raw", None)
        if bio_raw is None:
            raise RuntimeError("模型 forward 未返回 bio_raw，无法计算 bio baseline。")
        bio_raw = bio_raw.detach().cpu().float()

        for i, sid in enumerate(subject_ids.detach().cpu().view(-1).tolist()):
            sid = int(sid)
            feat = bio_raw[i]
            if sid not in sums:
                sums[sid] = torch.zeros_like(feat)
                sq_sums[sid] = torch.zeros_like(feat)
            sums[sid] += feat
            sq_sums[sid] += feat * feat
            counts[sid] += 1

    subject_bio_mu = {}
    subject_bio_std = {}
    for sid, total in sums.items():
        count = max(int(counts[sid]), 1)
        mu = total / count
        var = sq_sums[sid] / count - mu * mu
        subject_bio_mu[int(sid)] = mu
        subject_bio_std[int(sid)] = torch.sqrt(torch.clamp(var, min=0.0) + eps)

    return subject_bio_mu, subject_bio_std


# =========================================================
# Label utilities
# =========================================================

def normalize_train_group(train_group: str) -> str:
    group = str(train_group).lower().strip()
    if group in ["hc", "normal", "healthy", "control", "controls", "0_hc", "1_hc"]:
        return "hc"
    if group in ["dep", "depression", "mdd", "patient", "patients", "0_dep", "1_dep"]:
        return "dep"
    if group in ["all", "both", "mix", "mixed"]:
        return "all"
    raise ValueError("train_group 只能是 'hc' / 'dep' / 'all'。")


def get_targets(batch: Dict, device: torch.device):
    """
    统一得到训练标签。

    label4 约定：
        0 DEP_neu
        1 DEP_pos
        2 HC_neu
        3 HC_pos

    emotion_label：
        0 neutral
        1 positive

    diagnosis_label：
        0 DEP
        1 HC

    subject_target：
        优先使用 batch['domain_id']，否则使用 batch['subject_id']。
    """
    label4 = batch["label4"].to(device).long()

    if "emotion_label" in batch:
        emotion_label = batch["emotion_label"].to(device).long()
    else:
        emotion_label = (label4 % 2).long()

    # 为了避免原始 diagnosis_label 编码不统一，这里优先由 label4 稳定推出：0=DEP, 1=HC
    if "label4" in batch:
        diagnosis_label = (label4 >= 2).long()
    elif "diagnosis_label" in batch:
        diagnosis_label = batch["diagnosis_label"].to(device).long()
    else:
        raise KeyError("batch 中缺少 label4 / diagnosis_label，无法得到诊断标签。")

    if "domain_id" in batch:
        subject_target = batch["domain_id"].to(device).long()
    elif "subject_id" in batch:
        subject_target = batch["subject_id"].to(device).long()
    else:
        subject_target = None

    return label4, emotion_label, diagnosis_label, subject_target


# =========================================================
# Graph loss
# =========================================================

def flatten_upper_triangle(adj: torch.Tensor) -> torch.Tensor:
    _, n, _ = adj.shape
    idx = torch.triu_indices(n, n, offset=1, device=adj.device)
    return adj[:, idx[0], idx[1]]


def intra_class_graph_loss(
    adj_dense: torch.Tensor,
    labels: torch.Tensor,
    normalize_graph_vec: bool = True,
) -> torch.Tensor:
    """
    同类样本脑网络一致性约束。
    先建议 lambda_graph=0.0，主任务跑稳后再开。
    """
    graph_vec = flatten_upper_triangle(adj_dense)
    if normalize_graph_vec:
        graph_vec = F.normalize(graph_vec, p=2, dim=1)

    losses = []
    for c in labels.unique():
        mask = (labels == c)
        if mask.sum() < 2:
            continue
        g = graph_vec[mask]
        g_mean = g.mean(dim=0, keepdim=True)
        losses.append(((g - g_mean) ** 2).mean())

    if len(losses) == 0:
        return adj_dense.new_tensor(0.0)
    return torch.stack(losses).mean()


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss，用 emotion_label 做监督。

    目标：
        - 同一情绪类别的窗口特征互相拉近；
        - 不同情绪类别的窗口特征互相拉远。

    Args:
        features: [B, D]，建议是模型输出的 contrast_feat，已归一化也可以
        labels:   [B]，这里用 emotion_label: 0=neutral, 1=positive
        temperature: 温度系数，越小类间分离约束越强

    返回：
        标量 loss。若当前 batch 内没有任何正样本对，则返回 0，避免报错。
    """
    if features.ndim != 2:
        raise ValueError(f"features should be [B, D], got {tuple(features.shape)}")

    device = features.device
    batch_size = features.size(0)
    if batch_size <= 1:
        return features.new_tensor(0.0)

    labels = labels.view(-1, 1)
    features = F.normalize(features, p=2, dim=1)

    # [B, B] cosine similarity / temperature
    logits = torch.matmul(features, features.T) / max(float(temperature), eps)

    # 数值稳定
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # 去掉自己和自己的对比
    eye = torch.eye(batch_size, device=device, dtype=torch.bool)
    not_eye = ~eye

    # 正样本：同类且不是自己
    positive_mask = torch.eq(labels, labels.T).to(device) & not_eye

    # 分母：除自己之外的所有样本
    exp_logits = torch.exp(logits) * not_eye.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)

    positive_count = positive_mask.sum(dim=1)
    valid_anchor = positive_count > 0

    if valid_anchor.sum() == 0:
        return features.new_tensor(0.0)

    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_count.clamp_min(1)
    loss = -mean_log_prob_pos[valid_anchor].mean()
    return loss


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
            num_neg = len(preds) - num_pos
            if num_pos > num_neg:
                pred = 1
            elif num_neg > num_pos:
                pred = 0
            else:
                pred = 1 if mean_prob >= threshold else 0
        else:
            raise ValueError(f"Unknown vote_method: {vote_method}")

        trial_keys.append(key)
        trial_probs.append(mean_prob)
        trial_preds.append(pred)
        trial_labels.append(label)

    trial_acc = accuracy_score(trial_labels, trial_preds)
    trial_macro_f1 = f1_score(
        trial_labels,
        trial_preds,
        average="macro",
        labels=[0, 1],
        zero_division=0,
    )
    trial_cm = confusion_matrix(trial_labels, trial_preds, labels=[0, 1])

    return {
        "trial_acc": trial_acc,
        "trial_macro_f1": trial_macro_f1,
        "trial_confusion_matrix": trial_cm,
        "trial_keys": trial_keys,
        "trial_probs": trial_probs,
        "trial_preds": trial_preds,
        "trial_labels": trial_labels,
    }


def apply_subject_topk_prediction(trial_score_records, k_pos=4):
    """
    输入每个 trial 的 subject_id、trial_id、score_pos、prob_pos。
    对每个 subject 内部按 score_pos 排序，top k_pos 判为 1，其余判为 0。
    返回带 Emotion_label 的结果。
    """
    subject_trials = defaultdict(list)
    for r in trial_score_records:
        rec = dict(r)
        rec["subject_id"] = int(rec.get("subject_id", rec.get("user_id")))
        rec["trial_id"] = int(rec["trial_id"])
        rec["score_pos"] = float(rec.get("score_pos", rec.get("prob_pos", 0.0)))
        if "prob_pos" in rec:
            rec["prob_pos"] = float(rec["prob_pos"])
        subject_trials[rec["subject_id"]].append(rec)

    results = []
    for subject_id, records in subject_trials.items():
        records = sorted(records, key=lambda x: x["score_pos"], reverse=True)
        cur_k = int(k_pos) if len(records) == 8 else int(round(len(records) * 0.5))
        cur_k = max(0, min(cur_k, len(records)))
        positive_keys = {(r["subject_id"], r["trial_id"]) for r in records[:cur_k]}
        for r in records:
            out = dict(r)
            out["Emotion_label"] = 1 if (r["subject_id"], r["trial_id"]) in positive_keys else 0
            results.append(out)

    return sorted(results, key=lambda x: (x["subject_id"], x["trial_id"]))


def compute_subject_topk_trial_metrics(
    segment_records,
    k_pos=4,
    score_key="score_pos",
    fallback_prob_key="prob_pos",
):
    """
    先 window -> trial 聚合，再 subject 内 top-k。
    每个 subject 的 trial_score 最高的 k_pos 个 trial 判为 positive，其余判为 neutral。
    """
    trial_dict = defaultdict(list)
    for r in segment_records:
        key = (int(r["subject_id"]), int(r["trial_id"]))
        trial_dict[key].append(r)

    trial_score_records = []
    for (subject_id, trial_id), records in trial_dict.items():
        scores = [
            float(r[score_key] if score_key in r else r.get(fallback_prob_key, 0.0))
            for r in records
        ]
        probs = [float(r.get(fallback_prob_key, 0.0)) for r in records]
        labels = [int(r["label_emo"]) for r in records]
        trial_score_records.append({
            "subject_id": int(subject_id),
            "trial_id": int(trial_id),
            "score_pos": float(np.mean(scores)),
            "prob_pos": float(np.mean(probs)),
            "label_emo": int(round(float(np.mean(labels)))),
        })

    topk_records = apply_subject_topk_prediction(trial_score_records, k_pos=k_pos)
    label_map = {
        (int(r["subject_id"]), int(r["trial_id"])): int(r["label_emo"])
        for r in trial_score_records
    }
    score_map = {
        (int(r["subject_id"]), int(r["trial_id"])): float(r["score_pos"])
        for r in trial_score_records
    }
    prob_map = {
        (int(r["subject_id"]), int(r["trial_id"])): float(r["prob_pos"])
        for r in trial_score_records
    }

    topk_trial_keys = []
    topk_trial_scores = []
    topk_trial_probs = []
    topk_trial_preds = []
    topk_trial_labels = []

    for r in topk_records:
        key = (int(r["subject_id"]), int(r["trial_id"]))
        topk_trial_keys.append(key)
        topk_trial_scores.append(score_map[key])
        topk_trial_probs.append(prob_map[key])
        topk_trial_preds.append(int(r["Emotion_label"]))
        topk_trial_labels.append(label_map[key])

    if len(topk_trial_labels) == 0:
        return {
            "topk_trial_acc": 0.0,
            "topk_trial_macro_f1": 0.0,
            "topk_trial_confusion_matrix": np.zeros((2, 2), dtype=int),
            "topk_trial_per_class": {},
            "topk_trial_keys": [],
            "topk_trial_scores": [],
            "topk_trial_probs": [],
            "topk_trial_preds": [],
            "topk_trial_labels": [],
            "topk_subject_gap_mean": 0.0,
            "topk_subject_gap_std": 0.0,
        }

    topk_trial_acc = accuracy_score(topk_trial_labels, topk_trial_preds)
    topk_trial_macro_f1 = f1_score(
        topk_trial_labels,
        topk_trial_preds,
        average="macro",
        labels=[0, 1],
        zero_division=0,
    )
    topk_trial_cm = confusion_matrix(topk_trial_labels, topk_trial_preds, labels=[0, 1])
    p, r, f1, s = precision_recall_fscore_support(
        topk_trial_labels,
        topk_trial_preds,
        labels=[0, 1],
        zero_division=0,
    )
    topk_trial_per_class = {
        int(c): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(s[i])}
        for i, c in enumerate([0, 1])
    }

    gaps = []
    by_subject = defaultdict(list)
    for key, score, pred in zip(topk_trial_keys, topk_trial_scores, topk_trial_preds):
        subject_id, trial_id = key
        by_subject[int(subject_id)].append((int(trial_id), float(score), int(pred)))

    for records in by_subject.values():
        pos_scores = [score for _tid, score, pred in records if pred == 1]
        neg_scores = [score for _tid, score, pred in records if pred == 0]
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            gaps.append(float(np.mean(pos_scores) - np.mean(neg_scores)))

    return {
        "topk_trial_acc": topk_trial_acc,
        "topk_trial_macro_f1": topk_trial_macro_f1,
        "topk_trial_confusion_matrix": topk_trial_cm,
        "topk_trial_per_class": topk_trial_per_class,
        "topk_trial_keys": topk_trial_keys,
        "topk_trial_scores": topk_trial_scores,
        "topk_trial_probs": topk_trial_probs,
        "topk_trial_preds": topk_trial_preds,
        "topk_trial_labels": topk_trial_labels,
        "topk_subject_gap_mean": float(np.mean(gaps)) if len(gaps) > 0 else 0.0,
        "topk_subject_gap_std": float(np.std(gaps)) if len(gaps) > 0 else 0.0,
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
        improved = is_better_by_criteria(
            current_metrics=metrics,
            best_metrics=self.best_metrics,
            criteria=self.criteria,
            eps=self.min_delta,
        )
        if improved:
            self.best_metrics = copy.deepcopy(metrics)
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            self.should_stop = False
        else:
            if epoch >= self.warmup:
                self.num_bad_epochs += 1
            else:
                self.num_bad_epochs = 0
            if epoch >= self.warmup and self.num_bad_epochs >= self.patience:
                self.should_stop = True
        return improved, self.should_stop

    def state_dict(self):
        return {
            "criteria": self.criteria,
            "patience": self.patience,
            "warmup": self.warmup,
            "min_delta": self.min_delta,
            "best_epoch": self.best_epoch,
            "num_bad_epochs": self.num_bad_epochs,
            "should_stop": self.should_stop,
            "best_metrics": self.best_metrics,
        }


# =========================================================
# Dataset / split / weights
# =========================================================

def _subject_group_table(index_csv: str) -> pd.DataFrame:
    """
    返回每个 subject 的 group_binary：0=DEP，1=HC。
    优先用 label4 推断，避免 diagnosis_label 原始编码不统一。
    """
    df = pd.read_csv(index_csv)

    if "label4" in df.columns:
        tmp = df[["subject_id", "label4"]].copy()
        tmp["group_binary"] = (tmp["label4"].astype(int) >= 2).astype(int)  # 0=DEP, 1=HC
        subject_df = (
            tmp.groupby("subject_id")["group_binary"]
            .agg(lambda s: int(s.mode().iloc[0]))
            .reset_index()
        )
    elif "diagnosis_label" in df.columns:
        subject_df = df[["subject_id", "diagnosis_label"]].drop_duplicates().reset_index(drop=True)
        subject_df["group_binary"] = subject_df["diagnosis_label"].astype(int)
    else:
        raise KeyError("index_csv 中至少需要 label4 或 diagnosis_label。")

    subject_df["group_name"] = subject_df["group_binary"].map({0: "dep", 1: "hc"})
    return subject_df


def get_competition_subject_split(
    index_csv: str,
    fold: int = 0,
    n_splits: int = 5,
    seed: int = 42,
    train_group: str = "all",
):
    """
    跨被试划分。

    train_group='all'：HC/DEP 混合，用 StratifiedGroupKFold 保持诊断比例。
    train_group='hc' ：只选正常被试，用 KFold 随机按被试划分。
    train_group='dep'：只选抑郁被试，用 KFold 随机按被试划分。
    """
    train_group = normalize_train_group(train_group)
    subject_df = _subject_group_table(index_csv)

    if train_group == "hc":
        subject_df = subject_df[subject_df["group_binary"] == 1].reset_index(drop=True)
    elif train_group == "dep":
        subject_df = subject_df[subject_df["group_binary"] == 0].reset_index(drop=True)

    if len(subject_df) < n_splits:
        raise ValueError(
            f"train_group={train_group} 下只有 {len(subject_df)} 个被试，不能做 {n_splits} 折。"
        )

    if train_group == "all" and subject_df["group_binary"].nunique() > 1:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(
            splitter.split(
                X=subject_df["subject_id"],
                y=subject_df["group_binary"],
                groups=subject_df["subject_id"],
            )
        )
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(subject_df["subject_id"].values))

    train_idx, val_idx = splits[fold]
    train_subjects = subject_df.iloc[train_idx]["subject_id"].tolist()
    val_subjects = subject_df.iloc[val_idx]["subject_id"].tolist()

    return train_subjects, val_subjects, subject_df


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


def build_diagnosis_class_weights(index_csv: str, subject_ids):
    df = pd.read_csv(index_csv)
    df = df[df["subject_id"].isin(subject_ids)].reset_index(drop=True)

    if "label4" in df.columns:
        diag = (df["label4"].astype(int) >= 2).astype(int)  # 0=DEP, 1=HC
    elif "diagnosis_label" in df.columns:
        diag = df["diagnosis_label"].astype(int)
    else:
        raise KeyError("缺少 label4 / diagnosis_label。")

    counts = diag.value_counts().sort_index().reindex([0, 1], fill_value=1)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights.values, dtype=torch.float32)


# =========================================================
# Train one epoch
# =========================================================

def train_one_epoch_emotion_grl(
    model,
    loader,
    optimizer,
    criterion_emo,
    criterion_diag,
    criterion_subject,
    device,
    lambda_emo: float = 1.0,
    lambda_diag: float = 0.0,
    lambda_subject: float = 0.01,
    lambda_graph: float = 0.0,
    lambda_con: float = 0.05,
    con_temperature: float = 0.1,
    grl_diag: float = 0.0,
    grl_subject: float = 0.01,
    subject_de_mu=None,
    subject_de_std=None,
    subject_bio_mu=None,
    subject_bio_std=None,
):
    model.train()

    total_loss = 0.0
    total_emo_loss = 0.0
    total_diag_loss = 0.0
    total_subject_loss = 0.0
    total_graph_loss = 0.0
    total_con_loss = 0.0

    total_emo_correct = 0
    total_diag_correct = 0
    total_subject_correct = 0
    total_num = 0
    total_subject_num = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        x = batch["x"].to(device)
        de_feat = batch["de_feat"].to(device)
        _, y_emo, y_diag, y_subject = get_targets(batch, device)
        subject_ids_cpu = batch["subject_id"]
        de_mu_batch = gather_subject_baseline(subject_ids_cpu, subject_de_mu, device, dtype=de_feat.dtype)
        de_std_batch = gather_subject_baseline(subject_ids_cpu, subject_de_std, device, dtype=de_feat.dtype)
        bio_mu_batch = gather_subject_baseline(subject_ids_cpu, subject_bio_mu, device, dtype=de_feat.dtype)
        bio_std_batch = gather_subject_baseline(subject_ids_cpu, subject_bio_std, device, dtype=de_feat.dtype)

        optimizer.zero_grad(set_to_none=True)

        out = model(
            x,
            de_feat=de_feat,
            lambda_diag=grl_diag if lambda_diag > 0 else 0.0,
            lambda_subject=grl_subject if lambda_subject > 0 else 0.0,
            subject_id=subject_ids_cpu,
            subject_de_mu=de_mu_batch,
            subject_de_std=de_std_batch,
            subject_bio_mu=bio_mu_batch,
            subject_bio_std=bio_std_batch,
        )

        emo_logits = out["emo_logits"]
        diagnosis_logits = out["diagnosis_logits"]
        subject_logits = out["subject_logits"]
        adj_dense = out["adj_dense"]
        contrast_feat = out.get("contrast_feat", out.get("graph_feat", None))

        loss_emo = criterion_emo(emo_logits, y_emo)
        loss_graph = intra_class_graph_loss(adj_dense, y_emo)

        if lambda_con > 0:
            if contrast_feat is None:
                raise KeyError("模型输出中缺少 contrast_feat / graph_feat，无法计算对比学习损失。")
            loss_con = supervised_contrastive_loss(
                contrast_feat,
                y_emo,
                temperature=con_temperature,
            )
        else:
            loss_con = emo_logits.new_tensor(0.0)

        if lambda_diag > 0:
            loss_diag = criterion_diag(diagnosis_logits, y_diag)
        else:
            loss_diag = emo_logits.new_tensor(0.0)

        if lambda_subject > 0 and y_subject is not None:
            if int(y_subject.max().item()) >= subject_logits.size(1):
                raise ValueError(
                    f"subject target 最大值为 {int(y_subject.max().item())}，"
                    f"但 subject_logits 只有 {subject_logits.size(1)} 类。"
                    f"请增大 EmotionPretrainModel(num_subjects=...)，或检查 dataloader 的 domain_id。"
                )
            loss_subject = criterion_subject(subject_logits, y_subject)
        else:
            loss_subject = emo_logits.new_tensor(0.0)

        loss = (
            lambda_emo * loss_emo
            + lambda_con * loss_con
            + lambda_diag * loss_diag
            + lambda_subject * loss_subject
            + lambda_graph * loss_graph
        )

        loss.backward()
        optimizer.step()

        bsz = x.size(0)
        pred_emo = emo_logits.argmax(dim=1)
        pred_diag = diagnosis_logits.argmax(dim=1)
        pred_subject = subject_logits.argmax(dim=1)

        total_loss += loss.item() * bsz
        total_emo_loss += loss_emo.item() * bsz
        total_diag_loss += loss_diag.item() * bsz
        total_subject_loss += loss_subject.item() * bsz
        total_graph_loss += loss_graph.item() * bsz
        total_con_loss += loss_con.item() * bsz

        total_emo_correct += (pred_emo == y_emo).sum().item()
        total_diag_correct += (pred_diag == y_diag).sum().item()
        if y_subject is not None:
            total_subject_correct += (pred_subject == y_subject).sum().item()
            total_subject_num += bsz
        total_num += bsz

        pbar.set_postfix({
            "loss": f"{total_loss / max(total_num, 1):.4f}",
            "emo": f"{total_emo_loss / max(total_num, 1):.4f}",
            "con": f"{total_con_loss / max(total_num, 1):.4f}",
            "diag": f"{total_diag_loss / max(total_num, 1):.4f}",
            "subj": f"{total_subject_loss / max(total_num, 1):.4f}",
            "emo_acc": f"{total_emo_correct / max(total_num, 1):.4f}",
        })

    if total_num == 0:
        raise RuntimeError("训练集为空：请检查 train_group / train_subjects / Dataset。")

    return {
        "loss": total_loss / total_num,
        "emo_loss": total_emo_loss / total_num,
        "diag_loss": total_diag_loss / total_num,
        "subject_loss": total_subject_loss / total_num,
        "graph_loss": total_graph_loss / total_num,
        "con_loss": total_con_loss / total_num,
        "emotion_acc": total_emo_correct / total_num,
        "diagnosis_acc": total_diag_correct / total_num,
        "subject_acc": total_subject_correct / max(total_subject_num, 1),
    }


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def validate_one_epoch_emotion_grl(
    model,
    loader,
    criterion_emo,
    criterion_diag,
    criterion_subject,
    device,
    lambda_emo: float = 1.0,
    lambda_diag: float = 0.0,
    lambda_subject: float = 0.0,
    lambda_graph: float = 0.0,
    lambda_con: float = 0.05,
    con_temperature: float = 0.1,
    subject_de_mu=None,
    subject_de_std=None,
    subject_bio_mu=None,
    subject_bio_std=None,
):
    """验证一轮 — 已注释验证时恒为 0 的损失以加速。"""

    model.eval()

    total_loss = 0.0
    total_emo_loss = 0.0
    # total_diag_loss = 0.0     # 验证时 lambda_diag 恒为 0，已注释
    # total_subject_loss = 0.0  # 验证时 lambda_subject 恒为 0，已注释
    # total_graph_loss = 0.0    # 验证时 lambda_graph 恒为 0，已注释
    total_con_loss = 0.0
    total_num = 0

    all_emo_preds, all_emo_labels = [], []
    all_diag_preds, all_diag_labels = [], []
    # all_subject_preds, all_subject_labels = [], []  # 验证时不需要
    segment_records = []

    _need_con = lambda_con > 0

    pbar = tqdm(loader, desc="Val", leave=False)

    for batch in pbar:
        x = batch["x"].to(device)
        de_feat = batch["de_feat"].to(device)
        _, y_emo, y_diag, y_subject = get_targets(batch, device)
        subject_ids_cpu = batch["subject_id"]
        trial_ids_cpu = batch["trial_id"]
        de_mu_batch = gather_subject_baseline(subject_ids_cpu, subject_de_mu, device, dtype=de_feat.dtype)
        de_std_batch = gather_subject_baseline(subject_ids_cpu, subject_de_std, device, dtype=de_feat.dtype)
        bio_mu_batch = gather_subject_baseline(subject_ids_cpu, subject_bio_mu, device, dtype=de_feat.dtype)
        bio_std_batch = gather_subject_baseline(subject_ids_cpu, subject_bio_std, device, dtype=de_feat.dtype)

        out = model(
            x,
            de_feat=de_feat,
            lambda_diag=0.0,
            lambda_subject=0.0,
            subject_id=subject_ids_cpu,
            subject_de_mu=de_mu_batch,
            subject_de_std=de_std_batch,
            subject_bio_mu=bio_mu_batch,
            subject_bio_std=bio_std_batch,
        )

        emo_logits = out["emo_logits"]
        diagnosis_logits = out["diagnosis_logits"]
        subject_logits = out["subject_logits"]
        # adj_dense = out["adj_dense"]  # 验证时 lambda_graph=0，不计算 graph loss
        contrast_feat = out.get("contrast_feat", out.get("graph_feat", None))

        loss_emo = criterion_emo(emo_logits, y_emo)

        if _need_con:
            if contrast_feat is None:
                raise KeyError("模型输出中缺少 contrast_feat / graph_feat，无法计算对比学习损失。")
            loss_con = supervised_contrastive_loss(
                contrast_feat,
                y_emo,
                temperature=con_temperature,
            )
        else:
            loss_con = emo_logits.new_tensor(0.0)

        # ── 以下损失在验证时 lambda 恒为 0，已注释以加速 ──
        # loss_diag = criterion_diag(diagnosis_logits, y_diag) if lambda_diag > 0 else emo_logits.new_tensor(0.0)
        # loss_subject = ...
        # loss_graph = intra_class_graph_loss(adj_dense, y_emo)

        loss = lambda_emo * loss_emo + lambda_con * loss_con
        # 原始: + lambda_diag * loss_diag + lambda_subject * loss_subject + lambda_graph * loss_graph

        prob_emo = torch.softmax(emo_logits, dim=1)
        prob_pos = prob_emo[:, 1]
        score_pos = emo_logits[:, 1] - emo_logits[:, 0]

        pred_emo = emo_logits.argmax(dim=1)
        pred_diag = diagnosis_logits.argmax(dim=1)
        # pred_subject = subject_logits.argmax(dim=1)  # 验证时不需要

        for i in range(x.size(0)):
            segment_records.append({
                "subject_id": int(subject_ids_cpu[i]),
                "trial_id": int(trial_ids_cpu[i]),
                "prob_pos": float(prob_pos[i].detach().cpu()),
                "score_pos": float(score_pos[i].detach().cpu()),
                "pred_emo": int(pred_emo[i].detach().cpu()),
                "label_emo": int(y_emo[i].detach().cpu()),
                "pred_diag": int(pred_diag[i].detach().cpu()),
                "label_diag": int(y_diag[i].detach().cpu()),
            })

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_emo_loss += loss_emo.item() * bsz
        # total_diag_loss += loss_diag.item() * bsz     # 已注释
        # total_subject_loss += loss_subject.item() * bsz  # 已注释
        # total_graph_loss += loss_graph.item() * bsz   # 已注释
        total_con_loss += loss_con.item() * bsz
        total_num += bsz

        all_emo_preds.extend(pred_emo.detach().cpu().numpy().tolist())
        all_emo_labels.extend(y_emo.detach().cpu().numpy().tolist())
        all_diag_preds.extend(pred_diag.detach().cpu().numpy().tolist())
        all_diag_labels.extend(y_diag.detach().cpu().numpy().tolist())
        # if y_subject is not None:     # 验证时不需要
        #     all_subject_preds.extend(pred_subject.detach().cpu().numpy().tolist())
        #     all_subject_labels.extend(y_subject.detach().cpu().numpy().tolist())

        pbar.set_postfix({
            "loss": f"{total_loss / max(total_num, 1):.4f}",
            "emo_acc": f"{accuracy_score(all_emo_labels, all_emo_preds):.4f}" if len(all_emo_labels) > 0 else "0.0000",
        })

    if total_num == 0:
        raise RuntimeError("验证集为空：请检查 val_dataset / val_loader。")

    emotion_acc = accuracy_score(all_emo_labels, all_emo_preds)
    emotion_macro_f1 = f1_score(
        all_emo_labels,
        all_emo_preds,
        average="macro",
        labels=[0, 1],
        zero_division=0,
    )
    emotion_cm = confusion_matrix(all_emo_labels, all_emo_preds, labels=[0, 1])

    pe, re, f1e, se = precision_recall_fscore_support(
        all_emo_labels,
        all_emo_preds,
        labels=[0, 1],
        zero_division=0,
    )
    per_class_emo = {
        int(c): {
            "precision": float(pe[i]),
            "recall": float(re[i]),
            "f1": float(f1e[i]),
            "support": int(se[i]),
        }
        for i, c in enumerate([0, 1])
    }

    diagnosis_acc = accuracy_score(all_diag_labels, all_diag_preds)
    diagnosis_macro_f1 = f1_score(
        all_diag_labels,
        all_diag_preds,
        average="macro",
        labels=[0, 1],
        zero_division=0,
    )
    diagnosis_cm = confusion_matrix(all_diag_labels, all_diag_preds, labels=[0, 1])

    # ── subject_acc 在验证时不需要（lambda_subject 恒为 0），设 0 ──
    subject_acc = 0.0

    trial_metrics = compute_trial_level_metrics(segment_records, threshold=0.5, vote_method="hard")
    topk_trial_metrics = compute_subject_topk_trial_metrics(
        segment_records,
        k_pos=4,
        score_key="score_pos",
    )

    pt, rt, f1t, st = precision_recall_fscore_support(
        trial_metrics["trial_labels"],
        trial_metrics["trial_preds"],
        labels=[0, 1],
        zero_division=0,
    )
    trial_per_class_emo = {
        int(c): {
            "precision": float(pt[i]),
            "recall": float(rt[i]),
            "f1": float(f1t[i]),
            "support": int(st[i]),
        }
        for i, c in enumerate([0, 1])
    }

    return {
        "loss": total_loss / total_num,
        "emo_loss": total_emo_loss / total_num,
        "diag_loss": 0.0,       # 验证时 lambda_diag 恒为 0
        "subject_loss": 0.0,    # 验证时 lambda_subject 恒为 0
        "graph_loss": 0.0,      # 验证时 lambda_graph 恒为 0
        "con_loss": total_con_loss / total_num,

        "emotion_acc": emotion_acc,
        "emotion_macro_f1": emotion_macro_f1,
        "emotion_confusion_matrix": emotion_cm,
        "per_class_emo": per_class_emo,
        "segment_preds_emo": all_emo_preds,
        "segment_labels_emo": all_emo_labels,

        "diagnosis_acc": diagnosis_acc,
        "diagnosis_macro_f1": diagnosis_macro_f1,
        "diagnosis_confusion_matrix": diagnosis_cm,
        "subject_acc": subject_acc,

        "trial_acc": trial_metrics["trial_acc"],
        "trial_macro_f1": trial_metrics["trial_macro_f1"],
        "trial_confusion_matrix": trial_metrics["trial_confusion_matrix"],
        "trial_per_class_emo": trial_per_class_emo,
        "trial_keys": trial_metrics["trial_keys"],
        "trial_probs": trial_metrics["trial_probs"],
        "trial_preds": trial_metrics["trial_preds"],
        "trial_labels": trial_metrics["trial_labels"],

        "topk_trial_acc": topk_trial_metrics["topk_trial_acc"],
        "topk_trial_macro_f1": topk_trial_metrics["topk_trial_macro_f1"],
        "topk_trial_confusion_matrix": topk_trial_metrics["topk_trial_confusion_matrix"],
        "topk_trial_per_class": topk_trial_metrics["topk_trial_per_class"],
        "topk_trial_keys": topk_trial_metrics["topk_trial_keys"],
        "topk_trial_scores": topk_trial_metrics["topk_trial_scores"],
        "topk_trial_probs": topk_trial_metrics["topk_trial_probs"],
        "topk_trial_preds": topk_trial_metrics["topk_trial_preds"],
        "topk_trial_labels": topk_trial_metrics["topk_trial_labels"],
        "topk_subject_gap_mean": topk_trial_metrics["topk_subject_gap_mean"],
        "topk_subject_gap_std": topk_trial_metrics["topk_subject_gap_std"],

        "segment_records": segment_records,
    }


# =========================================================
# Checkpoint
# =========================================================

def save_best_checkpoint(
    save_dir: str,
    fold: int,
    best_name: str,
    model,
    optimizer,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    train_subjects,
    val_subjects,
    criteria,
    config: dict,
):
    os.makedirs(save_dir, exist_ok=True)

    model_file = "hc_best.pt" if best_name == "combined" else f"best_{best_name}_fold{fold}.pt"
    ckpt_path = os.path.join(save_dir, model_file)
    json_path = os.path.join(save_dir, f"best_{best_name}_fold{fold}_metrics.json")
    csv_path = os.path.join(save_dir, f"best_{best_name}_fold{fold}_summary.csv")

    ckpt = {
        "best_name": best_name,
        "epoch": epoch,
        "criteria": criteria,
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

    save_json(json_path, {
        "best_name": best_name,
        "epoch": epoch,
        "criteria": criteria,
        "criteria_readable": format_criteria(criteria),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "config": config,
        "checkpoint_path": ckpt_path,
    })

    row = {
        "best_name": best_name,
        "epoch": epoch,
        "criteria": format_criteria(criteria),
        "checkpoint_path": ckpt_path,
    }
    row.update(flatten_metrics_for_csv("train", train_metrics))
    row.update(flatten_metrics_for_csv("val", val_metrics))
    pd.DataFrame([row]).to_csv(csv_path, index=False, encoding="utf-8-sig")

    return ckpt_path, json_path, csv_path


# =========================================================
# Cross-subject training
# =========================================================

def train_competition_cross_subject(
    model,
    index_csv: str,
    fold: int,
    save_dir: str,
    dataset: str,
    rand: int,
    train_group: str = "dep",
    n_splits: int = 5,
    epochs: int = 80,
    batch_size: int = 128,
    subjects_per_batch: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    lambda_emo: float = 1.0,
    lambda_diag: float = 0.0,
    lambda_subject: float = 0.01,
    lambda_graph: float = 0.0,
    lambda_con: float = 0.05,
    con_temperature: float = 0.1,
    grl_diag: float = 0.0,
    grl_subject: float = 0.01,
    device: str = "cuda",
    run_seed: int = None,
    deterministic: bool = True,
    early_stop_patience: int = 25,
    early_stop_warmup: int = 15,
    early_stop_min_delta: float = 1e-6,
    early_stop_track: str = "combined",
    use_subject_relative_de: bool = True,
    use_subject_relative_bio: bool = True,
    bio_abs_scale: float = 0.3,
    relative_eps: float = 1e-6,
):
    os.makedirs(save_dir, exist_ok=True)
    train_group = normalize_train_group(train_group)

    if train_group in ["hc", "dep"] and lambda_diag > 0:
        print(
            f"[Warning] 当前 train_group={train_group}，诊断标签为单一类别，"
            f"diagnosis GRL 不具备有效对抗意义。建议 lambda_diag=0。"
        )

    if run_seed is None:
        run_seed = config.make_run_seed(rand, fold)

    set_global_seed(run_seed, deterministic=deterministic)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(run_seed)

    print(f"[Seed] rand={rand}, fold={fold}, run_seed={run_seed}, deterministic={deterministic}")
    print(f"[TrainGroup] {train_group}")

    # ── 统一交叉验证划分：三个模型共用同一套 StratifiedGroupKFold ──
    split = get_unified_subject_split(
        index_csv=index_csv,
        fold=fold,
        n_splits=n_splits,
        seed=rand,
    )
    if train_group == "hc":
        train_subjects = split["train_hc"]
        val_subjects = split["val_hc"]
    elif train_group == "dep":
        train_subjects = split["train_dep"]
        val_subjects = split["val_dep"]
    else:
        train_subjects = split["train_all"]
        val_subjects = split["val_all"]
    subject_df = split["subject_df"]

    print(f"Fold {fold}")
    print("train subjects:", len(train_subjects), train_subjects)
    print("val subjects:", len(val_subjects), val_subjects)
    print("group counts:")
    print(subject_df["group_name"].value_counts())

    if dataset == "com":
        train_dataset = Competition4ClassDataset(
            index_csv=index_csv,
            subject_ids=train_subjects,
            normalize=False,
        )
        val_dataset = Competition4ClassDataset(
            index_csv=index_csv,
            subject_ids=val_subjects,
            normalize=False,
        )
        collate_fn = comp4_collate_fn
    elif dataset == "seed":
        train_dataset = EEGWindowDataset(
            index_csv="seed_window_index.csv",
            subject_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            normalize=True,
            selected_emotions=["nue", "pos"],
            label_map={"nue": 0, "pos": 1},
        )
        val_dataset = EEGWindowDataset(
            index_csv="seed_window_index.csv",
            subject_ids=[13, 14, 15],
            normalize=True,
            selected_emotions=["nue", "pos"],
            label_map={"nue": 0, "pos": 1},
        )
        collate_fn = seed_collate_fn
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    effective_subjects_per_batch = min(int(subjects_per_batch), max(len(train_subjects), 1))
    train_batch_sampler = MultiSubjectBatchSampler(
        sample_subject_ids=train_dataset.sample_subject_ids,
        batch_size=batch_size,
        subjects_per_batch=effective_subjects_per_batch,
        drop_last=True,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    use_subject_relative_de = bool(use_subject_relative_de and getattr(model, "use_subject_relative_de", False))
    use_subject_relative_bio = bool(use_subject_relative_bio and getattr(model, "use_subject_relative_bio", False))
    print(
        f"[Subject Relative] DE={use_subject_relative_de}, "
        f"bio={use_subject_relative_bio}, bio_abs_scale={bio_abs_scale}, eps={relative_eps}"
    )

    subject_de_mu = subject_de_std = None
    subject_bio_mu = subject_bio_std = None
    val_subject_de_mu = val_subject_de_std = None
    val_subject_bio_mu = val_subject_bio_std = None

    if use_subject_relative_de or use_subject_relative_bio:
        train_baseline_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
        )
        val_baseline_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
        )
    else:
        train_baseline_loader = val_baseline_loader = None

    if use_subject_relative_de:
        print("[Subject Relative] computing train/val DE baselines from each split itself...")
        subject_de_mu, subject_de_std = compute_subject_de_baselines(train_dataset, eps=relative_eps)
        val_subject_de_mu, val_subject_de_std = compute_subject_de_baselines(val_dataset, eps=relative_eps)

    if use_subject_relative_bio:
        print("[Subject Relative] computing train/val bio baselines with model forward, labels are not used...")
        subject_bio_mu, subject_bio_std = compute_subject_bio_baselines(
            model,
            train_baseline_loader,
            device,
            subject_de_mu=subject_de_mu,
            subject_de_std=subject_de_std,
            eps=relative_eps,
        )
        val_subject_bio_mu, val_subject_bio_std = compute_subject_bio_baselines(
            model,
            val_baseline_loader,
            device,
            subject_de_mu=val_subject_de_mu,
            subject_de_std=val_subject_de_std,
            eps=relative_eps,
        )

    # -------- losses --------
    emotion_weights = build_emotion_class_weights(index_csv, train_subjects).to(device)
    diagnosis_weights = build_diagnosis_class_weights(index_csv, train_subjects).to(device)

    criterion_emo = torch.nn.CrossEntropyLoss(weight=emotion_weights, label_smoothing=0.03)
    criterion_diag = torch.nn.CrossEntropyLoss(weight=diagnosis_weights, label_smoothing=0.03)
    criterion_subject = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    config = {
        "index_csv": index_csv,
        "fold": fold,
        "rand": rand,
        "run_seed": run_seed,
        "deterministic": deterministic,
        "train_group": train_group,
        "n_splits": n_splits,
        "epochs": epochs,
        "batch_size": batch_size,
        "subjects_per_batch": subjects_per_batch,
        "effective_subjects_per_batch": effective_subjects_per_batch,
        "lr": lr,
        "weight_decay": weight_decay,
        "lambda_emo": lambda_emo,
        "lambda_diag": lambda_diag,
        "lambda_subject": lambda_subject,
        "lambda_graph": lambda_graph,
        "lambda_con": lambda_con,
        "con_temperature": con_temperature,
        "grl_diag": grl_diag,
        "grl_subject": grl_subject,
        "dataset": dataset,
        "early_stop_patience": early_stop_patience,
        "early_stop_warmup": early_stop_warmup,
        "early_stop_min_delta": early_stop_min_delta,
        "early_stop_track": early_stop_track,
        "use_subject_relative_de": use_subject_relative_de,
        "use_subject_relative_bio": use_subject_relative_bio,
        "bio_abs_scale": bio_abs_scale,
        "relative_eps": relative_eps,
    }

    best_trackers = {
        "combined": {
            "criteria": [
                ("topk_trial_macro_f1", "max"),
                ("topk_trial_acc", "max"),
                ("trial_macro_f1", "max"),
                ("trial_acc", "max"),
                ("emotion_macro_f1", "max"),
                ("emotion_acc", "max"),
                ("loss", "min"),
            ],
            "best_metrics": None,
            "best_epoch": None,
            "checkpoint_path": None,
        },
        "topk_trial_f1": {
            "criteria": [
                ("topk_trial_macro_f1", "max"),
                ("topk_trial_acc", "max"),
                ("topk_subject_gap_mean", "max"),
                ("loss", "min"),
            ],
            "best_metrics": None,
            "best_epoch": None,
            "checkpoint_path": None,
        },
        "topk_trial_acc": {
            "criteria": [
                ("topk_trial_acc", "max"),
                ("topk_trial_macro_f1", "max"),
                ("topk_subject_gap_mean", "max"),
                ("loss", "min"),
            ],
            "best_metrics": None,
            "best_epoch": None,
            "checkpoint_path": None,
        },
        "trial_f1": {
            "criteria": [
                ("trial_macro_f1", "max"),
                ("trial_acc", "max"),
                ("loss", "min"),
            ],
            "best_metrics": None,
            "best_epoch": None,
            "checkpoint_path": None,
        },
        "trial_acc": {
            "criteria": [
                ("trial_acc", "max"),
                ("trial_macro_f1", "max"),
                ("loss", "min"),
            ],
            "best_metrics": None,
            "best_epoch": None,
            "checkpoint_path": None,
        },
        "segment_emo_f1": {
            "criteria": [
                ("emotion_macro_f1", "max"),
                ("emotion_acc", "max"),
                ("trial_macro_f1", "max"),
                ("loss", "min"),
            ],
            "best_metrics": None,
            "best_epoch": None,
            "checkpoint_path": None,
        },
    }

    if early_stop_track not in best_trackers:
        raise ValueError(f"early_stop_track={early_stop_track} 不在 {list(best_trackers.keys())} 中。")

    early_stopper = CriteriaEarlyStopping(
        criteria=best_trackers[early_stop_track]["criteria"],
        patience=early_stop_patience,
        warmup=early_stop_warmup,
        min_delta=early_stop_min_delta,
    )

    print(
        f"[EarlyStopping] track={early_stop_track}, "
        f"criteria={format_criteria(early_stopper.criteria)}, "
        f"patience={early_stop_patience}, warmup={early_stop_warmup}"
    )

    history = []

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")

        train_metrics = train_one_epoch_emotion_grl(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion_emo=criterion_emo,
            criterion_diag=criterion_diag,
            criterion_subject=criterion_subject,
            device=device,
            lambda_emo=lambda_emo,
            lambda_diag=lambda_diag,
            lambda_subject=lambda_subject,
            lambda_graph=lambda_graph,
            lambda_con=lambda_con,
            con_temperature=con_temperature,
            grl_diag=grl_diag,
            grl_subject=grl_subject,
            subject_de_mu=subject_de_mu,
            subject_de_std=subject_de_std,
            subject_bio_mu=subject_bio_mu,
            subject_bio_std=subject_bio_std,
        )

        val_metrics = validate_one_epoch_emotion_grl(
            model=model,
            loader=val_loader,
            criterion_emo=criterion_emo,
            criterion_diag=criterion_diag,
            criterion_subject=criterion_subject,
            device=device,
            lambda_emo=lambda_emo,
            lambda_diag=0.0,
            lambda_subject=0.0,
            lambda_graph=lambda_graph,
            lambda_con=lambda_con,
            con_temperature=con_temperature,
            subject_de_mu=val_subject_de_mu,
            subject_de_std=val_subject_de_std,
            subject_bio_mu=val_subject_bio_mu,
            subject_bio_std=val_subject_bio_std,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Train] loss={train_metrics['loss']:.4f} "
            f"emo={train_metrics['emo_loss']:.4f} "
            f"con={train_metrics['con_loss']:.4f} "
            f"diag={train_metrics['diag_loss']:.4f} "
            f"subj={train_metrics['subject_loss']:.4f} "
            f"emo_acc={train_metrics['emotion_acc']:.4f} "
            f"diag_acc={train_metrics['diagnosis_acc']:.4f} "
            f"subj_acc={train_metrics['subject_acc']:.4f}"
        )
        print(
            f"[Val]   loss={val_metrics['loss']:.4f} "
            f"emo_loss={val_metrics['emo_loss']:.4f} "
            f"con_loss={val_metrics['con_loss']:.4f} "
            f"seg_emo_acc={val_metrics['emotion_acc']:.4f} "
            f"seg_emo_f1={val_metrics['emotion_macro_f1']:.4f} "
            f"trial_acc={val_metrics['trial_acc']:.4f} "
            f"trial_f1={val_metrics['trial_macro_f1']:.4f} "
            f"topk_trial_acc={val_metrics['topk_trial_acc']:.4f} "
            f"topk_trial_f1={val_metrics['topk_trial_macro_f1']:.4f} "
            f"topk_gap={val_metrics['topk_subject_gap_mean']:.4f} "
            f"diag_acc={val_metrics['diagnosis_acc']:.4f}"
        )

        hist_row = {"epoch": epoch, "lr": current_lr}
        hist_row.update(flatten_metrics_for_csv("train", train_metrics))
        hist_row.update(flatten_metrics_for_csv("val", val_metrics))
        history.append(hist_row)

        epoch_metrics_json = os.path.join(save_dir, f"epoch_{epoch:03d}_metrics.json")
        save_json(epoch_metrics_json, {
            "epoch": epoch,
            "lr": current_lr,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        })

        for best_name, tracker in best_trackers.items():
            criteria = tracker["criteria"]
            if is_better_by_criteria(val_metrics, tracker["best_metrics"], criteria):
                ckpt_path, json_path, csv_path = save_best_checkpoint(
                    save_dir=save_dir,
                    fold=fold,
                    best_name=best_name,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    train_subjects=train_subjects,
                    val_subjects=val_subjects,
                    criteria=criteria,
                    config=config,
                )
                tracker["best_metrics"] = copy.deepcopy(val_metrics)
                tracker["best_train_metrics"] = copy.deepcopy(train_metrics)
                tracker["best_epoch"] = epoch
                tracker["checkpoint_path"] = ckpt_path
                tracker["metrics_json_path"] = json_path
                tracker["summary_csv_path"] = csv_path
                print(f"保存 best_{best_name}: epoch={epoch}, criteria={format_criteria(criteria)}, path={ckpt_path}")

        early_improved, should_stop = early_stopper.step(val_metrics, epoch)
        print(
            f"[EarlyStopping] improved={early_improved}, "
            f"best_epoch={early_stopper.best_epoch}, "
            f"bad_epochs={early_stopper.num_bad_epochs}/{early_stop_patience}"
        )

        if should_stop:
            print(f"\n触发早停: epoch={epoch}，连续 {early_stopper.num_bad_epochs} 个 epoch 未提升。")
            early_stop_json = os.path.join(save_dir, f"early_stop_fold{fold}.json")
            save_json(early_stop_json, {
                "stopped": True,
                "stop_epoch": epoch,
                "early_stop_track": early_stop_track,
                "early_stopper": early_stopper.state_dict(),
                "config": config,
            })
            print(f"早停信息已保存到: {early_stop_json}")
            break

    early_stop_state_json = os.path.join(save_dir, f"early_stop_state_fold{fold}.json")
    save_json(early_stop_state_json, {
        "early_stop_track": early_stop_track,
        "early_stopper": early_stopper.state_dict(),
        "config": config,
    })

    history_df = pd.DataFrame(history)
    history_csv = os.path.join(save_dir, f"history_fold{fold}.csv")
    history_df.to_csv(history_csv, index=False, encoding="utf-8-sig")
    print(f"训练历史已保存到: {history_csv}")

    best_rows = []
    for best_name, tracker in best_trackers.items():
        row = {
            "best_name": best_name,
            "best_epoch": tracker["best_epoch"],
            "criteria": format_criteria(tracker["criteria"]),
            "checkpoint_path": tracker["checkpoint_path"],
        }
        if tracker["best_metrics"] is not None:
            row.update(flatten_metrics_for_csv("val", tracker["best_metrics"]))
        best_rows.append(row)

    best_summary_csv = os.path.join(save_dir, f"best_summary_fold{fold}.csv")
    pd.DataFrame(best_rows).to_csv(best_summary_csv, index=False, encoding="utf-8-sig")
    print(f"本折所有 best 汇总已保存到: {best_summary_csv}")

    return {
        "fold": fold,
        "rand": rand,
        "run_seed": run_seed,
        "train_group": train_group,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "history_csv": history_csv,
        "best_summary_csv": best_summary_csv,
        "best_models": best_trackers,
        "early_stop": early_stopper.state_dict(),
    }


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--batch_size", type=int, default=256)
    _args, _ = _parser.parse_known_args()
    _bs = _args.batch_size

    device = "cuda:0"

    # 可选："hc" / "dep" / "all"
    #   hc  : 只训练正常被试内部的 neutral vs positive
    #   dep : 只训练抑郁被试内部的 neutral vs positive
    #   all : 正常+抑郁一起训练情绪，同时可打开 diagnosis GRL
    train_group = "hc"

    version = f"{train_group}"
    model_params_root = "model_params"
    os.makedirs(version, exist_ok=True)
    os.makedirs(model_params_root, exist_ok=True)

    all_fold_rows = []
    all_best_rows_long = []
    combined_emotion_confusions = []
    combined_trial_confusions = []

    for repeat, rand in enumerate(config.HC_REPEAT_SEEDS):
        print(f"\n{'#' * 70}")
        print(f"开始 random seed = {rand} 的 5 折交叉验证 | train_group={train_group}")
        print(f"{'#' * 70}")

        seed_combined_rows = []
        n_splits = 5

        for fold in range(n_splits):
            print(f"\n{'=' * 50}")
            print(f"开始训练第 {fold + 1}/{n_splits} 折 | seed={rand} | group={train_group}")
            print(f"{'=' * 50}")

            run_seed = config.make_run_seed(rand, fold)
            set_global_seed(run_seed, deterministic=True)
            print(f"[Main Seed] rand={rand}, fold={fold}, run_seed={run_seed}")

            model = EmotionPretrainModel(
                sfreq=250.0,
                topk=6,
                dropout=0.2,
                num_subjects=60,
                emotion_nclass=2,
                diagnosis_nclass=2,
                contrast_dim=64,
                contrast_hidden_dim=128,
                use_subject_relative_de=True,
                use_subject_relative_bio=True,
                bio_abs_scale=0.3,
                relative_eps=1e-6,
                de_num_bands=5,
            )

            result = train_competition_cross_subject(
                model=model,
                index_csv="com_index_sub_2s.csv",
                fold=fold,
                save_dir=os.path.join(model_params_root, f"hc_reapt{repeat}_fold{fold}"),
                dataset="com",
                train_group=train_group,
                n_splits=n_splits,
                epochs=100,
                batch_size=_bs,
                subjects_per_batch=4,
                lr=1e-4,
                rand=rand,
                weight_decay=1e-4,
                num_workers=4,

                # 主任务：情绪识别
                lambda_emo=1.0,

                # 监督对比学习：同一情绪类拉近，不同情绪类拉远
                lambda_con=0.0,
                con_temperature=0.1,

                # train_group='hc'/'dep' 时诊断标签恒定，建议关闭
                # train_group='all' 时可以尝试 lambda_diag=0.01, grl_diag=0.01 起步
                lambda_diag=0.0,
                grl_diag=0.0,

                # 被试域对抗，建议小权重起步
                lambda_subject=0.0,
                grl_subject=0.0,

                # 图正则先关闭，主任务稳定后再开
                lambda_graph=0.0,

                device=device,
                run_seed=run_seed,
                deterministic=True,
                early_stop_track="combined",
                early_stop_patience=25,
                early_stop_warmup=15,
                early_stop_min_delta=1e-6,
                use_subject_relative_de=True,
                use_subject_relative_bio=True,
                bio_abs_scale=0.3,
                relative_eps=1e-6,
            )

            combined_tracker = result["best_models"]["combined"]
            combined_metrics = combined_tracker["best_metrics"]

            fold_row = {
                "rand": rand,
                "fold": fold,
                "run_seed": run_seed,
                "train_group": train_group,
                "best_name": "combined",
                "best_epoch": combined_tracker["best_epoch"],
                "checkpoint_path": combined_tracker["checkpoint_path"],
            }
            fold_row.update(flatten_metrics_for_csv("val", combined_metrics))
            all_fold_rows.append(fold_row)
            seed_combined_rows.append(fold_row)

            for best_name, tracker in result["best_models"].items():
                row = {
                    "rand": rand,
                    "fold": fold,
                    "run_seed": run_seed,
                    "train_group": train_group,
                    "best_name": best_name,
                    "best_epoch": tracker["best_epoch"],
                    "criteria": format_criteria(tracker["criteria"]),
                    "checkpoint_path": tracker["checkpoint_path"],
                }
                if tracker["best_metrics"] is not None:
                    row.update(flatten_metrics_for_csv("val", tracker["best_metrics"]))
                all_best_rows_long.append(row)

            if combined_metrics is not None:
                combined_emotion_confusions.append(combined_metrics["emotion_confusion_matrix"])
                combined_trial_confusions.append(combined_metrics["trial_confusion_matrix"])

            print(f"第 {fold + 1} 折 combined 最优结果:")
            print(f"best_epoch = {combined_tracker['best_epoch']}")
            print(f"val_segment_emo_acc = {combined_metrics['emotion_acc']:.4f}")
            print(f"val_segment_emo_f1 = {combined_metrics['emotion_macro_f1']:.4f}")
            print(f"val_trial_acc = {combined_metrics['trial_acc']:.4f}")
            print(f"val_trial_f1 = {combined_metrics['trial_macro_f1']:.4f}")
            print(f"val_topk_trial_acc = {combined_metrics['topk_trial_acc']:.4f}")
            print(f"val_topk_trial_f1 = {combined_metrics['topk_trial_macro_f1']:.4f}")
            print(f"val_topk_subject_gap = {combined_metrics['topk_subject_gap_mean']:.4f}")

        seed_df = pd.DataFrame(seed_combined_rows)
        seed_csv = os.path.join(version, f"seed{rand}_combined_5fold_results.csv")
        seed_df.to_csv(seed_csv, index=False, encoding="utf-8-sig")

        metric_cols = [
            "val_emotion_acc",
            "val_emotion_macro_f1",
            "val_trial_acc",
            "val_trial_macro_f1",
            "val_topk_trial_acc",
            "val_topk_trial_macro_f1",
            "val_topk_subject_gap_mean",
            "val_loss",
        ]

        summary_lines = []
        print(f"\n{'=' * 50}")
        print(f"Random seed {rand} 的 5 折 combined 平均结果 | group={train_group}")
        print(f"{'=' * 50}")

        for col in metric_cols:
            if col in seed_df.columns:
                mean_v = seed_df[col].mean()
                std_v = seed_df[col].std(ddof=0)
                print(f"{col}: {mean_v:.4f} ± {std_v:.4f}")
                summary_lines.append(f"{col}: {mean_v:.4f} ± {std_v:.4f}\n")

        with open(os.path.join(version, f"seed{rand}_combined_5fold_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"Random seed {rand} combined 5折交叉验证结果 | group={train_group}\n")
            f.writelines(summary_lines)
            f.write(f"\n详细结果 CSV: {seed_csv}\n")

    all_fold_df = pd.DataFrame(all_fold_rows)
    all_fold_csv = os.path.join(version, "all_combined_best_metrics.csv")
    all_fold_df.to_csv(all_fold_csv, index=False, encoding="utf-8-sig")
    print(f"\n全部 combined best 指标已保存到: {all_fold_csv}")

    all_best_df = pd.DataFrame(all_best_rows_long)
    all_best_csv = os.path.join(version, "all_parallel_best_metrics_long.csv")
    all_best_df.to_csv(all_best_csv, index=False, encoding="utf-8-sig")
    print(f"全部并列 best 指标已保存到: {all_best_csv}")

    np.save(os.path.join(version, "combined_emotion_confusions.npy"), np.array(combined_emotion_confusions, dtype=object))
    np.save(os.path.join(version, "combined_trial_confusions.npy"), np.array(combined_trial_confusions, dtype=object))

    if len(all_fold_df) > 0:
        metric_cols = [
            "val_emotion_acc",
            "val_emotion_macro_f1",
            "val_trial_acc",
            "val_trial_macro_f1",
            "val_topk_trial_acc",
            "val_topk_trial_macro_f1",
            "val_topk_subject_gap_mean",
            "val_loss",
        ]

        print(f"\n{'=' * 50}")
        print(f"所有 seed × fold 的 combined 总体结果 | group={train_group}")
        print(f"{'=' * 50}")

        overall_summary = []
        for col in metric_cols:
            if col in all_fold_df.columns:
                mean_v = all_fold_df[col].mean()
                std_v = all_fold_df[col].std(ddof=0)
                print(f"{col}: {mean_v:.4f} ± {std_v:.4f}")
                overall_summary.append({"metric": col, "mean": mean_v, "std": std_v})

        overall_summary_csv = os.path.join(version, "overall_combined_summary.csv")
        pd.DataFrame(overall_summary).to_csv(overall_summary_csv, index=False, encoding="utf-8-sig")
        print(f"总体统计已保存到: {overall_summary_csv}")

        # 平均 trial 情绪二分类混淆矩阵
        valid_trial_confs = [np.array(c, dtype=np.float32) for c in combined_trial_confusions if c is not None]
        if len(valid_trial_confs) > 0:
            trial_conf_stack = np.stack(valid_trial_confs, axis=0)
            avg_trial_conf = trial_conf_stack.mean(axis=0)
            row_sum = avg_trial_conf.sum(axis=1, keepdims=True)
            avg_trial_conf_norm = avg_trial_conf / np.maximum(row_sum, 1e-12)

            import matplotlib.pyplot as plt

            labels = ["neutral", "positive"]
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(avg_trial_conf_norm, interpolation="nearest", cmap="Blues")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Average Trial-level Emotion Confusion Matrix ({train_group})")

            thresh = avg_trial_conf_norm.max() / 2.0 if avg_trial_conf_norm.max() != 0 else 0.5
            for i in range(avg_trial_conf_norm.shape[0]):
                for j in range(avg_trial_conf_norm.shape[1]):
                    val = avg_trial_conf_norm[i, j]
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color="white" if val > thresh else "black",
                    )

            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig_path = os.path.join(version, "avg_trial_emotion_confusion.png")
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            print(f"平均 trial 情绪混淆矩阵图已保存到: {fig_path}")
    else:
        print("没有收集到任何 fold 结果，无法计算总体统计。")

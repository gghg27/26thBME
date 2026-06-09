# -*- coding: utf-8 -*-
"""
EEG 情绪二分类 + 四分类辅助任务跨被试训练脚本。

双任务架构：
    - 四分类头 (cls4_head)：z_conv → 4 类情绪 (DEP_neu, DEP_pos, HC_neu, HC_pos)
    - 二分类头 (cls2_head)：z_diag → 2 类情绪 (中性, 正性)
    - 域对抗头 (domain_head)：z_diag → 被试域分类 (GRL)

Loss:
    L = L_4cls(z_conv, label4) + λ_diag × L_2cls(z_diag, emotion_binary)
        + λ_domain × L_domain + λ_center × L_center

注意：
    二分类头做的是情绪正/负性分类（从 label4 % 2 推算），不是抑郁症诊断。
    如需做诊断二分类，请使用 batch["diagnosis_label"] 替换 y_2cls。
"""

import os
import json
import copy
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# V1 模块路径
V1_ROOT = Path(__file__).resolve().parent
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))

import config
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
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
from two_branch import TwoBranchModel
from utils.folds import get_unified_subject_split


# =========================================================
# Random seed utilities
# =========================================================

def set_global_seed(seed: int, deterministic: bool = True):
    """
    设置 Python / NumPy / PyTorch / CUDA 的随机种子，尽量保证实验可复现。

    deterministic=True 会让结果更稳定，但可能降低训练速度。
    如果某些 CUDA 算子报 deterministic 相关错误，可以把 deterministic 改为 False。
    """
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
    """
    DataLoader 多进程 worker 的随机种子。
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def validate_one_epoch_two_branch(
    model,
    loader,
    criterion_4cls,
    criterion_2cls,
    device,
    lambda_diag: float = 1.0,
    lambda_graph: float = 0.1,
):
    """
    验证一轮：双任务评估。

    输出指标：
        - 四分类（情绪）：acc, macro_f1, per_class_4
        - 二分类（情绪）：emotion_acc/emotion_macro_f1（segment 级）、
          trial_acc/trial_macro_f1（trial 级）、subject_acc/subject_macro_f1（subject 级）
    """
    model.eval()

    total_loss = 0.0
    total_ce_4cls = 0.0
    total_ce_2cls = 0.0

    total_correct_4cls = 0
    total_correct_2cls = 0
    total_num = 0

    all_preds_4cls = []
    all_labels_4cls = []
    all_preds_2cls = []
    all_labels_2cls = []
    segment_records = []

    _need_graph = lambda_graph > 0

    pbar = tqdm(loader, desc="Val", leave=False)

    for batch in pbar:
        x = batch["x"].to(device)
        y_4cls = batch["label4"].to(device).long()
        y_2cls = (y_4cls % 2).long()                    # 情绪二分类: 0=中性, 1=正性

        de_feat = batch["de_feat"].to(device)

        out = model(x, de_feat, lambda_dom=0.0, dataset_name="comp4")
        logits_4cls = out["logits_4cls"]
        logits_2cls = out["logits_2cls"]

        loss_4cls = criterion_4cls(logits_4cls, y_4cls)
        loss_2cls = criterion_2cls(logits_2cls, y_2cls)

        if _need_graph:
            adj_dense = out.get("adj_dense", None)
            if adj_dense is not None:
                loss_graph = intra_class_graph_loss(adj_dense, y_4cls)
            else:
                loss_graph = logits_4cls.new_tensor(0.0)
            loss = loss_4cls + lambda_diag * loss_2cls + lambda_graph * loss_graph
        else:
            loss = loss_4cls + lambda_diag * loss_2cls

        pred_4cls = logits_4cls.argmax(dim=1)
        pred_2cls = logits_2cls.argmax(dim=1)

        # 四分类概率：情绪正性概率 = P(class=1) + P(class=3)
        prob_4 = torch.softmax(logits_4cls, dim=1)
        prob_pos_emo = prob_4[:, 1] + prob_4[:, 3]

        # 二分类概率：情绪正性概率 = P(class=1)
        prob_2 = torch.softmax(logits_2cls, dim=1)
        prob_pos_2 = prob_2[:, 1]

        subject_ids_cpu = batch["subject_id"]
        trial_ids_cpu = batch["trial_id"]

        for i in range(x.size(0)):
            segment_records.append({
                "subject_id": int(subject_ids_cpu[i]),
                "trial_id": int(trial_ids_cpu[i]),
                "prob_emo4": float(prob_pos_emo[i].detach().cpu()),   # 四分类头正性概率 P(class=1)+P(class=3)
                "prob_emo2": float(prob_pos_2[i].detach().cpu()),    # 二分类头正性概率 P(class=1)
                "pred_emo": int(pred_2cls[i].detach().cpu()),
                "label_emo": int(y_2cls[i].detach().cpu()),
                "pred_cls": int(pred_4cls[i].detach().cpu()),
                "label_cls": int(y_4cls[i].detach().cpu()),
                "pred4": int(pred_4cls[i].detach().cpu()),
                "label4": int(y_4cls[i].detach().cpu()),
            })

        correct_4cls = (pred_4cls == y_4cls).sum().item()
        correct_2cls = (pred_2cls == y_2cls).sum().item()
        bsz = x.size(0)

        total_loss += loss.item() * bsz
        total_ce_4cls += loss_4cls.item() * bsz
        total_ce_2cls += loss_2cls.item() * bsz
        total_correct_4cls += correct_4cls
        total_correct_2cls += correct_2cls
        total_num += bsz

        all_preds_4cls.extend(pred_4cls.detach().cpu().numpy().tolist())
        all_labels_4cls.extend(y_4cls.detach().cpu().numpy().tolist())
        all_preds_2cls.extend(pred_2cls.detach().cpu().numpy().tolist())
        all_labels_2cls.extend(y_2cls.detach().cpu().numpy().tolist())

        pbar.set_postfix({
            "loss": f"{total_loss / max(total_num, 1):.4f}",
            "acc4": f"{total_correct_4cls / max(total_num, 1):.4f}",
            "acc2": f"{total_correct_2cls / max(total_num, 1):.4f}",
        })

    if total_num == 0:
        raise RuntimeError("验证集为空：total_num == 0，请检查 val_dataset / val_loader。")

    # ── 四分类指标 ──
    cls4_labels = [0, 1, 2, 3]
    acc_4cls = total_correct_4cls / total_num
    macro_f1_4cls = f1_score(all_labels_4cls, all_preds_4cls, average="macro", labels=cls4_labels, zero_division=0)
    cm_4cls = confusion_matrix(all_labels_4cls, all_preds_4cls, labels=cls4_labels)

    p4, r4, f14, s4 = precision_recall_fscore_support(
        all_labels_4cls, all_preds_4cls, labels=cls4_labels, zero_division=0,
    )
    per_class_4 = {
        int(c): {"precision": float(p4[i]), "recall": float(r4[i]), "f1": float(f14[i]), "support": int(s4[i])}
        for i, c in enumerate(cls4_labels)
    }

    # ── 二分类情绪指标（segment 级）──
    emotion_acc = total_correct_2cls / total_num
    emotion_macro_f1 = f1_score(all_labels_2cls, all_preds_2cls, average="macro", labels=[0, 1], zero_division=0)
    emotion_cm = confusion_matrix(all_labels_2cls, all_preds_2cls, labels=[0, 1])

    pe, re, f1e, se = precision_recall_fscore_support(
        all_labels_2cls, all_preds_2cls, labels=[0, 1], zero_division=0,
    )
    per_class_emo = {
        int(c): {"precision": float(pe[i]), "recall": float(re[i]), "f1": float(f1e[i]), "support": int(se[i])}
        for i, c in enumerate([0, 1])
    }

    # ── trial / subject 级聚合 ──
    trial_metrics = compute_trial_level_metrics(segment_records, threshold=0.5, vote_method="hard")
    subject_metrics = compute_subject_level_metrics(segment_records, threshold=0.5, vote_method="trial_hard")

    pt, rt, f1t, st = precision_recall_fscore_support(
        trial_metrics["trial_labels"], trial_metrics["trial_preds"], labels=[0, 1], zero_division=0,
    )
    trial_per_class_emo = {
        int(c): {"precision": float(pt[i]), "recall": float(rt[i]), "f1": float(f1t[i]), "support": int(st[i])}
        for i, c in enumerate([0, 1])
    }

    metrics = {
        "loss": total_loss / total_num,
        "ce_loss_4cls": total_ce_4cls / total_num,
        "ce_loss_2cls": total_ce_2cls / total_num,
        "ce_loss": (total_ce_4cls + lambda_diag * total_ce_2cls) / total_num,
        "graph_loss": 0.0,
        "contrast_loss": 0.0,

        # 四分类
        "acc_4cls": acc_4cls,
        "macro_f1_4cls": macro_f1_4cls,
        "confusion_matrix_4cls": cm_4cls,
        "per_class_4": per_class_4,
        "segment_preds_4": all_preds_4cls,
        "segment_labels_4": all_labels_4cls,

        # 兼容旧 key
        "acc": acc_4cls,
        "macro_f1": macro_f1_4cls,
        "confusion_matrix": cm_4cls,

        # 二分类情绪
        "emotion_acc": emotion_acc,
        "emotion_macro_f1": emotion_macro_f1,
        "emotion_confusion_matrix": emotion_cm,
        "per_class_emo": per_class_emo,
        "segment_preds_emo": all_preds_2cls,
        "segment_labels_emo": all_labels_2cls,

        # trial 级
        "trial_acc": trial_metrics["trial_acc"],
        "trial_macro_f1": trial_metrics["trial_macro_f1"],
        "trial_confusion_matrix": trial_metrics["trial_confusion_matrix"],
        "trial_per_class_emo": trial_per_class_emo,
        "trial_keys": trial_metrics["trial_keys"],
        "trial_probs": trial_metrics["trial_probs"],
        "trial_preds": trial_metrics["trial_preds"],
        "trial_labels": trial_metrics["trial_labels"],

        # subject 级
        "subject_acc": subject_metrics["subject_acc"],
        "subject_macro_f1": subject_metrics["subject_macro_f1"],
        "subject_confusion_matrix": subject_metrics["subject_confusion_matrix"],
        "subject_keys": subject_metrics["subject_keys"],
        "subject_probs": subject_metrics["subject_probs"],
        "subject_preds": subject_metrics["subject_preds"],
        "subject_labels": subject_metrics["subject_labels"],

        # 情绪二分类别名（subject / trial 级）
        "subject_emotion_acc": subject_metrics["subject_acc"],
        "subject_emotion_macro_f1": subject_metrics["subject_macro_f1"],
        "trial_emotion_acc": trial_metrics["trial_acc"],
        "trial_emotion_macro_f1": trial_metrics["trial_macro_f1"],

        "segment_records": segment_records,
    }

    return metrics


# =========================================================
# Train one epoch
# =========================================================

def train_one_epoch_two_branch(
    model,
    loader,
    optimizer,
    criterion_4cls,
    criterion_2cls,
    dom_criterion,
    device,
    lambda_diag: float = 1.0,
    lambda_domain: float = 0.0,
    grl_domain: float = 0.1,
    # 类中心损失（作用于 z_diag）
    center_criterion=None,
    lambda_center: float = 0.0,
    center_warmup_epochs: int = 10,
    current_epoch: int = 1,
):
    """
    双任务训练：四分类情绪 + 二分类情绪。

    L = L_4cls(z_conv, label4) + λ_diag × L_2cls(z_diag, emotion_binary)
        + λ_domain × L_domain + λ_center × L_center
    """
    model.train()

    total_loss = 0.0
    total_loss_4cls = 0.0
    total_loss_2cls = 0.0
    total_loss_dom = 0.0
    total_loss_center = 0.0
    total_correct_4cls = 0
    total_correct_2cls = 0
    total_num = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        x = batch["x"].to(device)
        y_4cls = batch["label4"].to(device).long()                       # 四分类标签
        y_2cls = (y_4cls % 2).long()                                     # 情绪二分类: 0=中性, 1=正性
        subject_id = batch["domain_id"].to(device).long()
        de_feat = batch["de_feat"].to(device)

        optimizer.zero_grad(set_to_none=True)

        out = model(x, de_feat=de_feat, lambda_dom=grl_domain, dataset_name="comp4")

        logits_4cls = out["logits_4cls"]
        logits_2cls = out["logits_2cls"]
        domain_logits = out.get("domain_logits", None)
        graph_feat = out.get("graph_feat", None)  # z_diag

        # ── 四分类情绪 loss ──
        loss_4cls = criterion_4cls(logits_4cls, y_4cls)

        # ── 二分类情绪 loss（中性 vs 正性）──
        loss_2cls = criterion_2cls(logits_2cls, y_2cls)

        # ── 类中心损失（作用于 z_diag）──
        if center_criterion is not None and lambda_center > 0 and graph_feat is not None:
            loss_center, center_info = center_criterion(graph_feat, y_2cls)
            if current_epoch <= center_warmup_epochs:
                center_weight = 0.0
            else:
                center_weight = float(lambda_center)
        else:
            loss_center = logits_4cls.new_tensor(0.0)
            center_weight = 0.0
            center_info = {"center_acc": 0.0}

        # ── 域对抗 loss ──
        if domain_logits is not None and lambda_domain > 0:
            loss_dom = dom_criterion(domain_logits, subject_id)
        else:
            loss_dom = logits_4cls.new_tensor(0.0)

        # ── 总 loss ──
        loss = (
            loss_4cls
            + lambda_diag * loss_2cls
            + center_weight * loss_center
            + lambda_domain * loss_dom
        )

        loss.backward()
        optimizer.step()

        pred_4cls = logits_4cls.argmax(dim=1)
        pred_2cls = logits_2cls.argmax(dim=1)
        bsz = x.size(0)

        total_loss += loss.item() * bsz
        total_loss_4cls += loss_4cls.item() * bsz
        total_loss_2cls += loss_2cls.item() * bsz
        total_loss_dom += loss_dom.item() * bsz
        total_loss_center += loss_center.item() * bsz
        total_correct_4cls += (pred_4cls == y_4cls).sum().item()
        total_correct_2cls += (pred_2cls == y_2cls).sum().item()
        total_num += bsz

        pbar.set_postfix({
            "loss": f"{total_loss / max(total_num, 1):.4f}",
            "L4": f"{total_loss_4cls / max(total_num, 1):.4f}",
            "L2": f"{total_loss_2cls / max(total_num, 1):.4f}",
            "dom": f"{total_loss_dom / max(total_num, 1):.4f}",
            "ctr": f"{total_loss_center / max(total_num, 1):.4f}",
            "acc4": f"{total_correct_4cls / max(total_num, 1):.4f}",
            "acc2": f"{total_correct_2cls / max(total_num, 1):.4f}",
        })

    if total_num == 0:
        raise RuntimeError("训练集为空：total_num == 0，请检查 train_dataset / train_loader。")

    metrics = {
        "loss": total_loss / total_num,
        "loss_4cls": total_loss_4cls / total_num,
        "loss_2cls": total_loss_2cls / total_num,
        "loss_dom": total_loss_dom / total_num,
        "loss_center": total_loss_center / total_num,
        "acc_4cls": total_correct_4cls / total_num,
        "acc_2cls": total_correct_2cls / total_num,
        "acc": total_correct_4cls / total_num,  # 兼容旧接口：acc = 四分类
    }
    return metrics


# =========================================================
# Dataset / sampler / split utilities
# =========================================================

def build_weighted_sampler_from_dataset(dataset):
    labels = dataset.df["label4"].to_numpy()
    class_counts = np.bincount(labels, minlength=4)
    class_weights = 1.0 / np.maximum(class_counts, 1)

    sample_weights = class_weights[labels]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def get_competition_subject_split(
    index_csv: str,
    fold: int = 0,
    n_splits: int = 5,
    seed: int = 42,
):
    df = pd.read_csv(index_csv)
    subject_df = df[["subject_id", "diagnosis_label"]].drop_duplicates().reset_index(drop=True)

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    splits = list(
        sgkf.split(
            X=subject_df["subject_id"],
            y=subject_df["diagnosis_label"],
            groups=subject_df["subject_id"],
        )
    )

    train_idx, val_idx = splits[fold]
    train_subjects = subject_df.iloc[train_idx]["subject_id"].tolist()
    val_subjects = subject_df.iloc[val_idx]["subject_id"].tolist()

    return train_subjects, val_subjects


# =========================================================
# Loss utilities
# =========================================================

def flatten_upper_triangle(adj: torch.Tensor) -> torch.Tensor:
    """
    adj: [B, N, N]
    return: [B, N*(N-1)/2]
    """
    bsz, n, _ = adj.shape
    idx = torch.triu_indices(n, n, offset=1, device=adj.device)
    vec = adj[:, idx[0], idx[1]]
    return vec


def intra_class_graph_loss(
    adj_dense: torch.Tensor,
    labels: torch.Tensor,
    normalize_graph_vec: bool = True,
) -> torch.Tensor:
    """
    adj_dense: [B, N, N]，未稀疏化脑网络
    labels:    [B]
    """
    graph_vec = flatten_upper_triangle(adj_dense)

    if normalize_graph_vec:
        graph_vec = F.normalize(graph_vec, p=2, dim=1)

    unique_labels = labels.unique()
    losses = []

    for c in unique_labels:
        mask = (labels == c)
        if mask.sum() < 2:
            continue

        g = graph_vec[mask]
        g_mean = g.mean(dim=0, keepdim=True)
        loss_c = ((g - g_mean) ** 2).mean()
        losses.append(loss_c)

    if len(losses) == 0:
        return adj_dense.new_tensor(0.0)

    return torch.stack(losses).mean()


def build_class_weights(index_csv: str, subject_ids):
    df = pd.read_csv(index_csv)
    subject_set = {str(s) for s in subject_ids}
    df = df[df["subject_id"].astype(str).isin(subject_set)].reset_index(drop=True)

    counts = df["label4"].value_counts().sort_index()
    counts = counts.reindex([0, 1, 2, 3], fill_value=1)

    weights = counts.sum() / counts
    weights = weights / weights.mean()

    return torch.tensor(weights.values, dtype=torch.float32)


def build_diagnosis_class_weights(index_csv: str, subject_ids):
    """
    为 diagnosis_label 二分类构建类别权重。

    作用：当 DEP/HC 样本量不均衡时，让 CE 对少数类更敏感。
    返回顺序固定为 [class0_weight, class1_weight]。
    """
    df = pd.read_csv(index_csv)
    subject_set = {str(s) for s in subject_ids}
    df = df[df["subject_id"].astype(str).isin(subject_set)].reset_index(drop=True)

    if "diagnosis_label" not in df.columns:
        raise KeyError("index_csv 中缺少 diagnosis_label，无法进行抑郁症/正常人二分类。")

    counts = df["diagnosis_label"].value_counts().sort_index()
    counts = counts.reindex([0, 1], fill_value=1)

    weights = counts.sum() / counts
    weights = weights / weights.mean()

    return torch.tensor(weights.values, dtype=torch.float32)


class RelationCosineContrastLoss(torch.nn.Module):
    """
    按指定类别关系做余弦相似度约束。

    label4:
        0: DEP_neu
        1: DEP_pos
        2: HC_neu
        3: HC_pos

    当前目标:
        1, 3 为正性情绪组，互相拉近；
        0, 2 为中性情绪组，互相拉近；
        1/3 与 0/2 拉开。
    """

    def __init__(
        self,
        positive_classes=(1, 3),
        negative_class=(2,),
        pos_target=0.8,
        neg_margin=0.2,
        use_cross_subject=True,
        use_cross_class_only=True,
        eps=1e-8,
    ):
        super().__init__()
        self.positive_classes = positive_classes
        self.negative_class = negative_class
        self.pos_target = pos_target
        self.neg_margin = neg_margin
        self.use_cross_subject = use_cross_subject
        self.use_cross_class_only = use_cross_class_only
        self.eps = eps

    def forward(self, features, labels, subjects=None):
        device = features.device
        B = features.size(0)

        features = F.normalize(features, dim=1)
        labels = labels.view(-1)

        sim = torch.matmul(features, features.T)
        eye = torch.eye(B, device=device).bool()

        pos_group = (labels == 1) | (labels == 3)
        neg_group = (labels == 0) | (labels == 2)

        pos_pos_mask = pos_group[:, None] & pos_group[None, :]
        neg_neg_mask = neg_group[:, None] & neg_group[None, :]

        pos_pos_mask = pos_pos_mask & (~eye)
        neg_neg_mask = neg_neg_mask & (~eye)

        if self.use_cross_subject and subjects is not None:
            subjects = subjects.view(-1)
            diff_subject = subjects[:, None] != subjects[None, :]
            pos_pos_mask = pos_pos_mask & diff_subject
            neg_neg_mask = neg_neg_mask & diff_subject
        else:
            diff_subject = None

        intra_mask = pos_pos_mask | neg_neg_mask

        inter_mask = (
            (pos_group[:, None] & neg_group[None, :])
            | (neg_group[:, None] & pos_group[None, :])
        )
        inter_mask = inter_mask & (~eye)

        if self.use_cross_subject and subjects is not None:
            inter_mask = inter_mask & diff_subject

        losses = []

        if intra_mask.sum() > 0:
            intra_loss = F.relu(self.pos_target - sim[intra_mask]).mean()
            losses.append(intra_loss)

        if inter_mask.sum() > 0:
            inter_loss = F.relu(sim[inter_mask] - self.neg_margin).mean()
            losses.append(inter_loss)

        if len(losses) == 0:
            return features.new_tensor(0.0)

        return torch.stack(losses).mean()


class ClassCenterContrastLoss(torch.nn.Module):
    """
    类中心对比 CE / Prototype CE。

    作用：
        1. 将特征 z 投影到对比空间；
        2. 维护 num_classes 个可学习类中心；
        3. 用 cosine(z, center) / tau 得到 center_logits；
        4. 对 center_logits 和 labels 做 CE。

    对情绪二分类：
        center_0: 中性情绪中心
        center_1: 正性情绪中心
    """

    def __init__(
        self,
        in_dim: int = 192,
        proj_dim: int = 64,
        num_classes: int = 2,
        tau: float = 0.1,
        dropout: float = 0.2,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.tau = float(tau)
        self.num_classes = int(num_classes)
        self.label_smoothing = float(label_smoothing)

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(in_dim, proj_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(proj_dim, proj_dim),
        )

        self.centers = torch.nn.Parameter(
            torch.randn(num_classes, proj_dim)
        )

        torch.nn.init.xavier_uniform_(self.centers)

    def forward(self, features, labels):
        """
        features: [B, in_dim]，建议传 out["graph_feat"]
        labels:   [B]，情绪二分类标签 (0=中性, 1=正性)
        """
        z = self.projector(features)
        z = F.normalize(z, dim=1)

        centers = F.normalize(self.centers, dim=1)

        center_logits = torch.matmul(z, centers.t()) / self.tau

        loss = F.cross_entropy(
            center_logits,
            labels.long(),
            label_smoothing=self.label_smoothing,
        )

        pred = center_logits.argmax(dim=1)
        acc = (pred == labels).float().mean()

        return loss, {
            "center_acc": float(acc.detach().cpu()),
            "center_logits": center_logits.detach(),
        }




# =========================================================
# Metrics utilities
# =========================================================

def compute_trial_level_metrics(segment_records, threshold=0.5, vote_method="hard"):
    """
    将 segment 级预测聚合成 trial 级情绪二分类结果。

    情绪二分类: 0=中性, 1=正性。
    使用二分类头概率 prob_emo2 和预测 pred_emo。
    """
    trial_dict = defaultdict(list)

    for r in segment_records:
        key = (int(r["subject_id"]), int(r["trial_id"]))
        trial_dict[key].append(r)

    trial_probs = []
    trial_preds = []
    trial_labels = []
    trial_keys = []

    for key, records in trial_dict.items():
        probs = [float(r["prob_emo2"]) for r in records]    # 二分类头正性概率
        preds = [int(r["pred_emo"]) for r in records]
        labels = [int(r["label_emo"]) for r in records]

        # 一个 trial 内标签通常一致；用多数投票比 round(mean) 更稳。
        label = int(np.bincount(labels, minlength=2).argmax())
        mean_prob = float(np.mean(probs))

        if vote_method == "prob":
            pred = 1 if mean_prob >= threshold else 0
        elif vote_method == "hard":
            counts = np.bincount(preds, minlength=2)
            if counts[1] > counts[0]:
                pred = 1
            elif counts[0] > counts[1]:
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


def compute_subject_level_metrics(segment_records, threshold=0.5, vote_method="trial_hard"):
    """
    将 segment 预测进一步聚合到 subject 级情绪二分类。

    subject_acc / subject_macro_f1 比 segment 或 trial 级指标
    更能反映模型在被试层面的泛化能力。

    聚合方式：
        1. 先按 trial 聚合 segment。
        2. 再按 subject 聚合 trial。
    """
    trial_metrics = compute_trial_level_metrics(
        segment_records,
        threshold=threshold,
        vote_method="hard" if vote_method in ["trial_hard", "hard"] else "prob",
    )

    subject_dict = defaultdict(list)
    for key, prob, pred, label in zip(
        trial_metrics["trial_keys"],
        trial_metrics["trial_probs"],
        trial_metrics["trial_preds"],
        trial_metrics["trial_labels"],
    ):
        subject_id, _trial_id = key
        subject_dict[int(subject_id)].append({
            "prob": float(prob),
            "pred": int(pred),
            "label": int(label),
        })

    subject_keys = []
    subject_probs = []
    subject_preds = []
    subject_labels = []

    for subject_id, records in subject_dict.items():
        labels = [r["label"] for r in records]
        preds = [r["pred"] for r in records]
        probs = [r["prob"] for r in records]

        label = int(np.bincount(labels, minlength=2).argmax())
        mean_prob = float(np.mean(probs))

        if vote_method in ["prob", "trial_prob"]:
            pred = 1 if mean_prob >= threshold else 0
        else:
            counts = np.bincount(preds, minlength=2)
            if counts[1] > counts[0]:
                pred = 1
            elif counts[0] > counts[1]:
                pred = 0
            else:
                pred = 1 if mean_prob >= threshold else 0

        subject_keys.append(subject_id)
        subject_probs.append(mean_prob)
        subject_preds.append(pred)
        subject_labels.append(label)

    if len(subject_labels) == 0:
        return {
            "subject_acc": 0.0,
            "subject_macro_f1": 0.0,
            "subject_confusion_matrix": np.zeros((2, 2), dtype=int),
            "subject_keys": [],
            "subject_probs": [],
            "subject_preds": [],
            "subject_labels": [],
        }

    subject_acc = accuracy_score(subject_labels, subject_preds)
    subject_macro_f1 = f1_score(
        subject_labels,
        subject_preds,
        average="macro",
        labels=[0, 1],
        zero_division=0,
    )
    subject_cm = confusion_matrix(subject_labels, subject_preds, labels=[0, 1])

    return {
        "subject_acc": subject_acc,
        "subject_macro_f1": subject_macro_f1,
        "subject_confusion_matrix": subject_cm,
        "subject_keys": subject_keys,
        "subject_probs": subject_probs,
        "subject_preds": subject_preds,
        "subject_labels": subject_labels,
    }


def to_serializable(obj):
    """
    将 numpy / torch / tuple 等对象转成 json 可保存的 Python 原生类型。
    """
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


def flatten_metrics_for_csv(prefix: str, metrics: dict) -> dict:
    """
    把指标展开成适合写入 CSV 的一维字典。
    """
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


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(obj), f, ensure_ascii=False, indent=2)


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
    """
    criteria 示例：
    [("trial_macro_f1", "max"), ("trial_acc", "max"), ("loss", "min")]
    """
    key = []
    for metric_name, mode in criteria:
        v = metric_value(metrics, metric_name, default=None)
        if v is None:
            v = -1e18 if mode == "max" else 1e18
        key.append(v if mode == "max" else -v)
    return tuple(key)


def is_better_by_criteria(current_metrics: dict, best_metrics: dict, criteria, eps: float = 1e-12) -> bool:
    """
    并列比较条件。
    """
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
    """
    基于多指标 criteria 的早停器。
    """

    def __init__(
        self,
        criteria,
        patience: int = 20,
        warmup: int = 10,
        min_delta: float = 1e-6,
    ):
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
# Checkpoint save
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
    """
    同时保存：
    1. pt：模型参数 + optimizer + 完整指标
    2. json：完整指标
    3. csv：标量指标摘要
    """
    os.makedirs(save_dir, exist_ok=True)

    model_file = "diag_best.pt" if best_name == "combined" else f"best_{best_name}_fold{fold}.pt"
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
    dataset,
    rand,
    n_splits: int = 5,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    lambda_diag: float = 1.0,
    lambda_graph: float = 0.1,
    lambda_domain: float = 0.0,
    grl_domain: float = 0.1,
    lambda_con: float = 0.05,
    device: str = "cuda",
    run_seed: int = None,
    deterministic: bool = False,
    early_stop_patience: int = 20,
    early_stop_warmup: int = 15,
    early_stop_min_delta: float = 1e-6,
    early_stop_track: str = "combined",
    # 类中心损失
    lambda_center: float = 0.0,
    center_tau: float = 0.1,
    center_dim: int = 64,
    center_warmup_epochs: int = 10,
):
    """
    双任务跨被试训练（TwoBranchModel）。

    L = L_4cls(z_conv, label4) + λ_diag × L_2cls(z_diag, emotion_binary)
        + λ_domain × L_domain + λ_center × L_center
    """
    os.makedirs(save_dir, exist_ok=True)

    if run_seed is None:
        run_seed = config.make_run_seed(rand, fold)

    # 这里主要控制 DataLoader、sampler、训练过程随机性；
    # 模型初始化前的种子需要在 __main__ 里创建 model 之前设置。
    set_global_seed(run_seed, deterministic=deterministic)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(run_seed)

    print(f"[Seed] rand={rand}, fold={fold}, run_seed={run_seed}, deterministic={deterministic}")

    # -------- subject-level split --------
    # ── 统一交叉验证划分：三个模型共用同一套 StratifiedGroupKFold ──
    split = get_unified_subject_split(
        index_csv=index_csv,
        fold=fold,
        n_splits=n_splits,
        seed=rand,
    )
    train_subjects = split["train_all"]
    val_subjects = split["val_all"]

    print(f"Fold {fold}")
    print("train subjects:", len(train_subjects), train_subjects)
    print("val subjects:", len(val_subjects), val_subjects)

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

    # -------- sampler --------
    train_batch_sampler = MultiSubjectBatchSampler(
        sample_subject_ids=train_dataset.sample_subject_ids,
        batch_size=batch_size,
        subjects_per_batch=4,
        drop_last=True,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collate_fn,
        num_workers=4,
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

    # -------- loss --------
    # 四分类情绪 loss（带类别权重）
    class_weights_4cls = build_class_weights(index_csv, train_subjects).to(device)
    print(f"[Class weights 4cls] weights={class_weights_4cls.detach().cpu().numpy().tolist()}")
    criterion_4cls = torch.nn.CrossEntropyLoss(
        weight=class_weights_4cls,
        label_smoothing=0.05,
    )

    # 二分类情绪 loss（带类别权重：0=中性, 1=正性）
    class_weights_2cls = build_emotion_class_weights(index_csv, train_subjects).to(device)
    print(f"[Class weights 2cls (emotion)] weights={class_weights_2cls.detach().cpu().numpy().tolist()}")
    criterion_2cls = torch.nn.CrossEntropyLoss(
        weight=class_weights_2cls,
        label_smoothing=0.05,
    )

    dom_criterion = torch.nn.CrossEntropyLoss()
    center_criterion = ClassCenterContrastLoss(
        in_dim=model.diag_dim if hasattr(model, "diag_dim") else 128,
        proj_dim=64,
        num_classes=2,
        tau=center_tau,
        dropout=0.2,
        label_smoothing=0.0,
    ).to(device)

    # -------- optimizer --------
    optim_params = list(model.parameters())

    if center_criterion is not None and lambda_center > 0:
        optim_params += list(center_criterion.parameters())

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    config_dict = {
        "index_csv": index_csv,
        "fold": fold,
        "rand": rand,
        "run_seed": run_seed,
        "deterministic": deterministic,
        "n_splits": n_splits,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "lambda_diag": lambda_diag,
        "lambda_graph": lambda_graph,
        "lambda_domain": lambda_domain,
        "grl_domain": grl_domain,
        "lambda_con": lambda_con,
        "dataset": dataset,
        "early_stop_patience": early_stop_patience,
        "early_stop_warmup": early_stop_warmup,
        "early_stop_min_delta": early_stop_min_delta,
        "early_stop_track": early_stop_track,
        "lambda_center": lambda_center,
        "center_tau": center_tau,
        "center_dim": center_dim,
        "center_warmup_epochs": center_warmup_epochs,
        "num_subjects": model.domain_head.num_subjects if hasattr(model, "domain_head") else 48,
    }

    # -------- 并列保存多个最优模型 --------
    best_trackers = {
        "combined": {
            "criteria": [
                ("subject_macro_f1", "max"),
                ("subject_acc", "max"),
                ("trial_macro_f1", "max"),
                ("trial_acc", "max"),
                ("emotion_macro_f1", "max"),
                ("emotion_acc", "max"),
                ("macro_f1", "max"),
                ("acc", "max"),
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
                ("subject_macro_f1", "max"),
                ("subject_acc", "max"),
                ("emotion_macro_f1", "max"),
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
                ("subject_acc", "max"),
                ("subject_macro_f1", "max"),
                ("emotion_acc", "max"),
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
        "four_class_f1": {
            "criteria": [
                ("macro_f1", "max"),
                ("acc", "max"),
                ("emotion_macro_f1", "max"),
                ("loss", "min"),
            ],
            "best_metrics": None,
            "best_epoch": None,
            "checkpoint_path": None,
        },
    }

    if early_stop_track not in best_trackers:
        raise ValueError(
            f"early_stop_track={early_stop_track} 不在 best_trackers 中，"
            f"可选值为: {list(best_trackers.keys())}"
        )

    early_stopper = CriteriaEarlyStopping(
        criteria=best_trackers[early_stop_track]["criteria"],
        patience=early_stop_patience,
        warmup=early_stop_warmup,
        min_delta=early_stop_min_delta,
    )

    print(
        f"[EarlyStopping] track={early_stop_track}, "
        f"criteria={format_criteria(early_stopper.criteria)}, "
        f"patience={early_stop_patience}, "
        f"warmup={early_stop_warmup}, "
        f"min_delta={early_stop_min_delta}, "
        f"lambda_center={lambda_center}, "
        f"center_warmup_epochs={center_warmup_epochs}"
    )

    history = []

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")

        train_metrics = train_one_epoch_two_branch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion_4cls=criterion_4cls,
            criterion_2cls=criterion_2cls,
            dom_criterion=dom_criterion,
            device=device,
            lambda_diag=lambda_diag,
            lambda_domain=lambda_domain,
            grl_domain=grl_domain,
            center_criterion=center_criterion,
            lambda_center=lambda_center,
            center_warmup_epochs=center_warmup_epochs,
            current_epoch=epoch,
        )

        val_metrics = validate_one_epoch_two_branch(
            model=model,
            loader=val_loader,
            criterion_4cls=criterion_4cls,
            criterion_2cls=criterion_2cls,
            device=device,
            lambda_diag=lambda_diag,
            lambda_graph=lambda_graph,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Train] loss={train_metrics['loss']:.4f} "
            f"L4={train_metrics['loss_4cls']:.4f} "
            f"L2={train_metrics['loss_2cls']:.4f} "
            f"dom={train_metrics['loss_dom']:.4f} "
            f"ctr={train_metrics['loss_center']:.4f} "
            f"acc4={train_metrics['acc_4cls']:.4f} "
            f"acc2={train_metrics['acc_2cls']:.4f}"
        )
        print(
            f"[Val]   loss={val_metrics['loss']:.4f} "
            f"L4={val_metrics['ce_loss_4cls']:.4f} "
            f"L2={val_metrics['ce_loss_2cls']:.4f} "
            f"acc4={val_metrics['acc_4cls']:.4f} "
            f"f1_4={val_metrics['macro_f1_4cls']:.4f} "
            f"seg_emo_acc={val_metrics['emotion_acc']:.4f} "
            f"seg_emo_f1={val_metrics['emotion_macro_f1']:.4f} "
            f"trial_emo_acc={val_metrics['trial_acc']:.4f} "
            f"trial_emo_f1={val_metrics['trial_macro_f1']:.4f} "
            f"subject_emo_acc={val_metrics['subject_acc']:.4f} "
            f"subject_emo_f1={val_metrics['subject_macro_f1']:.4f}"
        )

        hist_row = {
            "epoch": epoch,
            "lr": current_lr,
        }
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

        # 按多个标准并列保存最优模型
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
                    config=config_dict,
                )

                tracker["best_metrics"] = copy.deepcopy(val_metrics)
                tracker["best_train_metrics"] = copy.deepcopy(train_metrics)
                tracker["best_epoch"] = epoch
                tracker["checkpoint_path"] = ckpt_path
                tracker["metrics_json_path"] = json_path
                tracker["summary_csv_path"] = csv_path

                print(
                    f"保存 best_{best_name}: epoch={epoch}, "
                    f"criteria={format_criteria(criteria)}, "
                    f"path={ckpt_path}"
                )

        # -------- early stopping --------
        early_improved, should_stop = early_stopper.step(val_metrics, epoch)
        print(
            f"[EarlyStopping] improved={early_improved}, "
            f"best_epoch={early_stopper.best_epoch}, "
            f"bad_epochs={early_stopper.num_bad_epochs}/{early_stop_patience}"
        )

        if should_stop:
            print(
                f"\n触发早停: epoch={epoch}, "
                f"连续 {early_stopper.num_bad_epochs} 个 epoch "
                f"在 {early_stop_track} 标准上没有提升。"
            )
            early_stop_json = os.path.join(save_dir, f"early_stop_fold{fold}.json")
            save_json(early_stop_json, {
                "stopped": True,
                "stop_epoch": epoch,
                "early_stop_track": early_stop_track,
                "early_stopper": early_stopper.state_dict(),
                "config": config_dict,
            })
            print(f"早停信息已保存到: {early_stop_json}")
            break

    early_stop_state_json = os.path.join(save_dir, f"early_stop_state_fold{fold}.json")
    save_json(early_stop_state_json, {
        "early_stop_track": early_stop_track,
        "early_stopper": early_stopper.state_dict(),
        "config": config_dict,
    })
    print(f"Early stopping 状态已保存到: {early_stop_state_json}")

    history_df = pd.DataFrame(history)
    history_csv = os.path.join(save_dir, f"history_fold{fold}.csv")
    history_df.to_csv(history_csv, index=False, encoding="utf-8-sig")
    print(f"训练历史已保存到: {history_csv}")

    # 保存该折所有 best 的汇总
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
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "history_csv": history_csv,
        "best_summary_csv": best_summary_csv,
        "best_models": best_trackers,
        "early_stop": early_stopper.state_dict(),
    }


# =========================================================
# Emotion class weights（从 label4 推算情绪二分类权重）
# =========================================================

def build_emotion_class_weights(index_csv: str, subject_ids):
    """
    从 label4 列推算情绪二分类 (0=中性, 1=正性) 的类别权重。

    label4: 0=DEP_neu, 1=DEP_pos, 2=HC_neu, 3=HC_pos
    emotion = label4 % 2: 0=中性, 1=正性
    """
    df = pd.read_csv(index_csv)
    subject_set = {str(s) for s in subject_ids}
    df = df[df["subject_id"].astype(str).isin(subject_set)].reset_index(drop=True)

    if "label4" not in df.columns:
        raise KeyError("index_csv 中缺少 label4 列。")

    emotion_labels = df["label4"].astype(int) % 2
    counts = emotion_labels.value_counts().sort_index()
    counts = counts.reindex([0, 1], fill_value=1)

    weights = counts.sum() / counts
    weights = weights / weights.mean()

    return torch.tensor(weights.values, dtype=torch.float32)


# =========================================================
# Test prediction
# =========================================================

@torch.no_grad()
def predict_test_trial(
    model,
    test_csv: str,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 4,
    root: Path = ROOT,
) -> pd.DataFrame:
    """
    用单个 model 对测试集做 trial 级情绪二分类预测。

    Returns
    -------
    pd.DataFrame with columns: [user_id, trial_id, prob_pos, pred_emotion]
    """
    from utils.data import expand_window_index

    model.eval()

    # 加载测试 CSV，展开窗口索引
    test_df = pd.read_csv(test_csv)
    id_col = "user_id" if "user_id" in test_df.columns else "subject_id"
    print(f"[predict_test] test samples: {len(test_df)}, id_col={id_col}")

    test_df = expand_window_index(test_df, root=root)
    print(f"[predict_test] expanded windows: {len(test_df)}")

    # 构造无标签 dataset
    dataset = EEGWindowDataset(
        test_df.reset_index(drop=True),
        label_col=None,
        root=root,
        return_de=True,
    )

    # 简易 dict collate：EEGWindowDataset 返回 dict，需要自定义 collate
    from torch.utils.data._utils.collate import default_collate

    def _dict_collate(batch_list):
        keys = list(batch_list[0].keys())
        return {k: default_collate([b[k] for b in batch_list]) for k in keys}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_dict_collate,
    )

    all_user_ids = []
    all_trial_ids = []
    all_probs = []

    for batch in tqdm(loader, desc="Test-Pred", leave=False):
        x = batch["x"].to(device)
        de_feat = batch["de_feat"].to(device)

        out = model(x, de_feat, lambda_dom=0.0, dataset_name="comp4")
        logits_2cls = out["logits_2cls"]                          # [B, 2]
        prob_pos = torch.softmax(logits_2cls, dim=1)[:, 1]        # P(正性情绪)

        user_ids = batch[id_col].detach().cpu().numpy()
        trial_ids = batch["trial_id"].detach().cpu().numpy()
        probs = prob_pos.detach().cpu().numpy()

        all_user_ids.extend(user_ids.tolist())
        all_trial_ids.extend(trial_ids.tolist())
        all_probs.extend(probs.tolist())

    # 聚合到 trial 级（同 trial 内窗口取均值）
    pred_df = pd.DataFrame({
        "user_id": all_user_ids,
        "trial_id": all_trial_ids,
        "prob_window": all_probs,
    })

    trial_pred = (
        pred_df.groupby(["user_id", "trial_id"], as_index=False)
        .agg(prob_pos=("prob_window", "mean"), n_windows=("prob_window", "count"))
    )
    trial_pred["pred_emotion"] = (trial_pred["prob_pos"] >= 0.5).astype(int)

    print(f"[predict_test] trial predictions: {len(trial_pred)}")
    return trial_pred


def ensemble_test_predictions(
    model_paths: list[Path],
    test_csv: str,
    device: torch.device,
    save_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    root: Path = ROOT,
) -> pd.DataFrame:
    """
    加载所有 fold 模型，逐模型推理，soft voting 集成。

    Returns
    -------
    pd.DataFrame with columns: [user_id, trial_id, prob_pos_ensemble, pred_emotion, n_models]
    """
    all_trial_probs = []

    for i, ckpt_path in enumerate(model_paths):
        print(f"\n[ensemble] 模型 {i + 1}/{len(model_paths)}: {ckpt_path}")

        # 从 checkpoint 中读取 num_subjects
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config_ckpt = ckpt.get("config", {})
        num_subjects = int(config_ckpt.get("num_subjects", 48))

        model = TwoBranchModel(
            sfreq=250.0,
            prior_matrix=None,
            topk=6,
            dropout=0.2,
            num_subjects=num_subjects,
            num_classes_4=4,
            num_classes_2=2,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        trial_pred = predict_test_trial(
            model=model,
            test_csv=test_csv,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            root=root,
        )
        trial_pred = trial_pred[["user_id", "trial_id", "prob_pos"]].copy()
        trial_pred.rename(columns={"prob_pos": f"prob_fold_{i}"}, inplace=True)
        all_trial_probs.append(trial_pred)

    # ── Soft voting：所有 fold 概率取均值 ──
    merged = all_trial_probs[0]
    for df in all_trial_probs[1:]:
        merged = merged.merge(df, on=["user_id", "trial_id"], how="inner")

    prob_cols = [c for c in merged.columns if c.startswith("prob_fold_")]
    merged["prob_pos_ensemble"] = merged[prob_cols].mean(axis=1)
    merged["n_models"] = len(prob_cols)
    merged["pred_emotion"] = (merged["prob_pos_ensemble"] >= 0.5).astype(int)

    # ── 保存 ──
    os.makedirs(save_dir, exist_ok=True)
    probs_path = os.path.join(save_dir, "test_ensemble_probs.csv")
    merged.to_csv(probs_path, index=False, encoding="utf-8-sig")
    print(f"\n[ensemble] 集成概率已保存: {probs_path}")

    # 提交格式：user_id, trial_id, Emotion_label
    submission = merged[["user_id", "trial_id", "pred_emotion"]].copy()
    submission.rename(columns={"pred_emotion": "Emotion_label"}, inplace=True)
    submission_path = os.path.join(save_dir, "submission_test_ensemble.csv")
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")
    print(f"[ensemble] 提交文件已保存: {submission_path}")

    # 打印分布统计
    n_total = len(submission)
    n_pos = int(submission["Emotion_label"].sum())
    print(f"[ensemble] 预测分布: {n_pos}/{n_total} 正性 ({100 * n_pos / max(n_total, 1):.1f}%)")

    return merged


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    device = "cuda:0"
    version = "two_branch"
    model_params_root = "model_params"
    os.makedirs(version, exist_ok=True)
    os.makedirs(model_params_root, exist_ok=True)

    all_fold_rows = []
    all_best_rows_long = []
    all_ckpt_paths: list[Path] = []
    combined_4cls_confusions = []
    combined_segment_emo_confusions = []
    combined_trial_emo_confusions = []
    combined_subject_emo_confusions = []

    # 多随机种子 × 五折交叉验证
    for repeat, rand in enumerate(config.DIAG_REPEAT_SEEDS):
        print(f"\n{'#' * 70}")
        print(f"开始 random seed = {rand} 的 {5} 折交叉验证")
        print(f"{'#' * 70}")

        seed_combined_rows = []
        n_splits = 5

        for fold in range(n_splits):
            print(f"\n{'=' * 50}")
            print(f"开始训练第 {fold + 1}/{n_splits} 折 | seed={rand}")
            print(f"{'=' * 50}")

            # 关键：模型初始化前设置 seed
            run_seed = config.make_run_seed(rand, fold)
            set_global_seed(run_seed, deterministic=False)
            print(f"[Main Seed] rand={rand}, fold={fold}, run_seed={run_seed}")

            model = TwoBranchModel(
                sfreq=250.0,
                prior_matrix=None,
                topk=6,
                dropout=0.2,
                num_subjects=48,
                num_classes_4=4,
                num_classes_2=2,
            )

            result = train_competition_cross_subject(
                model=model,
                index_csv="com_index_sub_2s.csv",
                fold=fold,
                save_dir=os.path.join(model_params_root, f"two_branch_reapt{repeat}_fold{fold}"),
                dataset="com",
                n_splits=n_splits,
                epochs=100,
                batch_size=32,
                lr=1e-4,
                rand=rand,
                weight_decay=1e-4,
                num_workers=4,
                lambda_diag=0.5,
                lambda_graph=0.0,
                lambda_domain=0.0,
                grl_domain=0.0,
                lambda_con=0.0,
                device=device,
                run_seed=run_seed,
                deterministic=False,
                early_stop_track="combined",
                early_stop_patience=25,
                early_stop_warmup=15,
                early_stop_min_delta=1e-6,
            )

            combined_tracker = result["best_models"]["combined"]
            combined_metrics = combined_tracker["best_metrics"]

            fold_row = {
                "rand": rand,
                "fold": fold,
                "run_seed": run_seed,
                "best_name": "combined",
                "best_epoch": combined_tracker["best_epoch"],
                "checkpoint_path": combined_tracker["checkpoint_path"],
            }
            fold_row.update(flatten_metrics_for_csv("val", combined_metrics))
            all_fold_rows.append(fold_row)
            seed_combined_rows.append(fold_row)

            if combined_tracker["checkpoint_path"]:
                all_ckpt_paths.append(Path(combined_tracker["checkpoint_path"]))

            for best_name, tracker in result["best_models"].items():
                row = {
                    "rand": rand,
                    "fold": fold,
                    "run_seed": run_seed,
                    "best_name": best_name,
                    "best_epoch": tracker["best_epoch"],
                    "criteria": format_criteria(tracker["criteria"]),
                    "checkpoint_path": tracker["checkpoint_path"],
                }
                if tracker["best_metrics"] is not None:
                    row.update(flatten_metrics_for_csv("val", tracker["best_metrics"]))
                all_best_rows_long.append(row)

            if combined_metrics is not None:
                combined_4cls_confusions.append(combined_metrics["confusion_matrix"])
                combined_segment_emo_confusions.append(combined_metrics["emotion_confusion_matrix"])
                combined_trial_emo_confusions.append(combined_metrics["trial_confusion_matrix"])
                combined_subject_emo_confusions.append(combined_metrics["subject_confusion_matrix"])

            print(f"第 {fold + 1} 折 combined 最优结果:")
            print(f"best_epoch = {combined_tracker['best_epoch']}")
            print(f"val_acc_4cls = {combined_metrics['acc_4cls']:.4f}")
            print(f"val_macro_f1_4cls = {combined_metrics['macro_f1_4cls']:.4f}")
            print(f"val_segment_emo_acc = {combined_metrics['emotion_acc']:.4f}")
            print(f"val_segment_emo_f1 = {combined_metrics['emotion_macro_f1']:.4f}")
            print(f"val_trial_emo_acc = {combined_metrics['trial_acc']:.4f}")
            print(f"val_trial_emo_f1 = {combined_metrics['trial_macro_f1']:.4f}")
            print(f"val_subject_emo_acc = {combined_metrics['subject_acc']:.4f}")
            print(f"val_subject_emo_f1 = {combined_metrics['subject_macro_f1']:.4f}")

        seed_df = pd.DataFrame(seed_combined_rows)
        seed_csv = os.path.join(version, f"seed{rand}_combined_5fold_results.csv")
        seed_df.to_csv(seed_csv, index=False, encoding="utf-8-sig")

        metric_cols = [
            "val_loss",
            "val_acc_4cls",
            "val_macro_f1_4cls",
            "val_acc",
            "val_macro_f1",
            "val_emotion_acc",
            "val_emotion_macro_f1",
            "val_trial_acc",
            "val_trial_macro_f1",
            "val_subject_acc",
            "val_subject_macro_f1",
            "val_subject_emotion_acc",
            "val_subject_emotion_macro_f1",
        ]
        summary_lines = []
        print(f"\n{'=' * 50}")
        print(f"Random seed {rand} 的 5 折 combined 平均结果:")
        print(f"{'=' * 50}")
        for col in metric_cols:
            if col in seed_df.columns:
                mean_v = seed_df[col].mean()
                std_v = seed_df[col].std(ddof=0)
                print(f"{col}: {mean_v:.4f} ± {std_v:.4f}")
                summary_lines.append(f"{col}: {mean_v:.4f} ± {std_v:.4f}\n")

        with open(os.path.join(version, f"seed{rand}_combined_5fold_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"Random seed {rand} combined 5折交叉验证结果\n")
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

    np.save(os.path.join(version, "combined_4cls_confusions.npy"), np.array(combined_4cls_confusions, dtype=object))
    np.save(os.path.join(version, "combined_segment_emo_confusions.npy"), np.array(combined_segment_emo_confusions, dtype=object))
    np.save(os.path.join(version, "combined_trial_emo_confusions.npy"), np.array(combined_trial_emo_confusions, dtype=object))
    np.save(os.path.join(version, "combined_subject_emo_confusions.npy"), np.array(combined_subject_emo_confusions, dtype=object))

    if len(all_fold_df) > 0:
        metric_cols = [
            "val_loss",
            "val_acc_4cls",
            "val_macro_f1_4cls",
            "val_acc",
            "val_macro_f1",
            "val_emotion_acc",
            "val_emotion_macro_f1",
            "val_trial_acc",
            "val_trial_macro_f1",
            "val_subject_acc",
            "val_subject_macro_f1",
            "val_subject_emotion_acc",
            "val_subject_emotion_macro_f1",
        ]

        print(f"\n{'=' * 50}")
        print("所有 seed × fold 的 combined 总体结果:")
        print(f"{'=' * 50}")

        overall_summary = []
        for col in metric_cols:
            if col in all_fold_df.columns:
                mean_v = all_fold_df[col].mean()
                std_v = all_fold_df[col].std(ddof=0)
                print(f"{col}: {mean_v:.4f} ± {std_v:.4f}")
                overall_summary.append({
                    "metric": col,
                    "mean": mean_v,
                    "std": std_v,
                })

        overall_summary_csv = os.path.join(version, "overall_combined_summary.csv")
        pd.DataFrame(overall_summary).to_csv(overall_summary_csv, index=False, encoding="utf-8-sig")
        print(f"总体统计已保存到: {overall_summary_csv}")

        with open(os.path.join(version, "overall_combined_summary.txt"), "w", encoding="utf-8") as f:
            f.write("所有 seed × fold 的 combined 总体结果\n")
            for item in overall_summary:
                f.write(f"{item['metric']}: {item['mean']:.4f} ± {item['std']:.4f}\n")
            f.write(f"\n全部详细指标 CSV: {all_fold_csv}\n")
            f.write(f"全部并列 best 指标 CSV: {all_best_csv}\n")

        # 平均 segment 级情绪二分类混淆矩阵（四分类头视角: acc=四分类）
        valid_confs = [np.array(c, dtype=np.float32) for c in combined_4cls_confusions if c is not None]
        if len(valid_confs) > 0:
            conf_stack = np.stack(valid_confs, axis=0)
            avg_conf = conf_stack.mean(axis=0)
            row_sum = avg_conf.sum(axis=1, keepdims=True)
            avg_conf_norm = avg_conf / np.maximum(row_sum, 1e-12)

            import matplotlib.pyplot as plt

            labels_4cls = ["DEP_neu", "DEP_pos", "HC_neu", "HC_pos"]
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(avg_conf_norm, interpolation="nearest", cmap="Blues")
            ax.set_xticks(np.arange(len(labels_4cls)))
            ax.set_yticks(np.arange(len(labels_4cls)))
            ax.set_xticklabels(labels_4cls, rotation=45, ha="right")
            ax.set_yticklabels(labels_4cls)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Average 4-Class Emotion Confusion Matrix")

            thresh = avg_conf_norm.max() / 2.0 if avg_conf_norm.max() != 0 else 0.5
            for i in range(avg_conf_norm.shape[0]):
                for j in range(avg_conf_norm.shape[1]):
                    val = avg_conf_norm[i, j]
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
            fig_path = os.path.join(version, "avg_4cls_confusion.png")
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            print(f"平均四分类混淆矩阵图已保存到: {fig_path}")

        # 平均 trial 级情绪二分类混淆矩阵
        valid_trial_confs = [np.array(c, dtype=np.float32) for c in combined_trial_emo_confusions if c is not None]
        if len(valid_trial_confs) > 0:
            trial_conf_stack = np.stack(valid_trial_confs, axis=0)
            avg_trial_conf = trial_conf_stack.mean(axis=0)
            row_sum = avg_trial_conf.sum(axis=1, keepdims=True)
            avg_trial_conf_norm = avg_trial_conf / np.maximum(row_sum, 1e-12)

            import matplotlib.pyplot as plt

            labels_emo = ["Neutral", "Positive"]
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(avg_trial_conf_norm, interpolation="nearest", cmap="Blues")
            ax.set_xticks(np.arange(len(labels_emo)))
            ax.set_yticks(np.arange(len(labels_emo)))
            ax.set_xticklabels(labels_emo)
            ax.set_yticklabels(labels_emo)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Average Trial-level Emotion Confusion Matrix")

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
            print(f"平均 trial 级情绪二分类混淆矩阵图已保存到: {fig_path}")


        # 平均 subject 级情绪二分类混淆矩阵
        valid_subject_confs = [np.array(c, dtype=np.float32) for c in combined_subject_emo_confusions if c is not None]
        if len(valid_subject_confs) > 0:
            subject_conf_stack = np.stack(valid_subject_confs, axis=0)
            avg_subject_conf = subject_conf_stack.mean(axis=0)
            row_sum = avg_subject_conf.sum(axis=1, keepdims=True)
            avg_subject_conf_norm = avg_subject_conf / np.maximum(row_sum, 1e-12)

            import matplotlib.pyplot as plt

            labels_emo = ["Neutral", "Positive"]
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(avg_subject_conf_norm, interpolation="nearest", cmap="Blues")
            ax.set_xticks(np.arange(len(labels_emo)))
            ax.set_yticks(np.arange(len(labels_emo)))
            ax.set_xticklabels(labels_emo)
            ax.set_yticklabels(labels_emo)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Average Subject-level Emotion Confusion Matrix")

            thresh = avg_subject_conf_norm.max() / 2.0 if avg_subject_conf_norm.max() != 0 else 0.5
            for i in range(avg_subject_conf_norm.shape[0]):
                for j in range(avg_subject_conf_norm.shape[1]):
                    val = avg_subject_conf_norm[i, j]
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
            fig_path = os.path.join(version, "avg_subject_emotion_confusion.png")
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            print(f"平均 subject 级情绪二分类混淆矩阵图已保存到: {fig_path}")
    else:
        print("没有收集到任何 fold 结果，无法计算总体统计与混淆矩阵。")

    # =========================================================
    # Test 集成预测
    # =========================================================
    if len(all_ckpt_paths) > 0:
        test_csv = os.path.join(ROOT, "com_test_trial_index_2s.csv")
        if not os.path.exists(test_csv):
            test_csv = os.path.join(ROOT, "com_test_trial_index_10s.csv")
        if os.path.exists(test_csv):
            print(f"\n{'=' * 50}")
            print(f"Test 集成预测: {len(all_ckpt_paths)} 个模型")
            print(f"Test CSV: {test_csv}")
            print(f"{'=' * 50}")

            ensemble_test_predictions(
                model_paths=all_ckpt_paths,
                test_csv=test_csv,
                device=torch.device(device),
                save_dir=version,
                batch_size=32,
                num_workers=4,
                root=ROOT,
            )
        else:
            print(f"\n[WARNING] 测试 CSV 不存在: {test_csv}，跳过 test 预测。")
    else:
        print("没有收集到任何模型 checkpoint，跳过 test 预测。")

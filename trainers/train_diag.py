# -*- coding: utf-8 -*-
"""
EEG 抑郁症/正常人跨被试二分类训练脚本。

适用目标：
    diagnosis_label 二分类：0/1 分别代表你的数据集中定义的两个诊断类别。
    请确认 com_index_sub_2s.csv 中 diagnosis_label 的编码含义，常见是：
        0 = HC/正常人，1 = DEP/抑郁症
    如果你的编码相反，指标含义也相反，但训练逻辑不受影响。

主要修正：
1. 当 nclass=2 时，训练标签统一使用 batch["diagnosis_label"]。
2. 二分类也使用 diagnosis_label 的类别权重，缓解 DEP/HC 样本不均衡。
3. lambda_graph=0 或模型不返回 adj_dense 时，不再强制访问 out["adj_dense"]。
4. trial 级指标改为诊断二分类指标；额外增加 subject 级诊断指标。
5. 打印、保存和画图命名改为 diagnosis/diag，避免与 emotion 或 4-class 混淆。
6. 保留 nclass=4 分支兼容性，但本脚本默认 nclass=2。
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
from models.diagnosis_model import EmotionPretrainModel


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


def make_run_seed(base_seed: int, fold: int, extra: int = 0) -> int:
    """
    为每个 seed × fold 构造一个独立但可复现的 run_seed。
    """
    return int(base_seed) * 1000 + int(fold) + int(extra)


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def validate_one_epoch_comp4(
    model,
    loader,
    criterion_ce,
    contrast_criterion,
    device,
    nclass: int = 4,
    lambda_graph: float = 0.1,
):
    """
    验证 / 测试一轮，并返回尽可能完整的指标。
    """
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_graph = 0.0
    total_con = 0.0

    total_correct = 0
    total_num = 0
    all_preds = []
    all_labels = []

    total_emo_correct = 0
    all_emo_preds = []
    all_emo_labels = []
    segment_records = []

    pbar = tqdm(loader, desc="Val", leave=False)

    for batch in pbar:
        x = batch["x"].to(device)

        if nclass == 4:
            y = batch["label4"].to(device).long()
            y_emo = (y % 2).long()
        else:
            y = batch["diagnosis_label"].to(device).long()
            y_emo = y

        subject_id = batch["subject_id"].to(device).long()
        de_feat = batch["de_feat"].to(device)

        out = model(x, de_feat, 0, dataset_name="comp4")
        logits = out["logits"]
        adj_dense = out.get("adj_dense", None)
        contrast_feat = out.get("node_contrast_feat", None)

        if contrast_feat is not None:
            loss_con = contrast_criterion(
                features=contrast_feat,
                labels=y,
                subjects=subject_id,
            )
        else:
            loss_con = logits.new_tensor(0.0)

        loss_ce = criterion_ce(logits, y)
        if lambda_graph > 0 and adj_dense is not None:
            loss_graph = intra_class_graph_loss(adj_dense, y)
        else:
            loss_graph = logits.new_tensor(0.0)
        loss = loss_ce + lambda_graph * loss_graph

        pred = logits.argmax(dim=1)
        pred_emo = (pred % 2).long() if nclass == 4 else pred

        prob = torch.softmax(logits, dim=1)
        if nclass == 4:
            prob_pos = prob[:, 1] + prob[:, 3]
        else:
            prob_pos = prob[:, 1]

        subject_ids_cpu = batch["subject_id"]
        trial_ids_cpu = batch["trial_id"]

        for i in range(x.size(0)):
            segment_records.append({
                "subject_id": int(subject_ids_cpu[i]),
                "trial_id": int(trial_ids_cpu[i]),
                "prob_pos": float(prob_pos[i].detach().cpu()),  # 兼容原二分类 trial 聚合；nclass=2 时表示类别1概率
                "prob_diag1": float(prob_pos[i].detach().cpu()),
                "pred_emo": int(pred_emo[i].detach().cpu()),
                "pred_diag": int(pred_emo[i].detach().cpu()),
                "label_emo": int(y_emo[i].detach().cpu()),
                "label_diag": int(y_emo[i].detach().cpu()),
                "pred_cls": int(pred[i].detach().cpu()),
                "label_cls": int(y[i].detach().cpu()),
                "pred4": int(pred[i].detach().cpu()),  # 兼容旧字段；nclass=2 时不是四分类
                "label4": int(y[i].detach().cpu()),  # 兼容旧字段；nclass=2 时不是四分类
            })

        correct = (pred == y).sum().item()
        emo_correct = (pred_emo == y_emo).sum().item()
        bsz = x.size(0)

        total_loss += loss.item() * bsz
        total_ce += loss_ce.item() * bsz
        total_graph += loss_graph.item() * bsz
        total_con += loss_con.item() * bsz
        total_correct += correct
        total_emo_correct += emo_correct
        total_num += bsz

        all_preds.extend(pred.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())
        all_emo_preds.extend(pred_emo.detach().cpu().numpy().tolist())
        all_emo_labels.extend(y_emo.detach().cpu().numpy().tolist())

        pbar.set_postfix({
            "loss": f"{total_loss / max(total_num, 1):.4f}",
            "ce": f"{total_ce / max(total_num, 1):.4f}",
            "graph": f"{total_graph / max(total_num, 1):.4f}",
            "contrast_loss": f"{total_con / max(total_num, 1):.4f}",
            "acc": f"{total_correct / max(total_num, 1):.4f}",
            "emo_acc": f"{total_emo_correct / max(total_num, 1):.4f}",
        })

    if total_num == 0:
        raise RuntimeError("验证集为空：total_num == 0，请检查 val_dataset / val_loader。")

    cls_labels = list(range(nclass))
    acc = total_correct / total_num
    macro_f1 = f1_score(all_labels, all_preds, average="macro", labels=cls_labels, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=cls_labels)

    p4, r4, f14, s4 = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=cls_labels,
        zero_division=0,
    )
    per_class_4 = {
        int(c): {
            "precision": float(p4[i]),
            "recall": float(r4[i]),
            "f1": float(f14[i]),
            "support": int(s4[i]),
        }
        for i, c in enumerate(cls_labels)
    }

    emotion_acc = total_emo_correct / total_num
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

    trial_metrics = compute_trial_level_metrics(
        segment_records,
        threshold=0.5,
        vote_method="hard",
    )
    subject_metrics = compute_subject_level_metrics(
        segment_records,
        threshold=0.5,
        vote_method="trial_hard",
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

    metrics = {
        "loss": total_loss / total_num,
        "ce_loss": total_ce / total_num,
        "graph_loss": total_graph / total_num,
        "contrast_loss": total_con / total_num,

        "acc": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_4": per_class_4,
        "segment_preds_4": all_preds,
        "segment_labels_4": all_labels,

        "emotion_acc": emotion_acc,
        "emotion_macro_f1": emotion_macro_f1,
        "emotion_confusion_matrix": emotion_cm,
        "per_class_emo": per_class_emo,
        "segment_preds_emo": all_emo_preds,
        "segment_labels_emo": all_emo_labels,

        "trial_acc": trial_metrics["trial_acc"],
        "trial_macro_f1": trial_metrics["trial_macro_f1"],
        "trial_confusion_matrix": trial_metrics["trial_confusion_matrix"],
        "trial_per_class_emo": trial_per_class_emo,
        "trial_keys": trial_metrics["trial_keys"],
        "trial_probs": trial_metrics["trial_probs"],
        "trial_preds": trial_metrics["trial_preds"],
        "trial_labels": trial_metrics["trial_labels"],

        "subject_acc": subject_metrics["subject_acc"],
        "subject_macro_f1": subject_metrics["subject_macro_f1"],
        "subject_confusion_matrix": subject_metrics["subject_confusion_matrix"],
        "subject_keys": subject_metrics["subject_keys"],
        "subject_probs": subject_metrics["subject_probs"],
        "subject_preds": subject_metrics["subject_preds"],
        "subject_labels": subject_metrics["subject_labels"],

        # 诊断二分类别名：nclass=2 时，这些与 acc/macro_f1、emotion_acc/emotion_macro_f1 等价。
        "diagnosis_acc": acc,
        "diagnosis_macro_f1": macro_f1,
        "diagnosis_confusion_matrix": cm,
        "trial_diagnosis_acc": trial_metrics["trial_acc"],
        "trial_diagnosis_macro_f1": trial_metrics["trial_macro_f1"],
        "subject_diagnosis_acc": subject_metrics["subject_acc"],
        "subject_diagnosis_macro_f1": subject_metrics["subject_macro_f1"],

        "segment_records": segment_records,
    }

    return metrics


# =========================================================
# Train one epoch
# =========================================================

def train_one_epoch_comp4(
    model,
    loader,
    contrast_criterion,
    optimizer,
    criterion_ce,
    dom_criterion,
    device,
    nclass: int = 4,
    lambdad_domain: float = 1.0,
    grl_domain: float = 0.1,
    lambda_graph: float = 0.6,
    lambda_con: float = 0.05,
):
    """
    单次训练。
    """
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_graph = 0.0
    total_correct = 0
    total_num = 0
    total_dom_loss = 0.0
    total_con_loss = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        x = batch["x"].to(device)
        if nclass == 4:
            y = batch["label4"].to(device).long()
        else:
            y = batch["diagnosis_label"].to(device).long()

        subject_id = batch["domain_id"].to(device).long()
        de_feat = batch["de_feat"].to(device)

        optimizer.zero_grad(set_to_none=True)

        out = model(x, de_feat, grl_domain, dataset_name="comp4")
        logits = out["logits"]
        adj_dense = out.get("adj_dense", None)
        domain_logits = out.get("domain_logits", None)
        contrast_feat = out.get("node_contrast_feat", None)

        if contrast_feat is not None:
            loss_con = contrast_criterion(
                features=contrast_feat,
                labels=y,
                subjects=subject_id,
            )
        else:
            loss_con = logits.new_tensor(0.0)

        if domain_logits is not None:
            loss_dom = dom_criterion(domain_logits, subject_id)
        else:
            loss_dom = logits.new_tensor(0.0)

        loss_ce = criterion_ce(logits, y)
        if lambda_graph > 0 and adj_dense is not None:
            loss_graph = intra_class_graph_loss(adj_dense, y)
        else:
            loss_graph = logits.new_tensor(0.0)

        loss = (
            loss_ce
            + lambda_graph * loss_graph
            + lambdad_domain * loss_dom
            + lambda_con * loss_con
        )

        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1)
        correct = (pred == y).sum().item()
        bsz = x.size(0)

        total_loss += loss.item() * bsz
        total_con_loss += loss_con.item() * bsz
        total_ce += loss_ce.item() * bsz
        total_graph += loss_graph.item() * bsz
        total_dom_loss += loss_dom.item() * bsz
        total_correct += correct
        total_num += bsz

        pbar.set_postfix({
            "loss": f"{total_loss / max(total_num, 1):.4f}",
            "ce": f"{total_ce / max(total_num, 1):.4f}",
            "graph": f"{total_graph / max(total_num, 1):.4f}",
            "loss_dom": f"{total_dom_loss / max(total_num, 1):.4f}",
            "contrast_loss": f"{total_con_loss / max(total_num, 1):.4f}",
            "acc": f"{total_correct / max(total_num, 1):.4f}",
        })

    if total_num == 0:
        raise RuntimeError("训练集为空：total_num == 0，请检查 train_dataset / train_loader。")

    metrics = {
        "loss": total_loss / total_num,
        "ce_loss": total_ce / total_num,
        "graph_loss": total_graph / total_num,
        "loss_dom": total_dom_loss / total_num,
        "loss_con": total_con_loss / total_num,
        "acc": total_correct / total_num,
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
    df = df[df["subject_id"].isin(subject_ids)].reset_index(drop=True)

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
    df = df[df["subject_id"].isin(subject_ids)].reset_index(drop=True)

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


# =========================================================
# Metrics utilities
# =========================================================

def compute_trial_level_metrics(segment_records, threshold=0.5, vote_method="hard"):
    """
    根据 segment 级预测结果聚合成 trial 级二分类结果。

    对 nclass=2 的诊断任务：
        pred_emo / label_emo 实际表示 pred_diag / label_diag。
        prob_pos 实际表示类别 1 的概率，比如 P(DEP) 或 P(HC)，取决于你的 diagnosis_label 编码。
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
        probs = [float(r["prob_pos"]) for r in records]
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
    将 segment 预测进一步聚合到 subject 级。

    诊断二分类本质上是被试级任务，因此 subject_acc / subject_macro_f1
    比 segment_acc 或 trial_acc 更接近最终任务目标。

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
    nclass: int = 4,
    n_splits: int = 5,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    lambda_graph: float = 0.1,
    lambda_dom: float = 1.0,
    grl_domain: float = 0.1,
    lambda_con: float = 0.05,
    device: str = "cuda",
    run_seed: int = None,
    deterministic: bool = True,
    early_stop_patience: int = 20,
    early_stop_warmup: int = 15,
    early_stop_min_delta: float = 1e-6,
    early_stop_track: str = "combined",
):
    """
    跨被试训练。
    """
    os.makedirs(save_dir, exist_ok=True)

    if run_seed is None:
        run_seed = make_run_seed(rand, fold)

    # 这里主要控制 DataLoader、sampler、训练过程随机性；
    # 模型初始化前的种子需要在 __main__ 里创建 model 之前设置。
    set_global_seed(run_seed, deterministic=deterministic)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(run_seed)

    print(f"[Seed] rand={rand}, fold={fold}, run_seed={run_seed}, deterministic={deterministic}")

    # -------- subject-level split --------
    train_subjects, val_subjects = get_competition_subject_split(
        index_csv=index_csv,
        fold=fold,
        n_splits=n_splits,
        seed=rand,
    )

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
        num_workers=0,  # Windows 建议先用 0
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
    if nclass == 4:
        class_weights = build_class_weights(index_csv, train_subjects).to(device)
    else:
        class_weights = build_diagnosis_class_weights(index_csv, train_subjects).to(device)

    print(f"[Class weights] nclass={nclass}, weights={class_weights.detach().cpu().numpy().tolist()}")
    criterion_ce = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05,
    )

    dom_criterion = torch.nn.CrossEntropyLoss()
    contrast_criterion = RelationCosineContrastLoss(
        positive_classes=(1, 3),
        negative_class=(2,),
        pos_target=0.8,
        neg_margin=0.2,
        use_cross_subject=True,
        use_cross_class_only=True,
    )

    # -------- optimizer --------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    config = {
        "index_csv": index_csv,
        "fold": fold,
        "rand": rand,
        "run_seed": run_seed,
        "deterministic": deterministic,
        "nclass": nclass,
        "n_splits": n_splits,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "lambda_graph": lambda_graph,
        "lambda_dom": lambda_dom,
        "grl_domain": grl_domain,
        "lambda_con": lambda_con,
        "dataset": dataset,
        "early_stop_patience": early_stop_patience,
        "early_stop_warmup": early_stop_warmup,
        "early_stop_min_delta": early_stop_min_delta,
        "early_stop_track": early_stop_track,
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
        f"min_delta={early_stop_min_delta}"
    )

    history = []

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")

        train_metrics = train_one_epoch_comp4(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            nclass=nclass,
            criterion_ce=criterion_ce,
            contrast_criterion=contrast_criterion,
            dom_criterion=dom_criterion,
            device=device,
            grl_domain=grl_domain,
            lambda_graph=lambda_graph,
            lambdad_domain=lambda_dom,
            lambda_con=lambda_con,
        )

        val_metrics = validate_one_epoch_comp4(
            model=model,
            loader=val_loader,
            nclass=nclass,
            criterion_ce=criterion_ce,
            contrast_criterion=contrast_criterion,
            device=device,
            lambda_graph=lambda_graph,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Train] loss={train_metrics['loss']:.4f} "
            f"ce={train_metrics['ce_loss']:.4f} "
            f"graph={train_metrics['graph_loss']:.4f} "
            f"loss_dom={train_metrics['loss_dom']:.4f} "
            f"loss_con={train_metrics['loss_con']:.4f} "
            f"acc={train_metrics['acc']:.4f}"
        )
        print(
            f"[Val]   loss={val_metrics['loss']:.4f} "
            f"ce={val_metrics['ce_loss']:.4f} "
            f"graph={val_metrics['graph_loss']:.4f} "
            f"contrast_loss={val_metrics['contrast_loss']:.4f} "
            f"acc4={val_metrics['acc']:.4f} "
            f"macro_f1_4={val_metrics['macro_f1']:.4f} "
            f"seg_diag_acc={val_metrics['emotion_acc']:.4f} "
            f"seg_diag_f1={val_metrics['emotion_macro_f1']:.4f} "
            f"trial_diag_acc={val_metrics['trial_acc']:.4f} "
            f"trial_diag_f1={val_metrics['trial_macro_f1']:.4f} "
            f"subject_diag_acc={val_metrics['subject_acc']:.4f} "
            f"subject_diag_f1={val_metrics['subject_macro_f1']:.4f}"
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
                    config=config,
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
# Main
# =========================================================

if __name__ == "__main__":
    device = "cuda:0"
    nclass = 2  # 抑郁症/正常人二分类；训练标签来自 diagnosis_label
    version = "diagnoise_cls"
    model_params_root = "model_params"
    os.makedirs(version, exist_ok=True)
    os.makedirs(model_params_root, exist_ok=True)

    all_fold_rows = []
    all_best_rows_long = []
    combined_diagnosis_confusions = []
    combined_segment_diag_confusions = []
    combined_trial_diag_confusions = []
    combined_subject_diag_confusions = []

    # 多随机种子 × 五折交叉验证
    for repeat, rand in enumerate([20,42]):
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
            run_seed = make_run_seed(rand, fold)
            set_global_seed(run_seed, deterministic=True)
            print(f"[Main Seed] rand={rand}, fold={fold}, run_seed={run_seed}")

            model = EmotionPretrainModel(
                sfreq=250.0,
                prior_matrix=None,
                topk=6,
                dropout=0.2,
                num_subjects=48,
                nclass=nclass,
            )

            result = train_competition_cross_subject(
                model=model,
                index_csv="com_index_sub_10s.csv",
                fold=fold,
                save_dir=os.path.join(model_params_root, f"diag_reapt{repeat}_fold{fold}"),
                dataset="com",
                n_splits=n_splits,
                epochs=100,
                batch_size=128,
                nclass=nclass,
                lr=1e-4,
                rand=rand,
                weight_decay=1e-4,
                num_workers=4,
                grl_domain=0.0,
                lambda_graph=0.0,
                lambda_dom=0.0,
                lambda_con=0.0,
                device=device,
                run_seed=run_seed,
                deterministic=True,
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
                combined_diagnosis_confusions.append(combined_metrics["confusion_matrix"])
                combined_segment_diag_confusions.append(combined_metrics["emotion_confusion_matrix"])
                combined_trial_diag_confusions.append(combined_metrics["trial_confusion_matrix"])
                combined_subject_diag_confusions.append(combined_metrics["subject_confusion_matrix"])

            print(f"第 {fold + 1} 折 combined 最优结果:")
            print(f"best_epoch = {combined_tracker['best_epoch']}")
            print(f"val_segment_diag_acc = {combined_metrics['acc']:.4f}")
            print(f"val_segment_diag_macro_f1 = {combined_metrics['macro_f1']:.4f}")
            print(f"val_trial_diag_acc = {combined_metrics['trial_acc']:.4f}")
            print(f"val_trial_diag_f1 = {combined_metrics['trial_macro_f1']:.4f}")
            print(f"val_subject_diag_acc = {combined_metrics['subject_acc']:.4f}")
            print(f"val_subject_diag_f1 = {combined_metrics['subject_macro_f1']:.4f}")

        seed_df = pd.DataFrame(seed_combined_rows)
        seed_csv = os.path.join(version, f"seed{rand}_combined_5fold_results.csv")
        seed_df.to_csv(seed_csv, index=False, encoding="utf-8-sig")

        metric_cols = [
            "val_acc",
            "val_macro_f1",
            "val_emotion_acc",
            "val_emotion_macro_f1",
            "val_trial_acc",
            "val_trial_macro_f1",
            "val_subject_acc",
            "val_subject_macro_f1",
            "val_diagnosis_acc",
            "val_diagnosis_macro_f1",
            "val_loss",
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

    np.save(os.path.join(version, "combined_diagnosis_confusions.npy"), np.array(combined_diagnosis_confusions, dtype=object))
    np.save(os.path.join(version, "combined_segment_diag_confusions.npy"), np.array(combined_segment_diag_confusions, dtype=object))
    np.save(os.path.join(version, "combined_trial_diag_confusions.npy"), np.array(combined_trial_diag_confusions, dtype=object))
    np.save(os.path.join(version, "combined_subject_diag_confusions.npy"), np.array(combined_subject_diag_confusions, dtype=object))

    if len(all_fold_df) > 0:
        metric_cols = [
            "val_acc",
            "val_macro_f1",
            "val_emotion_acc",
            "val_emotion_macro_f1",
            "val_trial_acc",
            "val_trial_macro_f1",
            "val_subject_acc",
            "val_subject_macro_f1",
            "val_diagnosis_acc",
            "val_diagnosis_macro_f1",
            "val_loss",
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

        # 平均 segment 级诊断二分类混淆矩阵
        valid_confs = [np.array(c, dtype=np.float32) for c in combined_diagnosis_confusions if c is not None]
        if len(valid_confs) > 0:
            conf_stack = np.stack(valid_confs, axis=0)
            avg_conf = conf_stack.mean(axis=0)
            row_sum = avg_conf.sum(axis=1, keepdims=True)
            avg_conf_norm = avg_conf / np.maximum(row_sum, 1e-12)

            import matplotlib.pyplot as plt

            labels = ["Diag_0", "Diag_1"]
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(avg_conf_norm, interpolation="nearest", cmap="Blues")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Average Segment-level Diagnosis Confusion Matrix")

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
            fig_path = os.path.join(version, "avg_segment_diagnosis_confusion.png")
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            print(f"平均 segment 级诊断二分类混淆矩阵图已保存到: {fig_path}")

        # 平均 trial 级诊断二分类混淆矩阵
        valid_trial_confs = [np.array(c, dtype=np.float32) for c in combined_trial_diag_confusions if c is not None]
        if len(valid_trial_confs) > 0:
            trial_conf_stack = np.stack(valid_trial_confs, axis=0)
            avg_trial_conf = trial_conf_stack.mean(axis=0)
            row_sum = avg_trial_conf.sum(axis=1, keepdims=True)
            avg_trial_conf_norm = avg_trial_conf / np.maximum(row_sum, 1e-12)

            import matplotlib.pyplot as plt

            labels = ["Diag_0", "Diag_1"]
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(avg_trial_conf_norm, interpolation="nearest", cmap="Blues")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Average Trial-level Diagnosis Confusion Matrix")

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
            fig_path = os.path.join(version, "avg_trial_diagnosis_confusion.png")
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            print(f"平均 trial 级诊断二分类混淆矩阵图已保存到: {fig_path}")


        # 平均 subject 级诊断二分类混淆矩阵
        valid_subject_confs = [np.array(c, dtype=np.float32) for c in combined_subject_diag_confusions if c is not None]
        if len(valid_subject_confs) > 0:
            subject_conf_stack = np.stack(valid_subject_confs, axis=0)
            avg_subject_conf = subject_conf_stack.mean(axis=0)
            row_sum = avg_subject_conf.sum(axis=1, keepdims=True)
            avg_subject_conf_norm = avg_subject_conf / np.maximum(row_sum, 1e-12)

            import matplotlib.pyplot as plt

            labels = ["Diag_0", "Diag_1"]
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(avg_subject_conf_norm, interpolation="nearest", cmap="Blues")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Average Subject-level Diagnosis Confusion Matrix")

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
            fig_path = os.path.join(version, "avg_subject_diagnosis_confusion.png")
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            print(f"平均 subject 级诊断二分类混淆矩阵图已保存到: {fig_path}")
    else:
        print("没有收集到任何 fold 结果，无法计算总体统计与混淆矩阵。")

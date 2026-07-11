"""
Train Experiment A: ordered full-trial temporal aggregation plus two-stage SSAS.

两阶段训练流程：
  Stage 1（SSAS — Source Subject Adaptive Selection）：
    使用域判别器 + GRL（梯度反转层）学习域不变特征，
    并通过 MMD 对齐源域和目标域的分布，最后用投票机制为每个源被试分配权重。
  Stage 2（Expert Emotion Adaptation）：
    在 Stage 1 学到的特征基础上，使用健康/抑郁两个专家分支
    （hard expert）和混合预测头（mixture head），结合 ranking loss
    和熵正则化，进行精细化的情绪分类。

Quick check（小批量快速验证）:
  python V7/train_experiment_a.py --fold 0 --stage1_epochs 2 --stage2_epochs 2 --batch_size 4
Full run（完整训练）:
  python V7/train_experiment_a.py --all_folds --all_repeats --trial_num_windows 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# 项目根路径 — 确保可以从仓库根目录导入模块
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

import config
from dataloader import Competition4ClassDataset
from utils.data import expand_window_index, resolve_data_path
from utils.folds import get_unified_subject_split
from V7.experiment_a_model import (
    Stage1SSASSourceSelectionModel,   # Stage 1: 域不变特征 + 域分类器
    Stage2ExpertEmotionAdaptationModel,  # Stage 2: 双专家 + 混合预测
    hard_expert_emotion_loss,          # 硬专家情绪损失
    mixture_emotion_nll_loss,          # 混合预测 NLL 损失
    target_entropy_loss,               # 目标域熵正则化
    weighted_mmd_rbf,                  # 加权 RBF-MMD 距离
)
from V7.trial_sequence import TrialSequenceDataset, trial_sequence_collate


# ============================================================================
# 通用工具函数
# ============================================================================

def set_seed(seed: int, deterministic: bool = False) -> None:
    """固定 Python / NumPy / PyTorch 的随机种子，确保实验可复现。"""
    os.environ["PYTHONHASHSEED"] = str(seed); random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def natural_key(value: Any):
    """自然排序键：数字字符串按数值排序，其余按字典序，避免 "10" < "2" 的问题。"""
    text = str(value); return (0, int(text)) if text.isdigit() else (1, text)


def jsonable(value: Any):
    """递归将 numpy/torch/Path 等类型转换为可 JSON 序列化的纯 Python 对象。"""
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if torch.is_tensor(value): return value.detach().cpu().tolist()
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def save_json(path: Path, value: Any) -> None:
    """将任意 Python 对象保存为格式化的 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def move(batch: dict, device: torch.device) -> dict:
    """将 batch 中的所有 tensor 异步移动到指定设备（GPU/CPU）。"""
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def window_collate(samples: list[dict]) -> dict:
    """窗口级 batch 的 collate 函数：stack tensor，list 保持不变。"""
    return {key: torch.stack([x[key] for x in samples]) if torch.is_tensor(samples[0][key])
            else [x[key] for x in samples] for key in samples[0]}


# ============================================================================
# 数据集类
# ============================================================================


class DomainAwareCompetitionDataset(Competition4ClassDataset):
    """带域标签的受试者数据集。

    继承自 Competition4ClassDataset，额外为每条窗口数据附加：
      - domain_id：该受试者所属的域索引（source / val / test）
      - diagnosis_label：是否抑郁的二分类标签（由 label4 >= 2 转换而来）
    """

    def __init__(self, index_csv, subject_ids, domain_mapping, split_prefix, normalize=True,
                 use_label4_for_diagnosis=True):
        super().__init__(index_csv=index_csv, subject_ids=subject_ids, normalize=normalize)
        self.domain_mapping, self.split_prefix = domain_mapping, split_prefix
        self.use_label4_for_diagnosis = use_label4_for_diagnosis

    def __getitem__(self, idx):
        item = super().__getitem__(idx); subject = str(int(item["subject_id"]))
        key = f"{self.split_prefix}:{subject}"
        item["domain_id"] = torch.tensor(self.domain_mapping["key_to_domain"][key])
        if self.use_label4_for_diagnosis:
            item["diagnosis_label"] = (item["label4"] >= 2).long()
        item.update(user_id=subject, target_key=key)
        return item


class UnlabeledTargetDataset(Dataset):
    """无标签目标域数据集（用于测试集）。

    从 index CSV 读取测试数据路径，加载 EEG trial 和 DE 特征。
    因为测试集无真实标签，label4 / emotion_label / diagnosis_label 均填充为 0。
    """

    def __init__(self, index_csv, domain_mapping, normalize=True):
        raw = pd.read_csv(index_csv)
        self.df = expand_window_index(raw, root=ROOT).reset_index(drop=True)
        self.id_col = "user_id" if "user_id" in self.df else "subject_id"
        self.domain_mapping, self.normalize = domain_mapping, normalize
        users = sorted(self.df[self.id_col].astype(str).unique(), key=natural_key)
        self.user_to_int = {u: i for i, u in enumerate(users)}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 加载并裁剪 trial 时间序列
        trial = np.load(resolve_data_path(row["trial_path"], ROOT))
        start = int(row.get("start", 0)); end = int(row.get("end", trial.shape[-1]))
        x = trial[:, start:end].astype("float32", copy=False)
        # 通道级 z-score 归一化
        if self.normalize: x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)
        # 加载 DE 特征（可能是多窗口的，按 de_win_id 选取）
        de = np.load(resolve_data_path(row["de_path"], ROOT))
        if de.ndim >= 3 and "de_win_id" in row: de = de[int(row["de_win_id"])]
        uid = str(row[self.id_col]); key = f"test:{uid}"
        sid = int(row["subject_number"]) if "subject_number" in row and pd.notna(row["subject_number"]) else self.user_to_int[uid]
        return {"x": torch.from_numpy(x), "de_feat": torch.tensor(de, dtype=torch.float32),
                "label4": torch.tensor(0), "emotion_label": torch.tensor(0),
                "diagnosis_label": torch.tensor(0), "subject_id": torch.tensor(sid),
                "domain_id": torch.tensor(self.domain_mapping["key_to_domain"][key]),
                "trial_id": torch.tensor(int(row["trial_id"])), "user_id": uid, "target_key": key}


# ============================================================================
# 域映射 & 数据构建
# ============================================================================


def test_users(path: Optional[str]) -> list[str]:
    """从测试集 CSV 中读取所有用户 ID 列表。"""
    if not path or not Path(path).exists(): return []
    df = pd.read_csv(path); col = "user_id" if "user_id" in df else "subject_id"
    return sorted(df[col].astype(str).unique(), key=natural_key)


def domain_mapping(train: Iterable, val: Iterable, test_csv: Optional[str]) -> dict:
    """构建完整的域映射表。

    每个 subject 对应一个域 ID，划分为三类：
      - source: 训练用的源域受试者
      - val:    验证用受试者
      - test:   测试集用户

    Returns:
        key_to_domain:          "source:U001" → 域索引
        domain_to_key:          域索引 → "source:U001"
        source_domain_indices:  源域的所有域索引列表（用于 Stage 1 投票）
        num_domains:            域总数
    """
    train, val, test = map(lambda xs: sorted(map(str, xs), key=natural_key), (train, val, test_users(test_csv)))
    keys = [f"source:{x}" for x in train] + [f"val:{x}" for x in val] + [f"test:{x}" for x in test]
    lookup = {key: i for i, key in enumerate(keys)}
    return {"key_to_domain": lookup, "domain_to_key": {str(v): k for k, v in lookup.items()},
            "source_subjects": train, "val_subjects": val, "test_users": test,
            "source_domain_indices": [lookup[f"source:{x}"] for x in train], "num_domains": len(keys)}


def trial_loader(window_ds, trial_num_windows, batch_size, workers, name, shuffle=False, drop_last=False):
    """从窗口数据集构造 TrialSequenceDataset 和对应的 DataLoader。

    TrialSequenceDataset 将同一 trial 的窗口打包成序列，支持变长序列的 collate。
    """
    ds = TrialSequenceDataset(window_ds, trial_num_windows=trial_num_windows, name=name)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        drop_last=drop_last and len(ds) >= batch_size, num_workers=workers,
                        pin_memory=True, collate_fn=trial_sequence_collate)
    return ds, loader


def build_data(args, split, mapping):
    """构建训练/验证/测试的所有数据集和 DataLoader。

    返回：
      src_w, val_w          — 窗口级源域/验证域数据集（用于 baseline 统计）
      src, val              — trial 级源域/验证域数据集
      test_trial            — trial 级测试数据集（可能为 None）
      src_loader            — 源域训练 DataLoader（shuffle=True）
      val_loader            — 验证域 DataLoader
      target_loader         — 目标域训练 DataLoader（val + test，shuffle=True）
      target_vote           — 目标域投票 DataLoader（val + test，shuffle=False）
    """
    # 窗口级数据集：用于计算受试者基线统计量
    src_w = DomainAwareCompetitionDataset(args.index_csv, split["train_all"], mapping, "source",
                                           not args.no_normalize, not args.use_raw_diagnosis_label)
    val_w = DomainAwareCompetitionDataset(args.index_csv, split["val_all"], mapping, "val",
                                           not args.no_normalize, not args.use_raw_diagnosis_label)
    # trial 级数据集：窗口打包成 trial 序列
    src, src_loader = trial_loader(src_w, args.trial_num_windows, args.batch_size, args.num_workers, "source", True, True)
    val, val_loader = trial_loader(val_w, args.trial_num_windows, args.batch_size, args.num_workers, "val")
    # 目标域 = 验证域 + 测试域（用于域自适应训练）
    target_parts: list[Dataset] = [val]
    test_trial = None
    if args.test_csv and Path(args.test_csv).exists() and mapping["test_users"]:
        test_w = UnlabeledTargetDataset(args.test_csv, mapping, not args.no_normalize)
        test_trial = TrialSequenceDataset(test_w, args.trial_num_windows, "test")
        target_parts.append(test_trial)
    target = ConcatDataset(target_parts) if len(target_parts) > 1 else val
    target_train = DataLoader(target, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(target) >= args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, collate_fn=trial_sequence_collate)
    target_vote = DataLoader(target, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, collate_fn=trial_sequence_collate)
    return src_w, val_w, src, val, test_trial, src_loader, val_loader, target_train, target_vote


# ============================================================================
# 受试者基线统计量（DE 特征 & 生物标志物的均值/标准差）
# 用于 subject-relative 归一化：模型输入时减去受试者自身均值再除以标准差
# ============================================================================


def baseline_key(value) -> str:
    """将 subject ID 转换为统一的字符串键，用于 baseline 统计量的 key。"""
    if torch.is_tensor(value): value = value.item()
    return str(int(value)) if isinstance(value, (int, np.integer, float)) and float(value).is_integer() else str(value)


def compute_de_baseline(window_ds, eps=1e-6):
    """计算每个受试者的 DE 特征的均值 (mu) 和标准差 (std)。

    遍历所有窗口样本，按 subject 分组累积 sum 和 sum of squares，
    最后用 Welford 风格公式计算每个受试者的统计量。
    """
    sums, squares, count = {}, {}, defaultdict(int)
    for i in tqdm(range(len(window_ds)), desc="DE baseline", leave=False):
        item = window_ds[i]; key = baseline_key(item.get("target_key", item["subject_id"])); x = item["de_feat"].float()
        sums[key] = sums.get(key, torch.zeros_like(x)) + x
        squares[key] = squares.get(key, torch.zeros_like(x)) + x.square(); count[key] += 1
    mu = {k: v / count[k] for k, v in sums.items()}
    std = {k: (squares[k] / count[k] - mu[k].square()).clamp_min(0).add(eps).sqrt() for k in mu}
    return mu, std


@torch.no_grad()
def compute_bio_baseline(model, window_ds, device, de_mu, de_std, batch_size, workers, eps=1e-6):
    """计算受试者级别的生物标志物 (biomarker) 均值与标准差。

    需要先完成 DE baseline 计算（de_mu, de_std），然后通过模型 backbone
    前向传播得到 bio_raw 特征，再按 subject 分组建模统计量。
    与 V2 版本中的 subject-relative 方法保持一致。
    """
    backbone = model.shared_encoder.backbone
    if not getattr(backbone, "use_biomarkers", True):
        return {}, {}
    loader = DataLoader(window_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        collate_fn=window_collate)
    sums, squares, counts = {}, {}, defaultdict(int)
    was_training = model.training; model.eval()
    for batch in tqdm(loader, desc="Bio baseline", leave=False):
        batch = move(batch, device); keys = batch_keys(batch)
        def gather(store):
            """从 baseline dict 中按 batch key 顺序收集对应统计量。"""
            if not store or any(k not in store for k in keys): return None
            return torch.stack([store[k] for k in keys]).to(device=device, dtype=batch["de_feat"].dtype)
        out = backbone(batch["x"], de_feat=batch["de_feat"], subject_de_mu=gather(de_mu),
                       subject_de_std=gather(de_std), subject_bio_mu=None, subject_bio_std=None)
        raw = out.get("bio_raw")
        if raw is None:
            if was_training: model.train()
            return {}, {}
        for i, key in enumerate(keys):
            feat = raw[i].detach().cpu().float()
            sums[key] = sums.get(key, torch.zeros_like(feat)) + feat
            squares[key] = squares.get(key, torch.zeros_like(feat)) + feat.square(); counts[key] += 1
    if was_training: model.train()
    mu = {k: v / counts[k] for k, v in sums.items()}
    std = {k: (squares[k] / counts[k] - mu[k].square()).clamp_min(0).add(eps).sqrt() for k in mu}
    return mu, std


def batch_keys(batch):
    """从 batch 中提取每个样本的字符串 key（优先使用 target_key，否则用 subject_id）。"""
    values = batch.get("target_key", batch["subject_id"])
    return [baseline_key(x) for x in (values.detach().cpu().tolist() if torch.is_tensor(values) else values)]


def baseline_kwargs(batch, device, de_mu=None, de_std=None, bio_mu=None, bio_std=None):
    """根据 batch 中的样本 keys 从 baseline 字典中收集对应的统计量，
    构造成模型 forward 所需的 subject_relative 参数。

    Args:
        de_mu, de_std:   DE 特征的受试者均值/标准差
        bio_mu, bio_std: 生物标志物的受试者均值/标准差
    """
    keys = batch_keys(batch)
    def gather(store):
        if not store or any(k not in store for k in keys): return None
        return torch.stack([store[k] for k in keys]).to(device=device, dtype=batch["de_feat"].dtype)
    return {"subject_de_mu": gather(de_mu), "subject_de_std": gather(de_std),
            "subject_bio_mu": gather(bio_mu), "subject_bio_std": gather(bio_std)}


# ============================================================================
# 模型构建 & 前向传播
# ============================================================================


def model_kwargs(args):
    """从命令行参数中提取模型构造函数所需的关键字参数。"""
    return dict(sfreq=args.sfreq, topk=args.topk, dropout=args.dropout,
                use_biomarkers=not args.no_biomarkers, biomarker_dim=args.biomarker_dim,
                use_subject_relative_de=not args.no_subject_relative_de,
                use_subject_relative_bio=(not args.no_subject_relative_bio and not args.no_biomarkers),
                bio_abs_scale=args.bio_abs_scale, relative_eps=args.relative_eps, de_num_bands=args.de_num_bands,
                temporal_hidden_dim=args.temporal_hidden_dim, temporal_kernel_size=args.temporal_kernel_size,
                temporal_dilations=tuple(args.temporal_dilations), temporal_dropout=args.temporal_dropout)


def forward(model, batch, device, baselines, **kwargs):
    """统一的模型前向传播：自动注入 subject-relative baseline 统计量。

    Args:
        model:      模型实例
        batch:      数据 batch
        device:     目标设备
        baselines:  四元组 (de_mu, de_std, bio_mu, bio_std)
        **kwargs:   额外的 forward 参数（如 lambda_emo, lambda_diag 等 GRL 系数）
    """
    rel = baseline_kwargs(batch, device, *baselines)
    return model(batch["x"], batch["de_feat"], batch["window_mask"], **kwargs, **rel)


# ============================================================================
# Stage 1 训练 —— 域自适应源选择 (SSAS)
#
# 目标：学习域不变特征表示，同时利用 GRL 抑制情绪/诊断分类信息，
#       使特征对域判别有区分力但对情绪/诊断标签不可区分。
# 损失：domain CE + weighted MMD + emotion_GRL + diagnosis_GRL
# ============================================================================

def train_stage1(model, source_loader, target_loader, optimizer, device, args, source_base, target_base):
    """Stage 1 训练循环。

    交替从源域和目标域采样 batch，计算域判别损失 + MMD 对齐损失 +
    GRL 辅助损失（情绪 + 诊断），通过梯度反转层迫使编码器学习域不变特征。
    target_loader 通过 cycle 无限循环，确保与 source_loader 长度匹配。
    """
    model.train(); totals = defaultdict(float); n = 0; target_iter = cycle(target_loader)
    for step, source in enumerate(tqdm(source_loader, desc="Stage1 train", leave=False)):
        if args.max_batches > 0 and step >= args.max_batches: break
        source, target = move(source, device), move(next(target_iter), device); optimizer.zero_grad(set_to_none=True)
        so = forward(model, source, device, source_base, lambda_emo=args.grl_emo, lambda_diag=args.grl_diag)
        to = forward(model, target, device, target_base)
        logits = torch.cat((so["domain_logits"], to["domain_logits"])); labels = torch.cat((source["domain_id"], target["domain_id"])).long()
        parts = {"domain": F.cross_entropy(logits, labels), "mmd": weighted_mmd_rbf(so["z_mmd"], to["z_mmd"]),
                 "emotion": F.cross_entropy(so["emotion_logits_grl"], source["emotion_label"].long()),
                 "diagnosis": F.cross_entropy(so["diagnosis_logits_grl"], source["diagnosis_label"].long())}
        loss = args.lambda_domain*parts["domain"] + args.stage1_lambda_mmd*parts["mmd"] + args.lambda_emo_grl*parts["emotion"] + args.lambda_diag_grl*parts["diagnosis"]
        loss.backward(); optimizer.step(); b = len(source["x"]); n += b; totals["loss"] += loss.item()*b
        for k, v in parts.items(): totals[k] += v.item()*b
    return {k: v/max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def validate_stage1(model, source_loader, target_loader, device, args, source_base, target_base):
    """Stage 1 验证：计算各损失项在验证集上的平均值（不更新参数）。"""
    model.eval(); totals=defaultdict(float); n=0; target_iter=cycle(target_loader)
    for step, source in enumerate(source_loader):
        if args.max_batches > 0 and step >= args.max_batches: break
        source,target=move(source,device),move(next(target_iter),device)
        so=forward(model,source,device,source_base); to=forward(model,target,device,target_base)
        logits=torch.cat((so["domain_logits"],to["domain_logits"])); labels=torch.cat((source["domain_id"],target["domain_id"])).long()
        parts={"domain":F.cross_entropy(logits,labels),"mmd":weighted_mmd_rbf(so["z_mmd"],to["z_mmd"]),
               "emotion":F.cross_entropy(so["emotion_logits_grl"],source["emotion_label"].long()),
               "diagnosis":F.cross_entropy(so["diagnosis_logits_grl"],source["diagnosis_label"].long())}
        loss=args.lambda_domain*parts["domain"]+args.stage1_lambda_mmd*parts["mmd"]+args.lambda_emo_grl*parts["emotion"]+args.lambda_diag_grl*parts["diagnosis"]
        b=len(source["x"]); n+=b; totals["loss"]+=loss.item()*b
        for k,v in parts.items(): totals[k]+=v.item()*b
    return {k:v/max(n,1) for k,v in totals.items()}


@torch.no_grad()
def vote_source_weights(model, loader, mapping, device, baselines, args, save_dir):
    """Stage 1 投票：用训练好的域分类器对目标域 trial 投票，确定每个源受试者的重要性权重。

    投票模式：
      - hard:       每个 trial 投票给概率最大的源域（one-hot）
      - soft:       直接使用 softmax 概率
      - soft_conf:  用熵归一化的置信度对 softmax 概率加权

    最终权重经过平滑 (vote_smooth) 并归一化后缩放，使权重之和等于源域数量。
    """
    model.eval(); source_domains = mapping["source_domain_indices"]; domain_to_key = mapping["domain_to_key"]
    votes = torch.zeros(len(source_domains), dtype=torch.float64); rows = []
    for step, batch in enumerate(tqdm(loader, desc="Stage1 trial voting", leave=False)):
        if args.max_batches > 0 and step >= args.max_batches: break
        batch = move(batch, device); out = forward(model, batch, device, baselines)
        prob = torch.softmax(out["domain_logits"][:, source_domains] / max(args.tau_vote, 1e-6), 1)
        prob = prob / prob.sum(1, keepdim=True).clamp_min(1e-8)
        if args.vote_mode == "hard": contribution = F.one_hot(prob.argmax(1), len(source_domains)).float()
        elif args.vote_mode == "soft_conf":
            confidence = (1 + (prob * prob.clamp_min(1e-8).log()).sum(1) / np.log(max(len(source_domains), 2))).clamp_min(0)
            contribution = prob * confidence.pow(args.confidence_power).unsqueeze(1)
        else: contribution = prob
        votes += contribution.sum(0).cpu().double()
        for i in range(len(batch["x"])):
            rows.append({"user_id": str(batch["user_id"][i]), "trial_id": int(batch["trial_id"][i]),
                         "predicted_source": domain_to_key[str(source_domains[int(prob[i].argmax())])],
                         "confidence": float(prob[i].max())})
    raw = votes + args.vote_smooth; raw /= raw.sum().clamp_min(1e-8)
    weights = raw * len(raw)
    result = {domain_to_key[str(d)].split(":", 1)[1]: float(weights[i]) for i, d in enumerate(source_domains)}
    save_json(save_dir / "source_subject_weights.json", result)
    pd.DataFrame(rows).to_csv(save_dir / "source_subject_trial_votes.csv", index=False, encoding="utf-8-sig")
    return result


# ============================================================================
# Stage 2 训练 —— 专家情绪适应
#
# 目标：在 Stage 1 特征基础上，用双专家（健康/抑郁）+ 混合预测头
#       进行精细的情绪分类。引入 ranking loss 确保正样本得分 > 负样本，
#       以及目标域熵正则化防止过拟合。
# ============================================================================

def sample_weights(subject_ids, weights, device):
    """根据投票得到的源受试者权重表，为 batch 中每个样本采样对应权重。"""
    values = [weights.get(str(int(x)), 1.0) for x in subject_ids.detach().cpu().tolist()]
    return torch.tensor(values, device=device, dtype=torch.float32)


def ranking_loss(prob, labels, subjects, margin, max_pairs):
    """受试者内 pairwise ranking loss。

    对每个受试者，确保正样本（情绪=1）的 log-odds 得分高于负样本（情绪=0），
    差值不足 margin 时产生惩罚。为防止组合爆炸，最多采样 max_pairs 对。
    """
    terms = []
    score = torch.log(prob[:, 1].clamp_min(1e-8)) - torch.log(prob[:, 0].clamp_min(1e-8))
    for subject in subjects.unique():
        idx = subjects == subject; pos, neg = score[idx & (labels == 1)], score[idx & (labels == 0)]
        if pos.numel() and neg.numel():
            losses = F.relu(margin - pos[:, None] + neg[None, :]).flatten()
            if losses.numel() > max_pairs: losses = losses.topk(max_pairs).values
            terms.append(losses.mean())
    return torch.stack(terms).mean() if terms else score.new_tensor(0.0)


def train_stage2(model, source_loader, target_loader, optimizer, device, args, weights, source_base, target_base, epoch):
    """Stage 2 训练循环。

    损失组成：
      - expert:   硬专家情绪损失（健康专家 / 抑郁专家二选一）
      - mix:      混合预测 NLL 损失
      - diagnosis: 诊断分类辅助损失
      - mmd:      加权 MMD 对齐损失
      - subject:  受试者域判别损失（含 GRL）
      - entropy:  目标域熵正则化（鼓励预测更确定）
      - rank:     受试者内 ranking loss（warmup 后启用）
    """
    model.train(); totals = defaultdict(float); n = 0; target_iter = cycle(target_loader)
    for step, source in enumerate(tqdm(source_loader, desc="Stage2 train", leave=False)):
        if args.max_batches > 0 and step >= args.max_batches: break
        source, target = move(source, device), move(next(target_iter), device); optimizer.zero_grad(set_to_none=True)
        so = forward(model, source, device, source_base, lambda_subject=args.grl_subject)
        to = forward(model, target, device, target_base, lambda_subject=args.grl_subject)
        y, yd = source["emotion_label"].long(), source["diagnosis_label"].long(); sw = sample_weights(source["subject_id"], weights, device)
        domain_logits = torch.cat((so["subject_domain_logits"], to["subject_domain_logits"])); domain_y = torch.cat((source["domain_id"], target["domain_id"])).long()
        parts = {"expert": hard_expert_emotion_loss(so["hc_logits"], so["dep_logits"], y, yd, sw),
                 "mix": mixture_emotion_nll_loss(so["mix_prob"], y, sw),
                 "diagnosis": (F.cross_entropy(so["diag_logits"], yd, reduction="none") * sw).mean(),
                 "mmd": weighted_mmd_rbf(so["z_mmd"], to["z_mmd"], sw),
                 "subject": F.cross_entropy(domain_logits, domain_y),
                 "entropy": target_entropy_loss(to["mix_prob"]) if args.lambda_ent else so["z"].new_tensor(0.),
                 "rank": ranking_loss(so["mix_prob"], y, source["subject_id"], args.rank_margin, args.rank_max_pairs_per_subject) if args.lambda_rank and epoch > args.rank_warmup_epochs else so["z"].new_tensor(0.)}
        loss = args.lambda_expert*parts["expert"] + args.lambda_mix*parts["mix"] + args.lambda_diag*parts["diagnosis"] + args.stage2_lambda_mmd*parts["mmd"] + args.lambda_subject*parts["subject"] + args.lambda_ent*parts["entropy"] + args.lambda_rank*parts["rank"]
        loss.backward(); optimizer.step(); b = len(y); n += b; totals["loss"] += loss.item()*b
        for k,v in parts.items(): totals[k] += v.item()*b
    return {k:v/max(n,1) for k,v in totals.items()}


def apply_topk(frame: pd.DataFrame, k: int, score="prob_pos") -> np.ndarray:
    """Top-K 预测：每个用户选取 score 最高的 K 个 trial 标记为正类。

    这是竞赛评估的核心逻辑 — 每名用户最多预测 K 个正样本。
    """
    pred = np.zeros(len(frame), dtype=int)
    for _, positions in frame.groupby("user_id", sort=False).indices.items():
        positions = np.asarray(positions); chosen = positions[np.argsort(-frame.iloc[positions][score].to_numpy())[:min(k,len(positions))]]
        pred[chosen] = 1
    return pred


@torch.no_grad()
def predict_trials(model, loader, device, baselines, labeled=True, max_batches=0):
    """对 DataLoader 中所有 trial 进行推理，返回包含预测概率、得分、注意力的 DataFrame。

    Returns 字段：
      - prob_pos:      正类（情绪=1）预测概率
      - score_pos:     log-odds 得分 log(p1/p0)
      - pred_emo:      情绪预测类别
      - pred_diag:     诊断预测类别
      - temporal_attention_emotion/diag:  时间注意力权重（仅有效窗口）
    """
    model.eval(); rows=[]
    for step, batch in enumerate(tqdm(loader, desc="Trial prediction", leave=False)):
        if max_batches > 0 and step >= max_batches: break
        batch=move(batch,device); out=forward(model,batch,device,baselines); prob=out["mix_prob"]; diag=out["diag_logits"].argmax(1)
        score=torch.log(prob[:,1].clamp_min(1e-8))-torch.log(prob[:,0].clamp_min(1e-8))
        for i in range(len(prob)):
            row={"user_id":str(batch["user_id"][i]),"subject_id":int(batch["subject_id"][i]),"trial_id":int(batch["trial_id"][i]),
                 "prob_pos":float(prob[i,1]),"score_pos":float(score[i]),"pred_emo":int(prob[i].argmax()),"pred_diag":int(diag[i]),
                 "temporal_attention_emotion":json.dumps(out["temporal_attention_emotion"][i,batch["window_mask"][i]].cpu().tolist()),
                 "temporal_attention_diag":json.dumps(out["temporal_attention_diag"][i,batch["window_mask"][i]].cpu().tolist())}
            if labeled: row.update(label_emo=int(batch["emotion_label"][i]), label_diag=int(batch["diagnosis_label"][i]))
            rows.append(row)
    return pd.DataFrame(rows)


def validate(model, loader, device, baselines, k_pos, max_batches=0):
    """验证/测试：推理 + 计算 trial 级别和 top-K 级别的各项指标。

    Returns:
        trial_acc/macro_f1:   逐 trial 的准确率 / macro F1
        topk_trial_acc/macro_f1:  Top-K 筛选后的准确率 / macro F1
        diagnosis_acc/macro_f1:   诊断分类指标
        records:                完整预测 DataFrame
    """
    frame=predict_trials(model,loader,device,baselines,True,max_batches); pred=frame.pred_emo.to_numpy(); y=frame.label_emo.to_numpy()
    topk=apply_topk(frame,k_pos); yd=frame.label_diag.to_numpy(); dp=frame.pred_diag.to_numpy()
    return {"trial_acc":accuracy_score(y,pred),"trial_macro_f1":f1_score(y,pred,average="macro",zero_division=0),
            "trial_confusion_matrix":confusion_matrix(y,pred,labels=[0,1]),
            "topk_trial_acc":accuracy_score(y,topk),"topk_trial_macro_f1":f1_score(y,topk,average="macro",zero_division=0),
            "diagnosis_acc":accuracy_score(yd,dp),"diagnosis_macro_f1":f1_score(yd,dp,average="macro",zero_division=0),
            "records":frame}


def checkpoint_payload(model,args,mapping,weights,fold,repeat,seed,metrics):
    """构建完整的 checkpoint 字典，包含模型参数、配置、域映射和验证指标。"""
    return {"format":"V7_experiment_a","model_state":model.state_dict(),"model_config":model_kwargs(args),
            "temporal_config":model.temporal_config,"trial_num_windows":args.trial_num_windows,
            "backbone_config":{k:model_kwargs(args)[k] for k in ("sfreq","topk","dropout","biomarker_dim","de_num_bands")},
            "num_domains":mapping["num_domains"],"domain_mapping":mapping,"source_subject_weights":weights,
            "shared_mix_alpha":args.shared_mix_alpha,"fold":fold,"repeat":repeat,"seed":seed,
            "metrics":jsonable({k:v for k,v in metrics.items() if k!="records"}),"args":vars(args)}


def rebuild_stage2(ckpt,device):
    """从 checkpoint 重建 Stage 2 模型，用于恢复训练或集成推理。"""
    cfg=dict(ckpt["model_config"]); model=Stage2ExpertEmotionAdaptationModel(ckpt["num_domains"],shared_mix_alpha=ckpt.get("shared_mix_alpha",.7),**cfg).to(device)
    result=model.load_state_dict(ckpt["model_state"],strict=False)
    print(f"[checkpoint load] missing_keys={result.missing_keys}; unexpected_keys={result.unexpected_keys}")
    return model


def run_fold(args, fold, repeat, seed):
    """执行单折完整的两阶段训练流程。

    流程概览：
      1. 数据准备：划分 fold、构建域映射、计算 subject-relative baseline
      2. Stage 1：训练域判别器 + MMD 对齐 → 保存最佳模型
      3. 投票：对目标域 trial 投票得到源受试者权重
      4. Stage 2：用投票权重初始化专家模型，训练情绪分类头
      5. 评估：在验证集上计算 top-K 指标，可选输出测试集预测

    Returns:
        dict: fold / repeat / best_path / metrics
    """
    run_seed=config.make_run_seed(seed,fold); set_seed(run_seed,args.deterministic); device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir=Path(args.save_root)/f"experiment_a_repeat{repeat}_fold{fold}"; save_dir.mkdir(parents=True,exist_ok=True)
    split=get_unified_subject_split(args.index_csv,fold=fold,n_splits=args.n_splits,seed=seed); mapping=domain_mapping(split["train_all"],split["val_all"],args.test_csv); save_json(save_dir/"domain_mapping.json",mapping)
    # ---- 构建数据和 baseline ----
    src_w,val_w,src,val,test_ds,src_loader,val_loader,target_loader,target_vote=build_data(args,split,mapping)
    source_base=target_base=val_base=(None,None,None,None)
    # 计算 DE 特征的 subject-relative 基线统计量
    if not args.no_subject_relative_de:
        sm,ss=compute_de_baseline(src_w,args.relative_eps); vm,vs=compute_de_baseline(val_w,args.relative_eps)
        # 测试集和验证集的 key 不同（test:xxx vs val:xxx），需要分别计算后合并
        tm,ts=dict(vm),dict(vs)
        if test_ds is not None:
            test_window=test_ds.window_dataset; xm,xs=compute_de_baseline(test_window,args.relative_eps); tm.update(xm); ts.update(xs)
        source_base=(sm,ss,None,None); val_base=(vm,vs,None,None); target_base=(tm,ts,None,None)
    kwargs=model_kwargs(args); stage1=Stage1SSASSourceSelectionModel(mapping["num_domains"],**kwargs).to(device)
    # 计算 biomarker 的 subject-relative 基线统计量（需要先初始化 Stage1 模型）
    if not args.no_subject_relative_bio and not args.no_biomarkers:
        sbm,sbs=compute_bio_baseline(stage1,src_w,device,source_base[0],source_base[1],args.batch_size,args.num_workers,args.relative_eps)
        vbm,vbs=compute_bio_baseline(stage1,val_w,device,val_base[0],val_base[1],args.batch_size,args.num_workers,args.relative_eps)
        tbm,tbs=dict(vbm),dict(vbs)
        if test_ds is not None:
            xbm,xbs=compute_bio_baseline(stage1,test_ds.window_dataset,device,target_base[0],target_base[1],args.batch_size,args.num_workers,args.relative_eps); tbm.update(xbm); tbs.update(xbs)
        source_base=(source_base[0],source_base[1],sbm,sbs); val_base=(val_base[0],val_base[1],vbm,vbs); target_base=(target_base[0],target_base[1],tbm,tbs)

    # ---- Stage 1：域自适应源选择 ----
    opt=torch.optim.AdamW(stage1.parameters(),lr=args.lr_stage1,weight_decay=args.weight_decay); best=float("inf")
    for epoch in range(1,args.stage1_epochs+1):
        metrics=train_stage1(stage1,src_loader,target_loader,opt,device,args,source_base,target_base)
        val1=validate_stage1(stage1,val_loader,target_vote,device,args,val_base,target_base); print(f"[Stage1] epoch={epoch} train={metrics} val={val1}")
        if val1["loss"]<best: best=val1["loss"]; torch.save({"model_state":stage1.state_dict(),"temporal_config":stage1.temporal_config,"trial_num_windows":args.trial_num_windows,"model_config":kwargs,"domain_mapping":mapping,"fold":fold,"seed":run_seed},save_dir/"stage1_best.pt")
    # 重新加载最佳 Stage1 模型
    stage1_state=torch.load(save_dir/"stage1_best.pt",map_location=device,weights_only=False)["model_state"]
    stage1_loaded=stage1.load_state_dict(stage1_state,strict=False); print(f"[stage1 reload] missing_keys={stage1_loaded.missing_keys}; unexpected_keys={stage1_loaded.unexpected_keys}")

    # ---- 投票得到源受试者权重 ----
    weights=vote_source_weights(stage1,target_vote,mapping,device,target_base,args,save_dir)

    # ---- Stage 2：专家情绪适应 ----
    stage2=Stage2ExpertEmotionAdaptationModel(mapping["num_domains"],shared_mix_alpha=args.shared_mix_alpha,**kwargs).to(device)
    # 用 Stage1 的 shared_encoder 初始化 Stage2（迁移域不变特征）
    if not args.no_stage1_init:
        loaded=stage2.shared_encoder.load_state_dict(stage1.shared_encoder.state_dict(),strict=False)
        print(f"[stage2 init] missing_keys={loaded.missing_keys}; unexpected_keys={loaded.unexpected_keys}")
    opt=torch.optim.AdamW(stage2.parameters(),lr=args.lr_stage2,weight_decay=args.weight_decay); best_metric=-1.; best_path=save_dir/"stage2_best.pt"; patience=0
    for epoch in range(1,args.stage2_epochs+1):
        train_m=train_stage2(stage2,src_loader,target_loader,opt,device,args,weights,source_base,target_base,epoch)
        val_m=validate(stage2,val_loader,device,val_base,args.k_pos,args.max_batches); score=val_m["topk_trial_macro_f1"]
        print(f"[Stage2] epoch={epoch} loss={train_m['loss']:.4f} trial_f1={val_m['trial_macro_f1']:.4f} topk_f1={score:.4f}")
        # 早停机制：基于 topk_trial_macro_f1 的增量阈值
        if score>best_metric+args.stage2_early_stop_min_delta:
            best_metric=score; patience=0; torch.save(checkpoint_payload(stage2,args,mapping,weights,fold,repeat,run_seed,val_m),best_path)
        else: patience+=1
        if not args.no_stage2_early_stop and epoch>=args.stage2_early_stop_warmup and patience>=args.stage2_early_stop_patience: break

    # ---- 最终评估 ----
    ckpt=torch.load(best_path,map_location=device,weights_only=False); restored=rebuild_stage2(ckpt,device); final=validate(restored,val_loader,device,val_base,args.k_pos,args.max_batches); final["records"].to_csv(save_dir/"validation_trials.csv",index=False,encoding="utf-8-sig")
    if args.predict_test and test_ds is not None:
        loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=trial_sequence_collate); predict_trials(restored,loader,device,target_base,False,args.max_batches).to_csv(save_dir/"test_trials.csv",index=False,encoding="utf-8-sig")
    return {"fold":fold,"repeat":repeat,"best_path":str(best_path),"metrics":{k:v for k,v in final.items() if k!="records"}}


def ensemble(paths, args):
    """多折模型集成：加载所有 checkpoint，对测试集分别推理后取平均概率。

    每个 checkpoint 独立计算 subject-relative baseline 后推理，
    最终将多个模型的 prob_pos 和 score_pos 取均值，并用 Top-K 生成提交文件。
    """
    device=torch.device(args.device if torch.cuda.is_available() else "cpu"); frames=[]
    for i,path in enumerate(paths):
        ckpt=torch.load(path,map_location=device,weights_only=False); model=rebuild_stage2(ckpt,device); mapping=ckpt["domain_mapping"]
        test_w=UnlabeledTargetDataset(args.test_csv,mapping,not args.no_normalize); test_ds=TrialSequenceDataset(test_w,ckpt["trial_num_windows"],f"ensemble-{i}")
        loader=DataLoader(test_ds,batch_size=args.batch_size,collate_fn=trial_sequence_collate,num_workers=args.num_workers)
        dm=ds=None
        if ckpt["model_config"].get("use_subject_relative_de"):
            dm,ds=compute_de_baseline(test_w,args.relative_eps)
        bm=bs=None
        if ckpt["model_config"].get("use_subject_relative_bio"):
            bm,bs=compute_bio_baseline(model,test_w,device,dm,ds,args.batch_size,args.num_workers,args.relative_eps)
        frame=predict_trials(model,loader,device,(dm,ds,bm,bs),False).rename(columns={"prob_pos":f"prob_{i}","score_pos":f"score_{i}"}); frames.append(frame[["user_id","trial_id",f"prob_{i}",f"score_{i}"]])
    merged=frames[0]
    for frame in frames[1:]: merged=merged.merge(frame,on=["user_id","trial_id"],how="inner")
    merged["prob_pos"]=merged.filter(regex="^prob_").mean(1); merged["score_pos"]=merged.filter(regex="^score_").mean(1); merged["Emotion_label"]=apply_topk(merged,args.k_pos)
    out=Path(args.save_root)/"test_ensemble"; out.mkdir(parents=True,exist_ok=True); merged.to_csv(out/"test_ensemble_probs.csv",index=False,encoding="utf-8-sig"); merged[["user_id","trial_id","Emotion_label"]].to_csv(out/"submission_test_ensemble.csv",index=False,encoding="utf-8-sig")


def parse_args():
    """解析命令行参数，按功能分为以下几组：

    训练控制: fold/all_folds/repeat/all_repeats, epochs, batch_size, lr, early_stop
    模型架构: sfreq/topk/dropout/biomarker_dim/de_num_bands, temporal_*
    Baseline控制: no_biomarkers/no_normalize/no_subject_relative_de/no_subject_relative_bio
    Stage1 损失权重: lambda_domain/stage1_lambda_mmd/lambda_emo_grl/grl_emo/lambda_diag_grl/grl_diag
    Stage1 投票: vote_smooth/vote_mode/tau_vote/confidence_power
    Stage2 损失权重: lambda_expert/lambda_mix/lambda_diag/stage2_lambda_mmd/lambda_subject/grl_subject/lambda_ent/lambda_rank
    """
    p=argparse.ArgumentParser()
    # ---- 数据路径 ----
    p.add_argument("--index_csv",default="com_index_sub_2s.csv"); p.add_argument("--test_csv",default="com_test_trial_index_2s.csv"); p.add_argument("--save_root",default="model_params/V7_experiment_a")
    # ---- 实验组织 ----
    p.add_argument("--fold",type=int,default=0); p.add_argument("--all_folds",action="store_true"); p.add_argument("--n_splits",type=int,default=10); p.add_argument("--repeat",type=int,default=0); p.add_argument("--all_repeats",action="store_true")
    # ---- 训练超参数 ----
    p.add_argument("--stage1_epochs",type=int,default=5); p.add_argument("--stage2_epochs",type=int,default=25); p.add_argument("--batch_size",type=int,default=4); p.add_argument("--num_workers",type=int,default=0); p.add_argument("--device",default="cuda:0"); p.add_argument("--deterministic",action="store_true"); p.add_argument("--max_batches",type=int,default=0,help="debug only; 0 uses every batch")
    # ---- 时域聚合参数 ----
    p.add_argument("--trial_num_windows",type=int,default=0); p.add_argument("--temporal_hidden_dim",type=int,default=128); p.add_argument("--temporal_kernel_size",type=int,default=3); p.add_argument("--temporal_dilations",type=int,nargs="+",default=[1,2]); p.add_argument("--temporal_dropout",type=float,default=.3); p.add_argument("--lambda_window_aux",type=float,default=0.)
    # ---- 优化器 & 模型架构 ----
    p.add_argument("--lr_stage1",type=float,default=1e-4); p.add_argument("--lr_stage2",type=float,default=1e-4); p.add_argument("--weight_decay",type=float,default=1e-3); p.add_argument("--sfreq",type=float,default=250.); p.add_argument("--topk",type=int,default=8); p.add_argument("--dropout",type=float,default=.45); p.add_argument("--biomarker_dim",type=int,default=57); p.add_argument("--de_num_bands",type=int,default=5)
    # ---- Baseline 控制 ----
    p.add_argument("--no_biomarkers",action="store_true"); p.add_argument("--no_normalize",action="store_true"); p.add_argument("--no_subject_relative_de",action="store_true"); p.add_argument("--no_subject_relative_bio",action="store_true"); p.add_argument("--bio_abs_scale",type=float,default=.3); p.add_argument("--relative_eps",type=float,default=1e-6); p.add_argument("--use_raw_diagnosis_label",action="store_true")
    # ---- Stage 1 损失权重 ----
    p.add_argument("--lambda_domain",type=float,default=1.); p.add_argument("--stage1_lambda_mmd",type=float,default=.03); p.add_argument("--lambda_emo_grl",type=float,default=.001); p.add_argument("--grl_emo",type=float,default=.01); p.add_argument("--lambda_diag_grl",type=float,default=.001); p.add_argument("--grl_diag",type=float,default=.01)
    # ---- Stage 1 投票 ----
    p.add_argument("--vote_smooth",type=float,default=5.); p.add_argument("--vote_mode",choices=["hard","soft","soft_conf"],default="soft"); p.add_argument("--tau_vote",type=float,default=1.); p.add_argument("--confidence_power",type=float,default=1.)
    # ---- Stage 2 损失权重 ----
    p.add_argument("--lambda_expert",type=float,default=.5); p.add_argument("--lambda_mix",type=float,default=1.); p.add_argument("--lambda_diag",type=float,default=.01); p.add_argument("--stage2_lambda_mmd",type=float,default=.0003); p.add_argument("--lambda_subject",type=float,default=.0003); p.add_argument("--grl_subject",type=float,default=.001); p.add_argument("--lambda_ent",type=float,default=0.); p.add_argument("--lambda_rank",type=float,default=.1); p.add_argument("--rank_margin",type=float,default=.2); p.add_argument("--rank_warmup_epochs",type=int,default=3); p.add_argument("--rank_max_pairs_per_subject",type=int,default=128); p.add_argument("--shared_mix_alpha",type=float,default=.7)
    # ---- 训练策略控制 ----
    p.add_argument("--no_stage1_init",action="store_true"); p.add_argument("--no_stage2_early_stop",action="store_true"); p.add_argument("--stage2_early_stop_patience",type=int,default=5); p.add_argument("--stage2_early_stop_warmup",type=int,default=3); p.add_argument("--stage2_early_stop_min_delta",type=float,default=1e-6); p.add_argument("--k_pos",type=int,default=4); p.add_argument("--predict_test",action="store_true"); p.add_argument("--no_test_ensemble",action="store_true")
    return p.parse_args()


def main():
    """主入口：解析参数 → 遍历 folds/repeats → 逐折训练 → 保存汇总 → 可选集成。"""
    args=parse_args(); seeds=list(getattr(config,"V2_seed",[42])); repeats=range(len(seeds)) if args.all_repeats else [args.repeat]; folds=range(args.n_splits) if args.all_folds else [args.fold]; results=[]
    for repeat in repeats:
        seed=seeds[repeat] if repeat<len(seeds) else seeds[0]+31*repeat
        for fold in folds: results.append(run_fold(args,fold,repeat,int(seed)))
    root=Path(args.save_root); root.mkdir(parents=True,exist_ok=True); save_json(root/"all_fold_summary.json",results)
    if not args.no_test_ensemble and args.test_csv and Path(args.test_csv).exists(): ensemble([r["best_path"] for r in results],args)


if __name__=="__main__": main()

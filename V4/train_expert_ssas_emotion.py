# -*- coding: utf-8 -*-
"""Train two-stage SSAS + three-branch encoder + HC/DEP emotion experts.

Typical quick run:
    python V4/train_expert_ssas_emotion.py --fold 0 --stage1_epochs 2 --stage2_epochs 2

Full 5-fold run:
    python V4/train_expert_ssas_emotion.py --all_folds --stage1_epochs 20 --stage2_epochs 100
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

import config
from dataloader import Competition4ClassDataset
from utils.data import expand_window_index, resolve_data_path
from utils.folds import get_unified_subject_split
from V4.expert_ssas_emotion_model import (
    Stage1SSASSourceSelectionModel,
    Stage2ExpertEmotionAdaptationModel,
    hard_expert_emotion_loss,
    mixture_emotion_nll_loss,
    soft_conditional_mmd_rbf,
    target_entropy_loss,
    weighted_mmd_rbf,
)


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


def natural_key(value: Any):
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def to_jsonable(obj: Any):
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
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def save_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def dict_collate(batch: list[dict]) -> dict:
    out: dict[str, Any] = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _baseline_key(value: Any) -> str:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor baseline key, got shape={tuple(value.shape)}")
        return str(int(value.detach().cpu().item()))
    return str(value)


def _batch_baseline_keys(batch: dict, key_field: str = "target_key") -> list[str]:
    if key_field in batch:
        values = batch[key_field]
    else:
        values = batch["subject_id"]

    if torch.is_tensor(values):
        return [str(int(v)) for v in values.detach().cpu().view(-1).tolist()]
    return [_baseline_key(v) for v in values]


def gather_subject_baseline(
    keys,
    baseline_dict: Optional[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    """Gather per-sample subject-relative baseline tensors.

    Keys are usually target_key values such as source:3, val:3, test:P_test1.
    This avoids collisions between competition subject_id and test subject_number.
    """

    if baseline_dict is None:
        return None
    if torch.is_tensor(keys):
        key_list = [str(int(v)) for v in keys.detach().cpu().view(-1).tolist()]
    else:
        key_list = [_baseline_key(v) for v in keys]

    values = []
    for key in key_list:
        if key not in baseline_dict:
            return None
        values.append(torch.as_tensor(baseline_dict[key]))
    out = torch.stack(values, dim=0).to(device=device)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def get_subject_relative_kwargs(
    batch: dict,
    device: torch.device,
    dtype: torch.dtype,
    de_mu: Optional[dict[str, torch.Tensor]] = None,
    de_std: Optional[dict[str, torch.Tensor]] = None,
    bio_mu: Optional[dict[str, torch.Tensor]] = None,
    bio_std: Optional[dict[str, torch.Tensor]] = None,
    key_field: str = "target_key",
) -> dict:
    keys = _batch_baseline_keys(batch, key_field=key_field)
    return {
        "subject_de_mu": gather_subject_baseline(keys, de_mu, device, dtype=dtype),
        "subject_de_std": gather_subject_baseline(keys, de_std, device, dtype=dtype),
        "subject_bio_mu": gather_subject_baseline(keys, bio_mu, device, dtype=dtype),
        "subject_bio_std": gather_subject_baseline(keys, bio_std, device, dtype=dtype),
    }


def compute_subject_de_baselines(dataset: Dataset, key_field: str = "target_key", eps: float = 1e-6) -> tuple[dict, dict]:
    sums: dict[str, torch.Tensor] = {}
    sq_sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)

    for idx in tqdm(range(len(dataset)), desc="DE baseline", leave=False):
        item = dataset[idx]
        key = _baseline_key(item[key_field] if key_field in item else item["subject_id"])
        de_feat = torch.as_tensor(item["de_feat"], dtype=torch.float32)
        if key not in sums:
            sums[key] = torch.zeros_like(de_feat)
            sq_sums[key] = torch.zeros_like(de_feat)
        sums[key] += de_feat
        sq_sums[key] += de_feat * de_feat
        counts[key] += 1

    subject_de_mu: dict[str, torch.Tensor] = {}
    subject_de_std: dict[str, torch.Tensor] = {}
    for key, total in sums.items():
        count = max(int(counts[key]), 1)
        mu = total / count
        var = sq_sums[key] / count - mu * mu
        std = torch.sqrt(torch.clamp(var, min=0.0) + eps)
        subject_de_mu[key] = mu
        subject_de_std[key] = std
    return subject_de_mu, subject_de_std


@torch.no_grad()
def compute_subject_bio_baselines(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    subject_de_mu: Optional[dict[str, torch.Tensor]] = None,
    subject_de_std: Optional[dict[str, torch.Tensor]] = None,
    key_field: str = "target_key",
    eps: float = 1e-6,
) -> tuple[dict, dict]:
    model.eval()
    sums: dict[str, torch.Tensor] = {}
    sq_sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)

    for batch in tqdm(loader, desc="Bio baseline", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        de_feat = batch["de_feat"].to(device, non_blocking=True)
        keys = _batch_baseline_keys(batch, key_field=key_field)
        de_mu_batch = gather_subject_baseline(keys, subject_de_mu, device, dtype=de_feat.dtype)
        de_std_batch = gather_subject_baseline(keys, subject_de_std, device, dtype=de_feat.dtype)

        out = model(
            x,
            de_feat,
            subject_de_mu=de_mu_batch,
            subject_de_std=de_std_batch,
            subject_bio_mu=None,
            subject_bio_std=None,
        )
        bio_raw = out.get("bio_raw", None)
        if bio_raw is None:
            raise RuntimeError("Model forward did not return bio_raw; cannot compute bio baselines.")
        bio_raw = bio_raw.detach().cpu().float()

        for i, key in enumerate(keys):
            feat = bio_raw[i]
            if key not in sums:
                sums[key] = torch.zeros_like(feat)
                sq_sums[key] = torch.zeros_like(feat)
            sums[key] += feat
            sq_sums[key] += feat * feat
            counts[key] += 1

    subject_bio_mu: dict[str, torch.Tensor] = {}
    subject_bio_std: dict[str, torch.Tensor] = {}
    for key, total in sums.items():
        count = max(int(counts[key]), 1)
        mu = total / count
        var = sq_sums[key] / count - mu * mu
        std = torch.sqrt(torch.clamp(var, min=0.0) + eps)
        subject_bio_mu[key] = mu
        subject_bio_std[key] = std
    return subject_bio_mu, subject_bio_std


class DomainAwareCompetitionDataset(Competition4ClassDataset):
    """Labeled source/val dataset with global train/val/test domain ids."""

    def __init__(
        self,
        index_csv: str | Path,
        subject_ids: Iterable,
        domain_mapping: dict,
        split_prefix: str,
        normalize: bool = True,
        use_label4_for_diagnosis: bool = True,
    ) -> None:
        super().__init__(index_csv=index_csv, subject_ids=subject_ids, normalize=normalize)
        self.domain_mapping = domain_mapping
        self.split_prefix = split_prefix
        self.use_label4_for_diagnosis = bool(use_label4_for_diagnosis)

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        subject_id = int(item["subject_id"].item())
        subject_key = str(subject_id)
        domain_key = f"{self.split_prefix}:{subject_key}"
        item["domain_id"] = torch.tensor(
            int(self.domain_mapping["key_to_domain"][domain_key]),
            dtype=torch.long,
        )
        if self.use_label4_for_diagnosis and "label4" in item:
            # Project label4 to the requested convention: 0=DEP, 1=HC.
            diag = int(item["label4"].item() >= 2)
            item["diagnosis_label"] = torch.tensor(diag, dtype=torch.long)
        item["user_id"] = subject_key
        item["target_key"] = domain_key
        return item


class UnlabeledTargetDataset(Dataset):
    """Unlabeled test target windows with the same tensor keys as labeled batches."""

    def __init__(
        self,
        index_csv: str | Path,
        domain_mapping: dict,
        root: Path = ROOT,
        normalize: bool = True,
    ) -> None:
        self.index_csv = Path(index_csv)
        raw_df = pd.read_csv(self.index_csv)
        if raw_df.empty:
            raise ValueError(f"Empty target csv: {index_csv}")
        self.id_col = "user_id" if "user_id" in raw_df.columns else "subject_id"
        self.df = expand_window_index(raw_df, root=root).reset_index(drop=True)
        self.domain_mapping = domain_mapping
        self.root = Path(root)
        self.normalize = bool(normalize)
        self.user_to_int = {
            uid: i for i, uid in enumerate(sorted(self.df[self.id_col].astype(str).unique(), key=natural_key))
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        trial = np.load(resolve_data_path(row["trial_path"], self.root))
        start = int(row["start"]) if "start" in row.index and not pd.isna(row["start"]) else 0
        end = int(row["end"]) if "end" in row.index and not pd.isna(row["end"]) else trial.shape[-1]
        x = trial[:, start:end].astype("float32", copy=False)
        if self.normalize:
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-6
            x = (x - mean) / std

        de_arr = np.load(resolve_data_path(row["de_path"], self.root))
        if de_arr.ndim >= 3 and "de_win_id" in row.index:
            de_feat = de_arr[int(row["de_win_id"])]
        else:
            de_feat = de_arr
        de_feat = de_feat.astype("float32", copy=False)

        uid = str(row[self.id_col])
        domain_key = f"test:{uid}"
        if "subject_number" in row.index and not pd.isna(row["subject_number"]):
            subject_id = int(row["subject_number"])
        else:
            subject_id = self.user_to_int[uid]

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "de_feat": torch.tensor(de_feat, dtype=torch.float32),
            "label4": torch.tensor(0, dtype=torch.long),
            "emotion_label": torch.tensor(0, dtype=torch.long),
            "diagnosis_label": torch.tensor(0, dtype=torch.long),
            "subject_id": torch.tensor(subject_id, dtype=torch.long),
            "domain_id": torch.tensor(int(self.domain_mapping["key_to_domain"][domain_key]), dtype=torch.long),
            "trial_id": torch.tensor(int(row["trial_id"]), dtype=torch.long),
            "user_id": uid,
            "target_key": domain_key,
        }


def build_train_val_target_split(
    index_csv: str,
    fold: int = 0,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    return get_unified_subject_split(index_csv=index_csv, fold=fold, n_splits=n_splits, seed=seed)


def _read_test_users(test_csv: str | None) -> list[str]:
    if test_csv is None or not Path(test_csv).exists():
        return []
    df = pd.read_csv(test_csv)
    if df.empty:
        return []
    id_col = "user_id" if "user_id" in df.columns else "subject_id"
    return sorted(df[id_col].astype(str).unique().tolist(), key=natural_key)


def build_domain_id_mapping(
    train_subjects: Iterable,
    val_subjects: Iterable,
    test_csv: str | None = None,
) -> dict:
    train_subjects = sorted([str(s) for s in train_subjects], key=natural_key)
    val_subjects = sorted([str(s) for s in val_subjects], key=natural_key)
    test_users = _read_test_users(test_csv)

    keys = [f"source:{s}" for s in train_subjects]
    keys += [f"val:{s}" for s in val_subjects]
    keys += [f"test:{u}" for u in test_users]

    key_to_domain = {key: idx for idx, key in enumerate(keys)}
    source_subject_to_domain = {s: key_to_domain[f"source:{s}"] for s in train_subjects}
    val_subject_to_domain = {s: key_to_domain[f"val:{s}"] for s in val_subjects}
    test_user_to_domain = {u: key_to_domain[f"test:{u}"] for u in test_users}

    return {
        "key_to_domain": key_to_domain,
        "domain_to_key": {str(v): k for k, v in key_to_domain.items()},
        "source_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_users": test_users,
        "source_subject_to_domain": source_subject_to_domain,
        "val_subject_to_domain": val_subject_to_domain,
        "test_user_to_domain": test_user_to_domain,
        "source_domain_indices": [source_subject_to_domain[s] for s in train_subjects],
        "num_domains": len(key_to_domain),
    }


def create_source_loader(
    index_csv: str,
    train_subjects: Iterable,
    domain_mapping: dict,
    batch_size: int,
    num_workers: int,
    generator: Optional[torch.Generator] = None,
    normalize: bool = True,
    use_label4_for_diagnosis: bool = True,
) -> tuple[DomainAwareCompetitionDataset, DataLoader]:
    dataset = DomainAwareCompetitionDataset(
        index_csv=index_csv,
        subject_ids=train_subjects,
        domain_mapping=domain_mapping,
        split_prefix="source",
        normalize=normalize,
        use_label4_for_diagnosis=use_label4_for_diagnosis,
    )
    effective_drop_last = len(dataset) >= batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=effective_drop_last,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dict_collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return dataset, loader


def create_val_loader(
    index_csv: str,
    val_subjects: Iterable,
    domain_mapping: dict,
    batch_size: int,
    num_workers: int,
    normalize: bool = True,
    use_label4_for_diagnosis: bool = True,
) -> tuple[DomainAwareCompetitionDataset, DataLoader]:
    dataset = DomainAwareCompetitionDataset(
        index_csv=index_csv,
        subject_ids=val_subjects,
        domain_mapping=domain_mapping,
        split_prefix="val",
        normalize=normalize,
        use_label4_for_diagnosis=use_label4_for_diagnosis,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dict_collate,
        worker_init_fn=seed_worker,
    )
    return dataset, loader


def create_target_loader(
    val_dataset: Dataset,
    domain_mapping: dict,
    batch_size: int,
    num_workers: int,
    test_csv: str | None = None,
    shuffle: bool = True,
    drop_last: bool = True,
    generator: Optional[torch.Generator] = None,
    normalize_test: bool = True,
) -> tuple[Dataset, DataLoader]:
    datasets: list[Dataset] = [val_dataset]
    if test_csv is not None and Path(test_csv).exists() and domain_mapping.get("test_users"):
        test_dataset = UnlabeledTargetDataset(
            index_csv=test_csv,
            domain_mapping=domain_mapping,
            root=ROOT,
            normalize=normalize_test,
        )
        datasets.append(test_dataset)
        print(f"[target] using val + test target: val={len(val_dataset)}, test={len(test_dataset)}")
    else:
        print(f"[target] using val target only; test_csv={test_csv}")

    dataset = ConcatDataset(datasets) if len(datasets) > 1 else val_dataset
    effective_drop_last = bool(drop_last and len(dataset) >= batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=effective_drop_last,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dict_collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return dataset, loader


def build_emotion_class_weights(source_dataset: DomainAwareCompetitionDataset) -> torch.Tensor:
    labels = source_dataset.df["emotion_label"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=2)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_diag_class_weights(source_dataset: DomainAwareCompetitionDataset, use_label4_for_diagnosis: bool) -> torch.Tensor:
    if use_label4_for_diagnosis and "label4" in source_dataset.df.columns:
        labels = (source_dataset.df["label4"].astype(int).to_numpy() >= 2).astype(int)
    else:
        labels = source_dataset.df["diagnosis_label"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=2)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    loss = F.cross_entropy(
        logits,
        labels.long(),
        reduction="none",
        label_smoothing=float(label_smoothing),
    )
    if sample_weight is not None:
        loss = loss * sample_weight.to(device=loss.device, dtype=loss.dtype)
    return loss.mean()


def _scalar_stat(value: Any) -> Optional[float]:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_arch_output_stats(out: dict) -> dict[str, float]:
    stats: dict[str, float] = {}
    gates = out.get("fusion_gates") or {}
    gate_names = {
        "temporal": "fusion_gate_temporal",
        "graph": "fusion_gate_graph",
        "core": "fusion_gate_core",
        "bio": "fusion_gate_bio",
    }
    for gate_key, metric_key in gate_names.items():
        value = _scalar_stat(gates.get(gate_key))
        if value is not None:
            stats[metric_key] = value

    beta = _scalar_stat(out.get("graph_beta"))
    if beta is not None:
        stats["graph_beta"] = beta

    adj_learnable = out.get("adj_learnable")
    if torch.is_tensor(adj_learnable) and adj_learnable.numel() > 0:
        adj_learnable_float = adj_learnable.detach().float()
        stats["learnable_graph_mean"] = float(adj_learnable_float.mean().cpu().item())
        stats["learnable_graph_std"] = float(adj_learnable_float.std(unbiased=False).cpu().item())

    adj_dense = out.get("adj_dense")
    if torch.is_tensor(adj_dense) and adj_dense.numel() > 0 and beta is not None:
        stats["adj_final_mean"] = float(adj_dense.detach().float().mean().cpu().item())
    return stats


def format_stage2_arch_stats(metrics: dict) -> str:
    parts = []
    aliases = [
        ("gate_t", "fusion_gate_temporal"),
        ("gate_g", "fusion_gate_graph"),
        ("gate_core", "fusion_gate_core"),
        ("gate_bio", "fusion_gate_bio"),
        ("graph_beta", "graph_beta"),
    ]
    for label, key in aliases:
        value = metrics.get(key)
        if value is not None:
            parts.append(f"{label}={float(value):.4f}")
    return (" " + " ".join(parts)) if parts else ""


def train_stage1_one_epoch(
    model: Stage1SSASSourceSelectionModel,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_domain: float = 1.0,
    lambda_mmd: float = 0.01,
    lambda_emo_grl: float = 0.001,
    grl_emo: float = 0.01,
    lambda_diag_grl: float = 0.001,
    grl_diag: float = 0.01,
    lambda_weight_reg: float = 0.0,
    emotion_class_weight: Optional[torch.Tensor] = None,
    diag_class_weight: Optional[torch.Tensor] = None,
    source_de_mu: Optional[dict[str, torch.Tensor]] = None,
    source_de_std: Optional[dict[str, torch.Tensor]] = None,
    source_bio_mu: Optional[dict[str, torch.Tensor]] = None,
    source_bio_std: Optional[dict[str, torch.Tensor]] = None,
    target_de_mu: Optional[dict[str, torch.Tensor]] = None,
    target_de_std: Optional[dict[str, torch.Tensor]] = None,
    target_bio_mu: Optional[dict[str, torch.Tensor]] = None,
    target_bio_std: Optional[dict[str, torch.Tensor]] = None,
) -> dict:
    model.train()
    target_iter = cycle(target_loader)
    totals = defaultdict(float)
    total_n = 0
    emotion_class_weight = emotion_class_weight.to(device) if emotion_class_weight is not None else None
    diag_class_weight = diag_class_weight.to(device) if diag_class_weight is not None else None

    pbar = tqdm(source_loader, desc="Stage1-Train", leave=False)
    for source_batch in pbar:
        target_batch = next(target_iter)
        source_batch = move_batch_to_device(source_batch, device)
        target_batch = move_batch_to_device(target_batch, device)

        optimizer.zero_grad(set_to_none=True)
        source_rel_kwargs = get_subject_relative_kwargs(
            source_batch,
            device,
            dtype=source_batch["de_feat"].dtype,
            de_mu=source_de_mu,
            de_std=source_de_std,
            bio_mu=source_bio_mu,
            bio_std=source_bio_std,
        )
        target_rel_kwargs = get_subject_relative_kwargs(
            target_batch,
            device,
            dtype=target_batch["de_feat"].dtype,
            de_mu=target_de_mu,
            de_std=target_de_std,
            bio_mu=target_bio_mu,
            bio_std=target_bio_std,
        )
        source_out = model(
            source_batch["x"],
            source_batch["de_feat"],
            lambda_emo=grl_emo,
            lambda_diag=grl_diag,
            **source_rel_kwargs,
        )
        target_out = model(
            target_batch["x"],
            target_batch["de_feat"],
            lambda_emo=0.0,
            lambda_diag=0.0,
            **target_rel_kwargs,
        )

        domain_logits = torch.cat([source_out["domain_logits"], target_out["domain_logits"]], dim=0)
        domain_labels = torch.cat([source_batch["domain_id"].long(), target_batch["domain_id"].long()], dim=0)
        loss_domain = F.cross_entropy(domain_logits, domain_labels)

        loss_mmd = weighted_mmd_rbf(source_out["z_mmd"], target_out["z_mmd"])
        loss_emo = F.cross_entropy(
            source_out["emotion_logits_grl"],
            source_batch["emotion_label"].long(),
            weight=emotion_class_weight,
        )
        loss_diag = F.cross_entropy(
            source_out["diagnosis_logits_grl"],
            source_batch["diagnosis_label"].long(),
            weight=diag_class_weight,
        )
        loss_weight_reg = source_out["z"].new_tensor(0.0)

        loss = (
            lambda_domain * loss_domain
            + lambda_mmd * loss_mmd
            + lambda_emo_grl * loss_emo
            + lambda_diag_grl * loss_diag
            + lambda_weight_reg * loss_weight_reg
        )
        loss.backward()
        optimizer.step()

        bsz = source_batch["x"].size(0)
        total_n += bsz
        totals["loss"] += float(loss.item()) * bsz
        totals["loss_domain"] += float(loss_domain.item()) * bsz
        totals["loss_mmd"] += float(loss_mmd.item()) * bsz
        totals["loss_emo_grl"] += float(loss_emo.item()) * bsz
        totals["loss_diag_grl"] += float(loss_diag.item()) * bsz
        totals["domain_acc"] += int((domain_logits.argmax(dim=1) == domain_labels).sum().item())
        totals["domain_n"] += int(domain_labels.numel())
        pbar.set_postfix(
            {
                "loss": f"{totals['loss'] / max(total_n, 1):.4f}",
                "dom": f"{totals['loss_domain'] / max(total_n, 1):.4f}",
                "mmd": f"{totals['loss_mmd'] / max(total_n, 1):.4f}",
            }
        )

    metrics = {k: v / max(total_n, 1) for k, v in totals.items() if k not in ("domain_acc", "domain_n")}
    metrics["domain_acc"] = totals["domain_acc"] / max(totals["domain_n"], 1)
    return metrics


@torch.no_grad()
def validate_stage1(
    model: Stage1SSASSourceSelectionModel,
    source_loader: DataLoader,
    target_loader: DataLoader,
    device: torch.device,
    source_de_mu: Optional[dict[str, torch.Tensor]] = None,
    source_de_std: Optional[dict[str, torch.Tensor]] = None,
    source_bio_mu: Optional[dict[str, torch.Tensor]] = None,
    source_bio_std: Optional[dict[str, torch.Tensor]] = None,
    target_de_mu: Optional[dict[str, torch.Tensor]] = None,
    target_de_std: Optional[dict[str, torch.Tensor]] = None,
    target_bio_mu: Optional[dict[str, torch.Tensor]] = None,
    target_bio_std: Optional[dict[str, torch.Tensor]] = None,
) -> dict:
    model.eval()
    totals = defaultdict(float)
    total_n = 0
    target_iter = cycle(target_loader)
    steps = min(len(source_loader), len(target_loader))
    iterator = iter(source_loader)
    for _ in tqdm(range(steps), desc="Stage1-Val", leave=False):
        source_batch = move_batch_to_device(next(iterator), device)
        target_batch = move_batch_to_device(next(target_iter), device)
        source_rel_kwargs = get_subject_relative_kwargs(
            source_batch,
            device,
            dtype=source_batch["de_feat"].dtype,
            de_mu=source_de_mu,
            de_std=source_de_std,
            bio_mu=source_bio_mu,
            bio_std=source_bio_std,
        )
        target_rel_kwargs = get_subject_relative_kwargs(
            target_batch,
            device,
            dtype=target_batch["de_feat"].dtype,
            de_mu=target_de_mu,
            de_std=target_de_std,
            bio_mu=target_bio_mu,
            bio_std=target_bio_std,
        )
        source_out = model(source_batch["x"], source_batch["de_feat"], **source_rel_kwargs)
        target_out = model(target_batch["x"], target_batch["de_feat"], **target_rel_kwargs)

        domain_logits = torch.cat([source_out["domain_logits"], target_out["domain_logits"]], dim=0)
        domain_labels = torch.cat([source_batch["domain_id"].long(), target_batch["domain_id"].long()], dim=0)
        loss_domain = F.cross_entropy(domain_logits, domain_labels)
        loss_mmd = weighted_mmd_rbf(source_out["z_mmd"], target_out["z_mmd"])
        loss_emo = F.cross_entropy(source_out["emotion_logits_grl"], source_batch["emotion_label"].long())
        loss_diag = F.cross_entropy(source_out["diagnosis_logits_grl"], source_batch["diagnosis_label"].long())
        loss = loss_domain + 0.01 * loss_mmd + 0.001 * loss_emo + 0.001 * loss_diag

        bsz = source_batch["x"].size(0)
        total_n += bsz
        totals["loss"] += float(loss.item()) * bsz
        totals["loss_domain"] += float(loss_domain.item()) * bsz
        totals["loss_mmd"] += float(loss_mmd.item()) * bsz
        totals["domain_acc"] += int((domain_logits.argmax(dim=1) == domain_labels).sum().item())
        totals["domain_n"] += int(domain_labels.numel())

    metrics = {k: v / max(total_n, 1) for k, v in totals.items() if k not in ("domain_acc", "domain_n")}
    metrics["domain_acc"] = totals["domain_acc"] / max(totals["domain_n"], 1)
    return metrics


@torch.no_grad()
def compute_source_weights_by_classification_voting(
    model: Stage1SSASSourceSelectionModel,
    target_loader: DataLoader,
    domain_mapping: dict,
    device: torch.device,
    save_dir: str | Path,
    smooth: float = 5.0,
    vote_level: str = "trial",
    vote_mode: str = "soft",
    tau_vote: float = 1.0,
    confidence_power: float = 1.0,
    save_topk_probs: int = 5,
    target_de_mu: Optional[dict[str, torch.Tensor]] = None,
    target_de_std: Optional[dict[str, torch.Tensor]] = None,
    target_bio_mu: Optional[dict[str, torch.Tensor]] = None,
    target_bio_std: Optional[dict[str, torch.Tensor]] = None,
) -> dict:
    """
    Compute target-to-source subject weights with SSAS voting.

    vote_mode="hard" keeps the original SSAS-style hard voting behavior.
    vote_mode="soft" uses the source-subject probability vector as a soft vote.
    vote_mode="soft_conf" further weights soft votes by normalized confidence.
    Trial-level soft voting averages window probabilities first so each target
    trial contributes once, matching the final trial-level submission target.
    """
    if vote_level not in {"window", "trial"}:
        raise ValueError(f"Unknown vote_level={vote_level}; expected 'window' or 'trial'.")
    if vote_mode not in {"hard", "soft", "soft_conf"}:
        raise ValueError(f"Unknown vote_mode={vote_mode}; expected 'hard', 'soft' or 'soft_conf'.")
    tau_vote = max(float(tau_vote), 1e-6)
    save_topk_probs = max(1, int(save_topk_probs))

    model.eval()
    source_subjects = [str(s) for s in domain_mapping["source_subjects"]]
    num_source_subjects = len(source_subjects)
    source_domain_indices = torch.tensor(domain_mapping["source_domain_indices"], dtype=torch.long, device=device)
    domain_to_source = {
        int(domain_mapping["source_subject_to_domain"][s]): str(s)
        for s in source_subjects
    }
    source_domains = [int(domain_mapping["source_subject_to_domain"][s]) for s in source_subjects]
    eps = 1e-12
    log_num_sources = math.log(max(num_source_subjects, 2))
    topk = min(save_topk_probs, num_source_subjects)
    window_rows = []
    window_records = []

    def _topk_columns(prob_vec: np.ndarray) -> dict:
        if prob_vec.size == 0:
            return {}
        top_idx = np.argsort(-prob_vec)[:topk]
        data = {}
        for rank, idx in enumerate(top_idx, start=1):
            data[f"top{rank}_subject"] = source_subjects[int(idx)]
            data[f"top{rank}_domain"] = source_domains[int(idx)]
            data[f"top{rank}_prob"] = float(prob_vec[int(idx)])
        return data

    def _confidence(prob_vec: np.ndarray) -> float:
        entropy = -float(np.sum(prob_vec * np.log(prob_vec + eps)))
        conf = 1.0 - entropy / log_num_sources
        conf = max(0.0, min(1.0, conf))
        return float(conf ** float(confidence_power))

    for batch in tqdm(target_loader, desc="Stage1-Voting", leave=False):
        batch = move_batch_to_device(batch, device)
        target_rel_kwargs = get_subject_relative_kwargs(
            batch,
            device,
            dtype=batch["de_feat"].dtype,
            de_mu=target_de_mu,
            de_std=target_de_std,
            bio_mu=target_bio_mu,
            bio_std=target_bio_std,
        )
        out = model(batch["x"], batch["de_feat"], **target_rel_kwargs)
        source_logits = out["domain_logits"].index_select(dim=1, index=source_domain_indices)
        source_prob = torch.softmax(source_logits / tau_vote, dim=1)
        pred_local = source_prob.argmax(dim=1)
        prob_np = source_prob.detach().cpu().numpy()
        pred_domain = source_domain_indices[pred_local].detach().cpu().tolist()
        pred_subjects = [domain_to_source[int(d)] for d in pred_domain]

        for i, pred_subject in enumerate(pred_subjects):
            prob_vec = prob_np[i].astype(np.float64)
            confidence = _confidence(prob_vec)
            row = {
                "target_key": batch["target_key"][i],
                "user_id": batch["user_id"][i],
                "trial_id": int(batch["trial_id"][i].detach().cpu()),
                "pred_source_subject": pred_subject,
                "pred_source_domain": int(pred_domain[i]),
                "confidence": confidence,
            }
            row.update(_topk_columns(prob_vec))
            window_rows.append(row)
            window_records.append({**row, "_source_prob": prob_vec})

    vote_vec = np.zeros(num_source_subjects, dtype=np.float64)
    trial_rows = []

    if vote_level == "trial":
        grouped = defaultdict(list)
        for row in window_records:
            grouped[(row["target_key"], row["trial_id"])].append(row)
        for (target_key, trial_id), records in grouped.items():
            prob_stack = np.stack([r["_source_prob"] for r in records], axis=0)
            trial_prob = prob_stack.mean(axis=0)
            pred_local = int(np.argmax(trial_prob))
            confidence = _confidence(trial_prob)

            if vote_mode == "hard":
                vote_vec[pred_local] += 1.0
            elif vote_mode == "soft":
                vote_vec += trial_prob
            else:
                vote_vec += confidence * trial_prob

            first = records[0]
            row = {
                "target_key": target_key,
                "user_id": first["user_id"],
                "trial_id": int(trial_id),
                "n_windows": len(records),
                "pred_source_subject": source_subjects[pred_local],
                "pred_source_domain": source_domains[pred_local],
                "confidence": confidence,
            }
            row.update(_topk_columns(trial_prob))
            trial_rows.append(row)
    elif vote_level == "window":
        for record in window_records:
            prob_vec = record["_source_prob"]
            pred_local = int(np.argmax(prob_vec))
            confidence = float(record["confidence"])
            if vote_mode == "hard":
                vote_vec[pred_local] += 1.0
            elif vote_mode == "soft":
                vote_vec += prob_vec
            else:
                vote_vec += confidence * prob_vec

    smoothed_votes = vote_vec + float(smooth)
    weights = smoothed_votes / smoothed_votes.sum()
    weights_mean_one = weights * num_source_subjects
    vote_probs = vote_vec / max(float(vote_vec.sum()), eps)
    vote_entropy = -float(np.sum(vote_probs * np.log(vote_probs + eps)))
    confidences = (
        [float(r["confidence"]) for r in trial_rows]
        if vote_level == "trial"
        else [float(r["confidence"]) for r in window_rows]
    )
    result = {
        "type": "source_classification_voting",
        "vote_level": vote_level,
        "vote_mode": vote_mode,
        "tau_vote": float(tau_vote),
        "confidence_power": float(confidence_power),
        "smooth": float(smooth),
        "source_subjects": source_subjects,
        "votes": {s: float(v) for s, v in zip(source_subjects, vote_vec)},
        "counts": {s: int(round(v)) for s, v in zip(source_subjects, vote_vec)} if vote_mode == "hard" else {},
        "weights": {s: float(w) for s, w in zip(source_subjects, weights)},
        "weights_mean_one": {s: float(w) for s, w in zip(source_subjects, weights_mean_one)},
        "vote_entropy": vote_entropy,
        "num_target_windows": len(window_rows),
        "num_target_trials": len(trial_rows) if vote_level == "trial" else None,
    }
    if confidences:
        result["confidence_mean"] = float(np.mean(confidences))
        result["confidence_min"] = float(np.min(confidences))
        result["confidence_max"] = float(np.max(confidences))

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_json(save_dir / "source_subject_weights.json", result)
    pd.DataFrame(window_rows).to_csv(save_dir / "source_subject_votes.csv", index=False, encoding="utf-8-sig")
    if vote_level == "trial":
        pd.DataFrame(trial_rows).to_csv(save_dir / "source_subject_trial_votes.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([]).to_csv(save_dir / "source_subject_trial_votes.csv", index=False, encoding="utf-8-sig")
    save_json(save_dir / "domain_id_mapping.json", domain_mapping)

    print(f"[Stage1 Voting] vote_level={vote_level}, vote_mode={vote_mode}, tau_vote={tau_vote}")
    if vote_level == "trial":
        print(f"[Stage1 Voting] total target trials = {len(trial_rows)}")
    else:
        print(f"[Stage1 Voting] total target windows = {len(window_rows)}")
    print(f"[Stage1 Voting] vote entropy = {vote_entropy:.4f}")
    if vote_mode == "soft_conf" and confidences:
        print(
            f"[Stage1 Voting] confidence mean/min/max = "
            f"{np.mean(confidences):.4f}/{np.min(confidences):.4f}/{np.max(confidences):.4f}"
        )
    print("[Stage1 Voting] top source weights:")
    top_vote_rows = sorted(
        zip(source_subjects, vote_vec, weights, weights_mean_one),
        key=lambda item: item[1],
        reverse=True,
    )[:8]
    for subject, vote, weight, weight_mean_one in top_vote_rows:
        print(
            f"    {subject}, vote={vote:.4f}, "
            f"weight={weight:.6f}, weight_mean_one={weight_mean_one:.4f}"
        )
    return result


def _subject_weight_tensor(
    subject_ids: torch.Tensor,
    source_weights: dict[str, float],
    num_source_subjects: int,
    device: torch.device,
) -> torch.Tensor:
    default = (
        float(np.mean([float(v) for v in source_weights.values()]))
        if source_weights
        else 1.0 / max(num_source_subjects, 1)
    )
    values = [float(source_weights.get(str(int(s)), default)) for s in subject_ids.detach().cpu()]
    return torch.tensor(values, dtype=torch.float32, device=device)


def clip_source_weights(
    source_weights: dict[str, float],
    max_weight: float = 0.0,
    blend_uniform: float = 0.0,
    mean_one: bool = True,
) -> dict[str, float]:
    """Clip and smooth source-subject weights while preserving their scale."""
    if not source_weights:
        return {}

    keys = list(source_weights.keys())
    values = np.asarray([float(source_weights[k]) for k in keys], dtype=np.float64)
    if max_weight and max_weight > 0:
        values = np.minimum(values, float(max_weight))
    if blend_uniform and blend_uniform > 0:
        alpha = float(max(0.0, min(1.0, blend_uniform)))
        baseline = 1.0 if mean_one else 1.0 / max(len(keys), 1)
        values = (1.0 - alpha) * values + alpha * baseline

    target_sum = float(len(keys)) if mean_one else 1.0
    value_sum = float(values.sum())
    if value_sum > 0:
        values = values * (target_sum / value_sum)
    return {k: float(v) for k, v in zip(keys, values)}


def normalize_source_weight_dict(
    source_weights: dict[str, float],
    keys: Optional[Iterable] = None,
    mean_one: bool = True,
) -> dict[str, float]:
    if keys is None:
        keys = list(source_weights.keys())
    keys = [str(k) for k in keys]
    if not keys:
        return {}

    values = np.asarray([float(source_weights.get(k, 1.0)) for k in keys], dtype=np.float64)
    values = np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    target_sum = float(len(keys)) if mean_one else 1.0
    value_sum = float(values.sum())
    if value_sum <= 0:
        values = np.full_like(values, target_sum / max(len(keys), 1), dtype=np.float64)
    else:
        values = values * (target_sum / value_sum)
    return {k: float(v) for k, v in zip(keys, values)}


def summarize_source_weights(source_weights: dict[str, float]) -> dict[str, float]:
    if not source_weights:
        return {
            "source_weight_entropy": 0.0,
            "source_weight_max": 0.0,
            "source_weight_min": 0.0,
            "source_weight_mean": 0.0,
        }
    values = np.asarray([float(v) for v in source_weights.values()], dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total > 0:
        prob = values / total
        entropy = float(-(prob * np.log(np.clip(prob, 1e-12, None))).sum())
    else:
        entropy = 0.0
    return {
        "source_weight_entropy": entropy,
        "source_weight_max": float(values.max()) if values.size else 0.0,
        "source_weight_min": float(values.min()) if values.size else 0.0,
        "source_weight_mean": float(values.mean()) if values.size else 0.0,
    }


def source_weight_l1_change(old_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    keys = sorted(set(old_weights.keys()) | set(new_weights.keys()), key=natural_key)
    if not keys:
        return 0.0
    return float(sum(abs(float(new_weights.get(k, 0.0)) - float(old_weights.get(k, 0.0))) for k in keys))


@torch.no_grad()
def compute_dynamic_source_weights_by_mmd(
    model: Stage2ExpertEmotionAdaptationModel,
    source_dataset: Dataset,
    target_dataset: Dataset,
    domain_mapping: dict,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 0,
    source_de_mu: Optional[dict[str, torch.Tensor]] = None,
    source_de_std: Optional[dict[str, torch.Tensor]] = None,
    source_bio_mu: Optional[dict[str, torch.Tensor]] = None,
    source_bio_std: Optional[dict[str, torch.Tensor]] = None,
    target_de_mu: Optional[dict[str, torch.Tensor]] = None,
    target_de_std: Optional[dict[str, torch.Tensor]] = None,
    target_bio_mu: Optional[dict[str, torch.Tensor]] = None,
    target_bio_std: Optional[dict[str, torch.Tensor]] = None,
    dynamic_weight_tau: float = 1.0,
    dynamic_weight_min: float = 0.5,
    dynamic_weight_max: float = 2.0,
    max_samples_per_subject: int = 512,
    max_target_samples: int = 4096,
) -> tuple[dict[str, float], dict]:
    """Estimate source-subject weights from current Stage2 MMD distance to target."""

    source_subjects = [str(s) for s in domain_mapping.get("source_subjects", [])]
    default_weights = {subject_id: 1.0 for subject_id in source_subjects}
    if not source_subjects:
        return {}, {
            "dynamic_mmd_distance": {},
            "dynamic_weight_raw": {},
            "dynamic_weight_final": {},
            "dynamic_weight_entropy": 0.0,
            "dynamic_weight_max": 0.0,
            "dynamic_weight_min": 0.0,
            "dynamic_weight_mean": 0.0,
            "dynamic_mmd_distance_mean": 0.0,
            "dynamic_mmd_distance_min": 0.0,
            "dynamic_mmd_distance_max": 0.0,
        }

    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dict_collate,
        worker_init_fn=seed_worker,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dict_collate,
        worker_init_fn=seed_worker,
    )

    was_training = model.training
    model.eval()

    target_chunks = []
    target_seen = 0
    for batch in tqdm(target_loader, desc="Dynamic-MMD target", leave=False):
        batch = move_batch_to_device(batch, device)
        rel_kwargs = get_subject_relative_kwargs(
            batch,
            device,
            dtype=batch["de_feat"].dtype,
            de_mu=target_de_mu,
            de_std=target_de_std,
            bio_mu=target_bio_mu,
            bio_std=target_bio_std,
        )
        out = model(batch["x"], batch["de_feat"], lambda_subject=0.0, **rel_kwargs)
        z = out["z_mmd"].detach().cpu()
        if max_target_samples > 0:
            remaining = int(max_target_samples) - target_seen
            if remaining <= 0:
                break
            z = z[:remaining]
        target_chunks.append(z)
        target_seen += int(z.size(0))
        if max_target_samples > 0 and target_seen >= int(max_target_samples):
            break

    if not target_chunks:
        stats = summarize_source_weights(default_weights)
        if was_training:
            model.train()
        return default_weights, {
            "dynamic_mmd_distance": {subject_id: None for subject_id in source_subjects},
            "dynamic_weight_raw": dict(default_weights),
            "dynamic_weight_final": dict(default_weights),
            "dynamic_weight_entropy": stats["source_weight_entropy"],
            "dynamic_weight_max": stats["source_weight_max"],
            "dynamic_weight_min": stats["source_weight_min"],
            "dynamic_weight_mean": stats["source_weight_mean"],
            "dynamic_mmd_distance_mean": 0.0,
            "dynamic_mmd_distance_min": 0.0,
            "dynamic_mmd_distance_max": 0.0,
        }

    z_target_all = torch.cat(target_chunks, dim=0).to(device=device)
    source_feats_by_subject: dict[str, list[torch.Tensor]] = defaultdict(list)
    source_counts: dict[str, int] = defaultdict(int)
    max_per_subject = int(max_samples_per_subject)

    for batch in tqdm(source_loader, desc="Dynamic-MMD source", leave=False):
        batch = move_batch_to_device(batch, device)
        rel_kwargs = get_subject_relative_kwargs(
            batch,
            device,
            dtype=batch["de_feat"].dtype,
            de_mu=source_de_mu,
            de_std=source_de_std,
            bio_mu=source_bio_mu,
            bio_std=source_bio_std,
        )
        out = model(batch["x"], batch["de_feat"], lambda_subject=0.0, **rel_kwargs)
        z = out["z_mmd"].detach().cpu()
        subject_ids = batch["subject_id"].detach().cpu().view(-1).tolist()
        for i, sid in enumerate(subject_ids):
            subject_id = str(int(sid))
            if max_per_subject > 0 and source_counts[subject_id] >= max_per_subject:
                continue
            source_feats_by_subject[subject_id].append(z[i])
            source_counts[subject_id] += 1

    distances: dict[str, Optional[float]] = {}
    valid_distances = []
    for subject_id in source_subjects:
        feats = source_feats_by_subject.get(subject_id, [])
        if not feats:
            distances[subject_id] = None
            continue
        z_source_i = torch.stack(feats, dim=0).to(device=device)
        distance = weighted_mmd_rbf(z_source_i, z_target_all)
        distance_value = float(distance.detach().cpu().item())
        distances[subject_id] = distance_value
        valid_distances.append(distance_value)

    if not valid_distances:
        stats = summarize_source_weights(default_weights)
        if was_training:
            model.train()
        return default_weights, {
            "dynamic_mmd_distance": distances,
            "dynamic_weight_raw": dict(default_weights),
            "dynamic_weight_final": dict(default_weights),
            "dynamic_weight_entropy": stats["source_weight_entropy"],
            "dynamic_weight_max": stats["source_weight_max"],
            "dynamic_weight_min": stats["source_weight_min"],
            "dynamic_weight_mean": stats["source_weight_mean"],
            "dynamic_mmd_distance_mean": 0.0,
            "dynamic_mmd_distance_min": 0.0,
            "dynamic_mmd_distance_max": 0.0,
        }

    valid_array = np.asarray(valid_distances, dtype=np.float64)
    tau = max(float(dynamic_weight_tau), 1e-8)
    min_distance = float(valid_array.min())
    scores_by_subject: dict[str, float] = {}
    for subject_id, distance in distances.items():
        if distance is None:
            continue
        scores_by_subject[subject_id] = float(math.exp(-(float(distance) - min_distance) / tau))

    mean_score = float(np.mean(list(scores_by_subject.values()))) if scores_by_subject else 1.0
    raw_weights = dict(default_weights)
    if mean_score > 0:
        for subject_id, score in scores_by_subject.items():
            raw_weights[subject_id] = float(score / mean_score)

    final_weights = dict(raw_weights)
    min_w = float(dynamic_weight_min)
    max_w = float(dynamic_weight_max)
    if max_w > 0 and max_w >= min_w:
        final_weights = {k: float(np.clip(v, min_w, max_w)) for k, v in final_weights.items()}
    final_weights = normalize_source_weight_dict(final_weights, keys=source_subjects, mean_one=True)
    weight_stats = summarize_source_weights(final_weights)

    if was_training:
        model.train()

    return final_weights, {
        "dynamic_mmd_distance": distances,
        "dynamic_weight_raw": raw_weights,
        "dynamic_weight_final": final_weights,
        "dynamic_weight_entropy": weight_stats["source_weight_entropy"],
        "dynamic_weight_max": weight_stats["source_weight_max"],
        "dynamic_weight_min": weight_stats["source_weight_min"],
        "dynamic_weight_mean": weight_stats["source_weight_mean"],
        "dynamic_mmd_distance_mean": float(valid_array.mean()),
        "dynamic_mmd_distance_min": float(valid_array.min()),
        "dynamic_mmd_distance_max": float(valid_array.max()),
    }


def ema_update_source_weights(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    momentum: float = 0.8,
    blend_with_stage1: Optional[dict[str, float]] = None,
    stage1_alpha: float = 0.0,
    mean_one: bool = True,
) -> dict[str, float]:
    keys = sorted(set(old_weights.keys()) | set(new_weights.keys()), key=natural_key)
    if blend_with_stage1 is not None:
        keys = sorted(set(keys) | set(blend_with_stage1.keys()), key=natural_key)
    if not keys:
        return {}

    old_norm = normalize_source_weight_dict(old_weights, keys=keys, mean_one=mean_one)
    new_norm = normalize_source_weight_dict(new_weights, keys=keys, mean_one=mean_one)
    stage1_norm = (
        normalize_source_weight_dict(blend_with_stage1, keys=keys, mean_one=mean_one)
        if blend_with_stage1 is not None
        else None
    )

    momentum = float(max(0.0, min(1.0, momentum)))
    stage1_alpha = float(max(0.0, min(1.0, stage1_alpha)))
    updated = {}
    for key in keys:
        ema_w = momentum * old_norm[key] + (1.0 - momentum) * new_norm[key]
        if stage1_norm is not None and stage1_alpha > 0:
            ema_w = stage1_alpha * stage1_norm[key] + (1.0 - stage1_alpha) * ema_w
        updated[key] = float(ema_w)
    return normalize_source_weight_dict(updated, keys=keys, mean_one=mean_one)


def stage2_subject_ranking_loss(
    mix_prob: torch.Tensor,
    labels: torch.Tensor,
    subject_ids: torch.Tensor,
    trial_ids: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    margin: float = 0.2,
    max_pairs_per_subject: int = 128,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Subject-wise trial ranking loss aligned with final soft top-k inference.

    The score is log-odds of positive emotion. Window scores are averaged to a
    trial score before pairwise ranking positive trials above neutral trials.
    """
    score_pos = torch.log(mix_prob[:, 1].clamp_min(eps)) - torch.log(mix_prob[:, 0].clamp_min(eps))
    labels = labels.reshape(-1).long()
    subject_ids = subject_ids.reshape(-1)
    trial_ids = trial_ids.reshape(-1)
    sample_weight = sample_weight.reshape(-1) if sample_weight is not None else None

    loss_terms = []
    for sid in torch.unique(subject_ids):
        sub_mask = subject_ids == sid
        sub_scores = score_pos[sub_mask]
        sub_labels = labels[sub_mask]
        sub_trials = trial_ids[sub_mask]
        sub_weights = sample_weight[sub_mask] if sample_weight is not None else None

        trial_scores = []
        trial_labels = []
        trial_weights = []
        for tid in torch.unique(sub_trials):
            trial_mask = sub_trials == tid
            trial_scores.append(sub_scores[trial_mask].mean())
            trial_labels.append(sub_labels[trial_mask][0])
            if sub_weights is not None:
                trial_weights.append(sub_weights[trial_mask].mean())

        if len(trial_scores) < 2:
            continue

        trial_scores = torch.stack(trial_scores)
        trial_labels = torch.stack(trial_labels)
        pos_scores = trial_scores[trial_labels == 1]
        neg_scores = trial_scores[trial_labels == 0]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue

        pair_losses = F.softplus(float(margin) - (pos_scores[:, None] - neg_scores[None, :]))
        if trial_weights:
            trial_weights = torch.stack(trial_weights)
            pos_weights = trial_weights[trial_labels == 1]
            neg_weights = trial_weights[trial_labels == 0]
            pair_weights = torch.sqrt(pos_weights[:, None] * neg_weights[None, :]).clamp_min(eps)
            pair_losses = pair_losses * pair_weights

        if max_pairs_per_subject > 0 and pair_losses.numel() > max_pairs_per_subject:
            pair_losses = torch.topk(pair_losses.reshape(-1), k=int(max_pairs_per_subject)).values
        loss_terms.append(pair_losses.mean())

    if not loss_terms:
        return mix_prob.new_tensor(0.0)
    return torch.stack(loss_terms).mean()


def train_stage2_one_epoch(
    model: Stage2ExpertEmotionAdaptationModel,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    source_subject_weights: dict[str, float],
    num_source_subjects: int,
    lambda_expert: float = 0.5,
    lambda_mix: float = 1.0,
    lambda_diag: float = 0.02,
    diag_label_smoothing: float = 0.0,
    lambda_mmd: float = 0.0003,
    lambda_cond_mmd: float = 0.0,
    cond_mmd_warmup_epochs: int = 5,
    cond_mmd_conf_threshold: float = 0.7,
    cond_mmd_detach_target_prob: bool = True,
    lambda_subject: float = 0.0003,
    grl_subject: float = 0.001,
    lambda_ent: float = 0.0,
    lambda_mdc: float = 0.0,
    lambda_rank: float = 0.0,
    rank_margin: float = 0.2,
    rank_warmup_epochs: int = 3,
    rank_max_pairs_per_subject: int = 128,
    current_epoch: int = 1,
    source_de_mu: Optional[dict[str, torch.Tensor]] = None,
    source_de_std: Optional[dict[str, torch.Tensor]] = None,
    source_bio_mu: Optional[dict[str, torch.Tensor]] = None,
    source_bio_std: Optional[dict[str, torch.Tensor]] = None,
    target_de_mu: Optional[dict[str, torch.Tensor]] = None,
    target_de_std: Optional[dict[str, torch.Tensor]] = None,
    target_bio_mu: Optional[dict[str, torch.Tensor]] = None,
    target_bio_std: Optional[dict[str, torch.Tensor]] = None,
) -> dict:
    model.train()
    target_iter = cycle(target_loader)
    totals = defaultdict(float)
    total_n = 0
    all_mix_preds = []
    all_expert_preds = []
    all_emo_labels = []
    all_diag_preds = []
    all_diag_labels = []
    all_domain_preds = []
    all_domain_labels = []
    cond_mmd_stat_keys = [
        "cond_mmd_valid_classes",
        "cond_mmd_target_conf_mean",
        "cond_mmd_target_conf_ratio",
        "cond_mmd_target_pseudo_pos_rate",
        "cond_mmd_source_class0_weight",
        "cond_mmd_source_class1_weight",
        "cond_mmd_target_class0_weight",
        "cond_mmd_target_class1_weight",
    ]
    default_cond_mmd_stats = {key: 0.0 for key in cond_mmd_stat_keys}
    default_cond_mmd_stats["cond_mmd_valid_classes"] = 0
    arch_stat_keys: set[str] = set()

    pbar = tqdm(source_loader, desc="Stage2-Train", leave=False)
    for source_batch in pbar:
        target_batch = next(target_iter)
        source_batch = move_batch_to_device(source_batch, device)
        target_batch = move_batch_to_device(target_batch, device)

        optimizer.zero_grad(set_to_none=True)
        source_rel_kwargs = get_subject_relative_kwargs(
            source_batch,
            device,
            dtype=source_batch["de_feat"].dtype,
            de_mu=source_de_mu,
            de_std=source_de_std,
            bio_mu=source_bio_mu,
            bio_std=source_bio_std,
        )
        target_rel_kwargs = get_subject_relative_kwargs(
            target_batch,
            device,
            dtype=target_batch["de_feat"].dtype,
            de_mu=target_de_mu,
            de_std=target_de_std,
            bio_mu=target_bio_mu,
            bio_std=target_bio_std,
        )
        source_out = model(
            source_batch["x"],
            source_batch["de_feat"],
            lambda_subject=grl_subject,
            **source_rel_kwargs,
        )
        target_out = model(
            target_batch["x"],
            target_batch["de_feat"],
            lambda_subject=grl_subject,
            **target_rel_kwargs,
        )

        y_emo = source_batch["emotion_label"].long()
        y_diag = source_batch["diagnosis_label"].long()
        sample_weight = _subject_weight_tensor(
            source_batch["subject_id"],
            source_subject_weights,
            num_source_subjects,
            device,
        )

        loss_expert = hard_expert_emotion_loss(
            source_out["hc_logits"],
            source_out["dep_logits"],
            y_emo,
            y_diag,
            sample_weight=sample_weight,
        )
        loss_mix = mixture_emotion_nll_loss(source_out["mix_prob"], y_emo, sample_weight=sample_weight)
        loss_diag = _weighted_ce(
            source_out["diag_logits"],
            y_diag,
            sample_weight=sample_weight,
            label_smoothing=diag_label_smoothing,
        )

        domain_logits = torch.cat(
            [source_out["subject_domain_logits"], target_out["subject_domain_logits"]],
            dim=0,
        )
        domain_labels = torch.cat([source_batch["domain_id"].long(), target_batch["domain_id"].long()], dim=0)
        loss_subject_domain = F.cross_entropy(domain_logits, domain_labels)
        loss_mmd = weighted_mmd_rbf(source_out["z_mmd"], target_out["z_mmd"], source_weight=sample_weight)
        if lambda_cond_mmd > 0 and current_epoch > cond_mmd_warmup_epochs:
            loss_cond_mmd, cond_mmd_stats = soft_conditional_mmd_rbf(
                z_source=source_out["z_mmd"],
                z_target=target_out["z_mmd"],
                y_source=y_emo,
                target_prob=target_out["mix_prob"],
                source_sample_weight=sample_weight,
                num_classes=2,
                conf_threshold=cond_mmd_conf_threshold,
                detach_target_prob=cond_mmd_detach_target_prob,
            )
        else:
            loss_cond_mmd = source_out["z"].new_tensor(0.0)
            cond_mmd_stats = default_cond_mmd_stats
        loss_ent = target_entropy_loss(target_out["mix_prob"]) if lambda_ent > 0 else source_out["z"].new_tensor(0.0)
        loss_mdc = source_out["z"].new_tensor(0.0)
        if lambda_rank > 0 and current_epoch > rank_warmup_epochs:
            loss_rank = stage2_subject_ranking_loss(
                source_out["mix_prob"],
                y_emo,
                source_batch["subject_id"].long(),
                source_batch["trial_id"].long(),
                sample_weight=sample_weight,
                margin=rank_margin,
                max_pairs_per_subject=rank_max_pairs_per_subject,
            )
        else:
            loss_rank = source_out["z"].new_tensor(0.0)
        arch_stats = extract_arch_output_stats(source_out)
        arch_stat_keys.update(arch_stats.keys())

        loss = (
            lambda_expert * loss_expert
            + lambda_mix * loss_mix
            + lambda_diag * loss_diag
            + lambda_mmd * loss_mmd
            + lambda_cond_mmd * loss_cond_mmd
            + lambda_subject * loss_subject_domain
            + lambda_ent * loss_ent
            + lambda_mdc * loss_mdc
            + lambda_rank * loss_rank
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mix_pred = source_out["mix_prob"].argmax(dim=1)
            diag_pred = source_out["diag_logits"].argmax(dim=1)

            selected_logits = torch.empty_like(source_out["hc_logits"])
            hc_mask = y_diag == 1
            dep_mask = y_diag == 0
            selected_logits[hc_mask] = source_out["hc_logits"][hc_mask]
            selected_logits[dep_mask] = source_out["dep_logits"][dep_mask]
            expert_pred = selected_logits.argmax(dim=1)

            domain_pred = domain_logits.argmax(dim=1)

        bsz = source_batch["x"].size(0)
        total_n += bsz
        totals["loss"] += float(loss.item()) * bsz
        totals["loss_expert"] += float(loss_expert.item()) * bsz
        totals["loss_mix"] += float(loss_mix.item()) * bsz
        totals["loss_diag"] += float(loss_diag.item()) * bsz
        totals["loss_mmd"] += float(loss_mmd.item()) * bsz
        totals["loss_cond_mmd"] += float(loss_cond_mmd.item()) * bsz
        for key in cond_mmd_stat_keys:
            totals[key] += float(cond_mmd_stats.get(key, 0.0)) * bsz
        totals["loss_subject_domain"] += float(loss_subject_domain.item()) * bsz
        totals["loss_ent"] += float(loss_ent.item()) * bsz
        totals["loss_mdc"] += float(loss_mdc.item()) * bsz
        totals["loss_rank"] += float(loss_rank.item()) * bsz
        totals["mix_correct"] += int((mix_pred == y_emo).sum().item())
        totals["expert_correct"] += int((expert_pred == y_emo).sum().item())
        totals["diag_correct"] += int((diag_pred == y_diag).sum().item())
        totals["domain_correct"] += int((domain_pred == domain_labels).sum().item())
        totals["domain_n"] += int(domain_labels.numel())
        totals["hc_n"] += int(hc_mask.sum().item())
        totals["dep_n"] += int(dep_mask.sum().item())
        if hc_mask.any():
            totals["hc_expert_correct"] += int(
                (source_out["hc_logits"][hc_mask].argmax(dim=1) == y_emo[hc_mask]).sum().item()
            )
        if dep_mask.any():
            totals["dep_expert_correct"] += int(
                (source_out["dep_logits"][dep_mask].argmax(dim=1) == y_emo[dep_mask]).sum().item()
            )
        totals["pred_pos_mix"] += int((mix_pred == 1).sum().item())
        totals["true_pos_emo"] += int((y_emo == 1).sum().item())
        totals["pred_hc_diag"] += int((diag_pred == 1).sum().item())
        totals["true_hc_diag"] += int((y_diag == 1).sum().item())
        totals["sample_weight_sum"] += float(sample_weight.sum().item())
        for key, value in arch_stats.items():
            totals[key] += float(value) * bsz

        all_mix_preds.extend(mix_pred.detach().cpu().tolist())
        all_expert_preds.extend(expert_pred.detach().cpu().tolist())
        all_emo_labels.extend(y_emo.detach().cpu().tolist())
        all_diag_preds.extend(diag_pred.detach().cpu().tolist())
        all_diag_labels.extend(y_diag.detach().cpu().tolist())
        all_domain_preds.extend(domain_pred.detach().cpu().tolist())
        all_domain_labels.extend(domain_labels.detach().cpu().tolist())

        pbar.set_postfix(
            {
                "loss": f"{totals['loss'] / max(total_n, 1):.4f}",
                "exp": f"{totals['loss_expert'] / max(total_n, 1):.4f}",
                "mix": f"{totals['loss_mix'] / max(total_n, 1):.4f}",
                "mmd": f"{totals['loss_mmd'] / max(total_n, 1):.4f}",
                "cmmd": f"{totals['loss_cond_mmd'] / max(total_n, 1):.4f}",
                "rank": f"{totals['loss_rank'] / max(total_n, 1):.4f}",
                "acc": f"{totals['mix_correct'] / max(total_n, 1):.4f}",
                "diag": f"{totals['diag_correct'] / max(total_n, 1):.4f}",
            }
        )

    if total_n == 0:
        raise RuntimeError("Stage2 source loader is empty: total_n == 0.")

    emotion_labels = [0, 1]
    metrics = {
        "loss": totals["loss"] / total_n,
        "loss_expert": totals["loss_expert"] / total_n,
        "loss_mix": totals["loss_mix"] / total_n,
        "loss_diag": totals["loss_diag"] / total_n,
        "loss_mmd": totals["loss_mmd"] / total_n,
        "loss_cond_mmd": totals["loss_cond_mmd"] / total_n,
        "loss_subject_domain": totals["loss_subject_domain"] / total_n,
        "loss_ent": totals["loss_ent"] / total_n,
        "loss_mdc": totals["loss_mdc"] / total_n,
        "loss_rank": totals["loss_rank"] / total_n,
        "segment_acc": totals["mix_correct"] / total_n,
        "expert_acc": totals["expert_correct"] / total_n,
        "diag_acc": totals["diag_correct"] / total_n,
        "subject_domain_acc": totals["domain_correct"] / max(totals["domain_n"], 1),
        "hc_expert_acc": totals["hc_expert_correct"] / max(totals["hc_n"], 1),
        "dep_expert_acc": totals["dep_expert_correct"] / max(totals["dep_n"], 1),
        "hc_samples": totals["hc_n"],
        "dep_samples": totals["dep_n"],
        "pred_pos_rate": totals["pred_pos_mix"] / total_n,
        "true_pos_rate": totals["true_pos_emo"] / total_n,
        "pred_hc_rate": totals["pred_hc_diag"] / total_n,
        "true_hc_rate": totals["true_hc_diag"] / total_n,
        "sample_weight_mean": totals["sample_weight_sum"] / total_n,
    }
    for key in cond_mmd_stat_keys:
        metrics[key] = totals[key] / total_n
    for key in sorted(arch_stat_keys):
        metrics[key] = totals[key] / total_n

    metrics["emotion_macro_f1"] = f1_score(
        all_emo_labels,
        all_mix_preds,
        average="macro",
        labels=emotion_labels,
        zero_division=0,
    )
    metrics["expert_macro_f1"] = f1_score(
        all_emo_labels,
        all_expert_preds,
        average="macro",
        labels=emotion_labels,
        zero_division=0,
    )
    metrics["diag_macro_f1"] = f1_score(
        all_diag_labels,
        all_diag_preds,
        average="macro",
        labels=[0, 1],
        zero_division=0,
    )
    domain_label_set = sorted(set(all_domain_labels) | set(all_domain_preds))
    metrics["subject_domain_macro_f1"] = f1_score(
        all_domain_labels,
        all_domain_preds,
        average="macro",
        labels=domain_label_set,
        zero_division=0,
    )
    metrics["emotion_confusion_matrix"] = confusion_matrix(
        all_emo_labels,
        all_mix_preds,
        labels=emotion_labels,
    )
    metrics["expert_confusion_matrix"] = confusion_matrix(
        all_emo_labels,
        all_expert_preds,
        labels=emotion_labels,
    )
    metrics["diag_confusion_matrix"] = confusion_matrix(
        all_diag_labels,
        all_diag_preds,
        labels=[0, 1],
    )
    p, r, f1, support = precision_recall_fscore_support(
        all_emo_labels,
        all_mix_preds,
        labels=emotion_labels,
        zero_division=0,
    )
    metrics["per_class_emo"] = {
        int(cls): {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(emotion_labels)
    }

    # V1-compatible aliases used by history/summary code.
    metrics["acc_2cls"] = metrics["segment_acc"]
    metrics["acc"] = metrics["segment_acc"]
    metrics["macro_f1"] = metrics["emotion_macro_f1"]
    metrics["emotion_acc"] = metrics["segment_acc"]
    return metrics


def compute_trial_level_metrics(segment_records: list[dict], threshold: float = 0.5, vote_method: str = "hard") -> dict:
    trial_dict = defaultdict(list)
    for row in segment_records:
        trial_dict[(str(row["subject_id"]), int(row["trial_id"]))].append(row)

    trial_keys, trial_probs, trial_preds, trial_labels = [], [], [], []
    for key, records in trial_dict.items():
        probs = [float(r["prob_pos"]) for r in records]
        preds = [int(r["pred_emo"]) for r in records]
        labels = [int(r["label_emo"]) for r in records]
        mean_prob = float(np.mean(probs))
        label = int(np.bincount(labels, minlength=2).argmax())
        if vote_method == "prob":
            pred = int(mean_prob >= threshold)
        elif vote_method == "hard":
            counts = np.bincount(preds, minlength=2)
            pred = int(counts[1] > counts[0]) if counts[0] != counts[1] else int(mean_prob >= threshold)
        else:
            raise ValueError(f"Unknown vote_method={vote_method}")
        trial_keys.append(key)
        trial_probs.append(mean_prob)
        trial_preds.append(pred)
        trial_labels.append(label)

    return {
        "trial_acc": accuracy_score(trial_labels, trial_preds) if trial_labels else 0.0,
        "trial_macro_f1": f1_score(trial_labels, trial_preds, average="macro", labels=[0, 1], zero_division=0)
        if trial_labels
        else 0.0,
        "trial_confusion_matrix": confusion_matrix(trial_labels, trial_preds, labels=[0, 1])
        if trial_labels
        else np.zeros((2, 2), dtype=int),
        "trial_keys": trial_keys,
        "trial_probs": trial_probs,
        "trial_preds": trial_preds,
        "trial_labels": trial_labels,
    }


def compute_trial_topk_metrics(segment_records: list[dict], k_pos: int = 4) -> dict:
    """Validation-time V1-style per-subject soft top-K trial metrics."""

    trial_dict = defaultdict(list)
    for row in segment_records:
        subject_key = str(row.get("user_id", row["subject_id"]))
        trial_dict[(subject_key, int(row["trial_id"]))].append(row)

    trial_records = []
    for (subject_key, trial_id), records in trial_dict.items():
        probs = [float(r["prob_pos"]) for r in records]
        scores = [float(r.get("score_pos", 0.0)) for r in records]
        labels = [int(r["label_emo"]) for r in records]
        trial_records.append(
            {
                "user_id": subject_key,
                "trial_id": int(trial_id),
                "prob_pos": float(np.mean(probs)),
                "score_pos": float(np.mean(scores)),
                "label_emo": int(np.bincount(labels, minlength=2).argmax()),
            }
        )

    if not trial_records:
        return {
            "topk_trial_acc": 0.0,
            "topk_trial_macro_f1": 0.0,
            "topk_trial_confusion_matrix": np.zeros((2, 2), dtype=int),
            "topk_trial_keys": [],
            "topk_trial_preds": [],
            "topk_trial_labels": [],
            "topk_k_pos": int(k_pos),
        }

    topk_records = apply_subject_topk(trial_records, k_pos=k_pos)
    trial_keys = [(str(r["user_id"]), int(r["trial_id"])) for r in topk_records]
    trial_preds = [int(r["Emotion_label"]) for r in topk_records]
    trial_labels = [int(r["label_emo"]) for r in topk_records]
    return {
        "topk_trial_acc": accuracy_score(trial_labels, trial_preds),
        "topk_trial_macro_f1": f1_score(
            trial_labels,
            trial_preds,
            average="macro",
            labels=[0, 1],
            zero_division=0,
        ),
        "topk_trial_confusion_matrix": confusion_matrix(trial_labels, trial_preds, labels=[0, 1]),
        "topk_trial_keys": trial_keys,
        "topk_trial_preds": trial_preds,
        "topk_trial_labels": trial_labels,
        "topk_k_pos": int(k_pos),
    }


@torch.no_grad()
def validate_stage2_trial_level(
    model: Stage2ExpertEmotionAdaptationModel,
    val_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    k_pos: int = 4,
    lambda_diag: float = 0.02,
    diag_label_smoothing: float = 0.0,
    val_de_mu: Optional[dict[str, torch.Tensor]] = None,
    val_de_std: Optional[dict[str, torch.Tensor]] = None,
    val_bio_mu: Optional[dict[str, torch.Tensor]] = None,
    val_bio_std: Optional[dict[str, torch.Tensor]] = None,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_preds, all_labels = [], []
    all_diag_preds, all_diag_labels = [], []
    segment_records = []

    for batch in tqdm(val_loader, desc="Stage2-Val", leave=False):
        batch = move_batch_to_device(batch, device)
        val_rel_kwargs = get_subject_relative_kwargs(
            batch,
            device,
            dtype=batch["de_feat"].dtype,
            de_mu=val_de_mu,
            de_std=val_de_std,
            bio_mu=val_bio_mu,
            bio_std=val_bio_std,
        )
        out = model(batch["x"], batch["de_feat"], lambda_subject=0.0, **val_rel_kwargs)
        y_emo = batch["emotion_label"].long()
        y_diag = batch["diagnosis_label"].long()

        loss_mix = mixture_emotion_nll_loss(out["mix_prob"], y_emo)
        loss_diag = _weighted_ce(
            out["diag_logits"],
            y_diag,
            label_smoothing=diag_label_smoothing,
        )
        loss = loss_mix + float(lambda_diag) * loss_diag
        bsz = batch["x"].size(0)
        total_loss += float(loss.item()) * bsz
        total_n += bsz

        mix_prob = out["mix_prob"].clamp_min(1e-8)
        prob_pos = mix_prob[:, 1]
        score_pos = torch.log(mix_prob[:, 1]) - torch.log(mix_prob[:, 0])
        pred = out["mix_prob"].argmax(dim=1)
        diag_pred = out["diag_logits"].argmax(dim=1)
        all_preds.extend(pred.detach().cpu().tolist())
        all_labels.extend(y_emo.detach().cpu().tolist())
        all_diag_preds.extend(diag_pred.detach().cpu().tolist())
        all_diag_labels.extend(y_diag.detach().cpu().tolist())

        for i in range(bsz):
            segment_records.append(
                {
                    "subject_id": int(batch["subject_id"][i].detach().cpu()),
                    "user_id": str(batch["user_id"][i]) if "user_id" in batch else str(int(batch["subject_id"][i].detach().cpu())),
                    "trial_id": int(batch["trial_id"][i].detach().cpu()),
                    "prob_pos": float(prob_pos[i].detach().cpu()),
                    "score_pos": float(score_pos[i].detach().cpu()),
                    "pred_emo": int(pred[i].detach().cpu()),
                    "label_emo": int(y_emo[i].detach().cpu()),
                    "pred_diag": int(diag_pred[i].detach().cpu()),
                    "label_diag": int(y_diag[i].detach().cpu()),
                }
            )

    segment_acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    segment_macro_f1 = f1_score(all_labels, all_preds, average="macro", labels=[0, 1], zero_division=0) if all_labels else 0.0
    segment_cm = confusion_matrix(all_labels, all_preds, labels=[0, 1]) if all_labels else np.zeros((2, 2), dtype=int)
    p, r, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=[0, 1],
        zero_division=0,
    )
    per_class = {
        int(cls): {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate([0, 1])
    }
    trial_metrics = compute_trial_level_metrics(segment_records, threshold=threshold, vote_method="hard")
    topk_trial_metrics = compute_trial_topk_metrics(segment_records, k_pos=k_pos)

    metrics = {
        "loss": total_loss / max(total_n, 1),
        "segment_acc": segment_acc,
        "segment_macro_f1": segment_macro_f1,
        "segment_confusion_matrix": segment_cm,
        "per_class_emo": per_class,
        "diag_acc": accuracy_score(all_diag_labels, all_diag_preds) if all_diag_labels else 0.0,
        "diag_macro_f1": f1_score(all_diag_labels, all_diag_preds, average="macro", labels=[0, 1], zero_division=0)
        if all_diag_labels
        else 0.0,
        "segment_records": segment_records,
    }
    metrics.update(trial_metrics)
    metrics.update(topk_trial_metrics)

    # V1-compatible aliases for checkpoint criteria and summary tables.
    metrics["emotion_acc"] = metrics["segment_acc"]
    metrics["emotion_macro_f1"] = metrics["segment_macro_f1"]
    metrics["acc"] = metrics["segment_acc"]
    metrics["macro_f1"] = metrics["segment_macro_f1"]
    metrics["trial_emotion_acc"] = metrics["trial_acc"]
    metrics["trial_emotion_macro_f1"] = metrics["trial_macro_f1"]
    metrics["topk_trial_emotion_acc"] = metrics["topk_trial_acc"]
    metrics["topk_trial_emotion_macro_f1"] = metrics["topk_trial_macro_f1"]
    return metrics


def metric_value(metrics: dict, key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return default
    return float(value)


def make_compare_key(metrics: dict, criteria) -> tuple:
    key = []
    for metric_name, mode in criteria:
        value = metric_value(metrics, metric_name, default=None)
        if value is None:
            value = -1e18 if mode == "max" else 1e18
        key.append(value if mode == "max" else -value)
    return tuple(key)


def is_better_by_criteria(current_metrics: dict, best_metrics: Optional[dict], criteria, eps: float = 1e-12) -> bool:
    if best_metrics is None:
        return True

    cur_key = make_compare_key(current_metrics, criteria)
    best_key = make_compare_key(best_metrics, criteria)
    for cur_value, best_value in zip(cur_key, best_key):
        if cur_value > best_value + eps:
            return True
        if cur_value < best_value - eps:
            return False
    return False


def format_criteria(criteria) -> str:
    return " -> ".join([f"{name}({mode})" for name, mode in criteria])


def is_better(current: dict, best: Optional[dict]) -> bool:
    if best is None:
        return True
    criteria = [
        ("trial_macro_f1", "max"),
        ("trial_acc", "max"),
        ("segment_macro_f1", "max"),
        ("loss", "min"),
    ]
    return is_better_by_criteria(current, best, criteria)


def flatten_metrics_for_csv(prefix: str, metrics: dict) -> dict:
    row = {}
    for key, value in metrics.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, (int, float, np.integer, np.floating)):
            row[name] = float(value)
        elif isinstance(value, np.ndarray) and value.ndim == 2 and "confusion_matrix" in key:
            for i in range(value.shape[0]):
                for j in range(value.shape[1]):
                    row[f"{name}_{i}_{j}"] = float(value[i, j])
        elif isinstance(value, dict) and "per_class" in key:
            for cls, sub in value.items():
                for sub_key, sub_value in sub.items():
                    row[f"{name}_class{cls}_{sub_key}"] = float(sub_value)
    return row


def save_best_checkpoint(
    save_dir: str | Path,
    fold: int,
    best_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    train_subjects,
    val_subjects,
    criteria,
    config_dict: dict,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    extra_state: Optional[dict] = None,
    filename: Optional[str] = None,
) -> tuple[str, str, str]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = "stage2_best.pt" if best_name == "combined" else f"stage2_best_{best_name}_fold{fold}.pt"
    ckpt_path = save_dir / filename
    json_path = save_dir / f"{Path(filename).stem}_metrics.json"
    csv_path = save_dir / f"{Path(filename).stem}_summary.csv"

    ckpt = {
        "best_name": best_name,
        "epoch": epoch,
        "criteria": criteria,
        "criteria_readable": format_criteria(criteria),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": to_jsonable(train_metrics),
        "val_metrics": to_jsonable(val_metrics),
        "train_subjects": to_jsonable(train_subjects),
        "val_subjects": to_jsonable(val_subjects),
        "config": to_jsonable(config_dict),
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if extra_state is not None:
        ckpt["extra_state"] = to_jsonable(extra_state)
    torch.save(ckpt, ckpt_path)

    save_json(
        json_path,
        {
            "best_name": best_name,
            "epoch": epoch,
            "criteria": criteria,
            "criteria_readable": format_criteria(criteria),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "config": config_dict,
            "extra_state": extra_state or {},
            "checkpoint_path": str(ckpt_path),
        },
    )
    row = {
        "best_name": best_name,
        "epoch": epoch,
        "criteria": format_criteria(criteria),
        "checkpoint_path": str(ckpt_path),
    }
    row.update(flatten_metrics_for_csv("train", train_metrics))
    row.update(flatten_metrics_for_csv("val", val_metrics))
    pd.DataFrame([row]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return str(ckpt_path), str(json_path), str(csv_path)


def save_final_checkpoint(
    save_dir: str | Path,
    stage_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    train_subjects,
    val_subjects,
    config_dict: dict,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    extra_state: Optional[dict] = None,
) -> tuple[str, str, str]:
    """Save the final epoch snapshot without marking it as a best model."""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{stage_name}_final.pt"
    json_path = save_dir / f"{stage_name}_final_metrics.json"
    csv_path = save_dir / f"{stage_name}_final_summary.csv"

    ckpt = {
        "best_name": "final",
        "is_best": False,
        "epoch": epoch,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": to_jsonable(train_metrics),
        "val_metrics": to_jsonable(val_metrics),
        "train_subjects": to_jsonable(train_subjects),
        "val_subjects": to_jsonable(val_subjects),
        "config": to_jsonable(config_dict),
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if extra_state is not None:
        ckpt["extra_state"] = to_jsonable(extra_state)
    torch.save(ckpt, ckpt_path)

    save_json(
        json_path,
        {
            "best_name": "final",
            "is_best": False,
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "config": config_dict,
            "extra_state": extra_state or {},
            "checkpoint_path": str(ckpt_path),
        },
    )
    row = {
        "best_name": "final",
        "is_best": False,
        "epoch": epoch,
        "checkpoint_path": str(ckpt_path),
    }
    row.update(flatten_metrics_for_csv("train", train_metrics))
    row.update(flatten_metrics_for_csv("val", val_metrics))
    pd.DataFrame([row]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return str(ckpt_path), str(json_path), str(csv_path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    path = Path(path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def print_checkpoint_info(path: str | Path, ckpt: dict, prefix: str = "[checkpoint]") -> None:
    path = Path(path)
    config_dict = ckpt.get("config", {})
    extra_state = ckpt.get("extra_state", {})
    val_metrics = ckpt.get("val_metrics", {})
    train_metrics = ckpt.get("train_metrics", {})
    saved_time = "unknown"
    if path.exists():
        saved_time = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    def _fmt(metrics: dict, key: str) -> str:
        value = metrics.get(key, None)
        if value is None:
            return "NA"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    print(
        f"{prefix} path={path} | saved={saved_time} | "
        f"best_name={ckpt.get('best_name', extra_state.get('best_name', 'NA'))} | "
        f"epoch={ckpt.get('epoch', 'NA')}"
    )
    print(f"{prefix} criteria={ckpt.get('criteria_readable', format_criteria(ckpt.get('criteria', [])))}")
    print(
        f"{prefix} val: topk_f1={_fmt(val_metrics, 'topk_trial_macro_f1')} "
        f"trial_f1={_fmt(val_metrics, 'trial_macro_f1')} "
        f"emotion_f1={_fmt(val_metrics, 'emotion_macro_f1')} "
        f"diag_f1={_fmt(val_metrics, 'diag_macro_f1')} "
        f"loss={_fmt(val_metrics, 'loss')}"
    )
    print(
        f"{prefix} train: loss={_fmt(train_metrics, 'loss')} "
        f"emotion_loss={_fmt(train_metrics, 'emotion_loss')} "
        f"mix_loss={_fmt(train_metrics, 'mix_loss')} "
        f"diag_loss={_fmt(train_metrics, 'diag_loss')}"
    )
    print(
        f"{prefix} config: lr2={config_dict.get('lr_stage2', 'NA')} "
        f"weight_decay={config_dict.get('weight_decay', 'NA')} "
        f"dropout={config_dict.get('dropout', 'NA')} "
        f"lambda_expert={config_dict.get('lambda_expert', 'NA')} "
        f"lambda_mix={config_dict.get('lambda_mix', 'NA')} "
        f"lambda_diag={config_dict.get('lambda_diag', 'NA')} "
        f"stage2_lambda_mmd={config_dict.get('stage2_lambda_mmd', 'NA')}"
    )


def apply_subject_topk(trial_records: list[dict], k_pos: int = 4) -> list[dict]:
    """Apply V1-style per-subject soft top-K prediction on trial scores."""

    k_pos = 4 if int(k_pos) <= 0 else int(k_pos)
    by_subject = defaultdict(list)
    for record in trial_records:
        by_subject[str(record["user_id"])].append(dict(record))

    results = []
    for _subject_id, records in by_subject.items():
        records = sorted(records, key=lambda x: float(x.get("score_pos", 0.0)), reverse=True)
        cur_k = int(k_pos) if len(records) == 8 else max(1, int(round(len(records) * 0.5)))
        cur_k = max(0, min(cur_k, len(records)))
        positive_keys = {(r["user_id"], r["trial_id"]) for r in records[:cur_k]}

        for record in records:
            record["Emotion_label"] = 1 if (record["user_id"], record["trial_id"]) in positive_keys else 0
            results.append(record)

    return sorted(results, key=lambda x: (str(x["user_id"]), int(x["trial_id"])))


def build_stage2_best_trackers() -> dict:
    def tracker(criteria):
        return {
            "criteria": criteria,
            "best_metrics": None,
            "best_train_metrics": None,
            "best_epoch": None,
            "checkpoint_path": None,
            "metrics_json_path": None,
            "summary_csv_path": None,
        }

    return {
        "combined": tracker(
            [
                ("topk_trial_macro_f1", "max"),
                ("topk_trial_acc", "max"),
                ("trial_macro_f1", "max"),
                ("trial_acc", "max"),
                ("emotion_macro_f1", "max"),
                ("emotion_acc", "max"),
                ("diag_macro_f1", "max"),
                ("diag_acc", "max"),
                ("loss", "min"),
            ]
        ),
        "topk_trial_f1": tracker(
            [
                ("topk_trial_macro_f1", "max"),
                ("topk_trial_acc", "max"),
                ("trial_macro_f1", "max"),
                ("emotion_macro_f1", "max"),
                ("loss", "min"),
            ]
        ),
        "topk_trial_acc": tracker(
            [
                ("topk_trial_acc", "max"),
                ("topk_trial_macro_f1", "max"),
                ("trial_acc", "max"),
                ("emotion_acc", "max"),
                ("loss", "min"),
            ]
        ),
        "trial_f1": tracker(
            [
                ("trial_macro_f1", "max"),
                ("trial_acc", "max"),
                ("emotion_macro_f1", "max"),
                ("loss", "min"),
            ]
        ),
        "trial_acc": tracker(
            [
                ("trial_acc", "max"),
                ("trial_macro_f1", "max"),
                ("emotion_acc", "max"),
                ("loss", "min"),
            ]
        ),
        "segment_emo_f1": tracker(
            [
                ("emotion_macro_f1", "max"),
                ("emotion_acc", "max"),
                ("trial_macro_f1", "max"),
                ("loss", "min"),
            ]
        ),
        "diag_f1": tracker(
            [
                ("diag_macro_f1", "max"),
                ("diag_acc", "max"),
                ("trial_macro_f1", "max"),
                ("loss", "min"),
            ]
        ),
        "loss": tracker(
            [
                ("loss", "min"),
                ("trial_macro_f1", "max"),
                ("emotion_macro_f1", "max"),
            ]
        ),
    }


@torch.no_grad()
def predict_test_trial_level(
    model: Stage2ExpertEmotionAdaptationModel,
    test_csv: str,
    domain_mapping: dict,
    device: torch.device,
    save_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    threshold: float = 0.5,
    vote_method: str = "prob",
    k_pos: int = 4,
    normalize: bool = True,
) -> pd.DataFrame:
    dataset = UnlabeledTargetDataset(test_csv, domain_mapping=domain_mapping, root=ROOT, normalize=normalize)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dict_collate,
    )
    test_de_mu = test_de_std = None
    test_bio_mu = test_bio_std = None
    if getattr(model, "use_subject_relative_de", False) or getattr(model, "use_subject_relative_bio", False):
        print("[test] computing subject-relative test baselines from test windows...")
        if getattr(model, "use_subject_relative_de", False):
            test_de_mu, test_de_std = compute_subject_de_baselines(dataset, key_field="target_key")
        if getattr(model, "use_subject_relative_bio", False):
            test_bio_mu, test_bio_std = compute_subject_bio_baselines(
                model,
                loader,
                device,
                subject_de_mu=test_de_mu,
                subject_de_std=test_de_std,
                key_field="target_key",
            )

    model.eval()
    rows = []
    for batch in tqdm(loader, desc="Test-Predict", leave=False):
        batch = move_batch_to_device(batch, device)
        test_rel_kwargs = get_subject_relative_kwargs(
            batch,
            device,
            dtype=batch["de_feat"].dtype,
            de_mu=test_de_mu,
            de_std=test_de_std,
            bio_mu=test_bio_mu,
            bio_std=test_bio_std,
        )
        out = model(batch["x"], batch["de_feat"], lambda_subject=0.0, **test_rel_kwargs)
        mix_prob = out["mix_prob"].clamp_min(1e-8)
        prob_pos_tensor = mix_prob[:, 1]
        score_pos_tensor = torch.log(mix_prob[:, 1]) - torch.log(mix_prob[:, 0])
        prob_pos = prob_pos_tensor.detach().cpu().tolist()
        score_pos = score_pos_tensor.detach().cpu().tolist()
        pred_window = [int(p >= threshold) for p in prob_pos]
        for i, p in enumerate(prob_pos):
            rows.append(
                {
                    "user_id": batch["user_id"][i],
                    "trial_id": int(batch["trial_id"][i].detach().cpu()),
                    "prob_window": float(p),
                    "score_window": float(score_pos[i]),
                    "pred_window": int(pred_window[i]),
                }
            )
    pred_df = pd.DataFrame(rows)
    trial_df = (
        pred_df.groupby(["user_id", "trial_id"], as_index=False)
        .agg(
            prob_pos=("prob_window", "mean"),
            score_pos=("score_window", "mean"),
            hard_mean=("pred_window", "mean"),
            n_windows=("prob_window", "count"),
        )
        .sort_values(["user_id", "trial_id"])
    )
    trial_df["trial_prob"] = trial_df["prob_pos"]
    trial_df["pred_soft_threshold"] = (trial_df["prob_pos"] >= threshold).astype(int)
    trial_df["pred_hard_threshold"] = (trial_df["hard_mean"] >= 0.5).astype(int)

    topk_records = apply_subject_topk(
        trial_df[["user_id", "trial_id", "prob_pos", "score_pos"]].to_dict("records"),
        k_pos=k_pos,
    )
    topk_df = pd.DataFrame(topk_records)[["user_id", "trial_id", "Emotion_label"]].rename(
        columns={"Emotion_label": "pred_soft_topk"}
    )
    trial_df = trial_df.merge(topk_df, on=["user_id", "trial_id"], how="left")
    trial_df["pred_soft_topk"] = trial_df["pred_soft_topk"].fillna(0).astype(int)

    if vote_method == "hard":
        trial_df["Emotion_label"] = trial_df["pred_hard_threshold"]
    elif vote_method in {"soft_topk", "topk"}:
        trial_df["Emotion_label"] = trial_df["pred_soft_topk"]
    elif vote_method in {"prob", "soft_threshold"}:
        trial_df["Emotion_label"] = trial_df["pred_soft_threshold"]
    else:
        raise ValueError(f"Unknown vote_method={vote_method}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(save_dir / "test_window_probs.csv", index=False, encoding="utf-8-sig")
    trial_df.to_csv(save_dir / "test_trial_probs.csv", index=False, encoding="utf-8-sig")

    sub_soft_threshold = trial_df[["user_id", "trial_id", "pred_soft_threshold"]].copy()
    sub_soft_threshold.rename(columns={"pred_soft_threshold": "Emotion_label"}, inplace=True)
    sub_soft_threshold.to_csv(save_dir / "submission_soft_threshold.csv", index=False, encoding="utf-8-sig")

    sub_soft_topk = trial_df[["user_id", "trial_id", "pred_soft_topk"]].copy()
    sub_soft_topk.rename(columns={"pred_soft_topk": "Emotion_label"}, inplace=True)
    sub_soft_topk.to_csv(save_dir / "submission_soft_topk.csv", index=False, encoding="utf-8-sig")

    sub_hard_threshold = trial_df[["user_id", "trial_id", "pred_hard_threshold"]].copy()
    sub_hard_threshold.rename(columns={"pred_hard_threshold": "Emotion_label"}, inplace=True)
    sub_hard_threshold.to_csv(save_dir / "submission_hard_threshold.csv", index=False, encoding="utf-8-sig")

    submission = trial_df[["user_id", "trial_id", "Emotion_label"]].copy()
    submission.to_csv(save_dir / "submission_expert_ssas.csv", index=False, encoding="utf-8-sig")
    n_topk = int(sub_soft_topk["Emotion_label"].sum())
    n_selected = int(submission["Emotion_label"].sum())
    print(
        f"[test] soft-topk={k_pos}: {n_topk}/{len(sub_soft_topk)} positive; "
        f"selected={vote_method}: {n_selected}/{len(submission)} positive"
    )
    return trial_df


def build_stage2_model_from_checkpoint(ckpt: dict, device: torch.device) -> tuple[Stage2ExpertEmotionAdaptationModel, dict]:
    config_dict = ckpt.get("config", {}) or {}
    extra_state = ckpt.get("extra_state", {}) or {}
    domain_mapping = extra_state.get("domain_mapping") or ckpt.get("domain_mapping")
    if domain_mapping is None:
        raise KeyError("Checkpoint is missing domain_mapping; cannot rebuild target domain ids for test prediction.")

    use_biomarkers = not bool(config_dict.get("no_biomarkers", False))
    use_subject_relative_de = not bool(config_dict.get("no_subject_relative_de", False))
    use_subject_relative_bio = (
        (not bool(config_dict.get("no_subject_relative_bio", False)))
        and use_biomarkers
    )

    def _cfg_float(key: str, default: float) -> float:
        value = config_dict.get(key, default)
        return float(default if value is None else value)

    def _cfg_int(key: str, default: int) -> int:
        value = config_dict.get(key, default)
        return int(default if value is None else value)

    def _cfg_bool(key: str, default: bool) -> bool:
        value = config_dict.get(key, default)
        return bool(default if value is None else value)

    model = Stage2ExpertEmotionAdaptationModel(
        num_domains=int(domain_mapping["num_domains"]),
        sfreq=_cfg_float("sfreq", 250.0),
        topk=_cfg_int("topk", 8),
        dropout=_cfg_float("dropout", 0.35),
        use_biomarkers=use_biomarkers,
        biomarker_dim=_cfg_int("biomarker_dim", 57),
        use_subject_relative_de=use_subject_relative_de,
        use_subject_relative_bio=use_subject_relative_bio,
        bio_abs_scale=_cfg_float("bio_abs_scale", 0.3),
        relative_eps=_cfg_float("relative_eps", 1e-6),
        de_num_bands=_cfg_int("de_num_bands", 5),
        shared_mix_alpha=_cfg_float("shared_mix_alpha", 0.5),
        router_temperature=_cfg_float("router_temperature", 1.0),
        shared_expert_trunk=_cfg_bool("shared_expert_trunk", False),
        expert_hidden_dim=_cfg_int("expert_hidden_dim", 64),
        fusion_mode=config_dict.get("fusion_mode") or "concat",
        bio_gate_init=_cfg_float("bio_gate_init", -2.0),
        core_gate_init=_cfg_float("core_gate_init", 0.0),
        temporal_gate_init=_cfg_float("temporal_gate_init", 0.0),
        graph_gate_init=_cfg_float("graph_gate_init", 0.0),
        learnable_graph=_cfg_bool("learnable_graph", False),
        learnable_graph_rank=_cfg_int("learnable_graph_rank", 4),
        graph_mix_logit_init=_cfg_float("graph_mix_logit_init", 2.0),
        learnable_graph_dropout=_cfg_float("learnable_graph_dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, domain_mapping


@torch.no_grad()
def ensemble_test_predictions(
    model_paths: list[str | Path],
    test_csv: str,
    device: torch.device,
    save_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    threshold: float = 0.5,
    vote_method: str = "soft_topk",
    k_pos: int = 4,
    normalize: bool = True,
) -> pd.DataFrame:
    model_paths = [Path(p) for p in model_paths if p and Path(p).exists()]
    if not model_paths:
        raise ValueError("No valid checkpoint paths for V4 test ensemble.")

    save_dir = Path(save_dir)
    per_model_dir = save_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    all_trial_probs = []

    for i, ckpt_path in enumerate(model_paths):
        print(f"\n[V4 ensemble] model {i + 1}/{len(model_paths)}: {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path, device)
        print_checkpoint_info(ckpt_path, ckpt, prefix=f"[V4 ensemble ckpt {i + 1}/{len(model_paths)}]")
        model, domain_mapping = build_stage2_model_from_checkpoint(ckpt, device)
        model_dir = per_model_dir / f"model_{i:02d}_{ckpt_path.stem}"
        trial_df = predict_test_trial_level(
            model,
            test_csv,
            domain_mapping,
            device,
            model_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            threshold=threshold,
            vote_method=vote_method,
            k_pos=k_pos,
            normalize=normalize,
        )
        trial_df = trial_df[["user_id", "trial_id", "prob_pos", "score_pos"]].copy()
        trial_df.rename(
            columns={
                "prob_pos": f"prob_model_{i}",
                "score_pos": f"score_model_{i}",
            },
            inplace=True,
        )
        all_trial_probs.append(trial_df)

    merged = all_trial_probs[0]
    for df in all_trial_probs[1:]:
        merged = merged.merge(df, on=["user_id", "trial_id"], how="inner")

    prob_cols = [c for c in merged.columns if c.startswith("prob_model_")]
    score_cols = [c for c in merged.columns if c.startswith("score_model_")]
    merged["prob_pos_ensemble"] = merged[prob_cols].mean(axis=1)
    merged["score_pos_ensemble"] = merged[score_cols].mean(axis=1)
    merged["n_models"] = len(prob_cols)
    merged["pred_soft_threshold"] = (merged["prob_pos_ensemble"] >= threshold).astype(int)

    topk_records = apply_subject_topk(
        merged[["user_id", "trial_id", "prob_pos_ensemble", "score_pos_ensemble"]]
        .rename(columns={"prob_pos_ensemble": "prob_pos", "score_pos_ensemble": "score_pos"})
        .to_dict("records"),
        k_pos=k_pos,
    )
    topk_df = pd.DataFrame(topk_records)[["user_id", "trial_id", "Emotion_label"]].rename(
        columns={"Emotion_label": "pred_soft_topk"}
    )
    merged = merged.merge(topk_df, on=["user_id", "trial_id"], how="left")
    merged["pred_soft_topk"] = merged["pred_soft_topk"].fillna(0).astype(int)

    if vote_method in {"soft_topk", "topk"}:
        merged["Emotion_label"] = merged["pred_soft_topk"]
    elif vote_method in {"prob", "soft_threshold", "hard"}:
        merged["Emotion_label"] = merged["pred_soft_threshold"]
    else:
        raise ValueError(f"Unknown vote_method={vote_method}")

    save_dir.mkdir(parents=True, exist_ok=True)
    probs_path = save_dir / "test_ensemble_probs.csv"
    merged.to_csv(probs_path, index=False, encoding="utf-8-sig")

    sub_soft_threshold = merged[["user_id", "trial_id", "pred_soft_threshold"]].copy()
    sub_soft_threshold.rename(columns={"pred_soft_threshold": "Emotion_label"}, inplace=True)
    sub_soft_threshold.to_csv(save_dir / "submission_soft_threshold_ensemble.csv", index=False, encoding="utf-8-sig")

    sub_soft_topk = merged[["user_id", "trial_id", "pred_soft_topk"]].copy()
    sub_soft_topk.rename(columns={"pred_soft_topk": "Emotion_label"}, inplace=True)
    sub_soft_topk.to_csv(save_dir / "submission_soft_topk_ensemble.csv", index=False, encoding="utf-8-sig")

    submission = merged[["user_id", "trial_id", "Emotion_label"]].copy()
    submission_path = save_dir / "submission_test_ensemble.csv"
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")

    n_total = len(submission)
    n_pos = int(submission["Emotion_label"].sum())
    print(f"\n[V4 ensemble] probabilities saved: {probs_path}")
    print(f"[V4 ensemble] submission saved: {submission_path}")
    print(f"[V4 ensemble] selected={vote_method}, positives={n_pos}/{n_total} ({100 * n_pos / max(n_total, 1):.1f}%)")
    return merged


def load_stage1_encoder_into_stage2(
    stage1_model: Stage1SSASSourceSelectionModel,
    stage2_model: Stage2ExpertEmotionAdaptationModel,
) -> None:
    missing, unexpected = stage2_model.shared_encoder.load_state_dict(
        stage1_model.shared_encoder.state_dict(),
        strict=False,
    )
    print(f"[stage2-init] loaded Stage1 encoder; missing={len(missing)}, unexpected={len(unexpected)}")


def run_one_fold(args, fold: int, repeat_index: int, rand_seed: int) -> dict:
    run_seed = config.make_run_seed(rand_seed, fold)
    set_global_seed(run_seed, deterministic=args.deterministic)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_root) / f"expert_ssas_repeat{repeat_index}_fold{fold}"
    save_dir.mkdir(parents=True, exist_ok=True)

    split = build_train_val_target_split(args.index_csv, fold=fold, n_splits=args.n_splits, seed=rand_seed)
    domain_mapping = build_domain_id_mapping(split["train_all"], split["val_all"], args.test_csv)
    save_json(save_dir / "domain_id_mapping.json", domain_mapping)
    print(
        f"[fold] repeat={repeat_index} fold={fold} seed={rand_seed} run_seed={run_seed} "
        f"source_subjects={len(split['train_all'])} val_subjects={len(split['val_all'])} domains={domain_mapping['num_domains']}"
    )
    print_arch_config(args)
    config_dict = vars(args).copy()
    config_dict.update({"fold": fold, "repeat_index": repeat_index, "rand_seed": rand_seed, "run_seed": run_seed})
    print(
        f"[Checkpoint] save_warmup_epochs={args.save_warmup_epochs}: "
        f"epochs <= {args.save_warmup_epochs} will not update best checkpoints."
    )

    generator = torch.Generator()
    generator.manual_seed(run_seed)
    source_dataset, source_loader = create_source_loader(
        args.index_csv,
        split["train_all"],
        domain_mapping,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        generator=generator,
        normalize=not args.no_normalize,
        use_label4_for_diagnosis=not args.use_raw_diagnosis_label,
    )
    val_dataset, val_loader = create_val_loader(
        args.index_csv,
        split["val_all"],
        domain_mapping,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=not args.no_normalize,
        use_label4_for_diagnosis=not args.use_raw_diagnosis_label,
    )
    target_train_dataset, target_train_loader = create_target_loader(
        val_dataset,
        domain_mapping,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_csv=args.test_csv,
        shuffle=True,
        drop_last=True,
        generator=generator,
        normalize_test=not args.no_normalize,
    )
    target_vote_dataset, target_vote_loader = create_target_loader(
        val_dataset,
        domain_mapping,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_csv=args.test_csv,
        shuffle=False,
        drop_last=False,
        normalize_test=not args.no_normalize,
    )

    use_subject_relative_de = bool(not args.no_subject_relative_de)
    use_subject_relative_bio = bool((not args.no_subject_relative_bio) and (not args.no_biomarkers))
    print(
        f"[Subject Relative] DE={use_subject_relative_de}, "
        f"bio={use_subject_relative_bio}, bio_abs_scale={args.bio_abs_scale}, eps={args.relative_eps}"
    )

    source_de_mu = source_de_std = None
    val_de_mu = val_de_std = None
    target_de_mu = target_de_std = None
    source_bio_mu = source_bio_std = None
    val_bio_mu = val_bio_std = None
    target_bio_mu = target_bio_std = None

    if use_subject_relative_de:
        print("[Subject Relative] computing source/val/target DE baselines...")
        source_de_mu, source_de_std = compute_subject_de_baselines(source_dataset, eps=args.relative_eps)
        val_de_mu, val_de_std = compute_subject_de_baselines(val_dataset, eps=args.relative_eps)
        target_de_mu, target_de_std = compute_subject_de_baselines(target_vote_dataset, eps=args.relative_eps)

    source_baseline_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dict_collate,
        worker_init_fn=seed_worker,
    )

    stage1 = Stage1SSASSourceSelectionModel(
        num_domains=domain_mapping["num_domains"],
        sfreq=args.sfreq,
        topk=args.topk,
        dropout=args.dropout,
        use_biomarkers=not args.no_biomarkers,
        biomarker_dim=args.biomarker_dim,
        use_subject_relative_de=use_subject_relative_de,
        use_subject_relative_bio=use_subject_relative_bio,
        bio_abs_scale=args.bio_abs_scale,
        relative_eps=args.relative_eps,
        de_num_bands=args.de_num_bands,
        fusion_mode=args.fusion_mode,
        bio_gate_init=args.bio_gate_init,
        core_gate_init=args.core_gate_init,
        temporal_gate_init=args.temporal_gate_init,
        graph_gate_init=args.graph_gate_init,
        learnable_graph=args.learnable_graph,
        learnable_graph_rank=args.learnable_graph_rank,
        graph_mix_logit_init=args.graph_mix_logit_init,
        learnable_graph_dropout=args.learnable_graph_dropout,
    ).to(device)
    if use_subject_relative_bio:
        print("[Subject Relative] computing source/val/target bio baselines with Stage1 encoder...")
        source_bio_mu, source_bio_std = compute_subject_bio_baselines(
            stage1,
            source_baseline_loader,
            device,
            subject_de_mu=source_de_mu,
            subject_de_std=source_de_std,
            eps=args.relative_eps,
        )
        val_bio_mu, val_bio_std = compute_subject_bio_baselines(
            stage1,
            val_loader,
            device,
            subject_de_mu=val_de_mu,
            subject_de_std=val_de_std,
            eps=args.relative_eps,
        )
        target_bio_mu, target_bio_std = compute_subject_bio_baselines(
            stage1,
            target_vote_loader,
            device,
            subject_de_mu=target_de_mu,
            subject_de_std=target_de_std,
            eps=args.relative_eps,
        )

    optimizer1 = torch.optim.AdamW(stage1.parameters(), lr=args.lr_stage1, weight_decay=args.weight_decay)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=max(args.stage1_epochs, 1))
    emotion_weights = build_emotion_class_weights(source_dataset)
    diag_weights = build_diag_class_weights(source_dataset, use_label4_for_diagnosis=not args.use_raw_diagnosis_label)

    best_stage1 = None
    stage1_best_path = None
    history1 = []
    last_stage1_train_metrics = {}
    last_stage1_val_metrics = {}
    last_stage1_epoch = 0
    for epoch in range(1, args.stage1_epochs + 1):
        train_metrics = train_stage1_one_epoch(
            stage1,
            source_loader,
            target_train_loader,
            optimizer1,
            device,
            lambda_domain=args.lambda_domain,
            lambda_mmd=args.stage1_lambda_mmd,
            lambda_emo_grl=args.lambda_emo_grl,
            grl_emo=args.grl_emo,
            lambda_diag_grl=args.lambda_diag_grl,
            grl_diag=args.grl_diag,
            lambda_weight_reg=args.lambda_weight_reg,
            emotion_class_weight=emotion_weights,
            diag_class_weight=diag_weights,
            source_de_mu=source_de_mu,
            source_de_std=source_de_std,
            source_bio_mu=source_bio_mu,
            source_bio_std=source_bio_std,
            target_de_mu=target_de_mu,
            target_de_std=target_de_std,
            target_bio_mu=target_bio_mu,
            target_bio_std=target_bio_std,
        )
        val_stage1 = validate_stage1(
            stage1,
            source_loader,
            target_train_loader,
            device,
            source_de_mu=source_de_mu,
            source_de_std=source_de_std,
            source_bio_mu=source_bio_mu,
            source_bio_std=source_bio_std,
            target_de_mu=target_de_mu,
            target_de_std=target_de_std,
            target_bio_mu=target_bio_mu,
            target_bio_std=target_bio_std,
        )
        scheduler1.step()
        row = {"epoch": epoch, "lr": optimizer1.param_groups[0]["lr"]}
        row.update(flatten_metrics_for_csv("train", train_metrics))
        row.update(flatten_metrics_for_csv("val", val_stage1))
        history1.append(row)
        last_stage1_train_metrics = copy.deepcopy(train_metrics)
        last_stage1_val_metrics = copy.deepcopy(val_stage1)
        last_stage1_epoch = epoch
        print(f"[Stage1] epoch={epoch} train_loss={train_metrics['loss']:.4f} val_domain={val_stage1['loss_domain']:.4f}")
        if epoch <= args.save_warmup_epochs:
            print(f"[Stage1] skip best checkpoint during save warmup ({epoch}/{args.save_warmup_epochs})")
        elif best_stage1 is None or val_stage1["loss_domain"] < best_stage1["val"]["loss_domain"]:
            best_stage1 = {"epoch": epoch, "train": copy.deepcopy(train_metrics), "val": copy.deepcopy(val_stage1)}
            stage1_ckpt_path = save_dir / "stage1_best.pt"
            stage1_best_path = stage1_ckpt_path
            torch.save(
                {
                    "best_name": "stage1_domain",
                    "epoch": epoch,
                    "criteria": [("loss_domain", "min"), ("domain_acc", "max"), ("loss", "min")],
                    "criteria_readable": "loss_domain(min) -> domain_acc(max) -> loss(min)",
                    "model_state_dict": copy.deepcopy(stage1.state_dict()),
                    "optimizer_state_dict": optimizer1.state_dict(),
                    "scheduler_state_dict": scheduler1.state_dict(),
                    "train_metrics": to_jsonable(train_metrics),
                    "val_metrics": to_jsonable(val_stage1),
                    "train_subjects": to_jsonable(split["train_all"]),
                    "val_subjects": to_jsonable(split["val_all"]),
                    "domain_mapping": to_jsonable(domain_mapping),
                },
                stage1_ckpt_path,
            )
            save_json(
                save_dir / "stage1_best_metrics.json",
                {
                    "best_name": "stage1_domain",
                    "epoch": epoch,
                    "criteria": [("loss_domain", "min"), ("domain_acc", "max"), ("loss", "min")],
                    "criteria_readable": "loss_domain(min) -> domain_acc(max) -> loss(min)",
                    "train_metrics": train_metrics,
                    "val_metrics": val_stage1,
                    "train_subjects": split["train_all"],
                    "val_subjects": split["val_all"],
                    "checkpoint_path": str(stage1_ckpt_path),
                },
            )
            stage1_row = {
                "best_name": "stage1_domain",
                "epoch": epoch,
                "criteria": "loss_domain(min) -> domain_acc(max) -> loss(min)",
                "checkpoint_path": str(stage1_ckpt_path),
            }
            stage1_row.update(flatten_metrics_for_csv("train", train_metrics))
            stage1_row.update(flatten_metrics_for_csv("val", val_stage1))
            pd.DataFrame([stage1_row]).to_csv(save_dir / "stage1_best_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history1).to_csv(save_dir / "stage1_history.csv", index=False, encoding="utf-8-sig")
    if last_stage1_epoch > 0:
        save_final_checkpoint(
            save_dir=save_dir,
            stage_name="stage1",
            model=stage1,
            optimizer=optimizer1,
            epoch=last_stage1_epoch,
            train_metrics=last_stage1_train_metrics,
            val_metrics=last_stage1_val_metrics,
            train_subjects=split["train_all"],
            val_subjects=split["val_all"],
            config_dict=config_dict,
            scheduler=scheduler1,
            extra_state={"domain_mapping": domain_mapping},
        )
    if stage1_best_path is not None and stage1_best_path.exists():
        ckpt = torch.load(stage1_best_path, map_location=device)
        stage1.load_state_dict(ckpt["model_state_dict"])
    elif last_stage1_epoch > 0:
        print("[Stage1] no best checkpoint saved after warmup; using final Stage1 model for voting.")

    vote_result = compute_source_weights_by_classification_voting(
        stage1,
        target_vote_loader,
        domain_mapping,
        device,
        save_dir,
        smooth=args.vote_smooth,
        vote_level=args.vote_level,
        vote_mode=args.vote_mode,
        tau_vote=args.tau_vote,
        confidence_power=args.confidence_power,
        save_topk_probs=args.save_topk_probs,
        target_de_mu=target_de_mu,
        target_de_std=target_de_std,
        target_bio_mu=target_bio_mu,
        target_bio_std=target_bio_std,
    )
    source_weight_mode = "mean-one" if args.use_mean_one_source_weights else "sum-one"
    source_weights = vote_result["weights_mean_one"] if args.use_mean_one_source_weights else vote_result["weights"]
    raw_source_weights = dict(source_weights)
    source_weights = clip_source_weights(
        source_weights,
        max_weight=args.source_weight_clip,
        blend_uniform=args.source_weight_blend_uniform,
        mean_one=args.use_mean_one_source_weights,
    )
    if args.source_weight_clip > 0 or args.source_weight_blend_uniform > 0:
        source_weight_mode = (
            f"{source_weight_mode}; clip={args.source_weight_clip}; "
            f"blend_uniform={args.source_weight_blend_uniform}"
        )
        raw_top_weights = sorted(raw_source_weights.items(), key=lambda kv: kv[1], reverse=True)[:8]
        print(f"[Stage1] raw top source weights: {raw_top_weights}")
    top_weights = sorted(source_weights.items(), key=lambda kv: kv[1], reverse=True)[:8]
    print(f"[Stage1] source weight mode={source_weight_mode}; top source weights: {top_weights}")
    stage1_prior_source_weights = dict(raw_source_weights)
    source_weight_history = [
        {
            "epoch": 0,
            "source": "stage1_voting",
            "raw_stage1_weights": raw_source_weights,
            "source_weights": source_weights,
            **summarize_source_weights(source_weights),
        }
    ]

    stage2 = Stage2ExpertEmotionAdaptationModel(
        num_domains=domain_mapping["num_domains"],
        sfreq=args.sfreq,
        topk=args.topk,
        dropout=args.dropout,
        use_biomarkers=not args.no_biomarkers,
        biomarker_dim=args.biomarker_dim,
        use_subject_relative_de=use_subject_relative_de,
        use_subject_relative_bio=use_subject_relative_bio,
        bio_abs_scale=args.bio_abs_scale,
        relative_eps=args.relative_eps,
        de_num_bands=args.de_num_bands,
        shared_mix_alpha=args.shared_mix_alpha,
        router_temperature=args.router_temperature,
        shared_expert_trunk=args.shared_expert_trunk,
        expert_hidden_dim=args.expert_hidden_dim,
        fusion_mode=args.fusion_mode,
        bio_gate_init=args.bio_gate_init,
        core_gate_init=args.core_gate_init,
        temporal_gate_init=args.temporal_gate_init,
        graph_gate_init=args.graph_gate_init,
        learnable_graph=args.learnable_graph,
        learnable_graph_rank=args.learnable_graph_rank,
        graph_mix_logit_init=args.graph_mix_logit_init,
        learnable_graph_dropout=args.learnable_graph_dropout,
    ).to(device)
    print(
        f"[Stage2 Model] shared_expert_trunk={args.shared_expert_trunk}, "
        f"expert_hidden_dim={args.expert_hidden_dim}"
    )
    if not args.no_stage1_init:
        load_stage1_encoder_into_stage2(stage1, stage2)

    optimizer2 = torch.optim.AdamW(stage2.parameters(), lr=args.lr_stage2, weight_decay=args.weight_decay)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=max(args.stage2_epochs, 1))
    best_trackers = build_stage2_best_trackers()
    stage2_early_stop_track = args.stage2_early_stop_track
    if stage2_early_stop_track not in best_trackers:
        print(f"[Stage2 EarlyStop] track={stage2_early_stop_track} not found, fallback to topk_trial_f1")
        stage2_early_stop_track = "topk_trial_f1"
    stage2_early_stop_criteria = best_trackers[stage2_early_stop_track]["criteria"]
    stage2_early_best_metrics = None
    stage2_early_best_epoch = None
    stage2_bad_epochs = 0
    print(
        f"[Stage2 EarlyStop] enabled={not args.no_stage2_early_stop}, "
        f"track={stage2_early_stop_track}, criteria={format_criteria(stage2_early_stop_criteria)}, "
        f"patience={args.stage2_early_stop_patience}, warmup={args.stage2_early_stop_warmup}, "
        f"min_delta={args.stage2_early_stop_min_delta}"
    )
    history2 = []
    last_stage2_train_metrics = {}
    last_stage2_val_metrics = {}
    last_stage2_epoch = 0

    for epoch in range(1, args.stage2_epochs + 1):
        train_metrics = train_stage2_one_epoch(
            stage2,
            source_loader,
            target_train_loader,
            optimizer2,
            device,
            source_subject_weights=source_weights,
            num_source_subjects=len(domain_mapping["source_subjects"]),
            lambda_expert=args.lambda_expert,
            lambda_mix=args.lambda_mix,
            lambda_diag=args.lambda_diag,
            diag_label_smoothing=args.diag_label_smoothing,
            lambda_mmd=args.stage2_lambda_mmd,
            lambda_cond_mmd=args.lambda_cond_mmd,
            cond_mmd_warmup_epochs=args.cond_mmd_warmup_epochs,
            cond_mmd_conf_threshold=args.cond_mmd_conf_threshold,
            cond_mmd_detach_target_prob=args.cond_mmd_detach_target_prob,
            lambda_subject=args.lambda_subject,
            grl_subject=args.grl_subject,
            lambda_ent=args.lambda_ent,
            lambda_mdc=args.lambda_mdc,
            lambda_rank=args.lambda_rank,
            rank_margin=args.rank_margin,
            rank_warmup_epochs=args.rank_warmup_epochs,
            rank_max_pairs_per_subject=args.rank_max_pairs_per_subject,
            current_epoch=epoch,
            source_de_mu=source_de_mu,
            source_de_std=source_de_std,
            source_bio_mu=source_bio_mu,
            source_bio_std=source_bio_std,
            target_de_mu=target_de_mu,
            target_de_std=target_de_std,
            target_bio_mu=target_bio_mu,
            target_bio_std=target_bio_std,
        )
        val_metrics = validate_stage2_trial_level(
            stage2,
            val_loader,
            device,
            threshold=args.threshold,
            k_pos=args.k_pos,
            lambda_diag=args.lambda_diag,
            diag_label_smoothing=args.diag_label_smoothing,
            val_de_mu=val_de_mu,
            val_de_std=val_de_std,
            val_bio_mu=val_bio_mu,
            val_bio_std=val_bio_std,
        )
        scheduler2.step()
        row = {"epoch": epoch, "lr": optimizer2.param_groups[0]["lr"]}
        row.update(flatten_metrics_for_csv("train", train_metrics))
        row.update(flatten_metrics_for_csv("val", val_metrics))
        last_stage2_train_metrics = copy.deepcopy(train_metrics)
        last_stage2_val_metrics = copy.deepcopy(val_metrics)
        last_stage2_epoch = epoch
        print(
            f"[Stage2] epoch={epoch} loss={train_metrics['loss']:.4f} "
            f"train_emo_f1={train_metrics['emotion_macro_f1']:.4f} "
            f"train_diag_f1={train_metrics['diag_macro_f1']:.4f} "
            f"loss_cond_mmd={train_metrics.get('loss_cond_mmd', 0.0):.4f} "
            f"target_conf_mean={train_metrics.get('cond_mmd_target_conf_mean', 0.0):.4f} "
            f"target_pseudo_pos_rate={train_metrics.get('cond_mmd_target_pseudo_pos_rate', 0.0):.4f} "
            f"valid_classes={train_metrics.get('cond_mmd_valid_classes', 0.0):.2f} "
            f"rank={train_metrics.get('loss_rank', 0.0):.4f} "
            f"val_trial_f1={val_metrics['trial_macro_f1']:.4f} val_trial_acc={val_metrics['trial_acc']:.4f} "
            f"val_topk_f1={val_metrics['topk_trial_macro_f1']:.4f} val_topk_acc={val_metrics['topk_trial_acc']:.4f}"
            f"{format_stage2_arch_stats(train_metrics)}"
        )

        weight_stats = summarize_source_weights(source_weights)
        dynamic_history_fields = {
            "source_weight_entropy": weight_stats["source_weight_entropy"],
            "source_weight_max": weight_stats["source_weight_max"],
            "source_weight_min": weight_stats["source_weight_min"],
            "source_weight_l1_change": 0.0,
            "dynamic_mmd_distance_mean": 0.0,
            "dynamic_mmd_distance_min": 0.0,
            "dynamic_mmd_distance_max": 0.0,
        }
        update_every = max(1, int(args.source_weight_update_every))
        if args.dynamic_source_weight and epoch >= args.dynamic_weight_start_epoch and epoch % update_every == 0:
            old_weights = dict(source_weights)
            new_weights, dynamic_stats = compute_dynamic_source_weights_by_mmd(
                model=stage2,
                source_dataset=source_dataset,
                target_dataset=target_train_dataset,
                domain_mapping=domain_mapping,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                source_de_mu=source_de_mu,
                source_de_std=source_de_std,
                source_bio_mu=source_bio_mu,
                source_bio_std=source_bio_std,
                target_de_mu=target_de_mu,
                target_de_std=target_de_std,
                target_bio_mu=target_bio_mu,
                target_bio_std=target_bio_std,
                dynamic_weight_tau=args.dynamic_weight_tau,
                dynamic_weight_min=args.dynamic_weight_min,
                dynamic_weight_max=args.dynamic_weight_max,
                max_samples_per_subject=args.dynamic_max_samples_per_subject,
                max_target_samples=args.dynamic_max_target_samples,
            )
            source_weights = ema_update_source_weights(
                old_weights=old_weights,
                new_weights=new_weights,
                momentum=args.source_weight_ema,
                blend_with_stage1=stage1_prior_source_weights,
                stage1_alpha=args.dynamic_weight_alpha_stage1,
                mean_one=args.use_mean_one_source_weights,
            )
            source_weights = clip_source_weights(
                source_weights,
                max_weight=args.source_weight_clip,
                blend_uniform=args.source_weight_blend_uniform,
                mean_one=args.use_mean_one_source_weights,
            )
            l1_change = source_weight_l1_change(old_weights, source_weights)
            weight_stats = summarize_source_weights(source_weights)
            dynamic_history_fields = {
                "source_weight_entropy": weight_stats["source_weight_entropy"],
                "source_weight_max": weight_stats["source_weight_max"],
                "source_weight_min": weight_stats["source_weight_min"],
                "source_weight_l1_change": l1_change,
                "dynamic_mmd_distance_mean": float(dynamic_stats.get("dynamic_mmd_distance_mean", 0.0)),
                "dynamic_mmd_distance_min": float(dynamic_stats.get("dynamic_mmd_distance_min", 0.0)),
                "dynamic_mmd_distance_max": float(dynamic_stats.get("dynamic_mmd_distance_max", 0.0)),
            }
            dynamic_record = {
                "epoch": epoch,
                "old_weights": old_weights,
                "new_mmd_weights": new_weights,
                "updated_weights": source_weights,
                "dynamic_stats": dynamic_stats,
                "args": {
                    "source_weight_update_every": args.source_weight_update_every,
                    "source_weight_ema": args.source_weight_ema,
                    "dynamic_weight_tau": args.dynamic_weight_tau,
                    "dynamic_weight_alpha_stage1": args.dynamic_weight_alpha_stage1,
                    "dynamic_weight_min": args.dynamic_weight_min,
                    "dynamic_weight_max": args.dynamic_weight_max,
                    "dynamic_max_samples_per_subject": args.dynamic_max_samples_per_subject,
                    "dynamic_max_target_samples": args.dynamic_max_target_samples,
                },
                **dynamic_history_fields,
            }
            source_weight_history.append(dynamic_record)
            dynamic_json_path = save_dir / f"dynamic_source_weights_epoch_{epoch}.json"
            save_json(dynamic_json_path, dynamic_record)
            top_dynamic_weights = sorted(source_weights.items(), key=lambda kv: kv[1], reverse=True)[:8]
            print(
                f"[Dynamic Source Weight] epoch={epoch} saved={dynamic_json_path} "
                f"entropy={weight_stats['source_weight_entropy']:.4f} "
                f"max={weight_stats['source_weight_max']:.4f} min={weight_stats['source_weight_min']:.4f} "
                f"l1_change={l1_change:.4f} "
                f"mmd_mean/min/max={dynamic_history_fields['dynamic_mmd_distance_mean']:.4f}/"
                f"{dynamic_history_fields['dynamic_mmd_distance_min']:.4f}/"
                f"{dynamic_history_fields['dynamic_mmd_distance_max']:.4f}"
            )
            print(f"[Dynamic Source Weight] top source weights: {top_dynamic_weights}")

        row.update(dynamic_history_fields)
        history2.append(row)

        if epoch <= args.save_warmup_epochs:
            print(f"[Stage2] skip best checkpoint during save warmup ({epoch}/{args.save_warmup_epochs})")
        else:
            for best_name, tracker in best_trackers.items():
                criteria = tracker["criteria"]
                if is_better_by_criteria(val_metrics, tracker["best_metrics"], criteria):
                    ckpt_path, json_path, csv_path = save_best_checkpoint(
                        save_dir=save_dir,
                        fold=fold,
                        best_name=best_name,
                        model=stage2,
                        optimizer=optimizer2,
                        epoch=epoch,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        train_subjects=split["train_all"],
                        val_subjects=split["val_all"],
                        criteria=criteria,
                        config_dict=config_dict,
                        scheduler=scheduler2,
                        extra_state={
                            "domain_mapping": domain_mapping,
                            "arch_variant": args.arch_variant,
                            "resolved_arch": resolved_arch_state(args),
                            "source_subject_weights": source_weights,
                            "raw_source_subject_weights": raw_source_weights,
                            "initial_stage1_source_weights": stage1_prior_source_weights,
                            "source_weight_history": source_weight_history,
                            "source_weight_mode": source_weight_mode,
                            "source_weight_vote": vote_result,
                            "dynamic_source_weight": args.dynamic_source_weight,
                            "lambda_cond_mmd": args.lambda_cond_mmd,
                            "cond_mmd_conf_threshold": args.cond_mmd_conf_threshold,
                        },
                    )
                    tracker["best_metrics"] = copy.deepcopy(val_metrics)
                    tracker["best_train_metrics"] = copy.deepcopy(train_metrics)
                    tracker["best_epoch"] = epoch
                    tracker["checkpoint_path"] = ckpt_path
                    tracker["metrics_json_path"] = json_path
                    tracker["summary_csv_path"] = csv_path
                    print(
                        f"[Stage2] saved best_{best_name}: epoch={epoch}, "
                        f"criteria={format_criteria(criteria)}, path={ckpt_path}"
                    )

        if not args.no_stage2_early_stop:
            if epoch <= args.stage2_early_stop_warmup:
                warmup_improved = is_better_by_criteria(
                    val_metrics,
                    stage2_early_best_metrics,
                    stage2_early_stop_criteria,
                    eps=args.stage2_early_stop_min_delta,
                )
                if warmup_improved:
                    stage2_early_best_metrics = copy.deepcopy(val_metrics)
                    stage2_early_best_epoch = epoch
                print(
                    f"[Stage2 EarlyStop] warmup ({epoch}/{args.stage2_early_stop_warmup}), "
                    f"best_epoch={stage2_early_best_epoch}"
                )
            else:
                improved = is_better_by_criteria(
                    val_metrics,
                    stage2_early_best_metrics,
                    stage2_early_stop_criteria,
                    eps=args.stage2_early_stop_min_delta,
                )
                if improved:
                    stage2_early_best_metrics = copy.deepcopy(val_metrics)
                    stage2_early_best_epoch = epoch
                    stage2_bad_epochs = 0
                else:
                    stage2_bad_epochs += 1
                print(
                    f"[Stage2 EarlyStop] improved={improved}, "
                    f"best_epoch={stage2_early_best_epoch}, "
                    f"bad_epochs={stage2_bad_epochs}/{args.stage2_early_stop_patience}"
                )
                if stage2_bad_epochs >= args.stage2_early_stop_patience:
                    print(
                        f"[Stage2 EarlyStop] stop at epoch={epoch}; "
                        f"best_epoch={stage2_early_best_epoch}"
                    )
                    break

    stage2_history_csv = save_dir / "stage2_history.csv"
    pd.DataFrame(history2).to_csv(stage2_history_csv, index=False, encoding="utf-8-sig")
    stage2_final_path = None
    stage2_final_json = None
    stage2_final_csv = None
    if last_stage2_epoch > 0:
        stage2_final_path, stage2_final_json, stage2_final_csv = save_final_checkpoint(
            save_dir=save_dir,
            stage_name="stage2",
            model=stage2,
            optimizer=optimizer2,
            epoch=last_stage2_epoch,
            train_metrics=last_stage2_train_metrics,
            val_metrics=last_stage2_val_metrics,
            train_subjects=split["train_all"],
            val_subjects=split["val_all"],
            config_dict=config_dict,
            scheduler=scheduler2,
            extra_state={
                "domain_mapping": domain_mapping,
                "arch_variant": args.arch_variant,
                "resolved_arch": resolved_arch_state(args),
                "source_subject_weights": source_weights,
                "raw_source_subject_weights": raw_source_weights,
                "initial_stage1_source_weights": stage1_prior_source_weights,
                "source_weight_history": source_weight_history,
                "source_weight_mode": source_weight_mode,
                "source_weight_vote": vote_result,
                "dynamic_source_weight": args.dynamic_source_weight,
                "lambda_cond_mmd": args.lambda_cond_mmd,
                "cond_mmd_conf_threshold": args.cond_mmd_conf_threshold,
            },
        )

    best_rows = []
    for best_name, tracker in best_trackers.items():
        row = {
            "best_name": best_name,
            "best_epoch": tracker["best_epoch"],
            "criteria": format_criteria(tracker["criteria"]),
            "checkpoint_path": tracker["checkpoint_path"],
            "metrics_json_path": tracker["metrics_json_path"],
            "summary_csv_path": tracker["summary_csv_path"],
        }
        if tracker["best_metrics"] is not None:
            row.update(flatten_metrics_for_csv("val", tracker["best_metrics"]))
        best_rows.append(row)
    stage2_best_summary_csv = save_dir / f"stage2_best_summary_fold{fold}.csv"
    pd.DataFrame(best_rows).to_csv(stage2_best_summary_csv, index=False, encoding="utf-8-sig")
    print(f"[Stage2] history saved: {stage2_history_csv}")
    print(f"[Stage2] best summary saved: {stage2_best_summary_csv}")

    selected_best_name = args.predict_best_name
    if selected_best_name not in best_trackers:
        print(f"[Stage2] predict_best_name={selected_best_name} not found, fallback to combined")
        selected_best_name = "combined"
    selected_tracker = best_trackers[selected_best_name]
    best_path = selected_tracker["checkpoint_path"]
    best_val = selected_tracker["best_metrics"]
    if best_path is None and stage2_final_path is not None:
        print(
            "[Stage2] no best checkpoint was saved after warmup; "
            f"fallback to final checkpoint for prediction: {stage2_final_path}"
        )
        selected_best_name = "final_fallback"
        best_path = stage2_final_path
        best_val = last_stage2_val_metrics

    if args.predict_test and args.test_csv and Path(args.test_csv).exists() and best_path:
        ckpt = load_checkpoint(best_path, device)
        print_checkpoint_info(best_path, ckpt, prefix=f"[V4 predict ckpt {selected_best_name}]")
        stage2.load_state_dict(ckpt["model_state_dict"])
        trial_df = predict_test_trial_level(
            stage2,
            args.test_csv,
            domain_mapping,
            device,
            save_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold=args.threshold,
            vote_method=args.test_vote_method,
            k_pos=args.k_pos,
            normalize=not args.no_normalize,
        )
        print(f"[test] saved predictions, trials={len(trial_df)}")

    return {
        "fold": fold,
        "repeat_index": repeat_index,
        "rand_seed": rand_seed,
        "run_seed": run_seed,
        "save_dir": str(save_dir),
        "best_path": best_path,
        "best_name": selected_best_name,
        "best_val": best_val,
        "stage2_history_csv": str(stage2_history_csv),
        "stage2_best_summary_csv": str(stage2_best_summary_csv),
        "stage2_final_path": stage2_final_path,
        "stage2_final_metrics_json": stage2_final_json,
        "stage2_final_summary_csv": stage2_final_csv,
        "best_models": best_trackers,
    }


def resolve_arch_variant_args(args: argparse.Namespace) -> argparse.Namespace:
    variant_defaults = {
        "base": {
            "router_temperature": 1.0,
            "diag_label_smoothing": 0.0,
            "shared_expert_trunk": False,
            "fusion_mode": "concat",
            "learnable_graph": False,
        },
        "A": {
            "router_temperature": 1.5,
            "diag_label_smoothing": 0.05,
            "shared_expert_trunk": False,
            "fusion_mode": "concat",
            "learnable_graph": False,
        },
        "B": {
            "router_temperature": 1.5,
            "diag_label_smoothing": 0.05,
            "shared_expert_trunk": True,
            "fusion_mode": "concat",
            "learnable_graph": False,
        },
        "C": {
            "router_temperature": 1.5,
            "diag_label_smoothing": 0.05,
            "shared_expert_trunk": True,
            "fusion_mode": "scalar_gated",
            "learnable_graph": False,
        },
        "D": {
            "router_temperature": 1.5,
            "diag_label_smoothing": 0.05,
            "shared_expert_trunk": True,
            "fusion_mode": "scalar_gated",
            "learnable_graph": True,
        },
    }
    defaults = variant_defaults[args.arch_variant]
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def resolved_arch_state(args: argparse.Namespace) -> dict:
    return {
        "router_temperature": args.router_temperature,
        "diag_label_smoothing": args.diag_label_smoothing,
        "shared_expert_trunk": args.shared_expert_trunk,
        "fusion_mode": args.fusion_mode,
        "learnable_graph": args.learnable_graph,
    }


def print_arch_config(args: argparse.Namespace) -> None:
    print(f"[Arch] variant={args.arch_variant}")
    print(f"[Arch] router_temperature={args.router_temperature}")
    print(f"[Arch] diag_label_smoothing={args.diag_label_smoothing}")
    print(f"[Arch] shared_expert_trunk={args.shared_expert_trunk}")
    print(f"[Arch] fusion_mode={args.fusion_mode}")
    print(f"[Arch] learnable_graph={args.learnable_graph}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", type=str, default="com_index_sub_2s.csv")
    parser.add_argument("--test_csv", type=str, default="com_test_trial_index_2s.csv")
    parser.add_argument("--save_root", type=str, default="model_params/V4_expert_ssas")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--all_repeats", action="store_true")
    parser.add_argument("--stage1_epochs", type=int, default=5)
    parser.add_argument("--stage2_epochs", type=int, default=25)
    parser.add_argument("--save_warmup_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--lr_stage1", type=float, default=1e-4)
    parser.add_argument("--lr_stage2", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--biomarker_dim", type=int, default=57)
    parser.add_argument("--de_num_bands", type=int, default=5)
    parser.add_argument("--no_biomarkers", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--no_subject_relative_de", action="store_true")
    parser.add_argument("--no_subject_relative_bio", action="store_true")
    parser.add_argument("--bio_abs_scale", type=float, default=0.3)
    parser.add_argument("--relative_eps", type=float, default=1e-6)
    parser.add_argument("--use_raw_diagnosis_label", action="store_true")
    parser.add_argument(
        "--arch_variant",
        choices=["base", "A", "B", "C", "D"],
        default="base",
        help="Architecture ablation variant: base/A/B/C/D. Variants are cumulative.",
    )
    parser.add_argument("--router_temperature", type=float, default=None)
    parser.add_argument("--diag_label_smoothing", type=float, default=None)
    parser.add_argument("--shared_expert_trunk", action="store_true")
    parser.add_argument("--no_shared_expert_trunk", dest="shared_expert_trunk", action="store_false")
    parser.add_argument("--expert_hidden_dim", type=int, default=64)
    parser.add_argument(
        "--fusion_mode",
        choices=["concat", "scalar_gated"],
        default=None,
    )
    parser.add_argument("--bio_gate_init", type=float, default=-2.0)
    parser.add_argument("--core_gate_init", type=float, default=0.0)
    parser.add_argument("--graph_gate_init", type=float, default=0.0)
    parser.add_argument("--temporal_gate_init", type=float, default=0.0)
    parser.add_argument("--learnable_graph", action="store_true")
    parser.add_argument("--no_learnable_graph", dest="learnable_graph", action="store_false")
    parser.add_argument("--learnable_graph_rank", type=int, default=4)
    parser.add_argument("--graph_mix_logit_init", type=float, default=2.0)
    parser.add_argument("--learnable_graph_dropout", type=float, default=0.0)
    parser.set_defaults(shared_expert_trunk=None)
    parser.set_defaults(learnable_graph=None)

    parser.add_argument("--lambda_domain", type=float, default=1.0)
    parser.add_argument("--stage1_lambda_mmd", type=float, default=0.03)
    parser.add_argument("--lambda_emo_grl", type=float, default=0.001)
    parser.add_argument("--grl_emo", type=float, default=0.01)
    parser.add_argument("--lambda_diag_grl", type=float, default=0.001)
    parser.add_argument("--grl_diag", type=float, default=0.01)
    parser.add_argument("--lambda_weight_reg", type=float, default=0.0)
    parser.add_argument("--vote_smooth", type=float, default=5.0)
    parser.add_argument("--vote_level", choices=["window", "trial"], default="trial")
    parser.add_argument("--vote_mode", choices=["hard", "soft", "soft_conf"], default="soft")
    parser.add_argument("--tau_vote", type=float, default=1.0)
    parser.add_argument("--confidence_power", type=float, default=1.0)
    parser.add_argument("--save_topk_probs", type=int, default=5)
    parser.add_argument("--use_mean_one_source_weights", dest="use_mean_one_source_weights", action="store_true", default=True)
    parser.add_argument("--use_sum_one_source_weights", dest="use_mean_one_source_weights", action="store_false")
    parser.add_argument("--source_weight_clip", type=float, default=4.0)
    parser.add_argument("--source_weight_blend_uniform", type=float, default=0.0)

    parser.add_argument("--lambda_expert", type=float, default=0.5)
    parser.add_argument("--lambda_mix", type=float, default=1.0)
    parser.add_argument("--lambda_diag", type=float, default=0.01)
    parser.add_argument("--stage2_lambda_mmd", type=float, default=0.0003)
    parser.add_argument("--lambda_cond_mmd", type=float, default=0.0003)
    parser.add_argument("--cond_mmd_warmup_epochs", type=int, default=5)
    parser.add_argument("--cond_mmd_conf_threshold", type=float, default=0.7)
    parser.add_argument("--cond_mmd_detach_target_prob", action="store_true", default=True)
    parser.add_argument("--no_cond_mmd_detach_target_prob", dest="cond_mmd_detach_target_prob", action="store_false")
    parser.add_argument("--lambda_subject", type=float, default=0.0003)
    parser.add_argument("--grl_subject", type=float, default=0.001)
    parser.add_argument("--lambda_ent", type=float, default=0.0)
    parser.add_argument("--lambda_mdc", type=float, default=0.0)
    parser.add_argument("--lambda_rank", type=float, default=0.1)
    parser.add_argument("--rank_margin", type=float, default=0.2)
    parser.add_argument("--rank_warmup_epochs", type=int, default=3)
    parser.add_argument("--rank_max_pairs_per_subject", type=int, default=128)
    parser.add_argument("--shared_mix_alpha", type=float, default=0.7)
    parser.add_argument("--no_stage1_init", action="store_true")
    parser.add_argument("--no_stage2_early_stop", action="store_true")
    parser.add_argument("--stage2_early_stop_track", type=str, default="topk_trial_f1")
    parser.add_argument("--stage2_early_stop_patience", type=int, default=5)
    parser.add_argument("--stage2_early_stop_warmup", type=int, default=3)
    parser.add_argument("--stage2_early_stop_min_delta", type=float, default=1e-6)

    parser.add_argument("--dynamic_source_weight", action="store_true")
    parser.add_argument("--dynamic_weight_start_epoch", type=int, default=5)
    parser.add_argument("--source_weight_update_every", type=int, default=5)
    parser.add_argument("--source_weight_ema", type=float, default=0.8)
    parser.add_argument("--dynamic_weight_tau", type=float, default=1.0)
    parser.add_argument("--dynamic_weight_alpha_stage1", type=float, default=0.3)
    parser.add_argument("--dynamic_weight_min", type=float, default=0.5)
    parser.add_argument("--dynamic_weight_max", type=float, default=2.0)
    parser.add_argument("--dynamic_max_samples_per_subject", type=int, default=512)
    parser.add_argument("--dynamic_max_target_samples", type=int, default=4096)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--predict_test", action="store_true")
    parser.add_argument("--predict_best_name", type=str, default="topk_trial_f1")
    parser.add_argument(
        "--test_vote_method",
        choices=["prob", "soft_threshold", "hard", "soft_topk", "topk"],
        default="soft_topk",
    )
    parser.add_argument("--k_pos", type=int, default=4)
    parser.add_argument("--no_test_ensemble", action="store_true")
    parser.add_argument("--test_ensemble_dir", type=str, default="test_ensemble")
    return resolve_arch_variant_args(parser.parse_args())


def get_v4_seed_for_repeat(repeat: int) -> int:
    seed_list = list(getattr(config, "V4_seed", getattr(config, "V3_seed", getattr(config, "V2_seed", [42]))))
    if not seed_list:
        seed_list = [42]
    if repeat < 0:
        raise IndexError(f"repeat={repeat} 不能为负数。")
    if repeat < len(seed_list):
        return int(seed_list[repeat])

    derived = int(seed_list[0]) + int(repeat) * 31
    print(f"[config] V4 repeat={repeat} 超出预定义种子列表 {seed_list}，使用派生种子 {derived}")
    return derived


def main() -> None:
    args = parse_args()
    seed_list = list(getattr(config, "V4_seed", getattr(config, "V3_seed", getattr(config, "V2_seed", [42]))))
    repeats = range(len(seed_list)) if args.all_repeats else [args.repeat]
    folds = range(args.n_splits) if args.all_folds else [args.fold]
    results = []
    for repeat_index in repeats:
        rand_seed = get_v4_seed_for_repeat(repeat_index)
        for fold in folds:
            results.append(run_one_fold(args, fold=fold, repeat_index=repeat_index, rand_seed=rand_seed))

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for result in results:
        row = {
            "repeat_index": result["repeat_index"],
            "fold": result["fold"],
            "rand_seed": result["rand_seed"],
            "run_seed": result["run_seed"],
            "save_dir": result["save_dir"],
            "best_name": result.get("best_name"),
            "best_path": result["best_path"],
            "stage2_history_csv": result.get("stage2_history_csv"),
            "stage2_best_summary_csv": result.get("stage2_best_summary_csv"),
            "stage2_final_path": result.get("stage2_final_path"),
        }
        if result["best_val"] is not None:
            row.update(flatten_metrics_for_csv("val", result["best_val"]))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(save_root / "all_fold_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[done] summary saved to {save_root / 'all_fold_summary.csv'}")

    if not args.no_test_ensemble and args.test_csv and Path(args.test_csv).exists():
        model_paths = [result.get("best_path") for result in results if result.get("best_path")]
        model_paths = [p for p in model_paths if Path(p).exists()]
        if model_paths:
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            ensemble_dir = save_root / args.test_ensemble_dir
            ensemble_test_predictions(
                model_paths=model_paths,
                test_csv=args.test_csv,
                device=device,
                save_dir=ensemble_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                threshold=args.threshold,
                vote_method=args.test_vote_method,
                k_pos=args.k_pos,
                normalize=not args.no_normalize,
            )
        else:
            print("[V4 ensemble] no valid best checkpoints found; skip test ensemble.")
    elif args.no_test_ensemble:
        print("[V4 ensemble] skipped by --no_test_ensemble.")


if __name__ == "__main__":
    main()
    #训练
    # python3 V4/train_expert_ssas_emotion.py --all_folds --all_repeats --batch_size 200 --test_vote_method soft_topk --k_pos 4

    #python3 V4/train_expert_ssas_emotion.py --all_folds --all_repeats --batch_size 200

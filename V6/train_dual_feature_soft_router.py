# -*- coding: utf-8 -*-
"""Train V6_NoSSAS_DualFeature_SoftRouter.

Typical quick run:
    python V6/train_dual_feature_soft_router.py --fold 0 --epochs 2

Full fold run:
    python V6/train_dual_feature_soft_router.py --all_folds --epochs 100
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
from dataloader import Competition4ClassDataset
from dual_feature_soft_router_model import (
    DualFeatureSoftRouterModel,
    hard_expert_emotion_loss,
    mixture_emotion_nll_loss,
)
from utils.data import expand_window_index, resolve_data_path
from utils.folds import get_unified_subject_split

VERSION_NAME = "V6_NoSSAS_DualFeature_SoftRouter"


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


def compute_subject_de_baselines(
    dataset: Dataset,
    key_field: str = "target_key",
    eps: float = 1e-6,
) -> tuple[dict, dict]:
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
    backbone = getattr(getattr(model, "shared_encoder", None), "backbone", None)
    if backbone is not None and not getattr(backbone, "use_biomarkers", True):
        print("[Bio baseline] skipped because biomarkers are disabled.")
        return {}, {}

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
            print("[Bio baseline] skipped because model forward did not return bio_raw.")
            return {}, {}
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
    """Labeled source/val dataset with stable domain ids for compatibility."""

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
            item["diagnosis_label"] = torch.tensor(int(item["label4"].item() >= 2), dtype=torch.long)
        item["user_id"] = subject_key
        item["target_key"] = domain_key
        return item


class UnlabeledTargetDataset(Dataset):
    """Unlabeled test windows with the same tensor keys as labeled batches."""

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


def build_emotion_class_weights(source_dataset: DomainAwareCompetitionDataset) -> torch.Tensor:
    if "emotion_label" in source_dataset.df.columns:
        labels = source_dataset.df["emotion_label"].astype(int).to_numpy()
    else:
        labels = (source_dataset.df["label4"].astype(int).to_numpy() % 2).astype(int)
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


def _selected_expert_logits(hc_logits: torch.Tensor, dep_logits: torch.Tensor, y_diag: torch.Tensor) -> torch.Tensor:
    selected_logits = torch.empty_like(hc_logits)
    hc_mask = y_diag.long() == 1
    dep_mask = y_diag.long() == 0
    selected_logits[hc_mask] = hc_logits[hc_mask]
    selected_logits[dep_mask] = dep_logits[dep_mask]
    if (~(hc_mask | dep_mask)).any():
        selected_logits[~(hc_mask | dep_mask)] = dep_logits[~(hc_mask | dep_mask)]
    return selected_logits


def train_dual_router_one_epoch(
    model,
    source_loader,
    optimizer,
    device,
    lambda_mix=1.0,
    lambda_expert=0.5,
    lambda_shared=0.3,
    lambda_diag=0.1,
    emotion_class_weight=None,
    diag_class_weight=None,
    source_de_mu=None,
    source_de_std=None,
    source_bio_mu=None,
    source_bio_std=None,
):
    model.train()
    totals = defaultdict(float)
    total_n = 0
    all_mix_preds, all_shared_preds, all_expert_preds, all_emo_labels = [], [], [], []
    all_diag_preds, all_diag_labels = [], []

    emotion_class_weight = emotion_class_weight.to(device) if emotion_class_weight is not None else None
    diag_class_weight = diag_class_weight.to(device) if diag_class_weight is not None else None

    pbar = tqdm(source_loader, desc="V6-Train", leave=False)
    for batch in pbar:
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

        optimizer.zero_grad(set_to_none=True)
        out = model(batch["x"], batch["de_feat"], **rel_kwargs)

        y_emo = batch["emotion_label"].long()
        y_diag = batch["diagnosis_label"].long()

        loss_mix = mixture_emotion_nll_loss(out["mix_prob"], y_emo)
        loss_expert = hard_expert_emotion_loss(out["hc_logits"], out["dep_logits"], y_emo, y_diag)
        loss_shared = F.cross_entropy(out["shared_logits"], y_emo, weight=emotion_class_weight)
        loss_diag = F.cross_entropy(out["diag_logits"], y_diag, weight=diag_class_weight)
        loss = (
            float(lambda_mix) * loss_mix
            + float(lambda_expert) * loss_expert
            + float(lambda_shared) * loss_shared
            + float(lambda_diag) * loss_diag
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mix_pred = out["mix_prob"].argmax(dim=1)
            shared_pred = out["shared_logits"].argmax(dim=1)
            selected_logits = _selected_expert_logits(out["hc_logits"], out["dep_logits"], y_diag)
            expert_pred = selected_logits.argmax(dim=1)
            diag_pred = out["diag_logits"].argmax(dim=1)
            hc_mask = y_diag == 1
            dep_mask = y_diag == 0

        bsz = batch["x"].size(0)
        total_n += bsz
        totals["loss"] += float(loss.item()) * bsz
        totals["loss_mix"] += float(loss_mix.item()) * bsz
        totals["loss_expert"] += float(loss_expert.item()) * bsz
        totals["loss_shared"] += float(loss_shared.item()) * bsz
        totals["loss_diag"] += float(loss_diag.item()) * bsz
        totals["mix_correct"] += int((mix_pred == y_emo).sum().item())
        totals["shared_correct"] += int((shared_pred == y_emo).sum().item())
        totals["expert_correct"] += int((expert_pred == y_emo).sum().item())
        totals["diag_correct"] += int((diag_pred == y_diag).sum().item())
        totals["hc_n"] += int(hc_mask.sum().item())
        totals["dep_n"] += int(dep_mask.sum().item())
        if hc_mask.any():
            totals["hc_expert_correct"] += int((out["hc_logits"][hc_mask].argmax(dim=1) == y_emo[hc_mask]).sum().item())
        if dep_mask.any():
            totals["dep_expert_correct"] += int((out["dep_logits"][dep_mask].argmax(dim=1) == y_emo[dep_mask]).sum().item())
        totals["pred_pos_mix"] += int((mix_pred == 1).sum().item())
        totals["true_pos_emo"] += int((y_emo == 1).sum().item())
        totals["pred_hc_diag"] += int((diag_pred == 1).sum().item())
        totals["true_hc_diag"] += int((y_diag == 1).sum().item())

        all_mix_preds.extend(mix_pred.detach().cpu().tolist())
        all_shared_preds.extend(shared_pred.detach().cpu().tolist())
        all_expert_preds.extend(expert_pred.detach().cpu().tolist())
        all_emo_labels.extend(y_emo.detach().cpu().tolist())
        all_diag_preds.extend(diag_pred.detach().cpu().tolist())
        all_diag_labels.extend(y_diag.detach().cpu().tolist())

        pbar.set_postfix(
            {
                "loss": f"{totals['loss'] / max(total_n, 1):.4f}",
                "mix": f"{totals['loss_mix'] / max(total_n, 1):.4f}",
                "expert": f"{totals['loss_expert'] / max(total_n, 1):.4f}",
                "shared": f"{totals['loss_shared'] / max(total_n, 1):.4f}",
                "diag": f"{totals['loss_diag'] / max(total_n, 1):.4f}",
            }
        )

    if total_n == 0:
        raise RuntimeError("V6 source loader is empty: total_n == 0.")

    emotion_labels = [0, 1]
    metrics = {
        "loss": totals["loss"] / total_n,
        "loss_mix": totals["loss_mix"] / total_n,
        "loss_expert": totals["loss_expert"] / total_n,
        "loss_shared": totals["loss_shared"] / total_n,
        "loss_diag": totals["loss_diag"] / total_n,
        "segment_acc": totals["mix_correct"] / total_n,
        "emotion_acc": totals["mix_correct"] / total_n,
        "shared_acc": totals["shared_correct"] / total_n,
        "expert_acc": totals["expert_correct"] / total_n,
        "diag_acc": totals["diag_correct"] / total_n,
        "hc_expert_acc": totals["hc_expert_correct"] / max(totals["hc_n"], 1),
        "dep_expert_acc": totals["dep_expert_correct"] / max(totals["dep_n"], 1),
        "hc_samples": totals["hc_n"],
        "dep_samples": totals["dep_n"],
        "pred_pos_rate": totals["pred_pos_mix"] / total_n,
        "true_pos_rate": totals["true_pos_emo"] / total_n,
        "pred_hc_rate": totals["pred_hc_diag"] / total_n,
        "true_hc_rate": totals["true_hc_diag"] / total_n,
    }
    metrics["emotion_macro_f1"] = f1_score(
        all_emo_labels,
        all_mix_preds,
        average="macro",
        labels=emotion_labels,
        zero_division=0,
    )
    metrics["shared_macro_f1"] = f1_score(
        all_emo_labels,
        all_shared_preds,
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
    metrics["emotion_confusion_matrix"] = confusion_matrix(all_emo_labels, all_mix_preds, labels=emotion_labels)
    metrics["shared_confusion_matrix"] = confusion_matrix(all_emo_labels, all_shared_preds, labels=emotion_labels)
    metrics["expert_confusion_matrix"] = confusion_matrix(all_emo_labels, all_expert_preds, labels=emotion_labels)
    metrics["diag_confusion_matrix"] = confusion_matrix(all_diag_labels, all_diag_preds, labels=[0, 1])
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
    metrics["acc_2cls"] = metrics["segment_acc"]
    metrics["acc"] = metrics["segment_acc"]
    metrics["macro_f1"] = metrics["emotion_macro_f1"]
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
        if vote_method in {"prob", "soft_threshold"}:
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
def validate_dual_router_trial_level(
    model,
    val_loader,
    device,
    threshold=0.5,
    k_pos=4,
    lambda_mix=1.0,
    lambda_expert=0.5,
    lambda_shared=0.3,
    lambda_diag=0.1,
    val_de_mu=None,
    val_de_std=None,
    val_bio_mu=None,
    val_bio_std=None,
):
    model.eval()
    totals = defaultdict(float)
    total_n = 0
    all_preds, all_labels = [], []
    all_diag_preds, all_diag_labels = [], []
    segment_records = []

    for batch in tqdm(val_loader, desc="V6-Val", leave=False):
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
        out = model(batch["x"], batch["de_feat"], **val_rel_kwargs)
        y_emo = batch["emotion_label"].long()
        y_diag = batch["diagnosis_label"].long()

        loss_mix = mixture_emotion_nll_loss(out["mix_prob"], y_emo)
        loss_expert = hard_expert_emotion_loss(out["hc_logits"], out["dep_logits"], y_emo, y_diag)
        loss_shared = F.cross_entropy(out["shared_logits"], y_emo)
        loss_diag = F.cross_entropy(out["diag_logits"], y_diag)
        loss = (
            float(lambda_mix) * loss_mix
            + float(lambda_expert) * loss_expert
            + float(lambda_shared) * loss_shared
            + float(lambda_diag) * loss_diag
        )

        bsz = batch["x"].size(0)
        total_n += bsz
        totals["loss"] += float(loss.item()) * bsz
        totals["loss_mix"] += float(loss_mix.item()) * bsz
        totals["loss_expert"] += float(loss_expert.item()) * bsz
        totals["loss_shared"] += float(loss_shared.item()) * bsz
        totals["loss_diag"] += float(loss_diag.item()) * bsz

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
                    "user_id": str(batch["user_id"][i])
                    if "user_id" in batch
                    else str(int(batch["subject_id"][i].detach().cpu())),
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
        "loss": totals["loss"] / max(total_n, 1),
        "loss_mix": totals["loss_mix"] / max(total_n, 1),
        "loss_expert": totals["loss_expert"] / max(total_n, 1),
        "loss_shared": totals["loss_shared"] / max(total_n, 1),
        "loss_diag": totals["loss_diag"] / max(total_n, 1),
        "segment_acc": segment_acc,
        "segment_macro_f1": segment_macro_f1,
        "segment_confusion_matrix": segment_cm,
        "per_class_emo": per_class,
        "diag_acc": accuracy_score(all_diag_labels, all_diag_preds) if all_diag_labels else 0.0,
        "diag_macro_f1": f1_score(all_diag_labels, all_diag_preds, average="macro", labels=[0, 1], zero_division=0)
        if all_diag_labels
        else 0.0,
        "diag_confusion_matrix": confusion_matrix(all_diag_labels, all_diag_preds, labels=[0, 1])
        if all_diag_labels
        else np.zeros((2, 2), dtype=int),
        "segment_records": segment_records,
    }
    metrics.update(trial_metrics)
    metrics.update(topk_trial_metrics)

    metrics["emotion_acc"] = metrics["segment_acc"]
    metrics["emotion_macro_f1"] = metrics["segment_macro_f1"]
    metrics["acc"] = metrics["segment_acc"]
    metrics["macro_f1"] = metrics["segment_macro_f1"]
    metrics["trial_emotion_acc"] = metrics["trial_acc"]
    metrics["trial_emotion_macro_f1"] = metrics["trial_macro_f1"]
    metrics["topk_trial_emotion_acc"] = metrics["topk_trial_acc"]
    metrics["topk_trial_emotion_macro_f1"] = metrics["topk_trial_macro_f1"]
    return metrics


def metric_value(metrics: dict, key: str, default: Optional[float] = 0.0) -> Optional[float]:
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
        filename = "v6_best.pt" if best_name == "combined" else f"v6_best_{best_name}_fold{fold}.pt"
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
    saved_time = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if path.exists() else "unknown"

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
        f"version={extra_state.get('version', config_dict.get('version', 'NA'))} | "
        f"best_name={ckpt.get('best_name', 'NA')} | epoch={ckpt.get('epoch', 'NA')}"
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
        f"loss_mix={_fmt(train_metrics, 'loss_mix')} "
        f"loss_expert={_fmt(train_metrics, 'loss_expert')} "
        f"loss_shared={_fmt(train_metrics, 'loss_shared')} "
        f"loss_diag={_fmt(train_metrics, 'loss_diag')}"
    )
    print(
        f"{prefix} config: lr={config_dict.get('lr', 'NA')} "
        f"weight_decay={config_dict.get('weight_decay', 'NA')} "
        f"dropout={config_dict.get('dropout', 'NA')} "
        f"shared_mix_alpha={config_dict.get('shared_mix_alpha', 'NA')}"
    )


def apply_subject_topk(trial_records: list[dict], k_pos: int = 4) -> list[dict]:
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


def build_v6_best_trackers() -> dict:
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
        "trial_f1": tracker(
            [
                ("trial_macro_f1", "max"),
                ("trial_acc", "max"),
                ("emotion_macro_f1", "max"),
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
    model,
    test_csv: str,
    domain_mapping: dict,
    device: torch.device,
    save_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    threshold: float = 0.5,
    vote_method: str = "soft_topk",
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
        out = model(batch["x"], batch["de_feat"], **test_rel_kwargs)
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
    submission.to_csv(save_dir / "submission_v6_dual_router.csv", index=False, encoding="utf-8-sig")
    n_topk = int(sub_soft_topk["Emotion_label"].sum())
    n_selected = int(submission["Emotion_label"].sum())
    print(
        f"[test] soft-topk={k_pos}: {n_topk}/{len(sub_soft_topk)} positive; "
        f"selected={vote_method}: {n_selected}/{len(submission)} positive"
    )
    return trial_df


def ensemble_test_predictions_from_fold_outputs(
    results: list[dict],
    save_root: str | Path,
    threshold: float = 0.5,
    vote_method: str = "soft_topk",
    k_pos: int = 4,
) -> Optional[pd.DataFrame]:
    """Average per-fold V6 test trial predictions and write ensemble submissions.

    Each fold has already run predict_test_trial_level(), so we ensemble the saved
    trial-level probabilities/log-odds instead of forwarding all models again.
    """

    fold_frames = []
    for result in results:
        trial_path = Path(result["save_dir"]) / "test_trial_probs.csv"
        if not trial_path.exists():
            print(f"[V6 ensemble] skip missing trial prediction: {trial_path}")
            continue
        df = pd.read_csv(trial_path)
        required_cols = {"user_id", "trial_id", "prob_pos", "score_pos", "hard_mean"}
        missing = required_cols - set(df.columns)
        if missing:
            print(f"[V6 ensemble] skip {trial_path}; missing columns={sorted(missing)}")
            continue
        df = df.copy()
        df["source_fold"] = int(result["fold"])
        df["source_repeat"] = int(result["repeat_index"])
        df["source_best_name"] = str(result.get("best_name", ""))
        df["source_best_path"] = str(result.get("best_path", ""))
        fold_frames.append(df)

    if not fold_frames:
        print("[V6 ensemble] no fold test_trial_probs.csv files found; ensemble skipped.")
        return None

    all_pred_df = pd.concat(fold_frames, ignore_index=True)
    ensemble_dir = Path(save_root) / "v6_test_ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    all_pred_df.to_csv(ensemble_dir / "ensemble_member_trial_probs.csv", index=False, encoding="utf-8-sig")

    agg_kwargs = {
        "prob_pos": ("prob_pos", "mean"),
        "score_pos": ("score_pos", "mean"),
        "hard_mean": ("hard_mean", "mean"),
        "model_count": ("prob_pos", "count"),
    }
    if "n_windows" in all_pred_df.columns:
        agg_kwargs["n_windows_mean"] = ("n_windows", "mean")

    trial_df = (
        all_pred_df.groupby(["user_id", "trial_id"], as_index=False)
        .agg(**agg_kwargs)
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

    trial_df.to_csv(ensemble_dir / "ensemble_trial_probs.csv", index=False, encoding="utf-8-sig")

    sub_soft_threshold = trial_df[["user_id", "trial_id", "pred_soft_threshold"]].copy()
    sub_soft_threshold.rename(columns={"pred_soft_threshold": "Emotion_label"}, inplace=True)
    sub_soft_threshold.to_csv(ensemble_dir / "submission_v6_ensemble_soft_threshold.csv", index=False, encoding="utf-8-sig")

    sub_soft_topk = trial_df[["user_id", "trial_id", "pred_soft_topk"]].copy()
    sub_soft_topk.rename(columns={"pred_soft_topk": "Emotion_label"}, inplace=True)
    sub_soft_topk.to_csv(ensemble_dir / "submission_v6_ensemble_soft_topk.csv", index=False, encoding="utf-8-sig")

    sub_hard_threshold = trial_df[["user_id", "trial_id", "pred_hard_threshold"]].copy()
    sub_hard_threshold.rename(columns={"pred_hard_threshold": "Emotion_label"}, inplace=True)
    sub_hard_threshold.to_csv(ensemble_dir / "submission_v6_ensemble_hard_threshold.csv", index=False, encoding="utf-8-sig")

    submission = trial_df[["user_id", "trial_id", "Emotion_label"]].copy()
    submission.to_csv(ensemble_dir / "submission_v6_ensemble.csv", index=False, encoding="utf-8-sig")

    n_models = int(trial_df["model_count"].max()) if not trial_df.empty else 0
    n_selected = int(submission["Emotion_label"].sum()) if not submission.empty else 0
    print(
        f"[V6 ensemble] saved to {ensemble_dir}; models={n_models}, "
        f"trials={len(submission)}, selected={vote_method}: {n_selected}/{len(submission)} positive"
    )
    return trial_df


def _extra_state(domain_mapping: dict) -> dict:
    return {
        "domain_mapping": domain_mapping,
        "version": VERSION_NAME,
        "ssas_enabled": False,
        "mmd_enabled": False,
        "subject_grl_enabled": False,
        "trial_supcon_enabled": False,
    }


def run_one_fold(args, fold: int, repeat_index: int, rand_seed: int) -> dict:
    run_seed = config.make_run_seed(rand_seed, fold) if hasattr(config, "make_run_seed") else int(rand_seed) * 1000 + int(fold)
    set_global_seed(run_seed, deterministic=args.deterministic)
    requested_device = torch.device(args.device)
    device = requested_device if requested_device.type == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    if requested_device.type == "cuda" and device.type == "cpu":
        print(f"[device] requested {args.device} but CUDA is unavailable; using CPU.")

    save_dir = Path(args.save_root) / f"v6_dual_router_repeat{repeat_index}_fold{fold}"
    save_dir.mkdir(parents=True, exist_ok=True)

    split = build_train_val_target_split(args.index_csv, fold=fold, n_splits=args.n_splits, seed=rand_seed)
    domain_mapping = build_domain_id_mapping(split["train_all"], split["val_all"], args.test_csv)
    save_json(save_dir / "domain_id_mapping.json", domain_mapping)
    print(
        f"[V6 fold] repeat={repeat_index} fold={fold} seed={rand_seed} run_seed={run_seed} "
        f"source_subjects={len(split['train_all'])} val_subjects={len(split['val_all'])} "
        f"domains={domain_mapping['num_domains']}"
    )

    config_dict = vars(args).copy()
    config_dict.update(
        {
            "fold": fold,
            "repeat_index": repeat_index,
            "rand_seed": rand_seed,
            "run_seed": run_seed,
            "version": VERSION_NAME,
            "ssas_enabled": False,
            "mmd_enabled": False,
            "subject_grl_enabled": False,
            "trial_supcon_enabled": False,
        }
    )
    save_json(save_dir / "config.json", config_dict)
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

    use_subject_relative_de = bool(not args.no_subject_relative_de)
    use_subject_relative_bio = bool((not args.no_subject_relative_bio) and (not args.no_biomarkers))
    print(
        f"[Subject Relative] DE={use_subject_relative_de}, "
        f"bio={use_subject_relative_bio}, bio_abs_scale={args.bio_abs_scale}, eps={args.relative_eps}"
    )

    source_de_mu = source_de_std = None
    val_de_mu = val_de_std = None
    source_bio_mu = source_bio_std = None
    val_bio_mu = val_bio_std = None

    if use_subject_relative_de:
        print("[Subject Relative] computing source/val DE baselines...")
        source_de_mu, source_de_std = compute_subject_de_baselines(source_dataset, eps=args.relative_eps)
        val_de_mu, val_de_std = compute_subject_de_baselines(val_dataset, eps=args.relative_eps)

    model = DualFeatureSoftRouterModel(
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
    ).to(device)

    if use_subject_relative_bio:
        print("[Subject Relative] computing source/val bio baselines with V6 encoder...")
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
        source_bio_mu, source_bio_std = compute_subject_bio_baselines(
            model,
            source_baseline_loader,
            device,
            subject_de_mu=source_de_mu,
            subject_de_std=source_de_std,
            eps=args.relative_eps,
        )
        val_bio_mu, val_bio_std = compute_subject_bio_baselines(
            model,
            val_loader,
            device,
            subject_de_mu=val_de_mu,
            subject_de_std=val_de_std,
            eps=args.relative_eps,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    emotion_weights = build_emotion_class_weights(source_dataset)
    diag_weights = build_diag_class_weights(source_dataset, use_label4_for_diagnosis=not args.use_raw_diagnosis_label)

    best_trackers = build_v6_best_trackers()
    early_stop_track = args.early_stop_track
    if early_stop_track not in best_trackers:
        print(f"[V6 EarlyStop] track={early_stop_track} not found, fallback to combined")
        early_stop_track = "combined"
    early_stop_criteria = best_trackers[early_stop_track]["criteria"]
    early_best_metrics = None
    early_best_epoch = None
    bad_epochs = 0
    early_stop_enabled = int(args.early_stop_patience) > 0
    print(
        f"[V6 EarlyStop] enabled={early_stop_enabled}, track={early_stop_track}, "
        f"criteria={format_criteria(early_stop_criteria)}, patience={args.early_stop_patience}, "
        f"warmup={args.early_stop_warmup}, min_delta={args.early_stop_min_delta}"
    )

    history = []
    last_train_metrics = {}
    last_val_metrics = {}
    last_epoch = 0
    extra_state = _extra_state(domain_mapping)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_dual_router_one_epoch(
            model,
            source_loader,
            optimizer,
            device,
            lambda_mix=args.lambda_mix,
            lambda_expert=args.lambda_expert,
            lambda_shared=args.lambda_shared,
            lambda_diag=args.lambda_diag,
            emotion_class_weight=emotion_weights,
            diag_class_weight=diag_weights,
            source_de_mu=source_de_mu,
            source_de_std=source_de_std,
            source_bio_mu=source_bio_mu,
            source_bio_std=source_bio_std,
        )
        val_metrics = validate_dual_router_trial_level(
            model,
            val_loader,
            device,
            threshold=args.threshold,
            k_pos=args.k_pos,
            lambda_mix=args.lambda_mix,
            lambda_expert=args.lambda_expert,
            lambda_shared=args.lambda_shared,
            lambda_diag=args.lambda_diag,
            val_de_mu=val_de_mu,
            val_de_std=val_de_std,
            val_bio_mu=val_bio_mu,
            val_bio_std=val_bio_std,
        )
        scheduler.step()

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update(flatten_metrics_for_csv("train", train_metrics))
        row.update(flatten_metrics_for_csv("val", val_metrics))
        history.append(row)
        last_train_metrics = copy.deepcopy(train_metrics)
        last_val_metrics = copy.deepcopy(val_metrics)
        last_epoch = epoch
        pd.DataFrame(history).to_csv(save_dir / "v6_history.csv", index=False, encoding="utf-8-sig")

        print(
            f"[V6] epoch={epoch} loss={train_metrics['loss']:.4f} "
            f"loss_mix={train_metrics['loss_mix']:.4f} "
            f"loss_expert={train_metrics['loss_expert']:.4f} "
            f"loss_shared={train_metrics['loss_shared']:.4f} "
            f"loss_diag={train_metrics['loss_diag']:.4f} "
            f"train_emo_f1={train_metrics['emotion_macro_f1']:.4f} "
            f"train_diag_f1={train_metrics['diag_macro_f1']:.4f} "
            f"val_topk_f1={val_metrics['topk_trial_macro_f1']:.4f} "
            f"val_trial_f1={val_metrics['trial_macro_f1']:.4f}"
        )

        if epoch <= args.save_warmup_epochs:
            print(f"[V6] skip best checkpoint during save warmup ({epoch}/{args.save_warmup_epochs})")
        else:
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
                        train_subjects=split["train_all"],
                        val_subjects=split["val_all"],
                        criteria=criteria,
                        config_dict=config_dict,
                        scheduler=scheduler,
                        extra_state=extra_state,
                    )
                    tracker["best_metrics"] = copy.deepcopy(val_metrics)
                    tracker["best_train_metrics"] = copy.deepcopy(train_metrics)
                    tracker["best_epoch"] = epoch
                    tracker["checkpoint_path"] = ckpt_path
                    tracker["metrics_json_path"] = json_path
                    tracker["summary_csv_path"] = csv_path
                    print(f"[V6] saved best_{best_name}: epoch={epoch}, criteria={format_criteria(criteria)}, path={ckpt_path}")

        if early_stop_enabled:
            if epoch <= args.early_stop_warmup:
                if is_better_by_criteria(val_metrics, early_best_metrics, early_stop_criteria, eps=args.early_stop_min_delta):
                    early_best_metrics = copy.deepcopy(val_metrics)
                    early_best_epoch = epoch
                print(f"[V6 EarlyStop] warmup ({epoch}/{args.early_stop_warmup}), best_epoch={early_best_epoch}")
            else:
                improved = is_better_by_criteria(
                    val_metrics,
                    early_best_metrics,
                    early_stop_criteria,
                    eps=args.early_stop_min_delta,
                )
                if improved:
                    early_best_metrics = copy.deepcopy(val_metrics)
                    early_best_epoch = epoch
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                print(f"[V6 EarlyStop] improved={improved}, best_epoch={early_best_epoch}, bad_epochs={bad_epochs}/{args.early_stop_patience}")
                if bad_epochs >= args.early_stop_patience:
                    print(f"[V6 EarlyStop] stop at epoch={epoch}; best_epoch={early_best_epoch}")
                    break

    history_csv = save_dir / "v6_history.csv"
    pd.DataFrame(history).to_csv(history_csv, index=False, encoding="utf-8-sig")

    final_path = final_json = final_csv = None
    if last_epoch > 0:
        final_path, final_json, final_csv = save_final_checkpoint(
            save_dir=save_dir,
            stage_name="v6_dual_router",
            model=model,
            optimizer=optimizer,
            epoch=last_epoch,
            train_metrics=last_train_metrics,
            val_metrics=last_val_metrics,
            train_subjects=split["train_all"],
            val_subjects=split["val_all"],
            config_dict=config_dict,
            scheduler=scheduler,
            extra_state=extra_state,
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
    best_summary_csv = save_dir / f"v6_best_summary_fold{fold}.csv"
    pd.DataFrame(best_rows).to_csv(best_summary_csv, index=False, encoding="utf-8-sig")
    print(f"[V6] history saved: {history_csv}")
    print(f"[V6] best summary saved: {best_summary_csv}")

    selected_best_name = args.predict_best_name
    if selected_best_name not in best_trackers:
        print(f"[V6] predict_best_name={selected_best_name} not found, fallback to combined")
        selected_best_name = "combined"
    selected_tracker = best_trackers[selected_best_name]
    best_path = selected_tracker["checkpoint_path"]
    best_val = selected_tracker["best_metrics"]
    if best_path is None and final_path is not None:
        print(f"[V6] no best checkpoint was saved after warmup; fallback to final checkpoint for prediction: {final_path}")
        selected_best_name = "final_fallback"
        best_path = final_path
        best_val = last_val_metrics

    if args.predict_test and args.test_csv and Path(args.test_csv).exists() and best_path:
        ckpt = load_checkpoint(best_path, device)
        print_checkpoint_info(best_path, ckpt, prefix=f"[V6 predict ckpt {selected_best_name}]")
        model.load_state_dict(ckpt["model_state_dict"])
        trial_df = predict_test_trial_level(
            model,
            args.test_csv,
            domain_mapping,
            device,
            save_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold=args.threshold,
            vote_method=args.vote_method,
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
        "v6_history_csv": str(history_csv),
        "v6_best_summary_csv": str(best_summary_csv),
        "v6_final_path": final_path,
        "v6_final_metrics_json": final_json,
        "v6_final_summary_csv": final_csv,
        "best_models": best_trackers,
    }


def _parse_repeat_seeds(value: str | None) -> list[int]:
    if value is None or str(value).strip() == "":
        return [int(s) for s in getattr(config, "V2_seed", [42])]
    parts = [p.strip() for p in str(value).replace(";", ",").split(",") if p.strip()]
    return [int(p) for p in parts] if parts else [42]


def parse_args() -> argparse.Namespace: 
    parser = argparse.ArgumentParser(description=VERSION_NAME)
    parser.add_argument("--index_csv", type=str, default="com_index_sub_2s.csv")
    parser.add_argument("--test_csv", type=str, default="com_test_trial_index_2s.csv")
    parser.add_argument("--save_root", type=str, default="model_params/V6_NoSSAS_DualFeature_SoftRouter")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--repeat_seeds", type=str, default=",".join(str(s) for s in getattr(config, "V2_seed", [42])))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--biomarker_dim", type=int, default=57)
    parser.add_argument("--de_num_bands", type=int, default=5)
    parser.add_argument("--no_biomarkers", action="store_true")
    parser.add_argument("--no_subject_relative_de", action="store_true")
    parser.add_argument("--no_subject_relative_bio", action="store_true")
    parser.add_argument("--bio_abs_scale", type=float, default=0.3)
    parser.add_argument("--relative_eps", type=float, default=1e-6)
    parser.add_argument("--shared_mix_alpha", type=float, default=0.5)
    parser.add_argument("--lambda_mix", type=float, default=1.0)
    parser.add_argument("--lambda_expert", type=float, default=0.5)
    parser.add_argument("--lambda_shared", type=float, default=0.3)
    parser.add_argument("--lambda_diag", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--k_pos", type=int, default=4)
    parser.add_argument(
        "--vote_method",
        choices=["prob", "soft_threshold", "hard", "soft_topk", "topk"],
        default="soft_topk",
    )
    parser.add_argument("--predict_test", action="store_true")
    parser.add_argument("--predict_best_name", type=str, default="combined")
    parser.add_argument("--save_warmup_epochs", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_warmup", type=int, default=3)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-6)
    parser.add_argument("--early_stop_track", type=str, default="combined")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--use_raw_diagnosis_label", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repeat_seeds = _parse_repeat_seeds(args.repeat_seeds)
    folds = range(args.n_splits) if args.all_folds else [args.fold]
    results = []
    for repeat_index, rand_seed in enumerate(repeat_seeds):
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
            "v6_history_csv": result.get("v6_history_csv"),
            "v6_best_summary_csv": result.get("v6_best_summary_csv"),
            "v6_final_path": result.get("v6_final_path"),
        }
        if result["best_val"] is not None:
            row.update(flatten_metrics_for_csv("val", result["best_val"]))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(save_root / "all_fold_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[done] summary saved to {save_root / 'all_fold_summary.csv'}")

    if args.all_folds and args.predict_test:
        ensemble_test_predictions_from_fold_outputs(
            results,
            save_root=save_root,
            threshold=args.threshold,
            vote_method=args.vote_method,
            k_pos=args.k_pos,
        )


if __name__ == "__main__":
    main()
    #

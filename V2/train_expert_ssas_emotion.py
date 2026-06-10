# -*- coding: utf-8 -*-
"""Train two-stage SSAS + three-branch encoder + HC/DEP emotion experts.

Typical quick run:
    python V2/train_expert_ssas_emotion.py --fold 0 --stage1_epochs 2 --stage2_epochs 2

Full 5-fold run:
    python V2/train_expert_ssas_emotion.py --all_folds --stage1_epochs 20 --stage2_epochs 100
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from collections import Counter, defaultdict
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
from V2.expert_ssas_emotion_model import (
    Stage1SSASSourceSelectionModel,
    Stage2ExpertEmotionAdaptationModel,
    hard_expert_emotion_loss,
    mixture_emotion_nll_loss,
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


def _weighted_ce(logits: torch.Tensor, labels: torch.Tensor, sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.cross_entropy(logits, labels.long(), reduction="none")
    if sample_weight is not None:
        loss = loss * sample_weight.to(device=loss.device, dtype=loss.dtype)
    return loss.mean()


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
        source_out = model(
            source_batch["x"],
            source_batch["de_feat"],
            lambda_emo=grl_emo,
            lambda_diag=grl_diag,
        )
        target_out = model(target_batch["x"], target_batch["de_feat"], lambda_emo=0.0, lambda_diag=0.0)

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
        source_out = model(source_batch["x"], source_batch["de_feat"])
        target_out = model(target_batch["x"], target_batch["de_feat"])

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
    smooth: float = 1.0,
    vote_level: str = "window",
) -> dict:
    model.eval()
    source_subjects = [str(s) for s in domain_mapping["source_subjects"]]
    source_domain_indices = torch.tensor(domain_mapping["source_domain_indices"], dtype=torch.long, device=device)
    domain_to_source = {
        int(domain_mapping["source_subject_to_domain"][s]): str(s)
        for s in source_subjects
    }
    rows = []

    for batch in tqdm(target_loader, desc="Stage1-Voting", leave=False):
        batch = move_batch_to_device(batch, device)
        out = model(batch["x"], batch["de_feat"])
        source_logits = out["domain_logits"].index_select(dim=1, index=source_domain_indices)
        source_prob = torch.softmax(source_logits, dim=1)
        pred_local = source_prob.argmax(dim=1)
        conf = source_prob.max(dim=1).values
        pred_domain = source_domain_indices[pred_local].detach().cpu().tolist()
        pred_subjects = [domain_to_source[int(d)] for d in pred_domain]

        for i, pred_subject in enumerate(pred_subjects):
            rows.append(
                {
                    "target_key": batch["target_key"][i],
                    "user_id": batch["user_id"][i],
                    "trial_id": int(batch["trial_id"][i].detach().cpu()),
                    "pred_source_subject": pred_subject,
                    "pred_source_domain": int(pred_domain[i]),
                    "confidence": float(conf[i].detach().cpu()),
                }
            )

    if vote_level == "trial":
        trial_votes = []
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["target_key"], row["trial_id"])].append(row)
        for (_target_key, _trial_id), records in grouped.items():
            subject_counts = Counter(r["pred_source_subject"] for r in records)
            pred_subject = sorted(
                subject_counts.items(),
                key=lambda kv: (-kv[1], natural_key(kv[0])),
            )[0][0]
            trial_votes.append(pred_subject)
        counts = Counter(trial_votes)
    elif vote_level == "window":
        counts = Counter(row["pred_source_subject"] for row in rows)
    else:
        raise ValueError(f"Unknown vote_level={vote_level}")

    count_values = np.array([counts.get(s, 0) + float(smooth) for s in source_subjects], dtype=np.float64)
    weights = count_values / count_values.sum()
    weights_mean_one = weights * len(source_subjects)
    result = {
        "type": "source_classification_voting",
        "vote_level": vote_level,
        "smooth": float(smooth),
        "source_subjects": source_subjects,
        "counts": {s: int(counts.get(s, 0)) for s in source_subjects},
        "weights": {s: float(w) for s, w in zip(source_subjects, weights)},
        "weights_mean_one": {s: float(w) for s, w in zip(source_subjects, weights_mean_one)},
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_json(save_dir / "source_subject_weights.json", result)
    pd.DataFrame(rows).to_csv(save_dir / "source_subject_votes.csv", index=False, encoding="utf-8-sig")
    save_json(save_dir / "domain_id_mapping.json", domain_mapping)
    return result


def _subject_weight_tensor(
    subject_ids: torch.Tensor,
    source_weights: dict[str, float],
    num_source_subjects: int,
    device: torch.device,
) -> torch.Tensor:
    default = 1.0 / max(num_source_subjects, 1)
    values = [float(source_weights.get(str(int(s)), default)) for s in subject_ids.detach().cpu()]
    return torch.tensor(values, dtype=torch.float32, device=device)


def train_stage2_one_epoch(
    model: Stage2ExpertEmotionAdaptationModel,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    source_subject_weights: dict[str, float],
    num_source_subjects: int,
    lambda_expert: float = 1.0,
    lambda_mix: float = 0.5,
    lambda_diag: float = 0.1,
    lambda_mmd: float = 0.001,
    lambda_subject: float = 0.001,
    grl_subject: float = 0.001,
    lambda_ent: float = 0.0,
    lambda_mdc: float = 0.0,
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

    pbar = tqdm(source_loader, desc="Stage2-Train", leave=False)
    for source_batch in pbar:
        target_batch = next(target_iter)
        source_batch = move_batch_to_device(source_batch, device)
        target_batch = move_batch_to_device(target_batch, device)

        optimizer.zero_grad(set_to_none=True)
        source_out = model(source_batch["x"], source_batch["de_feat"], lambda_subject=grl_subject)
        target_out = model(target_batch["x"], target_batch["de_feat"], lambda_subject=grl_subject)

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
        loss_diag = _weighted_ce(source_out["diag_logits"], y_diag, sample_weight=sample_weight)

        domain_logits = torch.cat(
            [source_out["subject_domain_logits"], target_out["subject_domain_logits"]],
            dim=0,
        )
        domain_labels = torch.cat([source_batch["domain_id"].long(), target_batch["domain_id"].long()], dim=0)
        loss_subject_domain = F.cross_entropy(domain_logits, domain_labels)
        loss_mmd = weighted_mmd_rbf(source_out["z_mmd"], target_out["z_mmd"], source_weight=sample_weight)
        loss_ent = target_entropy_loss(target_out["mix_prob"]) if lambda_ent > 0 else source_out["z"].new_tensor(0.0)
        loss_mdc = source_out["z"].new_tensor(0.0)

        loss = (
            lambda_expert * loss_expert
            + lambda_mix * loss_mix
            + lambda_diag * loss_diag
            + lambda_mmd * loss_mmd
            + lambda_subject * loss_subject_domain
            + lambda_ent * loss_ent
            + lambda_mdc * loss_mdc
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
        totals["loss_subject_domain"] += float(loss_subject_domain.item()) * bsz
        totals["loss_ent"] += float(loss_ent.item()) * bsz
        totals["loss_mdc"] += float(loss_mdc.item()) * bsz
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
        "loss_subject_domain": totals["loss_subject_domain"] / total_n,
        "loss_ent": totals["loss_ent"] / total_n,
        "loss_mdc": totals["loss_mdc"] / total_n,
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


@torch.no_grad()
def validate_stage2_trial_level(
    model: Stage2ExpertEmotionAdaptationModel,
    val_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_preds, all_labels = [], []
    all_diag_preds, all_diag_labels = [], []
    segment_records = []

    for batch in tqdm(val_loader, desc="Stage2-Val", leave=False):
        batch = move_batch_to_device(batch, device)
        out = model(batch["x"], batch["de_feat"], lambda_subject=0.0)
        y_emo = batch["emotion_label"].long()
        y_diag = batch["diagnosis_label"].long()

        loss_mix = mixture_emotion_nll_loss(out["mix_prob"], y_emo)
        loss_diag = F.cross_entropy(out["diag_logits"], y_diag)
        loss = loss_mix + 0.1 * loss_diag
        bsz = batch["x"].size(0)
        total_loss += float(loss.item()) * bsz
        total_n += bsz

        prob_pos = out["mix_prob"][:, 1]
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
                    "trial_id": int(batch["trial_id"][i].detach().cpu()),
                    "prob_pos": float(prob_pos[i].detach().cpu()),
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

    # V1-compatible aliases for checkpoint criteria and summary tables.
    metrics["emotion_acc"] = metrics["segment_acc"]
    metrics["emotion_macro_f1"] = metrics["segment_macro_f1"]
    metrics["acc"] = metrics["segment_acc"]
    metrics["macro_f1"] = metrics["segment_macro_f1"]
    metrics["trial_emotion_acc"] = metrics["trial_acc"]
    metrics["trial_emotion_macro_f1"] = metrics["trial_macro_f1"]
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
                ("trial_macro_f1", "max"),
                ("trial_acc", "max"),
                ("emotion_macro_f1", "max"),
                ("emotion_acc", "max"),
                ("diag_macro_f1", "max"),
                ("diag_acc", "max"),
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
) -> pd.DataFrame:
    dataset = UnlabeledTargetDataset(test_csv, domain_mapping=domain_mapping, root=ROOT, normalize=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dict_collate,
    )
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="Test-Predict", leave=False):
        batch = move_batch_to_device(batch, device)
        out = model(batch["x"], batch["de_feat"], lambda_subject=0.0)
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

    stage1 = Stage1SSASSourceSelectionModel(
        num_domains=domain_mapping["num_domains"],
        sfreq=args.sfreq,
        topk=args.topk,
        dropout=args.dropout,
        use_biomarkers=not args.no_biomarkers,
        biomarker_dim=args.biomarker_dim,
        de_num_bands=args.de_num_bands,
    ).to(device)
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
        )
        val_stage1 = validate_stage1(stage1, source_loader, target_train_loader, device)
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
    )
    source_weights = vote_result["weights_mean_one"] if args.use_mean_one_source_weights else vote_result["weights"]
    top_weights = sorted(source_weights.items(), key=lambda kv: kv[1], reverse=True)[:8]
    print(f"[Stage1] top source weights: {top_weights}")

    stage2 = Stage2ExpertEmotionAdaptationModel(
        num_domains=domain_mapping["num_domains"],
        sfreq=args.sfreq,
        topk=args.topk,
        dropout=args.dropout,
        use_biomarkers=not args.no_biomarkers,
        biomarker_dim=args.biomarker_dim,
        de_num_bands=args.de_num_bands,
    ).to(device)
    if not args.no_stage1_init:
        load_stage1_encoder_into_stage2(stage1, stage2)

    optimizer2 = torch.optim.AdamW(stage2.parameters(), lr=args.lr_stage2, weight_decay=args.weight_decay)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=max(args.stage2_epochs, 1))
    best_trackers = build_stage2_best_trackers()
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
            lambda_mmd=args.stage2_lambda_mmd,
            lambda_subject=args.lambda_subject,
            grl_subject=args.grl_subject,
            lambda_ent=args.lambda_ent,
            lambda_mdc=args.lambda_mdc,
        )
        val_metrics = validate_stage2_trial_level(stage2, val_loader, device, threshold=args.threshold)
        scheduler2.step()
        row = {"epoch": epoch, "lr": optimizer2.param_groups[0]["lr"]}
        row.update(flatten_metrics_for_csv("train", train_metrics))
        row.update(flatten_metrics_for_csv("val", val_metrics))
        history2.append(row)
        last_stage2_train_metrics = copy.deepcopy(train_metrics)
        last_stage2_val_metrics = copy.deepcopy(val_metrics)
        last_stage2_epoch = epoch
        print(
            f"[Stage2] epoch={epoch} loss={train_metrics['loss']:.4f} "
            f"train_emo_f1={train_metrics['emotion_macro_f1']:.4f} "
            f"train_diag_f1={train_metrics['diag_macro_f1']:.4f} "
            f"val_trial_f1={val_metrics['trial_macro_f1']:.4f} val_trial_acc={val_metrics['trial_acc']:.4f}"
        )

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
                            "source_subject_weights": source_weights,
                            "source_weight_vote": vote_result,
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
                "source_subject_weights": source_weights,
                "source_weight_vote": vote_result,
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
        ckpt = torch.load(best_path, map_location=device)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", type=str, default="com_index_sub_2s.csv")
    parser.add_argument("--test_csv", type=str, default="com_test_trial_index_2s.csv")
    parser.add_argument("--save_root", type=str, default="model_params/V2_expert_ssas")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--all_repeats", action="store_true")
    parser.add_argument("--stage1_epochs", type=int, default=20)
    parser.add_argument("--stage2_epochs", type=int, default=100)
    parser.add_argument("--save_warmup_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--lr_stage1", type=float, default=1e-4)
    parser.add_argument("--lr_stage2", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--biomarker_dim", type=int, default=57)
    parser.add_argument("--de_num_bands", type=int, default=5)
    parser.add_argument("--no_biomarkers", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--use_raw_diagnosis_label", action="store_true")

    parser.add_argument("--lambda_domain", type=float, default=1.0)
    parser.add_argument("--stage1_lambda_mmd", type=float, default=0.03)
    parser.add_argument("--lambda_emo_grl", type=float, default=0.001)
    parser.add_argument("--grl_emo", type=float, default=0.01)
    parser.add_argument("--lambda_diag_grl", type=float, default=0.001)
    parser.add_argument("--grl_diag", type=float, default=0.01)
    parser.add_argument("--lambda_weight_reg", type=float, default=0.0)
    parser.add_argument("--vote_smooth", type=float, default=1.0)
    parser.add_argument("--vote_level", choices=["window", "trial"], default="window")
    parser.add_argument("--use_mean_one_source_weights", action="store_true")

    parser.add_argument("--lambda_expert", type=float, default=1.0)
    parser.add_argument("--lambda_mix", type=float, default=0.5)
    parser.add_argument("--lambda_diag", type=float, default=0.1)
    parser.add_argument("--stage2_lambda_mmd", type=float, default=0.001)
    parser.add_argument("--lambda_subject", type=float, default=0.001)
    parser.add_argument("--grl_subject", type=float, default=0.001)
    parser.add_argument("--lambda_ent", type=float, default=0.0)
    parser.add_argument("--lambda_mdc", type=float, default=0.0)
    parser.add_argument("--no_stage1_init", action="store_true")

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--predict_test", action="store_true")
    parser.add_argument("--predict_best_name", type=str, default="combined")
    parser.add_argument(
        "--test_vote_method",
        choices=["prob", "soft_threshold", "hard", "soft_topk", "topk"],
        default="soft_topk",
    )
    parser.add_argument("--k_pos", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repeats = range(len(config.DIAG_REPEAT_SEEDS)) if args.all_repeats else [args.repeat]
    folds = range(args.n_splits) if args.all_folds else [args.fold]
    results = []
    for repeat_index in repeats:
        rand_seed = config.get_seed_for_repeat("diag", repeat_index)
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


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Train and predict a DE-indexed three-branch emotion baseline.

Main objective:
    L = CE(emotion_2cls) + aux_weight * CE(label4)

Default aux_weight is 0.5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.de_three_branch_model import DEThreeBranchEmotionClassifier, build_model_from_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_data_path(path_value: Any, root: Path = ROOT) -> Path:
    text = str(path_value).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() and path.exists():
        return path

    cleaned = text
    while cleaned.startswith("../"):
        cleaned = cleaned[3:]
    cleaned_path = Path(cleaned)

    candidates = [
        root / path,
        root / cleaned_path,
        root / "data" / path.name,
        root / "testdata" / path.name,
    ]
    if cleaned.startswith("data/"):
        without_data = cleaned[len("data/") :]
        candidates.extend(
            [
                root / cleaned,
                root / without_data,
                root / "data" / without_data,
            ]
        )
    candidates.append(root / "data" / cleaned_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Data file not found: {path_value}")


def stable_int_id(value: Any) -> int:
    text = str(value)
    if text.lstrip("-").isdigit():
        return int(text)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


class DEWindowDataset(Dataset):
    """Window dataset reading raw EEG windows and their aligned DE features."""

    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        subject_ids: Iterable[Any] | None = None,
        root: Path = ROOT,
        normalize: bool = True,
        has_labels: bool = True,
    ) -> None:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            self.index_csv = None
        else:
            self.index_csv = Path(data)
            if not self.index_csv.is_absolute():
                self.index_csv = root / self.index_csv
            df = pd.read_csv(self.index_csv)

        if subject_ids is not None and "subject_id" in df.columns:
            subjects = {str(s) for s in subject_ids}
            df = df[df["subject_id"].astype(str).isin(subjects)].copy()

        if df.empty:
            raise ValueError("No rows left for DEWindowDataset.")
        if "de_path" not in df.columns:
            raise KeyError("CSV/DataFrame must contain de_path.")
        if "trial_path" not in df.columns:
            raise KeyError("CSV/DataFrame must contain trial_path for the three-branch encoder.")

        self.df = df.reset_index(drop=True)
        self.root = root
        self.normalize = bool(normalize)
        self.has_labels = bool(has_labels)
        self._trial_cache: dict[str, np.ndarray] = {}
        self._de_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_cached(self, cache: dict[str, np.ndarray], path_value: Any) -> np.ndarray:
        key = str(path_value)
        if key not in cache:
            cache[key] = np.load(resolve_data_path(path_value, self.root), mmap_mode="r")
        return cache[key]

    @staticmethod
    def _valid_int(row: pd.Series, key: str) -> int | None:
        if key not in row.index:
            return None
        value = row[key]
        if pd.isna(value):
            return None
        return int(value)

    def _load_x(self, row: pd.Series) -> np.ndarray:
        trial = self._load_cached(self._trial_cache, row["trial_path"])
        start = self._valid_int(row, "start")
        end = self._valid_int(row, "end")

        if start is None or end is None:
            trial_start = self._valid_int(row, "trial_start")
            trial_end = self._valid_int(row, "trial_end")
            if trial_start is not None and trial_end is not None and 0 <= trial_start < trial_end <= trial.shape[-1]:
                start, end = trial_start, trial_end
            else:
                start, end = 0, trial.shape[-1]

        x = np.asarray(trial[:, start:end], dtype=np.float32)
        if x.shape[-1] == 0:
            x = np.asarray(trial, dtype=np.float32)
        if self.normalize:
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-6
            x = (x - mean) / std
        return np.ascontiguousarray(x, dtype=np.float32)

    def _load_de(self, row: pd.Series) -> np.ndarray:
        de_arr = self._load_cached(self._de_cache, row["de_path"])
        win_id = self._valid_int(row, "de_win_id")
        if de_arr.ndim >= 3:
            if win_id is None:
                win_id = 0
            de_feat = de_arr[int(win_id)]
        else:
            de_feat = de_arr
        return np.ascontiguousarray(de_feat, dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        x = self._load_x(row)
        de_feat = self._load_de(row)

        label4 = int(row["label4"]) if self.has_labels and "label4" in row.index else 0
        if self.has_labels and "emotion_label" in row.index:
            emotion_label = int(row["emotion_label"])
        else:
            emotion_label = int(label4 % 2)

        subject_value = row.get("subject_id", row.get("subject_number", row.get("user_id", 0)))
        user_id = str(row.get("user_id", subject_value))
        trial_id = int(row.get("trial_id", 0))

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "de_feat": torch.tensor(de_feat, dtype=torch.float32),
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
            "label4": torch.tensor(label4, dtype=torch.long),
            "subject_id": torch.tensor(stable_int_id(subject_value), dtype=torch.long),
            "user_id": user_id,
            "trial_id": torch.tensor(trial_id, dtype=torch.long),
        }


def prepare_test_df(test_csv: str | Path) -> pd.DataFrame:
    path = Path(test_csv)
    if not path.is_absolute():
        path = ROOT / path
    df = pd.read_csv(path)
    if "de_win_id" in df.columns:
        return df
    if "n_windows" not in df.columns:
        raise RuntimeError("test_csv must contain de_win_id or n_windows.")

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        n_windows = int(row["n_windows"])
        for win_id in range(n_windows):
            item = row.to_dict()
            item["de_win_id"] = win_id
            rows.append(item)
    return pd.DataFrame(rows)


def collate_with_meta(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = ["x", "de_feat", "emotion_label", "label4", "subject_id", "trial_id"]
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    out["user_id"] = [item["user_id"] for item in batch]
    return out


def split_subjects(df: pd.DataFrame, val_ratio: float, seed: int) -> tuple[list[Any], list[Any]]:
    if "subject_id" not in df.columns:
        raise KeyError("Training CSV must contain subject_id for subject-level split.")

    subject_df = df.drop_duplicates("subject_id").copy()
    subjects = subject_df["subject_id"].tolist()
    if len(subjects) < 2:
        raise ValueError("Need at least two subjects for train/validation split.")

    if "label4" in subject_df.columns:
        stratify = (subject_df["label4"].astype(int) >= 2).astype(int)
    elif "diagnosis_label" in subject_df.columns:
        stratify = subject_df["diagnosis_label"].astype(int)
    else:
        stratify = None

    if stratify is not None and (stratify.value_counts().min() < 2 or stratify.nunique() < 2):
        stratify = None

    train_subjects, val_subjects = train_test_split(
        subjects,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return list(train_subjects), list(val_subjects)


def class_weight_from_labels(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = labels.astype(int)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def metric_block(y_true: list[int], y_pred: list[int], labels: list[int], prefix: str) -> dict[str, Any]:
    if not y_true:
        return {
            f"{prefix}_acc": 0.0,
            f"{prefix}_macro_f1": 0.0,
            f"{prefix}_confusion_matrix": np.zeros((len(labels), len(labels)), dtype=int),
            f"{prefix}_per_class": {},
        }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return {
        f"{prefix}_acc": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        f"{prefix}_confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        f"{prefix}_per_class": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
    }


def apply_subject_topk(trial_df: pd.DataFrame, k_pos: int = 4) -> pd.Series:
    labels = pd.Series(0, index=trial_df.index, dtype=int)
    subject_col = "user_id" if "user_id" in trial_df.columns else "subject_id"
    for _, group in trial_df.groupby(subject_col):
        cur_k = int(k_pos) if len(group) == 8 else max(1, int(round(len(group) * 0.5)))
        cur_k = max(0, min(cur_k, len(group)))
        top_index = group.sort_values("score_pos", ascending=False).head(cur_k).index
        labels.loc[top_index] = 1
    return labels


def trial_metrics(records: list[dict[str, Any]], k_pos: int) -> tuple[dict[str, Any], pd.DataFrame]:
    if not records:
        empty = pd.DataFrame()
        return {
            "trial_acc": 0.0,
            "trial_macro_f1": 0.0,
            "trial_confusion_matrix": np.zeros((2, 2), dtype=int),
            "topk_trial_acc": 0.0,
            "topk_trial_macro_f1": 0.0,
            "topk_trial_confusion_matrix": np.zeros((2, 2), dtype=int),
        }, empty

    df = pd.DataFrame(records)
    trial_df = (
        df.groupby(["user_id", "subject_id", "trial_id"], as_index=False)
        .agg(
            emotion_label=("emotion_label", "first"),
            prob_pos=("prob_pos", "mean"),
            score_pos=("score_pos", "mean"),
            pred_hard_rate=("pred_window", "mean"),
            n_windows=("prob_pos", "count"),
        )
    )
    trial_df["pred_threshold"] = (trial_df["prob_pos"] >= 0.5).astype(int)
    trial_df["pred_topk"] = apply_subject_topk(trial_df, k_pos=k_pos)

    y_true = trial_df["emotion_label"].astype(int).tolist()
    y_threshold = trial_df["pred_threshold"].astype(int).tolist()
    y_topk = trial_df["pred_topk"].astype(int).tolist()

    metrics = metric_block(y_true, y_threshold, labels=[0, 1], prefix="trial")
    topk = metric_block(y_true, y_topk, labels=[0, 1], prefix="topk_trial")
    metrics.update(topk)
    return metrics, trial_df


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    aux_weight: float,
    weight_2cls: torch.Tensor | None,
    weight_4cls: torch.Tensor | None,
    amp: bool,
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_loss_2 = 0.0
    total_loss_4 = 0.0
    total_n = 0
    y2_true: list[int] = []
    y2_pred: list[int] = []
    y4_true: list[int] = []
    y4_pred: list[int] = []

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        de_feat = batch["de_feat"].to(device, non_blocking=True)
        y2 = batch["emotion_label"].to(device, non_blocking=True).long()
        y4 = batch["label4"].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            out = model(x, de_feat)
            loss_2 = F.cross_entropy(out["logits_2cls"], y2, weight=weight_2cls)
            loss_4 = F.cross_entropy(out["logits_4cls"], y4, weight=weight_4cls)
            loss = loss_2 + float(aux_weight) * loss_4

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        bs = int(y2.size(0))
        total_loss += float(loss.detach().item()) * bs
        total_loss_2 += float(loss_2.detach().item()) * bs
        total_loss_4 += float(loss_4.detach().item()) * bs
        total_n += bs

        y2_true.extend(y2.detach().cpu().tolist())
        y2_pred.extend(out["logits_2cls"].argmax(dim=1).detach().cpu().tolist())
        y4_true.extend(y4.detach().cpu().tolist())
        y4_pred.extend(out["logits_4cls"].argmax(dim=1).detach().cpu().tolist())

    metrics = {
        "loss": total_loss / max(total_n, 1),
        "loss_2cls": total_loss_2 / max(total_n, 1),
        "loss_4cls": total_loss_4 / max(total_n, 1),
    }
    metrics.update(metric_block(y2_true, y2_pred, labels=[0, 1], prefix="emotion"))
    metrics.update(metric_block(y4_true, y4_pred, labels=[0, 1, 2, 3], prefix="four_class"))
    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    aux_weight: float,
    weight_2cls: torch.Tensor | None,
    weight_4cls: torch.Tensor | None,
    k_pos: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_loss_2 = 0.0
    total_loss_4 = 0.0
    total_n = 0
    y2_true: list[int] = []
    y2_pred: list[int] = []
    y4_true: list[int] = []
    y4_pred: list[int] = []
    records: list[dict[str, Any]] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        de_feat = batch["de_feat"].to(device, non_blocking=True)
        y2 = batch["emotion_label"].to(device, non_blocking=True).long()
        y4 = batch["label4"].to(device, non_blocking=True).long()

        out = model(x, de_feat)
        loss_2 = F.cross_entropy(out["logits_2cls"], y2, weight=weight_2cls)
        loss_4 = F.cross_entropy(out["logits_4cls"], y4, weight=weight_4cls)
        loss = loss_2 + float(aux_weight) * loss_4

        p2 = torch.softmax(out["logits_2cls"], dim=1)
        pred2 = p2.argmax(dim=1)
        pred4 = out["logits_4cls"].argmax(dim=1)
        score_pos = out["logits_2cls"][:, 1] - out["logits_2cls"][:, 0]

        bs = int(y2.size(0))
        total_loss += float(loss.item()) * bs
        total_loss_2 += float(loss_2.item()) * bs
        total_loss_4 += float(loss_4.item()) * bs
        total_n += bs

        y2_true.extend(y2.cpu().tolist())
        y2_pred.extend(pred2.cpu().tolist())
        y4_true.extend(y4.cpu().tolist())
        y4_pred.extend(pred4.cpu().tolist())

        for i in range(bs):
            records.append(
                {
                    "user_id": batch["user_id"][i],
                    "subject_id": int(batch["subject_id"][i].cpu().item()),
                    "trial_id": int(batch["trial_id"][i].cpu().item()),
                    "emotion_label": int(y2[i].cpu().item()),
                    "prob_pos": float(p2[i, 1].cpu().item()),
                    "score_pos": float(score_pos[i].cpu().item()),
                    "pred_window": int(pred2[i].cpu().item()),
                }
            )

    metrics = {
        "loss": total_loss / max(total_n, 1),
        "loss_2cls": total_loss_2 / max(total_n, 1),
        "loss_4cls": total_loss_4 / max(total_n, 1),
    }
    metrics.update(metric_block(y2_true, y2_pred, labels=[0, 1], prefix="emotion"))
    metrics.update(metric_block(y4_true, y4_pred, labels=[0, 1, 2, 3], prefix="four_class"))
    trial_metric, trial_df = trial_metrics(records, k_pos=k_pos)
    metrics.update(trial_metric)
    metrics["selection_score"] = metrics["trial_macro_f1"]
    return metrics, trial_df


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    metrics: dict[str, Any],
    model_config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "metrics": to_jsonable(metrics),
        "model_config": to_jsonable(model_config),
        "args": to_jsonable(vars(args)),
    }
    torch.save(payload, path)


def save_metrics_json(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(metrics), f, ensure_ascii=False, indent=2)


def load_model_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[DEThreeBranchEmotionClassifier, dict[str, Any]]:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("model_config", {})
    model = build_model_from_config(config).to(device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    return model, ckpt


def build_model_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sfreq": args.sfreq,
        "topk": args.topk,
        "dropout": args.dropout,
        "use_biomarkers": not args.no_biomarkers,
        "biomarker_dim": args.biomarker_dim,
        "de_num_bands": args.de_num_bands,
        "head_hidden_dim": args.head_hidden_dim,
    }


def train(args: argparse.Namespace, device: torch.device) -> Path:
    df = pd.read_csv(ROOT / args.index_csv if not Path(args.index_csv).is_absolute() else args.index_csv)
    train_subjects, val_subjects = split_subjects(df, val_ratio=args.val_ratio, seed=args.seed)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "split_subjects.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"train_subjects": [str(s) for s in train_subjects], "val_subjects": [str(s) for s in val_subjects]},
            f,
            ensure_ascii=False,
            indent=2,
        )

    train_ds = DEWindowDataset(args.index_csv, subject_ids=train_subjects, normalize=not args.no_normalize)
    val_ds = DEWindowDataset(args.index_csv, subject_ids=val_subjects, normalize=not args.no_normalize)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_with_meta,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_with_meta,
        drop_last=False,
    )

    model_config = build_model_config(args)
    model = DEThreeBranchEmotionClassifier(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    if args.use_class_weights:
        train_df = train_ds.df
        y2 = train_df["emotion_label"].astype(int).to_numpy() if "emotion_label" in train_df.columns else (train_df["label4"].astype(int).to_numpy() % 2)
        y4 = train_df["label4"].astype(int).to_numpy()
        weight_2cls = class_weight_from_labels(y2, num_classes=2, device=device)
        weight_4cls = class_weight_from_labels(y4, num_classes=4, device=device)
    else:
        weight_2cls = None
        weight_4cls = None

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_path = output_dir / "best_de_three_branch.pt"
    final_path = output_dir / "final_de_three_branch.pt"

    print(f"[data] train_windows={len(train_ds)} val_windows={len(val_ds)}")
    print(f"[loss] L = CE_2cls + {args.aux_weight} * CE_4cls")
    print(f"[save] warmup epochs before best checkpoint: {args.save_warmup_epochs}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            aux_weight=args.aux_weight,
            weight_2cls=weight_2cls,
            weight_4cls=weight_4cls,
            amp=args.amp and device.type == "cuda",
        )
        val_metrics, val_trial_df = evaluate(
            model,
            val_loader,
            device,
            aux_weight=args.aux_weight,
            weight_2cls=weight_2cls,
            weight_4cls=weight_4cls,
            k_pos=args.k_pos,
        )
        scheduler.step()

        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        row.update({f"train_{k}": v for k, v in train_metrics.items() if not isinstance(v, (dict, np.ndarray))})
        row.update({f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, (dict, np.ndarray))})
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")
        save_metrics_json(output_dir / "last_val_metrics.json", val_metrics)
        val_trial_df.to_csv(output_dir / "last_val_trial_probs.csv", index=False, encoding="utf-8-sig")

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_emo_f1={train_metrics['emotion_macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_emo_f1={val_metrics['emotion_macro_f1']:.4f} "
            f"val_trial_f1={val_metrics['trial_macro_f1']:.4f} "
            f"val_4cls_f1={val_metrics['four_class_macro_f1']:.4f}"
        )

        if epoch <= args.save_warmup_epochs:
            print(f"[save] skip best checkpoint during warmup ({epoch}/{args.save_warmup_epochs})")
            continue

        score = float(val_metrics["selection_score"])
        if score > best_score:
            best_score = score
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, val_metrics, model_config, args)
            save_metrics_json(output_dir / "best_metrics.json", val_metrics)
            val_trial_df.to_csv(output_dir / "best_val_trial_probs.csv", index=False, encoding="utf-8-sig")
            print(f"[save] best updated: score={best_score:.4f}, path={best_path}")

    final_metrics, final_trial_df = evaluate(
        model,
        val_loader,
        device,
        aux_weight=args.aux_weight,
        weight_2cls=weight_2cls,
        weight_4cls=weight_4cls,
        k_pos=args.k_pos,
    )
    save_checkpoint(final_path, model, optimizer, scheduler, args.epochs, final_metrics, model_config, args)
    save_metrics_json(output_dir / "final_metrics.json", final_metrics)
    final_trial_df.to_csv(output_dir / "final_val_trial_probs.csv", index=False, encoding="utf-8-sig")

    if not best_path.exists():
        print(f"[save] no best checkpoint after warmup, use final checkpoint: {final_path}")
        return final_path
    return best_path


@torch.no_grad()
def predict_test(
    model: torch.nn.Module,
    test_csv: str | Path,
    output_dir: str | Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    normalize: bool,
    k_pos: int,
    vote_method: str,
) -> None:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    test_df = prepare_test_df(test_csv)
    dataset = DEWindowDataset(test_df, normalize=normalize, has_labels=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_with_meta,
    )

    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        de_feat = batch["de_feat"].to(device, non_blocking=True)
        out = model(x, de_feat)
        p2 = torch.softmax(out["logits_2cls"], dim=1)
        p4 = torch.softmax(out["logits_4cls"], dim=1)
        score_pos = out["logits_2cls"][:, 1] - out["logits_2cls"][:, 0]
        pred_window = p2.argmax(dim=1)

        for i in range(x.size(0)):
            row = {
                "user_id": batch["user_id"][i],
                "subject_id": int(batch["subject_id"][i].cpu().item()),
                "trial_id": int(batch["trial_id"][i].cpu().item()),
                "prob_pos": float(p2[i, 1].cpu().item()),
                "score_pos": float(score_pos[i].cpu().item()),
                "pred_window": int(pred_window[i].cpu().item()),
            }
            for cls in range(4):
                row[f"prob4_{cls}"] = float(p4[i, cls].cpu().item())
            rows.append(row)

    window_df = pd.DataFrame(rows)
    window_df.to_csv(output_path / "test_window_probs.csv", index=False, encoding="utf-8-sig")

    agg_dict: dict[str, Any] = {
        "subject_id": ("subject_id", "first"),
        "prob_pos": ("prob_pos", "mean"),
        "score_pos": ("score_pos", "mean"),
        "pred_hard_rate": ("pred_window", "mean"),
        "n_windows": ("prob_pos", "count"),
    }
    for cls in range(4):
        agg_dict[f"prob4_{cls}"] = (f"prob4_{cls}", "mean")

    trial_df = window_df.groupby(["user_id", "trial_id"], as_index=False).agg(**agg_dict)
    trial_df["pred_soft_threshold"] = (trial_df["prob_pos"] >= 0.5).astype(int)
    trial_df["pred_hard_threshold"] = (trial_df["pred_hard_rate"] >= 0.5).astype(int)
    trial_df["pred_soft_topk"] = apply_subject_topk(trial_df, k_pos=k_pos)

    if vote_method in {"soft_threshold", "prob"}:
        trial_df["Emotion_label"] = trial_df["pred_soft_threshold"]
    elif vote_method in {"hard_threshold", "hard"}:
        trial_df["Emotion_label"] = trial_df["pred_hard_threshold"]
    elif vote_method in {"soft_topk", "topk"}:
        trial_df["Emotion_label"] = trial_df["pred_soft_topk"]
    else:
        raise ValueError(f"Unknown vote_method: {vote_method}")

    trial_df.to_csv(output_path / "test_trial_probs.csv", index=False, encoding="utf-8-sig")

    soft_threshold = trial_df[["user_id", "trial_id", "pred_soft_threshold"]].rename(
        columns={"pred_soft_threshold": "Emotion_label"}
    )
    soft_topk = trial_df[["user_id", "trial_id", "pred_soft_topk"]].rename(
        columns={"pred_soft_topk": "Emotion_label"}
    )
    hard_threshold = trial_df[["user_id", "trial_id", "pred_hard_threshold"]].rename(
        columns={"pred_hard_threshold": "Emotion_label"}
    )
    selected = trial_df[["user_id", "trial_id", "Emotion_label"]]

    soft_threshold.to_csv(output_path / "submission_soft_threshold.csv", index=False, encoding="utf-8-sig")
    soft_topk.to_csv(output_path / "submission_soft_topk.csv", index=False, encoding="utf-8-sig")
    hard_threshold.to_csv(output_path / "submission_hard_threshold.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(output_path / "submission_baseline_de_three_branch.csv", index=False, encoding="utf-8-sig")

    print(
        f"[predict] saved to {output_path}; "
        f"selected={vote_method}, positives={int(selected['Emotion_label'].sum())}/{len(selected)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DE three-branch encoder emotion baseline")
    parser.add_argument("--index_csv", type=str, default="com_index_sub_2s.csv")
    parser.add_argument("--test_csv", type=str, default="com_test_trial_index_2s.csv")
    parser.add_argument("--output_dir", type=str, default="baseline/runs/de_three_branch")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--predict_only", action="store_true")
    parser.add_argument("--predict_test", action="store_true")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--aux_weight", type=float, default=0.5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--save_warmup_epochs", type=int, default=1)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head_hidden_dim", type=int, default=128)
    parser.add_argument("--biomarker_dim", type=int, default=57)
    parser.add_argument("--de_num_bands", type=int, default=5)
    parser.add_argument("--no_biomarkers", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")

    parser.add_argument("--k_pos", type=int, default=4)
    parser.add_argument(
        "--test_vote_method",
        type=str,
        default="soft_topk",
        choices=["prob", "soft_threshold", "hard", "hard_threshold", "soft_topk", "topk"],
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[device] {device}")

    if args.predict_only:
        if not args.checkpoint:
            raise ValueError("--predict_only requires --checkpoint")
        model, ckpt = load_model_checkpoint(args.checkpoint, device)
        print(f"[load] checkpoint={args.checkpoint}, epoch={ckpt.get('epoch')}")
        predict_test(
            model,
            args.test_csv,
            args.output_dir,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            normalize=not args.no_normalize,
            k_pos=args.k_pos,
            vote_method=args.test_vote_method,
        )
        return

    best_or_final = train(args, device)
    if args.predict_test:
        model, ckpt = load_model_checkpoint(best_or_final, device)
        print(f"[load] prediction checkpoint={best_or_final}, epoch={ckpt.get('epoch')}")
        predict_test(
            model,
            args.test_csv,
            args.output_dir,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            normalize=not args.no_normalize,
            k_pos=args.k_pos,
            vote_method=args.test_vote_method,
        )


if __name__ == "__main__":
    main()


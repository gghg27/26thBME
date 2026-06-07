from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data import EEGWindowDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_loader(
    df: pd.DataFrame,
    *,
    label_col: str,
    root: Path,
    batch_size: int,
    shuffle: bool,
):
    from torch.utils.data import DataLoader

    dataset = EEGWindowDataset(df, label_col=label_col, root=root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, optimizer, criterion, device: str) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * y.numel()
        total_count += int(y.numel())
    return total_loss / total_count if total_count else 0.0


def collect_probabilities(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            if logits.ndim == 1 or logits.shape[-1] == 1:
                batch_probs = torch.sigmoid(logits.reshape(-1))
            else:
                batch_probs = torch.softmax(logits, dim=-1)[:, 1]
            probs.append(batch_probs.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
    if not probs:
        return np.array([], dtype="float32"), np.array([], dtype="int64")
    return np.concatenate(probs), np.concatenate(labels)


def evaluate_window_acc(model, loader, device: str) -> float:
    probs, labels = collect_probabilities(model, loader, device)
    if labels.size == 0:
        return 0.0
    preds = (probs >= 0.5).astype("int64")
    return float((preds == labels).mean())


def evaluate_group_acc(
    model,
    loader,
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    label_col: str,
    device: str,
) -> float:
    probs, _ = collect_probabilities(model, loader, device)
    if probs.size == 0:
        return 0.0
    tmp = df.reset_index(drop=True).copy()
    tmp["prob"] = probs
    grouped = tmp.groupby(group_cols, as_index=False).agg(
        prob=("prob", "mean"), true=(label_col, "first")
    )
    preds = (grouped["prob"].to_numpy() >= 0.5).astype("int64")
    return float((preds == grouped["true"].astype(int).to_numpy()).mean())

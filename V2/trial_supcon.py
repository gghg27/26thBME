# -*- coding: utf-8 -*-
"""Trial-level sampling and SupCon utilities for the V2_conf_trial_supcon run."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


TRIAL_SCALAR_KEYS = {
    "label4",
    "emotion_label",
    "diagnosis_label",
    "subject_id",
    "domain_id",
    "trial_id",
}


class TrialWindowDataset(Dataset):
    """Groups an existing window-level dataset into trial-level K-window samples."""

    def __init__(
        self,
        window_dataset: Dataset,
        num_windows_per_trial: int = 5,
        train: bool = True,
    ) -> None:
        if not hasattr(window_dataset, "df"):
            raise AttributeError("TrialWindowDataset expects a window dataset with a pandas df attribute.")
        self.window_dataset = window_dataset
        self.num_windows_per_trial = int(num_windows_per_trial)
        self.train = bool(train)
        self.df = window_dataset.df.reset_index(drop=True)
        self.trial_groups = self._build_trial_groups()
        if not self.trial_groups:
            raise ValueError("No trial groups were built from the window-level dataset.")

    def _row_int(self, row: Any, key: str, default: int = 0) -> int:
        if key in row.index and not pd.isna(row[key]):
            return int(row[key])
        return int(default)

    def _build_trial_groups(self) -> list[dict[str, Any]]:
        grouped: dict[tuple[int, int, int], dict[str, Any]] = {}
        use_label4_for_diagnosis = bool(getattr(self.window_dataset, "use_label4_for_diagnosis", False))

        for idx, row in self.df.iterrows():
            subject_id = self._row_int(row, "subject_id")
            trial_id = self._row_int(row, "trial_id")
            if "label4" in row.index:
                label4 = int(row["label4"])
            else:
                emotion = self._row_int(row, "emotion_label")
                diagnosis = self._row_int(row, "diagnosis_label")
                label4 = diagnosis * 2 + emotion
            emotion_label = self._row_int(row, "emotion_label", default=label4 % 2)
            if use_label4_for_diagnosis:
                diagnosis_label = int(label4 >= 2)
            else:
                diagnosis_label = self._row_int(row, "diagnosis_label", default=int(label4 >= 2))

            key = (subject_id, trial_id, label4)
            if key not in grouped:
                grouped[key] = {
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                    "label4": label4,
                    "emotion_label": emotion_label,
                    "diagnosis_label": diagnosis_label,
                    "window_indices": [],
                }
            grouped[key]["window_indices"].append(int(idx))

        return list(grouped.values())

    def __len__(self) -> int:
        return len(self.trial_groups)

    def _sample_window_indices(self, window_indices: list[int]) -> list[int]:
        k = self.num_windows_per_trial
        if self.train:
            if len(window_indices) >= k:
                return random.sample(window_indices, k)
            return random.choices(window_indices, k=k)

        positions = np.linspace(0, len(window_indices) - 1, k).round().astype(int)
        return [window_indices[int(pos)] for pos in positions]

    def __getitem__(self, trial_idx: int) -> dict:
        group = self.trial_groups[int(trial_idx)]
        selected_indices = self._sample_window_indices(group["window_indices"])
        items = [self.window_dataset[int(idx)] for idx in selected_indices]

        out: dict[str, Any] = {}
        for key in items[0].keys():
            values = [item[key] for item in items]
            if key in TRIAL_SCALAR_KEYS:
                out[key] = values[0]
            elif torch.is_tensor(values[0]):
                out[key] = torch.stack(values, dim=0)
            else:
                out[key] = values[0]
        out["window_indices"] = torch.tensor(selected_indices, dtype=torch.long)
        return out


class BalancedTrialBatchSampler(Sampler[list[int]]):
    """Balanced label4 trial sampler. It yields trial indices, not window indices."""

    def __init__(
        self,
        dataset: TrialWindowDataset,
        trials_per_class: int = 2,
        num_classes: int = 4,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.trials_per_class = int(trials_per_class)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.label_values = list(range(self.num_classes))
        self.class_to_indices: dict[int, list[int]] = {label: [] for label in self.label_values}
        self.trial_subjects: dict[int, int] = {}

        for idx, group in enumerate(dataset.trial_groups):
            label = int(group["label4"])
            self.class_to_indices.setdefault(label, []).append(idx)
            self.trial_subjects[idx] = int(group["subject_id"])

        self.expected_batch_size = self.trials_per_class * max(len(self.label_values), 1)
        self.num_batches = self._estimate_num_batches()

    def _estimate_num_batches(self) -> int:
        max_count = max((len(v) for v in self.class_to_indices.values()), default=0)
        if max_count <= 0:
            return 0
        return max(1, int(math.ceil(max_count / max(self.trials_per_class, 1))))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def _draw_for_class(
        self,
        label: int,
        rng: random.Random,
        subject_counts: Counter,
        class_pools: dict[int, list[int]],
        class_positions: dict[int, int],
    ) -> list[int]:
        indices = self.class_to_indices.get(label, [])
        if not indices:
            return []
        selected: list[int] = []
        for _ in range(self.trials_per_class):
            pool = class_pools[label]
            pos = class_positions[label]
            if pos >= len(pool):
                rng.shuffle(pool)
                pos = 0
            pick_pos = pos
            for candidate_pos in range(pos, len(pool)):
                candidate = pool[candidate_pos]
                if candidate not in selected and subject_counts[self.trial_subjects[candidate]] == 0:
                    pick_pos = candidate_pos
                    break
            pool[pos], pool[pick_pos] = pool[pick_pos], pool[pos]
            choice = pool[pos]
            class_positions[label] = pos + 1
            selected.append(choice)
            subject_counts[self.trial_subjects[choice]] += 1
        return selected

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        class_pools = {label: indices.copy() for label, indices in self.class_to_indices.items()}
        class_positions = {label: 0 for label in self.class_to_indices}
        for pool in class_pools.values():
            rng.shuffle(pool)
        for _ in range(self.num_batches):
            batch: list[int] = []
            subject_counts: Counter = Counter()
            for label in self.label_values:
                batch.extend(self._draw_for_class(label, rng, subject_counts, class_pools, class_positions))
            if not batch:
                continue
            rng.shuffle(batch)
            if self.drop_last and len(batch) < self.expected_batch_size:
                continue
            yield batch


def flatten_trial_batch(batch: dict) -> tuple[dict, int, int]:
    """Flatten [B, K, ...] trial tensors to [B*K, ...] window tensors."""

    if "x" not in batch or not torch.is_tensor(batch["x"]) or batch["x"].dim() < 2:
        raise KeyError("Trial batch must contain tensor key 'x' with shape [B, K, ...].")
    bsz, num_windows = int(batch["x"].shape[0]), int(batch["x"].shape[1])
    flat: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.dim() >= 2 and value.shape[0] == bsz and value.shape[1] == num_windows:
                flat[key] = value.reshape(bsz * num_windows, *value.shape[2:])
            elif value.dim() >= 1 and value.shape[0] == bsz:
                flat[key] = value.repeat_interleave(num_windows, dim=0)
            else:
                flat[key] = value
        elif isinstance(value, (list, tuple)) and len(value) == bsz:
            flat[key] = [item for item in value for _ in range(num_windows)]
        else:
            flat[key] = value
    return flat, bsz, num_windows


def confidence_weighted_aggregate(
    z_window: torch.Tensor,
    logits_window: torch.Tensor,
    tau_conf: float = 0.5,
    gamma: float = 0.6,
    detach_conf: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Confidence-weighted trial aggregation over K window embeddings."""

    prob = torch.softmax(logits_window, dim=-1)
    entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1)
    conf = 1.0 - entropy / math.log(2.0)
    if detach_conf:
        conf = conf.detach()
    tau = max(float(tau_conf), 1e-6)
    attn = torch.softmax(conf / tau, dim=1)
    k = z_window.size(1)
    uniform = torch.full_like(attn, 1.0 / max(k, 1))
    weight = (1.0 - float(gamma)) * uniform + float(gamma) * attn
    z_trial = torch.sum(z_window * weight.unsqueeze(-1), dim=1)
    return z_trial, weight, conf


class HierarchicalWeightedSupConLoss(nn.Module):
    """Weighted trial-level SupCon on z_emo trial embeddings."""

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 64,
        temperature: float = 0.1,
        same_label4_weight: float = 1.0,
        same_emotion_diff_diag_weight: float = 0.25,
        same_subject_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.same_label4_weight = float(same_label4_weight)
        self.same_emotion_diff_diag_weight = float(same_emotion_diff_diag_weight)
        self.same_subject_weight = float(same_subject_weight)
        self.projection_head = nn.Sequential(
            nn.Linear(int(in_dim), int(in_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_dim), int(proj_dim)),
        )

    def forward(
        self,
        z_trial: torch.Tensor,
        label4: torch.Tensor,
        emotion_label: torch.Tensor,
        diagnosis_label: torch.Tensor,
        subject_id: torch.Tensor,
    ) -> torch.Tensor:
        bsz = z_trial.size(0)
        if bsz <= 1:
            return z_trial.new_tensor(0.0)

        proj = F.normalize(self.projection_head(z_trial), dim=-1)
        sim = torch.matmul(proj, proj.T) / max(self.temperature, 1e-6)
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        label4 = label4.reshape(-1).long()
        emotion_label = emotion_label.reshape(-1).long()
        diagnosis_label = diagnosis_label.reshape(-1).long()
        subject_id = subject_id.reshape(-1)

        same_subject = subject_id[:, None] == subject_id[None, :]
        same_label4 = label4[:, None] == label4[None, :]
        same_emotion = emotion_label[:, None] == emotion_label[None, :]
        diff_diag = diagnosis_label[:, None] != diagnosis_label[None, :]

        pos_weight = torch.zeros_like(sim)
        pos_weight = torch.where(
            same_label4,
            torch.full_like(pos_weight, self.same_label4_weight),
            pos_weight,
        )
        pos_weight = torch.where(
            same_emotion & diff_diag,
            torch.full_like(pos_weight, self.same_emotion_diff_diag_weight),
            pos_weight,
        )
        pos_weight = torch.where(
            same_subject,
            torch.full_like(pos_weight, self.same_subject_weight),
            pos_weight,
        )

        logits_mask = torch.ones_like(sim)
        logits_mask.fill_diagonal_(0.0)
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_weight = pos_weight * logits_mask
        pos_sum = pos_weight.sum(dim=1)
        valid = pos_sum > 0
        if valid.sum() == 0:
            return z_trial.new_tensor(0.0)
        loss_i = -(pos_weight * log_prob).sum(dim=1) / (pos_sum + 1e-8)
        return loss_i[valid].mean()


def trial_weight_entropy(weight: torch.Tensor) -> torch.Tensor:
    return -(weight * torch.log(weight + 1e-8)).sum(dim=1)

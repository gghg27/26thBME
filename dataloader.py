from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


ROOT = Path(__file__).resolve().parent


def _resolve_path(path_value: str) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    if not path.is_absolute():
        path = ROOT / path
    if path.exists():
        return path
    raise FileNotFoundError(f"Data file not found: {path}")


class Competition4ClassDataset(Dataset):
    """Dataset matching the batch format expected by the imported trainers."""

    def __init__(
        self,
        index_csv: str | Path,
        subject_ids: Iterable | None = None,
        normalize: bool = False,
    ) -> None:
        self.index_csv = Path(index_csv)
        if not self.index_csv.is_absolute():
            self.index_csv = ROOT / self.index_csv
        self.df = pd.read_csv(self.index_csv)
        if subject_ids is not None:
            subjects = set(str(s) for s in subject_ids)
            self.df = self.df[self.df["subject_id"].astype(str).isin(subjects)].copy()
        self.df = self.df.reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows left in {self.index_csv} after subject filtering.")

        unique_subjects = sorted(self.df["subject_id"].unique().tolist(), key=lambda x: str(x))
        self.subject_to_domain = {subject: idx for idx, subject in enumerate(unique_subjects)}
        self.sample_subject_ids = self.df["subject_id"].tolist()
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        trial = np.load(_resolve_path(row["trial_path"]))
        start = int(row["start"]) if "start" in row.index else 0
        end = int(row["end"]) if "end" in row.index else trial.shape[-1]
        x = trial[:, start:end].astype("float32", copy=False)
        if self.normalize:
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-6
            x = (x - mean) / std

        de_arr = np.load(_resolve_path(row["de_path"]))
        if de_arr.ndim >= 3 and "de_win_id" in row.index:
            de_feat = de_arr[int(row["de_win_id"])]
        else:
            de_feat = de_arr
        de_feat = de_feat.astype("float32", copy=False)

        subject_id = row["subject_id"]
        diagnosis_label = int(row["diagnosis_label"]) if "diagnosis_label" in row.index else int(row["label4"] >= 2)
        emotion_label = int(row["emotion_label"]) if "emotion_label" in row.index else int(row["label4"] % 2)
        label4 = int(row["label4"]) if "label4" in row.index else diagnosis_label * 2 + emotion_label
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "de_feat": torch.tensor(de_feat, dtype=torch.float32),
            "label4": torch.tensor(label4, dtype=torch.long),
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
            "diagnosis_label": torch.tensor(diagnosis_label, dtype=torch.long),
            "subject_id": torch.tensor(int(subject_id), dtype=torch.long),
            "domain_id": torch.tensor(self.subject_to_domain[subject_id], dtype=torch.long),
            "trial_id": torch.tensor(int(row["trial_id"]), dtype=torch.long),
        }


def comp4_collate_fn(batch: list[dict]) -> dict:
    return {
        key: torch.stack([item[key] for item in batch], dim=0)
        for key in batch[0].keys()
    }


seed_collate_fn = comp4_collate_fn


class EEGWindowDataset(Competition4ClassDataset):
    def __init__(self, index_csv: str | Path, subject_ids=None, normalize=True, **kwargs) -> None:
        super().__init__(index_csv=index_csv, subject_ids=subject_ids, normalize=normalize)


class MultiSubjectBatchSampler(Sampler[list[int]]):
    """Simple sampler kept for compatibility with the imported trainer API."""

    def __init__(
        self,
        sample_subject_ids,
        batch_size: int,
        subjects_per_batch: int = 4,
        drop_last: bool = True,
        shuffle: bool = True,
    ) -> None:
        self.sample_subject_ids = list(sample_subject_ids)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.indices = list(range(len(self.sample_subject_ids)))

    def __iter__(self):
        indices = self.indices.copy()
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return int(np.ceil(len(self.indices) / max(self.batch_size, 1)))

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
DUAL_COLUMNS = (
    "trial_path_abs",
    "trial_path_rel",
    "de_path_abs",
    "de_path_rel",
)


def resolve_dual_path(value: str, root: Path = ROOT) -> Path:
    path = Path(str(value).replace("\\", "/"))
    candidates = [path] if path.is_absolute() else [root / path, root / "data" / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Dual-view data file not found: {value}")


def expand_dual_window_index(df: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    """Expand a trial-level dual index using the stored DE sequence length."""
    if "de_win_id" in df.columns:
        return df.reset_index(drop=True).copy()
    rows = []
    for _, row in df.iterrows():
        de = np.load(resolve_dual_path(row["de_path_abs"], root), mmap_mode="r")
        n_windows = int(row.get("n_windows", de.shape[0] if de.ndim >= 3 else 1))
        raw = np.load(resolve_dual_path(row["trial_path_abs"], root), mmap_mode="r")
        raw_len = int(raw.shape[-1])
        win_len = int(row.get("window_len", raw_len if n_windows == 1 else 500))
        step = int(row.get("step", win_len if n_windows == 1 else 250))
        for win_id in range(n_windows):
            item = row.to_dict()
            item["de_win_id"] = win_id
            item["start"] = win_id * step
            item["end"] = min(item["start"] + win_len, raw_len)
            rows.append(item)
    return pd.DataFrame(rows).reset_index(drop=True)


def validate_dual_index(
    df: pd.DataFrame,
    root: Path = ROOT,
    *,
    full_scan: bool = False,
) -> dict:
    missing = [column for column in DUAL_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Dual index missing columns: {missing}")
    if df.empty:
        raise ValueError("Dual index is empty.")
    indices = range(len(df)) if full_scan else sorted(set([0, len(df) // 2, len(df) - 1]))
    checked_pairs: set[tuple[str, str, str, str]] = set()
    for idx in indices:
        row = df.iloc[int(idx)]
        paths = tuple(str(row[column]) for column in DUAL_COLUMNS)
        if paths in checked_pairs:
            continue
        checked_pairs.add(paths)
        trial_abs = resolve_dual_path(paths[0], root)
        trial_rel = resolve_dual_path(paths[1], root)
        de_abs_path = resolve_dual_path(paths[2], root)
        de_rel_path = resolve_dual_path(paths[3], root)
        if trial_abs == trial_rel:
            raise AssertionError(f"abs/rel trial resolve to same file: {trial_abs}")
        if de_abs_path == de_rel_path:
            raise AssertionError(f"abs/rel DE resolve to same file: {de_abs_path}")
        x_abs = np.load(trial_abs, mmap_mode="r")
        x_rel = np.load(trial_rel, mmap_mode="r")
        de_abs = np.load(de_abs_path, mmap_mode="r")
        de_rel = np.load(de_rel_path, mmap_mode="r")
        if x_abs.shape != x_rel.shape:
            raise AssertionError(f"trial shape mismatch: {x_abs.shape} != {x_rel.shape}")
        if de_abs.shape != de_rel.shape:
            raise AssertionError(f"DE shape mismatch: {de_abs.shape} != {de_rel.shape}")
        for name, array in (("x_abs", x_abs), ("x_rel", x_rel), ("de_abs", de_abs), ("de_rel", de_rel)):
            if not np.isfinite(array).all():
                raise FloatingPointError(f"{name} contains NaN/Inf at index row {idx}")
    lengths = sorted((df["end"].astype(int) - df["start"].astype(int)).unique().tolist())
    starts = sorted(df["start"].astype(int).unique().tolist())
    steps = sorted(set(np.diff(starts).tolist())) if len(starts) > 1 else []
    return {"rows": len(df), "window_lengths": lengths, "candidate_steps": steps}


class DualViewCompetitionDataset(Dataset):
    """Window dataset with signal-level absolute and subject-relative views."""

    def __init__(
        self,
        index_csv: str | Path,
        subject_ids: Iterable | None = None,
        normalize: bool = False,
        root: Path = ROOT,
        validate: bool = True,
    ) -> None:
        if normalize:
            raise ValueError("Window normalization is forbidden for dual-signal data.")
        self.index_csv = Path(index_csv)
        if not self.index_csv.is_absolute():
            self.index_csv = root / self.index_csv
        self.root = Path(root)
        self.df = pd.read_csv(self.index_csv)
        if subject_ids is not None:
            allowed = {str(value) for value in subject_ids}
            self.df = self.df[self.df["subject_id"].astype(str).isin(allowed)].copy()
        self.df = self.df.reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows left in {self.index_csv}")
        if validate:
            self.integrity = validate_dual_index(self.df, self.root)
        subjects = sorted(self.df["subject_id"].unique().tolist(), key=str)
        self.subject_to_domain = {subject: idx for idx, subject in enumerate(subjects)}
        self.sample_subject_ids = self.df["subject_id"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[int(idx)]
        trial_abs = np.load(resolve_dual_path(row["trial_path_abs"], self.root))
        trial_rel = np.load(resolve_dual_path(row["trial_path_rel"], self.root))
        start, end = int(row["start"]), int(row["end"])
        x_abs = trial_abs[:, start:end].astype(np.float32, copy=False)
        x_rel = trial_rel[:, start:end].astype(np.float32, copy=False)
        de_abs_seq = np.load(resolve_dual_path(row["de_path_abs"], self.root))
        de_rel_seq = np.load(resolve_dual_path(row["de_path_rel"], self.root))
        win_id = int(row.get("de_win_id", 0))
        de_abs = (de_abs_seq[win_id] if de_abs_seq.ndim >= 3 else de_abs_seq).astype(np.float32, copy=False)
        de_rel = (de_rel_seq[win_id] if de_rel_seq.ndim >= 3 else de_rel_seq).astype(np.float32, copy=False)
        if x_abs.shape != x_rel.shape or de_abs.shape != de_rel.shape:
            raise AssertionError("Dual-view shapes changed after window selection.")
        for name, value in (("x_abs", x_abs), ("x_rel", x_rel), ("de_abs", de_abs), ("de_rel", de_rel)):
            if not np.isfinite(value).all():
                raise FloatingPointError(f"{name} contains NaN/Inf at dataset index {idx}")
        subject = row["subject_id"]
        label4 = int(row.get("label4", 0))
        emotion = int(row.get("emotion_label", label4 % 2))
        diagnosis = int(row.get("diagnosis_label", int(label4 >= 2)))
        return {
            "x_abs": torch.from_numpy(np.ascontiguousarray(x_abs)),
            "x_rel": torch.from_numpy(np.ascontiguousarray(x_rel)),
            "de_abs": torch.from_numpy(np.ascontiguousarray(de_abs)),
            "de_rel": torch.from_numpy(np.ascontiguousarray(de_rel)),
            "label4": torch.tensor(label4, dtype=torch.long),
            "emotion_label": torch.tensor(emotion, dtype=torch.long),
            "diagnosis_label": torch.tensor(diagnosis, dtype=torch.long),
            "subject_id": torch.tensor(int(subject), dtype=torch.long),
            "domain_id": torch.tensor(self.subject_to_domain[subject], dtype=torch.long),
            "trial_id": torch.tensor(int(row["trial_id"]), dtype=torch.long),
        }


Competition4ClassDataset = DualViewCompetitionDataset

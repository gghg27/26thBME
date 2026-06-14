# -*- coding: utf-8 -*-
"""Subject-set dataset for dual-stream EEG training and test prediction."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import N_CHANNELS, N_TRIALS, WIN_LEN, natural_key, resolve_path


class DualStreamSubjectSetDataset(Dataset):
    """Return one full subject/user with all 8 trials.

    Returned tensor shapes:
        x_abs: [8, K, 30, 2500]
        x_rel: [8, K, 30, 2500]
        de_abs/de_rel/de_z: [8, K, 30, 5]
        win_mask: [8, K]
    """

    def __init__(
        self,
        index_csv: str | Path,
        subject_ids: Iterable[Any] | None = None,
        *,
        mode: str = "train",
        root: str | Path = ROOT,
        max_windows: int | None = None,
        eps: float = 1e-6,
        check_paths: bool = True,
        debug: bool = True,
    ) -> None:
        self.index_csv = Path(index_csv)
        if not self.index_csv.is_absolute():
            self.index_csv = Path(root) / self.index_csv
        self.root = Path(root)
        self.mode = str(mode)
        self.eps = float(eps)
        self.debug = bool(debug)
        self._printed_first_item = False

        self.df = pd.read_csv(self.index_csv)
        if self.df.empty:
            raise ValueError(f"Empty index csv: {self.index_csv}")

        self.id_col = "user_id" if self.mode == "test" or "user_id" in self.df.columns else "subject_id"
        if subject_ids is not None and self.id_col == "subject_id":
            subjects = {str(subject) for subject in subject_ids}
            self.df = self.df[self.df["subject_id"].astype(str).isin(subjects)].copy()
        self.df = self.df.reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows left in {self.index_csv} after filtering.")

        self.trial_path_abs_col = "trial_path_abs" if "trial_path_abs" in self.df.columns else "trial_path"
        self.trial_path_rel_col = "trial_path_rel" if "trial_path_rel" in self.df.columns else self.trial_path_abs_col
        self.de_path_abs_col = "de_path_abs" if "de_path_abs" in self.df.columns else "de_path"
        self.has_real_rel = self.trial_path_rel_col != self.trial_path_abs_col

        if self.trial_path_abs_col not in self.df.columns:
            raise KeyError("Index must contain trial_path_abs or trial_path.")
        if self.de_path_abs_col not in self.df.columns:
            raise KeyError("Index must contain de_path_abs or de_path.")
        if "trial_id" not in self.df.columns:
            raise KeyError("Index must contain trial_id.")

        self.subject_keys = sorted(self.df[self.id_col].astype(str).unique().tolist(), key=natural_key)
        self.subject_to_rows = {
            key: self.df[self.df[self.id_col].astype(str) == key].copy().reset_index(drop=True)
            for key in self.subject_keys
        }
        self.max_windows = int(max_windows) if max_windows is not None else self._infer_max_windows()
        if self.max_windows <= 0:
            raise ValueError(f"Invalid max_windows={self.max_windows}")

        self.user_to_int = {user_id: idx for idx, user_id in enumerate(self.subject_keys)}

        if check_paths:
            self._check_path_columns()
        self._print_dataset_summary()

    def __len__(self) -> int:
        return len(self.subject_keys)

    def __getitem__(self, idx: int) -> dict:
        subject_key = self.subject_keys[idx]
        sdf = self.subject_to_rows[subject_key]
        trial_ids = sorted(sdf["trial_id"].astype(int).unique().tolist())
        if len(trial_ids) != N_TRIALS:
            raise ValueError(f"{subject_key}: expected {N_TRIALS} trials, got {trial_ids}")

        x_abs = np.zeros((N_TRIALS, self.max_windows, N_CHANNELS, WIN_LEN), dtype=np.float32)
        x_rel = np.zeros_like(x_abs)
        de_abs = np.zeros((N_TRIALS, self.max_windows, N_CHANNELS, 5), dtype=np.float32)
        win_mask = np.zeros((N_TRIALS, self.max_windows), dtype=np.float32)
        emotion_label = np.full((N_TRIALS,), -1, dtype=np.int64)
        trial_id_arr = np.zeros((N_TRIALS,), dtype=np.int64)

        for trial_slot, trial_id in enumerate(trial_ids):
            tdf = sdf[sdf["trial_id"].astype(int) == int(trial_id)].copy()
            sort_cols = [col for col in ["de_win_id", "start"] if col in tdf.columns]
            if sort_cols:
                tdf = tdf.sort_values(sort_cols)
            if len(tdf) > self.max_windows:
                raise ValueError(
                    f"{subject_key} trial {trial_id}: windows={len(tdf)} exceeds max_windows={self.max_windows}"
                )
            first = tdf.iloc[0]
            trial_abs_arr = self._load_trial(first[self.trial_path_abs_col], "abs")
            trial_rel_arr = self._load_trial(first[self.trial_path_rel_col], "rel")
            de_seq = np.load(resolve_path(first[self.de_path_abs_col], self.root)).astype(np.float32, copy=False)

            for win_slot, (_, row) in enumerate(tdf.iterrows()):
                start = int(row["start"]) if "start" in row.index and not pd.isna(row["start"]) else 0
                end = int(row["end"]) if "end" in row.index and not pd.isna(row["end"]) else start + WIN_LEN
                if end - start != WIN_LEN and trial_abs_arr.shape[-1] == WIN_LEN:
                    start, end = 0, WIN_LEN
                x_abs[trial_slot, win_slot] = self._slice_window(trial_abs_arr, start, end, subject_key, trial_id)
                x_rel[trial_slot, win_slot] = self._slice_window(trial_rel_arr, start, end, subject_key, trial_id)
                de_abs[trial_slot, win_slot] = self._select_de(de_seq, row)
                win_mask[trial_slot, win_slot] = 1.0

            trial_id_arr[trial_slot] = int(trial_id)
            emotion_label[trial_slot] = self._trial_emotion_label(tdf, trial_id)

        valid_de = de_abs[win_mask.astype(bool)]
        if valid_de.size == 0:
            raise ValueError(f"{subject_key}: no valid DE windows")
        subject_de_mu = valid_de.mean(axis=0, keepdims=True)
        subject_de_std = valid_de.std(axis=0, keepdims=True)
        subject_de_std = np.where(subject_de_std < self.eps, self.eps, subject_de_std)
        de_rel = de_abs - subject_de_mu.reshape(1, 1, N_CHANNELS, 5)
        de_z = de_rel / subject_de_std.reshape(1, 1, N_CHANNELS, 5)
        invalid = win_mask[..., None, None] <= 0
        de_rel = np.where(invalid, 0.0, de_rel)
        de_z = np.where(invalid, 0.0, de_z)

        diagnosis_label = self._subject_diagnosis_label(sdf)
        subject_id = self._subject_numeric_id(sdf, subject_key, idx)
        user_id = str(sdf["user_id"].iloc[0]) if "user_id" in sdf.columns else str(subject_key)

        item = {
            "x_abs": torch.tensor(x_abs, dtype=torch.float32),
            "x_rel": torch.tensor(x_rel, dtype=torch.float32),
            "de_abs": torch.tensor(de_abs, dtype=torch.float32),
            "de_rel": torch.tensor(de_rel, dtype=torch.float32),
            "de_z": torch.tensor(de_z, dtype=torch.float32),
            "win_mask": torch.tensor(win_mask, dtype=torch.float32),
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
            "diagnosis_label": torch.tensor(int(diagnosis_label), dtype=torch.long),
            "subject_id": torch.tensor(int(subject_id), dtype=torch.long),
            "user_id": user_id,
            "trial_id": torch.tensor(trial_id_arr, dtype=torch.long),
            "target_key": f"{self.mode}:{user_id}",
        }
        self._maybe_print_first_item(item)
        return item

    def _infer_max_windows(self) -> int:
        max_windows = 0
        for subject_key, sdf in self.subject_to_rows.items():
            trial_counts = sdf.groupby("trial_id").size()
            if len(trial_counts) != N_TRIALS:
                print(f"[dataset] WARNING {subject_key}: trial count={len(trial_counts)}, expected {N_TRIALS}")
            max_windows = max(max_windows, int(trial_counts.max()))
        return max_windows

    def _check_path_columns(self) -> None:
        checks = {
            "trial_path_abs": self.trial_path_abs_col,
            "trial_path_rel": self.trial_path_rel_col,
            "de_path_abs": self.de_path_abs_col,
        }
        for label, col in checks.items():
            unique_values = self.df[col].astype(str).drop_duplicates().tolist()
            ok = 0
            missing: list[str] = []
            for value in unique_values:
                try:
                    resolve_path(value, self.root)
                    ok += 1
                except FileNotFoundError:
                    if len(missing) < 5:
                        missing.append(value)
            print(f"[dataset paths] {label} col={col}: found {ok}/{len(unique_values)}")
            if missing:
                print(f"[dataset paths] missing examples for {label}: {missing}")
        if not self.has_real_rel:
            print("[dataset paths] WARNING trial_path_rel missing; rel stream falls back to abs path.")

    def _print_dataset_summary(self) -> None:
        print("\n" + "=" * 80)
        print(f"[dataset] csv={self.index_csv}")
        print(f"[dataset] mode={self.mode} subjects/users={len(self.subject_keys)} max_windows={self.max_windows}")
        trial_count_counter = Counter()
        win_count_counter = Counter()
        for subject_key, sdf in self.subject_to_rows.items():
            trial_counts = sdf.groupby("trial_id").size().sort_index()
            trial_count_counter[len(trial_counts)] += 1
            for _, count in trial_counts.items():
                win_count_counter[int(count)] += 1
            print(
                f"[dataset] {subject_key}: trials={len(trial_counts)} "
                f"windows_per_trial={trial_counts.astype(int).tolist()}"
            )
        print(f"[dataset] trial count distribution: {dict(sorted(trial_count_counter.items()))}")
        print(f"[dataset] window count distribution: {dict(sorted(win_count_counter.items()))}")

        if "diagnosis" in self.df.columns or "diagnosis_label" in self.df.columns or "label4" in self.df.columns:
            diag_labels = []
            for subject_key, sdf in self.subject_to_rows.items():
                label = self._subject_diagnosis_label(sdf)
                if label >= 0:
                    diag_labels.append(label)
            print(f"[dataset] diagnosis label distribution (0=DEP, 1=HC): {dict(sorted(Counter(diag_labels).items()))}")

        emo_labels = []
        for subject_key, sdf in self.subject_to_rows.items():
            for trial_id, tdf in sdf.groupby("trial_id"):
                label = self._trial_emotion_label(tdf, int(trial_id))
                if label >= 0:
                    emo_labels.append(label)
        print(f"[dataset] emotion label distribution (trial-level): {dict(sorted(Counter(emo_labels).items()))}")

    def _load_trial(self, path_value: Any, stream_name: str) -> np.ndarray:
        path = resolve_path(path_value, self.root)
        arr = np.load(path).astype(np.float32, copy=False)
        if arr.ndim != 2 or arr.shape[0] != N_CHANNELS:
            raise ValueError(f"{stream_name} trial path={path} must be [{N_CHANNELS}, T], got {arr.shape}")
        return arr

    @staticmethod
    def _slice_window(arr: np.ndarray, start: int, end: int, subject_key: str, trial_id: int) -> np.ndarray:
        if start < 0 or end > arr.shape[-1] or end <= start:
            raise ValueError(
                f"{subject_key} trial {trial_id}: invalid window start={start}, end={end}, trial_shape={arr.shape}"
            )
        x = arr[:, start:end]
        if x.shape != (N_CHANNELS, WIN_LEN):
            raise ValueError(f"{subject_key} trial {trial_id}: expected window {(N_CHANNELS, WIN_LEN)}, got {x.shape}")
        return x.astype(np.float32, copy=False)

    @staticmethod
    def _select_de(de_seq: np.ndarray, row: pd.Series) -> np.ndarray:
        if de_seq.ndim == 2:
            de = de_seq
        else:
            win_id = int(row["de_win_id"]) if "de_win_id" in row.index and not pd.isna(row["de_win_id"]) else 0
            if win_id >= de_seq.shape[0]:
                raise IndexError(f"de_win_id={win_id} out of range for de shape={de_seq.shape}")
            de = de_seq[win_id]
        if de.shape != (N_CHANNELS, 5):
            raise ValueError(f"Expected DE shape {(N_CHANNELS, 5)}, got {de.shape}")
        return de.astype(np.float32, copy=False)

    def _subject_diagnosis_label(self, sdf: pd.DataFrame) -> int:
        if self.mode == "test":
            return -1
        if "diagnosis" in sdf.columns:
            values = [str(v).upper() for v in sdf["diagnosis"].dropna().unique().tolist()]
            if "DEP" in values:
                return 0
            if "HC" in values:
                return 1
        if "label4" in sdf.columns:
            labels = sdf["label4"].dropna().astype(int)
            labels = labels[labels >= 0]
            if not labels.empty:
                return int(labels.mode().iloc[0] >= 2)
        if "diagnosis_label" in sdf.columns:
            labels = sdf["diagnosis_label"].dropna().astype(int)
            labels = labels[labels >= 0]
            if not labels.empty:
                return int(labels.mode().iloc[0])
        return -1

    def _trial_emotion_label(self, tdf: pd.DataFrame, trial_id: int) -> int:
        if self.mode == "test":
            return -1
        if "emotion_label" in tdf.columns:
            labels = tdf["emotion_label"].dropna().astype(int)
            labels = labels[labels >= 0]
            if not labels.empty:
                return int(labels.mode().iloc[0])
        if "label4" in tdf.columns:
            labels = tdf["label4"].dropna().astype(int)
            labels = labels[labels >= 0]
            if not labels.empty:
                return int(labels.mode().iloc[0] % 2)
        return 0 if int(trial_id) <= 4 else 1

    def _subject_numeric_id(self, sdf: pd.DataFrame, subject_key: str, idx: int) -> int:
        if "subject_id" in sdf.columns and not pd.isna(sdf["subject_id"].iloc[0]):
            return int(sdf["subject_id"].iloc[0])
        if "subject_number" in sdf.columns and not pd.isna(sdf["subject_number"].iloc[0]):
            return int(sdf["subject_number"].iloc[0])
        return int(self.user_to_int.get(str(subject_key), idx))

    def _maybe_print_first_item(self, item: dict) -> None:
        if not self.debug or self._printed_first_item:
            return
        self._printed_first_item = True
        print("[dataset sample] first item shapes:")
        for key in ["x_abs", "x_rel", "de_abs", "de_rel", "de_z", "win_mask"]:
            print(f"  {key}: {tuple(item[key].shape)}")
        print(f"  emotion_label: {item['emotion_label'].tolist()}")
        print(f"  diagnosis_label: {int(item['diagnosis_label'])}")


def dict_collate(batch: list[dict]) -> dict:
    out: dict[str, Any] = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values
    return out

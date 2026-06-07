from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class EEGWindowDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        label_col: str | None,
        root: Path | str = ".",
        feature_col: str = "de_path",
        return_de: bool = False,
    ) -> None:
        try:
            import torch
            from torch.utils.data import Dataset
        except ImportError as exc:
            raise ImportError("PyTorch is required for EEGWindowDataset") from exc

        class _Dataset(Dataset):
            def __len__(self_nonlocal) -> int:
                return len(df)

            def __getitem__(self_nonlocal, idx: int) -> Any:
                row = df.iloc[idx]
                if return_de:
                    x = load_raw_window(row, root=Path(root))
                    de_feat = load_feature(row, root=Path(root), feature_col=feature_col)
                else:
                    x = load_feature(row, root=Path(root), feature_col=feature_col)
                    de_feat = None
                x_tensor = torch.as_tensor(x, dtype=torch.float32)
                de_tensor = (
                    torch.as_tensor(de_feat, dtype=torch.float32)
                    if de_feat is not None
                    else None
                )
                if label_col is None:
                    if return_de:
                        return x_tensor, de_tensor, idx
                    return x_tensor, idx
                y = int(row[label_col])
                if return_de:
                    return x_tensor, de_tensor, torch.tensor(y, dtype=torch.long)
                return x_tensor, torch.tensor(y, dtype=torch.long)

        self._dataset = _Dataset()

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._dataset[idx]


def resolve_data_path(path_value: str, root: Path | str = ".") -> Path:
    root = Path(root)
    path_text = str(path_value).replace("\\", "/")
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    candidates = [path]
    if not path.exists():
        cleaned = path_text
        while cleaned.startswith("../"):
            cleaned = cleaned[3:]
        if cleaned.startswith("data/"):
            candidates.append(root / "testdata" / cleaned[len("data/") :])
        candidates.append(root / cleaned)
        candidates.append(root / "testdata" / Path(cleaned).name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Feature file not found. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_feature(row: pd.Series, *, root: Path | str = ".", feature_col: str = "de_path"):
    path = resolve_data_path(row[feature_col], root)
    arr = np.load(path)
    if arr.ndim >= 3 and "de_win_id" in row.index:
        win_id = int(row["de_win_id"])
        if win_id >= arr.shape[0]:
            raise IndexError(f"de_win_id={win_id} out of range for {path} shape {arr.shape}")
        arr = arr[win_id]
    return arr.astype("float32", copy=False)


def load_raw_window(row: pd.Series, *, root: Path | str = ".", raw_col: str = "trial_path"):
    path = resolve_data_path(row[raw_col], root)
    arr = np.load(path)
    start = int(row["start"]) if "start" in row.index and not pd.isna(row["start"]) else 0
    end = int(row["end"]) if "end" in row.index and not pd.isna(row["end"]) else arr.shape[-1]
    return arr[:, start:end].astype("float32", copy=False)


def expand_window_index(
    df: pd.DataFrame,
    *,
    root: Path | str = ".",
    feature_col: str = "de_path",
) -> pd.DataFrame:
    """Convert trial-level rows with n_windows into one row per feature window."""
    if "de_win_id" in df.columns:
        return df.reset_index(drop=True).copy()

    rows = []
    for _, row in df.iterrows():
        if "n_windows" in row.index and not pd.isna(row["n_windows"]):
            n_windows = int(row["n_windows"])
        else:
            path = resolve_data_path(row[feature_col], root)
            arr = np.load(path, mmap_mode="r")
            n_windows = int(arr.shape[0]) if arr.ndim >= 3 else 1
        if n_windows <= 0:
            raise ValueError(f"Invalid n_windows={n_windows} for row: {row.to_dict()}")
        raw_len = None
        if "trial_path" in row.index:
            raw = np.load(resolve_data_path(row["trial_path"], root), mmap_mode="r")
            raw_len = int(raw.shape[-1])
        if raw_len is not None and n_windows == 1:
            window_size = raw_len
            stride = raw_len
        elif raw_len is not None:
            window_size = max(1, int(round(2 * raw_len / (n_windows + 1))))
            stride = max(1, int(round((raw_len - window_size) / max(n_windows - 1, 1))))
        else:
            window_size = None
            stride = None
        for win_id in range(n_windows):
            item = row.to_dict()
            item["de_win_id"] = win_id
            if window_size is not None and "start" not in item:
                start = win_id * stride
                item["start"] = start
                item["end"] = min(start + window_size, raw_len)
            rows.append(item)
    return pd.DataFrame(rows).reset_index(drop=True)

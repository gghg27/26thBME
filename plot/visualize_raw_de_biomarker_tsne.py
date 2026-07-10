# -*- coding: utf-8 -*-
"""Visualize raw DE, biomarker, and fused raw features with t-SNE.

The script does not need a trained checkpoint. It reads raw EEG windows and
precomputed DE features from the competition index CSV, computes the
57-dimensional biomarker vector with the repository's BiologicalMarkerExtractor,
and also visualizes concat(zscore(DE), zscore(biomarker)).

Examples:
    conda run -n pytorch python plot/visualize_raw_de_biomarker_tsne.py ^
        --index_csv com_index_sub_2s.csv --data_root . ^
        --output_dir plot/raw_de_biomarker_tsne --device cuda

    conda run -n pytorch python B:/26thbme/v2_best/26thBME/plot/visualize_raw_de_biomarker_tsne.py ^
        --index_csv com_index_sub_2s.csv --data_root B:/26thbme/stage ^
        --output_dir B:/26thbme/stage/plot/raw_de_biomarker_tsne
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.dep_contrast_bio import BiologicalMarkerExtractor, RawSignalPLVGraphConstructor


CONDITION_ORDER = ["HC_neu", "HC_pos", "DEP_neu", "DEP_pos"]
LABEL4_TO_CONDITION = {
    0: "DEP_neu",
    1: "DEP_pos",
    2: "HC_neu",
    3: "HC_pos",
}
DIAGNOSIS_NAMES = {0: "DEP", 1: "HC"}
EMOTION_NAMES = {0: "neu", 1: "pos"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw DE and biomarker t-SNE visualization.")
    parser.add_argument("--index_csv", type=str, default="com_index_sub_2s.csv")
    parser.add_argument("--data_root", type=str, default=".", help="Base path for index_csv, trial_path, and de_path.")
    parser.add_argument("--output_dir", type=str, default="plot/raw_de_biomarker_tsne")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=800, help="Maximum samples used by t-SNE after optional trial aggregation.")
    parser.add_argument("--use_trial_level", type=int, default=1, help="1=average windows within each trial; 0=use window-level samples.")
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--num_channels", type=int, default=30)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--no_normalize_x", action="store_true", help="Do not z-score raw EEG per channel before biomarker extraction.")
    parser.add_argument("--normalize_de_per_window", action="store_true", help="Z-score DE across channels within each window.")
    parser.add_argument("--save_feature_npz", action="store_true", help="Also save DE and biomarker matrices.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_path(path_value: Any, data_root: Path) -> Path:
    text = str(path_value).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() and path.exists():
        return path

    cleaned = text
    while cleaned.startswith("../"):
        cleaned = cleaned[3:]
    cleaned_path = Path(cleaned)

    candidates = [
        data_root / path,
        data_root / cleaned_path,
        REPO_ROOT / path,
        REPO_ROOT / cleaned_path,
        data_root / "data" / cleaned_path.name,
        REPO_ROOT / "data" / cleaned_path.name,
    ]
    if cleaned.startswith("data/"):
        without_data = cleaned[len("data/") :]
        candidates.extend(
            [
                data_root / without_data,
                data_root / "data" / without_data,
                REPO_ROOT / without_data,
                REPO_ROOT / "data" / without_data,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Data file not found: {path_value}")


def infer_labels(row: pd.Series) -> tuple[int, int, int, str]:
    if "label4" in row.index and not pd.isna(row["label4"]):
        label4 = int(row["label4"])
        emotion_label = int(row["emotion_label"]) if "emotion_label" in row.index and not pd.isna(row["emotion_label"]) else label4 % 2
        diagnosis_label = 1 if label4 >= 2 else 0
    else:
        emotion_label = int(row["emotion_label"]) if "emotion_label" in row.index and not pd.isna(row["emotion_label"]) else 0
        if "diagnosis" in row.index and not pd.isna(row["diagnosis"]):
            diagnosis_label = 0 if str(row["diagnosis"]).upper().startswith("DEP") else 1
        else:
            raw_diag = int(row["diagnosis_label"]) if "diagnosis_label" in row.index and not pd.isna(row["diagnosis_label"]) else 0
            diagnosis_label = 1 if raw_diag == 0 else 0
        label4 = (2 if diagnosis_label == 1 else 0) + emotion_label
    condition = LABEL4_TO_CONDITION.get(label4, f"diag{diagnosis_label}_emo{emotion_label}")
    return diagnosis_label, emotion_label, label4, condition


class RawFeatureDataset(Dataset):
    def __init__(
        self,
        index_csv: str | Path,
        data_root: str | Path,
        normalize_x: bool = True,
        normalize_de_per_window: bool = False,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        csv_path = Path(index_csv)
        if not csv_path.is_absolute():
            csv_path = resolve_path(csv_path, self.data_root)
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"Empty index CSV: {csv_path}")
        for required in ["trial_path", "de_path"]:
            if required not in self.df.columns:
                raise KeyError(f"CSV must contain {required}.")
        self.normalize_x = bool(normalize_x)
        self.normalize_de_per_window = bool(normalize_de_per_window)
        self._trial_cache: dict[str, np.ndarray] = {}
        self._de_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_cached(self, cache: dict[str, np.ndarray], path_value: Any) -> np.ndarray:
        key = str(path_value)
        if key not in cache:
            cache[key] = np.load(resolve_path(path_value, self.data_root), mmap_mode="r")
        return cache[key]

    @staticmethod
    def _row_int(row: pd.Series, key: str, default: Optional[int] = None) -> Optional[int]:
        if key not in row.index or pd.isna(row[key]):
            return default
        return int(row[key])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        trial = self._load_cached(self._trial_cache, row["trial_path"])
        start = self._row_int(row, "start", 0) or 0
        end = self._row_int(row, "end", trial.shape[-1]) or trial.shape[-1]
        x = np.asarray(trial[:, start:end], dtype=np.float32)
        if self.normalize_x:
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-6
            x = (x - mean) / std

        de_arr = self._load_cached(self._de_cache, row["de_path"])
        de_win_id = self._row_int(row, "de_win_id", 0) or 0
        de_feat = np.asarray(de_arr[int(de_win_id)] if de_arr.ndim >= 3 else de_arr, dtype=np.float32)
        if self.normalize_de_per_window:
            mean = de_feat.mean(axis=0, keepdims=True)
            std = de_feat.std(axis=0, keepdims=True) + 1e-6
            de_feat = (de_feat - mean) / std

        diagnosis_label, emotion_label, label4, condition = infer_labels(row)
        subject_value = row.get("subject_id", row.get("subject_number", row.get("user_id", 0)))
        user_id = str(row.get("user_id", subject_value))
        trial_id = int(row.get("trial_id", idx))

        return {
            "x": torch.tensor(np.ascontiguousarray(x), dtype=torch.float32),
            "de_feat": torch.tensor(np.ascontiguousarray(de_feat), dtype=torch.float32),
            "subject_id": str(subject_value),
            "user_id": user_id,
            "trial_id": trial_id,
            "diagnosis_label": int(diagnosis_label),
            "emotion_label": int(emotion_label),
            "label4": int(label4),
            "condition": condition,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "de_feat": torch.stack([item["de_feat"] for item in batch], dim=0),
    }
    for key in ["subject_id", "user_id", "trial_id", "diagnosis_label", "emotion_label", "label4", "condition"]:
        out[key] = [item[key] for item in batch]
    return out


def flatten_feature(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@torch.no_grad()
def collect_records(
    loader: DataLoader,
    device: torch.device,
    sfreq: float,
    num_channels: int,
    topk: int,
) -> list[dict[str, Any]]:
    biomarker = BiologicalMarkerExtractor(sfreq=sfreq, num_channels=num_channels).to(device)
    plv = RawSignalPLVGraphConstructor(num_nodes=num_channels, topk=topk).to(device)
    biomarker.eval()
    plv.eval()

    records: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="Collect DE and biomarker features", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        de_feat = batch["de_feat"].to(device, non_blocking=True)
        plv_matrix = plv.compute_plv(x)
        bio = biomarker(x, de_feat=de_feat, plv_matrix=plv_matrix)["bio_raw"]

        de_np = de_feat.detach().cpu().float().numpy()
        bio_np = bio.detach().cpu().float().numpy()
        for i in range(de_np.shape[0]):
            records.append(
                {
                    "subject_id": str(batch["subject_id"][i]),
                    "user_id": str(batch["user_id"][i]),
                    "trial_id": int(batch["trial_id"][i]),
                    "diagnosis_label": int(batch["diagnosis_label"][i]),
                    "emotion_label": int(batch["emotion_label"][i]),
                    "label4": int(batch["label4"][i]),
                    "condition": str(batch["condition"][i]),
                    "de_feat": flatten_feature(de_np[i]),
                    "bio_raw": flatten_feature(bio_np[i]),
                }
            )
    return records


def mean_stack(values: list[Any]) -> Optional[np.ndarray]:
    arrays = [flatten_feature(value) for value in values if value is not None]
    if not arrays:
        return None
    common_len = Counter(arr.size for arr in arrays).most_common(1)[0][0]
    arrays = [arr for arr in arrays if arr.size == common_len]
    return np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)


def window_to_trial(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["user_id"]), int(record["trial_id"]))].append(record)

    trials: list[dict[str, Any]] = []
    for (_, _), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        first = rows[0]
        trials.append(
            {
                "subject_id": first["subject_id"],
                "user_id": first["user_id"],
                "trial_id": first["trial_id"],
                "diagnosis_label": first["diagnosis_label"],
                "emotion_label": first["emotion_label"],
                "label4": first["label4"],
                "condition": first["condition"],
                "n_windows": len(rows),
                "de_feat": mean_stack([row.get("de_feat") for row in rows]),
                "bio_raw": mean_stack([row.get("bio_raw") for row in rows]),
            }
        )
    return trials


def standardize_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    return ((x - mu) / sigma).astype(np.float32)


def compute_tsne(matrix: np.ndarray, seed: int) -> Optional[np.ndarray]:
    x = standardize_matrix(matrix)
    n_samples, n_features = x.shape
    if n_samples < 4 or n_features < 2:
        return None
    pca_dim = min(50, n_samples - 1, n_features)
    if n_features > pca_dim >= 2:
        x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
    perplexity = max(2, min(30, (n_samples - 1) // 3))
    if perplexity >= n_samples:
        return None
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed).fit_transform(x)


def labels_for(records: list[dict[str, Any]], mode: str) -> tuple[list[str], list[int]]:
    if mode == "diagnosis":
        numeric = [int(record["diagnosis_label"]) for record in records]
        labels = [DIAGNOSIS_NAMES.get(value, str(value)) for value in numeric]
    elif mode == "emotion":
        numeric = [int(record["emotion_label"]) for record in records]
        labels = [EMOTION_NAMES.get(value, str(value)) for value in numeric]
    else:
        labels = [str(record["condition"]) for record in records]
        numeric = [CONDITION_ORDER.index(label) if label in CONDITION_ORDER else -1 for label in labels]
    return labels, numeric


def maybe_silhouette(coords: np.ndarray, numeric_labels: list[int]) -> Optional[float]:
    labels = np.asarray(numeric_labels)
    valid = labels >= 0
    labels = labels[valid]
    coords = coords[valid]
    if coords.shape[0] < 4 or len(set(labels.tolist())) < 2 or len(set(labels.tolist())) >= coords.shape[0]:
        return None
    try:
        return float(silhouette_score(coords, labels))
    except Exception:
        return None


def stratified_sample(records: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(records) <= max_samples:
        return records
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("condition", "NA"))].append(record)
    sampled: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    per_group = max(1, max_samples // max(len(groups), 1))
    for key in sorted(groups):
        rows = groups[key]
        if len(rows) <= per_group:
            sampled.extend(rows)
        else:
            idx = rng.choice(len(rows), size=per_group, replace=False)
            chosen = set(int(i) for i in idx)
            sampled.extend([rows[int(i)] for i in idx])
            leftovers.extend([row for i, row in enumerate(rows) if i not in chosen])
    remaining = max_samples - len(sampled)
    if remaining > 0 and leftovers:
        idx = rng.choice(len(leftovers), size=min(remaining, len(leftovers)), replace=False)
        sampled.extend([leftovers[int(i)] for i in idx])
    return sampled[:max_samples]


def feature_matrix(records: list[dict[str, Any]], key: str) -> tuple[list[dict[str, Any]], Optional[np.ndarray]]:
    pairs = [(record, record.get(key)) for record in records if record.get(key) is not None]
    if len(pairs) < 4:
        return [], None
    common_len = Counter(np.asarray(value).size for _, value in pairs).most_common(1)[0][0]
    pairs = [(record, value) for record, value in pairs if np.asarray(value).size == common_len]
    if len(pairs) < 4:
        return [], None
    return [record for record, _ in pairs], np.stack([flatten_feature(value) for _, value in pairs], axis=0)


def fused_de_biomarker_matrix(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Optional[np.ndarray]]:
    pairs = [
        (record, record.get("de_feat"), record.get("bio_raw"))
        for record in records
        if record.get("de_feat") is not None and record.get("bio_raw") is not None
    ]
    if len(pairs) < 4:
        return [], None

    de_len = Counter(np.asarray(de_value).size for _, de_value, _ in pairs).most_common(1)[0][0]
    bio_len = Counter(np.asarray(bio_value).size for _, _, bio_value in pairs).most_common(1)[0][0]
    pairs = [
        (record, de_value, bio_value)
        for record, de_value, bio_value in pairs
        if np.asarray(de_value).size == de_len and np.asarray(bio_value).size == bio_len
    ]
    if len(pairs) < 4:
        return [], None

    de_matrix = np.stack([flatten_feature(de_value) for _, de_value, _ in pairs], axis=0)
    bio_matrix = np.stack([flatten_feature(bio_value) for _, _, bio_value in pairs], axis=0)
    fused = np.concatenate([standardize_matrix(de_matrix), standardize_matrix(bio_matrix)], axis=1).astype(np.float32)
    return [record for record, _, _ in pairs], fused


def get_feature_matrix(records: list[dict[str, Any]], key: str) -> tuple[list[dict[str, Any]], Optional[np.ndarray]]:
    if key == "de_bio_fused":
        return fused_de_biomarker_matrix(records)
    return feature_matrix(records, key)


def scatter_embedding(ax: plt.Axes, coords: np.ndarray, labels: list[str], title: str, score: Optional[float]) -> None:
    palette = {
        "HC": "#2CA02C",
        "DEP": "#D62728",
        "neu": "#4C78A8",
        "pos": "#F58518",
        "HC_neu": "#1B9E77",
        "HC_pos": "#66A61E",
        "DEP_neu": "#7570B3",
        "DEP_pos": "#E7298A",
    }

    def sort_key(label: str) -> tuple[int, str]:
        return (0, str(CONDITION_ORDER.index(label))) if label in CONDITION_ORDER else (1, label)

    for label in sorted(set(labels), key=sort_key):
        mask = np.asarray([item == label for item in labels])
        ax.scatter(coords[mask, 0], coords[mask, 1], s=28, alpha=0.82, label=label, color=palette.get(label), edgecolor="white", linewidth=0.3)
    score_text = "" if score is None or np.isnan(score) else f" | silhouette={score:.3f}"
    ax.set_title(f"{title}{score_text}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, fontsize=8, loc="best")


def save_figure(fig: plt.Figure, output_dir: Path, base_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{base_name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / f"{base_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tsne(records: list[dict[str, Any]], output_dir: Path, seed: int, max_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = stratified_sample(records, max_samples=max_samples, seed=seed)
    feature_specs = [
        ("de_feat", "DE features"),
        ("bio_raw", "Biomarker features"),
        ("de_bio_fused", "DE + biomarker fused features"),
    ]
    label_specs = [("label4", "four conditions", "condition"), ("diagnosis", "diagnosis", "diagnosis"), ("emotion", "emotion", "emotion")]

    coords_by_feature: dict[str, tuple[list[dict[str, Any]], np.ndarray, np.ndarray]] = {}
    coord_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for feature_key, feature_name in feature_specs:
        usable, matrix = get_feature_matrix(records, feature_key)
        if matrix is None:
            continue
        coords = compute_tsne(matrix, seed=seed)
        if coords is None:
            continue
        coords_by_feature[feature_key] = (usable, matrix, coords)
        for label_mode, _, _ in label_specs:
            labels, numeric = labels_for(usable, label_mode)
            score = maybe_silhouette(coords, numeric)
            summary_rows.append(
                {
                    "feature_type": feature_key,
                    "feature_name": feature_name,
                    "label_mode": label_mode,
                    "n_samples": len(usable),
                    "feature_dim": int(matrix.shape[1]),
                    "silhouette": score,
                }
            )
            for record, label, xy in zip(usable, labels, coords):
                coord_rows.append(
                    {
                        "feature_type": feature_key,
                        "feature_name": feature_name,
                        "label_mode": label_mode,
                        "label": label,
                        "subject_id": record.get("subject_id"),
                        "user_id": record.get("user_id"),
                        "trial_id": record.get("trial_id"),
                        "diagnosis_label": record.get("diagnosis_label"),
                        "emotion_label": record.get("emotion_label"),
                        "label4": record.get("label4"),
                        "condition": record.get("condition"),
                        "tsne1": float(xy[0]),
                        "tsne2": float(xy[1]),
                    }
                )

    for label_mode, label_title, suffix in label_specs:
        panels = []
        for feature_key, feature_name in feature_specs:
            if feature_key not in coords_by_feature:
                continue
            usable, _, coords = coords_by_feature[feature_key]
            labels, numeric = labels_for(usable, label_mode)
            panels.append((coords, labels, feature_name, maybe_silhouette(coords, numeric)))
        if not panels:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.7), squeeze=False)
        for ax, (coords, labels, feature_name, score) in zip(axes.ravel(), panels):
            scatter_embedding(ax, coords, labels, f"{feature_name} by {label_title}", score)
        save_figure(fig, output_dir, f"fig_raw_de_biomarker_tsne_by_{suffix}")

    coord_df = pd.DataFrame(coord_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not coord_df.empty:
        coord_df.to_csv(output_dir / "raw_de_biomarker_tsne_coordinates.csv", index=False, encoding="utf-8-sig")
    if not summary_df.empty:
        summary_df.to_csv(output_dir / "raw_de_biomarker_tsne_summary.csv", index=False, encoding="utf-8-sig")
    return coord_df, summary_df


def save_feature_npz(records: list[dict[str, Any]], output_dir: Path) -> None:
    payload: dict[str, Any] = {}
    for key in ["de_feat", "bio_raw", "de_bio_fused"]:
        usable, matrix = get_feature_matrix(records, key)
        if matrix is not None:
            payload[key] = matrix.astype(np.float32)
            payload[f"{key}_subject_id"] = np.asarray([record["subject_id"] for record in usable])
            payload[f"{key}_trial_id"] = np.asarray([record["trial_id"] for record in usable], dtype=np.int64)
            payload[f"{key}_label4"] = np.asarray([record["label4"] for record in usable], dtype=np.int64)
    if payload:
        np.savez(output_dir / "raw_de_biomarker_features.npz", **payload)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    setup_plot_style()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    dataset = RawFeatureDataset(
        index_csv=args.index_csv,
        data_root=data_root,
        normalize_x=not args.no_normalize_x,
        normalize_de_per_window=args.normalize_de_per_window,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    window_records = collect_records(loader, device=device, sfreq=args.sfreq, num_channels=args.num_channels, topk=args.topk)
    level = "trial" if int(args.use_trial_level) else "window"
    records = window_to_trial(window_records) if int(args.use_trial_level) else window_records

    meta_df = pd.DataFrame(
        [
            {
                "level": level,
                "subject_id": record.get("subject_id"),
                "user_id": record.get("user_id"),
                "trial_id": record.get("trial_id"),
                "diagnosis_label": record.get("diagnosis_label"),
                "emotion_label": record.get("emotion_label"),
                "label4": record.get("label4"),
                "condition": record.get("condition"),
                "n_windows": record.get("n_windows", 1),
            }
            for record in records
        ]
    )
    meta_df.to_csv(output_dir / f"raw_feature_{level}_metadata.csv", index=False, encoding="utf-8-sig")

    if args.save_feature_npz:
        save_feature_npz(records, output_dir)
    _, summary_df = plot_tsne(records, output_dir, seed=args.seed, max_samples=args.max_samples)

    print(f"[OK] Collected window records: {len(window_records)}")
    print(f"[OK] Visualized {level}-level records: {len(records)}")
    print(f"[OK] Saved raw DE/biomarker t-SNE outputs to: {output_dir}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

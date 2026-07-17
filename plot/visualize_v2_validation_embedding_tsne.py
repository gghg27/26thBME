# -*- coding: utf-8 -*-
"""Draw trial-level t-SNE plots for validation data encoded by V2 checkpoints.

V2 encodes individual EEG windows.  This script follows V2 validation data
loading and subject-relative normalization, then averages the encoded window
representations within each (subject, trial), so every t-SNE point is one
complete trial.

Single fold example:
    conda run -n pytorch python plot/visualize_v2_validation_embedding_tsne.py ^
        --checkpoint model_params/V2_expert_ssas/expert_ssas_repeat0_fold0/stage2_best.pt ^
        --index_csv com_index_sub_2s.csv --device cuda

All folds example:
    conda run -n pytorch python plot/visualize_v2_validation_embedding_tsne.py ^
        --all_folds --checkpoint_root model_params/V2_expert_ssas ^
        --repeat 0 --n_folds 10 --index_csv com_index_sub_2s.csv --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
V2_ROOT = REPO_ROOT / "V2"
if str(V2_ROOT) not in sys.path:
    # V2's original model module uses ``from pmg_backbone import ...``.
    sys.path.insert(0, str(V2_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from V2.train_expert_ssas_emotion import (
    DomainAwareCompetitionDataset,
    build_stage2_model_from_checkpoint,
    compute_subject_bio_baselines,
    compute_subject_de_baselines,
    dict_collate,
    get_subject_relative_kwargs,
    move_batch_to_device,
)


EMOTION_NAMES = {0: "neutral", 1: "positive"}
DIAGNOSIS_NAMES = {0: "DEP", 1: "HC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE of V2-encoded validation trials.")
    parser.add_argument(
        "--checkpoint",
        default="model_params/V2_expert_ssas/expert_ssas_repeat0_fold0/stage2_best.pt",
        help="V2 Stage-2 checkpoint for single-fold mode.",
    )
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--checkpoint_root", default="model_params/V2_expert_ssas")
    parser.add_argument(
        "--run_dir_pattern",
        default="expert_ssas_repeat{repeat}_fold{fold}",
        help="Directory pattern below checkpoint_root.",
    )
    parser.add_argument(
        "--checkpoint_name",
        default="stage2_best.pt",
        help="Checkpoint filename used in every fold directory.",
    )
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--index_csv", default="com_index_sub_2s.csv")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=0.0)
    parser.add_argument("--no_normalize", action="store_true")
    return parser.parse_args()


def resolve_input(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_domain_mapping(checkpoint: dict[str, Any]) -> dict[str, Any]:
    extra_state = checkpoint.get("extra_state", {}) or {}
    mapping = extra_state.get("domain_mapping") or checkpoint.get("domain_mapping")
    if mapping is None:
        raise KeyError("V2 checkpoint has no domain_mapping in extra_state or top level.")
    return mapping


def checkpoint_val_subjects(checkpoint: dict[str, Any], mapping: dict[str, Any]) -> list[str]:
    values = checkpoint.get("val_subjects") or mapping.get("val_subjects")
    if not values:
        raise KeyError("V2 checkpoint has no validation subject list.")
    return [str(value) for value in values]


def build_validation_data(
    checkpoint: dict[str, Any],
    index_csv: Path,
    batch_size: int,
    num_workers: int,
    normalize: bool,
) -> tuple[DomainAwareCompetitionDataset, DataLoader, dict[str, Any], list[str]]:
    mapping = checkpoint_domain_mapping(checkpoint)
    val_subjects = checkpoint_val_subjects(checkpoint, mapping)
    config = checkpoint.get("config", {}) or {}
    use_label4_for_diagnosis = not bool(config.get("use_raw_diagnosis_label", False))
    dataset = DomainAwareCompetitionDataset(
        index_csv=index_csv,
        subject_ids=val_subjects,
        domain_mapping=mapping,
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
    )
    return dataset, loader, mapping, val_subjects


def compute_validation_baselines(
    model: torch.nn.Module,
    dataset: DomainAwareCompetitionDataset,
    loader: DataLoader,
    device: torch.device,
    checkpoint: dict[str, Any],
) -> tuple[dict | None, dict | None, dict | None, dict | None]:
    config = checkpoint.get("config", {}) or {}
    use_de = not bool(config.get("no_subject_relative_de", False))
    use_bio = not bool(config.get("no_subject_relative_bio", False)) and not bool(
        config.get("no_biomarkers", False)
    )
    eps = float(config.get("relative_eps", 1e-6))
    de_mu = de_std = bio_mu = bio_std = None
    if use_de:
        de_mu, de_std = compute_subject_de_baselines(dataset, key_field="target_key", eps=eps)
    if use_bio:
        bio_mu, bio_std = compute_subject_bio_baselines(
            model,
            loader,
            device,
            subject_de_mu=de_mu,
            subject_de_std=de_std,
            key_field="target_key",
            eps=eps,
        )
    return de_mu, de_std, bio_mu, bio_std


@torch.inference_mode()
def encode_validation_windows(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    baselines: tuple[dict | None, dict | None, dict | None, dict | None],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model.eval()
    for batch in tqdm(loader, desc="Encode V2 validation windows"):
        batch = move_batch_to_device(batch, device)
        relative = get_subject_relative_kwargs(
            batch,
            device,
            dtype=batch["de_feat"].dtype,
            de_mu=baselines[0],
            de_std=baselines[1],
            bio_mu=baselines[2],
            bio_std=baselines[3],
        )
        output = model(batch["x"], batch["de_feat"], lambda_subject=0.0, **relative)
        z_emotion = output["z_emotion"].detach().cpu().float().numpy()
        z_diagnosis = output["z_diag"].detach().cpu().float().numpy()
        emotion_prob = output["mix_prob"].detach().cpu().float().numpy()
        diagnosis_prob = torch.softmax(output["diag_logits"], dim=1).detach().cpu().float().numpy()
        subjects = batch["subject_id"].detach().cpu().tolist()
        trials = batch["trial_id"].detach().cpu().tolist()
        true_emotions = batch["emotion_label"].detach().cpu().tolist()
        true_diagnoses = batch["diagnosis_label"].detach().cpu().tolist()
        users = batch.get("user_id", [str(value) for value in subjects])
        for index in range(len(subjects)):
            records.append(
                {
                    "user_id": str(users[index]),
                    "subject_id": int(subjects[index]),
                    "trial_id": int(trials[index]),
                    "true_emotion": int(true_emotions[index]),
                    "true_diagnosis": int(true_diagnoses[index]),
                    "emotion_feature": z_emotion[index],
                    "diagnosis_feature": z_diagnosis[index],
                    "emotion_prob": emotion_prob[index],
                    "diagnosis_prob": diagnosis_prob[index],
                }
            )
    if not records:
        raise RuntimeError("No V2 validation windows were encoded.")
    return records


def majority(values: list[int]) -> int:
    return int(np.bincount(np.asarray(values, dtype=np.int64), minlength=2).argmax())


def aggregate_trials(
    window_records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in window_records:
        groups[(record["user_id"], record["trial_id"])].append(record)

    rows: list[dict[str, Any]] = []
    emotion_features: list[np.ndarray] = []
    diagnosis_features: list[np.ndarray] = []
    for (user_id, trial_id), records in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        emotion_feature = np.stack([row["emotion_feature"] for row in records]).mean(axis=0)
        diagnosis_feature = np.stack([row["diagnosis_feature"] for row in records]).mean(axis=0)
        emotion_prob = np.stack([row["emotion_prob"] for row in records]).mean(axis=0)
        diagnosis_prob = np.stack([row["diagnosis_prob"] for row in records]).mean(axis=0)
        true_emotion = majority([row["true_emotion"] for row in records])
        true_diagnosis = majority([row["true_diagnosis"] for row in records])
        pred_emotion = int(emotion_prob.argmax())
        pred_diagnosis = int(diagnosis_prob.argmax())
        rows.append(
            {
                "user_id": user_id,
                "subject_id": int(records[0]["subject_id"]),
                "trial_id": int(trial_id),
                "n_windows": len(records),
                "true_emotion": true_emotion,
                "true_emotion_name": EMOTION_NAMES[true_emotion],
                "pred_emotion": pred_emotion,
                "pred_emotion_name": EMOTION_NAMES[pred_emotion],
                "prob_positive": float(emotion_prob[1]),
                "true_diagnosis": true_diagnosis,
                "true_diagnosis_name": DIAGNOSIS_NAMES[true_diagnosis],
                "pred_diagnosis": pred_diagnosis,
                "pred_diagnosis_name": DIAGNOSIS_NAMES[pred_diagnosis],
                "prob_hc": float(diagnosis_prob[1]),
            }
        )
        emotion_features.append(emotion_feature)
        diagnosis_features.append(diagnosis_feature)
    return pd.DataFrame(rows), np.stack(emotion_features), np.stack(diagnosis_features)


def standardize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (matrix - mean) / std


def compute_tsne(matrix: np.ndarray, seed: int, requested_perplexity: float) -> tuple[np.ndarray, float, int]:
    features = standardize(matrix)
    n_samples, n_features = features.shape
    if n_samples < 4:
        raise ValueError(f"t-SNE needs at least four trials, got {n_samples}.")
    pca_dim = min(50, n_samples - 1, n_features)
    if n_features > pca_dim:
        features = PCA(n_components=pca_dim, random_state=seed).fit_transform(features)
    perplexity = (
        float(requested_perplexity)
        if requested_perplexity > 0
        else float(max(2, min(30, (n_samples - 1) // 3)))
    )
    if not 0 < perplexity < n_samples:
        raise ValueError(f"perplexity must be in (0, {n_samples}), got {perplexity}.")
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)
    return coords, perplexity, pca_dim


def scatter_categories(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: list[str],
    title: str,
    palette: dict[str, str],
) -> None:
    for label in list(dict.fromkeys(labels)):
        mask = np.asarray([value == label for value in labels])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=34,
            alpha=0.82,
            color=palette[label],
            label=label,
            edgecolor="white",
            linewidth=0.35,
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, fontsize=8, loc="best")


def save_plot(frame: pd.DataFrame, output_dir: Path, fold_label: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    emotion_coords = frame[["emotion_tsne1", "emotion_tsne2"]].to_numpy()
    diagnosis_coords = frame[["diagnosis_tsne1", "diagnosis_tsne2"]].to_numpy()
    emotion_palette = {"neutral": "#4C78A8", "positive": "#F58518"}
    diagnosis_palette = {"DEP": "#D62728", "HC": "#2CA02C"}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.2))
    scatter_categories(
        axes[0, 0], emotion_coords, frame["true_emotion_name"].tolist(),
        "Emotion encoding | true emotion", emotion_palette,
    )
    scatter_categories(
        axes[0, 1], diagnosis_coords, frame["true_diagnosis_name"].tolist(),
        "Diagnosis encoding | true diagnosis", diagnosis_palette,
    )
    scatter_categories(
        axes[1, 0], emotion_coords, frame["pred_emotion_name"].tolist(),
        "Emotion encoding | predicted emotion", emotion_palette,
    )
    scatter_categories(
        axes[1, 1], diagnosis_coords, frame["pred_diagnosis_name"].tolist(),
        "Diagnosis encoding | predicted diagnosis", diagnosis_palette,
    )
    fig.suptitle(f"V2 model-encoded validation-trial distributions | {fold_label}", fontsize=13)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "v2_validation_embedding_tsne.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "v2_validation_embedding_tsne.pdf", bbox_inches="tight")
    plt.close(fig)


def run_one_checkpoint(
    args: argparse.Namespace,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(args.seed)
    index_csv = resolve_input(args.index_csv)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not index_csv.exists():
        raise FileNotFoundError(index_csv)
    print(f"[device] {device}")
    print(f"[checkpoint] {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, _ = build_stage2_model_from_checkpoint(checkpoint, device)
    model.eval()
    dataset, loader, _, val_subjects = build_validation_data(
        checkpoint,
        index_csv,
        args.batch_size,
        args.num_workers,
        normalize=not args.no_normalize,
    )
    print(f"[validation] subjects={val_subjects}; windows={len(dataset)}")
    baselines = compute_validation_baselines(model, dataset, loader, device, checkpoint)
    window_records = encode_validation_windows(model, loader, device, baselines)
    frame, emotion_features, diagnosis_features = aggregate_trials(window_records)
    emotion_coords, emotion_perplexity, emotion_pca_dim = compute_tsne(
        emotion_features, args.seed, args.perplexity
    )
    diagnosis_coords, diagnosis_perplexity, diagnosis_pca_dim = compute_tsne(
        diagnosis_features, args.seed, args.perplexity
    )
    frame[["emotion_tsne1", "emotion_tsne2"]] = emotion_coords
    frame[["diagnosis_tsne1", "diagnosis_tsne2"]] = diagnosis_coords
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        output_dir / "v2_validation_embedding_tsne_coordinates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        output_dir / "v2_validation_trial_embeddings.npz",
        emotion_embedding=emotion_features.astype(np.float32),
        diagnosis_embedding=diagnosis_features.astype(np.float32),
        user_id=frame["user_id"].astype(str).to_numpy(),
        trial_id=frame["trial_id"].to_numpy(dtype=np.int64),
    )
    fold = int(checkpoint.get("fold", -1))
    if fold < 0:
        name = checkpoint_path.parent.name
        try:
            fold = int(name.rsplit("fold", 1)[1])
        except (IndexError, ValueError):
            fold = -1
    summary = {
        "checkpoint": str(checkpoint_path),
        "fold": fold,
        "repeat": int(args.repeat),
        "best_name": checkpoint.get("best_name"),
        "index_csv": str(index_csv),
        "device": str(device),
        "validation_subjects": val_subjects,
        "n_windows": int(len(window_records)),
        "n_trials": int(len(frame)),
        "n_subjects": int(frame["user_id"].nunique()),
        "aggregation": "mean_encoded_windows_per_trial",
        "emotion_feature_dim": int(emotion_features.shape[1]),
        "diagnosis_feature_dim": int(diagnosis_features.shape[1]),
        "emotion_tsne_perplexity": emotion_perplexity,
        "diagnosis_tsne_perplexity": diagnosis_perplexity,
        "emotion_pca_dim": emotion_pca_dim,
        "diagnosis_pca_dim": diagnosis_pca_dim,
        "seed": int(args.seed),
    }
    (output_dir / "v2_validation_embedding_tsne_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fold_label = f"repeat{args.repeat}_fold{fold}" if fold >= 0 else checkpoint_path.parent.name
    save_plot(frame, output_dir, fold_label)
    print(
        f"[done] encoded {len(window_records)} windows -> {len(frame)} trials "
        f"from {frame['user_id'].nunique()} validation subjects"
    )
    print(f"[done] outputs: {output_dir}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    if not args.all_folds:
        checkpoint_path = resolve_input(args.checkpoint)
        output_dir = resolve_input(args.output_dir or "plot/v2_validation_embedding_tsne")
        run_one_checkpoint(args, checkpoint_path, output_dir, device)
        return

    checkpoint_root = resolve_input(args.checkpoint_root)
    output_root = resolve_input(
        args.output_dir or f"plot/v2_validation_embedding_tsne_repeat{args.repeat}_all_folds"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for fold in range(args.n_folds):
        run_dir = args.run_dir_pattern.format(repeat=args.repeat, fold=fold)
        checkpoint_path = checkpoint_root / run_dir / args.checkpoint_name
        fold_output = output_root / f"repeat{args.repeat}_fold{fold}"
        print(f"\n[all folds] fold {fold + 1}/{args.n_folds}")
        summaries.append(run_one_checkpoint(args, checkpoint_path, fold_output, device))
    (output_root / "all_folds_tsne_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_rows = [{**row, "validation_subjects": ",".join(row["validation_subjects"])} for row in summaries]
    pd.DataFrame(csv_rows).to_csv(
        output_root / "all_folds_tsne_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(f"[all folds done] {len(summaries)} folds -> {output_root}")


if __name__ == "__main__":
    main()

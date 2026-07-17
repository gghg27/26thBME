# -*- coding: utf-8 -*-
"""Visualize V7-encoded test or validation trials with t-SNE.

Each point is one complete trial.  The script reconstructs the V7 Stage-2
model from a checkpoint, repeats the subject-relative baseline computation
used during training/inference, and extracts the trial-level emotion and
diagnosis encoder features before the classifier heads.

Example (run from the repository root):
    conda run -n pytorch python plot/visualize_v7_test_embedding_tsne.py ^
        --checkpoint model_params/V7_experiment_a/experiment_a_repeat0_fold0/stage2_best.pt ^
        --split validation --index_csv com_index_sub_2s.csv ^
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

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
from torch.utils.data import DataLoader
from tqdm import tqdm

from V7.experiment_a_model import Stage2ExpertEmotionAdaptationModel
from V7.train_experiment_a import (
    DomainAwareCompetitionDataset,
    UnlabeledTargetDataset,
    baseline_kwargs,
    compute_bio_baseline,
    compute_de_baseline,
    move,
)
from V7.trial_sequence import TrialSequenceDataset, trial_sequence_collate


EMOTION_NAMES = {0: "neutral", 1: "positive"}
DIAGNOSIS_NAMES = {0: "DEP", 1: "HC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE of V7-encoded test or validation trials.")
    parser.add_argument(
        "--checkpoint",
        default="model_params/V7_non_res/experiment_a_repeat0_fold2/stage2_best.pt",
        help="V7 stage2_best.pt to visualize. One checkpoint defines one embedding space.",
    )
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument(
        "--all_folds",
        action="store_true",
        help="Draw validation t-SNE separately for every fold of one repeat.",
    )
    parser.add_argument("--checkpoint_root", default="model_params/V7_non_res")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--index_csv", default="com_index_sub_2s.csv", help="Labeled data index for validation.")
    parser.add_argument("--test_csv", default="com_test_trial_index_2s.csv")
    parser.add_argument(
        "--output_dir",
        default="",
        help="Default: plot/v7_non_res_<split>_embedding_tsne.",
    )
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_windows_per_batch",
        type=int,
        default=72,
        help="Automatically lower the trial batch size to keep GPU memory bounded.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--perplexity",
        type=float,
        default=0.0,
        help="t-SNE perplexity; <=0 chooses a value from the sample count.",
    )
    parser.add_argument(
        "--trial_num_windows",
        type=int,
        default=-1,
        help="-1 uses the checkpoint setting; 0 uses every window.",
    )
    parser.add_argument("--no_normalize", action="store_true", help="Disable the raw-signal z-score used by V7.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_input(path_value: str) -> Path:
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


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("format") != "V7_experiment_a":
        raise ValueError(
            f"Expected a V7 Stage-2 checkpoint, got format={checkpoint.get('format')!r}: {checkpoint_path}"
        )
    model = Stage2ExpertEmotionAdaptationModel(
        int(checkpoint["num_domains"]),
        shared_mix_alpha=float(checkpoint.get("shared_mix_alpha", 0.7)),
        **dict(checkpoint["model_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def validate_test_mapping(test_csv: Path, mapping: dict[str, Any]) -> None:
    frame = pd.read_csv(test_csv)
    id_col = "user_id" if "user_id" in frame.columns else "subject_id"
    missing = [
        user
        for user in sorted(frame[id_col].astype(str).unique())
        if f"test:{user}" not in mapping["key_to_domain"]
    ]
    if missing:
        raise ValueError(
            "The checkpoint domain_mapping does not contain these test users: "
            f"{missing}. Use the checkpoint trained with this test CSV."
        )


def compute_split_baselines(
    model: torch.nn.Module,
    window_dataset: Any,
    model_config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict | None, dict | None, dict | None, dict | None]:
    de_mu = de_std = bio_mu = bio_std = None
    relative_eps = float(model_config.get("relative_eps", 1e-6))
    if model_config.get("use_subject_relative_de", False):
        de_mu, de_std = compute_de_baseline(window_dataset, relative_eps)
    if model_config.get("use_subject_relative_bio", False) and model_config.get("use_biomarkers", True):
        bio_mu, bio_std = compute_bio_baseline(
            model,
            window_dataset,
            device,
            de_mu,
            de_std,
            batch_size,
            num_workers,
            relative_eps,
        )
    return de_mu, de_std, bio_mu, bio_std


@torch.inference_mode()
def extract_trial_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    baselines: tuple[dict | None, dict | None, dict | None, dict | None],
    split: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    emotion_features: list[np.ndarray] = []
    diagnosis_features: list[np.ndarray] = []

    for batch in tqdm(loader, desc=f"Encode V7 {split} trials"):
        batch = move(batch, device)
        relative = baseline_kwargs(batch, device, *baselines)
        output = model(batch["x"], batch["de_feat"], batch["window_mask"], **relative)
        emotion = output["z_end_emotion"].detach().cpu().float().numpy()
        diagnosis = output["z_end_diag"].detach().cpu().float().numpy()
        emotion_prob = output["mix_prob"].detach().cpu().float().numpy()
        diagnosis_prob = torch.softmax(output["diag_logits"], dim=1).detach().cpu().float().numpy()

        users = [str(value) for value in batch["user_id"]]
        trials = batch["trial_id"].detach().cpu().tolist()
        subject_ids = batch["subject_id"].detach().cpu().tolist()
        window_counts = batch["window_mask"].sum(dim=1).detach().cpu().tolist()
        true_emotions = batch["emotion_label"].detach().cpu().tolist() if split == "validation" else None
        true_diagnoses = batch["diagnosis_label"].detach().cpu().tolist() if split == "validation" else None
        for index, user in enumerate(users):
            pred_emotion = int(emotion_prob[index].argmax())
            pred_diagnosis = int(diagnosis_prob[index].argmax())
            row = {
                    "user_id": user,
                    "subject_id": int(subject_ids[index]),
                    "trial_id": int(trials[index]),
                    "n_windows": int(window_counts[index]),
                    "pred_emotion": pred_emotion,
                    "pred_emotion_name": EMOTION_NAMES[pred_emotion],
                    "prob_positive": float(emotion_prob[index, 1]),
                    "pred_diagnosis": pred_diagnosis,
                    "pred_diagnosis_name": DIAGNOSIS_NAMES[pred_diagnosis],
                    "prob_hc": float(diagnosis_prob[index, 1]),
                }
            if true_emotions is not None and true_diagnoses is not None:
                true_emotion = int(true_emotions[index])
                true_diagnosis = int(true_diagnoses[index])
                row.update(
                    true_emotion=true_emotion,
                    true_emotion_name=EMOTION_NAMES[true_emotion],
                    true_diagnosis=true_diagnosis,
                    true_diagnosis_name=DIAGNOSIS_NAMES[true_diagnosis],
                )
            rows.append(row)
        emotion_features.append(emotion)
        diagnosis_features.append(diagnosis)

    if not rows:
        raise RuntimeError(f"No {split} trials were encoded.")
    return pd.DataFrame(rows), np.concatenate(emotion_features), np.concatenate(diagnosis_features)


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
    if requested_perplexity > 0:
        perplexity = requested_perplexity
    else:
        perplexity = float(max(2, min(30, (n_samples - 1) // 3)))
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
    palette: dict[str, Any] | None = None,
) -> None:
    unique = list(dict.fromkeys(labels))
    fallback = plt.get_cmap("tab20")
    for index, label in enumerate(unique):
        mask = np.asarray([value == label for value in labels])
        color = palette.get(label) if palette else fallback(index % 20)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=34,
            alpha=0.82,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.35,
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=False, fontsize=7, loc="best", ncol=2 if len(unique) > 6 else 1)


def save_plot(frame: pd.DataFrame, output_dir: Path, split: str) -> None:
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
    pred_emotion_labels = frame["pred_emotion_name"].astype(str).tolist()
    pred_diagnosis_labels = frame["pred_diagnosis_name"].astype(str).tolist()

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.2))
    scatter_categories(
        axes[0, 0], emotion_coords,
        frame["true_emotion_name"].astype(str).tolist() if split == "validation" else pred_emotion_labels,
        f"Emotion encoding | {'true' if split == 'validation' else 'predicted'} emotion",
        {"neutral": "#4C78A8", "positive": "#F58518"},
    )
    scatter_categories(
        axes[0, 1], diagnosis_coords,
        frame["true_diagnosis_name"].astype(str).tolist() if split == "validation" else pred_diagnosis_labels,
        f"Diagnosis encoding | {'true' if split == 'validation' else 'predicted'} diagnosis",
        {"DEP": "#D62728", "HC": "#2CA02C"},
    )
    if split == "validation":
        scatter_categories(
            axes[1, 0], emotion_coords, pred_emotion_labels, "Emotion encoding | predicted emotion",
            {"neutral": "#4C78A8", "positive": "#F58518"},
        )
        scatter_categories(
            axes[1, 1], diagnosis_coords, pred_diagnosis_labels, "Diagnosis encoding | predicted diagnosis",
            {"DEP": "#D62728", "HC": "#2CA02C"},
        )
    else:
        subject_labels = frame["user_id"].astype(str).tolist()
        scatter_categories(axes[1, 0], emotion_coords, subject_labels, "Emotion encoding | test subject")
        scatter_categories(axes[1, 1], diagnosis_coords, subject_labels, "Diagnosis encoding | test subject")
    fig.suptitle(f"V7 model-encoded {split}-trial distributions", fontsize=13)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"v7_{split}_embedding_tsne.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"v7_{split}_embedding_tsne.pdf", bbox_inches="tight")
    plt.close(fig)


def run_one_checkpoint(
    args: argparse.Namespace,
    checkpoint_path: Path,
    output_dir: Path,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(args.seed)
    data_csv = resolve_input(args.test_csv if split == "test" else args.index_csv)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not data_csv.exists():
        raise FileNotFoundError(data_csv)

    print(f"[device] {device}")
    print(f"[checkpoint] {checkpoint_path}")
    model, checkpoint = load_model(checkpoint_path, device)
    mapping = checkpoint["domain_mapping"]
    if split == "test":
        validate_test_mapping(data_csv, mapping)
        window_dataset = UnlabeledTargetDataset(data_csv, mapping, normalize=not args.no_normalize)
    else:
        val_subjects = mapping.get("val_subjects", [])
        if not val_subjects:
            raise ValueError("The checkpoint has no val_subjects in its domain_mapping.")
        use_label4 = not bool(checkpoint.get("args", {}).get("use_raw_diagnosis_label", False))
        window_dataset = DomainAwareCompetitionDataset(
            data_csv,
            val_subjects,
            mapping,
            "val",
            normalize=not args.no_normalize,
            use_label4_for_diagnosis=use_label4,
        )
    trial_num_windows = (
        int(checkpoint.get("trial_num_windows", 0))
        if args.trial_num_windows < 0
        else args.trial_num_windows
    )
    trial_dataset = TrialSequenceDataset(window_dataset, trial_num_windows, f"V7-{split}-tSNE")
    max_trial_windows = max(
        min(len(group["window_indices"]), trial_num_windows)
        if trial_num_windows > 0
        else len(group["window_indices"])
        for group in trial_dataset.trial_groups
    )
    safe_batch_size = max(1, args.max_windows_per_batch // max_trial_windows)
    encoding_batch_size = min(args.batch_size, safe_batch_size)
    if encoding_batch_size < args.batch_size:
        print(
            f"[memory guard] max_trial_windows={max_trial_windows}; "
            f"encoding batch_size {args.batch_size} -> {encoding_batch_size}"
        )
    loader = DataLoader(
        trial_dataset,
        batch_size=encoding_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=trial_sequence_collate,
    )

    baselines = compute_split_baselines(
        model,
        window_dataset,
        dict(checkpoint["model_config"]),
        device,
        args.batch_size,
        args.num_workers,
    )
    frame, emotion_features, diagnosis_features = extract_trial_features(
        model, loader, device, baselines, split
    )
    emotion_coords, emotion_perplexity, emotion_pca_dim = compute_tsne(
        emotion_features, args.seed, args.perplexity
    )
    diagnosis_coords, diagnosis_perplexity, diagnosis_pca_dim = compute_tsne(
        diagnosis_features, args.seed, args.perplexity
    )
    frame[["emotion_tsne1", "emotion_tsne2"]] = emotion_coords
    frame[["diagnosis_tsne1", "diagnosis_tsne2"]] = diagnosis_coords

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"v7_{split}_embedding_tsne_coordinates.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output_dir / f"v7_{split}_trial_embeddings.npz",
        emotion_embedding=emotion_features.astype(np.float32),
        diagnosis_embedding=diagnosis_features.astype(np.float32),
        user_id=frame["user_id"].astype(str).to_numpy(),
        trial_id=frame["trial_id"].to_numpy(dtype=np.int64),
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "fold": int(checkpoint.get("fold", -1)),
        "repeat": int(checkpoint.get("repeat", -1)),
        "data_csv": str(data_csv),
        "device": str(device),
        "n_trials": int(len(frame)),
        "n_subjects": int(frame["user_id"].nunique()),
        "trial_num_windows": int(trial_num_windows),
        "emotion_feature_dim": int(emotion_features.shape[1]),
        "diagnosis_feature_dim": int(diagnosis_features.shape[1]),
        "emotion_tsne_perplexity": emotion_perplexity,
        "diagnosis_tsne_perplexity": diagnosis_perplexity,
        "emotion_pca_dim": emotion_pca_dim,
        "diagnosis_pca_dim": diagnosis_pca_dim,
        "seed": int(args.seed),
    }
    (output_dir / f"v7_{split}_embedding_tsne_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plot(frame, output_dir, split)
    print(f"[done] encoded {len(frame)} trials from {frame['user_id'].nunique()} {split} subjects")
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
        output_dir = resolve_input(args.output_dir or f"plot/v7_{args.split}_embedding_tsne")
        run_one_checkpoint(args, checkpoint_path, output_dir, args.split, device)
        return

    checkpoint_root = resolve_input(args.checkpoint_root)
    output_root = resolve_input(
        args.output_dir or f"plot/v7_validation_embedding_tsne_repeat{args.repeat}_all_folds"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for fold in range(args.n_folds):
        checkpoint_path = (
            checkpoint_root
            / f"experiment_a_repeat{args.repeat}_fold{fold}"
            / "stage2_best.pt"
        )
        fold_output = output_root / f"repeat{args.repeat}_fold{fold}"
        print(f"\n[all folds] fold {fold + 1}/{args.n_folds}")
        summaries.append(
            run_one_checkpoint(args, checkpoint_path, fold_output, "validation", device)
        )
    (output_root / "all_folds_tsne_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(
        output_root / "all_folds_tsne_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(f"[all folds done] {len(summaries)} folds -> {output_root}")


if __name__ == "__main__":
    main()

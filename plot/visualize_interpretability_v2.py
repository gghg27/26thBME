# -*- coding: utf-8 -*-
"""Interpretability and visualization utilities for the existing V2 model.

Example:
    python plot/visualize_interpretability_v2.py ^
        --ckpt model_params/V2_expert_ssas/fold0/stage2_best_topk_trial_f1_fold0.pt ^
        --output_dir plot/interpretability_outputs ^
        --split val --batch_size 64 --device cuda
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "V2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

import dataloader as dataloader_module
from V2.expert_ssas_emotion_model import Stage2ExpertEmotionAdaptationModel
from V2.train_expert_ssas_emotion import (
    DomainAwareCompetitionDataset,
    build_domain_id_mapping,
    build_train_val_target_split,
    compute_subject_bio_baselines,
    compute_subject_de_baselines,
    dict_collate,
    get_subject_relative_kwargs,
    load_checkpoint,
    move_batch_to_device,
    seed_worker,
)


CHANNELS_RAW = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ",
    "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ",
    "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
]
CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FT7", "FC3", "FCz",
    "FC4", "FT8", "T3", "C3", "Cz", "C4", "T4", "TP7", "CP3", "CPz",
    "CP4", "TP8", "T5", "P3", "Pz", "P4", "T6", "O1", "Oz", "O2",
]
CONDITION_ORDER = ["HC_neu", "HC_pos", "DEP_neu", "DEP_pos"]
LABEL4_TO_CONDITION = {
    0: "DEP_neu",
    1: "DEP_pos",
    2: "HC_neu",
    3: "HC_pos",
}
DIAGNOSIS_NAMES = {0: "DEP", 1: "HC"}
EMOTION_NAMES = {0: "neu", 1: "pos"}
FEATURE_KEYS = {
    "z_abs": ["z_abs", "z_diag", "feat_abs", "abs_feat"],
    "z_rel": ["z_rel", "z_emotion", "feat_rel", "rel_feat"],
    "z_fused": ["z_fused", "z", "z_core", "fused_feat"],
    "plv_adj": ["plv_adj", "plv_matrix", "plv_matrix_emotion", "adj_dense", "adj"],
    "route_weight": ["route_weight", "router_weight", "expert_weight"],
    "emotion_prob": ["emotion_prob", "mix_prob", "expert_mix_prob", "prob_emo"],
    "diagnosis_prob": ["diagnosis_prob", "diag_prob", "prob_diag"],
    "emotion_logits": ["emotion_logits", "logits", "mix_logits"],
    "diagnosis_logits": ["diagnosis_logits", "diag_logits"],
}

CHANNEL_POS = {
    "Fp1": (-0.35, 0.98), "Fp2": (0.35, 0.98),
    "F7": (-0.85, 0.66), "F3": (-0.45, 0.66), "Fz": (0.0, 0.70), "F4": (0.45, 0.66), "F8": (0.85, 0.66),
    "FT7": (-0.95, 0.35), "FC3": (-0.48, 0.35), "FCz": (0.0, 0.38), "FC4": (0.48, 0.35), "FT8": (0.95, 0.35),
    "T3": (-1.0, 0.0), "C3": (-0.50, 0.0), "Cz": (0.0, 0.0), "C4": (0.50, 0.0), "T4": (1.0, 0.0),
    "TP7": (-0.95, -0.35), "CP3": (-0.48, -0.35), "CPz": (0.0, -0.38), "CP4": (0.48, -0.35), "TP8": (0.95, -0.35),
    "T5": (-0.85, -0.66), "P3": (-0.45, -0.66), "Pz": (0.0, -0.70), "P4": (0.45, -0.66), "T6": (0.85, -0.66),
    "O1": (-0.35, -0.98), "Oz": (0.0, -1.03), "O2": (0.35, -0.98),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 interpretability figures and intermediate outputs.")
    parser.add_argument("--data_root", type=str, default=str(ROOT), help="Root used by the existing csv trial/de paths.")
    parser.add_argument("--index_csv", type=str, default="com_index_sub_2s.csv")
    parser.add_argument("--test_csv", type=str, default="com_test_trial_index_2s.csv")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="plot/interpretability_outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--top_k_edges", type=int, default=30)
    parser.add_argument("--max_occlusion_trials", type=int, default=200)
    parser.add_argument("--use_trial_level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--n_splits", type=int, default=None)
    parser.add_argument("--occlusion_target", choices=["true_label", "pred_label", "positive_prob", "neutral_prob"], default="true_label")
    parser.add_argument("--occlusion_edge_batch_size", type=int, default=64)
    parser.add_argument("--occlusion_fill", choices=["zero", "mean"], default="zero")
    parser.add_argument("--skip_occlusion", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--use_raw_diagnosis_label", action="store_true")
    parser.add_argument("--strict_load", action="store_true", help="Require an exact checkpoint/model state_dict match.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_existing_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = [base / path, ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def checkpoint_domain_mapping(ckpt: dict) -> Optional[dict]:
    extra_state = ckpt.get("extra_state", {}) or {}
    return extra_state.get("domain_mapping") or ckpt.get("domain_mapping")


def build_domain_mapping_if_needed(args: argparse.Namespace, ckpt: dict) -> tuple[dict, list, list]:
    data_root = Path(args.data_root).resolve()
    index_csv = resolve_existing_path(args.index_csv, data_root)
    test_csv = resolve_existing_path(args.test_csv, data_root)
    config_dict = ckpt.get("config", {}) or {}

    domain_mapping = checkpoint_domain_mapping(ckpt)
    train_subjects = ckpt.get("train_subjects")
    val_subjects = ckpt.get("val_subjects")

    if domain_mapping is not None:
        train_subjects = train_subjects or domain_mapping.get("source_subjects")
        val_subjects = val_subjects or domain_mapping.get("val_subjects")
    if train_subjects is None or val_subjects is None:
        fold = args.fold if args.fold is not None else int(config_dict.get("fold", 0))
        n_splits = args.n_splits if args.n_splits is not None else int(config_dict.get("n_splits", 5))
        split_seed = int(config_dict.get("rand_seed", args.seed))
        split = build_train_val_target_split(str(index_csv), fold=fold, n_splits=n_splits, seed=split_seed)
        train_subjects = split["train_all"]
        val_subjects = split["val_all"]
    if domain_mapping is None:
        domain_mapping = build_domain_id_mapping(train_subjects, val_subjects, str(test_csv) if test_csv.exists() else None)
    return domain_mapping, train_subjects, val_subjects


def load_model_state_compat(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    strict: bool = False,
) -> None:
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint model_state_dict must be a dict, got {type(state)!r}")

    if state and all(str(key).startswith("module.") for key in state.keys()):
        state = {str(key)[7:]: value for key, value in state.items()}

    if strict:
        model.load_state_dict(state, strict=True)
        return

    model_state = model.state_dict()
    compatible_state = {}
    incompatible = []
    for key, value in state.items():
        if key not in model_state:
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            incompatible.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible_state[key] = value

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    skipped_unexpected = [key for key in state.keys() if key not in model_state]
    if missing:
        print(f"[WARN] Missing checkpoint keys loaded with current defaults: {len(missing)}")
        print("[WARN] Missing examples:", ", ".join(list(missing)[:8]))
    if skipped_unexpected:
        print(f"[WARN] Ignored unexpected checkpoint keys: {len(skipped_unexpected)}")
        print("[WARN] Unexpected examples:", ", ".join(skipped_unexpected[:8]))
    if incompatible:
        print(f"[WARN] Ignored shape-mismatched checkpoint keys: {len(incompatible)}")
        examples = [f"{key}: ckpt{src} != model{dst}" for key, src, dst in incompatible[:5]]
        print("[WARN] Shape mismatch examples:", "; ".join(examples))
    if unexpected:
        print(f"[WARN] load_state_dict reported unexpected compatible keys: {len(unexpected)}")


def build_stage2_model(
    ckpt: dict,
    domain_mapping: dict,
    device: torch.device,
    strict_load: bool = False,
) -> Stage2ExpertEmotionAdaptationModel:
    config_dict = ckpt.get("config", {}) or {}
    use_biomarkers = not bool(config_dict.get("no_biomarkers", False))
    use_subject_relative_de = not bool(config_dict.get("no_subject_relative_de", False))
    use_subject_relative_bio = (not bool(config_dict.get("no_subject_relative_bio", False))) and use_biomarkers
    model = Stage2ExpertEmotionAdaptationModel(
        num_domains=int(domain_mapping["num_domains"]),
        sfreq=float(config_dict.get("sfreq", 250.0)),
        topk=int(config_dict.get("topk", 8)),
        dropout=float(config_dict.get("dropout", 0.35)),
        use_biomarkers=use_biomarkers,
        biomarker_dim=int(config_dict.get("biomarker_dim", 57)),
        use_subject_relative_de=use_subject_relative_de,
        use_subject_relative_bio=use_subject_relative_bio,
        bio_abs_scale=float(config_dict.get("bio_abs_scale", 0.3)),
        relative_eps=float(config_dict.get("relative_eps", 1e-6)),
        de_num_bands=int(config_dict.get("de_num_bands", 5)),
        shared_mix_alpha=float(config_dict.get("shared_mix_alpha", 0.5)),
    ).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    load_model_state_compat(model, state, strict=strict_load)
    model.eval()
    return model


def build_dataset_and_loader(
    args: argparse.Namespace,
    domain_mapping: dict,
    train_subjects: list,
    val_subjects: list,
) -> tuple[Dataset, DataLoader]:
    data_root = Path(args.data_root).resolve()
    dataloader_module.ROOT = data_root
    index_csv = resolve_existing_path(args.index_csv, data_root)
    normalize = not bool(args.no_normalize)
    use_label4_for_diagnosis = not bool(args.use_raw_diagnosis_label)

    source_dataset = DomainAwareCompetitionDataset(
        index_csv=index_csv,
        subject_ids=train_subjects,
        domain_mapping=domain_mapping,
        split_prefix="source",
        normalize=normalize,
        use_label4_for_diagnosis=use_label4_for_diagnosis,
    )
    val_dataset = DomainAwareCompetitionDataset(
        index_csv=index_csv,
        subject_ids=val_subjects,
        domain_mapping=domain_mapping,
        split_prefix="val",
        normalize=normalize,
        use_label4_for_diagnosis=use_label4_for_diagnosis,
    )
    if args.split == "train":
        dataset = source_dataset
    elif args.split == "val":
        dataset = val_dataset
    else:
        dataset = ConcatDataset([source_dataset, val_dataset])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dict_collate,
        worker_init_fn=seed_worker,
    )
    return dataset, loader


def tensor_to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    arr = np.asarray(value)
    return arr


def first_output(out: dict, keys: list[str]) -> Any:
    for key in keys:
        value = out.get(key, None)
        if value is not None:
            return value
    return None


def softmax_numpy(logits: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if logits is None:
        return None
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


def reduce_plv_array(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim > 3:
        arr = arr.mean(axis=1)
    if arr.ndim != 3:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = 0.5 * (arr + np.swapaxes(arr, -1, -2))
    return arr


def standardize_output(out: dict) -> dict[str, Optional[np.ndarray]]:
    emotion_prob = tensor_to_numpy(first_output(out, FEATURE_KEYS["emotion_prob"]))
    if emotion_prob is None:
        emotion_prob = softmax_numpy(tensor_to_numpy(first_output(out, FEATURE_KEYS["emotion_logits"])))
    diagnosis_prob = tensor_to_numpy(first_output(out, FEATURE_KEYS["diagnosis_prob"]))
    if diagnosis_prob is None:
        diagnosis_prob = softmax_numpy(tensor_to_numpy(first_output(out, FEATURE_KEYS["diagnosis_logits"])))

    route_weight = tensor_to_numpy(first_output(out, FEATURE_KEYS["route_weight"]))
    if route_weight is None and diagnosis_prob is not None and diagnosis_prob.shape[1] >= 2:
        route_weight = np.stack([diagnosis_prob[:, 1], diagnosis_prob[:, 0]], axis=1).astype(np.float32)

    return {
        "emotion_prob": emotion_prob,
        "diagnosis_prob": diagnosis_prob,
        "plv_adj": reduce_plv_array(tensor_to_numpy(first_output(out, FEATURE_KEYS["plv_adj"]))),
        "z_abs": tensor_to_numpy(first_output(out, FEATURE_KEYS["z_abs"])),
        "z_rel": tensor_to_numpy(first_output(out, FEATURE_KEYS["z_rel"])),
        "z_fused": tensor_to_numpy(first_output(out, FEATURE_KEYS["z_fused"])),
        "route_weight": route_weight,
    }


def batch_value(batch: dict, key: str, i: int, default: Any = None) -> Any:
    if key not in batch:
        return default
    value = batch[key]
    if torch.is_tensor(value):
        return value[i].detach().cpu().item()
    return value[i]


def infer_condition(label4: Optional[int], diagnosis_label: Optional[int], emotion_label: Optional[int]) -> str:
    if label4 is not None and int(label4) in LABEL4_TO_CONDITION:
        return LABEL4_TO_CONDITION[int(label4)]
    diag_name = DIAGNOSIS_NAMES.get(int(diagnosis_label), f"diag{diagnosis_label}") if diagnosis_label is not None else "diagNA"
    emo_name = EMOTION_NAMES.get(int(emotion_label), f"emo{emotion_label}") if emotion_label is not None else "emoNA"
    return f"{diag_name}_{emo_name}"


def compute_optional_baselines(
    model: Stage2ExpertEmotionAdaptationModel,
    dataset: Dataset,
    loader: DataLoader,
    device: torch.device,
    relative_eps: float,
) -> tuple[Optional[dict], Optional[dict], Optional[dict], Optional[dict]]:
    de_mu = de_std = bio_mu = bio_std = None
    if getattr(model, "use_subject_relative_de", False):
        print("[Info] Computing subject-relative DE baselines ...")
        de_mu, de_std = compute_subject_de_baselines(dataset, key_field="target_key", eps=relative_eps)
    if getattr(model, "use_subject_relative_bio", False):
        print("[Info] Computing subject-relative biomarker baselines ...")
        bio_mu, bio_std = compute_subject_bio_baselines(
            model,
            loader,
            device,
            subject_de_mu=de_mu,
            subject_de_std=de_std,
            key_field="target_key",
            eps=relative_eps,
        )
    return de_mu, de_std, bio_mu, bio_std


@torch.no_grad()
def collect_window_records(
    model: Stage2ExpertEmotionAdaptationModel,
    loader: DataLoader,
    device: torch.device,
    de_mu: Optional[dict],
    de_std: Optional[dict],
    bio_mu: Optional[dict],
    bio_std: Optional[dict],
) -> list[dict]:
    records: list[dict] = []
    model.eval()
    for batch in tqdm(loader, desc="Collect V2 interpretability records", leave=False):
        batch = move_batch_to_device(batch, device)
        rel_kwargs = get_subject_relative_kwargs(
            batch,
            device,
            dtype=batch["de_feat"].dtype,
            de_mu=de_mu,
            de_std=de_std,
            bio_mu=bio_mu,
            bio_std=bio_std,
        )
        out = model(
            batch["x"],
            batch["de_feat"],
            lambda_subject=0.0,
            return_features=True,
            return_plv=True,
            return_route=True,
            return_logits=True,
            **rel_kwargs,
        )
        std = standardize_output(out)
        bsz = int(batch["x"].size(0))
        for i in range(bsz):
            diagnosis_label = int(batch_value(batch, "diagnosis_label", i, 0))
            emotion_label = int(batch_value(batch, "emotion_label", i, 0))
            label4 = int(batch_value(batch, "label4", i, diagnosis_label * 2 + emotion_label))
            emotion_prob = None if std["emotion_prob"] is None else std["emotion_prob"][i].astype(np.float32)
            diagnosis_prob = None if std["diagnosis_prob"] is None else std["diagnosis_prob"][i].astype(np.float32)
            record = {
                "subject_id": int(batch_value(batch, "subject_id", i, i)),
                "user_id": str(batch_value(batch, "user_id", i, batch_value(batch, "subject_id", i, i))),
                "target_key": str(batch_value(batch, "target_key", i, batch_value(batch, "subject_id", i, i))),
                "trial_id": int(batch_value(batch, "trial_id", i, i)),
                "diagnosis_label": diagnosis_label,
                "emotion_label": emotion_label,
                "label4": label4,
                "condition": infer_condition(label4, diagnosis_label, emotion_label),
                "emotion_prob": emotion_prob,
                "diagnosis_prob": diagnosis_prob,
                "emotion_pred": int(np.argmax(emotion_prob)) if emotion_prob is not None else None,
                "diagnosis_pred": int(np.argmax(diagnosis_prob)) if diagnosis_prob is not None else None,
                "emotion_confidence": float(np.max(emotion_prob)) if emotion_prob is not None else np.nan,
                "plv_adj": None if std["plv_adj"] is None else std["plv_adj"][i].astype(np.float32),
                "z_abs": None if std["z_abs"] is None else std["z_abs"][i].reshape(-1).astype(np.float32),
                "z_rel": None if std["z_rel"] is None else std["z_rel"][i].reshape(-1).astype(np.float32),
                "z_fused": None if std["z_fused"] is None else std["z_fused"][i].reshape(-1).astype(np.float32),
                "route_weight": None if std["route_weight"] is None else std["route_weight"][i].reshape(-1).astype(np.float32),
                "_x": batch["x"][i].detach().cpu().float().numpy().astype(np.float32),
                "_de_feat": batch["de_feat"][i].detach().cpu().float().numpy().astype(np.float32),
            }
            records.append(record)
    return records


def mean_stack(values: list[Any]) -> Optional[np.ndarray]:
    arrays = [np.asarray(v, dtype=np.float32) for v in values if v is not None]
    if not arrays:
        return None
    shape = arrays[0].shape
    if any(arr.shape != shape for arr in arrays):
        warnings.warn(f"Cannot average arrays with inconsistent shapes: {[arr.shape for arr in arrays[:5]]}")
        return arrays[0]
    return np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)


def window_to_trial_aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[Any, Any], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record.get("subject_id"), record.get("trial_id"))].append(record)

    trial_records: list[dict] = []
    array_keys = [
        "plv_adj", "z_abs", "z_rel", "z_fused", "emotion_prob", "diagnosis_prob",
        "route_weight", "_x", "_de_feat",
    ]
    label_keys = ["diagnosis_label", "emotion_label", "label4", "condition"]
    for (subject_id, trial_id), rows in sorted(grouped.items(), key=lambda kv: (str(kv[0][0]), int(kv[0][1]))):
        first = rows[0]
        trial = {
            "subject_id": subject_id,
            "user_id": first.get("user_id", str(subject_id)),
            "target_key": first.get("target_key", str(subject_id)),
            "trial_id": trial_id,
            "n_windows": len(rows),
        }
        for key in label_keys:
            values = [row.get(key) for row in rows if row.get(key) is not None]
            unique = list(dict.fromkeys(values))
            if len(unique) > 1:
                warnings.warn(f"Inconsistent {key} within subject={subject_id}, trial={trial_id}: {unique}")
            trial[key] = unique[0] if unique else None
        for key in array_keys:
            trial[key] = mean_stack([row.get(key) for row in rows])
        if trial.get("emotion_prob") is not None:
            trial["emotion_pred"] = int(np.argmax(trial["emotion_prob"]))
            trial["emotion_confidence"] = float(np.max(trial["emotion_prob"]))
        else:
            trial["emotion_pred"] = first.get("emotion_pred")
            trial["emotion_confidence"] = first.get("emotion_confidence", np.nan)
        if trial.get("diagnosis_prob") is not None:
            trial["diagnosis_pred"] = int(np.argmax(trial["diagnosis_prob"]))
        else:
            trial["diagnosis_pred"] = first.get("diagnosis_pred")
        if trial.get("condition") is None:
            trial["condition"] = infer_condition(trial.get("label4"), trial.get("diagnosis_label"), trial.get("emotion_label"))
        if trial.get("emotion_label") is not None and trial.get("emotion_pred") is not None:
            trial["emotion_correct"] = int(trial["emotion_label"] == trial["emotion_pred"])
        else:
            trial["emotion_correct"] = None
        trial_records.append(trial)
    return trial_records


def strip_private(records: list[dict]) -> list[dict]:
    stripped = []
    for record in records:
        stripped.append({k: v for k, v in record.items() if not k.startswith("_")})
    return stripped


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_figure(fig: plt.Figure, output_dir: Path, base_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{base_name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / f"{base_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def upper_edges(matrix: np.ndarray, top_k: Optional[int] = None, by_abs: bool = False) -> list[tuple[int, int, float]]:
    n = int(matrix.shape[0])
    edges = [(i, j, float(matrix[i, j])) for i in range(n) for j in range(i + 1, n)]
    edges.sort(key=lambda e: abs(e[2]) if by_abs else e[2], reverse=True)
    if top_k is not None:
        edges = edges[: max(0, int(top_k))]
    return edges


def append_top_edges(
    rows: list[dict],
    figure_name: str,
    matrix: np.ndarray,
    top_k: int,
    edge_type: str,
    by_abs: bool = True,
) -> None:
    for rank, (i, j, value) in enumerate(upper_edges(matrix, top_k=top_k, by_abs=by_abs), start=1):
        rows.append(
            {
                "figure_name": figure_name,
                "edge_i": int(i),
                "edge_j": int(j),
                "channel_i": CHANNELS[i] if i < len(CHANNELS) else f"node{i}",
                "channel_j": CHANNELS[j] if j < len(CHANNELS) else f"node{j}",
                "value": float(value),
                "rank": int(rank),
                "edge_type": edge_type,
            }
        )


def draw_connectome(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    top_k: int = 30,
    signed: bool = False,
    edge_set: Optional[set[tuple[int, int]]] = None,
) -> None:
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    if edge_set is None:
        edges = upper_edges(matrix, top_k=top_k, by_abs=signed)
    else:
        edges = sorted([(i, j, float(matrix[i, j])) for i, j in edge_set], key=lambda e: abs(e[2]), reverse=True)
    max_abs = max([abs(v) for _, _, v in edges], default=1.0)

    circle = plt.Circle((0, 0), 1.08, edgecolor="#555555", facecolor="none", linewidth=0.8)
    ax.add_patch(circle)
    for i, j, value in edges:
        if i >= len(CHANNELS) or j >= len(CHANNELS):
            continue
        x1, y1 = CHANNEL_POS[CHANNELS[i]]
        x2, y2 = CHANNEL_POS[CHANNELS[j]]
        color = "#D55E00" if (not signed or value >= 0) else "#0072B2"
        alpha = 0.25 + 0.65 * min(abs(value) / max_abs, 1.0)
        width = 0.5 + 2.2 * min(abs(value) / max_abs, 1.0)
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=width, zorder=1)

    for idx, name in enumerate(CHANNELS[:n]):
        x, y = CHANNEL_POS.get(name, (0.0, 0.0))
        ax.scatter([x], [y], s=24, color="#222222", zorder=3)
        ax.text(x, y + 0.035, name, ha="center", va="bottom", fontsize=6.5)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")


def plot_matrix_grid(
    matrices: list[np.ndarray],
    titles: list[str],
    output_dir: Path,
    base_name: str,
    cmap: str,
    diverging: bool = False,
    ncols: int = 2,
) -> None:
    valid = [np.asarray(m, dtype=float) for m in matrices]
    if diverging:
        vmax = max(float(np.nanmax(np.abs(m))) for m in valid)
        vmax = max(vmax, 1e-6)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        vmin = vmax_value = None
    else:
        vmin = min(float(np.nanmin(m)) for m in valid)
        vmax_value = max(float(np.nanmax(m)) for m in valid)
        norm = None
    nrows = int(math.ceil(len(valid) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    last_im = None
    for ax, matrix, title in zip(axes.ravel(), valid, titles):
        last_im = ax.imshow(matrix, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax_value)
        ax.set_title(title)
        ax.set_xticks(range(len(CHANNELS)))
        ax.set_yticks(range(len(CHANNELS)))
        ax.set_xticklabels(CHANNELS, rotation=90)
        ax.set_yticklabels(CHANNELS)
    for ax in axes.ravel()[len(valid):]:
        ax.axis("off")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.72)
    save_figure(fig, output_dir, base_name)


def plot_connectome_grid(
    matrices: list[np.ndarray],
    titles: list[str],
    output_dir: Path,
    base_name: str,
    top_k: int,
    signed: bool = False,
    ncols: int = 2,
) -> None:
    nrows = int(math.ceil(len(matrices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows), squeeze=False)
    for ax, matrix, title in zip(axes.ravel(), matrices, titles):
        draw_connectome(ax, matrix, title, top_k=top_k, signed=signed)
    for ax in axes.ravel()[len(matrices):]:
        ax.axis("off")
    save_figure(fig, output_dir, base_name)


def mean_plv_by_condition(trial_records: list[dict]) -> dict[str, np.ndarray]:
    subject_condition: dict[tuple[Any, str], list[np.ndarray]] = defaultdict(list)
    for record in trial_records:
        plv = record.get("plv_adj")
        condition = record.get("condition")
        if plv is None or condition not in CONDITION_ORDER:
            continue
        subject_condition[(record.get("subject_id"), condition)].append(plv)

    per_subject_mean: dict[str, list[np.ndarray]] = defaultdict(list)
    for (_, condition), values in subject_condition.items():
        per_subject_mean[condition].append(np.mean(np.stack(values, axis=0), axis=0))

    result = {}
    for condition, values in per_subject_mean.items():
        result[condition] = np.mean(np.stack(values, axis=0), axis=0).astype(np.float32)
    return result


def make_figures_1_to_3(mean_plv: dict[str, np.ndarray], output_dir: Path, top_k: int, top_rows: list[dict]) -> dict[str, np.ndarray]:
    missing = [cond for cond in CONDITION_ORDER if cond not in mean_plv]
    if missing:
        warnings.warn(f"Missing conditions for Figures 1-3: {missing}. Available={list(mean_plv)}")
        return {}

    fig1_mats = [mean_plv[cond] for cond in CONDITION_ORDER]
    plot_matrix_grid(fig1_mats, CONDITION_ORDER, output_dir, "fig1_mean_plv_matrix_4conditions", cmap="viridis", diverging=False, ncols=2)
    plot_connectome_grid(fig1_mats, CONDITION_ORDER, output_dir, "fig1_mean_plv_connectome_4conditions", top_k=top_k, signed=False, ncols=2)
    for cond, matrix in zip(CONDITION_ORDER, fig1_mats):
        append_top_edges(top_rows, f"fig1_mean_plv_{cond}", matrix, top_k, "PLV data-level top edge", by_abs=False)
    print("[OK] Saved Figure 1 mean PLV matrices and connectomes")

    diagnosis_diff = 0.5 * ((mean_plv["DEP_neu"] - mean_plv["HC_neu"]) + (mean_plv["DEP_pos"] - mean_plv["HC_pos"]))
    plot_matrix_grid(
        [diagnosis_diff],
        ["Diagnosis-related PLV difference: DEP - HC"],
        output_dir,
        "fig2_diagnosis_plv_difference_matrix",
        cmap="RdBu_r",
        diverging=True,
        ncols=1,
    )
    plot_connectome_grid(
        [diagnosis_diff],
        ["Diagnosis-related PLV difference: DEP - HC"],
        output_dir,
        "fig2_diagnosis_plv_difference_connectome",
        top_k=top_k,
        signed=True,
        ncols=1,
    )
    append_top_edges(top_rows, "diagnosis_diff", diagnosis_diff, top_k, "DEP>HC if positive; DEP<HC if negative", by_abs=True)
    print("[OK] Saved Figure 2 diagnosis-related PLV difference")

    delta_hc = mean_plv["HC_pos"] - mean_plv["HC_neu"]
    delta_dep = mean_plv["DEP_pos"] - mean_plv["DEP_neu"]
    delta_interaction = delta_dep - delta_hc
    fig3_titles = ["HC: pos - neu", "DEP: pos - neu", "Interaction: DEP emotion change - HC emotion change"]
    fig3_mats = [delta_hc, delta_dep, delta_interaction]
    plot_matrix_grid(fig3_mats, fig3_titles, output_dir, "fig3_emotion_plv_change_matrix", cmap="RdBu_r", diverging=True, ncols=3)
    plot_connectome_grid(fig3_mats, fig3_titles, output_dir, "fig3_emotion_plv_change_connectome", top_k=top_k, signed=True, ncols=3)
    for name, matrix in zip(["emotion_change_HC", "emotion_change_DEP", "emotion_change_interaction"], fig3_mats):
        append_top_edges(top_rows, name, matrix, top_k, "emotion PLV change", by_abs=True)
    print("[OK] Saved Figure 3 emotion-induced PLV changes")

    return {
        "diagnosis_diff": diagnosis_diff.astype(np.float32),
        "delta_emo_HC": delta_hc.astype(np.float32),
        "delta_emo_DEP": delta_dep.astype(np.float32),
        "delta_interaction": delta_interaction.astype(np.float32),
    }


def repeated_baseline(
    baseline_dict: Optional[dict[str, torch.Tensor]],
    key: str,
    repeats: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if baseline_dict is None or key not in baseline_dict:
        return None
    value = torch.as_tensor(baseline_dict[key], device=device, dtype=dtype)
    return value.unsqueeze(0).expand(repeats, *value.shape).contiguous()


def select_occlusion_records(records: list[dict], max_trials: int, seed: int) -> list[dict]:
    usable = [r for r in records if r.get("plv_adj") is not None and r.get("_x") is not None and r.get("_de_feat") is not None]
    if max_trials <= 0 or len(usable) <= max_trials:
        return usable
    rng = np.random.default_rng(seed)

    def score(record: dict) -> tuple[int, float, float]:
        correct = record.get("emotion_correct")
        correct_score = int(correct == 1) if correct is not None else 0
        confidence = float(record.get("emotion_confidence", 0.0))
        return (correct_score, confidence, float(rng.random()))

    usable.sort(key=score, reverse=True)
    return usable[:max_trials]


@torch.no_grad()
def compute_edge_occlusion_importance(
    model: Stage2ExpertEmotionAdaptationModel,
    trial_records: list[dict],
    device: torch.device,
    de_mu: Optional[dict],
    de_std: Optional[dict],
    bio_mu: Optional[dict],
    bio_std: Optional[dict],
    target: str = "true_label",
    max_trials: int = 200,
    edge_batch_size: int = 64,
    fill: str = "zero",
    seed: int = 42,
) -> dict[str, Any]:
    selected = select_occlusion_records(trial_records, max_trials=max_trials, seed=seed)
    if not selected:
        warnings.warn("No usable trial records for edge occlusion importance.")
        return {}

    n = int(selected[0]["plv_adj"].shape[0])
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sums = {
        "all": np.zeros((n, n), dtype=np.float64),
        "HC": np.zeros((n, n), dtype=np.float64),
        "DEP": np.zeros((n, n), dtype=np.float64),
    }
    counts = defaultdict(int)
    model.eval()
    edge_batch_size = max(1, int(edge_batch_size))

    for record in tqdm(selected, desc="Edge occlusion", leave=False):
        x = torch.as_tensor(record["_x"], device=device, dtype=torch.float32).unsqueeze(0)
        de_feat = torch.as_tensor(record["_de_feat"], device=device, dtype=torch.float32).unsqueeze(0)
        adj_np = np.asarray(record["plv_adj"], dtype=np.float32)
        adj = torch.as_tensor(adj_np, device=device, dtype=torch.float32).unsqueeze(0)
        key = str(record.get("target_key", record.get("subject_id")))
        rel_kwargs_1 = {
            "subject_de_mu": repeated_baseline(de_mu, key, 1, device, de_feat.dtype),
            "subject_de_std": repeated_baseline(de_std, key, 1, device, de_feat.dtype),
            "subject_bio_mu": repeated_baseline(bio_mu, key, 1, device, de_feat.dtype),
            "subject_bio_std": repeated_baseline(bio_std, key, 1, device, de_feat.dtype),
        }
        base_out = model(
            x,
            de_feat,
            lambda_subject=0.0,
            override_plv_adj=adj,
            return_features=False,
            return_plv=True,
            return_route=False,
            return_logits=True,
            **rel_kwargs_1,
        )
        base_prob = standardize_output(base_out)["emotion_prob"]
        if base_prob is None:
            continue
        if target == "positive_prob":
            target_class = 1
        elif target == "neutral_prob":
            target_class = 0
        elif target == "pred_label" or record.get("emotion_label") is None:
            target_class = int(np.argmax(base_prob[0]))
        else:
            target_class = int(record.get("emotion_label", np.argmax(base_prob[0])))
        p_original = float(base_prob[0, target_class])

        trial_importance = np.zeros((n, n), dtype=np.float64)
        fill_value = 0.0
        if fill == "mean":
            off_diag = adj_np[~np.eye(n, dtype=bool)]
            fill_value = float(np.nanmean(off_diag)) if off_diag.size else 0.0

        for start in range(0, len(edges), edge_batch_size):
            edge_chunk = edges[start: start + edge_batch_size]
            m = len(edge_chunk)
            adj_batch = np.repeat(adj_np[None, :, :], m, axis=0)
            for k, (i, j) in enumerate(edge_chunk):
                adj_batch[k, i, j] = fill_value
                adj_batch[k, j, i] = fill_value
            x_rep = x.expand(m, -1, -1).contiguous()
            de_rep = de_feat.expand(m, *de_feat.shape[1:]).contiguous()
            adj_rep = torch.as_tensor(adj_batch, device=device, dtype=torch.float32)
            rel_kwargs_m = {
                "subject_de_mu": repeated_baseline(de_mu, key, m, device, de_feat.dtype),
                "subject_de_std": repeated_baseline(de_std, key, m, device, de_feat.dtype),
                "subject_bio_mu": repeated_baseline(bio_mu, key, m, device, de_feat.dtype),
                "subject_bio_std": repeated_baseline(bio_std, key, m, device, de_feat.dtype),
            }
            masked_out = model(
                x_rep,
                de_rep,
                lambda_subject=0.0,
                override_plv_adj=adj_rep,
                return_features=False,
                return_plv=True,
                return_route=False,
                return_logits=True,
                **rel_kwargs_m,
            )
            masked_prob = standardize_output(masked_out)["emotion_prob"]
            if masked_prob is None:
                continue
            diffs = p_original - masked_prob[:, target_class]
            for diff, (i, j) in zip(diffs, edge_chunk):
                trial_importance[i, j] = trial_importance[j, i] = float(diff)

        sums["all"] += trial_importance
        counts["all"] += 1
        diag = int(record.get("diagnosis_label", -1))
        group = "HC" if diag == 1 else "DEP" if diag == 0 else None
        if group is not None:
            sums[group] += trial_importance
            counts[group] += 1

    result = {}
    for group, matrix in sums.items():
        if counts[group] > 0:
            result[group] = (matrix / counts[group]).astype(np.float32)
        else:
            result[group] = np.zeros_like(matrix, dtype=np.float32)
    result["counts"] = dict(counts)
    result["selected_records"] = len(selected)
    return result


def plot_importance_figures(
    importance: dict[str, Any],
    output_dir: Path,
    top_k: int,
    top_rows: list[dict],
) -> None:
    if not importance or "all" not in importance:
        return
    imp_all = importance["all"]
    plot_matrix_grid(
        [imp_all],
        ["Model decision-related PLV edge importance"],
        output_dir,
        "fig4_model_edge_importance_matrix_all",
        cmap="RdBu_r",
        diverging=True,
        ncols=1,
    )
    plot_connectome_grid(
        [imp_all],
        ["Model decision-related PLV edge importance"],
        output_dir,
        "fig4_model_edge_importance_connectome_all",
        top_k=top_k,
        signed=True,
        ncols=1,
    )
    append_top_edges(top_rows, "model_importance_all", imp_all, top_k, "model decision-related edge", by_abs=True)

    for group in ["HC", "DEP"]:
        matrix = importance.get(group)
        if matrix is None:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
        vmax = max(float(np.nanmax(np.abs(matrix))), 1e-6)
        im = axes[0].imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
        axes[0].set_title(f"{group}: importance matrix")
        axes[0].set_xticks(range(len(CHANNELS)))
        axes[0].set_yticks(range(len(CHANNELS)))
        axes[0].set_xticklabels(CHANNELS, rotation=90)
        axes[0].set_yticklabels(CHANNELS)
        fig.colorbar(im, ax=axes[0], shrink=0.78)
        draw_connectome(axes[1], matrix, f"{group}: top edges", top_k=top_k, signed=True)
        save_figure(fig, output_dir, f"fig4_model_edge_importance_{group}")
        append_top_edges(top_rows, f"model_importance_{group}", matrix, top_k, f"{group} important edge", by_abs=True)
    print("[OK] Saved Figure 4 model edge occlusion importance")


def mean_plv_strength(trial_records: list[dict]) -> Optional[np.ndarray]:
    values = [record.get("plv_adj") for record in trial_records if record.get("plv_adj") is not None]
    if not values:
        return None
    return np.mean(np.stack(values, axis=0), axis=0).astype(np.float32)


def rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def plot_plv_vs_importance(
    mean_plv_matrix: Optional[np.ndarray],
    importance_all: Optional[np.ndarray],
    output_dir: Path,
    top_k: int,
) -> None:
    if mean_plv_matrix is None or importance_all is None:
        return
    edges = upper_edges(importance_all, top_k=None, by_abs=False)
    x = np.asarray([mean_plv_matrix[i, j] for i, j, _ in edges], dtype=float)
    y = np.asarray([importance_all[i, j] for i, j, _ in edges], dtype=float)
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan
    rx, ry = rankdata(x), rankdata(y)
    spearman = float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 and np.std(rx) > 0 and np.std(ry) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(x, y, s=18, alpha=0.75, color="#4C78A8", edgecolor="white", linewidth=0.3)
    ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean PLV strength")
    ax.set_ylabel("Model edge importance")
    ax.set_title("PLV strength vs model importance")
    ax.text(
        0.02,
        0.98,
        f"Pearson r={pearson:.3f}\nSpearman rho={spearman:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )
    for i, j, value in upper_edges(importance_all, top_k=top_k, by_abs=True):
        label = f"{CHANNELS[i]}-{CHANNELS[j]}"
        ax.annotate(label, (mean_plv_matrix[i, j], value), fontsize=6.5, alpha=0.85)
    save_figure(fig, output_dir, "fig5_plv_strength_vs_model_importance")
    print("[OK] Saved Figure 5 PLV strength vs model importance")


def matrix_from_edges(source: np.ndarray, edges: set[tuple[int, int]]) -> np.ndarray:
    out = np.zeros_like(source, dtype=np.float32)
    for i, j in edges:
        out[i, j] = out[j, i] = source[i, j]
    return out


def plot_hc_dep_overlap(importance: dict[str, Any], output_dir: Path, top_k: int) -> None:
    imp_hc = importance.get("HC") if importance else None
    imp_dep = importance.get("DEP") if importance else None
    if imp_hc is None or imp_dep is None:
        return
    hc_edges = {(i, j) for i, j, _ in upper_edges(imp_hc, top_k=top_k, by_abs=True)}
    dep_edges = {(i, j) for i, j, _ in upper_edges(imp_dep, top_k=top_k, by_abs=True)}
    common_edges = hc_edges & dep_edges
    hc_specific = hc_edges - dep_edges
    dep_specific = dep_edges - hc_edges
    common_matrix = matrix_from_edges(0.5 * (imp_hc + imp_dep), common_edges)
    hc_matrix = matrix_from_edges(imp_hc, hc_specific)
    dep_matrix = matrix_from_edges(imp_dep, dep_specific)

    titles = ["Common important edges", "HC-specific important edges", "DEP-specific important edges"]
    matrices = [common_matrix, hc_matrix, dep_matrix]
    plot_connectome_grid(matrices, titles, output_dir, "fig6_HC_DEP_model_edge_overlap", top_k=top_k, signed=True, ncols=3)
    plot_connectome_grid([hc_matrix], ["HC-specific important edges"], output_dir, "fig6_HC_specific_model_edges", top_k=top_k, signed=True, ncols=1)
    plot_connectome_grid([dep_matrix], ["DEP-specific important edges"], output_dir, "fig6_DEP_specific_model_edges", top_k=top_k, signed=True, ncols=1)
    print("[OK] Saved Figure 6 HC/DEP model edge overlap")


def labels_for_embedding(records: list[dict], mode: str) -> tuple[list[str], list[int]]:
    if mode == "diagnosis":
        labels = [DIAGNOSIS_NAMES.get(int(r.get("diagnosis_label", -1)), "NA") for r in records]
        numeric = [int(r.get("diagnosis_label", -1)) for r in records]
    elif mode == "emotion":
        labels = [EMOTION_NAMES.get(int(r.get("emotion_label", -1)), "NA") for r in records]
        numeric = [int(r.get("emotion_label", -1)) for r in records]
    else:
        labels = [str(r.get("condition", "NA")) for r in records]
        numeric = [CONDITION_ORDER.index(label) if label in CONDITION_ORDER else -1 for label in labels]
    return labels, numeric


def plot_embedding(
    coords: np.ndarray,
    labels: list[str],
    title: str,
    output_dir: Path,
    base_name: str,
    silhouette: Optional[float],
) -> None:
    palette = {
        "HC": "#2CA02C", "DEP": "#D62728",
        "neu": "#4C78A8", "pos": "#F58518",
        "HC_neu": "#1B9E77", "HC_pos": "#66A61E", "DEP_neu": "#7570B3", "DEP_pos": "#E7298A",
    }
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    def label_sort_key(label: str) -> tuple[int, Any]:
        return (0, CONDITION_ORDER.index(label)) if label in CONDITION_ORDER else (1, str(label))

    for label in sorted(set(labels), key=label_sort_key):
        mask = np.asarray([x == label for x in labels])
        ax.scatter(coords[mask, 0], coords[mask, 1], s=28, label=label, alpha=0.82, color=palette.get(label, None), edgecolor="white", linewidth=0.3)
    score_text = "" if silhouette is None or np.isnan(silhouette) else f" | silhouette={silhouette:.3f}"
    ax.set_title(f"{title}{score_text}")
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.legend(frameon=False, fontsize=8, loc="best")
    save_figure(fig, output_dir, base_name)


def maybe_silhouette(coords: np.ndarray, numeric_labels: list[int]) -> Optional[float]:
    labels = np.asarray(numeric_labels)
    valid = labels >= 0
    labels = labels[valid]
    coords = coords[valid]
    if coords.shape[0] < 4 or len(set(labels.tolist())) < 2 or len(set(labels.tolist())) >= coords.shape[0]:
        return None
    try:
        return float(silhouette_score(coords, labels))
    except Exception as exc:
        warnings.warn(f"Could not compute silhouette score: {exc}")
        return None


def run_embeddings(trial_records: list[dict], output_dir: Path, seed: int) -> pd.DataFrame:
    embedding_specs = [
        ("z_abs", "diagnosis", "fig7_tsne_z_abs_by_diagnosis", "t-SNE z_abs by diagnosis"),
        ("z_rel", "emotion", "fig7_tsne_z_rel_by_emotion", "t-SNE z_rel by emotion"),
        ("z_fused", "label4", "fig7_tsne_z_fused_by_label4", "t-SNE z_fused by label4"),
    ]
    coord_rows: list[dict] = []
    for z_key, label_mode, base_name, title in embedding_specs:
        usable = [r for r in trial_records if r.get(z_key) is not None]
        if len(usable) < 4:
            warnings.warn(f"Skip {z_key} embedding: not enough samples.")
            continue
        X = np.stack([np.asarray(r[z_key], dtype=np.float32).reshape(-1) for r in usable], axis=0)
        perplexity = max(2, min(30, (len(usable) - 1) // 3))
        if perplexity >= len(usable):
            warnings.warn(f"Skip {z_key} t-SNE: perplexity={perplexity}, n={len(usable)}.")
            continue
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(X)
        labels, numeric = labels_for_embedding(usable, label_mode)
        score = maybe_silhouette(coords, numeric)
        plot_embedding(coords, labels, title, output_dir, base_name, score)
        for record, (dim1, dim2) in zip(usable, coords):
            coord_rows.append(
                {
                    "method": "tsne",
                    "subject_id": record.get("subject_id"),
                    "trial_id": record.get("trial_id"),
                    "diagnosis_label": record.get("diagnosis_label"),
                    "emotion_label": record.get("emotion_label"),
                    "label4": record.get("label4"),
                    "z_type": z_key,
                    "dim1": float(dim1),
                    "dim2": float(dim2),
                }
            )

    try:
        import umap  # type: ignore

        for z_key, label_mode, base_name, title in [
            ("z_abs", "diagnosis", "fig7_umap_z_abs_by_diagnosis", "UMAP z_abs by diagnosis"),
            ("z_rel", "emotion", "fig7_umap_z_rel_by_emotion", "UMAP z_rel by emotion"),
            ("z_fused", "label4", "fig7_umap_z_fused_by_label4", "UMAP z_fused by label4"),
        ]:
            usable = [r for r in trial_records if r.get(z_key) is not None]
            if len(usable) < 4:
                continue
            X = np.stack([np.asarray(r[z_key], dtype=np.float32).reshape(-1) for r in usable], axis=0)
            reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(usable) - 1), random_state=seed)
            coords = reducer.fit_transform(X)
            labels, numeric = labels_for_embedding(usable, label_mode)
            score = maybe_silhouette(coords, numeric)
            plot_embedding(coords, labels, title, output_dir, base_name, score)
            for record, (dim1, dim2) in zip(usable, coords):
                coord_rows.append(
                    {
                        "method": "umap",
                        "subject_id": record.get("subject_id"),
                        "trial_id": record.get("trial_id"),
                        "diagnosis_label": record.get("diagnosis_label"),
                        "emotion_label": record.get("emotion_label"),
                        "label4": record.get("label4"),
                        "z_type": z_key,
                        "dim1": float(dim1),
                        "dim2": float(dim2),
                    }
                )
    except Exception as exc:
        print(f"[WARN] UMAP unavailable or failed. Skip UMAP figures. Details: {exc}")

    df = pd.DataFrame(coord_rows)
    if not df.empty:
        df.to_csv(output_dir / "fig7_embedding_coordinates.csv", index=False, encoding="utf-8-sig")
        df.to_csv(output_dir / "feature_embeddings.csv", index=False, encoding="utf-8-sig")
        print("[OK] Saved Figure 7 feature-space embeddings")
    return df


def route_weight_dataframe(trial_records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in trial_records:
        route = record.get("route_weight")
        if route is None or len(route) < 2:
            continue
        route = np.asarray(route, dtype=float)
        rows.append(
            {
                "subject_id": record.get("subject_id"),
                "trial_id": record.get("trial_id"),
                "diagnosis_label": record.get("diagnosis_label"),
                "diagnosis_name": DIAGNOSIS_NAMES.get(int(record.get("diagnosis_label", -1)), "NA"),
                "emotion_label": record.get("emotion_label"),
                "emotion_pred": record.get("emotion_pred"),
                "emotion_correct": record.get("emotion_correct"),
                "emotion_confidence": record.get("emotion_confidence"),
                "w_HC_expert": float(route[0]),
                "w_DEP_expert": float(route[1]),
                "route_margin": float(abs(route[0] - route[1])),
                "route_uncertainty": float(1.0 - abs(route[0] - route[1])),
            }
        )
    return pd.DataFrame(rows)


def plot_route_weights(trial_records: list[dict], output_dir: Path) -> pd.DataFrame:
    df = route_weight_dataframe(trial_records)
    if df.empty:
        print("[WARN] No route_weight found. Skip Figure 8.")
        return df
    df.to_csv(output_dir / "route_weights.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    groups = [
        df.loc[df["diagnosis_name"] == "HC", "w_HC_expert"].dropna().to_numpy(),
        df.loc[df["diagnosis_name"] == "HC", "w_DEP_expert"].dropna().to_numpy(),
        df.loc[df["diagnosis_name"] == "DEP", "w_HC_expert"].dropna().to_numpy(),
        df.loc[df["diagnosis_name"] == "DEP", "w_DEP_expert"].dropna().to_numpy(),
    ]
    ax.boxplot(groups, labels=["HC: w_HC", "HC: w_DEP", "DEP: w_HC", "DEP: w_DEP"], patch_artist=True)
    ax.set_ylabel("Route weight")
    ax.set_title("Route weight by diagnosis")
    ax.set_ylim(-0.02, 1.02)
    save_figure(fig, output_dir, "fig8_route_weight_boxplot_by_diagnosis")

    pivot = df.pivot_table(index="subject_id", columns="trial_id", values="w_DEP_expert", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(6.0, 0.28 * pivot.shape[1]), max(4.0, 0.25 * pivot.shape[0])))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title("w_DEP_expert by subject and trial")
    ax.set_xlabel("trial_id")
    ax.set_ylabel("subject_id")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.astype(str), rotation=90)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.astype(str))
    fig.colorbar(im, ax=ax, shrink=0.78)
    save_figure(fig, output_dir, "fig8_route_weight_heatmap_by_subject_trial")

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    if df["emotion_correct"].notna().any():
        incorrect = df.loc[df["emotion_correct"] == 0, "route_margin"].dropna().to_numpy()
        correct = df.loc[df["emotion_correct"] == 1, "route_margin"].dropna().to_numpy()
        ax.boxplot([incorrect, correct], labels=["incorrect", "correct"], patch_artist=True)
        ax.set_ylabel("Route margin")
        ax.set_title("Route margin vs emotion correctness")
    else:
        ax.scatter(df["route_margin"], df["emotion_confidence"], s=24, alpha=0.75)
        ax.set_xlabel("Route margin")
        ax.set_ylabel("Emotion confidence")
        ax.set_title("Route margin vs emotion confidence")
    save_figure(fig, output_dir, "fig8_route_weight_vs_correctness")
    print("[OK] Saved Figure 8 route weight distributions")
    return df


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    setup_plot_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    ckpt = load_checkpoint(args.ckpt, device)
    domain_mapping, train_subjects, val_subjects = build_domain_mapping_if_needed(args, ckpt)
    model = build_stage2_model(ckpt, domain_mapping, device, strict_load=args.strict_load)
    print("[OK] Loaded model checkpoint")

    dataset, loader = build_dataset_and_loader(args, domain_mapping, train_subjects, val_subjects)
    relative_eps = float((ckpt.get("config", {}) or {}).get("relative_eps", 1e-6))
    de_mu, de_std, bio_mu, bio_std = compute_optional_baselines(model, dataset, loader, device, relative_eps)

    window_records = collect_window_records(model, loader, device, de_mu, de_std, bio_mu, bio_std)
    save_pickle(output_dir / "records_window_level.pkl", strip_private(window_records))
    print(f"[OK] Collected window-level records: {len(window_records)}")

    trial_records = window_to_trial_aggregate(window_records) if int(args.use_trial_level) else window_records
    save_pickle(output_dir / "records_trial_level.pkl", strip_private(trial_records))
    print(f"[OK] Aggregated trial-level records: {len(trial_records)}")

    top_rows: list[dict] = []
    mean_plv = mean_plv_by_condition(trial_records)
    if mean_plv:
        np.savez(output_dir / "mean_plv_by_condition.npz", **mean_plv)
    diff_mats = make_figures_1_to_3(mean_plv, output_dir, args.top_k_edges, top_rows)
    if diff_mats:
        np.savez(output_dir / "plv_difference_matrices.npz", **diff_mats)

    importance = {}
    if args.skip_occlusion:
        print("[WARN] Skip edge occlusion by --skip_occlusion.")
    else:
        importance = compute_edge_occlusion_importance(
            model,
            trial_records,
            device,
            de_mu=de_mu,
            de_std=de_std,
            bio_mu=bio_mu,
            bio_std=bio_std,
            target=args.occlusion_target,
            max_trials=args.max_occlusion_trials,
            edge_batch_size=args.occlusion_edge_batch_size,
            fill=args.occlusion_fill,
            seed=args.seed,
        )
        if importance:
            np.savez(
                output_dir / "edge_occlusion_importance.npz",
                importance_all=importance.get("all"),
                importance_HC=importance.get("HC"),
                importance_DEP=importance.get("DEP"),
                counts=np.asarray([importance.get("counts", {}).get(k, 0) for k in ["all", "HC", "DEP"]], dtype=np.int64),
            )
            plot_importance_figures(importance, output_dir, args.top_k_edges, top_rows)
            plot_plv_vs_importance(mean_plv_strength(trial_records), importance.get("all"), output_dir, args.top_k_edges)
            plot_hc_dep_overlap(importance, output_dir, args.top_k_edges)

    run_embeddings(trial_records, output_dir, args.seed)
    route_df = plot_route_weights(trial_records, output_dir)
    if route_df.empty and not (output_dir / "route_weights.csv").exists():
        pd.DataFrame().to_csv(output_dir / "route_weights.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(top_rows).to_csv(output_dir / "top_edges_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] All interpretability figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

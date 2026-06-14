# -*- coding: utf-8 -*-
"""Ensemble V6 checkpoints on the unlabeled test set.

This script is for the common case where training has already produced V6
checkpoint files, but test predictions were not generated during training.

Examples:
    python V6/ensemble_predict.py --model_dir model_params/V6_NoSSAS_DualFeature_SoftRouter

    python V6/ensemble_predict.py --model_paths path/to/fold0/v6_best.pt path/to/fold1/v6_best.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Optional

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dual_feature_soft_router_model import DualFeatureSoftRouterModel
from train_dual_feature_soft_router import (
    UnlabeledTargetDataset,
    apply_subject_topk,
    compute_subject_bio_baselines,
    compute_subject_de_baselines,
    dict_collate,
    get_subject_relative_kwargs,
    move_batch_to_device,
    natural_key,
)


def to_jsonable(obj: Any):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def save_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    path = Path(path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def split_path_values(values: Optional[list[str]]) -> list[Path]:
    if not values:
        return []
    paths: list[Path] = []
    for value in values:
        for part in str(value).replace(";", ",").split(","):
            text = part.strip().strip('"').strip("'")
            if text:
                paths.append(Path(text))
    return paths


def discover_model_paths(
    model_dir: str | Path,
    pattern: str = "v6_best.pt",
    recursive: bool = True,
) -> list[Path]:
    root = Path(model_dir)
    if not root.exists():
        return []
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted({p.resolve() for p in iterator if p.is_file()}, key=lambda p: str(p))


def get_test_users(test_csv: str | Path) -> list[str]:
    df = pd.read_csv(test_csv)
    if df.empty:
        return []
    id_col = "user_id" if "user_id" in df.columns else "subject_id"
    return sorted(df[id_col].astype(str).unique().tolist(), key=natural_key)


def ensure_test_domain_mapping(domain_mapping: Optional[dict], test_csv: str | Path) -> dict:
    """Ensure test users exist in domain_mapping for UnlabeledTargetDataset."""

    mapping = copy.deepcopy(domain_mapping) if domain_mapping else {}
    key_to_domain = dict(mapping.get("key_to_domain", {}))
    test_users = list(mapping.get("test_users", []))
    test_user_set = set(str(u) for u in test_users)

    next_id = max([int(v) for v in key_to_domain.values()], default=-1) + 1
    for uid in get_test_users(test_csv):
        uid = str(uid)
        key = f"test:{uid}"
        if key not in key_to_domain:
            key_to_domain[key] = next_id
            next_id += 1
        if uid not in test_user_set:
            test_users.append(uid)
            test_user_set.add(uid)

    mapping["key_to_domain"] = key_to_domain
    mapping["domain_to_key"] = {str(v): k for k, v in key_to_domain.items()}
    mapping["test_users"] = sorted([str(u) for u in test_users], key=natural_key)
    mapping["test_user_to_domain"] = {
        str(u): int(key_to_domain[f"test:{u}"])
        for u in mapping["test_users"]
        if f"test:{u}" in key_to_domain
    }
    mapping["num_domains"] = len(key_to_domain)
    mapping.setdefault("source_subjects", [])
    mapping.setdefault("val_subjects", [])
    mapping.setdefault("source_subject_to_domain", {})
    mapping.setdefault("val_subject_to_domain", {})
    mapping.setdefault("source_domain_indices", [])
    return mapping


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> tuple[DualFeatureSoftRouterModel, dict, dict]:
    config_dict = ckpt.get("config", {}) or {}
    extra_state = ckpt.get("extra_state", {}) or {}
    domain_mapping = extra_state.get("domain_mapping") or ckpt.get("domain_mapping") or {}

    use_biomarkers = not bool(config_dict.get("no_biomarkers", False))
    use_subject_relative_de = not bool(config_dict.get("no_subject_relative_de", False))
    use_subject_relative_bio = (not bool(config_dict.get("no_subject_relative_bio", False))) and use_biomarkers

    model = DualFeatureSoftRouterModel(
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, domain_mapping, config_dict


@torch.no_grad()
def predict_one_checkpoint(
    ckpt_path: str | Path,
    test_csv: str | Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    threshold: float,
    normalize: bool,
    model_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ckpt_path = Path(ckpt_path)
    ckpt = load_checkpoint(ckpt_path, device)
    model, domain_mapping, config_dict = build_model_from_checkpoint(ckpt, device)
    domain_mapping = ensure_test_domain_mapping(domain_mapping, test_csv)
    epoch_value = ckpt.get("epoch", -1)
    if epoch_value is None:
        epoch_value = -1

    dataset = UnlabeledTargetDataset(test_csv, domain_mapping=domain_mapping, root=ROOT, normalize=normalize)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dict_collate,
    )

    test_de_mu = test_de_std = None
    test_bio_mu = test_bio_std = None
    if getattr(model, "use_subject_relative_de", False):
        test_de_mu, test_de_std = compute_subject_de_baselines(dataset, key_field="target_key")
    if getattr(model, "use_subject_relative_bio", False):
        test_bio_mu, test_bio_std = compute_subject_bio_baselines(
            model,
            loader,
            device,
            subject_de_mu=test_de_mu,
            subject_de_std=test_de_std,
            key_field="target_key",
        )

    rows = []
    for batch in tqdm(loader, desc=f"Predict model {model_index}", leave=False):
        batch = move_batch_to_device(batch, device)
        rel_kwargs = get_subject_relative_kwargs(
            batch,
            device,
            dtype=batch["de_feat"].dtype,
            de_mu=test_de_mu,
            de_std=test_de_std,
            bio_mu=test_bio_mu,
            bio_std=test_bio_std,
        )
        out = model(batch["x"], batch["de_feat"], **rel_kwargs)
        mix_prob = out["mix_prob"].clamp_min(1e-8)
        prob_pos_tensor = mix_prob[:, 1]
        score_pos_tensor = torch.log(mix_prob[:, 1]) - torch.log(mix_prob[:, 0])
        prob_pos = prob_pos_tensor.detach().cpu().tolist()
        score_pos = score_pos_tensor.detach().cpu().tolist()
        pred_window = [int(p >= threshold) for p in prob_pos]

        for i, p in enumerate(prob_pos):
            rows.append(
                {
                    "model_index": int(model_index),
                    "checkpoint_path": str(ckpt_path),
                    "best_name": str(ckpt.get("best_name", "")),
                    "epoch": int(epoch_value),
                    "user_id": batch["user_id"][i],
                    "trial_id": int(batch["trial_id"][i].detach().cpu()),
                    "prob_window": float(p),
                    "score_window": float(score_pos[i]),
                    "pred_window": int(pred_window[i]),
                }
            )

    window_df = pd.DataFrame(rows)
    trial_df = (
        window_df.groupby(["model_index", "checkpoint_path", "best_name", "epoch", "user_id", "trial_id"], as_index=False)
        .agg(
            prob_pos=("prob_window", "mean"),
            score_pos=("score_window", "mean"),
            hard_mean=("pred_window", "mean"),
            n_windows=("prob_window", "count"),
        )
        .sort_values(["model_index", "user_id", "trial_id"])
    )

    metadata = {
        "model_index": int(model_index),
        "checkpoint_path": str(ckpt_path),
        "best_name": ckpt.get("best_name", ""),
        "epoch": int(epoch_value),
        "criteria_readable": ckpt.get("criteria_readable", ""),
        "version": (ckpt.get("extra_state", {}) or {}).get("version", config_dict.get("version", "")),
        "config": config_dict,
    }
    return window_df, trial_df, metadata


def make_ensemble_outputs(
    member_window_df: pd.DataFrame,
    member_trial_df: pd.DataFrame,
    save_dir: str | Path,
    threshold: float,
    vote_method: str,
    k_pos: int,
) -> pd.DataFrame:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    member_window_df.to_csv(save_dir / "ensemble_member_window_probs.csv", index=False, encoding="utf-8-sig")
    member_trial_df.to_csv(save_dir / "ensemble_member_trial_probs.csv", index=False, encoding="utf-8-sig")

    trial_df = (
        member_trial_df.groupby(["user_id", "trial_id"], as_index=False)
        .agg(
            prob_pos=("prob_pos", "mean"),
            score_pos=("score_pos", "mean"),
            hard_mean=("hard_mean", "mean"),
            model_count=("prob_pos", "count"),
            n_windows_mean=("n_windows", "mean"),
        )
        .sort_values(["user_id", "trial_id"])
    )
    trial_df["trial_prob"] = trial_df["prob_pos"]
    trial_df["pred_soft_threshold"] = (trial_df["prob_pos"] >= threshold).astype(int)
    trial_df["pred_hard_threshold"] = (trial_df["hard_mean"] >= 0.5).astype(int)

    topk_records = apply_subject_topk(
        trial_df[["user_id", "trial_id", "prob_pos", "score_pos"]].to_dict("records"),
        k_pos=k_pos,
    )
    topk_df = pd.DataFrame(topk_records)[["user_id", "trial_id", "Emotion_label"]].rename(
        columns={"Emotion_label": "pred_soft_topk"}
    )
    trial_df = trial_df.merge(topk_df, on=["user_id", "trial_id"], how="left")
    trial_df["pred_soft_topk"] = trial_df["pred_soft_topk"].fillna(0).astype(int)

    if vote_method == "hard":
        trial_df["Emotion_label"] = trial_df["pred_hard_threshold"]
    elif vote_method in {"soft_topk", "topk"}:
        trial_df["Emotion_label"] = trial_df["pred_soft_topk"]
    elif vote_method in {"prob", "soft_threshold"}:
        trial_df["Emotion_label"] = trial_df["pred_soft_threshold"]
    else:
        raise ValueError(f"Unknown vote_method={vote_method}")

    trial_df.to_csv(save_dir / "ensemble_trial_probs.csv", index=False, encoding="utf-8-sig")

    sub_soft_threshold = trial_df[["user_id", "trial_id", "pred_soft_threshold"]].copy()
    sub_soft_threshold.rename(columns={"pred_soft_threshold": "Emotion_label"}, inplace=True)
    sub_soft_threshold.to_csv(save_dir / "submission_v6_ensemble_soft_threshold.csv", index=False, encoding="utf-8-sig")

    sub_soft_topk = trial_df[["user_id", "trial_id", "pred_soft_topk"]].copy()
    sub_soft_topk.rename(columns={"pred_soft_topk": "Emotion_label"}, inplace=True)
    sub_soft_topk.to_csv(save_dir / "submission_v6_ensemble_soft_topk.csv", index=False, encoding="utf-8-sig")

    sub_hard_threshold = trial_df[["user_id", "trial_id", "pred_hard_threshold"]].copy()
    sub_hard_threshold.rename(columns={"pred_hard_threshold": "Emotion_label"}, inplace=True)
    sub_hard_threshold.to_csv(save_dir / "submission_v6_ensemble_hard_threshold.csv", index=False, encoding="utf-8-sig")

    submission = trial_df[["user_id", "trial_id", "Emotion_label"]].copy()
    submission.to_csv(save_dir / "submission_v6_ensemble.csv", index=False, encoding="utf-8-sig")
    return trial_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble V6 checkpoints for test prediction.")
    parser.add_argument("--model_dir", type=str, default="model_params/V6_NoSSAS_DualFeature_SoftRouter")
    parser.add_argument("--pattern", type=str, default="v6_best.pt")
    parser.add_argument("--no_recursive", action="store_true")
    parser.add_argument("--model_paths", nargs="*", default=None)
    parser.add_argument("--test_csv", type=str, default="com_test_trial_index_2s.csv")
    parser.add_argument("--save_dir", type=str, default="model_params/V6_NoSSAS_DualFeature_SoftRouter/v6_checkpoint_ensemble")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--k_pos", type=int, default=4)
    parser.add_argument(
        "--vote_method",
        choices=["prob", "soft_threshold", "hard", "soft_topk", "topk"],
        default="soft_topk",
    )
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--limit_models", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_device = torch.device(args.device)
    device = requested_device if requested_device.type == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    if requested_device.type == "cuda" and device.type == "cpu":
        print(f"[V6 ensemble] requested {args.device} but CUDA is unavailable; using CPU.")

    explicit_paths = split_path_values(args.model_paths)
    if explicit_paths:
        model_paths = [p.resolve() for p in explicit_paths]
    else:
        model_paths = discover_model_paths(args.model_dir, pattern=args.pattern, recursive=not args.no_recursive)

    model_paths = [p for p in model_paths if p.exists() and p.is_file()]
    if args.limit_models and args.limit_models > 0:
        model_paths = model_paths[: int(args.limit_models)]
    if not model_paths:
        raise FileNotFoundError(
            "No V6 checkpoints found. Use --model_dir/--pattern or pass paths with --model_paths."
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[V6 ensemble] checkpoints={len(model_paths)}")
    for i, path in enumerate(model_paths, start=1):
        print(f"[V6 ensemble] {i:02d}: {path}")

    window_frames = []
    trial_frames = []
    manifest = []
    for model_index, ckpt_path in enumerate(model_paths, start=1):
        window_df, trial_df, metadata = predict_one_checkpoint(
            ckpt_path,
            test_csv=args.test_csv,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold=args.threshold,
            normalize=not args.no_normalize,
            model_index=model_index,
        )
        window_frames.append(window_df)
        trial_frames.append(trial_df)
        manifest.append(metadata)
        print(
            f"[V6 ensemble] model {model_index}/{len(model_paths)} done: "
            f"windows={len(window_df)}, trials={len(trial_df)}, checkpoint={ckpt_path}"
        )

    member_window_df = pd.concat(window_frames, ignore_index=True)
    member_trial_df = pd.concat(trial_frames, ignore_index=True)
    trial_df = make_ensemble_outputs(
        member_window_df,
        member_trial_df,
        save_dir=save_dir,
        threshold=args.threshold,
        vote_method=args.vote_method,
        k_pos=args.k_pos,
    )
    save_json(
        save_dir / "ensemble_manifest.json",
        {
            "test_csv": args.test_csv,
            "model_count": len(model_paths),
            "vote_method": args.vote_method,
            "threshold": args.threshold,
            "k_pos": args.k_pos,
            "checkpoints": manifest,
            "outputs": {
                "member_window_probs": str(save_dir / "ensemble_member_window_probs.csv"),
                "member_trial_probs": str(save_dir / "ensemble_member_trial_probs.csv"),
                "ensemble_trial_probs": str(save_dir / "ensemble_trial_probs.csv"),
                "submission": str(save_dir / "submission_v6_ensemble.csv"),
            },
        },
    )
    n_selected = int(trial_df["Emotion_label"].sum()) if not trial_df.empty else 0
    print(
        f"[V6 ensemble] saved to {save_dir}; models={len(model_paths)}, "
        f"trials={len(trial_df)}, selected={args.vote_method}: {n_selected}/{len(trial_df)} positive"
    )
    print(f"[V6 ensemble] final submission: {save_dir / 'submission_v6_ensemble.csv'}")


if __name__ == "__main__":
    main()

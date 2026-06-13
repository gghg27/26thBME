# -*- coding: utf-8 -*-
"""Standalone V5 ensemble test prediction — no training, just inference.

Scans a V5_expert_ssas save_root directory for best checkpoints across all
folds and repeats, then runs ensemble_test_predictions on the collected models.

Usage:
    # Default: use all topk_trial_f1 best checkpoints from V5_expert_ssas
    python V5/ensemble_predict.py

    # Custom save root and best_name
    python V5/ensemble_predict.py --save_root model_params/V5_expert_ssas --predict_best_name combined

    # All options
    python V5/ensemble_predict.py --save_root model_params/V5_expert_ssas \\
        --test_csv com_test_trial_index_2s.csv \\
        --predict_best_name topk_trial_f1 \\
        --test_vote_method soft_topk --k_pos 4 \\
        --trial_pooling attn_conf --trial_attn_tau 1.0 \\
        --output_dir test_ensemble
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

# ---------------------------------------------------------------------------
# Reuse everything from the V5 training module — no copy-paste
# ---------------------------------------------------------------------------
from V5.train_expert_ssas_emotion import (  # type: ignore[import-not-resolved]
    ensemble_test_predictions,
    load_checkpoint,
    print_checkpoint_info,
    build_stage2_model_from_checkpoint,
    natural_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_best_checkpoints(
    save_root: str | Path,
    predict_best_name: str = "topk_trial_f1",
    fallback_to_combined: bool = True,
    fallback_to_final: bool = True,
) -> list[Path]:
    """Scan *save_root*/expert_ssas_repeat*_fold*/ for best checkpoints.

    Priority order for each fold/repeat:
        1. stage2_best_{best_name}_fold{N}.pt
        2. stage2_best.pt  (combined, if fallback_to_combined)
        3. stage2_final.pt (if fallback_to_final)
    """
    root = Path(save_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Save root does not exist: {root}")

    # Collect all fold/repeat directories
    fold_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and "expert_ssas_repeat" in d.name],
        key=lambda d: natural_key(d.name),
    )
    if not fold_dirs:
        raise FileNotFoundError(
            f"No expert_ssas_repeat*_fold* subdirectories found under {root}"
        )

    checkpoint_paths: list[Path] = []
    missing: list[str] = []

    for fold_dir in fold_dirs:
        found: Path | None = None

        # 1) Named best checkpoint — try exact filename first
        if predict_best_name == "combined":
            candidates = [fold_dir / "stage2_best.pt"]
        else:
            fold_num = _extract_fold_number(fold_dir.name)
            candidates = []
            if fold_num is not None:
                candidates.append(
                    fold_dir / f"stage2_best_{predict_best_name}_fold{fold_num}.pt"
                )
            # also try without fold suffix (in case of unusual naming)
            candidates.append(fold_dir / f"stage2_best_{predict_best_name}.pt")
            # glob fallback: any file starting with this best_name pattern
            candidates.extend(
                sorted(
                    fold_dir.glob(f"stage2_best_{predict_best_name}*.pt"),
                    key=lambda p: natural_key(p.name),
                )
            )

        for candidate in candidates:
            if candidate.exists():
                found = candidate
                break

        # 2) Fallback to combined
        if found is None and fallback_to_combined and predict_best_name != "combined":
            combined = fold_dir / "stage2_best.pt"
            if combined.exists():
                print(f"[scan] {fold_dir.name}: '{predict_best_name}' not found, "
                      f"falling back to combined")
                found = combined

        # 3) Fallback to final
        if found is None and fallback_to_final:
            final = fold_dir / "stage2_final.pt"
            if final.exists():
                print(f"[scan] {fold_dir.name}: no best checkpoint found, "
                      f"falling back to final")
                found = final

        if found is not None:
            checkpoint_paths.append(found)
        else:
            missing.append(fold_dir.name)

    if missing:
        print(f"\n[scan] WARNING: {len(missing)} fold dir(s) have no usable checkpoint:")
        for m in missing:
            print(f"        - {m}")

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints found in {root}.  "
            f"Searched for best_name='{predict_best_name}' across "
            f"{len(fold_dirs)} fold directories."
        )

    print(f"\n[scan] Found {len(checkpoint_paths)} checkpoint(s) across "
          f"{len(fold_dirs)} fold/repeat dirs:")
    for p in checkpoint_paths:
        print(f"        {p}")
    return checkpoint_paths


def _extract_fold_number(dirname: str) -> int | None:
    """'expert_ssas_repeat0_fold3' → 3, or None on failure."""
    for part in dirname.split("_"):
        if part.startswith("fold") and part[4:].isdigit():
            return int(part[4:])
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V5 standalone ensemble test prediction (no training)",
    )
    parser.add_argument(
        "--save_root", type=str, default="model_params/V5_expert_ssas",
        help="Root directory containing expert_ssas_repeat*_fold*/ subdirectories",
    )
    parser.add_argument(
        "--test_csv", type=str, default="com_test_trial_index_2s.csv",
        help="Test trial index CSV",
    )
    parser.add_argument(
        "--predict_best_name", type=str, default="topk_trial_f1",
        help="Which best checkpoint to use per fold "
             "(topk_trial_f1 | trial_f1 | combined | loss | ...)",
    )
    parser.add_argument(
        "--test_vote_method", type=str,
        choices=["prob", "soft_threshold", "hard", "soft_topk", "topk"],
        default="soft_topk",
        help="Final test submission voting method",
    )
    parser.add_argument("--k_pos", type=int, default=4, help="Top-K positive trials per subject")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    parser.add_argument(
        "--trial_pooling", type=str, choices=["mean", "attn_conf"], default="attn_conf",
        help="Window→trial aggregation method",
    )
    parser.add_argument(
        "--trial_attn_tau", type=float, default=1.0,
        help="Temperature for attn_conf pooling",
    )
    parser.add_argument("--batch_size", type=int, default=200, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--no_normalize", action="store_true",
        help="Disable per-window z-score normalization",
    )
    parser.add_argument(
        "--no_fallback", action="store_true",
        help="Disable fallback to combined/final checkpoints",
    )
    parser.add_argument(
        "--output_dir", type=str, default="test_ensemble",
        help="Output subdirectory name inside save_root",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (cuda / cpu)",
    )
    parser.add_argument(
        "--checkpoint_paths", type=str, nargs="*", default=None,
        help="Explicit list of checkpoint paths (bypasses auto-scan)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    save_root = Path(args.save_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[V5 ensemble] device={device}  save_root={save_root}")

    # Resolve checkpoint list
    if args.checkpoint_paths:
        model_paths = [Path(p) for p in args.checkpoint_paths if Path(p).exists()]
        if not model_paths:
            raise FileNotFoundError("No valid paths in --checkpoint_paths")
        print(f"[V5 ensemble] Using {len(model_paths)} explicit checkpoint(s)")
    else:
        model_paths = find_best_checkpoints(
            save_root=save_root,
            predict_best_name=args.predict_best_name,
            fallback_to_combined=not args.no_fallback,
            fallback_to_final=not args.no_fallback,
        )

    # Quick header info for each checkpoint
    print("\n" + "=" * 72)
    print(f"[V5 ensemble] {len(model_paths)} models → {args.test_vote_method} k={args.k_pos}")
    print("=" * 72)
    for i, p in enumerate(model_paths):
        try:
            ckpt = load_checkpoint(p, device)
            print_checkpoint_info(p, ckpt, prefix=f"[ckpt {i + 1}/{len(model_paths)}]")
        except Exception as exc:
            print(f"[ckpt {i + 1}/{len(model_paths)}] {p} — skip (load error: {exc})")

    # Run ensemble
    output_dir = save_root / args.output_dir
    print(f"\n[V5 ensemble] output → {output_dir}")
    ensemble_test_predictions(
        model_paths=[str(p) for p in model_paths],
        test_csv=args.test_csv,
        device=device,
        save_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        vote_method=args.test_vote_method,
        k_pos=args.k_pos,
        normalize=not args.no_normalize,
        trial_pooling=args.trial_pooling,
        trial_attn_tau=args.trial_attn_tau,
    )

    print(f"\n[done] Ensemble results saved to {output_dir}")
    print(f"  submission_test_ensemble.csv     — main submission")
    print(f"  submission_soft_topk_ensemble.csv — soft top-K version")
    print(f"  test_ensemble_probs.csv          — full per-trial probabilities")


if __name__ == "__main__":
    main()

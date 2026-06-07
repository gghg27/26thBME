from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.folds import require_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble fold-level test predictions.")
    parser.add_argument("--predictions_dir", type=Path, default=ROOT / "predictions")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "predictions" / "ensemble")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.predictions_dir.glob("reapt*_fold*/test_two_stage_preds.csv"))
    files += sorted(args.predictions_dir.glob("repeat*_fold*/test_two_stage_preds.csv"))
    files += sorted(args.predictions_dir.glob("repeat*/test_two_stage_preds.csv"))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No test_two_stage_preds.csv files found under {args.predictions_dir}"
        )

    frames = []
    for path in files:
        df = pd.read_csv(path)
        require_columns(df, ["user_id", "trial_id", "p_final"], path)
        df = df[["user_id", "trial_id", "p_final"]].copy()
        df["source_file"] = path.as_posix()
        frames.append(df)
        print(f"[ensemble] loaded {path}")

    all_preds = pd.concat(frames, ignore_index=True)
    probs = (
        all_preds.groupby(["user_id", "trial_id"], as_index=False)
        .agg(p_final_ensemble=("p_final", "mean"), n_models=("p_final", "count"))
        .sort_values(["user_id", "trial_id"])
    )
    probs["Emotion_label"] = (probs["p_final_ensemble"] >= args.threshold).astype(int)
    submission = probs[["user_id", "trial_id", "Emotion_label"]].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    probs_path = args.output_dir / "test_10fold_probs.csv"
    xlsx_path = args.output_dir / "submission.xlsx"
    probs.to_csv(probs_path, index=False)
    try:
        submission.to_excel(xlsx_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Writing submission.xlsx requires openpyxl or xlsxwriter. "
            f"CSV probabilities were saved to {probs_path}."
        ) from exc
    print(f"[ensemble] wrote {probs_path}")
    print(f"[ensemble] wrote {xlsx_path}")


if __name__ == "__main__":
    main()

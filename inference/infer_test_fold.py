from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.checkpoint import load_meta
from utils.data import expand_window_index
from utils.folds import combined_model_param_dir, reapt_name, require_columns
from utils.predict import (
    aggregate_subject_prob,
    aggregate_trial_prob,
    load_model_for_inference,
    predict_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage test inference.")
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--combined_meta", type=Path, default=None)
    parser.add_argument("--diag_csv", type=Path, default=ROOT / "com_test_trial_index_10s.csv")
    parser.add_argument("--emotion_csv", type=Path, default=ROOT / "com_test_trial_index_2s.csv")
    parser.add_argument("--predictions_dir", type=Path, default=ROOT / "predictions")
    parser.add_argument("--model_params_dir", type=Path, default=ROOT / "model_params")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_combined(args: argparse.Namespace) -> dict:
    path = args.combined_meta or (
        combined_model_param_dir(args.repeat, args.fold, args.model_params_dir)
        / "combined_meta.json"
    )
    legacy_path = (
        args.model_params_dir
        / f"repeat{args.repeat}_fold{args.fold}"
        / "combined_meta.json"
    )
    if args.combined_meta is None and not path.exists() and legacy_path.exists():
        path = legacy_path
    if not path.exists():
        raise FileNotFoundError(f"combined_meta not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("repeat", -1)) != args.repeat or int(data.get("fold", -1)) != args.fold:
        raise ValueError(f"combined_meta repeat/fold mismatch: {path}")
    return data


def detect_id_col(df: pd.DataFrame) -> str:
    if "user_id" in df.columns:
        return "user_id"
    if "subject_id" in df.columns:
        return "subject_id"
    raise ValueError("Test CSV must contain either 'user_id' or 'subject_id'.")


def main() -> None:
    args = parse_args()

    import torch

    if not args.diag_csv.exists():
        raise FileNotFoundError(f"Test diagnosis index CSV not found: {args.diag_csv}")
    if not args.emotion_csv.exists():
        raise FileNotFoundError(f"Test emotion index CSV not found: {args.emotion_csv}")

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    combined = load_combined(args)
    metas = {
        task: load_meta(resolve_path(entry["meta"]))
        for task, entry in combined["models"].items()
    }
    models = {
        task: load_model_for_inference(
            metas[task], resolve_path(combined["models"][task]["path"]), device=device
        )
        for task in ("diagnosis", "hc_emotion", "dep_emotion")
    }

    diag_df = pd.read_csv(args.diag_csv)
    emotion_df = pd.read_csv(args.emotion_csv)
    diag_id_col = detect_id_col(diag_df)
    emotion_id_col = detect_id_col(emotion_df)
    require_columns(diag_df, [diag_id_col, "de_path"], args.diag_csv)
    require_columns(emotion_df, [emotion_id_col, "trial_id", "de_path"], args.emotion_csv)
    diag_df = expand_window_index(diag_df, root=ROOT)
    emotion_df = expand_window_index(emotion_df, root=ROOT)
    print(
        f"[infer-test] diagnosis windows={len(diag_df)} "
        f"emotion windows={len(emotion_df)}"
    )

    diag_df["p_dep"] = predict_windows(
        model=models["diagnosis"],
        df=diag_df,
        root=ROOT,
        device=device,
        batch_size=args.batch_size,
        logit_key="diagnosis_logits",
    )
    subject_probs = aggregate_subject_prob(diag_df, "p_dep", diag_id_col).rename(
        columns={diag_id_col: "user_id"}
    )

    emotion_df["p_pos_hc_window"] = predict_windows(
        model=models["hc_emotion"],
        df=emotion_df,
        root=ROOT,
        device=device,
        batch_size=args.batch_size,
        logit_key="emo_logits",
    )
    emotion_df["p_pos_dep_window"] = predict_windows(
        model=models["dep_emotion"],
        df=emotion_df,
        root=ROOT,
        device=device,
        batch_size=args.batch_size,
        logit_key="emo_logits",
    )
    p_hc = aggregate_trial_prob(
        emotion_df, "p_pos_hc_window", emotion_id_col, "p_pos_hc"
    ).rename(columns={emotion_id_col: "user_id"})
    p_dep = aggregate_trial_prob(
        emotion_df, "p_pos_dep_window", emotion_id_col, "p_pos_dep"
    ).rename(columns={emotion_id_col: "user_id"})
    preds = (
        p_hc.merge(p_dep, on=["user_id", "trial_id"], how="inner")
        .merge(subject_probs, on="user_id", how="left")
    )
    if preds["p_dep_subject"].isna().any():
        missing = preds.loc[preds["p_dep_subject"].isna(), "user_id"].unique()[:10]
        raise ValueError(f"Missing diagnosis probability for test users: {missing}")

    preds["p_final"] = (
        (1.0 - preds["p_dep_subject"]) * preds["p_pos_hc"]
        + preds["p_dep_subject"] * preds["p_pos_dep"]
    )
    preds["pred_emotion"] = (preds["p_final"] >= 0.5).astype(int)
    preds.insert(0, "fold", args.fold)
    preds.insert(0, "repeat", args.repeat)
    preds = preds[
        [
            "repeat",
            "fold",
            "user_id",
            "trial_id",
            "p_dep_subject",
            "p_pos_hc",
            "p_pos_dep",
            "p_final",
            "pred_emotion",
        ]
    ]

    pred_dir = args.predictions_dir / reapt_name(args.repeat, args.fold)
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / "test_two_stage_preds.csv"
    preds.to_csv(out_path, index=False)
    print(f"[infer-test] wrote {out_path}")


if __name__ == "__main__":
    main()

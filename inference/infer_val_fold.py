from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config
from utils.checkpoint import load_meta
from utils.folds import (
    combined_model_param_dir,
    reapt_name,
    require_columns,
    save_json,
)
from utils.metrics import accuracy, macro_f1
from utils.predict import (
    aggregate_subject_prob,
    aggregate_trial_prob,
    load_model_for_inference,
    predict_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage validation inference.")
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for StratifiedGroupKFold. "
        "Defaults to matching train_diag.py: [20, 42][repeat].",
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--combined_meta", type=Path, default=None)
    parser.add_argument("--diag_csv", type=Path, default=ROOT / "com_index_sub_10s.csv")
    parser.add_argument("--emotion_csv", type=Path, default=ROOT / "com_index_sub_2s.csv")
    parser.add_argument("--predictions_dir", type=Path, default=ROOT / "predictions")
    parser.add_argument("--results_dir", type=Path, default=ROOT / "results")
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
    if data.get("status") != "ready":
        raise ValueError(f"combined_meta status is not ready: {path}")
    return data


def get_val_subjects(
    index_csv: Path,
    fold: int,
    n_splits: int,
    seed: int,
) -> list[str]:
    """
    使用 sklearn StratifiedGroupKFold 即时划分验证集被试。

    与 train_diag.py 的 get_competition_subject_split() 完全一致：
    - 按 diagnosis_label 分层
    - 按 subject_id 分组（同一被试不会跨训练/验证集）
    """
    df = pd.read_csv(index_csv)
    subject_df = (
        df[["subject_id", "diagnosis_label"]]
        .drop_duplicates("subject_id")
        .reset_index(drop=True)
    )

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    splits = list(
        sgkf.split(
            X=subject_df["subject_id"],
            y=subject_df["diagnosis_label"],
            groups=subject_df["subject_id"],
        )
    )

    _, val_idx = splits[fold]
    val_subjects = subject_df.iloc[val_idx]["subject_id"].tolist()
    return [str(s) for s in val_subjects]


def filter_val_subjects(df: pd.DataFrame, val_subjects: list[str]) -> pd.DataFrame:
    """只保留属于验证集被试的行。"""
    val_set = set(val_subjects)
    result = df[df["subject_id"].astype(str).isin(val_set)].copy()
    return result.reset_index(drop=True)


def main() -> None:
    args = parse_args()

    import torch

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    # ── 确定 seed：与 train_diag.py 的 enumerate([20, 42]) 对齐 ────
    if args.seed is not None:
        seed = args.seed
    elif args.repeat < len(config.DIAG_REPEAT_SEEDS):
        seed = config.DIAG_REPEAT_SEEDS[args.repeat]
    else:
        raise ValueError(
            f"--repeat={args.repeat} 超出 diag 种子列表 {config.DIAG_REPEAT_SEEDS}，"
            f"请显式指定 --seed。"
        )
    print(f"[infer-val] repeat={args.repeat} → seed={seed}")

    # ── sklearn 即时划分（替代原来的 load_fold + filter_subjects）───
    val_subjects = get_val_subjects(
        index_csv=args.diag_csv,
        fold=args.fold,
        n_splits=args.n_splits,
        seed=seed,
    )
    print(
        f"[infer-val] val_subjects (sklearn StratifiedGroupKFold, "
        f"seed={seed}, fold={args.fold}): "
        f"n={len(val_subjects)}"
    )

    # ── 加载模型 ──────────────────────────────────────────────────
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

    # ── 加载并过滤验证数据 ────────────────────────────────────────
    diag_df = pd.read_csv(args.diag_csv)
    emotion_df = pd.read_csv(args.emotion_csv)
    require_columns(
        diag_df,
        ["subject_id", "diagnosis_label", "trial_id", "de_path", "de_win_id"],
        args.diag_csv,
    )
    require_columns(
        emotion_df,
        ["subject_id", "diagnosis_label", "emotion_label", "trial_id", "de_path", "de_win_id"],
        args.emotion_csv,
    )

    # 用 sklearn 划分出的验证被试过滤（替代原来的 filter_subjects + fold_data）
    diag_df = filter_val_subjects(diag_df, val_subjects)
    emotion_df = filter_val_subjects(emotion_df, val_subjects)
    if diag_df.empty or emotion_df.empty:
        raise ValueError("Validation diagnosis or emotion data is empty.")

    print(f"[infer-val] diagnosis windows={len(diag_df)} emotion windows={len(emotion_df)}")

    # ── 阶段一：诊断推理 ──────────────────────────────────────────
    diag_df["p_dep"] = predict_windows(
        model=models["diagnosis"],
        df=diag_df,
        root=ROOT,
        device=device,
        batch_size=args.batch_size,
        logit_key="diagnosis_logits",
    )
    subject_probs = aggregate_subject_prob(diag_df, "p_dep", "subject_id")
    subject_truth = diag_df.groupby("subject_id", as_index=False).agg(
        true_diag=("diagnosis_label", "first")
    )
    subject_probs = subject_probs.merge(subject_truth, on="subject_id", how="left")
    subject_probs["pred_diag"] = (subject_probs["p_dep_subject"] >= 0.5).astype(int)

    # ── 阶段二：情绪推理 ──────────────────────────────────────────
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

    # ── 聚合与融合 ────────────────────────────────────────────────
    p_hc = aggregate_trial_prob(emotion_df, "p_pos_hc_window", "subject_id", "p_pos_hc")
    p_dep = aggregate_trial_prob(emotion_df, "p_pos_dep_window", "subject_id", "p_pos_dep")
    truth = emotion_df.groupby(["subject_id", "trial_id"], as_index=False).agg(
        true_diag=("diagnosis_label", "first"),
        true_emotion=("emotion_label", "first"),
    )
    preds = (
        truth.merge(p_hc, on=["subject_id", "trial_id"], how="left")
        .merge(p_dep, on=["subject_id", "trial_id"], how="left")
        .merge(subject_probs[["subject_id", "p_dep_subject", "pred_diag"]], on="subject_id", how="left")
    )
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
            "subject_id",
            "trial_id",
            "true_diag",
            "p_dep_subject",
            "pred_diag",
            "true_emotion",
            "p_pos_hc",
            "p_pos_dep",
            "p_final",
            "pred_emotion",
        ]
    ]

    # ── 计算指标 ──────────────────────────────────────────────────
    # 注意：hc_emotion_acc / dep_emotion_acc 分别评估的是
    # true_diag==0 (DEP) 和 true_diag==1 (HC) 子集上的情绪识别准确率
    metrics = {
        "repeat": args.repeat,
        "fold": args.fold,
        "seed": seed,
        "diag_subject_acc": accuracy(subject_probs["true_diag"], subject_probs["pred_diag"]),
        "emotion_trial_acc_soft": accuracy(preds["true_emotion"], preds["pred_emotion"]),
        "emotion_macro_f1_soft": macro_f1(preds["true_emotion"], preds["pred_emotion"]),
        "hc_emotion_acc": accuracy(
            preds.loc[preds["true_diag"].astype(int) == 1, "true_emotion"],
            preds.loc[preds["true_diag"].astype(int) == 1, "pred_emotion"],
        ),
        "dep_emotion_acc": accuracy(
            preds.loc[preds["true_diag"].astype(int) == 0, "true_emotion"],
            preds.loc[preds["true_diag"].astype(int) == 0, "pred_emotion"],
        ),
    }

    # ── 保存结果 ──────────────────────────────────────────────────
    pred_dir = args.predictions_dir / reapt_name(args.repeat, args.fold)
    pred_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(pred_dir / "val_two_stage_preds.csv", index=False)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, args.results_dir / f"{reapt_name(args.repeat, args.fold)}_metrics.json")
    print(f"[infer-val] wrote {pred_dir / 'val_two_stage_preds.csv'}")
    print(f"[infer-val] metrics={metrics}")


if __name__ == "__main__":
    main()

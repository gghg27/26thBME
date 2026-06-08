from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config
from utils.checkpoint import load_meta
from utils.folds import (
    combined_model_param_dir,
    get_unified_subject_split,
    reapt_name,
    require_columns,
    save_json,
)
from utils.metrics import accuracy, macro_f1
from utils.predict import (
    aggregate_subject_hard_vote,
    aggregate_subject_prob,
    aggregate_trial_hard_vote,
    aggregate_trial_prob,
    load_model_for_inference,
    predict_windows,
)

# ── 从 checkpoint meta 中读取训练时的 train/val 被试划分 ──────────
def _load_training_split(meta_path: str | Path) -> tuple[set[str], set[str]]:
    """
    尝试从 meta JSON 中读取训练时保存的 train_subjects / val_subjects。
    返回 (train_set, val_set)，若缺失则返回空集。
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return set(), set()
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set(), set()
    train = {str(s) for s in (meta.get("train_subjects") or [])}
    val = {str(s) for s in (meta.get("val_subjects") or [])}
    return train, val


def _load_subjects_from_checkpoint(checkpoint_path: str | Path) -> tuple[set[str], set[str]]:
    """直接从 .pt checkpoint 中读取 train_subjects / val_subjects。"""
    import torch

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        return set(), set()
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return set(), set()
    if not isinstance(ckpt, dict):
        return set(), set()
    train = {str(s) for s in (ckpt.get("train_subjects") or [])}
    val = {str(s) for s in (ckpt.get("val_subjects") or [])}
    return train, val


def _check_overlap(model_name: str, eval_val: set[str], train_train: set[str]) -> None:
    """检查评估用的验证被试是否与模型训练集有重叠（即数据泄露）。"""
    overlap = eval_val & train_train
    if overlap:
        print(
            f"[WARNING] {model_name}: {len(overlap)} 个评估验证被试出现在该模型训练集中！"
            f" 重叠被试: {sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
        )
    else:
        print(f"[OK] {model_name}: 评估验证被试与该模型训练集无重叠。")


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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--hard_voting",
        action="store_true",
        default=config.VAL_HARD_VOTING,
        help="Enable hard voting mode: majority vote across windows + hard diagnosis routing. "
        "Default from config.VAL_HARD_VOTING.",
    )
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
    使用项目统一划分函数即时重建验证集被试。

    与训练脚本一致：优先按 label4 推断 0=DEP, 1=HC，再按 subject_id 分组。
    """
    split = get_unified_subject_split(
        index_csv=index_csv,
        fold=fold,
        n_splits=n_splits,
        seed=seed,
    )
    return [str(s) for s in split["val_all"]]


def filter_val_subjects(df: pd.DataFrame, val_subjects: list[str]) -> pd.DataFrame:
    """只保留属于验证集被试的行。"""
    val_set = set(val_subjects)
    result = df[df["subject_id"].astype(str).isin(val_set)].copy()
    return result.reset_index(drop=True)


def add_diagnosis_truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一生成诊断真值列，避免 diagnosis_label 编码方向不一致。

    当前训练 CSV 里 diagnosis_label 是 DEP=1, HC=0；但部分训练代码
    会用 label4 >= 2 推出 0=DEP, 1=HC。这里显式生成:
      - true_dep: 1=DEP, 0=HC，用于和 p_dep / pred_diag 比较
      - true_group: "DEP" / "HC"，用于 HC/DEP 子集指标
    """
    result = df.copy()

    if "diagnosis" in result.columns:
        diag_text = result["diagnosis"].astype(str).str.upper()
        result["true_group"] = diag_text.where(diag_text.isin(["DEP", "HC"]), None)
        result["true_dep"] = (diag_text == "DEP").astype(int)
        return result

    if "label4" in result.columns:
        label4 = result["label4"].astype(int)
        is_hc = label4 >= 2
        result["true_group"] = is_hc.map({True: "HC", False: "DEP"})
        result["true_dep"] = (~is_hc).astype(int)
        return result

    if "diagnosis_label" in result.columns:
        result["true_dep"] = result["diagnosis_label"].astype(int)
        result["true_group"] = result["true_dep"].map({1: "DEP", 0: "HC"})
        return result

    raise KeyError("需要 diagnosis / label4 / diagnosis_label 中至少一列来确定诊断真值。")


def main() -> None:
    args = parse_args()

    import torch

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    voting_mode = "hard" if args.hard_voting else "soft"
    print(f"[infer-val] voting_mode={voting_mode} (hard=majority vote + hard routing, soft=mean prob + soft fusion)")

    # ── 确定 seed：与训练脚本共用 config.get_seed_for_repeat ────
    if args.seed is not None:
        seed = args.seed
    else:
        seed = config.get_seed_for_repeat("diagnosis", args.repeat)
    print(f"[infer-val] repeat={args.repeat} → seed={seed}")

    # ── 加载模型（需要先加载 combined_meta 才能读各个模型的 meta）───
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

    # ── 确定验证被试：优先从训练时保存的 val_subjects 读取 ────────
    #   Step 1: 尝试从诊断模型的 meta JSON 读取
    diag_meta_path = resolve_path(combined["models"]["diagnosis"]["meta"])
    _, diag_val_from_meta = _load_training_split(diag_meta_path)

    #   Step 2: 如果 meta 中没有，尝试从 .pt checkpoint 直接读
    if not diag_val_from_meta:
        diag_ckpt_path = resolve_path(combined["models"]["diagnosis"]["path"])
        _, diag_val_from_meta = _load_subjects_from_checkpoint(diag_ckpt_path)

    #   Step 3: 都读不到才用 sklearn 即时重建（并打印警告）
    if diag_val_from_meta:
        val_subjects = sorted(diag_val_from_meta)
        print(
            f"[infer-val] val_subjects 来自训练时 checkpoint: "
            f"n={len(val_subjects)}"
        )
    else:
        val_subjects = get_val_subjects(
            index_csv=args.diag_csv,
            fold=args.fold,
            n_splits=args.n_splits,
            seed=seed,
        )
        print(
            f"[WARNING] checkpoint 中无 val_subjects，使用 sklearn 即时重建: "
            f"n={len(val_subjects)} — 可能与训练时划分不一致！"
        )

    # ── 检查数据泄露：评估用的 val_subjects 是否出现在训练集中 ────
    eval_val_set = set(val_subjects)
    checkpoint_splits: dict[str, tuple[set[str], set[str]]] = {}
    for task in ("diagnosis", "hc_emotion", "dep_emotion"):
        meta_path = resolve_path(combined["models"][task]["meta"])
        train_train_set, task_val_set = _load_training_split(meta_path)
        if not train_train_set and not task_val_set:
            # meta 没有则从 checkpoint 直接读
            ckpt_path = resolve_path(combined["models"][task]["path"])
            train_train_set, task_val_set = _load_subjects_from_checkpoint(ckpt_path)
        checkpoint_splits[task] = (train_train_set, task_val_set)
        if train_train_set:
            _check_overlap(task, eval_val_set, train_train_set)
        else:
            print(f"[INFO] {task}: checkpoint 中无 train_subjects，无法检查重叠。")

    diag_val_set = checkpoint_splits.get("diagnosis", (set(), set()))[1]
    hc_val_set = checkpoint_splits.get("hc_emotion", (set(), set()))[1]
    dep_val_set = checkpoint_splits.get("dep_emotion", (set(), set()))[1]
    if diag_val_set:
        if diag_val_set == eval_val_set:
            print(f"[OK] diagnosis val_subjects 与本次评估 val_subjects 完全一致。")
        else:
            print(
                "[WARNING] diagnosis checkpoint val_subjects 与本次评估 val_subjects 不一致: "
                f"only_eval={sorted(eval_val_set - diag_val_set)} "
                f"only_ckpt={sorted(diag_val_set - eval_val_set)}"
            )
    if hc_val_set or dep_val_set:
        emotion_val_union = hc_val_set | dep_val_set
        if emotion_val_union == eval_val_set:
            print(
                "[OK] hc_emotion.val_subjects ∪ dep_emotion.val_subjects "
                "与本次评估 val_subjects 完全一致。"
            )
        else:
            print(
                "[WARNING] HC/DEP 情绪 checkpoint 的 val_subjects 合集与本次评估不一致: "
                f"only_eval={sorted(eval_val_set - emotion_val_union)} "
                f"only_emotion_ckpt={sorted(emotion_val_union - eval_val_set)}"
            )
        print(
            f"[infer-val] checkpoint val counts: "
            f"diagnosis={len(diag_val_set) if diag_val_set else 'NA'}, "
            f"hc={len(hc_val_set) if hc_val_set else 'NA'}, "
            f"dep={len(dep_val_set) if dep_val_set else 'NA'}"
        )

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

    diag_df = add_diagnosis_truth_columns(diag_df)
    emotion_df = add_diagnosis_truth_columns(emotion_df)

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
    if args.hard_voting:
        # 硬投票：每个窗口投 0/1，被试级别多数表决
        subject_probs = aggregate_subject_hard_vote(diag_df, "p_dep", "subject_id")
        # pred_diag 已由多数表决确定
    else:
        # 软投票：窗口概率均值 → 阈值 0.5
        subject_probs = aggregate_subject_prob(diag_df, "p_dep", "subject_id")
        subject_probs["pred_diag"] = (subject_probs["p_dep_subject"] >= 0.5).astype(int)

    subject_truth = diag_df.groupby("subject_id", as_index=False).agg(
        true_diag=("true_dep", "first"),
        true_group=("true_group", "first"),
    )
    subject_probs = subject_probs.merge(subject_truth, on="subject_id", how="left")

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
    if args.hard_voting:
        # 硬投票：每个窗口投 0/1，trial 级别多数表决
        p_hc = aggregate_trial_hard_vote(emotion_df, "p_pos_hc_window", "subject_id", "p_pos_hc")
        p_dep = aggregate_trial_hard_vote(emotion_df, "p_pos_dep_window", "subject_id", "p_pos_dep")
        p_hc.rename(columns={"pred_hard": "pred_emotion_hc_model"}, inplace=True)
        p_dep.rename(columns={"pred_hard": "pred_emotion_dep_model"}, inplace=True)
    else:
        # 软投票：窗口概率均值
        p_hc = aggregate_trial_prob(emotion_df, "p_pos_hc_window", "subject_id", "p_pos_hc")
        p_dep = aggregate_trial_prob(emotion_df, "p_pos_dep_window", "subject_id", "p_pos_dep")
    truth = emotion_df.groupby(["subject_id", "trial_id"], as_index=False).agg(
        true_diag=("true_dep", "first"),
        true_group=("true_group", "first"),
        true_emotion=("emotion_label", "first"),
    )
    preds = (
        truth.merge(p_hc, on=["subject_id", "trial_id"], how="left")
        .merge(p_dep, on=["subject_id", "trial_id"], how="left")
        .merge(subject_probs[["subject_id", "p_dep_subject", "pred_diag"]], on="subject_id", how="left")
    )

    if args.hard_voting:
        # 硬投票融合：诊断结果硬路由到对应情绪模型
        # pred_diag=0 (HC) → 使用 HC 模型多数表决结果
        # pred_diag=1 (DEP) → 使用 DEP 模型多数表决结果
        preds["pred_emotion"] = np.where(
            preds["pred_diag"] == 0,
            preds["pred_emotion_hc_model"],
            preds["pred_emotion_dep_model"],
        )
        # p_final: 保留对应被选中模型的窗口概率均值，用于参考
        preds["p_final"] = np.where(
            preds["pred_diag"] == 0,
            preds["p_pos_hc"],
            preds["p_pos_dep"],
        )
    else:
        # 软投票融合：诊断概率加权插值 HC/DEP 情绪模型输出
        preds["p_final"] = (
            (1.0 - preds["p_dep_subject"]) * preds["p_pos_hc"]
            + preds["p_dep_subject"] * preds["p_pos_dep"]
        )
        preds["pred_emotion_hc_model"] = (preds["p_pos_hc"] >= 0.5).astype(int)
        preds["pred_emotion_dep_model"] = (preds["p_pos_dep"] >= 0.5).astype(int)
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
            "true_group",
            "p_dep_subject",
            "pred_diag",
            "true_emotion",
            "p_pos_hc",
            "p_pos_dep",
            "pred_emotion_hc_model",
            "pred_emotion_dep_model",
            "p_final",
            "pred_emotion",
        ]
    ]

    # ── 计算指标 ──────────────────────────────────────────────────
    # hc_emotion_acc / dep_emotion_acc: 最终软路由预测在对应真实组上的指标。
    # hc_model_emotion_acc / dep_model_emotion_acc: 单个情绪模型在对应真实组上的指标，
    # 更适合与 train_hc.py / train_dep.py 训练时的 val trial_acc 对齐。
    hc_mask = preds["true_group"].astype(str).str.upper() == "HC"
    dep_mask = preds["true_group"].astype(str).str.upper() == "DEP"
    hc_true = preds.loc[hc_mask, "true_emotion"]
    hc_final_pred = preds.loc[hc_mask, "pred_emotion"]
    hc_model_pred = preds.loc[hc_mask, "pred_emotion_hc_model"]
    dep_true = preds.loc[dep_mask, "true_emotion"]
    dep_final_pred = preds.loc[dep_mask, "pred_emotion"]
    dep_model_pred = preds.loc[dep_mask, "pred_emotion_dep_model"]

    emotion_suffix = "hard" if args.hard_voting else "soft"
    diag_suffix = "hard" if args.hard_voting else "soft"

    metrics = {
        "repeat": args.repeat,
        "fold": args.fold,
        "seed": seed,
        "voting_mode": voting_mode,
        "n_val_subjects": int(len(val_subjects)),
        "n_hc_trials": int(hc_mask.sum()),
        "n_dep_trials": int(dep_mask.sum()),
        f"diag_subject_acc_{diag_suffix}": accuracy(
            subject_probs["true_diag"], subject_probs["pred_diag"]
        ),
        f"emotion_trial_acc_{emotion_suffix}": accuracy(
            preds["true_emotion"], preds["pred_emotion"]
        ),
        f"emotion_macro_f1_{emotion_suffix}": macro_f1(
            preds["true_emotion"], preds["pred_emotion"]
        ),
        f"hc_emotion_acc_{emotion_suffix}": accuracy(hc_true, hc_final_pred),
        f"hc_emotion_f1_{emotion_suffix}": macro_f1(hc_true, hc_final_pred),
        f"dep_emotion_acc_{emotion_suffix}": accuracy(dep_true, dep_final_pred),
        f"dep_emotion_f1_{emotion_suffix}": macro_f1(dep_true, dep_final_pred),
        f"hc_model_emotion_acc_{emotion_suffix}": accuracy(hc_true, hc_model_pred),
        f"hc_model_emotion_f1_{emotion_suffix}": macro_f1(hc_true, hc_model_pred),
        f"dep_model_emotion_acc_{emotion_suffix}": accuracy(dep_true, dep_model_pred),
        f"dep_model_emotion_f1_{emotion_suffix}": macro_f1(dep_true, dep_model_pred),
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

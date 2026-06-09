# -*- coding: utf-8 -*-
"""
V1 测试集集成预测脚本。

扫描 params 文件夹下所有模型 checkpoint，逐模型推理并 soft voting 集成，
输出 trial 级预测标签。

用法:
    python V1/predict_test_ensemble.py                           # 使用默认参数
    python V1/predict_test_ensemble.py --params_dir model_params --test_csv com_test_trial_index_2s.csv

在脚本顶部 MODEL_CONFIG 中指定要加载的模型类和 checkpoint 匹配模式。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V1_ROOT = Path(__file__).resolve().parent
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# =========================================================
# MODEL CONFIG —— 在此处指定要加载的模型
# =========================================================

# 可选:
#   ("two_branch",                    "TwoBranchModel")
#   ("two_branch_ssas",               "TwoBranchModel")
#   ("two_branch_subject_relative",   "TwoBranchModel")
MODEL_MODULE = "two_branch"               # V1 下的模块名（不含 .py）
MODEL_CLASS  = "TwoBranchModel"           # 模块中的模型类名

# checkpoint 搜索模式：params_dir 下哪些子目录包含模型
CKPT_GLOB_PATTERN = "two_branch_reapt*_fold*"   # 目录名匹配模式 (glob)
CKPT_FILENAME     = "best.pt"                    # 目录内的 checkpoint 文件名

# 模型初始化参数（传递给 MODEL_CLASS.__init__）
MODEL_KWARGS: dict = dict(
    sfreq=250.0,
    prior_matrix=None,
    topk=6,
    dropout=0.2,
    num_subjects=48,
    num_classes_4=4,
    num_classes_2=2,
)

# 如果模型来自 subject_relative 版本，需额外参数
# MODEL_KWARGS.update(use_subject_relative_de=False, use_subject_relative_bio=False)

# 推理参数
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 输出
OUTPUT_DIR = ROOT / "V1" / "test_submissions"

# =========================================================
# Test prediction
# =========================================================


def load_model_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """加载 checkpoint 并返回 model (eval mode)。"""
    # 动态导入
    import importlib
    module = importlib.import_module(MODEL_MODULE)
    model_cls = getattr(module, MODEL_CLASS)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 尝试从 checkpoint config 中读取 num_subjects
    ckpt_config = ckpt.get("config", {})
    kwargs = dict(MODEL_KWARGS)
    if "num_subjects" in ckpt_config:
        kwargs["num_subjects"] = int(ckpt_config["num_subjects"])

    # 兼容 subject_relative 版本
    if "use_subject_relative_de" in ckpt_config:
        kwargs["use_subject_relative_de"] = bool(ckpt_config["use_subject_relative_de"])
        kwargs["use_subject_relative_bio"] = bool(ckpt_config["use_subject_relative_bio"])

    model = model_cls(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def find_checkpoints(params_dir: Path) -> list[Path]:
    """扫描 params_dir，返回所有匹配的 checkpoint 路径。"""
    matches = sorted(params_dir.glob(CKPT_GLOB_PATTERN))
    ckpts = []
    for subdir in matches:
        if not subdir.is_dir():
            continue
        ckpt_path = subdir / CKPT_FILENAME
        if ckpt_path.exists():
            ckpts.append(ckpt_path)
        else:
            # 也尝试 combined_best_fold*.pt 等常见命名
            alt = sorted(subdir.glob("best*.pt"))
            if alt:
                ckpts.append(alt[0])
    return ckpts


def _dict_collate(batch_list):
    from torch.utils.data._utils.collate import default_collate
    keys = list(batch_list[0].keys())
    return {k: default_collate([b[k] for b in batch_list]) for k in keys}


@torch.no_grad()
def predict_single_model(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    device: torch.device,
    id_col: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
) -> pd.DataFrame:
    """
    单个模型对 test_df（窗口级）做推理，返回 trial 级 prob_pos。
    """
    from dataloader import EEGWindowDataset

    dataset = EEGWindowDataset(
        test_df.reset_index(drop=True),
        label_col=None,
        root=ROOT,
        return_de=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_dict_collate,
    )

    user_ids = []
    trial_ids = []
    probs = []

    for batch in tqdm(loader, desc="Predict", leave=False):
        x = batch["x"].to(device)
        de_feat = batch["de_feat"].to(device)

        out = model(x, de_feat, lambda_dom=0.0, dataset_name="comp4")

        # 优先用二分类头的正性概率
        if "logits_2cls" in out:
            prob_pos = torch.softmax(out["logits_2cls"], dim=1)[:, 1]
        else:
            prob_pos = torch.softmax(out["logits"], dim=1)[:, 1]

        user_ids.extend(batch[id_col].detach().cpu().numpy().tolist())
        trial_ids.extend(batch["trial_id"].detach().cpu().numpy().tolist())
        probs.extend(prob_pos.detach().cpu().numpy().tolist())

    # 聚合到 trial 级（同 trial 内窗口取均值）
    pred_df = pd.DataFrame({
        "user_id": user_ids,
        "trial_id": trial_ids,
        "prob_window": probs,
    })
    trial_pred = (
        pred_df.groupby(["user_id", "trial_id"], as_index=False)
        .agg(prob_pos=("prob_window", "mean"), n_windows=("prob_window", "count"))
    )
    return trial_pred


def ensemble_and_save(
    all_probs: list[pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    """Soft voting: 所有模型概率取均值。"""
    merged = all_probs[0]
    for i, df in enumerate(all_probs[1:]):
        merged = merged.merge(df, on=["user_id", "trial_id"], how="inner", suffixes=("", f"_m{i+1}"))

    prob_cols = [c for c in merged.columns if c.startswith("prob_pos")]
    merged["prob_pos_ensemble"] = merged[prob_cols].mean(axis=1)
    merged["n_models"] = len(prob_cols)
    merged["Emotion_label"] = (merged["prob_pos_ensemble"] >= 0.5).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 完整结果
    full_path = output_dir / "test_ensemble_probs.csv"
    merged.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"[save] 完整概率: {full_path}")

    # 提交格式
    submission = merged[["user_id", "trial_id", "Emotion_label"]].copy()
    sub_path = output_dir / "submission.csv"
    submission.to_csv(sub_path, index=False, encoding="utf-8-sig")
    print(f"[save] 提交文件: {sub_path}")

    # 统计
    n_total = len(submission)
    n_pos = int(submission["Emotion_label"].sum())
    print(f"[stats] 总 trial: {n_total}, 正性: {n_pos} ({100*n_pos/max(n_total,1):.1f}%)")

    return merged


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="V1 测试集集成预测")
    parser.add_argument("--params_dir", type=Path, default=ROOT / "model_params",
                        help="包含模型 checkpoint 的目录")
    parser.add_argument("--test_csv", type=Path,
                        default=ROOT / "com_test_trial_index_2s.csv",
                        help="测试集 CSV 路径")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR,
                        help="输出目录")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="推理 batch size")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="推理设备")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 查找 checkpoint ──
    print(f"[scan] params_dir={args.params_dir}")
    print(f"[scan] pattern={CKPT_GLOB_PATTERN} / {CKPT_FILENAME}")
    ckpt_paths = find_checkpoints(args.params_dir)
    if not ckpt_paths:
        print(f"[ERROR] 在 {args.params_dir} 下未找到匹配的 checkpoint。")
        print(f"        期望 pattern: {CKPT_GLOB_PATTERN}/*/{CKPT_FILENAME}")
        return
    print(f"[scan] 找到 {len(ckpt_paths)} 个 checkpoint:")
    for p in ckpt_paths:
        print(f"  {p}")

    # ── 加载测试数据 ──
    test_csv = args.test_csv
    if not test_csv.exists():
        alt = ROOT / "com_test_trial_index_10s.csv"
        print(f"[warn] {test_csv} 不存在，尝试 {alt}")
        test_csv = alt
    if not test_csv.exists():
        raise FileNotFoundError(f"测试 CSV 不存在: {args.test_csv}")

    print(f"[data] test_csv={test_csv}")
    test_df = pd.read_csv(test_csv)
    id_col = "user_id" if "user_id" in test_df.columns else "subject_id"
    print(f"[data] test samples={len(test_df)}, id_col={id_col}")

    # 展开窗口索引
    from utils.data import expand_window_index
    test_df = expand_window_index(test_df, root=ROOT)
    print(f"[data] expanded windows={len(test_df)}")

    # ── 逐模型推理 ──
    all_trial_probs = []
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"\n[{i+1}/{len(ckpt_paths)}] {ckpt_path}")
        model = load_model_from_checkpoint(ckpt_path, device)
        trial_pred = predict_single_model(
            model=model,
            test_df=test_df,
            device=device,
            id_col=id_col,
            batch_size=args.batch_size,
        )
        trial_pred = trial_pred[["user_id", "trial_id", "prob_pos"]].copy()
        trial_pred.rename(columns={"prob_pos": f"prob_pos"}, inplace=True)
        all_trial_probs.append(trial_pred)

    # ── 集成 ──
    print(f"\n[ensemble] soft voting over {len(all_trial_probs)} models...")
    ensemble_and_save(all_trial_probs, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()

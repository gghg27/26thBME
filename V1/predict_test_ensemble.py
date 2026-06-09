# -*- coding: utf-8 -*-
"""
V1 测试集集成预测脚本（独立于训练流程）。

用法:
    python V1/predict_test_ensemble.py                           # 默认参数
    python V1/predict_test_ensemble.py --batch_size 256
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
# Config — 在此修改要加载的模型
# =========================================================

# 模型模块和类名
MODEL_MODULE = "two_branch_subject_relative"
MODEL_CLASS  = "TwoBranchModel"

# checkpoint 搜索
PARAMS_DIR       = ROOT / "model_params"
CKPT_GLOB_PATTERN = "two_branch_reapt*_fold*"
CKPT_FILENAME     = "best.pt"

# 模型初始化参数（checkpoint config 中的会覆盖此处默认值）
MODEL_KWARGS = dict(
    sfreq=250.0, prior_matrix=None, topk=6, dropout=0.2,
    num_subjects=48, num_classes_4=4, num_classes_2=2,
    use_subject_relative_de=False, use_subject_relative_bio=False,
    bio_abs_scale=0.3, relative_eps=1e-6,
)

# 推理参数
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# 测试 CSV
TEST_CSV = ROOT / "com_test_trial_index_2s.csv"

# 输出
OUTPUT_DIR = V1_ROOT / "test_submissions"

# 被试内 top-K 参数（每个 subject 取 top k_pos 个 trial 判为正性）
USE_TOPK = True
K_POS = 4


# =========================================================
# Helpers
# =========================================================

def find_checkpoints(params_dir: Path) -> list[Path]:
    matches = sorted(params_dir.glob(CKPT_GLOB_PATTERN))
    ckpts = []
    for subdir in matches:
        if not subdir.is_dir():
            continue
        ckpt_path = subdir / CKPT_FILENAME
        if ckpt_path.exists():
            ckpts.append(ckpt_path)
        else:
            alt = sorted(subdir.glob("best*.pt"))
            if alt:
                ckpts.append(alt[0])
    return ckpts


def load_model(ckpt_path: Path, device: torch.device):
    import importlib
    module = importlib.import_module(MODEL_MODULE)
    model_cls = getattr(module, MODEL_CLASS)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})

    kwargs = dict(MODEL_KWARGS)
    for key in ["num_subjects", "use_subject_relative_de", "use_subject_relative_bio",
                "bio_abs_scale", "relative_eps"]:
        if key in ckpt_config:
            kwargs[key] = type(MODEL_KWARGS.get(key, 0))(ckpt_config[key])

    model = model_cls(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _dict_collate(batch_list):
    from torch.utils.data._utils.collate import default_collate
    # 分离 tensor 字段和非 tensor 字段
    tensor_keys = ["x", "de_feat", "subject_id", "trial_id"]
    result = {}
    for k in tensor_keys:
        if k in batch_list[0]:
            result[k] = default_collate([b[k] for b in batch_list])
    # 非 tensor 字段直接传 list
    for k in ["_uid", "_tid"]:
        if k in batch_list[0]:
            result[k] = [b[k] for b in batch_list]
    return result


def prepare_test_df(test_csv: Path) -> tuple[pd.DataFrame, str]:
    test_df = pd.read_csv(test_csv)
    id_col = "user_id" if "user_id" in test_df.columns else "subject_id"
    print(f"[data] test_csv={test_csv}, samples={len(test_df)}, id_col={id_col}")

    if "de_win_id" in test_df.columns:
        print("[data] 已有 de_win_id，无需展开")
        return test_df, id_col

    from utils.data import expand_window_index
    try:
        test_df = expand_window_index(test_df, root=ROOT)
    except (FileNotFoundError, OSError) as e:
        print(f"[warn] expand_window_index 失败: {e}")
        print("[warn] 用 n_windows 列直接展开...")
        if "n_windows" not in test_df.columns:
            raise RuntimeError("测试 CSV 缺少 de_win_id / n_windows 列。")
        rows = []
        for _, row in test_df.iterrows():
            nw = int(row["n_windows"])
            for win_id in range(nw):
                item = row.to_dict()
                item["de_win_id"] = win_id
                rows.append(item)
        test_df = pd.DataFrame(rows)
        print(f"[data] 用 n_windows 展开: {len(test_df)} 窗口")

    print(f"[data] 展开后窗口数: {len(test_df)}")
    return test_df, id_col


# =========================================================
# Prediction
# =========================================================

class _TestDataset(torch.utils.data.Dataset):
    """轻量测试 Dataset，直接接受 DataFrame，处理 trial_path 缺失的情况。"""

    def __init__(self, df: pd.DataFrame, root: Path):
        self.df = df.reset_index(drop=True)
        self.root = root
        self._trial_cache: dict[str, np.ndarray | None] = {}

    def _resolve(self, path_val: str) -> Path:
        """解析数据文件路径，兼容 test CSV 中的相对路径。"""
        text = str(path_val).replace("\\", "/")
        p = Path(text)
        if p.is_absolute():
            return p

        # 候选路径（与 utils/data.py 的 resolve_data_path 保持一致）
        candidates = [
            self.root / p,                           # root + 原始相对路径
            self.root / "data" / p.name,             # root/data/文件名
            self.root / "testdata" / p.name,         # root/testdata/文件名
        ]
        # 去掉开头的 ../ 后再试
        cleaned = text
        while cleaned.startswith("../"):
            cleaned = cleaned[3:]
        if cleaned.startswith("data/"):
            candidates.append(self.root / cleaned[len("data/"):])       # 去掉 data/ 前缀直接放 root
            candidates.append(self.root / "data" / cleaned[len("data/"):])  # root/data/ 下
        candidates.append(self.root / cleaned)
        candidates.append(self.root / "data" / Path(cleaned).name)

        for c in candidates:
            if c.exists():
                return c
        # 都不存在则返回第一个候选用于报错
        return candidates[0]

    def _load_trial(self, path_val: str) -> np.ndarray | None:
        if path_val in self._trial_cache:
            return self._trial_cache[path_val]
        try:
            arr = np.load(self._resolve(path_val), mmap_mode="r")
            self._trial_cache[path_val] = arr
            return arr
        except (FileNotFoundError, OSError):
            self._trial_cache[path_val] = None
            return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # raw signal
        if "trial_path" in row.index:
            trial_arr = self._load_trial(row["trial_path"])
            if trial_arr is not None:
                start = int(row.get("start", 0)) if "start" in row.index else 0
                end = int(row.get("end", trial_arr.shape[-1])) if "end" in row.index else trial_arr.shape[-1]
                x = trial_arr[:, start:end].astype("float32", copy=False)
            else:
                n_ch = 30; t_len = 500
                x = np.zeros((n_ch, t_len), dtype="float32")
        else:
            n_ch = 30; t_len = 500
            x = np.zeros((n_ch, t_len), dtype="float32")

        # DE features
        de_path = self._resolve(row["de_path"])
        de_arr = np.load(de_path)
        if de_arr.ndim >= 3 and "de_win_id" in row.index:
            de_feat = de_arr[int(row["de_win_id"])]
        else:
            de_feat = de_arr
        de_feat = de_feat.astype("float32", copy=False)

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "de_feat": torch.tensor(de_feat, dtype=torch.float32),
            "subject_id": torch.tensor(0, dtype=torch.long),
            "trial_id": torch.tensor(int(row["trial_id"]), dtype=torch.long),
            "_uid": str(row.get("user_id", row.get("subject_id", ""))),
            "_tid": int(row["trial_id"]),
        }


@torch.no_grad()
def predict_single_model(
    model, test_df: pd.DataFrame, device: torch.device,
    id_col: str, batch_size: int,
) -> pd.DataFrame:
    dataset = _TestDataset(test_df, ROOT)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=_dict_collate,
    )

    user_ids, trial_ids, probs, scores = [], [], [], []
    for batch in tqdm(loader, desc="Predict", leave=False):
        x = batch["x"].to(device)
        de_feat = batch["de_feat"].to(device)

        out = model(x, de_feat, lambda_dom=0.0, dataset_name="comp4")

        if "logits_2cls" in out:
            p = torch.softmax(out["logits_2cls"], dim=1)
        else:
            p = torch.softmax(out["logits"], dim=1)
        prob_pos = p[:, 1]

        # score_pos: 四分类头的正性概率 P(class=1)+P(class=3)
        if "logits_4cls" in out:
            p4 = torch.softmax(out["logits_4cls"], dim=1)
            score_pos = p4[:, 1] + p4[:, 3]
        else:
            score_pos = prob_pos

        user_ids.extend(batch["_uid"])
        trial_ids.extend(batch["_tid"])
        probs.extend(prob_pos.detach().cpu().numpy().tolist())
        scores.extend(score_pos.detach().cpu().numpy().tolist())

    pred_df = pd.DataFrame({
        "user_id": user_ids, "trial_id": trial_ids,
        "prob_window": probs, "score_window": scores,
    })
    trial_pred = (
        pred_df.groupby(["user_id", "trial_id"], as_index=False)
        .agg(prob_pos=("prob_window", "mean"), score_pos=("score_window", "mean"),
             n_windows=("prob_window", "count"))
    )
    return trial_pred


# =========================================================
# Ensemble & Top-K
# =========================================================

def apply_subject_topk(trial_records: list[dict], k_pos: int = 4) -> list[dict]:
    """被试内按 score_pos 排序，top K 判 1，其余判 0。"""
    from collections import defaultdict
    by_subject = defaultdict(list)
    for r in trial_records:
        by_subject[str(r["user_id"])].append(dict(r))

    results = []
    for sid, records in by_subject.items():
        records = sorted(records, key=lambda x: float(x.get("score_pos", 0)), reverse=True)
        cur_k = int(k_pos) if len(records) == 8 else max(1, int(round(len(records) * 0.5)))
        cur_k = max(0, min(cur_k, len(records)))
        pos_keys = {(r["user_id"], r["trial_id"]) for r in records[:cur_k]}
        for r in records:
            r["Emotion_label"] = 1 if (r["user_id"], r["trial_id"]) in pos_keys else 0
            results.append(r)
    return results


def ensemble_and_save(all_probs: list[pd.DataFrame], output_dir: Path):
    # soft voting
    merged = all_probs[0]
    for i, df in enumerate(all_probs[1:]):
        merged = merged.merge(df, on=["user_id", "trial_id"], how="inner",
                              suffixes=("", f"_m{i+1}"))

    prob_cols = [c for c in merged.columns if c.startswith("prob_pos")]
    score_cols = [c for c in merged.columns if c.startswith("score_pos")]

    merged["prob_pos_ensemble"] = merged[prob_cols].mean(axis=1)
    merged["score_pos_ensemble"] = merged[score_cols].mean(axis=1)
    merged["n_models"] = len(prob_cols)
    merged["pred_threshold"] = (merged["prob_pos_ensemble"] >= 0.5).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 完整结果（threshold 版本）
    merged.to_csv(output_dir / "test_ensemble_probs.csv", index=False, encoding="utf-8-sig")
    print(f"[save] 完整概率: {output_dir / 'test_ensemble_probs.csv'}")

    # threshold 提交
    sub_th = merged[["user_id", "trial_id", "pred_threshold"]].copy()
    sub_th.rename(columns={"pred_threshold": "Emotion_label"}, inplace=True)
    sub_th.to_csv(output_dir / "submission_threshold.csv", index=False, encoding="utf-8-sig")
    n_pos = int(sub_th["Emotion_label"].sum())
    print(f"[save] submission (threshold): {output_dir / 'submission_threshold.csv'}")
    print(f"       正性: {n_pos}/{len(sub_th)} ({100*n_pos/max(len(sub_th),1):.1f}%)")

    # top-K 提交
    if USE_TOPK:
        trial_records = []
        for _, row in merged.iterrows():
            trial_records.append({
                "user_id": row["user_id"], "trial_id": row["trial_id"],
                "prob_pos": row["prob_pos_ensemble"], "score_pos": row["score_pos_ensemble"],
            })
        topk_results = apply_subject_topk(trial_records, k_pos=K_POS)
        sub_topk = pd.DataFrame(topk_results)[["user_id", "trial_id", "Emotion_label"]]
        sub_topk.to_csv(output_dir / "submission_topk.csv", index=False, encoding="utf-8-sig")
        n_pos_tk = int(sub_topk["Emotion_label"].sum())
        print(f"[save] submission (topk={K_POS}): {output_dir / 'submission_topk.csv'}")
        print(f"       正性: {n_pos_tk}/{len(sub_topk)} ({100*n_pos_tk/max(len(sub_topk),1):.1f}%)")

    return merged


# =========================================================
# Main
# =========================================================

def main():
    global USE_TOPK, K_POS  # noqa: PLW0603

    parser = argparse.ArgumentParser(description="V1 测试集集成预测")
    parser.add_argument("--params_dir", type=Path, default=PARAMS_DIR)
    parser.add_argument("--test_csv", type=Path, default=TEST_CSV)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--no_topk", action="store_true", help="禁用 top-K 预测")
    parser.add_argument("--k_pos", type=int, default=0, help="top-K 中每被试正性 trial 数（0=使用默认4）")
    args = parser.parse_args()

    if args.no_topk:
        USE_TOPK = False
    if args.k_pos > 0:
        K_POS = args.k_pos

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 查找 checkpoint ──
    print(f"[scan] params_dir={args.params_dir}")
    print(f"[scan] pattern={CKPT_GLOB_PATTERN} / {CKPT_FILENAME}")
    ckpt_paths = find_checkpoints(args.params_dir)
    if not ckpt_paths:
        print(f"[ERROR] 未找到 checkpoint。")
        return
    print(f"[scan] 找到 {len(ckpt_paths)} 个 checkpoint:")
    for p in ckpt_paths:
        print(f"  {p}")

    # ── 准备测试数据 ──
    test_csv = args.test_csv
    if not test_csv.exists():
        alt = ROOT / "com_test_trial_index_10s.csv"
        if alt.exists():
            test_csv = alt
            print(f"[data] 使用备选 CSV: {test_csv}")
        else:
            raise FileNotFoundError(f"测试 CSV 不存在: {args.test_csv}")
    test_df, id_col = prepare_test_df(test_csv)

    # ── 逐模型推理 ──
    all_trial = []
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"\n[{i+1}/{len(ckpt_paths)}] {ckpt_path}")
        model = load_model(ckpt_path, device)
        trial_pred = predict_single_model(model, test_df, device, id_col, args.batch_size)
        trial_pred = trial_pred[["user_id", "trial_id", "prob_pos", "score_pos"]].copy()
        trial_pred.rename(columns={"prob_pos": f"prob_pos", "score_pos": f"score_pos"}, inplace=True)
        all_trial.append(trial_pred)

    # ── 集成 ──
    print(f"\n[ensemble] soft voting over {len(all_trial)} models...")
    ensemble_and_save(all_trial, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()

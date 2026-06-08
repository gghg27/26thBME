#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键批量推理脚本 —— 自动发现所有已训练的 (repeat, fold)，批量执行。

用法：
    python run_pipeline.py status           # 查看 model_params 中有哪些 fold 就绪
    python run_pipeline.py combine          # 对所有就绪的 fold 执行 combine
    python run_pipeline.py val              # 对所有 fold 做验证集推理，汇总平均指标
    python run_pipeline.py test             # 对所有 fold 做测试集推理 + 集成投票
    python run_pipeline.py all              # combine → val → test（全流程）

限定范围：
    python run_pipeline.py combine --repeat 0           # 只跑 repeat=0 的所有 fold
    python run_pipeline.py val --repeat 0 --fold 2      # 只跑特定一折
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

import numpy as np

from utils.folds import reapt_name

# ── 自动发现 ──────────────────────────────────────────────────────

def discover_folds(model_params_dir: Path) -> list[tuple[int, int]]:
    """
    扫描 model_params/，找到三个模型 checkpoint 都存在的 (repeat, fold)。

    支持两种目录命名模式：
      - 新版: diag_reapt{r}_fold{f}/   (训练脚本当前使用)
      - 旧版: repeat{r}_fold{f}/       (历史遗留，需检查内部文件)
    """
    if not model_params_dir.exists():
        return []

    def _scan(prefix: str) -> set[tuple[int, int]]:
        # 新版命名: diag_reapt0_fold1
        pat_new = re.compile(rf"{prefix}_reapt(\d+)_fold(\d+)")
        # 旧版命名: diag_repeat0_fold1 (极少见但兼容)
        pat_old = re.compile(rf"{prefix}_repeat(\d+)_fold(\d+)")
        pairs: set[tuple[int, int]] = set()
        for item in model_params_dir.iterdir():
            if not item.is_dir():
                continue
            m = pat_new.match(item.name) or pat_old.match(item.name)
            if m:
                pairs.add((int(m.group(1)), int(m.group(2))))
        return pairs

    # 新版命名：每个模型有独立目录
    ready_new = sorted(_scan("diag") & _scan("hc") & _scan("dep"))

    # 旧版命名：三个模型文件在同一个 repeat{r}_fold{f} 目录下
    # 检查目录内是否有 diag_best.pt / hc_best.pt / dep_best.pt
    legacy_pairs: set[tuple[int, int]] = set()
    legacy_pat = re.compile(r"repeat(\d+)_fold(\d+)")
    for item in model_params_dir.iterdir():
        if not item.is_dir():
            continue
        m = legacy_pat.match(item.name)
        if not m:
            continue
        r, f = int(m.group(1)), int(m.group(2))
        has_all = all((item / f"{p}_best.pt").exists() for p in ("diag", "hc", "dep"))
        if has_all:
            legacy_pairs.add((r, f))

    # 合并新旧，去重排序
    all_pairs = sorted(set(ready_new) | legacy_pairs)
    return all_pairs


def filter_folds(
    folds: list[tuple[int, int]],
    repeat: int | None,
    fold: int | None,
) -> list[tuple[int, int]]:
    result = folds
    if repeat is not None:
        result = [(r, f) for r, f in result if r == repeat]
    if fold is not None:
        result = [(r, f) for r, f in result if f == fold]
    return result


# ── 执行工具 ──────────────────────────────────────────────────────

def run_script(script: str, *extra_args: str) -> bool:
    """用 subprocess 调用项目内脚本，继承 stdout/stderr。"""
    cmd = [sys.executable, str(ROOT / script), *extra_args]
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode == 0


# ── 各步骤 ────────────────────────────────────────────────────────

def step_status(folds: list[tuple[int, int]], model_params_dir: Path) -> None:
    print(f"model_params 目录: {model_params_dir}")
    print(f"三模型就绪的 (repeat, fold): {len(folds)} 个")
    if folds:
        for r, f in folds:
            print(f"  repeat={r}  fold={f}")
    else:
        print("  (无 — 请先运行训练脚本)")

    if model_params_dir.exists():
        all_dirs = sorted(d.name for d in model_params_dir.iterdir() if d.is_dir())
        if all_dirs:
            print(f"\n所有子目录 ({len(all_dirs)} 个):")
            for d in all_dirs:
                print(f"  {d}")


def step_combine(folds: list[tuple[int, int]]) -> list[tuple[int, int]]:
    succeeded: list[tuple[int, int]] = []
    for r, f in folds:
        print(f"\n{'─' * 50}")
        print(f"[combine] repeat={r} fold={f}")
        ok = run_script("scripts/combine_fold_models.py",
                         "--repeat", str(r), "--fold", str(f))
        if ok:
            succeeded.append((r, f))
        else:
            print(f"[combine] repeat={r} fold={f} 失败，跳过后续与此 fold 相关的步骤")
    return succeeded


def step_val(folds: list[tuple[int, int]]) -> None:
    results_dir = ROOT / "results"
    all_metrics: list[dict] = []

    for r, f in folds:
        print(f"\n{'─' * 50}")
        print(f"[val] repeat={r} fold={f}")
        ok = run_script("inference/infer_val_fold.py",
                        "--repeat", str(r), "--fold", str(f))
        if ok:
            path = results_dir / f"{reapt_name(r, f)}_metrics.json"
            if path.exists():
                all_metrics.append(json.loads(path.read_text(encoding="utf-8")))

    if not all_metrics:
        print("\n无成功的验证结果。")
        return

    # 从第一个 metrics 文件推断实际 key 名（带 _soft/_hard 后缀）
    first = all_metrics[0]
    candidate_suffixes: dict[str, list[str]] = {
        "diag_subject_acc":       ["diag_subject_acc_soft", "diag_subject_acc_hard"],
        "emotion_trial_acc":      ["emotion_trial_acc_soft", "emotion_trial_acc_hard"],
        "emotion_macro_f1":       ["emotion_macro_f1_soft", "emotion_macro_f1_hard"],
        "hc_emotion_acc":         ["hc_emotion_acc_soft", "hc_emotion_acc_hard"],
        "hc_emotion_f1":          ["hc_emotion_f1_soft", "hc_emotion_f1_hard"],
        "dep_emotion_acc":        ["dep_emotion_acc_soft", "dep_emotion_acc_hard"],
        "dep_emotion_f1":         ["dep_emotion_f1_soft", "dep_emotion_f1_hard"],
        "hc_model_emotion_acc":   ["hc_model_emotion_acc_soft", "hc_model_emotion_acc_hard"],
        "hc_model_emotion_f1":    ["hc_model_emotion_f1_soft", "hc_model_emotion_f1_hard"],
        "dep_model_emotion_acc":  ["dep_model_emotion_acc_soft", "dep_model_emotion_acc_hard"],
        "dep_model_emotion_f1":   ["dep_model_emotion_f1_soft", "dep_model_emotion_f1_hard"],
    }
    keys: list[str] = []
    for candidates in candidate_suffixes.values():
        for c in candidates:
            if c in first:
                keys.append(c)
                break

    if not keys:
        print("\n无法识别指标 key，请检查 metrics JSON 格式。")
        return

    print(f"\n{'=' * 50}")
    print(f"验证集 {len(all_metrics)} 折 平均指标")
    print(f"{'=' * 50}")
    for key in keys:
        vals = [m[key] for m in all_metrics if key in m]
        if vals:
            print(f"  {key}:  {np.mean(vals):.4f}  ±  {np.std(vals, ddof=0):.4f}")

    # 保存汇总 JSON
    summary = {
        "n_folds": len(all_metrics),
        "averages": {
            key: float(np.mean([m[key] for m in all_metrics if key in m]))
            for key in keys
        },
        "per_fold": all_metrics,
    }
    out = results_dir / "val_summary_all_folds.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总已保存: {out}")


def step_test(folds: list[tuple[int, int]]) -> None:
    # 逐 fold 推理
    for r, f in folds:
        print(f"\n{'─' * 50}")
        print(f"[test] repeat={r} fold={f}")
        run_script("inference/infer_test_fold.py",
                   "--repeat", str(r), "--fold", str(f))

    # 集成投票
    print(f"\n{'─' * 50}")
    print("[ensemble] 集成所有 fold 投票")
    run_script("inference/ensemble_test.py")


# ── 主入口 ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="一键批量推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        choices=["status", "combine", "val", "test", "all"],
    )
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--model_params_dir", type=Path,
                        default=ROOT / "model_params")
    args = parser.parse_args()

    folds = discover_folds(args.model_params_dir)
    folds = filter_folds(folds, args.repeat, args.fold)

    if args.command == "status":
        step_status(folds, args.model_params_dir)
        return

    if not folds:
        print("未找到三模型就绪的 (repeat, fold)。")
        print("请先运行 train_diag.py / train_hc.py / train_dep.py。")
        print("使用 'python run_pipeline.py status' 查看详情。")
        return

    print(f"将处理 {len(folds)} 个 fold:")
    for r, f in folds:
        print(f"  repeat={r}  fold={f}")

    if args.command in ("combine", "all"):
        folds = step_combine(folds)

    if args.command in ("val", "all"):
        step_val(folds)

    if args.command in ("test", "all"):
        step_test(folds)

    print("\n完成。")


if __name__ == "__main__":
    main()

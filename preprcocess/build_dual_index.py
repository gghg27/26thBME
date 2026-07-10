from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build(args) -> pd.DataFrame:
    source = pd.read_csv(args.metadata_csv)
    rows = []
    for _, row in source.iterrows():
        old_trial = Path(str(row[args.trial_column]).replace("\\", "/")).name
        old_de = Path(str(row[args.de_column]).replace("\\", "/")).name
        trial_abs = args.trial_abs_root / old_trial
        trial_rel = args.trial_rel_root / old_trial
        de_abs_path = args.de_abs_root / old_de
        de_rel_path = args.de_rel_root / old_de
        for path in (trial_abs, trial_rel, de_abs_path, de_rel_path):
            if not path.exists():
                raise FileNotFoundError(path)
        x_abs = np.load(trial_abs, mmap_mode="r")
        x_rel = np.load(trial_rel, mmap_mode="r")
        de_abs = np.load(de_abs_path, mmap_mode="r")
        de_rel = np.load(de_rel_path, mmap_mode="r")
        if x_abs.shape != x_rel.shape or de_abs.shape != de_rel.shape:
            raise AssertionError(f"Dual shape mismatch for {old_trial}")
        item = row.to_dict()
        item.update({
            "trial_path_abs": rel(trial_abs),
            "trial_path_rel": rel(trial_rel),
            "de_path_abs": rel(de_abs_path),
            "de_path_rel": rel(de_rel_path),
        })
        item.pop(args.trial_column, None)
        item.pop(args.de_column, None)
        if "label4" in item and int(item["label4"]) >= 0:
            item["diagnosis_label"] = int(int(item["label4"]) >= 2)  # 0=DEP, 1=HC
        if "de_win_id" in item:
            rows.append(item)
        else:
            n_windows = int(item.get("n_windows", de_abs.shape[0]))
            for win_id in range(n_windows):
                expanded = dict(item)
                expanded.update({
                    "start": win_id * args.step,
                    "end": win_id * args.step + args.win_len,
                    "de_win_id": win_id,
                })
                rows.append(expanded)
    out = pd.DataFrame(rows)
    first = [
        "subject_id", "user_id", "file_name", "diagnosis", "diagnosis_label",
        "emotion", "emotion_label", "label4", "trial_id", "trial_path_abs",
        "trial_path_rel", "de_path_abs", "de_path_rel", "start", "end", "de_win_id",
    ]
    out = out[[column for column in first if column in out.columns] + [
        column for column in out.columns if column not in first
    ]]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    lengths = sorted((out["end"].astype(int) - out["start"].astype(int)).unique())
    print(
        f"[Dual Index] saved={args.out_csv}, rows={len(out)}, "
        f"window_lengths={lengths}, configured_step={args.step}"
    )
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--trial_abs_root", type=Path, required=True)
    parser.add_argument("--trial_rel_root", type=Path, required=True)
    parser.add_argument("--de_abs_root", type=Path, required=True)
    parser.add_argument("--de_rel_root", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--trial_column", default="trial_path")
    parser.add_argument("--de_column", default="de_path")
    parser.add_argument("--win_len", type=int, default=500)
    parser.add_argument("--step", type=int, default=250)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())

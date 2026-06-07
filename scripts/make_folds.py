from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.folds import require_columns, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create subject-level repeated folds.")
    parser.add_argument("--csv", type=Path, default=ROOT / "com_index_sub_10s.csv")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "folds")
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def native(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def stratified_subject_folds(
    subjects: pd.DataFrame, *, n_splits: int, seed: int
) -> list[list]:
    folds = [[] for _ in range(n_splits)]
    if "diagnosis_label" in subjects.columns:
        groups = subjects.groupby("diagnosis_label")["subject_id"].apply(list).to_dict()
        for label, ids in sorted(groups.items()):
            ids = [native(x) for x in ids]
            random.Random(seed + int(label) * 997).shuffle(ids)
            for idx, subject_id in enumerate(ids):
                folds[idx % n_splits].append(subject_id)
    else:
        ids = [native(x) for x in subjects["subject_id"].tolist()]
        random.Random(seed).shuffle(ids)
        for idx, subject_id in enumerate(ids):
            folds[idx % n_splits].append(subject_id)
    return [sorted(fold, key=lambda x: str(x)) for fold in folds]


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    require_columns(df, ["subject_id"], args.csv)
    subject_cols = ["subject_id"]
    if "diagnosis_label" in df.columns:
        subject_cols.append("diagnosis_label")
    subjects = df[subject_cols].drop_duplicates("subject_id").reset_index(drop=True)
    all_subjects = [native(x) for x in subjects["subject_id"].tolist()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[folds] subjects={len(all_subjects)} repeats={args.n_repeats} "
        f"splits={args.n_splits} seed={args.seed}"
    )
    for repeat in range(args.n_repeats):
        repeat_seed = args.seed + repeat * 1009
        val_folds = stratified_subject_folds(
            subjects, n_splits=args.n_splits, seed=repeat_seed
        )
        for fold, val_subjects in enumerate(val_folds):
            val_set = set(str(s) for s in val_subjects)
            train_subjects = [
                native(s) for s in all_subjects if str(s) not in val_set
            ]
            payload = {
                "repeat": repeat,
                "fold": fold,
                "seed": repeat_seed,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
            }
            out_path = args.output_dir / f"repeat{repeat}_fold{fold}.json"
            save_json(payload, out_path)
            print(
                f"[folds] wrote {out_path} "
                f"train={len(train_subjects)} val={len(val_subjects)}"
            )


if __name__ == "__main__":
    main()

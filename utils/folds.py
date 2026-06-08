from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def fold_name(repeat: int, fold: int) -> str:
    return f"repeat{repeat}_fold{fold}"


def reapt_name(repeat: int, fold: int) -> str:
    return f"reapt{repeat}_fold{fold}"


def task_fold_name(task_prefix: str, repeat: int, fold: int) -> str:
    return f"{task_prefix}_{reapt_name(repeat, fold)}"


def fold_json_path(repeat: int, fold: int, folds_dir: Path | str = "folds") -> Path:
    path = Path(folds_dir)
    if not path.is_absolute():
        path = project_root() / path
    return path / f"{fold_name(repeat, fold)}.json"


def model_param_dir(
    repeat: int, fold: int, model_params_dir: Path | str = "model_params"
) -> Path:
    path = Path(model_params_dir)
    if not path.is_absolute():
        path = project_root() / path
    return path / fold_name(repeat, fold)


def task_model_param_dir(
    task_prefix: str,
    repeat: int,
    fold: int,
    model_params_dir: Path | str = "model_params",
) -> Path:
    path = Path(model_params_dir)
    if not path.is_absolute():
        path = project_root() / path
    return path / task_fold_name(task_prefix, repeat, fold)


def combined_model_param_dir(
    repeat: int, fold: int, model_params_dir: Path | str = "model_params"
) -> Path:
    return task_model_param_dir("combined", repeat, fold, model_params_dir)


def load_fold(repeat: int, fold: int, folds_dir: Path | str = "folds") -> dict:
    path = fold_json_path(repeat, fold, folds_dir)
    if not path.exists():
        raise FileNotFoundError(f"Fold file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("repeat", -1)) != repeat or int(data.get("fold", -1)) != fold:
        raise ValueError(f"Fold file {path} does not match repeat={repeat}, fold={fold}")
    for key in ("train_subjects", "val_subjects"):
        if key not in data:
            raise KeyError(f"Fold file {path} is missing '{key}'")
    return data


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def require_columns(df: pd.DataFrame, columns: Iterable[str], csv_path: Path | str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")


def filter_subjects(
    df: pd.DataFrame,
    subjects: Iterable,
    diagnosis_label: int | None = None,
) -> pd.DataFrame:
    subjects = set(str(s) for s in subjects)
    result = df[df["subject_id"].astype(str).isin(subjects)].copy()
    if diagnosis_label is not None:
        result = result[result["diagnosis_label"].astype(int) == diagnosis_label].copy()
    return result.reset_index(drop=True)


# ── 统一交叉验证划分 ──────────────────────────────────────────────

def _subject_group_table(index_csv: str | Path) -> pd.DataFrame:
    """
    返回每个 subject 的 group_binary：0=DEP，1=HC。
    优先用 label4 推断，避免 diagnosis_label 原始编码不统一。
    """
    df = pd.read_csv(index_csv)

    if "label4" in df.columns:
        tmp = df[["subject_id", "label4"]].copy()
        tmp["group_binary"] = (tmp["label4"].astype(int) >= 2).astype(int)  # 0=DEP, 1=HC
        subject_df = (
            tmp.groupby("subject_id")["group_binary"]
            .agg(lambda s: int(s.mode().iloc[0]))
            .reset_index()
        )
    elif "diagnosis_label" in df.columns:
        subject_df = (
            df[["subject_id", "diagnosis_label"]]
            .drop_duplicates("subject_id")
            .reset_index(drop=True)
        )
        subject_df["group_binary"] = subject_df["diagnosis_label"].astype(int)
    else:
        raise KeyError("index_csv 中至少需要 label4 或 diagnosis_label。")

    subject_df["group_name"] = subject_df["group_binary"].map({0: "dep", 1: "hc"})
    return subject_df


def get_unified_subject_split(
    index_csv: str | Path,
    fold: int = 0,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """
    使用 StratifiedGroupKFold 在 **全部被试** 上做统一划分，
    三个模型（诊断 / HC情绪 / DEP情绪）共用同一套 train/val 被试分区。

    返回:
        {
            "train_all":      [...],   # 训练折全部被试
            "val_all":        [...],   # 验证折全部被试
            "train_hc":       [...],   # 训练折中的 HC 被试
            "val_hc":         [...],   # 验证折中的 HC 被试
            "train_dep":      [...],   # 训练折中的 DEP 被试
            "val_dep":        [...],   # 验证折中的 DEP 被试
            "subject_df":     DataFrame,
            "fold":           int,
            "seed":           int,
        }
    """
    subject_df = _subject_group_table(index_csv)

    if len(subject_df) < n_splits:
        raise ValueError(f"只有 {len(subject_df)} 个被试，无法做 {n_splits} 折。")

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    splits = list(
        sgkf.split(
            X=subject_df["subject_id"],
            y=subject_df["group_binary"],
            groups=subject_df["subject_id"],
        )
    )

    train_idx, val_idx = splits[fold]

    train_all = [str(s) for s in subject_df.iloc[train_idx]["subject_id"].tolist()]
    val_all = [str(s) for s in subject_df.iloc[val_idx]["subject_id"].tolist()]

    train_df = subject_df.iloc[train_idx]
    val_df = subject_df.iloc[val_idx]

    train_hc = [str(s) for s in train_df[train_df["group_binary"] == 1]["subject_id"].tolist()]
    val_hc = [str(s) for s in val_df[val_df["group_binary"] == 1]["subject_id"].tolist()]
    train_dep = [str(s) for s in train_df[train_df["group_binary"] == 0]["subject_id"].tolist()]
    val_dep = [str(s) for s in val_df[val_df["group_binary"] == 0]["subject_id"].tolist()]

    return {
        "train_all": train_all,
        "val_all": val_all,
        "train_hc": train_hc,
        "val_hc": val_hc,
        "train_dep": train_dep,
        "val_dep": val_dep,
        "subject_df": subject_df,
        "fold": fold,
        "seed": seed,
    }

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


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

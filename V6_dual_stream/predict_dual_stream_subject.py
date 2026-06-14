# -*- coding: utf-8 -*-
"""Predict test submissions with a trained V6 dual-stream subject model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import natural_key
from dual_stream_dataset import DualStreamSubjectSetDataset, dict_collate
from dual_stream_subject_model import DualStreamSubjectEmotionModel


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model_from_config(config_dict: dict) -> DualStreamSubjectEmotionModel:
    return DualStreamSubjectEmotionModel(
        sfreq=float(config_dict.get("sfreq", 250.0)),
        topk=int(config_dict.get("topk", 8)),
        dropout=float(config_dict.get("dropout", 0.35)),
        use_biomarkers=not bool(config_dict.get("no_biomarkers", False)),
        biomarker_dim=int(config_dict.get("biomarker_dim", 57)),
        de_num_bands=int(config_dict.get("de_num_bands", 5)),
        shared_mix_alpha=float(config_dict.get("shared_mix_alpha", 0.5)),
        share_abs_rel_encoder=bool(config_dict.get("share_abs_rel_encoder", False)),
        hidden_dim=int(config_dict.get("hidden_dim", 128)),
        relative_eps=float(config_dict.get("relative_eps", 1e-6)),
        encode_chunk_size=int(config_dict.get("encode_chunk_size", 0)),
    )


def apply_subject_topk(trial_records: list[dict], k_pos: int = 4, verbose: bool = True) -> list[dict]:
    by_user: dict[str, list[dict]] = {}
    for record in trial_records:
        by_user.setdefault(str(record["user_id"]), []).append(dict(record))

    out: list[dict] = []
    for user_id, records in by_user.items():
        records = sorted(records, key=lambda row: float(row["score_pos"]), reverse=True)
        cur_k = int(k_pos) if len(records) == 8 else max(1, int(round(len(records) * 0.5)))
        cur_k = max(0, min(cur_k, len(records)))
        positive = {(row["user_id"], int(row["trial_id"])) for row in records[:cur_k]}
        pos_count = 0
        for row in records:
            row["Emotion_label"] = 1 if (row["user_id"], int(row["trial_id"])) in positive else 0
            pos_count += int(row["Emotion_label"])
            out.append(row)
        if verbose:
            print(f"[topk] user={user_id} trials={len(records)} positive={pos_count} expected={cur_k} ok={pos_count == cur_k}")
    return sorted(out, key=lambda row: (natural_key(row["user_id"]), int(row["trial_id"])))


def sort_submission(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_user_sort_text"] = df["user_id"].astype(str).str.replace(r"\d+$", "", regex=True)
    df["_user_sort_num"] = df["user_id"].astype(str).str.extract(r"(\d+)$").fillna("-1").astype(int)
    df = (
        df.sort_values(["_user_sort_text", "_user_sort_num", "trial_id"])
        .drop(columns=["_user_sort_text", "_user_sort_num"])
        .reset_index(drop=True)
    )
    return df


@torch.no_grad()
def predict_test(
    model: torch.nn.Module,
    test_csv: str | Path,
    save_dir: str | Path,
    device: torch.device,
    *,
    batch_size: int = 4,
    num_workers: int = 0,
    threshold: float = 0.5,
    k_pos: int = 4,
) -> pd.DataFrame:
    dataset = DualStreamSubjectSetDataset(
        test_csv,
        mode="test",
        root=ROOT,
        check_paths=True,
        debug=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dict_collate,
    )

    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="V6-dual-test", leave=False):
        batch = move_batch_to_device(batch, device)
        out = model(
            x_abs=batch["x_abs"],
            x_rel=batch["x_rel"],
            de_abs=batch["de_abs"],
            de_z=batch["de_z"],
            win_mask=batch["win_mask"],
        )
        mix_prob = out["mix_prob"].clamp_min(1e-8)
        prob_neu = mix_prob[..., 0]
        prob_pos = mix_prob[..., 1]
        score_pos = torch.log(prob_pos) - torch.log(prob_neu)
        pred_soft_threshold = (prob_pos >= float(threshold)).long()
        batch_size_cur, n_trials = prob_pos.shape
        for i in range(batch_size_cur):
            user_id = str(batch["user_id"][i])
            for j in range(n_trials):
                rows.append(
                    {
                        "user_id": user_id,
                        "subject_id": int(batch["subject_id"][i].detach().cpu()),
                        "trial_id": int(batch["trial_id"][i, j].detach().cpu()),
                        "prob_neu": float(prob_neu[i, j].detach().cpu()),
                        "prob_pos": float(prob_pos[i, j].detach().cpu()),
                        "score_pos": float(score_pos[i, j].detach().cpu()),
                        "pred_soft_threshold": int(pred_soft_threshold[i, j].detach().cpu()),
                        "diag_prob_dep": float(out["diag_prob"][i, 0].detach().cpu()),
                        "diag_prob_hc": float(out["diag_prob"][i, 1].detach().cpu()),
                        "pred_diag": int(out["diag_logits"][i].argmax(dim=-1).detach().cpu()),
                    }
                )

    trial_df = pd.DataFrame(rows)
    if trial_df.empty:
        raise RuntimeError("No test predictions were produced.")
    trial_df = sort_submission(trial_df)

    topk_records = apply_subject_topk(
        trial_df[["user_id", "trial_id", "prob_pos", "score_pos"]].to_dict("records"),
        k_pos=k_pos,
        verbose=True,
    )
    topk_df = pd.DataFrame(topk_records)[["user_id", "trial_id", "Emotion_label"]].rename(
        columns={"Emotion_label": "pred_soft_topk"}
    )
    trial_df = trial_df.merge(topk_df, on=["user_id", "trial_id"], how="left")
    trial_df["pred_soft_topk"] = trial_df["pred_soft_topk"].fillna(0).astype(int)
    trial_df["Emotion_label"] = trial_df["pred_soft_topk"]
    trial_df = sort_submission(trial_df)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trial_df.to_csv(save_dir / "test_trial_probs.csv", index=False, encoding="utf-8-sig")

    sub_topk = trial_df[["user_id", "trial_id", "pred_soft_topk"]].copy()
    sub_topk.rename(columns={"pred_soft_topk": "Emotion_label"}, inplace=True)
    sub_topk = sort_submission(sub_topk)
    sub_topk.to_csv(save_dir / "submission_soft_topk.csv", index=False, encoding="utf-8-sig")

    sub_threshold = trial_df[["user_id", "trial_id", "pred_soft_threshold"]].copy()
    sub_threshold.rename(columns={"pred_soft_threshold": "Emotion_label"}, inplace=True)
    sub_threshold = sort_submission(sub_threshold)
    sub_threshold.to_csv(save_dir / "submission_soft_threshold.csv", index=False, encoding="utf-8-sig")

    selected_topk = int(sub_topk["Emotion_label"].sum())
    selected_threshold = int(sub_threshold["Emotion_label"].sum())
    print(f"[predict] saved test_trial_probs.csv to {save_dir}")
    print(f"[predict] submission_soft_topk positives={selected_topk}/{len(sub_topk)}")
    print(f"[predict] submission_soft_threshold positives={selected_threshold}/{len(sub_threshold)}")
    return trial_df


def predict_from_checkpoint(
    ckpt_path: str | Path,
    test_csv: str | Path,
    save_dir: str | Path,
    device: torch.device | str,
    *,
    batch_size: int = 4,
    num_workers: int = 0,
    threshold: float = 0.5,
    k_pos: int = 4,
) -> pd.DataFrame:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[device] requested {device}, but CUDA is unavailable; using CPU.")
        device = torch.device("cpu")
    ckpt = load_checkpoint(ckpt_path, device)
    config_dict = ckpt.get("config", {})
    model = build_model_from_config(config_dict).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"[checkpoint] path={ckpt_path} epoch={ckpt.get('epoch')} "
        f"fold={ckpt.get('fold')} version={ckpt.get('version', config_dict.get('version'))}"
    )
    return predict_test(
        model,
        test_csv,
        save_dir,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        threshold=threshold,
        k_pos=k_pos,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with V6 dual-stream subject checkpoint.")
    parser.add_argument("--test_csv", type=str, default="data/com_test_dual_stream_window_index_10s.csv")
    parser.add_argument("--test_ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="outputs/V6_dual_stream/test_predict")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--k_pos", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict_from_checkpoint(
        ckpt_path=args.test_ckpt,
        test_csv=args.test_csv,
        save_dir=args.save_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        k_pos=args.k_pos,
    )


if __name__ == "__main__":
    main()

"""Predict test trials by window averaging followed by 10-fold model averaging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from V9.configs.bfgcn_config import BFGCNConfig
from V9.datasets.eeg_bfgcn_dataset import EEGBFGCNDataset
from V9.models.bfgcn import BFGCN, extract_interpretability_outputs
from V9.utils.checkpoint import load_checkpoint
from V9.utils.trial_aggregation import aggregate_window_probabilities, natural_sort_trials


def _metadata(batch) -> pd.DataFrame:
    values = {}
    for key in (
        "subject_id", "user_id", "trial_id", "original_trial_id", "pseudo_trial_id",
        "window_id", "emotion_label", "diagnosis_label",
    ):
        item = batch[key]
        values[key] = item.detach().cpu().tolist() if torch.is_tensor(item) else list(item)
    return pd.DataFrame(values)


@torch.no_grad()
def predict_one_fold(
    checkpoint_path: Path,
    dataset: EEGBFGCNDataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    interpretability_path: Path | None = None,
) -> pd.DataFrame:
    state = load_checkpoint(checkpoint_path, device)
    config = BFGCNConfig.from_dict(state["config"])
    model = BFGCN(config).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    probabilities: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    for batch_index, batch in enumerate(loader):
        outputs = model(batch["de"].to(device), batch["plv"].to(device))
        probabilities.append(torch.softmax(outputs["emotion_logits"], dim=-1).cpu().numpy())
        metadata.append(_metadata(batch))
        if batch_index == 0 and interpretability_path is not None:
            extract_interpretability_outputs(model, batch, interpretability_path, device)
    return aggregate_window_probabilities(
        np.concatenate(probabilities), pd.concat(metadata, ignore_index=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="full_bfgcn")
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--test-index", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--save-interpretability", action="store_true")
    args = parser.parse_args()

    base = BFGCNConfig()
    checkpoint_root = Path(args.checkpoint_root or base.checkpoint_dir)
    if args.experiment != "full_bfgcn" and args.checkpoint_root is None:
        checkpoint_root = checkpoint_root / args.experiment
    checkpoint_paths = [checkpoint_root / f"fold_{fold:02d}" / "best_model.pt" for fold in range(1, args.folds + 1)]
    missing = [str(path) for path in checkpoint_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing fold checkpoints:\n" + "\n".join(missing))
    first_state = load_checkpoint(checkpoint_paths[0], "cpu")
    data_config = BFGCNConfig.from_dict(first_state["config"])
    if args.test_index:
        data_config.test_index_csv = str(Path(args.test_index).resolve())
    dataset = EEGBFGCNDataset(data_config.test_index_csv, config=data_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_trials = []
    for index, checkpoint_path in enumerate(checkpoint_paths, start=1):
        interpretation = None
        if args.save_interpretability and index == 1:
            interpretation = Path(args.output_dir or base.output_dir) / "interpretability_first_batch.npz"
        trials = predict_one_fold(
            checkpoint_path, dataset, args.batch_size, device, args.num_workers, interpretation
        )
        trials = trials.rename(columns={"prob_0": f"fold_{index:02d}_prob_0", "prob_1": f"fold_{index:02d}_prob_1"})
        fold_trials.append(trials)
        print(f"predicted fold {index:02d}: {len(trials)} trials")

    keys = ["subject_id", "user_id", "original_trial_id", "pseudo_trial_id", "trial_id"]
    merged = fold_trials[0][keys + ["fold_01_prob_0", "fold_01_prob_1"]]
    for index, frame in enumerate(fold_trials[1:], start=2):
        merged = merged.merge(
            frame[keys + [f"fold_{index:02d}_prob_0", f"fold_{index:02d}_prob_1"]],
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    neutral_columns = [f"fold_{index:02d}_prob_0" for index in range(1, args.folds + 1)]
    positive_columns = [f"fold_{index:02d}_prob_1" for index in range(1, args.folds + 1)]
    merged["prob_neutral"] = merged[neutral_columns].mean(axis=1)
    merged["prob_positive"] = merged[positive_columns].mean(axis=1)
    merged["Emotion_label"] = (merged["prob_positive"] > merged["prob_neutral"]).astype(int)
    merged["trial_id"] = merged["original_trial_id"].astype(int)
    output = natural_sort_trials(
        merged[["user_id", "trial_id", "Emotion_label", "prob_neutral", "prob_positive"]]
    )
    output_dir = Path(args.output_dir or base.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probability_path = output_dir / "bfgcn_test_probabilities.csv"
    submission_path = output_dir / "bfgcn_submission.xlsx"
    output.to_csv(probability_path, index=False, encoding="utf-8-sig")
    output[["user_id", "trial_id", "Emotion_label"]].to_excel(submission_path, index=False)
    print(f"saved {probability_path}")
    print(f"saved {submission_path}")


if __name__ == "__main__":
    main()


from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.checkpoint import extract_state_dict, instantiate_model, torch_load
from utils.data import EEGWindowDataset


def load_model_for_inference(meta: dict, checkpoint_path: Path, device: str = "cpu"):
    import torch

    checkpoint = torch_load(checkpoint_path, map_location=device)
    model = instantiate_model(meta)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def class1_prob(logits) -> np.ndarray:
    import torch

    if logits.ndim == 1 or logits.shape[-1] == 1:
        return torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy()
    return torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()


def select_logits(output, logit_key: str | None = None):
    if not isinstance(output, dict):
        return output
    if logit_key is not None and logit_key in output:
        return output[logit_key]
    for key in ("logits", "emo_logits", "emotion_logits", "diagnosis_logits"):
        if key in output:
            return output[key]
    raise KeyError(f"Model output does not contain logits. Keys: {list(output.keys())}")


def predict_windows(
    *,
    model,
    df: pd.DataFrame,
    root: Path,
    device: str = "cpu",
    batch_size: int = 128,
    logit_key: str | None = None,
) -> pd.Series:
    import torch
    from torch.utils.data import DataLoader

    dataset = EEGWindowDataset(
        df.reset_index(drop=True),
        label_col=None,
        root=root,
        return_de=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs = np.zeros(len(df), dtype="float32")
    with torch.no_grad():
        for x, de_feat, idx in loader:
            x = x.to(device)
            de_feat = de_feat.to(device)
            output = model(x, de_feat)
            logits = select_logits(output, logit_key)
            if hasattr(idx, "detach"):
                idx = idx.detach().cpu().numpy()
            probs[np.asarray(idx)] = class1_prob(logits)
    return pd.Series(probs, index=df.index, name="prob")


def aggregate_subject_prob(df: pd.DataFrame, prob_col: str, id_col: str) -> pd.DataFrame:
    return (
        df.groupby(id_col, as_index=False)[prob_col]
        .mean()
        .rename(columns={prob_col: "p_dep_subject"})
    )


def aggregate_trial_prob(
    df: pd.DataFrame, prob_col: str, id_col: str, out_col: str
) -> pd.DataFrame:
    return (
        df.groupby([id_col, "trial_id"], as_index=False)[prob_col]
        .mean()
        .rename(columns={prob_col: out_col})
    )

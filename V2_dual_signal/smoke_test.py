from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from V2_dual_signal.expert_ssas_emotion_model import (
    Stage1SSASSourceSelectionModel,
    Stage2ExpertEmotionAdaptationModel,
    mixture_emotion_nll_loss,
)
from V2_dual_signal.dataloader import DualViewCompetitionDataset
from V2_dual_signal.trial_supcon import TrialWindowDataset, flatten_trial_batch


def make_batch(batch_size: int = 2, window_len: int = 500):
    x_abs = torch.randn(batch_size, 30, window_len) * 8.0
    channel_mean = x_abs.mean(dim=-1, keepdim=True)
    channel_std = x_abs.std(dim=-1, keepdim=True).clamp_min(1e-6)
    x_rel = (x_abs - channel_mean) / channel_std
    de_abs = torch.randn(batch_size, 30, 5)
    de_rel = torch.randn(batch_size, 30, 5)
    return {"x_abs": x_abs, "x_rel": x_rel, "de_abs": de_abs, "de_rel": de_rel}


def load_synthetic_train_and_test_samples():
    temp = tempfile.TemporaryDirectory(prefix="v2_dual_smoke_")
    root = Path(temp.name)
    rows = []
    for subject_id in (0, 1):
        x_abs = (np.random.randn(30, 500) * 8.0).astype(np.float32)
        x_rel = ((x_abs - x_abs.mean(1, keepdims=True)) / (x_abs.std(1, keepdims=True) + 1e-6)).astype(np.float32)
        de_abs = np.random.randn(1, 30, 5).astype(np.float32)
        de_rel = np.random.randn(1, 30, 5).astype(np.float32)
        paths = {}
        for key, value in (("trial_abs", x_abs), ("trial_rel", x_rel), ("de_abs", de_abs), ("de_rel", de_rel)):
            path = root / f"subject{subject_id}_{key}.npy"
            np.save(path, value)
            paths[key] = str(path)
        rows.append({
            "subject_id": subject_id, "file_name": f"S{subject_id}.mat",
            "diagnosis": "DEP" if subject_id == 0 else "HC",
            "diagnosis_label": subject_id, "emotion": "neu", "emotion_label": 0,
            "label4": subject_id * 2, "trial_id": 1,
            "trial_path_abs": paths["trial_abs"], "trial_path_rel": paths["trial_rel"],
            "de_path_abs": paths["de_abs"], "de_path_rel": paths["de_rel"],
            "start": 0, "end": 500, "de_win_id": 0,
        })
    train_csv, test_csv = root / "train.csv", root / "test.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)
    pd.DataFrame(rows).to_csv(test_csv, index=False)
    train = DualViewCompetitionDataset(train_csv, root=root)
    test = DualViewCompetitionDataset(test_csv, root=root)
    train_sample = train[np.random.randint(len(train))]
    test_sample = test[np.random.randint(len(test))]
    trial_dataset = TrialWindowDataset(train, num_windows_per_trial=2, train=False)
    trial_items = [trial_dataset[0], trial_dataset[1]]
    trial_batch = {
        key: torch.stack([item[key] for item in trial_items], dim=0)
        for key in trial_items[0]
        if torch.is_tensor(trial_items[0][key])
    }
    flat, bsz, windows = flatten_trial_batch(trial_batch)
    assert flat["x_abs"].shape[:2] == (bsz * windows, 30)
    assert flat["de_rel"].shape[:2] == (bsz * windows, 30)
    batch = {
        key: torch.stack([train[0][key], train[1][key]], dim=0)
        for key in ("x_abs", "x_rel", "de_abs", "de_rel")
    }
    print(
        f"dataset samples: train x_abs={tuple(train_sample['x_abs'].shape)}, "
        f"test x_rel={tuple(test_sample['x_rel'].shape)}; "
        f"trial flatten={tuple(flat['x_abs'].shape)}"
    )
    return temp, batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--with_biomarkers", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    temp, disk_batch = load_synthetic_train_and_test_samples()
    batch = {key: value.to(device) for key, value in disk_batch.items()}
    labels = torch.tensor([0, 1], device=device)
    kwargs = dict(num_domains=3, dropout=0.1, use_biomarkers=args.with_biomarkers)

    stage1 = Stage1SSASSourceSelectionModel(**kwargs).to(device)
    stage2 = Stage2ExpertEmotionAdaptationModel(**kwargs).to(device)
    for model in (stage1, stage2):
        assert hasattr(model, "shared_encoder")
        assert not hasattr(model, "encoder_abs") and not hasattr(model, "encoder_rel")
        ids_before = tuple(id(p) for p in model.shared_encoder.parameters())
        ids_after = tuple(id(p) for p in model.shared_encoder.parameters())
        assert ids_before == ids_after

    optimizer = torch.optim.AdamW(
        list(stage1.parameters()) + list(stage2.parameters()), lr=1e-5
    )
    optimizer.zero_grad(set_to_none=True)
    out1 = stage1(**batch, lambda_emo=0.01, lambda_diag=0.01)
    out2 = stage2(**batch, lambda_subject=0.01)
    loss = (
        F.cross_entropy(out1["domain_logits"], labels)
        + F.cross_entropy(out1["emotion_logits_grl"], labels)
        + F.cross_entropy(out1["diagnosis_logits_grl"], labels)
        + mixture_emotion_nll_loss(out2["mix_prob"], labels)
        + F.cross_entropy(out2["diag_logits"], labels)
    )
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in list(stage1.parameters()) + list(stage2.parameters()) if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    optimizer.step()

    stage1.eval()
    stage2.eval()
    with torch.no_grad():
        val_out = stage1(**batch)
        test_out = stage2(**batch)
    assert torch.allclose(test_out["mix_prob"].sum(1), torch.ones(2, device=device), atol=1e-5)
    assert torch.allclose(test_out["diag_prob"].sum(1), torch.ones(2, device=device), atol=1e-5)
    print(f"loss={loss.item():.6f}; finite gradients={len(grads)}")
    print(f"z_abs.shape={tuple(test_out['z_abs'].shape)}")
    print(f"z_rel.shape={tuple(test_out['z_rel'].shape)}")
    for key in ("diag_logits", "shared_logits", "hc_logits", "dep_logits", "mix_prob"):
        print(f"{key}.shape={tuple(test_out[key].shape)}")
    print(f"mix_prob row sum={test_out['mix_prob'].sum(1).tolist()}")
    print(f"diag_prob row sum={test_out['diag_prob'].sum(1).tolist()}")
    print(f"validation z_rel.shape={tuple(val_out['z_rel'].shape)}; test forward=ok")
    temp.cleanup()


if __name__ == "__main__":
    main()

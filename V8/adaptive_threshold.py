"""Subject-wise adaptive decision threshold for V7 Experiment A.

This module deliberately has no dependency on the Stage 2 model.  It consumes
complete trial-score collections, so target-subject statistics are unsupervised
and calibration labels can only come from source subjects.
"""

from __future__ import annotations

import copy
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def encode_group_ids(values, device=None):
    """Map arbitrary (including string) subject IDs to contiguous integers."""
    strings = [str(x) for x in values]
    ordered = list(dict.fromkeys(strings))
    value_to_group = {value: i for i, value in enumerate(ordered)}
    ids = torch.tensor([value_to_group[x] for x in strings], dtype=torch.long, device=device)
    return ids, {i: value for value, i in value_to_group.items()}


class SubjectAdaptiveThreshold(nn.Module):
    def __init__(self, min_std: float = 0.1, init_temperature: float = 1.0) -> None:
        super().__init__()
        if min_std <= 0 or init_temperature <= 1e-4:
            raise ValueError("min_std must be positive and init_temperature must exceed 1e-4")
        self.min_std = float(min_std)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        # inverse softplus(init_temperature - 1e-4)
        target = float(init_temperature) - 1e-4
        self.raw_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(target))))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature) + 1e-4

    def forward(self, score: torch.Tensor, group_id: torch.Tensor, detach_score: bool = True) -> dict:
        if score.ndim != 1 or group_id.ndim != 1 or len(score) != len(group_id):
            raise ValueError("score and group_id must be equally sized one-dimensional tensors")
        score = score.detach() if detach_score else score
        group_id = group_id.to(device=score.device, dtype=torch.long)
        sample_threshold = torch.empty_like(score)
        thresholds, means, stds = {}, {}, {}
        for group in torch.unique(group_id, sorted=True):
            mask = group_id == group
            group_score = score[mask]
            mean = group_score.mean()
            std = group_score.std(unbiased=False).clamp_min(self.min_std)
            threshold = mean + self.alpha * std + self.beta
            key = int(group.item())
            means[key], stds[key], thresholds[key] = mean, std, threshold
            sample_threshold[mask] = threshold
        probability = torch.sigmoid((score - sample_threshold) / self.temperature)
        return {
            "adaptive_prob": probability,
            "adaptive_label": (score >= sample_threshold).long(),
            "sample_threshold": sample_threshold,
            "group_thresholds": thresholds,
            "group_means": means,
            "group_stds": stds,
            "alpha": self.alpha,
            "beta": self.beta,
            "temperature": self.temperature,
        }


def _subject_loss(prob, label, group_id, module, lambda_f1, lambda_reg, eps=1e-8):
    losses, bces, f1s = [], [], []
    for group in torch.unique(group_id):
        mask = group_id == group
        q, y = prob[mask].clamp(eps, 1 - eps), label[mask]
        class_terms = []
        if (y == 1).any(): class_terms.append(-torch.log(q[y == 1]).mean())
        if (y == 0).any(): class_terms.append(-torch.log1p(-q[y == 0]).mean())
        bce = torch.stack(class_terms).mean()
        tp = (q * y).sum(); fp = (q * (1 - y)).sum(); fn = ((1 - q) * y).sum()
        f1_pos = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        qn, yn = 1 - q, 1 - y
        tpn = (qn * yn).sum(); fpn = (qn * (1 - yn)).sum(); fnn = ((1 - qn) * yn).sum()
        f1_neg = (2 * tpn + eps) / (2 * tpn + fpn + fnn + eps)
        soft_f1 = (f1_pos + f1_neg) / 2
        bces.append(bce); f1s.append(soft_f1); losses.append(bce + lambda_f1 * (1 - soft_f1))
    bce = torch.stack(bces).mean(); soft_f1 = torch.stack(f1s).mean()
    regularization = module.alpha.square() + module.beta.square() + torch.log(module.temperature).square()
    return torch.stack(losses).mean() + lambda_reg * regularization, bce, soft_f1


def _metrics(prob: torch.Tensor, label: torch.Tensor) -> dict:
    pred = (prob >= 0.5).long().detach().cpu().numpy()
    true = label.long().detach().cpu().numpy()
    return {"acc": accuracy_score(true, pred),
            "macro_f1": f1_score(true, pred, average="macro", zero_division=0)}


def _evaluate(module, score, label, groups, lambda_f1, lambda_reg):
    module.eval()
    with torch.no_grad():
        out = module(score, groups)
        loss, bce, soft_f1 = _subject_loss(out["adaptive_prob"], label, groups, module, lambda_f1, lambda_reg)
        metrics = _metrics(out["adaptive_prob"], label)
    return {"loss": float(loss), "bce": float(bce), "soft_macro_f1": float(soft_f1), **metrics}, out


def train_threshold(records: pd.DataFrame, args, save_dir: Path, base_stage2_checkpoint: str,
                    device: torch.device):
    """Fit only three threshold parameters using labeled source trial records."""
    required = {"user_id", "score_pos", "label_emo"}
    if not required.issubset(records.columns):
        raise ValueError(f"source threshold records missing {sorted(required - set(records.columns))}")
    users = sorted(records.user_id.astype(str).unique())
    rng = np.random.default_rng(args.threshold_seed)
    shuffled = list(rng.permutation(users))
    can_split = len(users) >= 3 and 0 < args.threshold_val_ratio < 1
    if can_split:
        n_val = min(len(users) - 1, max(1, int(round(len(users) * args.threshold_val_ratio))))
        val_users, train_users = sorted(shuffled[:n_val]), sorted(shuffled[n_val:])
    else:
        warnings.warn("Too few source subjects for threshold validation; using all subjects for training without early stopping.")
        train_users, val_users = users, []

    def tensors(selected):
        frame = records[records.user_id.astype(str).isin(selected)].copy()
        group, mapping = encode_group_ids(frame.user_id.astype(str).tolist(), device)
        return frame, torch.tensor(frame.score_pos.to_numpy(), dtype=torch.float32, device=device), \
            torch.tensor(frame.label_emo.to_numpy(), dtype=torch.float32, device=device), group, mapping

    train_frame, train_score, train_y, train_group, _ = tensors(train_users)
    val_data = tensors(val_users) if val_users else None
    module = SubjectAdaptiveThreshold(args.threshold_min_std, args.threshold_init_temperature).to(device)
    initial = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.threshold_lr,
                                  weight_decay=args.threshold_weight_decay)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimizer_ids != {id(p) for p in module.parameters()}:
        raise RuntimeError("threshold optimizer contains unexpected parameters")
    history, best_state, best_epoch, best_value, patience = [], None, 0, float("inf"), 0
    for epoch in range(1, args.threshold_epochs + 1):
        module.train(); optimizer.zero_grad(set_to_none=True)
        out = module(train_score, train_group, detach_score=True)
        loss, bce, soft_f1 = _subject_loss(out["adaptive_prob"], train_y, train_group, module,
                                           args.lambda_threshold_f1, args.lambda_threshold_reg)
        loss.backward(); optimizer.step()
        train_metrics, train_out = _evaluate(module, train_score, train_y, train_group,
                                              args.lambda_threshold_f1, args.lambda_threshold_reg)
        if val_data:
            _, vs, vy, vg, _ = val_data
            val_metrics, _ = _evaluate(module, vs, vy, vg, args.lambda_threshold_f1, args.lambda_threshold_reg)
            monitored = val_metrics["loss"]
        else:
            val_metrics = {"loss": np.nan, "acc": np.nan, "macro_f1": np.nan}
            monitored = train_metrics["loss"]
        thresholds = np.asarray([float(value) for value in train_out["group_thresholds"].values()])
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}, "alpha": float(module.alpha),
               "beta": float(module.beta), "temperature": float(module.temperature),
               "mean_subject_threshold": float(np.mean(thresholds)),
               "min_subject_threshold": float(np.min(thresholds)),
               "max_subject_threshold": float(np.max(thresholds))}
        history.append(row)
        if monitored < best_value - args.threshold_min_delta:
            best_value, best_epoch, patience = monitored, epoch, 0
            best_state = copy.deepcopy(module.state_dict())
        else:
            patience += 1
        if val_data and patience >= args.threshold_patience:
            break
    module.load_state_dict(best_state if best_state is not None else module.state_dict())
    changed = any(not torch.equal(initial[k], module.state_dict()[k].detach().cpu()) for k in initial)
    if not changed:
        raise RuntimeError("Adaptive-threshold parameters did not change from initialization")
    train_metrics, _ = _evaluate(module, train_score, train_y, train_group,
                                  args.lambda_threshold_f1, args.lambda_threshold_reg)
    if val_data:
        _, vs, vy, vg, _ = val_data
        val_metrics, _ = _evaluate(module, vs, vy, vg, args.lambda_threshold_f1, args.lambda_threshold_reg)
    else:
        val_metrics = {}
    history_path = save_dir / "adaptive_threshold_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
    checkpoint = {
        "threshold_state_dict": module.state_dict(), "base_stage2_checkpoint": str(base_stage2_checkpoint),
        "threshold_epoch": best_epoch, "threshold_train_metrics": train_metrics,
        "threshold_val_metrics": val_metrics,
        "threshold_config": {"min_std": args.threshold_min_std,
                             "init_temperature": args.threshold_init_temperature,
                             "epochs": args.threshold_epochs, "lr": args.threshold_lr,
                             "weight_decay": args.threshold_weight_decay,
                             "patience": args.threshold_patience, "min_delta": args.threshold_min_delta,
                             "val_ratio": args.threshold_val_ratio, "seed": args.threshold_seed,
                             "lambda_f1": args.lambda_threshold_f1,
                             "lambda_reg": args.lambda_threshold_reg},
        "threshold_train_subjects": train_users, "threshold_val_subjects": val_users,
        "score_type": "log_mix_probability_ratio",
        "formula": "mean_subject + alpha * std_subject + beta",
    }
    path = save_dir / "adaptive_threshold_best.pt"
    torch.save(checkpoint, path)
    print(f"[adaptive threshold] best_epoch={best_epoch} alpha={float(module.alpha):.6f} "
          f"beta={float(module.beta):.6f} temperature={float(module.temperature):.6f}")
    return module, checkpoint, path


def load_threshold(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("threshold_config", {})
    module = SubjectAdaptiveThreshold(config.get("min_std", 0.1),
                                      config.get("init_temperature", 1.0)).to(device)
    module.load_state_dict(checkpoint["threshold_state_dict"]); module.eval()
    return module, checkpoint


def apply_threshold(frame: pd.DataFrame, module: SubjectAdaptiveThreshold, device: torch.device,
                    labeled: bool = False):
    """Apply one model's threshold after collecting every trial of each subject."""
    result = frame.copy()
    group, reverse = encode_group_ids(result.user_id.astype(str).tolist(), device)
    score = torch.tensor(result.score_pos.to_numpy(), dtype=torch.float32, device=device)
    module.eval()
    with torch.no_grad(): out = module(score, group)
    result["prob_neutral"] = 1.0 - result.prob_pos
    result["prob_positive"] = result.prob_pos
    result["score_positive"] = result.score_pos
    result["subject_score_mean"] = [float(out["group_means"][int(g)]) for g in group]
    result["subject_score_std"] = [float(out["group_stds"][int(g)]) for g in group]
    result["adaptive_threshold"] = out["sample_threshold"].cpu().numpy()
    result["adaptive_probability"] = out["adaptive_prob"].cpu().numpy()
    result["Emotion_label"] = out["adaptive_label"].cpu().numpy()
    summary = []
    for gid, user in reverse.items():
        mask = group.cpu().numpy() == gid; part = result.loc[mask]
        row: dict[str, Any] = {"user_id": user, "num_trials": int(mask.sum()),
             "score_mean": float(out["group_means"][gid]), "score_std": float(out["group_stds"][gid]),
             "threshold": float(out["group_thresholds"][gid]),
             "pred_positive_count": int(part.Emotion_label.sum())}
        if labeled and "label_emo" in part:
            y, pred = part.label_emo.to_numpy(), part.Emotion_label.to_numpy()
            row.update(true_positive_count=int(y.sum()), accuracy=accuracy_score(y, pred),
                       macro_f1=f1_score(y, pred, average="macro", zero_division=0))
        else:
            row.update(true_positive_count=None, accuracy=None, macro_f1=None)
        summary.append(row)
    return result, pd.DataFrame(summary)


def adaptive_validation_metrics(frame: pd.DataFrame) -> dict:
    y, pred = frame.label_emo.to_numpy(), frame.Emotion_label.to_numpy()
    fixed = (frame.prob_pos.to_numpy() >= 0.5).astype(int)
    return {"adaptive_trial_acc": accuracy_score(y, pred),
            "adaptive_trial_macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "adaptive_trial_confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]),
            "adaptive_positive_rate": float(pred.mean()),
            "adaptive_threshold_mean": float(frame.adaptive_threshold.mean()),
            "adaptive_threshold_std": float(frame.adaptive_threshold.std(ddof=0)),
            "fixed_trial_acc": accuracy_score(y, fixed),
            "fixed_trial_macro_f1": f1_score(y, fixed, average="macro", zero_division=0)}

# -*- coding: utf-8 -*-
"""Dual-stream subject-level diagnosis and trial-level emotion model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from V6.pmg_backbone import BrainGraphBackbone


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.extend([nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ThreeBranchEncoderWrapper(nn.Module):
    """Wrap the existing PMG backbone as forward(x, de_feat)."""

    def __init__(
        self,
        sfreq: float = 250.0,
        topk: int = 8,
        dropout: float = 0.2,
        prior_matrix: Optional[torch.Tensor] = None,
        use_biomarkers: bool = True,
        biomarker_dim: int = 57,
        use_subject_relative_de: bool = False,
        use_subject_relative_bio: bool = False,
        bio_abs_scale: float = 0.3,
        relative_eps: float = 1e-6,
        de_num_bands: int = 5,
    ) -> None:
        super().__init__()
        self.backbone = BrainGraphBackbone(
            sfreq=sfreq,
            node_dim=64,
            attn_dim=16,
            topk=topk,
            prior_matrix=prior_matrix,
            dropout=dropout,
            use_biomarkers=use_biomarkers,
            biomarker_dim=biomarker_dim,
            use_subject_relative_de=use_subject_relative_de,
            use_subject_relative_bio=use_subject_relative_bio,
            bio_abs_scale=bio_abs_scale,
            relative_eps=relative_eps,
            de_num_bands=de_num_bands,
        )
        self.out_dim = int(self.backbone.out_dim)

    def forward(self, x: torch.Tensor, de_feat: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        out = self.backbone(x, de_feat=de_feat, **kwargs)
        if "z" not in out:
            raise KeyError("BrainGraphBackbone must return 'z'.")
        return out


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    mask = mask.to(device=x.device, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=dim).clamp_min(eps)
    return (x * mask).sum(dim=dim) / denom


def masked_std(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    mean = masked_mean(x, mask, dim=dim, eps=eps)
    expanded_mean = mean.unsqueeze(dim)
    mask_float = mask.to(device=x.device, dtype=x.dtype)
    while mask_float.ndim < x.ndim:
        mask_float = mask_float.unsqueeze(-1)
    var = masked_mean((x - expanded_mean).pow(2), mask, dim=dim, eps=eps)
    return torch.sqrt(var.clamp_min(0.0) + eps)


class DualStreamSubjectEmotionModel(nn.Module):
    """Absolute/relative dual stream with subject-level diagnosis routing."""

    def __init__(
        self,
        sfreq: float = 250.0,
        topk: int = 8,
        dropout: float = 0.35,
        prior_matrix: Optional[torch.Tensor] = None,
        emotion_classes: int = 2,
        diagnosis_classes: int = 2,
        use_biomarkers: bool = True,
        biomarker_dim: int = 57,
        de_num_bands: int = 5,
        shared_mix_alpha: float = 0.5,
        share_abs_rel_encoder: bool = False,
        hidden_dim: int = 128,
        relative_eps: float = 1e-6,
        encode_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        self.shared_mix_alpha = float(max(0.0, min(1.0, shared_mix_alpha)))
        self.share_abs_rel_encoder = bool(share_abs_rel_encoder)
        self.relative_eps = float(relative_eps)
        self.encode_chunk_size = int(encode_chunk_size)

        encoder_kwargs = dict(
            sfreq=sfreq,
            topk=topk,
            dropout=dropout,
            prior_matrix=prior_matrix,
            use_biomarkers=use_biomarkers,
            biomarker_dim=biomarker_dim,
            use_subject_relative_de=False,
            use_subject_relative_bio=False,
            bio_abs_scale=0.3,
            relative_eps=relative_eps,
            de_num_bands=de_num_bands,
        )
        self.abs_encoder = ThreeBranchEncoderWrapper(**encoder_kwargs)
        self.rel_encoder = self.abs_encoder if self.share_abs_rel_encoder else ThreeBranchEncoderWrapper(**encoder_kwargs)
        self.encoder_dim = int(self.abs_encoder.out_dim)

        self.abs_trial_pool = nn.Sequential(
            nn.Linear(self.encoder_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.encoder_dim),
        )
        self.rel_trial_pool = nn.Sequential(
            nn.Linear(self.encoder_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.encoder_dim),
        )
        self.subject_mlp = nn.Sequential(
            nn.Linear(self.encoder_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.encoder_dim),
        )
        self.diagnosis_head = MLPHead(
            in_dim=self.encoder_dim,
            out_dim=diagnosis_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_layer_norm=True,
        )
        self.emotion_feature_mlp = nn.Sequential(
            nn.Linear(self.encoder_dim * 3 + diagnosis_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.encoder_dim),
            nn.LayerNorm(self.encoder_dim),
            nn.GELU(),
        )
        self.shared_emotion_head = MLPHead(self.encoder_dim, emotion_classes, hidden_dim=hidden_dim, dropout=dropout)
        self.hc_emotion_expert = MLPHead(self.encoder_dim, emotion_classes, hidden_dim=hidden_dim, dropout=dropout)
        self.dep_emotion_expert = MLPHead(self.encoder_dim, emotion_classes, hidden_dim=hidden_dim, dropout=dropout)

    def _encode_windows(
        self,
        encoder: ThreeBranchEncoderWrapper,
        x: torch.Tensor,
        de_feat: torch.Tensor,
        z_key: str,
    ) -> torch.Tensor:
        batch, trials, windows, channels, time_len = x.shape
        flat_x = x.reshape(batch * trials * windows, channels, time_len)
        flat_de = de_feat.reshape(batch * trials * windows, channels, de_feat.shape[-1])
        chunk_size = int(self.encode_chunk_size)
        if chunk_size <= 0 or chunk_size >= flat_x.size(0):
            enc = encoder(flat_x, flat_de)
            z = enc.get(z_key, enc["z"])
        else:
            z_chunks = []
            for start in range(0, flat_x.size(0), chunk_size):
                end = min(start + chunk_size, flat_x.size(0))
                enc = encoder(flat_x[start:end], flat_de[start:end])
                z_chunks.append(enc.get(z_key, enc["z"]))
            z = torch.cat(z_chunks, dim=0)
        return z.reshape(batch, trials, windows, -1)

    def _pool_windows(
        self,
        z_win: torch.Tensor,
        win_mask: torch.Tensor,
        pool: nn.Module,
    ) -> torch.Tensor:
        trial_mean = masked_mean(z_win, win_mask, dim=2, eps=self.relative_eps)
        trial_std = masked_std(z_win, win_mask, dim=2, eps=self.relative_eps)
        return pool(torch.cat([trial_mean, trial_std], dim=-1))

    def forward(
        self,
        x_abs: torch.Tensor,
        x_rel: torch.Tensor,
        de_abs: torch.Tensor,
        de_z: torch.Tensor,
        win_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_abs_win = self._encode_windows(self.abs_encoder, x_abs, de_abs, z_key="z_diag")
        z_rel_win = self._encode_windows(self.rel_encoder, x_rel, de_z, z_key="z_emotion")

        h_abs_trial = self._pool_windows(z_abs_win, win_mask, self.abs_trial_pool)
        h_rel_trial = self._pool_windows(z_rel_win, win_mask, self.rel_trial_pool)

        sub_mean = h_abs_trial.mean(dim=1)
        sub_std = h_abs_trial.std(dim=1, unbiased=False)
        sub_max = h_abs_trial.max(dim=1).values
        h_subject_abs = self.subject_mlp(torch.cat([sub_mean, sub_std, sub_max], dim=-1))
        diag_logits = self.diagnosis_head(h_subject_abs)
        diag_prob = torch.softmax(diag_logits, dim=-1)

        mu_abs = h_abs_trial.mean(dim=1, keepdim=True)
        std_abs = h_abs_trial.std(dim=1, keepdim=True, unbiased=False)
        h_abs_rel = h_abs_trial - mu_abs
        h_abs_z = h_abs_rel / (std_abs + self.relative_eps)
        diag_context = diag_prob[:, None, :].expand(-1, h_abs_trial.size(1), -1)

        h_emo = torch.cat([h_rel_trial, h_abs_rel, h_abs_z, diag_context], dim=-1)
        h_emo = self.emotion_feature_mlp(h_emo)

        shared_logits = self.shared_emotion_head(h_emo)
        hc_logits = self.hc_emotion_expert(h_emo)
        dep_logits = self.dep_emotion_expert(h_emo)

        prob_shared = torch.softmax(shared_logits, dim=-1)
        prob_hc = torch.softmax(hc_logits, dim=-1)
        prob_dep = torch.softmax(dep_logits, dim=-1)
        p_dep = diag_prob[:, 0:1]
        p_hc = diag_prob[:, 1:2]
        expert_mix_prob = p_dep[:, None, :] * prob_dep + p_hc[:, None, :] * prob_hc
        expert_mix_prob = expert_mix_prob / expert_mix_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mix_prob = self.shared_mix_alpha * prob_shared + (1.0 - self.shared_mix_alpha) * expert_mix_prob
        mix_prob = mix_prob / mix_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return {
            "diag_logits": diag_logits,
            "diag_prob": diag_prob,
            "shared_logits": shared_logits,
            "hc_logits": hc_logits,
            "dep_logits": dep_logits,
            "prob_shared": prob_shared,
            "prob_hc": prob_hc,
            "prob_dep": prob_dep,
            "expert_mix_prob": expert_mix_prob,
            "mix_prob": mix_prob,
            "h_subject_abs": h_subject_abs,
            "h_abs_trial": h_abs_trial,
            "h_rel_trial": h_rel_trial,
            "h_abs_rel": h_abs_rel,
            "h_abs_z": h_abs_z,
            "z_abs_win": z_abs_win,
            "z_rel_win": z_rel_win,
        }


def mixture_emotion_nll_loss(mix_prob: torch.Tensor, y_emo: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y_emo = y_emo.long()
    valid = y_emo >= 0
    if not valid.any():
        return mix_prob.new_tensor(0.0)
    probs = mix_prob[valid]
    labels = y_emo[valid]
    p_true = probs[torch.arange(probs.size(0), device=probs.device), labels]
    return -torch.log(p_true.clamp_min(eps)).mean()


def hard_expert_emotion_loss(
    hc_logits: torch.Tensor,
    dep_logits: torch.Tensor,
    y_emo: torch.Tensor,
    y_diag: torch.Tensor,
) -> torch.Tensor:
    y_emo = y_emo.long()
    y_diag = y_diag.long()
    valid = (y_emo >= 0) & (y_diag[:, None] >= 0)
    if not valid.any():
        return hc_logits.new_tensor(0.0)
    selected_logits = torch.empty_like(hc_logits)
    hc_mask = y_diag == 1
    dep_mask = y_diag == 0
    selected_logits[hc_mask] = hc_logits[hc_mask]
    selected_logits[dep_mask] = dep_logits[dep_mask]
    other_mask = ~(hc_mask | dep_mask)
    if other_mask.any():
        selected_logits[other_mask] = dep_logits[other_mask]
    return F.cross_entropy(selected_logits[valid], y_emo[valid])


def subject_ranking_loss(
    mix_prob: torch.Tensor,
    y_emo: torch.Tensor,
    margin: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    y_emo = y_emo.long()
    score_pos = torch.log(mix_prob[..., 1].clamp_min(eps)) - torch.log(mix_prob[..., 0].clamp_min(eps))
    losses = []
    for batch_idx in range(score_pos.size(0)):
        labels = y_emo[batch_idx]
        valid = labels >= 0
        pos_scores = score_pos[batch_idx][valid & (labels == 1)]
        neu_scores = score_pos[batch_idx][valid & (labels == 0)]
        if pos_scores.numel() == 0 or neu_scores.numel() == 0:
            continue
        diff = pos_scores[:, None] - neu_scores[None, :]
        losses.append(F.softplus(float(margin) - diff).mean())
    if not losses:
        return mix_prob.new_tensor(0.0)
    return torch.stack(losses).mean()

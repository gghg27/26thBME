# -*- coding: utf-8 -*-
"""V6 one-stage dual-feature soft-router model.

This model keeps the current PMG/BrainGraphBackbone dual feature design:
z_emotion is used by emotion heads, while z_diag is used by the diagnosis
router. SSAS, MMD, GRL, and target-domain adaptation heads are intentionally
not part of this version.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .pmg_backbone import BrainGraphBackbone
except ImportError:  # pragma: no cover - supports direct script execution from V6/
    from pmg_backbone import BrainGraphBackbone


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
        layers.extend(
            [
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ThreeBranchEncoderWrapper(nn.Module):
    """Unifies BrainGraphBackbone behind forward(x, de_feat, **kwargs)."""

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

    def forward(
        self,
        x: torch.Tensor,
        de_feat: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        out = self.backbone(x, de_feat=de_feat, **kwargs)
        if "z" not in out:
            raise KeyError("BrainGraphBackbone must return a fused 'z' feature.")
        return out


class DualFeatureSoftRouterModel(nn.Module):
    """Diagnosis-guided HC/DEP expert router without SSAS adaptation."""

    def __init__(
        self,
        sfreq: float = 250.0,
        topk: int = 8,
        dropout: float = 0.2,
        prior_matrix: Optional[torch.Tensor] = None,
        emotion_classes: int = 2,
        diagnosis_classes: int = 2,
        use_biomarkers: bool = True,
        biomarker_dim: int = 57,
        use_subject_relative_de: bool = False,
        use_subject_relative_bio: bool = False,
        bio_abs_scale: float = 0.3,
        relative_eps: float = 1e-6,
        de_num_bands: int = 5,
        shared_mix_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.use_subject_relative_de = bool(use_subject_relative_de)
        self.use_subject_relative_bio = bool(use_subject_relative_bio)
        self.bio_abs_scale = float(bio_abs_scale)
        self.relative_eps = float(relative_eps)
        self.shared_mix_alpha = float(max(0.0, min(1.0, shared_mix_alpha)))
        self.shared_encoder = ThreeBranchEncoderWrapper(
            sfreq=sfreq,
            topk=topk,
            dropout=dropout,
            prior_matrix=prior_matrix,
            use_biomarkers=use_biomarkers,
            biomarker_dim=biomarker_dim,
            use_subject_relative_de=self.use_subject_relative_de,
            use_subject_relative_bio=self.use_subject_relative_bio,
            bio_abs_scale=self.bio_abs_scale,
            relative_eps=self.relative_eps,
            de_num_bands=de_num_bands,
        )
        in_dim = self.shared_encoder.out_dim
        self.in_dim = in_dim
        self.diagnosis_router = MLPHead(
            in_dim=in_dim,
            out_dim=diagnosis_classes,
            hidden_dim=64,
            dropout=dropout,
        )
        self.shared_emotion_head = MLPHead(
            in_dim=in_dim,
            out_dim=emotion_classes,
            hidden_dim=64,
            dropout=dropout,
        )
        self.hc_emotion_expert = MLPHead(
            in_dim=in_dim,
            out_dim=emotion_classes,
            hidden_dim=64,
            dropout=dropout,
        )
        self.dep_emotion_expert = MLPHead(
            in_dim=in_dim,
            out_dim=emotion_classes,
            hidden_dim=64,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        de_feat: torch.Tensor,
        subject_de_mu: Optional[torch.Tensor] = None,
        subject_de_std: Optional[torch.Tensor] = None,
        subject_bio_mu: Optional[torch.Tensor] = None,
        subject_bio_std: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc = self.shared_encoder(
            x,
            de_feat=de_feat,
            subject_de_mu=subject_de_mu,
            subject_de_std=subject_de_std,
            subject_bio_mu=subject_bio_mu,
            subject_bio_std=subject_bio_std,
        )
        z_emotion = enc.get("z_emotion", enc["z"])
        z_diag_feat = enc.get("z_diag", enc["z"])

        diag_logits = self.diagnosis_router(z_diag_feat)
        shared_logits = self.shared_emotion_head(z_emotion)
        hc_logits = self.hc_emotion_expert(z_emotion)
        dep_logits = self.dep_emotion_expert(z_emotion)

        diag_prob = torch.softmax(diag_logits, dim=1)
        prob_shared = torch.softmax(shared_logits, dim=1)
        prob_hc = torch.softmax(hc_logits, dim=1)
        prob_dep = torch.softmax(dep_logits, dim=1)
        p_dep = diag_prob[:, 0:1]
        p_hc = diag_prob[:, 1:2]
        expert_mix_prob = p_dep * prob_dep + p_hc * prob_hc
        expert_mix_prob = expert_mix_prob / expert_mix_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mix_prob = self.shared_mix_alpha * prob_shared + (1.0 - self.shared_mix_alpha) * expert_mix_prob
        mix_prob = mix_prob / mix_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
        log_mix_prob = torch.log(mix_prob.clamp_min(1e-8))

        out = dict(enc)
        out.update(
            {
                "z": z_emotion,
                "z_emotion": z_emotion,
                "z_emo": z_emotion,
                "z_diag": z_diag_feat,
                "diag_logits": diag_logits,
                "diagnosis_logits": diag_logits,
                "shared_logits": shared_logits,
                "hc_logits": hc_logits,
                "dep_logits": dep_logits,
                "mix_prob": mix_prob,
                "expert_mix_prob": expert_mix_prob,
                "prob_shared": prob_shared,
                "prob_hc": prob_hc,
                "prob_dep": prob_dep,
                "diag_prob": diag_prob,
                "logits": log_mix_prob,
                "emotion_logits": log_mix_prob,
            }
        )
        return out


def mixture_emotion_nll_loss(
    mix_prob: torch.Tensor,
    y_emo: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    y_emo = y_emo.long()
    p_true = mix_prob[torch.arange(mix_prob.size(0), device=mix_prob.device), y_emo]
    return -torch.log(p_true.clamp_min(eps)).mean()


def hard_expert_emotion_loss(
    hc_logits: torch.Tensor,
    dep_logits: torch.Tensor,
    y_emo: torch.Tensor,
    y_diag: torch.Tensor,
) -> torch.Tensor:
    """Use HC expert for y_diag==1 and DEP expert for y_diag==0."""

    y_emo = y_emo.long()
    y_diag = y_diag.long()
    selected_logits = torch.empty_like(hc_logits)
    hc_mask = y_diag == 1
    dep_mask = y_diag == 0
    selected_logits[hc_mask] = hc_logits[hc_mask]
    selected_logits[dep_mask] = dep_logits[dep_mask]
    if (~(hc_mask | dep_mask)).any():
        selected_logits[~(hc_mask | dep_mask)] = dep_logits[~(hc_mask | dep_mask)]
    return F.cross_entropy(selected_logits, y_emo)

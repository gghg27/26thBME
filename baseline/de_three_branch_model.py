# -*- coding: utf-8 -*-
"""Three-branch encoder baseline for emotion classification.

The backbone is the existing BrainGraphBackbone in models.dep_contrast_bio:
multi-scale temporal convolution + PLV graph + biomarkers. The main head is
emotion binary classification, and the auxiliary head is four-class
diagnosis-emotion classification.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.dep_contrast_bio import BrainGraphBackbone


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DEThreeBranchEmotionClassifier(nn.Module):
    """Three-branch encoder with a primary 2-class emotion head and 4-class head."""

    def __init__(
        self,
        sfreq: float = 250.0,
        topk: int = 8,
        dropout: float = 0.2,
        use_biomarkers: bool = True,
        biomarker_dim: int = 57,
        de_num_bands: int = 5,
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = BrainGraphBackbone(
            sfreq=sfreq,
            node_dim=64,
            attn_dim=16,
            topk=topk,
            prior_matrix=None,
            dropout=dropout,
            use_biomarkers=use_biomarkers,
            biomarker_dim=biomarker_dim,
            de_num_bands=de_num_bands,
        )
        self.in_dim = int(self.backbone.out_dim)
        self.logits_2cls_head = MLPHead(
            in_dim=self.in_dim,
            out_dim=2,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )
        self.logits_4cls_head = MLPHead(
            in_dim=self.in_dim,
            out_dim=4,
            hidden_dim=head_hidden_dim,
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
        enc = self.backbone(
            x,
            de_feat=de_feat,
            subject_de_mu=subject_de_mu,
            subject_de_std=subject_de_std,
            subject_bio_mu=subject_bio_mu,
            subject_bio_std=subject_bio_std,
        )
        z = enc["z"]
        logits_2cls = self.logits_2cls_head(z)
        logits_4cls = self.logits_4cls_head(z)
        out = dict(enc)
        out.update(
            {
                "logits": logits_2cls,
                "logits_2cls": logits_2cls,
                "emotion_logits": logits_2cls,
                "logits_4cls": logits_4cls,
                "graph_feat": z,
            }
        )
        return out


def build_model_from_config(config: dict) -> DEThreeBranchEmotionClassifier:
    return DEThreeBranchEmotionClassifier(
        sfreq=float(config.get("sfreq", 250.0)),
        topk=int(config.get("topk", 8)),
        dropout=float(config.get("dropout", 0.2)),
        use_biomarkers=bool(config.get("use_biomarkers", True)),
        biomarker_dim=int(config.get("biomarker_dim", 57)),
        de_num_bands=int(config.get("de_num_bands", 5)),
        head_hidden_dim=int(config.get("head_hidden_dim", 128)),
    )


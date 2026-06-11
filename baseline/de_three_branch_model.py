# -*- coding: utf-8 -*-
"""DE-only three-branch baseline for emotion classification.

Input is precomputed differential entropy features with shape [B, C, K],
normally [B, 30, 5]. The main head predicts binary emotion and the auxiliary
head predicts the 4-class diagnosis-emotion label.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MultiScaleDEConvBranch(nn.Module):
    """Multi-scale 1D convolutions over the channel axis of DE features."""

    def __init__(self, num_bands: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(num_bands, hidden_dim, kernel_size=k, padding=k // 2)
                for k in (1, 3, 5)
            ]
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 6),
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, de_feat: torch.Tensor) -> torch.Tensor:
        x = de_feat.transpose(1, 2).contiguous()  # [B, K, C]
        pooled = []
        for conv in self.convs:
            h = F.gelu(conv(x))
            pooled.append(h.mean(dim=-1))
            pooled.append(h.amax(dim=-1))
        return self.proj(torch.cat(pooled, dim=-1))


class DEChannelRelationBranch(nn.Module):
    """Learned channel relation branch built from DE node embeddings."""

    def __init__(self, num_bands: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.LayerNorm(num_bands),
            nn.Linear(num_bands, hidden_dim),
            nn.GELU(),
        )
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, de_feat: torch.Tensor) -> torch.Tensor:
        node = self.node_proj(de_feat)  # [B, C, H]
        scale = max(node.size(-1), 1) ** 0.5
        attn = torch.softmax(torch.matmul(self.q(node), self.k(node).transpose(1, 2)) / scale, dim=-1)
        graph_node = torch.matmul(attn, self.v(node))
        readout = torch.cat([graph_node.mean(dim=1), graph_node.amax(dim=1)], dim=-1)
        return self.out(readout)


class DEStatBranch(nn.Module):
    """Global DE statistics and flattened channel-band descriptors."""

    def __init__(self, num_channels: int, num_bands: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        stat_dim = num_channels * num_bands + 2 * num_bands + num_channels + 4
        self.net = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, de_feat: torch.Tensor) -> torch.Tensor:
        flat = de_feat.flatten(start_dim=1)
        band_mean = de_feat.mean(dim=1)
        band_std = de_feat.std(dim=1, unbiased=False)
        channel_mean = de_feat.mean(dim=2)
        global_stats = torch.stack(
            [
                de_feat.mean(dim=(1, 2)),
                de_feat.std(dim=(1, 2), unbiased=False),
                de_feat.amin(dim=(1, 2)),
                de_feat.amax(dim=(1, 2)),
            ],
            dim=-1,
        )
        return self.net(torch.cat([flat, band_mean, band_std, channel_mean, global_stats], dim=-1))


class DEOnlyThreeBranchEncoder(nn.Module):
    """Three DE-only branches fused into one feature vector."""

    def __init__(
        self,
        num_channels: int = 30,
        num_bands: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_bands = int(num_bands)
        self.conv_branch = MultiScaleDEConvBranch(self.num_bands, hidden_dim, dropout)
        self.relation_branch = DEChannelRelationBranch(self.num_bands, hidden_dim, dropout)
        self.stat_branch = DEStatBranch(self.num_channels, self.num_bands, hidden_dim, dropout)
        self.out_dim = hidden_dim * 3
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.out_dim),
            nn.Linear(self.out_dim, self.out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _match_shape(self, de_feat: torch.Tensor) -> torch.Tensor:
        if de_feat.ndim != 3:
            raise ValueError(f"de_feat should be [B,C,K], got {tuple(de_feat.shape)}")
        if de_feat.size(1) != self.num_channels:
            raise ValueError(f"Expected C={self.num_channels}, got C={de_feat.size(1)}.")
        if de_feat.size(-1) == self.num_bands:
            return de_feat
        if de_feat.size(-1) > self.num_bands:
            return de_feat[..., : self.num_bands]
        pad = self.num_bands - de_feat.size(-1)
        return F.pad(de_feat, (0, pad), mode="constant", value=0.0)

    def forward(self, de_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        de_feat = torch.nan_to_num(self._match_shape(de_feat.float()), nan=0.0, posinf=0.0, neginf=0.0)
        conv_feat = self.conv_branch(de_feat)
        relation_feat = self.relation_branch(de_feat)
        stat_feat = self.stat_branch(de_feat)
        z = self.fusion(torch.cat([conv_feat, relation_feat, stat_feat], dim=-1))
        return {
            "z": z,
            "de_feat": de_feat,
            "de_conv_feat": conv_feat,
            "de_relation_feat": relation_feat,
            "de_stat_feat": stat_feat,
        }


class DEThreeBranchEmotionClassifier(nn.Module):
    """DE-only encoder with a primary 2-class emotion head and 4-class head."""

    def __init__(
        self,
        num_channels: int = 30,
        de_num_bands: int = 5,
        encoder_hidden_dim: int = 128,
        head_hidden_dim: int = 128,
        dropout: float = 0.2,
        sfreq: float = 250.0,
        topk: int = 8,
        use_biomarkers: bool = False,
        biomarker_dim: int = 57,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.de_num_bands = int(de_num_bands)
        self.sfreq = float(sfreq)
        self.topk = int(topk)
        self.use_biomarkers = bool(use_biomarkers)
        self.biomarker_dim = int(biomarker_dim)
        self.encoder = DEOnlyThreeBranchEncoder(
            num_channels=self.num_channels,
            num_bands=self.de_num_bands,
            hidden_dim=encoder_hidden_dim,
            dropout=dropout,
        )
        self.in_dim = int(self.encoder.out_dim)
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
        de_feat: torch.Tensor,
        maybe_de_feat: Optional[torch.Tensor] = None,
        **_: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if maybe_de_feat is not None:
            de_feat = maybe_de_feat
        enc = self.encoder(de_feat)
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
        num_channels=int(config.get("num_channels", 30)),
        de_num_bands=int(config.get("de_num_bands", 5)),
        encoder_hidden_dim=int(config.get("encoder_hidden_dim", config.get("head_hidden_dim", 128))),
        head_hidden_dim=int(config.get("head_hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.2)),
        sfreq=float(config.get("sfreq", 250.0)),
        topk=int(config.get("topk", 8)),
        use_biomarkers=bool(config.get("use_biomarkers", False)),
        biomarker_dim=int(config.get("biomarker_dim", 57)),
    )

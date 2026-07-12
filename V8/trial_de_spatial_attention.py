"""Trial-level DE temporal aggregation and channel-to-channel attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TrialDESpatialAttention(nn.Module):
    """Build one shared spatial attention matrix for every complete trial."""

    def __init__(self, de_bands: int = 5, hidden_dim: int = 32,
                 attention_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.de_bands = int(de_bands)
        self.attention_dim = int(attention_dim)
        self.de_projection = nn.Sequential(
            nn.LayerNorm(de_bands), nn.Linear(de_bands, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, attention_dim),
        )
        self.temporal_attention = nn.Linear(attention_dim, 1)
        self.query = nn.Linear(attention_dim, attention_dim)
        self.key = nn.Linear(attention_dim, attention_dim)

    def forward(self, de_feat: torch.Tensor, window_mask: torch.Tensor) -> dict:
        if de_feat.ndim != 4:
            raise ValueError(f"de_feat must be [B,T,N,{self.de_bands}], got {tuple(de_feat.shape)}")
        if de_feat.shape[-1] != self.de_bands or window_mask.shape != de_feat.shape[:2]:
            raise ValueError("DE bands or window_mask shape mismatch")
        mask = window_mask.bool()
        if (~mask.any(dim=1)).any():
            raise ValueError("each trial must contain at least one valid window")
        embedding = self.de_projection(torch.nan_to_num(de_feat, nan=0.0, posinf=0.0, neginf=0.0))
        logits = self.temporal_attention(embedding).squeeze(-1)  # [B,T,N]
        expanded_mask = mask.unsqueeze(-1)
        logits = logits.masked_fill(~expanded_mask, torch.finfo(logits.dtype).min)
        temporal = torch.softmax(logits, dim=1)
        temporal = torch.where(expanded_mask, temporal, torch.zeros_like(temporal))
        temporal = temporal / temporal.sum(dim=1, keepdim=True).clamp_min(1e-6)
        trial_embedding = (temporal.unsqueeze(-1) * embedding).sum(dim=1)
        q, k = self.query(trial_embedding), self.key(trial_embedding)
        spatial = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_dim)
        spatial = torch.softmax(spatial, dim=-1)
        spatial = 0.5 * (spatial + spatial.transpose(-1, -2))
        eye = torch.eye(spatial.shape[-1], device=spatial.device, dtype=torch.bool).unsqueeze(0)
        spatial = spatial.masked_fill(eye, 0.0)
        spatial = torch.nan_to_num(spatial, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        return {"spatial_attention": spatial, "temporal_de_attention": temporal,
                "trial_de_embedding": trial_embedding}

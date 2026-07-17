"""Masked non-causal temporal convolution and attention pooling."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class _ResidualTemporalBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd so temporal length is preserved")
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(self.activation(y))
        y = self.norm( y)
        return y * mask.unsqueeze(-1).to(y.dtype)


class TemporalTrialAggregator(nn.Module):
    """Aggregate ``[B,T,D]`` window features into one trial representation."""

    def __init__(
        self,
        input_dim: int = 185,
        hidden_dim: int = 128,
        output_dim: int = 185,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not dilations:
            raise ValueError("dilations cannot be empty")
        self.config = {
            "input_dim": int(input_dim), "hidden_dim": int(hidden_dim),
            "output_dim": int(output_dim), "kernel_size": int(kernel_size),
            "dilations": [int(d) for d in dilations], "dropout": float(dropout),
        }
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            _ResidualTemporalBlock(hidden_dim, kernel_size, int(d), dropout) for d in dilations
        ])
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, z_seq: torch.Tensor, window_mask: torch.Tensor):
        if z_seq.ndim != 3 or window_mask.shape != z_seq.shape[:2]:
            raise ValueError("expected z_seq [B,T,D] and window_mask [B,T]")
        mask = window_mask.bool()
        if (~mask.any(dim=1)).any():
            raise ValueError("every trial must contain at least one valid window")
        h = self.input_projection(z_seq) * mask.unsqueeze(-1).to(z_seq.dtype)
        for block in self.blocks:
            h = block(h, mask)
        score = self.attention(h).squeeze(-1).masked_fill(~mask, torch.finfo(h.dtype).min)
        weight = torch.softmax(score, dim=1)
        weight = weight.masked_fill(~mask, 0.0)
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = torch.sum(weight.unsqueeze(-1) * h, dim=1)
        return self.output_projection(pooled), weight, h


"""Graph construction and multi-order propagation layers for BF-GCN."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def normalize_adjacency(adjacency: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Symmetrize, constrain nonnegative, add self-loops, and D^-1/2 A D^-1/2 normalize."""
    if adjacency.ndim not in (2, 3) or adjacency.shape[-1] != adjacency.shape[-2]:
        raise ValueError(f"adjacency must be [N,N] or [B,N,N], got {tuple(adjacency.shape)}")
    adjacency = 0.5 * (adjacency + adjacency.transpose(-1, -2))
    adjacency = F.relu(adjacency)
    nodes = adjacency.shape[-1]
    eye = torch.eye(nodes, device=adjacency.device, dtype=adjacency.dtype)
    adjacency = adjacency + eye
    degree = adjacency.sum(dim=-1).clamp_min(eps)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt.unsqueeze(-1) * adjacency * inv_sqrt.unsqueeze(-2)


class LearnableAdjacency(nn.Module):
    """Global trainable graph whose device/dtype follows the module parameters."""

    def __init__(self, num_nodes: int) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        initial = torch.full((num_nodes, num_nodes), -2.0)
        initial.fill_diagonal_(0.0)
        self.raw = nn.Parameter(initial)

    def forward(self) -> torch.Tensor:
        symmetric = 0.5 * (self.raw + self.raw.transpose(0, 1))
        return normalize_adjacency(F.softplus(symmetric))


class MultiOrderGraphConv(nn.Module):
    """Learnable combination of I, A, ..., A^K neighborhood propagation.

    This is BF-GCN-style multi-order adjacency propagation, not the strict
    Chebyshev-Laplacian recurrence used by ChebNet.
    """

    def __init__(self, input_dim: int, output_dim: int, order: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        if order < 0:
            raise ValueError("order must be nonnegative")
        self.order = int(order)
        self.projections = nn.ModuleList(
            nn.Linear(input_dim, output_dim, bias=index == 0) for index in range(self.order + 1)
        )
        self.order_logits = nn.Parameter(torch.zeros(self.order + 1))
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _propagate(adjacency: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim == 2:
            return torch.einsum("nm,bmf->bnf", adjacency, features)
        return torch.bmm(adjacency, features)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Map ``features[B,N,F]`` with ``adjacency[N,N]`` or ``[B,N,N]``."""
        if features.ndim != 3:
            raise ValueError(f"features must be [B,N,F], got {tuple(features.shape)}")
        if adjacency.shape[-2:] != features.shape[1:2] * 2:
            expected = (features.shape[1], features.shape[1])
            raise ValueError(f"adjacency nodes {tuple(adjacency.shape[-2:])} != feature nodes {expected}")
        propagated = features
        outputs = [self.projections[0](propagated)]
        for index in range(1, self.order + 1):
            propagated = self._propagate(adjacency, propagated)
            outputs.append(self.projections[index](propagated))
        weights = torch.softmax(self.order_logits, dim=0)
        combined = sum(weights[index] * value for index, value in enumerate(outputs))
        return self.dropout(F.gelu(self.norm(combined)))


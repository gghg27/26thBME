"""Spatial-attention PLV fusion followed by an explicit order-2 Chebyshev block."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SpatialAttentionChebConv(nn.Module):
    """Chebyshev K=2 means T0, T1 and T2 (three independently transformed terms)."""

    cheb_order = 2
    cheb_num_terms = 3

    def __init__(self, in_dim: int = 64, out_dim: int = 64, dropout: float = 0.2,
                 gamma_init: float = 0.15, eps: float = 1e-6) -> None:
        super().__init__()
        if not 0 < gamma_init < 1:
            raise ValueError("gamma_init must be in (0, 1)")
        self.in_dim, self.out_dim, self.eps = int(in_dim), int(out_dim), float(eps)
        self.raw_gamma = nn.Parameter(torch.tensor(math.log(gamma_init / (1 - gamma_init))))
        self.linear0 = nn.Linear(in_dim, out_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.activation, self.dropout, self.norm = nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(out_dim)

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gamma)

    def forward(self, node_features: torch.Tensor, plv_adj: torch.Tensor,
                spatial_attention: torch.Tensor, return_graph_debug: bool = False) -> dict:
        if node_features.ndim != 3 or plv_adj.shape != spatial_attention.shape:
            raise ValueError("expected node_features [M,N,F] and matching graph tensors [M,N,N]")
        n = node_features.shape[1]
        plv = torch.nan_to_num(plv_adj, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        spatial = torch.nan_to_num(spatial_attention, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        attended = plv * (n * spatial)
        effective = (1 - self.gamma) * plv + self.gamma * attended
        effective = torch.nan_to_num(effective, nan=0.0, posinf=0.0, neginf=0.0)
        effective = 0.5 * (effective + effective.transpose(-1, -2)).clamp_min(0.0)
        eye = torch.eye(n, device=effective.device, dtype=effective.dtype).unsqueeze(0)
        effective = effective * (1 - eye) + eye
        degree = effective.sum(dim=-1).clamp_min(self.eps)
        inv_sqrt = degree.rsqrt()
        adjacency_norm = inv_sqrt.unsqueeze(-1) * effective * inv_sqrt.unsqueeze(-2)
        laplacian = eye - adjacency_norm
        # lambda_max=2: L_tilde = 2L/lambda_max - I = L - I.
        laplacian_tilde = 2.0 * laplacian / 2.0 - eye
        t0 = node_features
        t1 = torch.bmm(laplacian_tilde, t0)
        t2 = 2.0 * torch.bmm(laplacian_tilde, t1) - t0
        t0, t1, t2 = (torch.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
                      for term in (t0, t1, t2))
        output = self.linear0(t0) + self.linear1(t1) + self.linear2(t2)
        output = self.dropout(self.activation(output))
        if self.in_dim == self.out_dim:
            output = output + node_features
        output = self.norm(torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0))
        result = {"output": output, "gamma": self.gamma}
        if return_graph_debug:
            result.update(effective_adj=effective, laplacian=laplacian,
                          laplacian_tilde=laplacian_tilde, t0=t0, t1=t1, t2=t2)
        return result

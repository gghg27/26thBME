"""Modular BF-GCN implementation for five-band DE and PLV inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from V9.configs.bfgcn_config import BFGCNConfig
from V9.models.graph_layers import LearnableAdjacency, MultiOrderGraphConv, normalize_adjacency


class BFGCN(nn.Module):
    """Three-branch BF-GCN with configurable ablations."""

    def __init__(self, config: BFGCNConfig) -> None:
        super().__init__()
        self.config = config
        if config.graph_conv_type != "multi_order":
            raise NotImplementedError(
                f"graph_conv_type={config.graph_conv_type!r}; V9 currently implements 'multi_order' only"
            )
        if config.pool_type != "mean_max":
            raise NotImplementedError(f"pool_type={config.pool_type!r}; V9 implements 'mean_max'")
        if not (config.use_learnable_graph or config.use_functional_graph):
            raise ValueError("At least one of learnable/functional graph must be enabled")
        if config.use_common_branch and not (config.use_learnable_graph and config.use_functional_graph):
            raise ValueError("The common branch requires both learnable and functional graphs")

        bands = len(config.frequency_bands)
        self.learnable_graph = LearnableAdjacency(config.num_channels)
        self.band_attention_logits = nn.Parameter(torch.zeros(bands))
        self.band_attention_network = nn.Linear(bands, bands)
        conv_args = (bands, config.hidden_dim, config.graph_order, config.dropout)
        self.learnable_specific = MultiOrderGraphConv(*conv_args) if config.use_learnable_graph else None
        self.functional_specific = MultiOrderGraphConv(*conv_args) if config.use_functional_graph else None
        self.common_conv = MultiOrderGraphConv(*conv_args) if config.use_common_branch else None
        self.branch_score = nn.Sequential(
            nn.Linear(config.hidden_dim, max(config.hidden_dim // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(config.hidden_dim // 2, 1), 1, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.classifier_dim),
            nn.LayerNorm(config.classifier_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_dim, 2),
        )

    def _functional_graph(self, plv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if plv.ndim != 4:
            raise ValueError(f"plv must be [B,N,N,F], got {tuple(plv.shape)}")
        batch, nodes_a, nodes_b, bands = plv.shape
        expected = (self.config.num_channels, self.config.num_channels, len(self.config.frequency_bands))
        if (nodes_a, nodes_b, bands) != expected:
            raise ValueError(f"plv trailing dimensions must be {expected}, got {tuple(plv.shape[1:])}")
        if not torch.isfinite(plv).all():
            raise ValueError("plv contains NaN or Inf")
        if self.config.use_band_attention:
            # Sample-conditioned scores use the mean connectivity in each band,
            # with a global trainable bias retaining interpretable band priors.
            band_summary = plv.mean(dim=(1, 2))  # [B,F]
            logits = self.band_attention_network(band_summary) + self.band_attention_logits[:bands]
            band_attention = torch.softmax(logits, dim=-1)
        else:
            band_attention = plv.new_full((batch, bands), 1.0 / bands)
        functional = torch.einsum("bijf,bf->bij", plv, band_attention)
        return normalize_adjacency(functional), band_attention

    def forward(self, de: torch.Tensor, plv: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Run BF-GCN: DE ``[B,N,F]`` and PLV ``[B,N,N,F]`` to binary logits."""
        if de.ndim != 3:
            raise ValueError(f"de must be [B,N,F], got {tuple(de.shape)}")
        expected_de = (self.config.num_channels, len(self.config.frequency_bands))
        if tuple(de.shape[1:]) != expected_de:
            raise ValueError(f"de trailing dimensions must be {expected_de}, got {tuple(de.shape[1:])}")
        if not torch.isfinite(de).all():
            raise ValueError("de contains NaN or Inf")
        if plv.device != de.device or plv.dtype != de.dtype:
            plv = plv.to(device=de.device, dtype=de.dtype)

        learnable_adj = self.learnable_graph() if self.config.use_learnable_graph else None
        functional_adj: torch.Tensor | None = None
        band_attention: torch.Tensor | None = None
        if self.config.use_functional_graph:
            functional_adj, band_attention = self._functional_graph(plv)

        branches: list[torch.Tensor] = []
        if self.learnable_specific is not None and learnable_adj is not None:
            branches.append(self.learnable_specific(de, learnable_adj))
        if self.functional_specific is not None and functional_adj is not None:
            branches.append(self.functional_specific(de, functional_adj))
        if self.common_conv is not None and learnable_adj is not None and functional_adj is not None:
            common_learnable = self.common_conv(de, learnable_adj)
            common_functional = self.common_conv(de, functional_adj)
            branches.append(0.5 * (common_learnable + common_functional))
        if not branches:
            raise RuntimeError("No graph branch produced an output")

        # [B,R,N,H], where R dynamically follows enabled branches.
        stacked = torch.stack(branches, dim=1)
        if self.config.use_branch_attention and len(branches) > 1:
            scores = self.branch_score(stacked).squeeze(-1)  # [B,R,N]
            branch_attention = torch.softmax(scores, dim=1)
        else:
            branch_attention = stacked.new_full(stacked.shape[:3], 1.0 / len(branches))
        node_feature = (stacked * branch_attention.unsqueeze(-1)).sum(dim=1)  # [B,N,H]
        mean_feature = node_feature.mean(dim=1)
        max_feature = node_feature.max(dim=1).values
        graph_feature = torch.cat((mean_feature, max_feature), dim=-1)  # [B,2H]
        emotion_logits = self.classifier(graph_feature)  # [B,2]
        return {
            "emotion_logits": emotion_logits,
            "graph_feature": graph_feature,
            "node_feature": node_feature,
            "learnable_adj": learnable_adj,
            "functional_adj": functional_adj,
            "band_attention": band_attention,
            "branch_attention": branch_attention,
        }


@torch.no_grad()
def extract_interpretability_outputs(
    model: BFGCN,
    batch: dict[str, Any],
    output_path: str | Path,
    device: torch.device | str | None = None,
) -> Path:
    """Save graph/attention outputs for one batch to a compressed ``.npz`` file."""
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    outputs = model(batch["de"].to(target_device), batch["plv"].to(target_device))
    probabilities = torch.softmax(outputs["emotion_logits"], dim=-1)

    def array(name: str) -> np.ndarray:
        value = outputs[name]
        return np.asarray([]) if value is None else value.detach().cpu().numpy()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        learnable_adj=array("learnable_adj"),
        functional_adj=array("functional_adj"),
        band_attention=array("band_attention"),
        branch_attention=array("branch_attention"),
        node_feature=array("node_feature"),
        graph_feature=array("graph_feature"),
        emotion_prob=probabilities.cpu().numpy(),
        subject_id=np.asarray(batch["subject_id"]),
        trial_id=np.asarray(batch["trial_id"].cpu()),
    )
    return path

"""CPU forward/backward coverage for full BF-GCN and required ablations."""

import pytest
import torch

from V9.configs.bfgcn_config import BFGCNConfig
from V9.models.bfgcn import BFGCN


@pytest.mark.parametrize(
    ("learnable", "functional", "common", "branches"),
    [(True, False, False, 1), (False, True, False, 1), (True, True, False, 2), (True, True, True, 3)],
)
def test_required_ablation_shapes(
    learnable: bool, functional: bool, common: bool, branches: int
) -> None:
    config = BFGCNConfig(
        use_learnable_graph=learnable,
        use_functional_graph=functional,
        use_common_branch=common,
        hidden_dim=16,
        classifier_dim=8,
    )
    model = BFGCN(config)
    de = torch.randn(2, config.num_channels, len(config.frequency_bands))
    raw_plv = torch.rand(2, config.num_channels, config.num_channels, len(config.frequency_bands))
    plv = 0.5 * (raw_plv + raw_plv.transpose(1, 2))
    outputs = model(de, plv)
    assert outputs["emotion_logits"].shape == (2, 2)
    assert outputs["node_feature"].shape == (2, config.num_channels, config.hidden_dim)
    assert outputs["graph_feature"].shape == (2, config.hidden_dim * 2)
    assert outputs["branch_attention"].shape == (2, branches, config.num_channels)
    outputs["emotion_logits"].sum().backward()

# -*- coding: utf-8 -*-
"""
SSAS (Source Selection and Adaptation) 框架适配版 HC 情绪模型。

基于 hc_trail_con.py 的 BrainGraphBackbone，加入:
    - DomainDiscriminator: 带 GRL 的域判别器
    - SourceSelectionModel: Stage 1 源选择模型
    - SSASEmotionModel: Stage 2/3 最终模型（情绪主任务 + 对比学习 + 域对抗）
    - mmd_loss: 多核 MMD 分布距离 loss

训练脚本: trainers/train_hc_ssas.py
"""

import math
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入共用 backbone 和基础组件
from models.hc_trail_con import (
    BrainGraphBackbone,
    ClassificationHead,
    ContrastProjectionHead,
)

# 从 diag_model_ssas 复用 GradReverse
from models.diag_model_ssas import GradReverse


# =========================================================
# Domain discriminator
# =========================================================

class DomainDiscriminator(nn.Module):
    """SSAS 域判别器: 小 MLP + 可选的 GRL。"""

    def __init__(
        self,
        in_dim: int,
        num_domains: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, feat: torch.Tensor, lambda_grl: float = 0.0) -> torch.Tensor:
        if lambda_grl and lambda_grl > 0:
            feat = GradReverse.apply(feat, lambda_grl)
        return self.net(feat)


# =========================================================
# Stage 1: Source Selection Model
# =========================================================

class SourceSelectionModel(nn.Module):
    """
    Stage 1 源选择模型:
        F_ss = BrainGraphBackbone
        C_ss = emotion classifier (2-class: neutral/positive)
        D_ss = multi-source + target domain classifier (K+1 outputs)

    D_ss 输出:
        0..K-1: 源域被试
        K: 目标域
    """

    def __init__(
        self,
        num_source_domains: int,
        sfreq: float = 250.0,
        topk: int = 6,
        dropout: float = 0.2,
        emotion_classes: int = 2,
        domain_hidden_dim: int = 128,
        use_biomarkers: bool = True,
        biomarker_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_source_domains = int(num_source_domains)
        self.num_domains = self.num_source_domains + 1
        self.target_domain_id = self.num_source_domains

        self.backbone = BrainGraphBackbone(
            sfreq=sfreq,
            node_dim=64,
            attn_dim=16,
            topk=topk,
            prior_matrix=None,
            dropout=dropout,
            use_biomarkers=use_biomarkers,
            biomarker_dim=biomarker_dim,
        )
        self.in_dim = self.backbone.out_dim
        self.classifier = ClassificationHead(
            in_dim=self.in_dim,
            num_classes=emotion_classes,
            hidden_dim=64,
            dropout=dropout,
        )
        self.domain_discriminator = DomainDiscriminator(
            in_dim=self.in_dim,
            num_domains=self.num_domains,
            hidden_dim=domain_hidden_dim,
            dropout=dropout,
        )

    def extract_features(self, x: torch.Tensor, de_feat: torch.Tensor) -> torch.Tensor:
        return self.backbone(x, de_feat=de_feat)["z"]

    def forward(
        self,
        x: torch.Tensor,
        de_feat: torch.Tensor,
        lambda_domain: float = 0.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        out = self.backbone(x, de_feat=de_feat)
        z = out["z"]
        emo_logits = self.classifier(z)
        domain_logits = self.domain_discriminator(z, lambda_grl=lambda_domain)
        out["z"] = z
        out["graph_feat"] = z
        out["emo_logits"] = emo_logits
        out["logits"] = emo_logits            # 兼容旧接口
        out["domain_logits"] = domain_logits
        return out


# =========================================================
# Stage 2/3: SSAS Emotion Model
# =========================================================

class SSASEmotionModel(nn.Module):
    """
    Stage 2/3 最终模型:
        F = BrainGraphBackbone
        C_emo = emotion classifier (2-class, no GRL)
        C_contrast = contrast projector (for trial compactness)
        D_as = binary source-vs-target domain discriminator (with GRL)
    """

    def __init__(
        self,
        sfreq: float = 250.0,
        topk: int = 6,
        dropout: float = 0.2,
        emotion_classes: int = 2,
        contrast_dim: int = 64,
        contrast_hidden_dim: int = 128,
        domain_hidden_dim: int = 128,
        use_biomarkers: bool = True,
        biomarker_dim: int = 64,
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
        )
        self.in_dim = self.backbone.out_dim

        self.emotion_head = ClassificationHead(
            in_dim=self.in_dim,
            num_classes=emotion_classes,
            hidden_dim=64,
            dropout=dropout,
        )

        self.contrast_projector = ContrastProjectionHead(
            in_dim=self.in_dim,
            hidden_dim=contrast_hidden_dim,
            out_dim=contrast_dim,
            dropout=dropout,
        )

        self.domain_discriminator = DomainDiscriminator(
            in_dim=self.in_dim,
            num_domains=2,   # binary: source vs target
            hidden_dim=domain_hidden_dim,
            dropout=dropout,
        )

    def extract_features(self, x: torch.Tensor, de_feat: torch.Tensor) -> torch.Tensor:
        return self.backbone(x, de_feat=de_feat)["z"]

    def forward(
        self,
        x: torch.Tensor,
        de_feat: torch.Tensor,
        lambda_domain: float = 0.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        out = self.backbone(x, de_feat=de_feat)
        z = out["z"]

        emo_logits = self.emotion_head(z)
        contrast_feat = self.contrast_projector(z)
        domain_logits = self.domain_discriminator(z, lambda_grl=lambda_domain)

        out["emo_logits"] = emo_logits
        out["logits"] = emo_logits             # 兼容
        out["contrast_feat"] = contrast_feat    # trial compactness 用
        out["domain_logits"] = domain_logits
        out["graph_feat"] = z
        return out


# =========================================================
# MMD loss
# =========================================================

def gaussian_kernel(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None,
) -> torch.Tensor:
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)
    l2_distance = ((total0 - total1) ** 2).sum(dim=2)

    if fix_sigma is None:
        n_samples = total.size(0)
        bandwidth = l2_distance.detach().sum() / max(n_samples * n_samples - n_samples, 1)
    else:
        bandwidth = torch.as_tensor(fix_sigma, dtype=total.dtype, device=total.device)

    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    kernels = 0.0
    for i in range(kernel_num):
        kernels = kernels + torch.exp(-l2_distance / (bandwidth * (kernel_mul ** i)).clamp_min(1e-6))
    return kernels


def mmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None,
) -> torch.Tensor:
    if source.size(0) == 0 or target.size(0) == 0:
        return source.new_tensor(0.0)
    kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    n_source = source.size(0)
    xx = kernels[:n_source, :n_source].mean()
    yy = kernels[n_source:, n_source:].mean()
    xy = kernels[:n_source, n_source:].mean()
    yx = kernels[n_source:, :n_source].mean()
    return xx + yy - xy - yx

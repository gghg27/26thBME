# -*- coding: utf-8 -*-
"""Two-stage SSAS models with the existing three-branch EEG encoder.

The shared encoder is the BrainGraphBackbone from models.dep_contrast_bio:
multi-scale temporal convolution branch + PLV graph branch + biomarker branch.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# from models.dep_contrast_bio import BrainGraphBackbone
from pmg_gat_backbone import BrainGraphBackbone

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_grl: float):
        ctx.lambda_grl = float(lambda_grl)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_grl * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_grl: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambda_grl)


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


class MMDHead(nn.Module):
    """Projection head for source-target MMD."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MultiDomainHead(nn.Module):
    """Multi-domain classifier. Set lambda_grl > 0 to adversarially train encoder."""

    def __init__(
        self,
        in_dim: int,
        num_domains: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_domains = int(num_domains)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_domains),
        )

    def forward(self, z: torch.Tensor, lambda_grl: float = 0.0) -> torch.Tensor:
        if lambda_grl > 0:
            z = grad_reverse(z, lambda_grl)
        return self.net(z)


class ThreeBranchEncoderWrapper(nn.Module):
    """Unifies the existing three-branch backbone behind forward(x, de_feat)."""

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


class Stage1SSASSourceSelectionModel(nn.Module):
    """Stage 1: domain-focused source selection model."""

    def __init__(
        self,
        num_domains: int,
        sfreq: float = 250.0,
        topk: int = 8,
        dropout: float = 0.2,
        prior_matrix: Optional[torch.Tensor] = None,
        emotion_classes: int = 2,
        diagnosis_classes: int = 2,
        domain_hidden_dim: int = 128,
        mmd_hidden_dim: int = 128,
        mmd_dim: int = 64,
        use_biomarkers: bool = True,
        biomarker_dim: int = 57,
        use_subject_relative_de: bool = False,
        use_subject_relative_bio: bool = False,
        bio_abs_scale: float = 0.3,
        relative_eps: float = 1e-6,
        de_num_bands: int = 5,
    ) -> None:
        super().__init__()
        self.use_subject_relative_de = bool(use_subject_relative_de)
        self.use_subject_relative_bio = bool(use_subject_relative_bio)
        self.bio_abs_scale = float(bio_abs_scale)
        self.relative_eps = float(relative_eps)
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
        self.domain_head = MultiDomainHead(
            in_dim=in_dim,
            num_domains=num_domains,
            hidden_dim=domain_hidden_dim,
            dropout=dropout,
        )
        self.emotion_head = MLPHead(
            in_dim=in_dim,
            out_dim=emotion_classes,
            hidden_dim=64,
            dropout=dropout,
        )
        self.diagnosis_head = MLPHead(
            in_dim=in_dim,
            out_dim=diagnosis_classes,
            hidden_dim=64,
            dropout=dropout,
        )
        self.mmd_head = MMDHead(
            in_dim=in_dim,
            hidden_dim=mmd_hidden_dim,
            out_dim=mmd_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        de_feat: torch.Tensor,
        lambda_emo: float = 0.0,
        lambda_diag: float = 0.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        enc = self.shared_encoder(x, de_feat=de_feat, **kwargs)
        z = enc["z"]
        z_emo = grad_reverse(z, lambda_emo) if lambda_emo > 0 else z
        z_diag = grad_reverse(z, lambda_diag) if lambda_diag > 0 else z

        out = dict(enc)
        out.update(
            {
                "z": z,
                "z_mmd": self.mmd_head(z),
                "domain_logits": self.domain_head(z, lambda_grl=0.0),
                "emotion_logits_grl": self.emotion_head(z_emo),
                "diagnosis_logits_grl": self.diagnosis_head(z_diag),
            }
        )
        return out


class Stage2ExpertEmotionAdaptationModel(nn.Module):
    """Stage 2: diagnosis router + HC/DEP emotion experts + target adaptation."""

    def __init__(
        self,
        num_domains: int,
        sfreq: float = 250.0,
        topk: int = 8,
        dropout: float = 0.2,
        prior_matrix: Optional[torch.Tensor] = None,
        emotion_classes: int = 2,
        diagnosis_classes: int = 2,
        domain_hidden_dim: int = 128,
        mmd_hidden_dim: int = 128,
        mmd_dim: int = 64,
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
        self.mmd_head = MMDHead(
            in_dim=in_dim,
            hidden_dim=mmd_hidden_dim,
            out_dim=mmd_dim,
            dropout=dropout,
        )
        self.subject_domain_head = MultiDomainHead(
            in_dim=in_dim,
            num_domains=num_domains,
            hidden_dim=domain_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        de_feat: torch.Tensor,
        lambda_subject: float = 0.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        enc = self.shared_encoder(x, de_feat=de_feat, **kwargs)
        z = enc["z"]

        diag_logits = self.diagnosis_router(z)
        shared_logits = self.shared_emotion_head(z)
        hc_logits = self.hc_emotion_expert(z)
        dep_logits = self.dep_emotion_expert(z)

        diag_prob = torch.softmax(diag_logits, dim=1)
        prob_shared = torch.softmax(shared_logits, dim=1)
        prob_dep = torch.softmax(dep_logits, dim=1)
        prob_hc = torch.softmax(hc_logits, dim=1)
        p_dep = diag_prob[:, 0:1]
        p_hc = diag_prob[:, 1:2]
        expert_mix_prob = p_dep * prob_dep + p_hc * prob_hc
        expert_mix_prob = expert_mix_prob / expert_mix_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mix_prob = self.shared_mix_alpha * prob_shared + (1.0 - self.shared_mix_alpha) * expert_mix_prob
        mix_prob = mix_prob / mix_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)

        out = dict(enc)
        out.update(
            {
                "z": z,
                "z_mmd": self.mmd_head(z),
                "diag_logits": diag_logits,
                "shared_logits": shared_logits,
                "hc_logits": hc_logits,
                "dep_logits": dep_logits,
                "mix_prob": mix_prob,
                "expert_mix_prob": expert_mix_prob,
                "subject_domain_logits": self.subject_domain_head(
                    z,
                    lambda_grl=lambda_subject,
                ),
                "prob_shared": prob_shared,
                "prob_hc": prob_hc,
                "prob_dep": prob_dep,
                "diag_prob": diag_prob,
                "logits": mix_prob,
            }
        )
        return out


def mixture_emotion_nll_loss(
    mix_prob: torch.Tensor,
    y_emo: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    y_emo = y_emo.long()
    p_true = mix_prob[torch.arange(mix_prob.size(0), device=mix_prob.device), y_emo]
    loss = -torch.log(p_true.clamp_min(eps))
    if sample_weight is not None:
        loss = loss * sample_weight.to(device=loss.device, dtype=loss.dtype)
    return loss.mean()


def hard_expert_emotion_loss(
    hc_logits: torch.Tensor,
    dep_logits: torch.Tensor,
    y_emo: torch.Tensor,
    y_diag: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
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

    ce = F.cross_entropy(selected_logits, y_emo, reduction="none")
    if sample_weight is not None:
        ce = ce * sample_weight.to(device=ce.device, dtype=ce.dtype)
    return ce.mean()


def _rbf_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None,
) -> torch.Tensor:
    total = torch.cat([x, y], dim=0)
    l2 = torch.cdist(total, total, p=2).pow(2)
    if fix_sigma is None:
        n = total.size(0)
        bandwidth = l2.detach().sum() / max(n * n - n, 1)
    else:
        bandwidth = torch.as_tensor(fix_sigma, device=total.device, dtype=total.dtype)
    bandwidth = bandwidth.clamp_min(1e-6)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    kernels = 0.0
    for i in range(kernel_num):
        kernels = kernels + torch.exp(-l2 / (bandwidth * (kernel_mul ** i)).clamp_min(1e-6))
    return kernels


def weighted_mmd_rbf(
    z_source: torch.Tensor,
    z_target: torch.Tensor,
    source_weight: Optional[torch.Tensor] = None,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None,
) -> torch.Tensor:
    """Global RBF MMD with optional source sample weights."""

    if z_source.size(0) == 0 or z_target.size(0) == 0:
        return z_source.new_tensor(0.0)

    kernels = _rbf_kernel(
        z_source,
        z_target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    ns = z_source.size(0)
    nt = z_target.size(0)
    k_ss = kernels[:ns, :ns]
    k_tt = kernels[ns:, ns:]
    k_st = kernels[:ns, ns:]

    if source_weight is None:
        ws = torch.full((ns,), 1.0 / ns, device=z_source.device, dtype=z_source.dtype)
    else:
        ws = source_weight.to(device=z_source.device, dtype=z_source.dtype).flatten()
        ws = ws.clamp_min(0.0)
        ws = ws / ws.sum().clamp_min(1e-8)
    wt = torch.full((nt,), 1.0 / nt, device=z_target.device, dtype=z_target.dtype)

    loss_ss = (ws[:, None] * ws[None, :] * k_ss).sum()
    loss_tt = (wt[:, None] * wt[None, :] * k_tt).sum()
    loss_st = (ws[:, None] * wt[None, :] * k_st).sum()
    return loss_ss + loss_tt - 2.0 * loss_st


def target_entropy_loss(mix_prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    entropy = -(mix_prob.clamp_min(eps) * torch.log(mix_prob.clamp_min(eps))).sum(dim=1)
    return entropy.mean()

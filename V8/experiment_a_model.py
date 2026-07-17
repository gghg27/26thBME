"""Trial-level two-stage SSAS models for Experiment A."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from V8.de_plv_graph_backbone import DEPLVGraphBackbone
from V8.temporal_aggregator import TemporalTrialAggregator


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.scale * grad, None


def grad_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradReverse.apply(x, scale)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiDomainHead(MLPHead):
    def forward(self, x: torch.Tensor, lambda_grl: float = 0.0) -> torch.Tensor:
        return super().forward(grad_reverse(x, lambda_grl) if lambda_grl > 0 else x)


class MMDHead(MLPHead):
    pass


class TrialEncoder(nn.Module):
    """Encode valid windows only, scatter them, then aggregate emotion/diagnosis separately."""

    def __init__(
        self,
        dropout: float = 0.45,
        num_nodes: int = 30,
        graph_hidden_dim: int = 64,
        band_embed_dim: int = 64,
        window_embed_dim: int = 128,
        cheb_order: int = 3,
        num_graph_layers: int = 2,
        share_band_encoder: bool = True,
        graph_dropout: float = 0.3,
        use_subject_relative_de: bool = True,
        relative_eps: float = 1e-6,
        de_num_bands: int = 5,
        temporal_hidden_dim: int = 128,
        temporal_kernel_size: int = 3,
        temporal_dilations=(1, 2),
        temporal_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = DEPLVGraphBackbone(
            num_nodes=num_nodes, num_bands=de_num_bands,
            graph_hidden_dim=graph_hidden_dim, band_embed_dim=band_embed_dim,
            window_embed_dim=window_embed_dim, cheb_order=cheb_order,
            num_graph_layers=num_graph_layers, dropout=graph_dropout,
            use_subject_relative_de=use_subject_relative_de, relative_eps=relative_eps,
            share_band_encoder=share_band_encoder,
        )
        self.out_dim = int(self.backbone.out_dim)
        temporal_kwargs = dict(
            input_dim=self.out_dim, hidden_dim=temporal_hidden_dim, output_dim=self.out_dim,
            kernel_size=temporal_kernel_size, dilations=temporal_dilations, dropout=temporal_dropout,
        )
        self.emotion_temporal_aggregator = TemporalTrialAggregator(**temporal_kwargs)
        self.diagnosis_temporal_aggregator = TemporalTrialAggregator(**temporal_kwargs)
        self.temporal_config = self.emotion_temporal_aggregator.config.copy()

    @staticmethod
    def _expand_trial_kwarg(value, mask: torch.Tensor):
        if not torch.is_tensor(value) or value.shape[0] != mask.shape[0]:
            return value
        expanded = value.unsqueeze(1).expand(mask.shape[0], mask.shape[1], *value.shape[1:])
        return expanded[mask]

    def forward(self, de_feat: torch.Tensor, plv_feat: torch.Tensor,
                window_mask: torch.Tensor, **kwargs) -> dict:
        if de_feat.ndim != 4:
            raise ValueError(f"TrialEncoder expects de_feat [B,T,N,5], got {tuple(de_feat.shape)}")
        if plv_feat.ndim != 5:
            raise ValueError(f"TrialEncoder expects plv_feat [B,T,5,N,N], got {tuple(plv_feat.shape)}")
        mask = window_mask.bool()
        if mask.shape != de_feat.shape[:2] or mask.shape != plv_feat.shape[:2]:
            raise ValueError(
                f"trial inputs disagree: de={tuple(de_feat.shape)}, plv={tuple(plv_feat.shape)}, "
                f"mask={tuple(mask.shape)}"
            )
        if (~mask.any(dim=1)).any():
            raise ValueError("every trial must contain at least one valid window")
        window_kwargs = {key: self._expand_trial_kwarg(value, mask) for key, value in kwargs.items()}
        enc = self.backbone(de_feat=de_feat[mask], plv_feat=plv_feat[mask], **window_kwargs)
        z_emo_valid = enc.get("z_emotion", enc["z"])
        z_diag_valid = enc.get("z_diag", enc["z"])
        z_emotion_seq = z_emo_valid.new_zeros((*mask.shape, z_emo_valid.shape[-1]))
        z_diag_seq = z_diag_valid.new_zeros((*mask.shape, z_diag_valid.shape[-1]))
        z_emotion_seq[mask] = z_emo_valid
        z_diag_seq[mask] = z_diag_valid
        z_end_emotion, attn_emo, temporal_emo = self.emotion_temporal_aggregator(z_emotion_seq, mask)
        z_end_diag, attn_diag, temporal_diag = self.diagnosis_temporal_aggregator(z_diag_seq, mask)
        out = {
            "z": z_end_emotion, "z_end": z_end_emotion,
            "z_emotion": z_end_emotion, "z_emo": z_end_emotion, "z_diag": z_end_diag,
            "z_end_emotion": z_end_emotion, "z_end_diag": z_end_diag,
            "z_emotion_seq": z_emotion_seq, "z_diag_seq": z_diag_seq,
            "temporal_attention_emotion": attn_emo, "temporal_attention_diag": attn_diag,
            "temporal_features_emotion": temporal_emo, "temporal_features_diag": temporal_diag,
            "window_mask": mask,
        }
        # Retain valid-window diagnostic features without pretending they are trial tensors.
        for key, value in enc.items():
            if key not in out and key not in {"z", "z_emotion", "z_diag"}:
                out[f"window_{key}"] = value
        return out


class _BaseTrialSSAS(nn.Module):
    def __init__(self, **encoder_kwargs) -> None:
        super().__init__()
        self.shared_encoder = TrialEncoder(**encoder_kwargs)
        self.in_dim = self.shared_encoder.out_dim

    @property
    def temporal_config(self) -> dict:
        return self.shared_encoder.temporal_config


class Stage1SSASSourceSelectionModel(_BaseTrialSSAS):
    def __init__(self, num_domains: int, emotion_classes: int = 2, diagnosis_classes: int = 2,
                 domain_hidden_dim: int = 128, mmd_hidden_dim: int = 128, mmd_dim: int = 64, **kwargs) -> None:
        super().__init__(**kwargs)
        self.domain_head = MultiDomainHead(self.in_dim, num_domains, domain_hidden_dim, kwargs.get("dropout", .2))
        self.emotion_head = MLPHead(self.in_dim, emotion_classes, 64, kwargs.get("dropout", .2))
        self.diagnosis_head = MLPHead(self.in_dim, diagnosis_classes, 64, kwargs.get("dropout", .2))
        self.mmd_head = MMDHead(self.in_dim, mmd_dim, mmd_hidden_dim, kwargs.get("dropout", .2))

    def forward(self, de_feat, plv_feat, window_mask, lambda_emo: float = 0.0,
                lambda_diag: float = 0.0, **kwargs):
        out = self.shared_encoder(de_feat, plv_feat, window_mask, **kwargs)
        z_emo, z_diag = out["z_end_emotion"], out["z_end_diag"]
        out.update({
            "z_mmd": self.mmd_head(z_emo), "domain_logits": self.domain_head(z_emo),
            "emotion_logits_grl": self.emotion_head(grad_reverse(z_emo, lambda_emo) if lambda_emo > 0 else z_emo),
            "diagnosis_logits_grl": self.diagnosis_head(grad_reverse(z_diag, lambda_diag) if lambda_diag > 0 else z_diag),
        })
        out["emotion_logits"] = out["emotion_logits_grl"]
        out["diagnosis_logits"] = out["diagnosis_logits_grl"]
        return out


class Stage2ExpertEmotionAdaptationModel(_BaseTrialSSAS):
    def __init__(self, num_domains: int, emotion_classes: int = 2, diagnosis_classes: int = 2,
                 domain_hidden_dim: int = 128, mmd_hidden_dim: int = 128, mmd_dim: int = 64,
                 shared_mix_alpha: float = 0.7, **kwargs) -> None:
        super().__init__(**kwargs)
        dropout = kwargs.get("dropout", .2)
        self.shared_mix_alpha = float(min(1.0, max(0.0, shared_mix_alpha)))
        self.diagnosis_router = MLPHead(self.in_dim, diagnosis_classes, 64, dropout)
        self.shared_emotion_head = MLPHead(self.in_dim, emotion_classes, 64, dropout)
        self.hc_emotion_expert = MLPHead(self.in_dim, emotion_classes, 64, dropout)
        self.dep_emotion_expert = MLPHead(self.in_dim, emotion_classes, 64, dropout)
        self.mmd_head = MMDHead(self.in_dim, mmd_dim, mmd_hidden_dim, dropout)
        self.subject_domain_head = MultiDomainHead(self.in_dim, num_domains, domain_hidden_dim, dropout)

    def forward(self, de_feat, plv_feat, window_mask, lambda_subject: float = 0.0, **kwargs):
        out = self.shared_encoder(de_feat, plv_feat, window_mask, **kwargs)
        z_emo, z_diag = out["z_end_emotion"], out["z_end_diag"]
        diag_logits = self.diagnosis_router(z_diag)
        shared_logits = self.shared_emotion_head(z_emo)
        hc_logits, dep_logits = self.hc_emotion_expert(z_emo), self.dep_emotion_expert(z_emo)
        diag_prob = torch.softmax(diag_logits, 1)
        prob_shared, prob_hc, prob_dep = map(lambda q: torch.softmax(q, 1), (shared_logits, hc_logits, dep_logits))
        expert_mix_prob = diag_prob[:, 0:1] * prob_dep + diag_prob[:, 1:2] * prob_hc
        mix_prob = self.shared_mix_alpha * prob_shared + (1 - self.shared_mix_alpha) * expert_mix_prob
        mix_prob = mix_prob / mix_prob.sum(1, keepdim=True).clamp_min(1e-8)
        out.update({
            "z_mmd": self.mmd_head(z_emo), "diag_logits": diag_logits, "diagnosis_logits": diag_logits,
            "shared_logits": shared_logits, "hc_logits": hc_logits, "dep_logits": dep_logits,
            "mix_prob": mix_prob, "expert_mix_prob": expert_mix_prob,
            "subject_domain_logits": self.subject_domain_head(z_emo, lambda_subject),
            "prob_shared": prob_shared, "prob_hc": prob_hc, "prob_dep": prob_dep, "diag_prob": diag_prob,
            "logits": mix_prob, "emotion_logits": torch.log(mix_prob.clamp_min(1e-8)),
        })
        return out


def mixture_emotion_nll_loss(mix_prob, labels, sample_weight=None):
    loss = -torch.log(mix_prob[torch.arange(len(labels), device=labels.device), labels.long()].clamp_min(1e-8))
    return (loss * sample_weight).mean() if sample_weight is not None else loss.mean()


def hard_expert_emotion_loss(hc_logits, dep_logits, emotion_labels, diagnosis_labels, sample_weight=None):
    chosen = torch.where((diagnosis_labels.long() == 1).unsqueeze(1), hc_logits, dep_logits)
    loss = F.cross_entropy(chosen, emotion_labels.long(), reduction="none")
    return (loss * sample_weight).mean() if sample_weight is not None else loss.mean()


def _rbf(x, y):
    total = torch.cat((x, y)); dist = torch.cdist(total, total).square()
    n = total.shape[0]; bandwidth = (dist.detach().sum() / max(n * n - n, 1)).clamp_min(1e-6)
    return sum(torch.exp(-dist / (bandwidth / 4 * (2 ** i)).clamp_min(1e-6)) for i in range(5))


def weighted_mmd_rbf(source, target, source_weight=None):
    if source.numel() == 0 or target.numel() == 0:
        return source.new_tensor(0.0)
    kernel = _rbf(source, target); ns, nt = len(source), len(target)
    ws = torch.ones(ns, device=source.device) if source_weight is None else source_weight.clamp_min(0)
    ws = ws / ws.sum().clamp_min(1e-8); wt = torch.full((nt,), 1 / nt, device=target.device)
    return ((ws[:, None] * ws) * kernel[:ns, :ns]).sum() + ((wt[:, None] * wt) * kernel[ns:, ns:]).sum() - 2 * ((ws[:, None] * wt) * kernel[:ns, ns:]).sum()


def target_entropy_loss(prob):
    return -(prob.clamp_min(1e-8) * torch.log(prob.clamp_min(1e-8))).sum(1).mean()

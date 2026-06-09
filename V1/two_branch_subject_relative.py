# -*- coding: utf-8 -*-
# 版本A：TSception temporal branch 替换构图前通道内特征提取器
# 在v2上加对比学习

# 将构图改为映射为30维向量

# 原网络，特征构图，被试域对抗
# 多尺度卷积


import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# 从主模型文件导入 biomarker 提取器（避免代码重复）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from models.diag_model_ssas import DepressionBiomarkerExtractor, _safe_std


# =========================================================
# Utility functions
# =========================================================

class SubjectRelativeDEAdapter(nn.Module):
    """
    Map subject-relative DE input back to the original band dimension.

    Input shape:  [B, C, 3*K] for concat(de_feat, de_rel, de_z)
    Output shape: [B, C, K]
    """

    def __init__(self, in_bands: int, out_bands: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_bands),
            nn.Linear(in_bands, out_bands),
        )

    def forward(self, de_input: torch.Tensor) -> torch.Tensor:
        return self.net(de_input)


class SubjectRelativeBiomarkerExtractor(DepressionBiomarkerExtractor):
    """
    DepressionBiomarkerExtractor with an optional subject-relative bio_raw path.

    If subject_bio_mu/std are absent, the forward path is exactly the original
    projector(norm(bio_raw)) behavior.
    """

    def __init__(
            self,
            *args,
            use_subject_relative_bio: bool = False,
            bio_abs_scale: float = 0.3,
            relative_eps: float = 1e-6,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_subject_relative_bio = bool(use_subject_relative_bio)
        self.bio_abs_scale = float(bio_abs_scale)
        self.relative_eps = float(relative_eps)

        if self.use_subject_relative_bio:
            hidden_dim = self.projector[0].out_features
            out_dim = self.projector[3].out_features
            dropout_p = self.projector[2].p if hasattr(self.projector[2], "p") else 0.2
            self.relative_projector = nn.Sequential(
                nn.LayerNorm(2 * self.raw_dim),
                nn.Linear(2 * self.raw_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout_p * 0.5),
            )

    def forward(
            self,
            x: torch.Tensor,
            de_feat: torch.Tensor,
            plv_matrix: torch.Tensor,
            hjorth: Optional[torch.Tensor] = None,
            return_raw: bool = False,
            subject_bio_mu: Optional[torch.Tensor] = None,
            subject_bio_std: Optional[torch.Tensor] = None,
    ):
        if self.detach_biomarkers:
            with torch.no_grad():
                raw = self._raw_features(
                    x=x.detach(),
                    de_feat=de_feat.detach(),
                    plv_matrix=plv_matrix.detach(),
                    hjorth=None if hjorth is None else hjorth.detach(),
                )
        else:
            raw = self._raw_features(x=x, de_feat=de_feat, plv_matrix=plv_matrix, hjorth=hjorth)

        raw = raw.to(device=x.device, dtype=x.dtype)
        bio_raw_rel = None
        bio_raw_z = None
        bio_input = raw

        use_relative = (
            self.use_subject_relative_bio
            and hasattr(self, "relative_projector")
            and subject_bio_mu is not None
            and subject_bio_std is not None
        )

        if use_relative:
            subject_bio_mu = subject_bio_mu.to(device=x.device, dtype=x.dtype)
            subject_bio_std = subject_bio_std.to(device=x.device, dtype=x.dtype)
            bio_raw_rel = raw - subject_bio_mu
            bio_raw_z = bio_raw_rel / (subject_bio_std + self.relative_eps)
            bio_input = torch.cat([self.bio_abs_scale * raw, bio_raw_z], dim=-1)
            bio_input = torch.nan_to_num(bio_input, nan=0.0, posinf=0.0, neginf=0.0)
            z = self.relative_projector(bio_input)
        else:
            z = self.projector(self.norm(raw))

        if return_raw:
            return z, raw, {
                "bio_raw_rel": bio_raw_rel,
                "bio_raw_z": bio_raw_z,
                "bio_input": bio_input,
            }
        return z

def masked_topk_adjacency(
        adj: torch.Tensor,
        k: int,
        symmetrize_after: bool = True
) -> torch.Tensor:
    """
    Keep top-k edges for each node in adjacency matrix.

    Args:
        adj: [B, N, N]
        k: number of strongest neighbors to keep per node
        symmetrize_after: whether to symmetrize after top-k masking

    Returns:
        sparse_adj: [B, N, N]
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    bsz, n, _ = adj.shape
    k = min(k, n)

    topk_vals, topk_idx = torch.topk(adj, k=k, dim=-1)
    mask = torch.zeros_like(adj)
    mask.scatter_(-1, topk_idx, 1.0)

    sparse_adj = adj * mask

    if symmetrize_after:
        sparse_adj = 0.5 * (sparse_adj + sparse_adj.transpose(-1, -2))

    return sparse_adj


def normalize_adjacency(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}

    Args:
        adj: [B, N, N]

    Returns:
        norm_adj: [B, N, N]
    """
    degree = adj.sum(dim=-1)  # [B, N]
    deg_inv_sqrt = torch.pow(degree + eps, -0.5)
    d_left = deg_inv_sqrt.unsqueeze(-1)
    d_right = deg_inv_sqrt.unsqueeze(-2)
    norm_adj = d_left * adj * d_right
    return norm_adj


def make_identity_batch(batch_size: int, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    eye = torch.eye(n, device=device, dtype=dtype)
    return eye.unsqueeze(0).expand(batch_size, -1, -1)


# =========================================================
# Spectral feature extractor
# =========================================================

class BandPowerExtractor(nn.Module):
    """
    Compute log band power from FFT for each channel.

    Input:
        x: [B, C, T]
    Output:
        band_power: [B, C, num_bands]
    """

    def __init__(
            self,
            sfreq: float = 250.0,
            bands: Optional[List[Tuple[float, float]]] = None,
            eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.sfreq = sfreq
        self.eps = eps
        if bands is None:
            # theta, alpha, beta, gamma
            bands = [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
        self.bands = bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            band_power: [B, C, K]
        """
        if x.ndim != 3:
            raise ValueError(f"x should be [B, C, T], got shape {tuple(x.shape)}")

        bsz, ch, t = x.shape

        # [T_freq]
        freqs = torch.fft.rfftfreq(t, d=1.0 / self.sfreq).to(x.device)

        # [B, C, T_freq]
        fft_vals = torch.fft.rfft(x, dim=-1)
        power = (fft_vals.real ** 2 + fft_vals.imag ** 2) / max(t, 1)

        features = []
        for low, high in self.bands:
            band_mask = (freqs >= low) & (freqs < high)
            if band_mask.sum() == 0:
                band_feat = torch.zeros(bsz, ch, device=x.device, dtype=x.dtype)
            else:
                band_feat = power[..., band_mask].mean(dim=-1)
            band_feat = torch.log(band_feat + self.eps)
            features.append(band_feat)

        band_power = torch.stack(features, dim=-1)  # [B, C, K]
        return band_power


class LearnableDEDiagonalFusion(nn.Module):
    """
    将 4 个节律的 DE 特征通过可学习权重融合后，作为脑网络对角线。

    输入:
        de_band: [B, C, 4]
            B: batch size
            C: 通道数，比如 30
            4: 四个节律 DE 特征

    输出:
        diag_value: [B, C]
            每个样本、每个通道的对角线值
    """

    def __init__(self, num_bands=5):
        super().__init__()

        # 这就是可学习的四个节律权重
        # 初始化为 0，softmax 后一开始就是平均权重 [0.25, 0.25, 0.25, 0.25]
        self.band_logits = nn.Parameter(torch.zeros(num_bands))

        self.de_extractor = DifferentialEntropyExtractor(sfreq=250)

        # 对 DE 做归一化，避免某个节律因为数值尺度大而天然占优势
        self.de_norm = nn.LayerNorm(num_bands)

        # 将融合后的 DE 映射到 0~1，适配 sigmoid 注意力邻接矩阵
        self.diag_scale = nn.Parameter(torch.tensor(1.0))
        self.diag_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, de_feat):
        """
        de_band: [B, C, 4]
        """
        de_band = de_feat # [B, C, 4]

        # 1. 对四个频带的 DE 做归一化
        de_band = self.de_norm(de_band)

        # 2. softmax 得到可学习节律权重
        band_weight = F.softmax(self.band_logits, dim=0)  # [4]

        # 3. 加权融合四个节律
        de_fused = torch.sum(de_band * band_weight.view(1, 1, -1), dim=-1)  # [B, C]

        # 4. 映射到 0~1，作为对角线自连接强度
        diag_value = torch.sigmoid(self.diag_scale * de_fused + self.diag_bias)  # [B, C]

        return diag_value, band_weight


class BandPowerDiagonalMapper(nn.Module):
    """
    将每个通道的 4 个节律平均功率映射为注意力图主对角线值。

    输入:
        x: [B, C, T]

    输出:
        diag_values: [B, C]
        band_power:  [B, C, 4]

    说明:
        - band_power 使用 BandPowerExtractor 计算 theta/alpha/beta/gamma 的 log power。
        - Linear(4, 1) 得到每个通道的自连接强度。
        - 默认 diag_mode="centered"，范围约为 [0.5, 1.5]。
          并且线性层零初始化，所以训练初始时 diag_values = 1，
          等价于原来的固定 self-loop，不会一开始破坏原模型。
    """

    def __init__(
            self,
            sfreq: float = 250.0,
            bands: Optional[List[Tuple[float, float]]] = None,
            diag_mode: str = "centered",
            eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.band_power_extractor = BandPowerExtractor(
            sfreq=sfreq,
            bands=bands,
            eps=eps,
        )
        num_bands = len(self.band_power_extractor.bands)
        self.band_norm = nn.LayerNorm(num_bands)
        self.diag_mapper = nn.Linear(num_bands, 1)
        self.diag_mode = diag_mode

        # 关键初始化：初始时 diag_values=1，等价于原始 self-loop。
        nn.init.zeros_(self.diag_mapper.weight)
        nn.init.zeros_(self.diag_mapper.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, T]

        Returns:
            diag_values: [B, C]
            band_power:  [B, C, 4]
        """
        band_power = self.band_power_extractor(x)  # [B, C, 4]
        band_power_norm = self.band_norm(band_power)  # [B, C, 4]
        raw_diag = self.diag_mapper(band_power_norm).squeeze(-1)  # [B, C]

        if self.diag_mode == "centered":
            # 推荐：保留原 self-loop 尺度，范围约 [0.5, 1.5]
            diag_values = 1.0 + 0.5 * torch.tanh(raw_diag)
        elif self.diag_mode == "sigmoid":
            # 范围 [0, 1]，可能会比原始 self-loop 更弱
            diag_values = torch.sigmoid(raw_diag)
        elif self.diag_mode == "softplus":
            # 范围 (0, +inf)，保证非负
            diag_values = F.softplus(raw_diag)
        elif self.diag_mode == "raw":
            # 不太推荐，可能出现负自连接
            diag_values = raw_diag
        else:
            raise ValueError(f"Unknown diag_mode: {self.diag_mode}")

        return diag_values, band_power


def add_power_diag_self_loop(
        adj_masked: torch.Tensor,
        eye: torch.Tensor,
        diag_values: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    用功率特征对角线替代原来的固定 self-loop。

    Args:
        adj_masked:  [B, N, N]，非对角线注意力连接
        eye:         [B, N, N]，单位阵 batch
        diag_values: [B, N]，每个通道的功率自连接强度

    Returns:
        adj_with_self: [B, N, N]
    """

    if diag_values.ndim != 2:
        raise ValueError(f"diag_values should be [B, N], got {tuple(diag_values.shape)}")

    B, N, _ = adj_masked.shape
    if diag_values.shape != (B, N):
        raise ValueError(
            f"diag_values shape should be {(B, N)}, got {tuple(diag_values.shape)}"
        )

    diag_values = diag_values.to(device=adj_masked.device, dtype=adj_masked.dtype)
    return adj_masked + torch.diag_embed(diag_values)


# =========================================================
# Node feature encoder
# =========================================================

class TemporalEncoder(nn.Module):
    """
    Shared temporal encoder applied channel-wise.

    Input:
        x: [B, C, T]
    Output:
        h_raw: [B, C, out_dim]
    """

    def __init__(self, out_dim: int = 96) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, out_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 用1维卷积，共用一套权重，分别作用在每个通道上
        bsz, ch, t = x.shape
        # reshape to [B*C, 1, T]
        x = x.reshape(bsz * ch, 1, t)
        h = self.conv(x)  # [B*C, out_dim, 1]
        h = h.squeeze(-1)  # [B*C, out_dim]
        h = h.reshape(bsz, ch, self.out_dim)
        return h


class SpectralMLP(nn.Module):
    """
    Map band power features to node spectral embedding.
    """

    def __init__(self, in_dim: int = 4, hidden_dim: int = 128, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30 * in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, K]
        x = x.flatten(-2)
        return self.net(x)


class MultiScaleTemporalEncoder(nn.Module):
    """
    TSception-style temporal branch 版通道内时间编码器。

    目的：替换原来的多尺度卷积前端，但保持输出接口不变：
        输入  x: [B, C, T]
        输出  h: [B, C, out_dim]

    设计要点：
    - 每个 EEG 通道独立共享同一套 temporal branch，避免提前混合通道；
    - 多个卷积核长度按 250Hz 采样率下的短/中时间尺度设计；
    - concat 后使用 depthwise-separable temporal block，控制参数量；
    - 使用 attention pooling 汇聚时间维，保留关键时间片信息。
    """

    def __init__(
            self,
            out_dim: int = 96,
            branch_dim: int = 12,
            dropout: float = 0.3,
            kernel_sizes: Optional[List[int]] = None,
    ):
        super().__init__()

        # 对 250Hz EEG，约对应 0.068s、0.132s、0.260s、0.500s。
        # 使用奇数卷积核，保证 padding 后各分支时间长度一致。
        if kernel_sizes is None:
            kernel_sizes = [17, 33, 65, 125]

        self.kernel_sizes = kernel_sizes
        self.branch_dim = branch_dim
        self.out_dim = out_dim

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, branch_dim, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(branch_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
            for k in kernel_sizes
        ])

        in_dim = branch_dim * len(kernel_sizes)

        self.temporal_block = nn.Sequential(
            # 降低时间长度，减少后续计算与过拟合风险
            nn.AvgPool1d(kernel_size=4, stride=4),

            # depthwise temporal conv：每个特征通道独立建模时间变化
            nn.Conv1d(
                in_dim,
                in_dim,
                kernel_size=7,
                padding=3,
                groups=in_dim,
                bias=False,
            ),
            # pointwise conv：融合不同尺度的 temporal branch
            nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            # 轻量 dilated depthwise temporal conv，扩大一点感受野
            nn.Conv1d(
                out_dim,
                out_dim,
                kernel_size=7,
                padding=6,
                dilation=2,
                groups=out_dim,
                bias=False,
            ),
            nn.Conv1d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # attention pooling over time
        self.pool_attn = nn.Sequential(
            nn.Conv1d(out_dim, out_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(out_dim // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            h: [B, C, out_dim]
        """
        if x.ndim != 3:
            raise ValueError(f"x should be [B, C, T], got {tuple(x.shape)}")

        B, C, T = x.shape
        x = x.reshape(B * C, 1, T)

        feats = [branch(x) for branch in self.branches]
        h = torch.cat(feats, dim=1)  # [B*C, branch_dim*num_branches, T]

        h = self.temporal_block(h)  # [B*C, out_dim, T']

        attn = self.pool_attn(h)  # [B*C, 1, T']
        attn = torch.softmax(attn, dim=-1)
        h = (h * attn).sum(dim=-1)  # [B*C, out_dim]

        h = h.reshape(B, C, self.out_dim)
        return h


class NodeFeatureContrastProjector(nn.Module):
    """
    对 NodeFeatureEncoder 输出的 node_features 做对比学习。

    输入:
        node_features: [B, C, D]

    输出:
        contrast_feat: [B, out_dim]
    """

    def __init__(
            self,
            node_dim: int = 64,
            hidden_dim: int = 128,
            out_dim: int = 64,
            dropout: float = 0.2,
    ):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(node_dim * 30, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, node_features):
        """
        node_features: [B, C, D]
        """

        feat = node_features.flatten(-2)  # [B, C*D]

        contrast_feat = self.projector(feat)
        contrast_feat = F.normalize(contrast_feat, dim=1)

        return contrast_feat


class DifferentialEntropyExtractor(nn.Module):
    """
    Compute Differential Entropy features for each EEG channel and frequency band.

    对每个通道、每个频带计算差分熵:
        DE = 0.5 * log(2 * pi * e * sigma^2)

    这里用 FFT 频带功率近似该频带下信号方差 sigma^2。

    Input:
        x: [B, C, T]

    Output:
        de_features: [B, C, num_bands]
    """

    def __init__(
            self,
            sfreq: float = 250.0,
            bands: Optional[List[Tuple[float, float]]] = None,
            eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.sfreq = sfreq
        self.eps = eps

        if bands is None:
            # theta, alpha, beta, gamma
            bands = [
                (4.0, 8.0),
                (8.0, 13.0),
                (13.0, 30.0),
                (30.0, 45.0),
            ]

        self.bands = bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]

        Returns:
            de_features: [B, C, K]
        """
        if x.ndim != 3:
            raise ValueError(f"x should be [B, C, T], got shape {tuple(x.shape)}")

        bsz, ch, t = x.shape

        # 频率轴: [T_freq]
        freqs = torch.fft.rfftfreq(
            t,
            d=1.0 / self.sfreq
        ).to(device=x.device)

        # FFT: [B, C, T_freq]
        fft_vals = torch.fft.rfft(x, dim=-1)

        # 功率谱，近似每个频点上的能量
        power = (fft_vals.real ** 2 + fft_vals.imag ** 2) / max(t, 1)

        features = []

        for low, high in self.bands:
            band_mask = (freqs >= low) & (freqs < high)

            if band_mask.sum() == 0:
                band_var = torch.zeros(
                    bsz,
                    ch,
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                # 用该频带平均功率近似该频带方差 sigma^2
                band_var = power[..., band_mask].mean(dim=-1)

            # Differential Entropy:
            # DE = 0.5 * log(2*pi*e*sigma^2)
            de_feat = 0.5 * torch.log(
                2.0 * math.pi * math.e * band_var + self.eps
            )

            features.append(de_feat)

        de_features = torch.stack(features, dim=-1)  # [B, C, K]
        return de_features


def hjorth_parameters(x, eps=1e-8):
    """
    x: [B, C, T]

    return:
        hjorth: [B, C, 3]
        分别是 Activity, Mobility, Complexity
    """

    # 原始信号方差
    var_x = torch.var(x, dim=-1, unbiased=False)  # [B, C]

    # 一阶差分
    dx = x[:, :, 1:] - x[:, :, :-1]
    var_dx = torch.var(dx, dim=-1, unbiased=False)  # [B, C]

    # 二阶差分
    ddx = dx[:, :, 1:] - dx[:, :, :-1]
    var_ddx = torch.var(ddx, dim=-1, unbiased=False)  # [B, C]

    activity = var_x

    mobility = torch.sqrt(var_dx / (var_x + eps))

    mobility_dx = torch.sqrt(var_ddx / (var_dx + eps))
    complexity = mobility_dx / (mobility + eps)

    hjorth = torch.stack(
        [activity, mobility, complexity],
        dim=-1
    )  # [B, C, 3]

    hjorth = hjorth.flatten(-2)

    return hjorth


class NodeFeatureEncoder(nn.Module):
    """
    Full node feature encoder:
    - temporal CNN branch
    - spectral band-power branch


    Output:
        H: [B, C, node_dim]
    """

    def __init__(
            self,
            sfreq: float = 250.0,
            temporal_dim: int = 96,
            spectral_dim: int = 32,
    ) -> None:
        super().__init__()
        self.temporal_encoder = MultiScaleTemporalEncoder(
            out_dim=temporal_dim,
            branch_dim=64,  # TSception-lite：降低每个尺度的通道数，减少过拟合
            dropout=0.3,
        )
        self.de_extractor = DifferentialEntropyExtractor(sfreq=sfreq)
        self.hjorth_extractor = hjorth_parameters
        self.spectral_mlp = SpectralMLP(in_dim=4, hidden_dim=128, out_dim=spectral_dim)
        self.node_dim = temporal_dim

    def forward(self, x: torch.Tensor) -> dict:
        h_raw = self.temporal_encoder(x)  # [B, C, 96]
        band_power = self.de_extractor(x)  # [B, C, 4]
        hjorth = self.hjorth_extractor(x)  # [B, C, 3]
        h_spec = self.spectral_mlp(band_power)  # [B,32]

        return {
            "h_raw": h_raw,
            "h_spec": h_spec,
            "hjorth": hjorth,
        }


# ========================================================
# 共享映射+通道调制的构图方式
# ========================================================
class PairSpecificGraphConstructor(nn.Module):
    """
    Pair-specific attention graph constructor.

    核心思想：
    原来每个节点只有一个 q/k 向量；
    现在每个节点针对不同目标节点生成不同的 pair-specific p/q 向量。

    Input:
        h: [B, N, D]

    Output:
        adj_norm:   [B, N, N]
        adj_masked: [B, N, N]
        adj_dense:  [B, N, N]
        scores:     [B, N, N]
    """

    def __init__(
            self,
            node_dim: int = 128,
            attn_dim: int = 16,
            num_nodes: int = 30,
            topk: int = 8,
            prior_matrix: Optional[torch.Tensor] = None,
            init_prior_strength: float = 0.5,
            channel_emb_dim: int = 16,
            pair_hidden_dim: int = 64,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.node_dim = node_dim
        self.attn_dim = attn_dim
        self.num_nodes = num_nodes
        self.topk = topk

        # 共享基础映射：防止参数爆炸
        self.p_base_proj = nn.Linear(node_dim, attn_dim, bias=False)
        self.q_base_proj = nn.Linear(node_dim, attn_dim, bias=False)

        # 通道身份嵌入
        self.src_channel_emb = nn.Embedding(num_nodes, channel_emb_dim)
        self.tgt_channel_emb = nn.Embedding(num_nodes, channel_emb_dim)

        # 根据通道对 (i, j) 生成调制参数
        # 输出:
        # gamma_p, beta_p, gamma_q, beta_q, pair_bias
        self.pair_mlp = nn.Sequential(
            nn.Linear(channel_emb_dim * 2, pair_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_dim, 4 * attn_dim + 1)
        )

        # 关键：最后一层初始化为 0
        # 这样模型初始时退化为原始共享 QK attention，不会一开始就太不稳定
        nn.init.zeros_(self.pair_mlp[-1].weight)
        nn.init.zeros_(self.pair_mlp[-1].bias)

        self.p_norm = nn.LayerNorm(attn_dim)
        self.q_norm = nn.LayerNorm(attn_dim)

        self.prior_strength_param = nn.Parameter(
            torch.tensor(float(init_prior_strength)).log()
        )

        if prior_matrix is not None:
            if prior_matrix.ndim != 2 or prior_matrix.shape[0] != prior_matrix.shape[1]:
                raise ValueError("prior_matrix must be [N, N].")
            self.register_buffer("prior_matrix", prior_matrix.float())
        else:
            self.prior_matrix = None

    def prior_strength(self) -> torch.Tensor:
        return F.softplus(self.prior_strength_param)

    def get_pair_modulation(self, device, dtype):
        """
        生成每一对通道 (i, j) 的调制参数。

        Returns:
            gamma_p:   [N, N, d]
            beta_p:    [N, N, d]
            gamma_q:   [N, N, d]
            beta_q:    [N, N, d]
            pair_bias: [N, N]
        """
        N = self.num_nodes
        idx = torch.arange(N, device=device)

        src_emb = self.src_channel_emb(idx)  # [N, E]
        tgt_emb = self.tgt_channel_emb(idx)  # [N, E]

        src_pair = src_emb[:, None, :].expand(N, N, -1)  # [N, N, E]
        tgt_pair = tgt_emb[None, :, :].expand(N, N, -1)  # [N, N, E]

        pair_feat = torch.cat([src_pair, tgt_pair], dim=-1)  # [N, N, 2E]
        pair_param = self.pair_mlp(pair_feat).to(dtype=dtype)  # [N, N, 4d + 1]

        d = self.attn_dim

        gamma_p = pair_param[..., 0:d]
        beta_p = pair_param[..., d:2 * d]
        gamma_q = pair_param[..., 2 * d:3 * d]
        beta_q = pair_param[..., 3 * d:4 * d]
        pair_bias = pair_param[..., 4 * d].squeeze(-1)  # [N, N]

        # 防止调制过强，训练初期更稳定
        gamma_p = 0.5 * torch.tanh(gamma_p)
        gamma_q = 0.5 * torch.tanh(gamma_q)

        return gamma_p, beta_p, gamma_q, beta_q, pair_bias

    def forward(self, h: torch.Tensor, diag_values: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: [B, N, D]

        Returns:
            dict:
                adj_norm:   [B, N, N]
                adj_masked: [B, N, N]
                adj_dense:  [B, N, N]
                scores:     [B, N, N]
        """
        if h.ndim != 3:
            raise ValueError(f"h should be [B, N, D], got {tuple(h.shape)}")

        B, N, D = h.shape

        if N != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, but got {N}")

        # 基础 p/q 向量
        p_base = self.p_base_proj(h)  # [B, N, d]
        q_base = self.q_base_proj(h)  # [B, N, d]

        gamma_p, beta_p, gamma_q, beta_q, pair_bias = self.get_pair_modulation(
            device=h.device,
            dtype=h.dtype,
        )

        # --------------------------------------------------
        # 构造 pair-specific p/q
        # --------------------------------------------------
        # p_pair[b, i, j, :] 表示：节点 i 面向节点 j 的 p 向量
        # q_pair[b, i, j, :] 表示：节点 j 面向节点 i 的 q 向量
        # --------------------------------------------------

        p_pair = p_base.unsqueeze(2)  # [B, N, 1, d]
        q_pair = q_base.unsqueeze(1)  # [B, 1, N, d]

        p_pair = p_pair * (1.0 + gamma_p.unsqueeze(0)) + beta_p.unsqueeze(0)
        q_pair = q_pair * (1.0 + gamma_q.unsqueeze(0)) + beta_q.unsqueeze(0)

        p_pair = self.p_norm(p_pair)
        q_pair = self.q_norm(q_pair)

        # pair-specific attention score
        scores = (p_pair * q_pair).sum(dim=-1) / math.sqrt(self.attn_dim)  # [B, N, N]

        # 加通道对偏置
        scores = scores + pair_bias.unsqueeze(0)

        # 加先验图
        if self.prior_matrix is not None:
            prior = self.prior_matrix.to(device=h.device, dtype=h.dtype)
            scores = scores + self.prior_strength() * prior.unsqueeze(0)

        adj_dense = torch.sigmoid(scores)

        # 对无向脑网络，做对称化
        adj_dense = 0.5 * (adj_dense + adj_dense.transpose(-1, -2))

        # 去掉自连接
        eye = make_identity_batch(B, N, h.device, h.dtype)
        adj_dense = adj_dense * (1.0 - eye)

        # 稀疏化
        adj_masked = masked_topk_adjacency(
            adj_dense,
            k=self.topk,
            symmetrize_after=True
        )

        # 加 self-loop：这里不再固定加单位阵，
        # 而是可选地用 4 节律功率映射得到的 diag_values 替代主对角线。
        adj_with_self = add_power_diag_self_loop(adj_masked, eye, diag_values)

        # 归一化
        adj_norm = normalize_adjacency(adj_with_self)

        return {
            "adj_norm": adj_norm,
            "adj_masked": adj_masked,
            "adj_dense": adj_dense,
            "scores": scores,
        }


# ======================================
# 通道特征映射为30一维向量构图
# ======================================

class TargetSpecificPQGraphConstructor(nn.Module):
    """
    目标通道特异 P/Q 注意力构图模块。

    对每个通道特征 h_i:
        h_i -> P_i: [N, d]
        h_i -> Q_i: [N, d]

    然后:
        score_ij = <P_i[j], Q_j[i]> / sqrt(d)

    Input:
        h: [B, N, D]

    Output:
        adj_norm:   [B, N, N]
        adj_masked: [B, N, N]
        adj_dense:  [B, N, N]
        scores:     [B, N, N]
        p_pair:     [B, N, N, d]
        q_pair:     [B, N, N, d]
    """

    def __init__(
            self,
            node_dim: int = 128,
            pair_dim: int = 16,  # p,q的维度
            num_nodes: int = 30,
            topk: int = 8,
            prior_matrix=None,
            init_prior_strength: float = 0.5,
            dropout: float = 0.1,
            use_channel_emb: bool = True,
            symmetrize: bool = True,
            use_mlp: bool = True,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.num_nodes = num_nodes
        self.topk = topk
        self.use_channel_emb = use_channel_emb
        self.symmetrize = symmetrize
        self.use_mlp = use_mlp

        if use_channel_emb:
            self.channel_emb = nn.Embedding(num_nodes, node_dim)

        # --------------------------------------------------
        # 关键部分：
        # 每一个目标通道 j 都有一个独立的 p 映射头
        # 每一个源通道 i 都有一个独立的 q 映射头
        # --------------------------------------------------
        if use_mlp:
            self.p_heads = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(node_dim),
                    nn.Linear(node_dim, node_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(node_dim, pair_dim),
                )
                for _ in range(num_nodes)
            ])

            self.q_heads = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(node_dim),
                    nn.Linear(node_dim, node_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(node_dim, pair_dim),
                )
                for _ in range(num_nodes)
            ])
        else:
            self.p_heads = nn.ModuleList([
                nn.Linear(node_dim, pair_dim)
                for _ in range(num_nodes)
            ])

            self.q_heads = nn.ModuleList([
                nn.Linear(node_dim, pair_dim)
                for _ in range(num_nodes)
            ])

        self.p_norm = nn.LayerNorm(pair_dim)
        self.q_norm = nn.LayerNorm(pair_dim)

        self.prior_strength_param = nn.Parameter(
            torch.tensor(float(init_prior_strength)).log()
        )

        if prior_matrix is not None:
            if prior_matrix.ndim != 2 or prior_matrix.shape[0] != prior_matrix.shape[1]:
                raise ValueError("prior_matrix must be [N, N].")
            self.register_buffer("prior_matrix", prior_matrix.float())
        else:
            self.prior_matrix = None

    def prior_strength(self):
        return F.softplus(self.prior_strength_param)

    def forward(self, h: torch.Tensor, diag_values: Optional[torch.Tensor] = None):
        """
        h: [B, N, D]
        """
        if h.ndim != 3:
            raise ValueError(f"h should be [B, N, D], got {tuple(h.shape)}")

        B, N, D = h.shape

        if N != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, but got {N}")

        device = h.device
        dtype = h.dtype

        # --------------------------------------------------
        # 1) 加入通道身份嵌入
        # --------------------------------------------------
        if self.use_channel_emb:
            idx = torch.arange(N, device=device)
            ch_emb = self.channel_emb(idx).to(dtype=dtype).unsqueeze(0)  # [1, N, D]
            h = h + ch_emb

        # --------------------------------------------------
        # 2) 每个 h_i 生成 P_i ∈ [N, d]
        # --------------------------------------------------
        # p_list[j]: 所有源通道面向目标通道 j 的 p 向量
        # shape: [B, N, d]
        p_list = []
        for j in range(N):
            p_j = self.p_heads[j](h)  # [B, N, d]
            p_list.append(p_j)

        # p_pair[b, i, j, :] = 通道 i 面向通道 j 的 p 向量
        p_pair = torch.stack(p_list, dim=2)  # [B, N, N, d]

        # --------------------------------------------------
        # 3) 每个 h_j 生成 Q_j ∈ [N, d]
        # --------------------------------------------------
        # q_list[i]: 所有目标通道面向源通道 i 的 q 向量
        # shape: [B, N, d]
        q_list = []
        for i in range(N):
            q_i = self.q_heads[i](h)  # [B, N, d]
            q_list.append(q_i)

        # q_raw[b, j, i, :] = 通道 j 面向通道 i 的 q 向量
        q_raw = torch.stack(q_list, dim=2)  # [B, N, N, d]

        # 对齐成 q_pair[b, i, j, :] = 通道 j 面向通道 i 的 q 向量
        q_pair = q_raw.transpose(1, 2)  # [B, N, N, d]

        # --------------------------------------------------
        # 4) 归一化 P/Q，稳定点积
        # --------------------------------------------------
        p_pair = self.p_norm(p_pair)
        q_pair = self.q_norm(q_pair)

        # --------------------------------------------------
        # 5) 注意力公式:
        # score_ij = <p_ij, q_ji> / sqrt(d)
        # --------------------------------------------------
        scores = (p_pair * q_pair).sum(dim=-1) / math.sqrt(self.pair_dim)
        # scores: [B, N, N]

        # --------------------------------------------------
        # 6) 加先验图
        # --------------------------------------------------
        if self.prior_matrix is not None:
            prior = self.prior_matrix.to(device=device, dtype=dtype)
            scores = scores + self.prior_strength() * prior.unsqueeze(0)

        # --------------------------------------------------
        # 7) 如果构建无向脑网络，则对称化
        # --------------------------------------------------
        if self.symmetrize:
            scores = 0.5 * (scores + scores.transpose(-1, -2))

        # --------------------------------------------------
        # 8) 映射为 0~1 边权重
        # --------------------------------------------------
        adj_dense = torch.sigmoid(scores)

        # 去自连接
        eye = make_identity_batch(B, N, device, dtype)
        adj_dense = adj_dense * (1.0 - eye)

        # Top-k 稀疏化
        adj_masked = masked_topk_adjacency(
            adj_dense,
            k=self.topk,
            symmetrize_after=self.symmetrize,
        )

        # 加 self-loop：这里不再固定加单位阵，
        # 而是可选地用 4 节律功率映射得到的 diag_values 替代主对角线。
        adj_with_self = add_power_diag_self_loop(adj_masked, eye, diag_values)

        # 图归一化
        adj_norm = normalize_adjacency(adj_with_self)

        return {
            "adj_norm": adj_norm,
            "adj_masked": adj_masked,
            "adj_dense": adj_dense,
            "scores": scores,
            "p_pair": p_pair,
            "q_pair": q_pair,
        }


# =========================================================
# Prior-guided graph constructor
# =========================================================

class PriorGuidedGraphConstructor(nn.Module):
    """
    Single-head attention graph constructor:
    A = sigmoid(QK^T / sqrt(dk) + lambda * B)

    Then:
    - symmetrize
    - zero diagonal
    - top-k
    - add self-loop
    - normalize

    Input:
        H: [B, N, D]
    Output:
        A_norm: [B, N, N]
        A_raw: [B, N, N]  (before self-loop normalization, after masking)
    """

    def __init__(
            self,
            node_dim: int = 128,
            attn_dim: int = 32,
            topk: int = 8,
            prior_matrix: Optional[torch.Tensor] = None,
            init_prior_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(node_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(node_dim, attn_dim, bias=False)
        self.topk = topk
        self.attn_dim = attn_dim

        # learnable positive scalar via softplus
        self.prior_strength_param = nn.Parameter(
            torch.tensor(float(init_prior_strength)).log()
        )

        if prior_matrix is not None:
            if prior_matrix.ndim != 2 or prior_matrix.shape[0] != prior_matrix.shape[1]:
                raise ValueError("prior_matrix must be [N, N].")
            self.register_buffer("prior_matrix", prior_matrix.float())
        else:
            self.prior_matrix = None

    def prior_strength(self) -> torch.Tensor:
        return F.softplus(self.prior_strength_param)

    def forward(self, h: torch.Tensor, diag_values: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: [B, N, D]

        Returns:
            dict with:
                "adj_norm": normalized adjacency [B, N, N]
                "adj_masked": masked adjacency before self-loop norm [B, N, N]
                "adj_dense": dense adjacency after sigmoid symmetrization [B, N, N]
                "scores": raw scores before sigmoid [B, N, N]
        """
        if h.ndim != 3:
            raise ValueError(f"h should be [B, N, D], got {tuple(h.shape)}")

        bsz, n, _ = h.shape
        q = self.q_proj(h)  # [B, N, dk]
        k = self.k_proj(h)  # [B, N, dk]

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attn_dim)  # [B, N, N]

        # 加先验网络矩阵
        if self.prior_matrix is not None:
            prior = self.prior_matrix.to(device=h.device, dtype=h.dtype)
            scores = scores + self.prior_strength() * prior.unsqueeze(0)

        adj_dense = torch.sigmoid(scores)

        # Symmetrize
        adj_dense = 0.5 * (adj_dense + adj_dense.transpose(-1, -2))

        # Zero diagonal before top-k
        eye = make_identity_batch(bsz, n, h.device, h.dtype)
        adj_dense = adj_dense * (1.0 - eye)

        # Top-k sparsification
        adj_masked = masked_topk_adjacency(adj_dense, k=self.topk, symmetrize_after=True)

        # Add self-loop：这里不再固定加单位阵，
        # 而是可选地用 4 节律功率映射得到的 diag_values 替代主对角线。
        adj_with_self = add_power_diag_self_loop(adj_masked, eye, diag_values)

        # Normalize
        adj_norm = normalize_adjacency(adj_with_self)

        return {
            "adj_norm": adj_norm,
            "adj_masked": adj_masked,
            "adj_dense": adj_dense,
            "scores": scores,
        }


# =========================================================
# Weighted GCN
# =========================================================

class WeightedGCNLayer(nn.Module):
    """
    H' = LN( A H W ) -> GELU -> Dropout
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, Din]
            adj: [B, N, N]
        Returns:
            out: [B, N, Dout]
        """
        x = self.linear(x)  # [B, N, Dout]
        x = torch.matmul(adj, x)  # [B, N, Dout]
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class WeightedGCNEncoder(nn.Module):
    """
    2-layer GCN + residual
    """

    def __init__(
            self,
            node_dim: int = 128,
            hidden_dim: int = 128,
            dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gcn1 = WeightedGCNLayer(node_dim, hidden_dim, dropout=dropout)
        self.gcn2 = WeightedGCNLayer(hidden_dim, hidden_dim, dropout=dropout)

        self.use_residual = (node_dim == hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            adj: [B, N, N]
        Returns:
            h: [B, N, hidden_dim]
        """
        residual = x
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)

        # 加残差
        if self.use_residual:
            h = h + residual
        return h


class GatedWeightedGCNEncoder(nn.Module):
    def __init__(self, node_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.gcn1 = WeightedGCNLayer(node_dim, hidden_dim, dropout=dropout)
        self.gcn2 = WeightedGCNLayer(hidden_dim, node_dim, dropout=dropout)

        # sigmoid(-3) ≈ 0.047，初始时图影响很小
        self.graph_gate = nn.Parameter(torch.tensor(-3.0))

        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x, adj):
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)

        alpha = torch.sigmoid(self.graph_gate)

        out = self.norm(x + alpha * h)

        return out



class FlattenGraphReadout(nn.Module):
    """
    展平式图读出。

    输入:
        h: [B, N, D]
            B: batch size
            N: 节点数，EEG 中为 30 个通道
            D: 节点特征维度，当前为 64

    输出:
        z: [B, out_dim]
            当前 out_dim=64

    读出方式:
        [B, 30, 64]
        -> flatten [B, 30*64]
        -> Linear(30*64, 256)
        -> GELU
        -> Dropout
        -> Linear(256, 64)
    """

    def __init__(
            self,
            node_dim: int = 64,
            num_nodes: int = 30,
            hidden_dim: int = 256,
            out_dim: int = 64,
            dropout: float = 0.2,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.num_nodes = num_nodes
        self.in_dim = node_dim * num_nodes
        self.out_dim = out_dim

        self.projector = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor):
        """
        h: [B, N, D]
        """
        if h.ndim != 3:
            raise ValueError(f"h should be [B, N, D], got {tuple(h.shape)}")

        B, N, D = h.shape
        if N != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, but got {N}")
        if D != self.node_dim:
            raise ValueError(f"Expected node_dim={self.node_dim}, but got {D}")

        graph_feat = h.flatten(-2)  # [B, N*D] = [B, 30*64]
        z = self.projector(graph_feat)  # [B, 64]

        # 为了兼容原来的 z_or, weight = self.readout(...) 写法，这里返回 None 作为 weight。
        return z, None

# =========================================================
# Graph readout
# =========================================================

class GraphReadout(nn.Module):
    """
    Global mean + global max pooling, followed by projection MLP.
    """

    def __init__(
            self,
            node_dim: int = 80,
            proj_hidden_dim: int = 128,
            out_dim: int = 64,
            dropout: float = 0.2,
    ) -> None:
        super().__init__()
        in_dim = node_dim * 30
        self.projector = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, proj_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, N, D]
        Returns:
            z: [B, out_dim]
        """
        # g_mean = h.mean(dim=1)      # [B, D]
        # g_max = h.max(dim=1).values # [B, D]
        # g = torch.cat([g_mean, g_max], dim=-1)
        g = h.flatten(-2)

        z = self.projector(g)
        return z


class RawSignalPLVGraphConstructor(nn.Module):
    """
    基于原始 EEG 信号的固定 PLV 脑网络构图模块。

    说明：
    - 输入原始信号 x: [B, N, T]，当前为 [B, 30, T]；
    - 在时间维 T 上用 Hilbert analytic signal 得到相位；
    - 用 PLV_ij = |mean_t(exp(j * (phase_i - phase_j)))| 构造通道间连接；
    - 不包含任何可学习构图参数；
    - 后续保持和原图分支一致：
        去自连接 -> TopK -> 加 DE 对角线 self-loop -> 图归一化。

    Input:
        x: [B, N, T]
        diag_values: [B, N]

    Output:
        adj_norm:   [B, N, N]
        adj_masked: [B, N, N]
        adj_dense:  [B, N, N]
        scores:     [B, N, N]，这里直接返回 PLV 分数
        plv_matrix: [B, N, N]
    """

    def __init__(
            self,
            num_nodes: int = 30,
            topk: int = 8,
            symmetrize: bool = True,
            eps: float = 1e-6,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.topk = topk
        self.symmetrize = symmetrize
        self.eps = eps

    @staticmethod
    def _analytic_signal(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        用 FFT 实现 Hilbert analytic signal。

        Args:
            x: real tensor
            dim: 做 Hilbert 变换的维度

        Returns:
            analytic: complex tensor，与 x 形状相同
        """
        if dim < 0:
            dim = x.ndim + dim

        n = x.size(dim)
        Xf = torch.fft.fft(x, n=n, dim=dim)

        h = torch.zeros(n, device=x.device, dtype=x.dtype)
        if n % 2 == 0:
            h[0] = 1.0
            h[n // 2] = 1.0
            h[1:n // 2] = 2.0
        else:
            h[0] = 1.0
            h[1:(n + 1) // 2] = 2.0

        shape = [1] * x.ndim
        shape[dim] = n
        h = h.view(*shape)

        analytic = torch.fft.ifft(Xf * h, n=n, dim=dim)
        return analytic

    def compute_plv(self, x: torch.Tensor) -> torch.Tensor:
        """
        基于原始 EEG 信号计算 PLV 矩阵。

        Args:
            x: [B, N, T]

        Returns:
            plv: [B, N, N]，范围约为 [0, 1]
        """
        # FFT / Hilbert 对 float16、bfloat16 支持不稳定，因此先转 float32 计算。
        x_float = x.float()

        # 去掉每个通道时间序列的 DC 分量，避免相位主要受均值影响。
        x_float = x_float - x_float.mean(dim=-1, keepdim=True)

        analytic = self._analytic_signal(x_float, dim=-1)  # [B, N, T], complex
        phase = torch.angle(analytic)  # [B, N, T]

        # 用复数矩阵乘法避免显式构造 [B, N, N, T]，显存更稳定。
        phase_complex = torch.complex(torch.cos(phase), torch.sin(phase))  # [B, N, T]
        plv_complex = torch.matmul(
            phase_complex,
            phase_complex.conj().transpose(-1, -2)
        ) / max(x.size(-1), 1)  # [B, N, N]

        plv = torch.abs(plv_complex)  # [B, N, N]

        # 理论上 PLV 已经对称，这里再做一次数值对称化，保证稳定。
        if self.symmetrize:
            plv = 0.5 * (plv + plv.transpose(-1, -2))

        return plv.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor, diag_values: Optional[torch.Tensor] = None):
        """
        x: [B, N, T]，当前为原始 EEG 信号。
        """
        if x.ndim != 3:
            raise ValueError(f"x should be [B, N, T], got {tuple(x.shape)}")

        B, N, T = x.shape
        if N != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, but got {N}")

        device = x.device
        dtype = x.dtype

        # 1. 固定公式计算原始信号 PLV 脑网络。
        plv_matrix = self.compute_plv(x)  # [B, N, N]
        scores = plv_matrix

        # 2. 去掉自连接。自连接后面由 DE 对角线 diag_values 加回来。
        eye = make_identity_batch(B, N, device, dtype)
        adj_dense = plv_matrix * (1.0 - eye)

        # 3. Top-k 稀疏化，保持和原图模块一致。
        adj_masked = masked_topk_adjacency(
            adj_dense,
            k=self.topk,
            symmetrize_after=self.symmetrize,
        )

        # 4. 加 DE 对角线 self-loop。
        adj_with_self = add_power_diag_self_loop(
            adj_masked,
            eye,
            diag_values,
        )

        # 5. 图归一化，作为 GCN 输入。
        adj_norm = normalize_adjacency(adj_with_self)

        return {
            "adj_norm": adj_norm,
            "adj_masked": adj_masked,
            "adj_dense": adj_dense,
            "scores": scores,
            "plv_matrix": plv_matrix,
        }


class EdgeVectorGraphConstructor(nn.Module):
    """
    轻量边向量构图模块。

    核心思想:
        不再为每条边单独建立 P-head 和 Q-head。

        对每个节点 h_i:
            h_i -> edge_vec_i: [N, edge_dim]

        其中:
            edge_vec_i[j] 表示通道 i 面向通道 j 的边向量。

        然后:
            score_ij = <edge_vec_i[j], edge_vec_j[i]> / sqrt(edge_dim)

        再用可学习的 edge_weight[i,j] 调制:
            score_ij = score_ij * scale_ij

    输入:
        h: [B, N, D]

    输出:
        adj_norm:   [B, N, N]
        adj_masked: [B, N, N]
        adj_dense:  [B, N, N]
        scores:     [B, N, N]
        edge_vec:   [B, N, N, edge_dim]
    """

    def __init__(
            self,
            node_dim: int = 32,
            edge_dim: int = 8,
            num_nodes: int = 30,
            topk: int = 8,
            prior_matrix=None,
            init_prior_strength: float = 0.5,
            dropout: float = 0.1,
            use_channel_emb: bool = True,
            symmetrize: bool = False,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_nodes = num_nodes
        self.topk = topk
        self.use_channel_emb = use_channel_emb
        self.symmetrize = symmetrize

        if use_channel_emb:
            self.channel_emb = nn.Embedding(num_nodes, node_dim)

        # 共享映射:
        # h_i: [D] -> [N * edge_dim]
        # 等价于为每个通道生成一个 [N, edge_dim] 的边向量表
        self.edge_proj = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Dropout(dropout),
            nn.Linear(node_dim, num_nodes * edge_dim, bias=False),
        )

        self.edge_norm = nn.LayerNorm(edge_dim)

        # 你说的 [900, 1] 可学习参数
        # 用 [N, N] 表示，每个 i-j 一 个权重
        self.edge_weight = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        self.prior_strength_param = nn.Parameter(
            torch.tensor(float(init_prior_strength)).log()
        )

        if prior_matrix is not None:
            if prior_matrix.ndim != 2 or prior_matrix.shape[0] != prior_matrix.shape[1]:
                raise ValueError("prior_matrix must be [N, N].")
            self.register_buffer("prior_matrix", prior_matrix.float())
        else:
            self.prior_matrix = None

    def prior_strength(self):
        return F.softplus(self.prior_strength_param)

    def forward(self, h: torch.Tensor, diag_values: Optional[torch.Tensor] = None):
        """
        h: [B, N, D]
        """
        if h.ndim != 3:
            raise ValueError(f"h should be [B, N, D], got {tuple(h.shape)}")

        B, N, D = h.shape

        if N != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, but got {N}")

        device = h.device
        dtype = h.dtype

        # 1. 加通道身份嵌入
        if self.use_channel_emb:
            idx = torch.arange(N, device=device)
            ch_emb = self.channel_emb(idx).to(dtype=dtype).unsqueeze(0)  # [1, N, D]
            h = h + ch_emb

        # 2. 每个通道映射出 [N, edge_dim]
        edge_vec = self.edge_proj(h)  # [B, N, N*edge_dim]
        edge_vec = edge_vec.view(B, N, N, self.edge_dim)  # [B, N, N, edge_dim]
        edge_vec = self.edge_norm(edge_vec)

        # edge_ij: 通道 i 面向通道 j 的向量
        edge_ij = edge_vec                         # [B, i, j, d]

        # edge_ji: 通道 j 面向通道 i 的向量
        edge_ji = edge_vec.transpose(1, 2)         # [B, i, j, d]

        # 3. 点积得到边分数
        scores = (edge_ij * edge_ji).sum(dim=-1) / math.sqrt(self.edge_dim)  # [B, N, N]

        # 4. 用 [N, N] 可学习权重调制不同边
        # 初始化时 scale=1，避免一开始破坏原始分数
        edge_scale = 1.0 + 0.5 * torch.tanh(self.edge_weight)  # 范围约 [0.5, 1.5]
        scores = scores * edge_scale.unsqueeze(0)

        # 5. 加结构先验
        if self.prior_matrix is not None:
            prior = self.prior_matrix.to(device=device, dtype=dtype)
            scores = scores + self.prior_strength() * prior.unsqueeze(0)

        # 6. 是否对称化
        # 如果你希望 i->j 和 j->i 不同，就保持 symmetrize=False
        if self.symmetrize:
            scores = 0.5 * (scores + scores.transpose(-1, -2))

        # 7. sigmoid 得到边权
        adj_dense = torch.sigmoid(scores)

        # 8. 去掉自连接
        eye = make_identity_batch(B, N, device, dtype)
        adj_dense = adj_dense * (1.0 - eye)

        # 9. Top-k 稀疏化
        adj_masked = masked_topk_adjacency(
            adj_dense,
            k=self.topk,
            symmetrize_after=self.symmetrize,
        )

        # 10. 加 DE 对角线 self-loop
        adj_with_self = add_power_diag_self_loop(
            adj_masked,
            eye,
            diag_values,
        )

        # 11. 图归一化
        adj_norm = normalize_adjacency(adj_with_self)

        return {
            "adj_norm": adj_norm,
            "adj_masked": adj_masked,
            "adj_dense": adj_dense,
            "scores": scores,
            "edge_vec": edge_vec,
            "edge_scale": edge_scale,
        }



class EdgeSpecificPQGraphConstructor(nn.Module):
    """
    边特异 P/Q 注意力构图模块。

    每一条有向边 (i -> j) 都有独立的 P 映射：
        p_ij = P_ij(h_i)

    每一条反向查询边 (j -> i) 都有独立的 Q 映射：
        q_ji = Q_ji(h_j)

    然后:
        score_ij = <p_ij, q_ji> / sqrt(d)

    因此:
        P 映射数量: N * N
        Q 映射数量: N * N

    当 N=30 时:
        P-head = 900 个
        Q-head = 900 个
        总映射头 = 1800 个

    Input:
        h: [B, N, D]

    Output:
        adj_norm:   [B, N, N]
        adj_masked: [B, N, N]
        adj_dense:  [B, N, N]
        scores:     [B, N, N]
        p_pair:     [B, N, N, d]
        q_pair:     [B, N, N, d]
    """

    def __init__(
            self,
            node_dim: int = 80,
            pair_dim: int = 16,
            num_nodes: int = 30,
            topk: int = 8,
            prior_matrix=None,
            init_prior_strength: float = 0.5,
            dropout: float = 0.1,
            use_channel_emb: bool = True,
            symmetrize: bool = True,
            use_mlp: bool = True,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.num_nodes = num_nodes
        self.topk = topk
        self.use_channel_emb = use_channel_emb
        self.symmetrize = symmetrize
        self.use_mlp = use_mlp

        if use_channel_emb:
            self.channel_emb = nn.Embedding(num_nodes, node_dim)

        def make_head():
            if use_mlp:
                return nn.Sequential(
                    nn.LayerNorm(node_dim),
                    nn.Linear(node_dim, node_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(node_dim, pair_dim),
                )
            else:
                return nn.Linear(node_dim, pair_dim)

        # --------------------------------------------------
        # 关键变化 1：
        # p_heads[i][j] 表示：
        # 源通道 i 面向目标通道 j 的独立 P 映射
        # --------------------------------------------------
        self.p_heads = nn.ModuleList([
            nn.ModuleList([
                make_head()
                for j in range(num_nodes)
            ])
            for i in range(num_nodes)
        ])

        # --------------------------------------------------
        # 关键变化 2：
        # q_heads[j][i] 表示：
        # 目标通道 j 面向源通道 i 的独立 Q 映射
        # --------------------------------------------------
        self.q_heads = nn.ModuleList([
            nn.ModuleList([
                make_head()
                for i in range(num_nodes)
            ])
            for j in range(num_nodes)
        ])

        self.p_norm = nn.LayerNorm(pair_dim)
        self.q_norm = nn.LayerNorm(pair_dim)

        self.prior_strength_param = nn.Parameter(
            torch.tensor(float(init_prior_strength)).log()
        )

        if prior_matrix is not None:
            if prior_matrix.ndim != 2 or prior_matrix.shape[0] != prior_matrix.shape[1]:
                raise ValueError("prior_matrix must be [N, N].")
            self.register_buffer("prior_matrix", prior_matrix.float())
        else:
            self.prior_matrix = None

    def prior_strength(self):
        return F.softplus(self.prior_strength_param)

    def forward(self, h: torch.Tensor, diag_values: Optional[torch.Tensor] = None):
        """
        h: [B, N, D]
        """
        if h.ndim != 3:
            raise ValueError(f"h should be [B, N, D], got {tuple(h.shape)}")

        B, N, D = h.shape

        if N != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, but got {N}")

        device = h.device
        dtype = h.dtype

        # --------------------------------------------------
        # 1) 加入通道身份嵌入
        # --------------------------------------------------
        if self.use_channel_emb:
            idx = torch.arange(N, device=device)
            ch_emb = self.channel_emb(idx).to(dtype=dtype).unsqueeze(0)  # [1, N, D]
            h = h + ch_emb

        # --------------------------------------------------
        # 2) 构造 p_pair
        #
        # p_pair[b, i, j, :] = p_heads[i][j](h_i)
        #
        # 即：
        # 源通道 i 面向目标通道 j 的 P 向量
        # 每一个 i-j 对都有独立映射
        # --------------------------------------------------
        p_rows = []

        for i in range(N):
            h_i = h[:, i, :]  # [B, D]

            p_cols = []
            for j in range(N):
                p_ij = self.p_heads[i][j](h_i)  # [B, d]
                p_cols.append(p_ij)

            p_i = torch.stack(p_cols, dim=1)  # [B, N, d]
            p_rows.append(p_i)

        p_pair = torch.stack(p_rows, dim=1)  # [B, N, N, d]

        # --------------------------------------------------
        # 3) 构造 q_pair
        #
        # q_pair[b, i, j, :] = q_heads[j][i](h_j)
        #
        # 注意：
        # score_ij = <p_ij, q_ji>
        #
        # 所以 q_pair 的位置 [i, j] 存的是：
        # 目标通道 j 面向源通道 i 的 Q 向量
        # --------------------------------------------------
        q_rows = []

        for i in range(N):
            q_cols = []

            for j in range(N):
                h_j = h[:, j, :]  # [B, D]
                q_ji = self.q_heads[j][i](h_j)  # [B, d]
                q_cols.append(q_ji)

            q_i = torch.stack(q_cols, dim=1)  # [B, N, d]
            q_rows.append(q_i)

        q_pair = torch.stack(q_rows, dim=1)  # [B, N, N, d]

        # --------------------------------------------------
        # 4) 归一化 P/Q
        # --------------------------------------------------
        p_pair = self.p_norm(p_pair)
        q_pair = self.q_norm(q_pair)

        # --------------------------------------------------
        # 5) 计算边分数
        #
        # score_ij = <p_ij, q_ji> / sqrt(d)
        # --------------------------------------------------
        scores = (p_pair * q_pair).sum(dim=-1) / math.sqrt(self.pair_dim)  # [B, N, N]

        # --------------------------------------------------
        # 6) 加先验图
        # --------------------------------------------------
        if self.prior_matrix is not None:
            prior = self.prior_matrix.to(device=device, dtype=dtype)
            scores = scores + self.prior_strength() * prior.unsqueeze(0)

        # --------------------------------------------------
        # 7) 对称化
        # --------------------------------------------------
        if self.symmetrize:
            scores = 0.5 * (scores + scores.transpose(-1, -2))

        # --------------------------------------------------
        # 8) sigmoid 得到边权
        # --------------------------------------------------
        adj_dense = torch.sigmoid(scores)

        # 去自连接
        eye = make_identity_batch(B, N, device, dtype)
        adj_dense = adj_dense * (1.0 - eye)

        # Top-k 稀疏化
        adj_masked = masked_topk_adjacency(
            adj_dense,
            k=self.topk,
            symmetrize_after=self.symmetrize,
        )

        # 加 self-loop：这里不再固定加单位阵，
        # 而是可选地用 4 节律功率映射得到的 diag_values 替代主对角线。
        adj_with_self = add_power_diag_self_loop(adj_masked, eye, diag_values)

        # 图归一化
        adj_norm = normalize_adjacency(adj_with_self)

        return {
            "adj_norm": adj_norm,
            "adj_masked": adj_masked,
            "adj_dense": adj_dense,
            "scores": scores,
            "p_pair": p_pair,
            "q_pair": q_pair,
        }


# =========================================================
# Full backbone
# =========================================================

class BrainGraphBackbone(nn.Module):
    """
    三分支 EEG backbone（V1 升级版）。

    分支 1：多尺度卷积节点特征分支 → z_conv [B, 64]
        EEG x -> NodeFeatureEncoder/MultiScaleTemporalEncoder -> node_features -> FlattenGraphReadout

    分支 2：原始信号 PLV 脑网络分支 → z_plv [B, 64]
        EEG x -> RawSignalPLVGraphConstructor
              -> 加同样的 DE 对角线 self-loop
              -> WeightedGCNEncoder
              -> FlattenGraphReadout

    分支 3：显式抑郁 EEG biomarker 分支 → z_bio [B, 64]
        频带统计 + 左右不对称 + Hjorth + PLV 图指标 + 非线性复杂度

    输出结构（与 V1 原版不同）：
        - z_conv  [B, 64]  → 用于四分类情绪头（不再与 z_plv 拼接）
        - z_diag  [B, 128] → z_plv ⊕ z_bio，用于二分类诊断头
    """

    def __init__(
            self,
            sfreq: float = 250.0,
            node_dim: int = 64,
            attn_dim: int = 16,
            topk: int = 8,
            prior_matrix: Optional[torch.Tensor] = None,
            dropout: float = 0.2,
            use_subject_relative_de: bool = False,
            use_subject_relative_bio: bool = False,
            bio_abs_scale: float = 0.3,
            relative_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.use_subject_relative_de = bool(use_subject_relative_de)
        self.use_subject_relative_bio = bool(use_subject_relative_bio)
        self.bio_abs_scale = float(bio_abs_scale)
        self.relative_eps = float(relative_eps)
        self.num_de_bands = 4

        self.node_encoder = NodeFeatureEncoder(sfreq=sfreq, temporal_dim=64, spectral_dim=16)
        if self.node_encoder.node_dim != node_dim:
            raise ValueError(
                f"Configured node_dim={node_dim}, but encoder outputs {self.node_encoder.node_dim}."
            )

        # 保留原来的 DE 对角线融合模块。
        self.power_diag_mapper = BandPowerDiagonalMapper(
            sfreq=sfreq,
            diag_mode="centered",
        )
        self.leranable_diag = LearnableDEDiagonalFusion(num_bands=4)
        if self.use_subject_relative_de:
            self.de_relative_adapter = SubjectRelativeDEAdapter(
                in_bands=self.num_de_bands * 3,
                out_bands=self.num_de_bands,
            )

        # 保留原来的节点特征对比分支。
        self.node_contrast_projector = NodeFeatureContrastProjector(
            node_dim=node_dim,
            hidden_dim=128,
            out_dim=64,
            dropout=dropout,
        )

        # =====================================================
        # 分支 1：原来的多尺度卷积特征分支
        # [B, 30, 64] -> flatten -> 256 -> 64
        # =====================================================
        self.conv_readout = FlattenGraphReadout(
            node_dim=node_dim,
            num_nodes=30,
            hidden_dim=256,
            out_dim=64,
            dropout=dropout,
        )

        # =====================================================
        # 分支 2：原始信号 PLV 脑网络分支
        # 原始 EEG x: [B, 30, T] -> PLV graph: [B, 30, 30]
        # PLV graph + DE 对角线 -> GCN -> 图读出
        # =====================================================
        self.plv_graph_constructor = RawSignalPLVGraphConstructor(
            num_nodes=30,
            topk=topk,
            symmetrize=True,
        )

        self.plv_graph_encoder = WeightedGCNEncoder(
            node_dim=node_dim,
            hidden_dim=node_dim,
            dropout=dropout,
        )

        self.plv_readout = FlattenGraphReadout(
            node_dim=node_dim,
            num_nodes=30,
            hidden_dim=256,
            out_dim=64,
            dropout=dropout,
        )

        # =====================================================
        # 分支 3：显式抑郁 EEG biomarker 分支（新增）
        # 频带统计 + 左右不对称 + Hjorth + PLV 图指标 + 非线性复杂度
        # 输出 z_bio: [B, 64]
        # =====================================================
        self.biomarker_extractor = SubjectRelativeBiomarkerExtractor(
            sfreq=sfreq,
            num_channels=30,
            num_de_bands=4,
            use_sampen_from_de_feat=True,
            sampen_index=-1,
            out_dim=64,
            hidden_dim=128,
            dropout=dropout,
            detach_biomarkers=True,
            use_subject_relative_bio=self.use_subject_relative_bio,
            bio_abs_scale=self.bio_abs_scale,
            relative_eps=self.relative_eps,
        )

        # ── 输出维度说明 ──
        # z_conv:  [B, 64]   → 四分类情绪头
        # z_diag:  [B, 128]  → z_plv ⊕ z_bio → 二分类诊断头
        self.conv_dim = 64          # z_conv 维度
        self.diag_dim = 64 + 64     # z_plv (64) + z_bio (64) = 128
        self.out_dim = self.conv_dim + self.diag_dim  # 兼容旧接口

        # 保留原来的 Hjorth projector 定义，避免外部加载旧代码时报属性缺失。
        self.h_projector = nn.Sequential(
            nn.Linear(90, 16),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
            self,
            x: torch.Tensor,
            de_feat,
            subject_de_mu: Optional[torch.Tensor] = None,
            subject_de_std: Optional[torch.Tensor] = None,
            subject_bio_mu: Optional[torch.Tensor] = None,
            subject_bio_std: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, T]
            de_feat: [B, C, num_bands]  前4维=频带DE，最后一维=SampEn

        Returns:
            dict:
                z_conv:  [B, 64]   多尺度卷积特征（→ 四分类头）
                z_plv:   [B, 64]   PLV 图读出特征
                z_bio:   [B, 64]   biomarker 特征
                z_diag:  [B, 128]  z_plv ⊕ z_bio（→ 二分类诊断头）
                z:       [B, 192]  三路拼接（兼容旧接口）
                ...
        """
        # =====================================================
        # 0. Optional subject-relative DE. Only de_feat and bio_raw are
        # relative-normalized in this first version.
        # =====================================================
        de_rel = None
        de_z = None
        de_input = de_feat
        de_for_model = de_feat
        use_de_relative = (
            self.use_subject_relative_de
            and hasattr(self, "de_relative_adapter")
            and subject_de_mu is not None
            and subject_de_std is not None
        )
        if use_de_relative:
            subject_de_mu = subject_de_mu.to(device=de_feat.device, dtype=de_feat.dtype)
            subject_de_std = subject_de_std.to(device=de_feat.device, dtype=de_feat.dtype)
            de_band = de_feat[..., :self.num_de_bands]
            de_mu = subject_de_mu[..., :self.num_de_bands]
            de_std = subject_de_std[..., :self.num_de_bands]
            de_rel = de_band - de_mu
            de_z = de_rel / (de_std + self.relative_eps)
            de_input = torch.cat([de_band, de_rel, de_z], dim=-1)
            de_input = torch.nan_to_num(de_input, nan=0.0, posinf=0.0, neginf=0.0)
            de_band_for_model = self.de_relative_adapter(de_input)
            if de_feat.size(-1) > self.num_de_bands:
                de_for_model = torch.cat([de_band_for_model, de_feat[..., self.num_de_bands:]], dim=-1)
            else:
                de_for_model = de_band_for_model

        # =====================================================
        # 1. 原来的多尺度卷积节点特征
        # =====================================================
        h = self.node_encoder(x)
        node_features = h["h_raw"]  # [B, C, 64]
        hjorth = h["hjorth"]

        # 保留原来的对比学习特征。
        node_contrast_feat = self.node_contrast_projector(node_features)  # [B, 64]

        # =====================================================
        # 2. 分支 1：多尺度卷积分支直接读出 → z_conv
        # =====================================================
        z_conv, _ = self.conv_readout(node_features)  # [B, 64]

        # =====================================================
        # 3. DE 对角线
        # =====================================================
        de_band_for_diag = de_for_model[..., :4]
        power_diag_values, band_weight = self.leranable_diag(de_band_for_diag)  # [B, C]

        # =====================================================
        # 4. 分支 2：原始信号 PLV 构图 + DE 对角线 + GCN + 图读出 → z_plv
        # =====================================================
        plv_graph_dict = self.plv_graph_constructor(
            x,
            diag_values=power_diag_values,
        )
        plv_adj_norm = plv_graph_dict["adj_norm"]

        plv_node_embeddings = self.plv_graph_encoder(
            node_features,
            plv_adj_norm,
        )  # [B, C, 64]

        z_plv, _ = self.plv_readout(plv_node_embeddings)  # [B, 64]

        # =====================================================
        # 5. 分支 3：显式 biomarker 特征（新增） → z_bio
        # =====================================================
        z_bio, bio_raw, bio_aux = self.biomarker_extractor(
            x=x,
            de_feat=de_for_model,
            plv_matrix=plv_graph_dict["plv_matrix"],
            hjorth=hjorth,
            return_raw=True,
            subject_bio_mu=subject_bio_mu,
            subject_bio_std=subject_bio_std,
        )  # z_bio: [B, 64]

        # =====================================================
        # 6. 重组输出
        # =====================================================
        z_diag = torch.cat([z_plv, z_bio], dim=-1)   # [B, 128] → 二分类诊断头
        z = torch.cat([z_conv, z_plv, z_bio], dim=-1)  # [B, 192] → 兼容旧接口

        return {
            # ── 新结构：按用途分离 ──
            "z_conv": z_conv,            # [B, 64]   → 四分类情绪头
            "z_plv": z_plv,              # [B, 64]   PLV 分支出
            "z_bio": z_bio,              # [B, 64]   biomarker 分支
            "z_diag": z_diag,            # [B, 128]  → 二分类诊断头

            # ── 兼容旧接口 ──
            "z": z,                      # [B, 192]  三路全拼接
            "z_or": z_plv,
            "graph_feat": z_diag,        # 二分类头用的特征，兼容 center loss 等

            "node_features": node_features,
            "node_contrast_feat": node_contrast_feat,
            "node_embeddings": plv_node_embeddings,

            # 图相关
            "adj_norm": plv_graph_dict["adj_norm"],
            "adj_masked": plv_graph_dict["adj_masked"],
            "adj_dense": plv_graph_dict["adj_dense"],
            "scores": plv_graph_dict["scores"],

            "plv_adj_norm": plv_graph_dict["adj_norm"],
            "plv_adj_masked": plv_graph_dict["adj_masked"],
            "plv_adj_dense": plv_graph_dict["adj_dense"],
            "plv_matrix": plv_graph_dict["plv_matrix"],
            "plv_node_embeddings": plv_node_embeddings,

            "power_diag_values": power_diag_values,
            "de_band_weight": band_weight,
            "hjorth": hjorth,
            "bio_raw_features": bio_raw,
            "bio_raw": bio_raw,
            "de_rel": de_rel,
            "de_z": de_z,
            "de_input": de_input,
            "de_for_model": de_for_model,
            "bio_raw_rel": bio_aux.get("bio_raw_rel"),
            "bio_raw_z": bio_aux.get("bio_raw_z"),
            "bio_input": bio_aux.get("bio_input"),
        }


# =========================================================
# Heads
# =========================================================

class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SubjectDomainHead(nn.Module):
    def __init__(self, in_dim, num_subjects, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.num_subjects = num_subjects
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_subjects)
        )

    def forward(self, feat, lambda_grl=1.0):
        feat = grad_reverse(feat, lambda_grl)
        logits = self.net(feat)
        return logits


##GRL层
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambda_grl, None


def grad_reverse(x, lambda_grl=1.0):
    return GradReverse.apply(x, lambda_grl)


# =========================================================
# Pretraining model
# =========================================================

class EmotionPretrainModel(nn.Module):
    """
    Shared backbone + dataset-specific heads
    - SEED: 3 classes
    - SEED-IV: 4 classes
    """

    def __init__(
            self,
            nclass: int = 4,
            sfreq: float = 250.0,
            prior_matrix: Optional[torch.Tensor] = None,
            num_subjects: int = 15,
            topk: int = 8,
            dropout: float = 0.2,
            use_subject_relative_de: bool = False,
            use_subject_relative_bio: bool = False,
            bio_abs_scale: float = 0.3,
            relative_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.backbone = BrainGraphBackbone(
            sfreq=sfreq,
            node_dim=64,
            attn_dim=16,
            topk=topk,
            prior_matrix=prior_matrix,
            dropout=dropout,
            use_subject_relative_de=use_subject_relative_de,
            use_subject_relative_bio=use_subject_relative_bio,
            bio_abs_scale=bio_abs_scale,
            relative_eps=relative_eps,
        )
        self.in_dim = 128
        self.seed_head = ClassificationHead(in_dim=self.in_dim, num_classes=2, hidden_dim=64, dropout=dropout)
        self.seed4_head = ClassificationHead(in_dim=self.in_dim, num_classes=4, hidden_dim=64, dropout=dropout)
        self.com4_head = ClassificationHead(in_dim=self.in_dim, num_classes=nclass, hidden_dim=64, dropout=dropout)

        self.domain_head = SubjectDomainHead(in_dim=self.in_dim, num_subjects=num_subjects, hidden_dim=64,
                                             dropout=dropout)

    def forward(
            self,
            x: torch.Tensor,
            de_feat:torch.Tensor,
            lambda_dom,
            dataset_name: str,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, T]
            lambda_dom:域对抗权重
            dataset_name: "seed" or "seed4"
        """
        out = self.backbone(x, de_feat=de_feat, **kwargs)

        z = out["z"]
        adj_norm = out["adj_norm"]

        # 分类头
        domain_logits = None
        if dataset_name.lower() == "seed":
            logits = self.seed_head(z)
        elif dataset_name.lower() in ["seed4", "seed-iv", "seed_iv"]:
            logits = self.seed4_head(z)
        elif dataset_name.lower() in ["comp4", "competition"]:
            logits = self.com4_head(z)
            domain_logits = self.domain_head(z, lambda_dom)
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        out["domain_logits"] = domain_logits
        out["logits"] = logits
        out["adj_norm"] = adj_norm
        out["graph_feat"] = z
        return out


# =========================================================
# Two-branch model（V1 升级版：三分支 backbone + 双分类头）
# =========================================================

class TwoBranchModel(nn.Module):
    """
    V1 升级版模型：三分支 backbone + 独立分类头。

    结构：
        EEG x + de_feat
        -> BrainGraphBackbone (3 branches)
        -> z_conv  [B, 64]  ──→ cls4_head  ──→ 四分类情绪 logits
        -> z_diag  [B, 128] ──→ cls2_head  ──→ 二分类诊断 logits
        -> z_diag  [B, 128] ──→ domain_head (GRL) ──→ 被试域 logits

    两个 loss：
        L = L_4cls(z_conv, label4) + λ_diag × L_2cls(z_diag, diagnosis_label) + λ_dom × L_domain
    """

    def __init__(
            self,
            sfreq: float = 250.0,
            prior_matrix: Optional[torch.Tensor] = None,
            num_subjects: int = 60,
            topk: int = 8,
            dropout: float = 0.2,
            num_classes_4: int = 4,
            num_classes_2: int = 2,
            use_subject_relative_de: bool = False,
            use_subject_relative_bio: bool = False,
            bio_abs_scale: float = 0.3,
            relative_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.use_subject_relative_de = bool(use_subject_relative_de)
        self.use_subject_relative_bio = bool(use_subject_relative_bio)
        self.bio_abs_scale = float(bio_abs_scale)
        self.relative_eps = float(relative_eps)

        self.backbone = BrainGraphBackbone(
            sfreq=sfreq,
            node_dim=64,
            attn_dim=16,
            topk=topk,
            prior_matrix=prior_matrix,
            dropout=dropout,
            use_subject_relative_de=self.use_subject_relative_de,
            use_subject_relative_bio=self.use_subject_relative_bio,
            bio_abs_scale=self.bio_abs_scale,
            relative_eps=self.relative_eps,
        )

        conv_dim = self.backbone.conv_dim    # 64
        diag_dim = self.backbone.diag_dim    # 128

        # 四分类情绪头：z_conv → 4 classes
        self.cls4_head = ClassificationHead(
            in_dim=conv_dim,
            num_classes=num_classes_4,
            hidden_dim=64,
            dropout=dropout,
        )

        # 二分类诊断头：z_diag → 2 classes
        self.cls2_head = ClassificationHead(
            in_dim=diag_dim,
            num_classes=num_classes_2,
            hidden_dim=64,
            dropout=dropout,
        )

        # 被试域对抗头：z_diag → num_subjects（通过 GRL）
        self.domain_head = SubjectDomainHead(
            in_dim=diag_dim,
            num_subjects=num_subjects,
            hidden_dim=64,
            dropout=dropout,
        )

        # 记录维度供外部使用
        self.in_dim = diag_dim          # 兼容 center loss 等需要 in_dim 的场景
        self.conv_dim = conv_dim
        self.diag_dim = diag_dim

    def forward(
            self,
            x: torch.Tensor,
            de_feat: torch.Tensor,
            lambda_dom: float = 0.0,
            dataset_name: str = "comp4",
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, T]
            de_feat: [B, C, K]
            lambda_dom: 域对抗 GRL 系数
            dataset_name: 兼容旧接口，实际分类头由 backbone 输出决定

        Returns:
            dict:
                logits:          四分类情绪 logits（兼容旧接口）
                logits_4cls:     四分类情绪 logits [B, 4]
                logits_2cls:     二分类诊断 logits [B, 2]
                domain_logits:   被试域分类 logits
                graph_feat:      z_diag [B, 128]
                z_conv, z_plv, z_bio, z_diag, z:  各分支特征
        """
        out = self.backbone(x, de_feat=de_feat, **kwargs)

        z_conv = out["z_conv"]          # [B, 64]
        z_diag = out["z_diag"]          # [B, 128]

        # 四分类情绪头
        logits_4cls = self.cls4_head(z_conv)

        # 二分类诊断头
        logits_2cls = self.cls2_head(z_diag)

        # 被试域对抗头
        domain_logits = self.domain_head(z_diag, lambda_grl=lambda_dom)

        out["logits"] = logits_4cls             # 兼容旧接口：logits = 四分类
        out["logits_4cls"] = logits_4cls
        out["logits_2cls"] = logits_2cls
        out["domain_logits"] = domain_logits
        out["graph_feat"] = z_diag              # 二分类侧特征
        out["z_diag"] = z_diag
        return out


# =========================================================
# Prior matrix builder
# =========================================================

def build_distance_prior(
        channel_coords: torch.Tensor,
        sigma: float = 0.35,
        normalize_to_minus1_1: bool = True,
) -> torch.Tensor:
    """
    Build distance-based prior matrix.

    Args:
        channel_coords: [N, 2] or [N, 3]
        sigma: distance kernel width
        normalize_to_minus1_1: whether to rescale prior to [-1, 1]

    Returns:
        prior: [N, N]
    """
    if channel_coords.ndim != 2:
        raise ValueError("channel_coords must be [N, D].")

    dist = torch.cdist(channel_coords.float(), channel_coords.float(), p=2)  # [N, N]
    prior = torch.exp(-(dist ** 2) / (2 * sigma ** 2))

    # zero diagonal before rescaling
    n = prior.shape[0]
    prior = prior * (1.0 - torch.eye(n, dtype=prior.dtype, device=prior.device))

    if normalize_to_minus1_1:
        p_min = prior.min()
        p_max = prior.max()
        prior = (prior - p_min) / (p_max - p_min + 1e-6)  # [0,1]
        prior = prior * 2.0 - 1.0  # [-1,1]

    return prior


def build_symmetry_prior(
        num_channels: int,
        symmetric_pairs: List[Tuple[int, int]],
        normalize_to_minus1_1: bool = True,
) -> torch.Tensor:
    """
    Build left-right symmetry prior.

    Args:
        num_channels: N
        symmetric_pairs: list of channel index pairs, e.g. [(0,1), (2,3)]
    """
    prior = torch.zeros(num_channels, num_channels, dtype=torch.float32)
    for i, j in symmetric_pairs:
        prior[i, j] = 1.0
        prior[j, i] = 1.0

    if normalize_to_minus1_1:
        prior = prior * 2.0 - 1.0  # 1 -> 1, 0 -> -1

    return prior


def combine_priors(
        distance_prior: torch.Tensor,
        symmetry_prior: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        beta: float = 0.3,
) -> torch.Tensor:
    """
    Combine priors:
        B = alpha * B_dist + beta * B_sym
    """
    prior = alpha * distance_prior
    if symmetry_prior is not None:
        prior = prior + beta * symmetry_prior
    return prior


# =========================================================
# Example usage
# =========================================================

if __name__ == "__main__":
    # Example fake input
    batch_size = 4
    num_channels = 30
    time_points = 2500

    x = torch.randn(batch_size, num_channels, time_points)
    de_feat=torch.randn(4,30,5)

    # Fake channel coordinates [N, 2]
    coords = torch.rand(num_channels, 2)

    # Example symmetric pairs (you should replace with your real mapping)
    symmetric_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

    # 计算先验网络
    dist_prior = build_distance_prior(coords, sigma=0.35)
    sym_prior = build_symmetry_prior(num_channels, symmetric_pairs)
    prior = combine_priors(dist_prior, sym_prior, alpha=1.0, beta=0.3)

    # Pretraining model
    pretrain_model = EmotionPretrainModel(
        sfreq=250.0,
        prior_matrix=prior,
        topk=8,
        nclass=2,
        dropout=0.2,
    )

    out_seed = pretrain_model(x,de_feat=de_feat, lambda_dom=0.1, dataset_name="comp4")
    print("SEED logits:", out_seed["logits"].shape)  # [B, 3]
    print("Graph embedding:", out_seed["z"].shape)  # [B, 128]
    print("Adjacency:", out_seed["adj_norm"].shape)  # [B, 30, 30]
    print("Power diag:", out_seed["power_diag_values"].shape)  # [B, 30]


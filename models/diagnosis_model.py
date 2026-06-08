# -*- coding: utf-8 -*-
# 版本A：TSception temporal branch 替换构图前通道内特征提取器
# 在v2上加对比学习

# 将构图改为映射为30维向量

# 原网络，特征构图，被试域对抗
# 多尺度卷积


import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# =========================================================
# Utility functions
# =========================================================

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
# Depression biomarker branch
# =========================================================

DEFAULT_30CH_NAMES = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "FT7", "FC3", "FCZ", "FC4", "FT8",
    "T3", "C3", "CZ", "C4", "T4",
    "TP7", "CP3", "CPZ", "CP4", "TP8",
    "T5", "P3", "PZ", "P4", "T6",
    "O1", "OZ", "O2",
]


def _safe_std(x: torch.Tensor, dim, keepdim: bool = False, eps: float = 1e-6):
    """unbiased=False 的稳定 std，避免单元素区域产生 nan。"""
    return torch.sqrt(torch.var(x, dim=dim, unbiased=False, keepdim=keepdim) + eps)


class DepressionBiomarkerExtractor(nn.Module):
    """
    显式抑郁 EEG biomarker 分支。

    输入:
        x:          [B, C, T] 原始 EEG 片段
        de_feat:    [B, C, K] 默认前 num_de_bands 维为频带 DE / log-power，最后一维为已计算好的 SampEn
        plv_matrix: [B, C, C] PLV 功能连接矩阵
        hjorth:     [B, C*3] 或 [B, C, 3]

    输出:
        z_bio:      [B, out_dim]

    特征包括：
        1) 频带统计特征：全脑 + 脑区 mean/std
        2) 左右不对称特征：right-left 与 abs(right-left)
        3) Hjorth 参数：Activity / Mobility / Complexity
        4) PLV 图指标：degree、边权统计、近似 clustering、脑区/半球连接
        5) 非线性复杂度：Higuchi FD、Permutation Entropy、Line Length、ZCR
        6) SampEn：直接从 de_feat 最后一维读取，不从 x 重复计算

    说明：
        - biomarker 原始特征默认 no_grad 计算，只训练后面的 MLP，避免显著增加显存。
        - 如果你的 30 通道顺序不是 DEFAULT_30CH_NAMES，需要修改 channel_names。
    """

    def __init__(
            self,
            sfreq: float = 250.0,
            num_channels: int = 30,
            num_de_bands: int = 4,
            use_sampen_from_de_feat: bool = True,
            sampen_index: int = -1,
            out_dim: int = 64,
            hidden_dim: int = 128,
            dropout: float = 0.2,
            channel_names: Optional[List[str]] = None,
            hfd_kmax: int = 8,
            perm_order: int = 3,
            perm_delay: int = 2,
            nonlinear_max_points: int = 512,
            detach_biomarkers: bool = True,
    ):
        super().__init__()
        self.sfreq = float(sfreq)
        self.num_channels = int(num_channels)
        # de_feat 默认格式：前 num_de_bands 维是频带 DE / log-power，最后一维是 SampEn。
        # 如果你的 de_feat 是 5 个频带 DE + 1 个 SampEn，可以把 num_de_bands 改为 5。
        self.num_de_bands = int(num_de_bands)
        self.use_sampen_from_de_feat = bool(use_sampen_from_de_feat)
        self.sampen_index = int(sampen_index)
        self.out_dim = int(out_dim)
        self.hfd_kmax = int(hfd_kmax)
        self.perm_order = int(perm_order)
        self.perm_delay = int(perm_delay)
        self.nonlinear_max_points = int(nonlinear_max_points)
        self.detach_biomarkers = bool(detach_biomarkers)

        if channel_names is None:
            channel_names = DEFAULT_30CH_NAMES[:num_channels]
        self.channel_names = [str(c).upper() for c in channel_names]
        self.name_to_idx = {name: i for i, name in enumerate(self.channel_names)}

        self.region_groups = self._build_region_groups()
        self.asym_pairs = self._build_asymmetry_pairs()
        self.left_indices, self.right_indices = self._build_hemisphere_indices()

        self.raw_dim = self._calc_raw_dim()
        self.norm = nn.LayerNorm(self.raw_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.raw_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

    def _idx(self, names: List[str]) -> List[int]:
        out = []
        for name in names:
            key = name.upper()
            if key in self.name_to_idx:
                idx = self.name_to_idx[key]
                if 0 <= idx < self.num_channels:
                    out.append(idx)
        return out

    def _build_region_groups(self):
        groups = {
            "frontal": self._idx(["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8"]),
            "frontocentral": self._idx(["FT7", "FC3", "FCZ", "FC4", "FT8"]),
            "central": self._idx(["T3", "C3", "CZ", "C4", "T4"]),
            "centroparietal": self._idx(["TP7", "CP3", "CPZ", "CP4", "TP8"]),
            "parietal": self._idx(["T5", "P3", "PZ", "P4", "T6"]),
            "occipital": self._idx(["O1", "OZ", "O2"]),
        }
        # 避免因为通道名不匹配导致某些组为空。
        valid = {k: v for k, v in groups.items() if len(v) > 0}
        if len(valid) == 0:
            valid = {"all": list(range(self.num_channels))}
        return valid

    def _build_asymmetry_pairs(self):
        pair_names = [
            ("FP1", "FP2"),
            ("F7", "F8"),
            ("F3", "F4"),
            ("FT7", "FT8"),
            ("FC3", "FC4"),
            ("T3", "T4"),
            ("C3", "C4"),
            ("TP7", "TP8"),
            ("CP3", "CP4"),
            ("T5", "T6"),
            ("P3", "P4"),
            ("O1", "O2"),
        ]
        pairs = []
        for left, right in pair_names:
            if left in self.name_to_idx and right in self.name_to_idx:
                li, ri = self.name_to_idx[left], self.name_to_idx[right]
                if li < self.num_channels and ri < self.num_channels:
                    pairs.append((li, ri))
        # 兜底：如果通道名不匹配，则用相邻左右通道对。
        if len(pairs) == 0:
            half = self.num_channels // 2
            pairs = [(i, i + half) for i in range(half) if i + half < self.num_channels]
        return pairs

    def _build_hemisphere_indices(self):
        left_names = ["FP1", "F7", "F3", "FT7", "FC3", "T3", "C3", "TP7", "CP3", "T5", "P3", "O1"]
        right_names = ["FP2", "F8", "F4", "FT8", "FC4", "T4", "C4", "TP8", "CP4", "T6", "P4", "O2"]
        left = self._idx(left_names)
        right = self._idx(right_names)
        if len(left) == 0 or len(right) == 0:
            half = self.num_channels // 2
            left = list(range(half))
            right = list(range(half, self.num_channels))
        return left, right

    def _calc_raw_dim(self):
        K = self.num_de_bands
        R = len(self.region_groups)
        P = len(self.asym_pairs)

        # 频带统计：全脑 mean/std + 每个脑区 mean/std
        freq_dim = 2 * K + 2 * R * K

        # 左右不对称：每对、每频带，right-left 与 abs(right-left)
        asym_dim = 2 * P * K

        # Hjorth 原始 30*3
        hjorth_dim = self.num_channels * 3

        # 非线性：HFD / PermEntropy / LineLength / ZCR，做全脑 mean/std + 脑区 mean/std
        nonlinear_k = 4
        nonlinear_dim = 2 * nonlinear_k + 2 * R * nonlinear_k

        # SampEn 已经在 de_feat 里时，直接对 SampEn 做全脑/脑区统计 + 左右不对称。
        # 这样不会从原始 x 中重复计算样本熵。
        sampen_dim = 0
        if self.use_sampen_from_de_feat:
            sampen_dim = (2 + 2 * R) + 2 * P

        # PLV 图指标：degree mean/std/max + edge mean/std/max + clustering mean/std
        # + frontal / parietal / occipital / frontoparietal / interhemi / left-intra / right-intra / global-density
        graph_dim = 3 + 3 + 2 + 8

        return int(freq_dim + asym_dim + hjorth_dim + nonlinear_dim + sampen_dim + graph_dim)

    def _fix_band_dim(self, de_feat: torch.Tensor) -> torch.Tensor:
        """把频带 DE 维固定到 self.num_de_bands。"""
        K = de_feat.size(-1)
        if K == self.num_de_bands:
            return de_feat
        if K > self.num_de_bands:
            return de_feat[..., :self.num_de_bands]
        pad = de_feat.new_zeros(*de_feat.shape[:-1], self.num_de_bands - K)
        return torch.cat([de_feat, pad], dim=-1)

    def _split_de_and_sampen(self, de_feat: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        将 de_feat 拆成频带 DE 和样本熵。

        默认约定：
            de_feat[..., :num_de_bands] 是频带 DE / log-power；
            de_feat[..., sampen_index] 是已经预处理好的 SampEn。

        如果 de_feat 维度不够，SampEn 用 0 兜底，但维度仍保持一致。
        """
        de_feat = de_feat.float()
        de_band = self._fix_band_dim(de_feat[..., :min(de_feat.size(-1), self.num_de_bands)])

        sampen = None
        if self.use_sampen_from_de_feat:
            K = de_feat.size(-1)
            idx = self.sampen_index if self.sampen_index >= 0 else K + self.sampen_index
            if 0 <= idx < K and idx >= self.num_de_bands:
                sampen = de_feat[..., idx:idx + 1]
            else:
                sampen = de_feat.new_zeros(de_feat.size(0), de_feat.size(1), 1)

        return de_band, sampen

    def _sampen_features(self, sampen: Optional[torch.Tensor], B: int, device, dtype) -> torch.Tensor:
        """
        从 de_feat 中已经计算好的 SampEn 提取统计特征，不再从 x 重复计算样本熵。
        输出维度固定为：(2 + 2*R) + 2*P。
        """
        if not self.use_sampen_from_de_feat:
            return torch.zeros(B, 0, device=device, dtype=dtype)

        expected_dim = (2 + 2 * len(self.region_groups)) + 2 * len(self.asym_pairs)
        if sampen is None:
            return torch.zeros(B, expected_dim, device=device, dtype=dtype)

        s = torch.nan_to_num(sampen.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if s.ndim == 2:
            s = s.unsqueeze(-1)

        stat = self._region_stats_3d(s)  # [B, 2+2R]

        asym_parts = []
        for li, ri in self.asym_pairs:
            left = s[:, li, :]
            right = s[:, ri, :]
            diff = right - left
            asym_parts.append(diff)
            asym_parts.append(torch.abs(diff))
        asym = torch.cat(asym_parts, dim=-1) if asym_parts else s.new_zeros(B, 0)

        out = torch.cat([stat, asym], dim=-1).to(device=device, dtype=dtype)
        if out.size(-1) != expected_dim:
            if out.size(-1) > expected_dim:
                out = out[:, :expected_dim]
            else:
                pad = out.new_zeros(B, expected_dim - out.size(-1))
                out = torch.cat([out, pad], dim=-1)
        return out

    def _region_stats_3d(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: [B, C, D]
        return: [B, 2*D + 2*R*D]
        """
        parts = [feat.mean(dim=1), _safe_std(feat, dim=1)]
        for _, idx in self.region_groups.items():
            idx_t = torch.tensor(idx, device=feat.device, dtype=torch.long)
            region = feat.index_select(dim=1, index=idx_t)
            parts.append(region.mean(dim=1))
            parts.append(_safe_std(region, dim=1))
        return torch.cat(parts, dim=-1)

    def _asymmetry_features(self, de_feat: torch.Tensor) -> torch.Tensor:
        parts = []
        for li, ri in self.asym_pairs:
            left = de_feat[:, li, :]   # [B, K]
            right = de_feat[:, ri, :]  # [B, K]
            diff = right - left
            parts.append(diff)
            parts.append(torch.abs(diff))
        return torch.cat(parts, dim=-1) if parts else de_feat.new_zeros(de_feat.size(0), 0)

    def _safe_group_mean(self, A: torch.Tensor, rows: List[int], cols: Optional[List[int]] = None) -> torch.Tensor:
        """A: [B, C, C]，返回指定子矩阵均值 [B,1]。"""
        if cols is None:
            cols = rows
        if len(rows) == 0 or len(cols) == 0:
            return A.new_zeros(A.size(0), 1)
        r = torch.tensor(rows, device=A.device, dtype=torch.long)
        c = torch.tensor(cols, device=A.device, dtype=torch.long)
        sub = A.index_select(1, r).index_select(2, c)
        return sub.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)

    def _plv_graph_features(self, plv_matrix: torch.Tensor) -> torch.Tensor:
        B, N, _ = plv_matrix.shape
        A = plv_matrix.float()
        eye = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
        A = A * (1.0 - eye)

        degree = A.sum(dim=-1) / max(N - 1, 1)  # [B,N]
        deg_stats = torch.stack([
            degree.mean(dim=1),
            _safe_std(degree, dim=1),
            degree.max(dim=1).values,
        ], dim=-1)

        idx = torch.triu_indices(N, N, offset=1, device=A.device)
        edges = A[:, idx[0], idx[1]]
        edge_stats = torch.stack([
            edges.mean(dim=1),
            _safe_std(edges, dim=1),
            edges.max(dim=1).values,
        ], dim=-1)

        # 近似 weighted clustering：diag(A^3) / (degree_count*(degree_count-1))
        A2 = torch.matmul(A, A)
        A3 = torch.matmul(A2, A)
        triangles = torch.diagonal(A3, dim1=1, dim2=2)
        deg_raw = A.sum(dim=-1)
        clustering = triangles / (deg_raw * (deg_raw - 1.0) + 1e-6)
        clustering = torch.nan_to_num(clustering, nan=0.0, posinf=0.0, neginf=0.0)
        cluster_stats = torch.stack([
            clustering.mean(dim=1),
            _safe_std(clustering, dim=1),
        ], dim=-1)

        frontal = self.region_groups.get("frontal", [])
        parietal = self.region_groups.get("parietal", []) + self.region_groups.get("centroparietal", [])
        occipital = self.region_groups.get("occipital", [])

        frontal_conn = self._safe_group_mean(A, frontal)
        parietal_conn = self._safe_group_mean(A, parietal)
        occipital_conn = self._safe_group_mean(A, occipital)
        frontoparietal_conn = self._safe_group_mean(A, frontal, parietal)
        interhemi_conn = self._safe_group_mean(A, self.left_indices, self.right_indices)
        left_intra = self._safe_group_mean(A, self.left_indices)
        right_intra = self._safe_group_mean(A, self.right_indices)
        density = edges.mean(dim=1, keepdim=True)

        return torch.cat([
            deg_stats,
            edge_stats,
            cluster_stats,
            frontal_conn,
            parietal_conn,
            occipital_conn,
            frontoparietal_conn,
            interhemi_conn,
            left_intra,
            right_intra,
            density,
        ], dim=-1)

    def _downsample_for_nonlinear(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        if T <= self.nonlinear_max_points:
            return x
        y = F.adaptive_avg_pool1d(x.reshape(B * C, 1, T), self.nonlinear_max_points)
        return y.reshape(B, C, self.nonlinear_max_points)

    def _higuchi_fd(self, x: torch.Tensor) -> torch.Tensor:
        """近似 Higuchi Fractal Dimension，输出 [B,C]。"""
        x = self._downsample_for_nonlinear(x.float())
        B, C, T = x.shape
        kmax = min(self.hfd_kmax, max(2, T // 4))
        Lk_list = []
        k_values = []

        for k in range(1, kmax + 1):
            Lm_vals = []
            for m in range(k):
                seq = x[:, :, m::k]
                n = seq.size(-1)
                if n < 2:
                    continue
                diff_sum = torch.abs(seq[:, :, 1:] - seq[:, :, :-1]).sum(dim=-1)
                scale = (T - 1.0) / (max(n - 1, 1) * k)
                Lm_vals.append(diff_sum * scale)
            if len(Lm_vals) == 0:
                continue
            Lk = torch.stack(Lm_vals, dim=0).mean(dim=0) / k  # [B,C]
            Lk_list.append(Lk.clamp_min(1e-8))
            k_values.append(k)

        if len(Lk_list) < 2:
            return x.new_zeros(B, C)

        log_L = torch.log(torch.stack(Lk_list, dim=-1))  # [B,C,K]
        log_inv_k = torch.log(x.new_tensor([1.0 / k for k in k_values]))  # [K]
        xm = log_inv_k.mean()
        ym = log_L.mean(dim=-1, keepdim=True)
        numerator = ((log_inv_k - xm).view(1, 1, -1) * (log_L - ym)).sum(dim=-1)
        denominator = ((log_inv_k - xm) ** 2).sum().clamp_min(1e-8)
        hfd = numerator / denominator
        return torch.nan_to_num(hfd, nan=0.0, posinf=0.0, neginf=0.0)

    def _permutation_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Permutation Entropy，默认 order=3，输出 [B,C]。"""
        x = self._downsample_for_nonlinear(x.float())
        B, C, T = x.shape
        order = self.perm_order
        delay = self.perm_delay
        win_span = (order - 1) * delay + 1
        L = T - win_span + 1
        if L <= 1:
            return x.new_zeros(B, C)

        seqs = []
        for i in range(order):
            seqs.append(x[..., i * delay:i * delay + L])
        patterns = torch.stack(seqs, dim=-1)  # [B,C,L,order]
        ranks = torch.argsort(patterns, dim=-1)

        code = torch.zeros(B, C, L, device=x.device, dtype=torch.long)
        for i in range(order):
            code = code * order + ranks[..., i]

        num_codes = order ** order
        hist = F.one_hot(code, num_classes=num_codes).float().mean(dim=-2)  # [B,C,num_codes]
        prob = hist.clamp_min(1e-8)
        ent = -(prob * torch.log(prob)).sum(dim=-1)
        ent = ent / math.log(math.factorial(order))
        return torch.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0)

    def _nonlinear_features(self, x: torch.Tensor) -> torch.Tensor:
        xs = x.float()
        hfd = self._higuchi_fd(xs)
        pe = self._permutation_entropy(xs)
        diff = xs[:, :, 1:] - xs[:, :, :-1]
        line_length = torch.mean(torch.abs(diff), dim=-1)
        zcr = torch.mean((xs[:, :, 1:] * xs[:, :, :-1] < 0).float(), dim=-1)
        nl = torch.stack([hfd, pe, line_length, zcr], dim=-1)  # [B,C,4]
        return self._region_stats_3d(nl)

    def _hjorth_features(self, hjorth: torch.Tensor, B: int, device, dtype) -> torch.Tensor:
        if hjorth is None:
            return torch.zeros(B, self.num_channels * 3, device=device, dtype=dtype)
        if hjorth.ndim == 3:
            h = hjorth.reshape(B, -1)
        else:
            h = hjorth
        if h.size(-1) != self.num_channels * 3:
            # 维度不一致时做兜底 padding / truncation。
            if h.size(-1) > self.num_channels * 3:
                h = h[:, :self.num_channels * 3]
            else:
                pad = h.new_zeros(B, self.num_channels * 3 - h.size(-1))
                h = torch.cat([h, pad], dim=-1)
        # Activity 等可能尺度较大，压一下尺度。
        h = torch.sign(h) * torch.log1p(torch.abs(h))
        return h

    def _raw_features(self, x: torch.Tensor, de_feat: torch.Tensor, plv_matrix: torch.Tensor, hjorth: Optional[torch.Tensor]):
        B = x.size(0)
        de_band, sampen = self._split_de_and_sampen(de_feat)
        de_band = torch.nan_to_num(de_band, nan=0.0, posinf=0.0, neginf=0.0)

        # 注意：频带统计和左右不对称只使用前 num_de_bands 个频带 DE，
        # 不把 SampEn 混进频带 DE 或图 self-loop。
        freq_stats = self._region_stats_3d(de_band)
        asym = self._asymmetry_features(de_band)
        hj = self._hjorth_features(hjorth, B=B, device=x.device, dtype=de_band.dtype)
        graph = self._plv_graph_features(plv_matrix)
        nonlinear = self._nonlinear_features(x)
        sampen_stats = self._sampen_features(sampen, B=B, device=x.device, dtype=de_band.dtype)

        feat = torch.cat([freq_stats, asym, hj, graph, nonlinear, sampen_stats], dim=-1)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        if feat.size(-1) != self.raw_dim:
            raise RuntimeError(
                f"Biomarker raw feature dim mismatch: got {feat.size(-1)}, expected {self.raw_dim}. "
                f"请检查 channel_names / num_bands / region_groups。"
            )
        return feat

    def forward(
            self,
            x: torch.Tensor,
            de_feat: torch.Tensor,
            plv_matrix: torch.Tensor,
            hjorth: Optional[torch.Tensor] = None,
            return_raw: bool = False,
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
        z = self.projector(self.norm(raw))
        if return_raw:
            return z, raw
        return z

# =========================================================
# Full backbone
# =========================================================

class BrainGraphBackbone(nn.Module):
    """
    双分支 EEG backbone。

    分支 1：原来的多尺度卷积节点特征分支
        EEG x -> NodeFeatureEncoder/MultiScaleTemporalEncoder -> node_features -> FlattenGraphReadout

    分支 2：原始信号 PLV 脑网络分支
        EEG x -> RawSignalPLVGraphConstructor
              -> 加同样的 DE 对角线 self-loop
              -> WeightedGCNEncoder
              -> FlattenGraphReadout

    最终：
        z = concat(z_conv, z_plv) = [B, 128]
    """

    def __init__(
            self,
            sfreq: float = 250.0,
            node_dim: int = 64,
            attn_dim: int = 16,
            topk: int = 8,
            prior_matrix: Optional[torch.Tensor] = None,
            dropout: float = 0.2,
    ) -> None:
        super().__init__()

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
        # 分支 3：显式抑郁 EEG biomarker 分支
        # 包含：频带统计、左右不对称、Hjorth、PLV 图指标、非线性复杂度
        # 输出 z_bio: [B, 64]
        # =====================================================
        self.biomarker_extractor = DepressionBiomarkerExtractor(
            sfreq=sfreq,
            num_channels=30,
            num_de_bands=4,
            use_sampen_from_de_feat=True,
            sampen_index=-1,
            out_dim=64,
            hidden_dim=128,
            dropout=dropout,
            detach_biomarkers=True,
        )

        # 最终拼接三个分支：
        # z_conv [B,64] + z_plv [B,64] + z_bio [B,64] = z [B,192]
        self.out_dim = 64 + 64 + self.biomarker_extractor.out_dim

        # 保留原来的 Hjorth projector 定义，避免外部加载旧代码时报属性缺失。
        self.h_projector = nn.Sequential(
            nn.Linear(90, 16),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, de_feat) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, T]
            de_feat: [B, C, num_bands]

        Returns:
            dict:
                z: [B, 128]
                z_conv: [B, 64]
                z_plv: [B, 64]
                node_features: [B, C, 64]
                plv_adj_norm: [B, C, C]
        """
        # =====================================================
        # 1. 原来的多尺度卷积节点特征
        # =====================================================
        h = self.node_encoder(x)
        node_features = h["h_raw"]  # [B, C, 64]
        hjorth = h["hjorth"]

        # 保留原来的对比学习特征。


        # =====================================================
        # 2. 分支 1：多尺度卷积分支直接读出
        # =====================================================
        z_conv, _ = self.conv_readout(node_features)  # [B, 64]

        # =====================================================
        # 3. DE 对角线：两个图相关模块共用同一套 DE self-loop 逻辑
        # =====================================================
        # de_feat 默认前 4 维是频带 DE，最后一维是 SampEn。
        # 图 self-loop 只使用频带 DE，不把 SampEn 混进去。
        de_band_for_diag = de_feat[..., :4]
        power_diag_values, band_weight = self.leranable_diag(de_band_for_diag)  # [B, C]

        # =====================================================
        # 4. 分支 2：原始信号 PLV 构图 + DE 对角线 + GCN + 图读出
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
        # 5. 分支 3：显式 biomarker 特征
        # =====================================================
        z_bio, bio_raw = self.biomarker_extractor(
            x=x,
            de_feat=de_feat,
            plv_matrix=plv_graph_dict["plv_matrix"],
            hjorth=hjorth,
            return_raw=True,
        )  # z_bio: [B,64]

        # =====================================================
        # 6. 三个分支拼接
        # =====================================================
        z = torch.cat([z_conv, z_plv, z_bio], dim=-1)  # [B, 192]

        return {
            "z": z,
            "z_conv": z_conv,
            "z_plv": z_plv,
            "z_bio": z_bio,
            "bio_raw_features": bio_raw,

            # 为了兼容旧训练代码，保留 z_or 字段；
            # 当前 z_or 指向 PLV 图分支读出特征。
            "z_or": z_plv,

            "node_features": node_features,

            "node_embeddings": plv_node_embeddings,

            # 为了兼容旧训练代码，adj_norm / adj_dense 默认返回 PLV 分支图。
            "adj_norm": plv_graph_dict["adj_norm"],
            "adj_masked": plv_graph_dict["adj_masked"],
            "adj_dense": plv_graph_dict["adj_dense"],
            "scores": plv_graph_dict["scores"],

            # 新增更明确的 PLV 图字段。
            "plv_adj_norm": plv_graph_dict["adj_norm"],
            "plv_adj_masked": plv_graph_dict["adj_masked"],
            "plv_adj_dense": plv_graph_dict["adj_dense"],
            "plv_matrix": plv_graph_dict["plv_matrix"],
            "plv_node_embeddings": plv_node_embeddings,

            "power_diag_values": power_diag_values,
            "de_band_weight": band_weight,
            "hjorth": hjorth,
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
    二分支 backbone + 三分类头版本。

    结构：
        EEG x + de_feat
        -> BrainGraphBackbone
        -> z = concat(z_conv, z_plv)  # [B, 128]
        -> diagnosis_head: 诊断二分类，不加 GRL，主任务
        -> emotion_head:   情绪二分类，经过 GRL，辅助对抗任务
        -> subject_head:   被试分类，经过 GRL，辅助对抗任务

    注意：
        - 原来的 seed_head / seed4_head / com4_head 已删除。
        - out["logits"] 默认等于 diagnosis_logits，方便兼容旧训练代码中 logits 的写法。
        - emotion_logits / subject_logits 虽然是分类头输出，但由于输入特征先过 GRL，
          对 backbone 的梯度方向会反转。
    """

    def __init__(
            self,
            sfreq: float = 250.0,
            prior_matrix: Optional[torch.Tensor] = None,
            num_subjects: int = 60,
            topk: int = 8,
            dropout: float = 0.2,
            diagnosis_classes: int = 2,
            emotion_classes: int = 2,
            # 保留 nclass 参数只是为了兼容旧 main 里 EmotionPretrainModel(..., nclass=xxx) 的写法
            nclass: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.backbone = BrainGraphBackbone(
            sfreq=sfreq,
            node_dim=64,
            attn_dim=16,
            topk=topk,
            prior_matrix=prior_matrix,
            dropout=dropout,
        )

        # 现在 backbone 输出 z = concat(z_conv, z_plv, z_bio)，默认维度为 192。
        self.in_dim = self.backbone.out_dim
        self.diagnosis_head = ClassificationHead(
            in_dim=self.in_dim,
            num_classes=diagnosis_classes,
            hidden_dim=64,
            dropout=dropout,
        )

        # 情绪头：用于情绪二分类，但通过 GRL 反向约束 backbone。
        self.emotion_head = SubjectDomainHead(
            in_dim=self.in_dim,
            num_subjects=emotion_classes,
            hidden_dim=64,
            dropout=dropout,
        )

        # 被试头：用于被试分类，也通过 GRL 反向约束 backbone。
        self.subject_head = SubjectDomainHead(
            in_dim=self.in_dim,
            num_subjects=num_subjects,
            hidden_dim=64,
            dropout=dropout,
        )

    def forward(
            self,
            x: torch.Tensor,
            de_feat: torch.Tensor,
            lambda_emo: float = 0.0,
            lambda_subject: float = 0.0,
            # 下面两个参数用于兼容旧代码，不再决定分类头选择
            lambda_dom: Optional[float] = None,
            dataset_name: Optional[str] = None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, T]
            de_feat: [B, C, K]
            lambda_emo: 情绪 GRL 系数，只影响 backbone 接收的情绪头梯度方向和强度
            lambda_subject: 被试 GRL 系数，只影响 backbone 接收的被试头梯度方向和强度
        """
        out = self.backbone(x, de_feat=de_feat)
        z = out["z"]  # [B, 128]

        diagnosis_logits = self.diagnosis_head(z)
        emotion_logits = self.emotion_head(z, lambda_grl=lambda_emo)
        subject_logits = self.subject_head(z, lambda_grl=lambda_subject)

        out["diagnosis_logits"] = diagnosis_logits
        out["emotion_logits"] = emotion_logits
        out["subject_logits"] = subject_logits

        # 兼容旧训练脚本中 out["logits"] 的用法：现在 logits 默认代表诊断二分类。
        out["logits"] = diagnosis_logits
        out["domain_logits"] = subject_logits
        out["graph_feat"] = z
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


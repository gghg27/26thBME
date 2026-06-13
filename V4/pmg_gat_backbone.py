import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.dep_contrast_bio import NodeFeatureEncoder,NodeFeatureContrastProjector, FlattenGraphReadout, RawSignalPLVGraphConstructor, WeightedGCNEncoder, BiologicalMarkerExtractor, normalize_adjacency
from models.dep_contrast_bio import BandPowerDiagonalMapper, LearnableDEDiagonalFusion, SubjectRelativeDEAdapter

# =========================================================
# Region PLV-prior Multi-head Attention (第二版 PMG)
# =========================================================

class RegionPLVPriorAttention(nn.Module):
    """
    对 region_features 做多头 self-attention，并使用 region PLV prior 作为 attention log-bias。

    输入:
        x:            [B, R, F]       region_features
        region_prior: [B, R, R]       行归一化后的 region PLV prior (含 self-loop)

    输出:
        region_embeddings: [B, R, embed_dim]
        region_attn:       [B, num_heads, R, R]
        region_attn_mean:  [B, R, R]

    Attention 公式:
        scores = Q @ K^T * scale + plv_prior_scale * log(region_prior)
        attn = softmax(scores)
        out = attn @ V

    结构:
        x -> Multi-head PLV-prior Attention -> dropout -> residual + LayerNorm
          -> FFN -> residual + LayerNorm -> region_embeddings
    """

    def __init__(
            self,
            embed_dim: int = 64,
            num_heads: int = 4,
            dropout: float = 0.2,
            eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads   # 16
        self.scale = self.head_dim ** -0.5
        self.eps = eps

        # ---- Q / K / V 投影 ----
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # ---- 输出投影 ----
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # ---- dropout ----
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # ---- LayerNorm ----
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ---- FFN ----
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

        # ---- 可学习的 PLV prior scale ----
        # 如果训练后 plv_prior_scale 变成极端负数导致数值不稳定，
        # 后续可改为 F.softplus(self.plv_prior_scale) 来强制为正。
        self.plv_prior_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
            self,
            x: torch.Tensor,
            region_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x:            [B, R, F]      region_features,  R=5, F=64
            region_prior: [B, R, R]      行归一化的 region PLV prior

        Returns:
            region_embeddings: [B, R, embed_dim]
            attn_dict: {
                "region_attn":       [B, num_heads, R, R],
                "region_attn_mean":  [B, R, R],
                "plv_prior_scale":   scalar tensor (detached for logging),
            }
        """
        B, R, F = x.shape
        assert F == self.embed_dim, f"Expected feature dim {self.embed_dim}, got {F}"
        assert R == region_prior.size(-1), \
            f"region_prior last dim {region_prior.size(-1)} != R={R}"

        # ---- 1. Q / K / V 投影 ----
        # [B, R, embed_dim] -> [B, num_heads, R, head_dim]
        Q = self.q_proj(x).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [B, num_heads, R, head_dim]

        # ---- 2. Scaled dot-product scores ----
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores: [B, num_heads, R, R]

        # ---- 3. PLV log-bias ----
        # log_prior: [B, R, R]
        log_prior = torch.log(region_prior.clamp(min=self.eps))
        # 数值稳定：减去每行最大值
        scores = scores - scores.max(dim=-1, keepdim=True).values
        scores = scores + self.plv_prior_scale * log_prior.unsqueeze(1)
        # scores: [B, num_heads, R, R]

        # ---- 4. Attention + output ----
        attn = torch.softmax(scores, dim=-1)               # [B, num_heads, R, R]
        attn = self.attn_dropout(attn)
        attn_out = torch.matmul(attn, V)                    # [B, num_heads, R, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, R, self.embed_dim)
        # attn_out: [B, R, embed_dim]

        # ---- 5. Residual + LayerNorm ----
        residual = x
        x_out = self.norm1(residual + self.proj_dropout(self.out_proj(attn_out)))

        # ---- 6. FFN + Residual + LayerNorm ----
        residual = x_out
        x_out = self.norm2(residual + self.ffn(x_out))
        # x_out: [B, R, embed_dim]

        # ---- 7. 构建返回字典 ----
        region_attn_mean = attn.mean(dim=1)  # [B, R, R]

        attn_dict = {
            "region_attn": attn,                                # [B, num_heads, R, R]
            "region_attn_mean": region_attn_mean,               # [B, R, R]
            "plv_prior_scale": self.plv_prior_scale.detach(),   # scalar (for logging)
        }

        return x_out, attn_dict


# =========================================================
# 保留旧版 RegionGCN 以便兼容旧 checkpoint（不再参与前向计算）
# =========================================================

class RegionGCN(nn.Module):
    """
    轻量两层 Region GCN，在脑区级图上做消息传递（第一版 PMG 使用）。

    输入:
        x:   [B, R, F]  region_features
        adj: [B, R, R]  region_adj_norm

    输出:
        h:   [B, R, F]  region_embeddings

    注意：第二版 PMG 已使用 RegionPLVPriorAttention 替代此类，
         仅保留以兼容旧 checkpoint 加载。
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x:   [B, R, F]
        # adj: [B, R, R]
        residual = x
        h = torch.bmm(adj, x)          # [B, R, F]
        h = self.fc1(h)                 # [B, R, F]
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = torch.bmm(adj, h)          # [B, R, F]
        h = self.fc2(h)                 # [B, R, F]
        h = self.norm2(h)
        h = self.act(h)

        if h.shape == residual.shape:
            h = h + residual

        return h


# =========================================================
# Region PLV PMG Encoder
# =========================================================

class RegionPLVPMGEncoder(nn.Module):
    """
    PMG (Parallel Multi-scale Graph) 编码器 —— 第二版。

    结构:
        Local PLV-GCN (通道级)
        + Region PLV-prior Multi-head Attention (脑区级)  ← 替换原 Region GCN
        + Fusion

    输入:
        node_features: [B, 30, 64]
        plv_adj_norm:  [B, 30, 30]
        plv_matrix:    [B, 30, 30]

    输出 dict:
        z_pmg:                  [B, 64]
        z_local:                [B, 64]
        z_region:               [B, 64]
        local_node_embeddings:  [B, 30, 64]
        region_features:        [B, 5, 64]
        region_adj:             [B, 5, 5]
        region_adj_norm:        [B, 5, 5]
        region_prior:           [B, 5, 5]   (行归一化，用于 attention log-bias)
        region_embeddings:      [B, 5, 64]
        region_attn:            [B, 4, 5, 5]
        region_attn_mean:       [B, 5, 5]
        plv_prior_scale:        scalar
    """

    # 5 个脑区分组（按 30 通道顺序）
    REGION_GROUPS = {
        "frontal":           [0, 1, 2, 3, 4, 5, 6],
        "fronto_central":    [7, 8, 9, 10, 11],
        "central_temporal":  [12, 13, 14, 15, 16],
        "centro_parietal":   [17, 18, 19, 20, 21],
        "parieto_occipital": [22, 23, 24, 25, 26, 27, 28, 29],
    }

    def __init__(
            self,
            node_dim: int = 64,
            hidden_dim: int = 64,
            num_nodes: int = 30,
            num_heads: int = 4,
            dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.num_nodes = num_nodes
        self.num_regions = len(self.REGION_GROUPS)  # 5
        self.num_heads = num_heads

        # ---- 构建并注册 region_map: [5, 30] ----
        region_map = torch.zeros(self.num_regions, num_nodes)  # [5, 30]
        for r_idx, ch_list in enumerate(self.REGION_GROUPS.values()):
            weight = 1.0 / len(ch_list)
            for ch in ch_list:
                region_map[r_idx, ch] = weight
        self.register_buffer("region_map", region_map, persistent=True)
        # region_map: [5, 30]，每一行是一个 region 的平均池化权重

        # ---- 1. Local GCN：通道级 PLV-GCN ----
        self.local_gcn = WeightedGCNEncoder(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        # 输入 node_features [B, 30, 64] + plv_adj_norm [B, 30, 30]
        # 输出 local_node_embeddings [B, 30, 64]

        # ---- 2. Local readout ----
        self.local_readout = FlattenGraphReadout(
            node_dim=node_dim,
            num_nodes=num_nodes,
            hidden_dim=256,
            out_dim=node_dim,
            dropout=dropout,
        )
        # 输入 local_node_embeddings [B, 30, 64]
        # 输出 z_local [B, 64]

        # ---- 3. Region PLV-prior Multi-head Attention (替换原 RegionGCN) ----
        self.region_attn = RegionPLVPriorAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            dropout=dropout,
            eps=1e-6,
        )
        # 输入 region_features [B, 5, 64] + region_prior [B, 5, 5]
        # 输出 region_embeddings [B, 5, 64] + attn_dict

        # ---- 4. Region readout: mean + max -> Linear -> 64 ----
        self.region_readout = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 输入 [region_mean; region_max] [B, 128]
        # 输出 z_region [B, 64]

        # ---- 5. PMG fusion: concat(z_local, z_region) -> 64 ----
        self.pmg_fusion = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 输入 [z_local; z_region] [B, 128]
        # 输出 z_pmg [B, 64]

    @staticmethod
    def _build_region_adj(
            plv_matrix: torch.Tensor,
            region_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从通道级 PLV 矩阵聚合得到脑区级邻接矩阵，并构建 attention 用的 region_prior。

        Args:
            plv_matrix: [B, 30, 30]
            region_map: [5, 30]

        Returns:
            region_adj:      [B, 5, 5]  原始聚合 + self-loop
            region_adj_norm: [B, 5, 5]  D^{-1/2} A D^{-1/2} 归一化 (兼容旧字段)
            region_prior:    [B, 5, 5]  行归一化 + self-loop (用于 attention log-bias)
        """
        R = region_map.size(0)  # 5

        # einsum: ri,bij,sj->brs
        #   r: region index, s: region index, i/j: channel index
        region_adj = torch.einsum("ri,bij,sj->brs", region_map, plv_matrix, region_map)
        # [B, 5, 5]

        # 数值稳定性处理
        region_adj = torch.nan_to_num(region_adj, nan=0.0, posinf=0.0, neginf=0.0)
        region_adj = 0.5 * (region_adj + region_adj.transpose(1, 2))  # 对称化
        region_adj = region_adj.clamp(min=0.0)

        # 加 self-loop（不再叠加 DE 对角线，因为 DE 对角线已在通道级 PLV 图中处理）
        eye = torch.eye(R, device=region_adj.device, dtype=region_adj.dtype).unsqueeze(0)  # [1, 5, 5]
        region_adj_self = region_adj + eye  # [B, 5, 5]

        # 图归一化: D^{-1/2} A D^{-1/2} (兼容旧字段)
        region_adj_norm = normalize_adjacency(region_adj_self)  # [B, 5, 5]

        # region_prior: 行归一化，用于 attention log-bias
        # row-sum = D, clamp 避免除零
        region_prior = region_adj_self / region_adj_self.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        # region_prior: [B, 5, 5]

        return region_adj, region_adj_norm, region_prior

    def forward(
            self,
            node_features: torch.Tensor,
            plv_adj_norm: torch.Tensor,
            plv_matrix: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_features: [B, 30, 64]
            plv_adj_norm:  [B, 30, 30]
            plv_matrix:    [B, 30, 30]

        Returns:
            dict with z_pmg, z_local, z_region, region_attn, etc.
        """
        # ---- 1. Local PLV-GCN: 通道级图卷积 ----
        # local_node_embeddings: [B, 30, 64]
        local_node_embeddings = self.local_gcn(node_features, plv_adj_norm)

        # ---- 2. Local readout: [B, 30, 64] -> z_local [B, 64] ----
        z_local, _ = self.local_readout(local_node_embeddings)

        # ---- 3. Region feature pooling: [B, 30, 64] -> [B, 5, 64] ----
        # region_map: [5, 30], einsum: rc,bcf->brf
        region_features = torch.einsum(
            "rc,bcf->brf",
            self.region_map.to(node_features.device),
            local_node_embeddings,
        )
        # region_features: [B, 5, 64]

        # ---- 4. Region PLV adjacency: [B, 30, 30] -> [B, 5, 5] ----
        region_adj, region_adj_norm, region_prior = self._build_region_adj(
            plv_matrix, self.region_map
        )
        # region_adj:      [B, 5, 5]  原始聚合 + self-loop
        # region_adj_norm: [B, 5, 5]  D^{-1/2} A D^{-1/2} (兼容旧字段)
        # region_prior:    [B, 5, 5]  行归一化 (用于 attention log-bias)

        # ---- 5. Region PLV-prior Attention: [B, 5, 64] + [B, 5, 5] -> [B, 5, 64] ----
        region_embeddings, region_attn_dict = self.region_attn(region_features, region_prior)
        # region_embeddings:  [B, 5, 64]
        # region_attn_dict:
        #   "region_attn":       [B, num_heads, 5, 5]
        #   "region_attn_mean":  [B, 5, 5]
        #   "plv_prior_scale":   scalar (detached)

        # ---- 6. Region readout: mean + max -> 64 ----
        region_mean = region_embeddings.mean(dim=1)       # [B, 64]
        region_max = region_embeddings.max(dim=1).values   # [B, 64]
        z_region = self.region_readout(
            torch.cat([region_mean, region_max], dim=-1)
        )  # [B, 128] -> [B, 64]

        # ---- 7. PMG fusion: concat(z_local, z_region) -> 64 ----
        z_pmg = self.pmg_fusion(
            torch.cat([z_local, z_region], dim=-1)
        )  # [B, 128] -> [B, 64]

        return {
            "z_pmg": z_pmg,                              # [B, 64]
            "z_local": z_local,                          # [B, 64]
            "z_region": z_region,                        # [B, 64]
            "local_node_embeddings": local_node_embeddings,  # [B, 30, 64]
            "region_features": region_features,          # [B, 5, 64]
            "region_adj": region_adj,                    # [B, 5, 5]
            "region_adj_norm": region_adj_norm,          # [B, 5, 5]  兼容旧字段
            "region_prior": region_prior,                # [B, 5, 5]  用于 attention log-bias
            "region_embeddings": region_embeddings,      # [B, 5, 64]
            # ---- 新字段：region attention ----
            "region_attn": region_attn_dict["region_attn"],              # [B, num_heads, 5, 5]
            "region_attn_mean": region_attn_dict["region_attn_mean"],    # [B, 5, 5]
            "region_prior_adj": region_prior,                            # [B, 5, 5]  兼容别名
            "plv_prior_scale": region_attn_dict["plv_prior_scale"],      # scalar
        }


# =========================================================
# Full backbone
# =========================================================

class BrainGraphBackbone(nn.Module):
    """
    双分支 EEG backbone (PMG 第二版)。

    分支 1：原来的多尺度卷积节点特征分支
        EEG x -> NodeFeatureEncoder/MultiScaleTemporalEncoder -> node_features -> FlattenGraphReadout
        -> z_conv: [B, 64]

    分支 2 (第二版 PMG)：Region PLV PMG 脑网络分支
        EEG x -> RawSignalPLVGraphConstructor
              -> RegionPLVPMGEncoder
                  (Local PLV-GCN + Region feature pooling
                   + Region PLV adj + Region PLV-prior Multi-head Attention  ← 替换 Region GCN
                   + Local/Region readout + Fusion)
              -> z_pmg: [B, 64]

    最终：
        z_core = concat(z_conv, z_pmg) = [B, 128]
        z = concat(z_core, z_bio) = [B, 128 + biomarker_dim]
    """

    def __init__(
            self,
            sfreq: float = 250.0,
            node_dim: int = 64,
            attn_dim: int = 16,
            topk: int = 8,
            prior_matrix: Optional[torch.Tensor] = None,
            dropout: float = 0.2,
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
        self.de_num_bands = int(de_num_bands)

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
        self.leranable_diag = LearnableDEDiagonalFusion(num_bands=self.de_num_bands)
        if self.use_subject_relative_de:
            self.de_relative_adapter = SubjectRelativeDEAdapter(
                in_bands=self.de_num_bands * 3,
                out_bands=self.de_num_bands,
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
        # 分支 2 (新版 PMG)：Region PLV PMG Encoder
        # Local PLV-GCN + Region PLV-prior Attention + Fusion
        # =====================================================
        self.pmg_encoder = RegionPLVPMGEncoder(
            node_dim=node_dim,
            hidden_dim=node_dim,
            num_nodes=30,
            num_heads=4,
            dropout=dropout,
        )

        # 生物标志物分支：频带统计 / 左右不对称 / Hjorth / PLV图指标 / 非线性特征。
        self.use_biomarkers = bool(use_biomarkers)
        self.biomarker_dim = int(biomarker_dim) if self.use_biomarkers else 0
        if self.use_biomarkers:
            self.biomarker_extractor = BiologicalMarkerExtractor(
                sfreq=sfreq,
                num_channels=30,
                use_subject_relative_bio=self.use_subject_relative_bio,
                bio_abs_scale=self.bio_abs_scale,
                relative_eps=self.relative_eps,
            )
        else:
            self.biomarker_extractor = None

        # 最终拼接：z_conv [B,64] + z_pmg [B,64] + z_bio [B,biomarker_dim]
        self.core_dim = 128
        self.out_dim = self.core_dim + self.biomarker_dim

        # 保留原来的 Hjorth projector 定义，避免外部加载旧代码时报属性缺失；
        # 但当前双分支版本不再把 Hjorth 拼进最终 z。
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
        # 0. Subject-relative DE。若 baseline 缺失则自动退化为原始 de_feat。
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
            and de_feat.size(-1) == self.de_num_bands
        )
        if use_de_relative:
            subject_de_mu = subject_de_mu.to(device=de_feat.device, dtype=de_feat.dtype)
            subject_de_std = subject_de_std.to(device=de_feat.device, dtype=de_feat.dtype)
            de_rel = de_feat - subject_de_mu
            de_z = de_rel / (subject_de_std + self.relative_eps)
            de_input = torch.cat([de_feat, de_rel, de_z], dim=-1)
            de_input = torch.nan_to_num(de_input, nan=0.0, posinf=0.0, neginf=0.0)
            de_for_model = self.de_relative_adapter(de_input)

        # =====================================================
        # 1. 原来的多尺度卷积节点特征
        # =====================================================
        h = self.node_encoder(x)
        node_features = h["h_raw"]  # [B, C, 64]
        hjorth = h["hjorth"]

        # 保留原来的对比学习特征。
        node_contrast_feat = self.node_contrast_projector(node_features)  # [B, 64]

        # =====================================================
        # 2. 分支 1：多尺度卷积分支直接读出
        # =====================================================
        z_conv, _ = self.conv_readout(node_features)  # [B, 64]

        # =====================================================
        # 3. DE 对角线：两个图相关模块共用同一套 DE self-loop 逻辑
        # =====================================================
        power_diag_values, band_weight = self.leranable_diag(de_for_model)  # [B, C]

        # =====================================================
        # 4. 分支 2 (第二版 PMG)：PLV 构图 + PMG 编码器
        #    Local PLV-GCN + Region PLV-prior Attention + Fusion
        #    node_features: [B, 30, 64]
        #    plv_adj_norm:  [B, 30, 30]
        #    plv_matrix:    [B, 30, 30]
        #    -> z_pmg:      [B, 64]
        # =====================================================
        plv_graph_dict = self.plv_graph_constructor(
            x,
            diag_values=power_diag_values,
        )
        plv_adj_norm = plv_graph_dict["adj_norm"]

        # ---- PMG 编码器 ----
        pmg_dict = self.pmg_encoder(
            node_features=node_features,
            plv_adj_norm=plv_adj_norm,
            plv_matrix=plv_graph_dict["plv_matrix"],
        )
        z_pmg = pmg_dict["z_pmg"]                              # [B, 64]
        pmg_node_embeddings = pmg_dict["local_node_embeddings"]  # [B, 30, 64]

        # （旧 plv_graph_encoder / plv_readout 模块保留在 state_dict 中以兼容旧 checkpoint，
        #   但 forward 中不再使用旧 PLV-GCN 路径参与 z_core，完全由 PMG 替代。）

        # =====================================================
        # 5. 生物标志物分支 + 最终拼接
        #    z_core = concat(z_conv, z_pmg): [B, 128]
        # =====================================================
        z_core = torch.cat([z_conv, z_pmg], dim=-1)  # [B, 128]

        bio_dict = {}
        if self.use_biomarkers and self.biomarker_extractor is not None:
            bio_dict = self.biomarker_extractor(
                x,
                de_feat=de_for_model,
                hjorth=hjorth,
                plv_matrix=plv_graph_dict["plv_matrix"],
                subject_bio_mu=subject_bio_mu,
                subject_bio_std=subject_bio_std,
            )
            z_bio = bio_dict["bio_feat"]  # [B, biomarker_dim]
            z = torch.cat([z_core, z_bio], dim=-1)
        else:
            z_bio = z_core.new_zeros(z_core.size(0), 0)
            z = z_core

        return {
            "z": z,
            "z_core": z_core,
            "z_bio": z_bio,
            "z_conv": z_conv,

            # ---- PMG 主输出 ----
            "z_pmg": z_pmg,                              # [B, 64]  PMG 融合特征

            # ---- 兼容旧字段名，指向 PMG 结果 ----
            "z_plv": z_pmg,                              # 兼容旧代码：指向 z_pmg
            "z_or": z_pmg,                               # 兼容旧代码：指向 z_pmg

            "node_embeddings": pmg_node_embeddings,       # 兼容旧代码：指向 PMG local_node_embeddings
            "pmg_node_embeddings": pmg_node_embeddings,   # [B, 30, 64]
            "plv_node_embeddings": pmg_node_embeddings,   # 兼容旧代码
            "local_node_embeddings": pmg_dict["local_node_embeddings"],  # [B, 30, 64]

            # ---- PMG 中间特征 ----
            "z_local": pmg_dict["z_local"],              # [B, 64]
            "z_region": pmg_dict["z_region"],             # [B, 64]
            "region_features": pmg_dict["region_features"],      # [B, 5, 64]
            "region_adj": pmg_dict["region_adj"],                # [B, 5, 5]
            "region_adj_norm": pmg_dict["region_adj_norm"],      # [B, 5, 5]
            "region_prior": pmg_dict["region_prior"],            # [B, 5, 5]  行归一化 prior
            "region_prior_adj": pmg_dict["region_prior_adj"],    # [B, 5, 5]  兼容别名
            "region_embeddings": pmg_dict["region_embeddings"],  # [B, 5, 64]

            # ---- Region attention 可视化字段 ----
            "region_attn": pmg_dict["region_attn"],              # [B, 4, 5, 5]
            "region_attn_mean": pmg_dict["region_attn_mean"],    # [B, 5, 5]
            "plv_prior_scale": pmg_dict["plv_prior_scale"],      # scalar

            "node_features": node_features,
            "node_contrast_feat": node_contrast_feat,

            # ---- PLV 图字段（保持兼容）----
            "adj_norm": plv_graph_dict["adj_norm"],
            "adj_masked": plv_graph_dict["adj_masked"],
            "adj_dense": plv_graph_dict["adj_dense"],
            "scores": plv_graph_dict["scores"],
            "plv_adj_norm": plv_graph_dict["adj_norm"],
            "plv_adj_masked": plv_graph_dict["adj_masked"],
            "plv_adj_dense": plv_graph_dict["adj_dense"],
            "plv_matrix": plv_graph_dict["plv_matrix"],

            # ---- 生物标志物 ----
            "bio_feat": z_bio,
            "bio_raw": bio_dict.get("bio_raw", None),
            "bio_raw_rel": bio_dict.get("bio_raw_rel", None),
            "bio_raw_z": bio_dict.get("bio_raw_z", None),
            "bio_input": bio_dict.get("bio_input", None),
            "bio_freq_feat": bio_dict.get("bio_freq_feat", None),
            "bio_asym_feat": bio_dict.get("bio_asym_feat", None),
            "bio_hjorth_feat": bio_dict.get("bio_hjorth_feat", None),
            "bio_plv_feat": bio_dict.get("bio_plv_feat", None),
            "bio_nonlinear_feat": bio_dict.get("bio_nonlinear_feat", None),

            # ---- 其他 ----
            "power_diag_values": power_diag_values,
            "de_band_weight": band_weight,
            "de_rel": de_rel,
            "de_z": de_z,
            "de_input": de_input,
            "de_for_model": de_for_model,
            "hjorth": hjorth,
        }

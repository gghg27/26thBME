import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.dep_contrast_bio import NodeFeatureEncoder,NodeFeatureContrastProjector, FlattenGraphReadout, RawSignalPLVGraphConstructor, WeightedGCNEncoder, BiologicalMarkerExtractor, normalize_adjacency
from models.dep_contrast_bio import BandPowerDiagonalMapper, LearnableDEDiagonalFusion, SubjectRelativeDEAdapter

# =========================================================
# Region GCN
# =========================================================

class RegionGCN(nn.Module):
    """
    轻量两层 Region GCN，在脑区级图上做消息传递。

    输入:
        x:   [B, R, F]  region_features
        adj: [B, R, R]  region_adj_norm

    输出:
        h:   [B, R, F]  region_embeddings
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
    PMG (Parallel Multi-scale Graph) 编码器 —— 第一版。

    结构:
        Local PLV-GCN (通道级)
        + Region PLV-GCN (脑区级)
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
        region_embeddings:      [B, 5, 64]
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
            dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.num_nodes = num_nodes
        self.num_regions = len(self.REGION_GROUPS)  # 5

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

        # ---- 3. Region GCN ----
        self.region_gcn = RegionGCN(
            in_dim=node_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        # 输入 region_features [B, 5, 64] + region_adj_norm [B, 5, 5]
        # 输出 region_embeddings [B, 5, 64]

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从通道级 PLV 矩阵聚合得到脑区级邻接矩阵。

        Args:
            plv_matrix: [B, 30, 30]
            region_map: [5, 30]

        Returns:
            region_adj: [B, 5, 5]
        """
        # einsum: ri,bij,sj->brs
        #   r: region index, s: region index, i/j: channel index
        region_adj = torch.einsum("ri,bij,sj->brs", region_map, plv_matrix, region_map)
        # [B, 5, 5]

        # 数值稳定性处理
        region_adj = torch.nan_to_num(region_adj, nan=0.0, posinf=0.0, neginf=0.0)
        region_adj = 0.5 * (region_adj + region_adj.transpose(1, 2))  # 对称化
        region_adj = region_adj.clamp(min=0.0)

        # 加 self-loop（不再叠加 DE 对角线，因为 DE 对角线已在通道级 PLV 图中处理）
        eye = torch.eye(
            region_adj.size(1),
            device=region_adj.device,
            dtype=region_adj.dtype,
        ).unsqueeze(0)  # [1, 5, 5]
        region_adj = region_adj + eye  # [B, 5, 5]

        # 图归一化: D^{-1/2} A D^{-1/2}
        region_adj_norm = normalize_adjacency(region_adj)  # [B, 5, 5]

        return region_adj, region_adj_norm

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
            dict with z_pmg, z_local, z_region, etc.
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
        region_adj, region_adj_norm = self._build_region_adj(
            plv_matrix, self.region_map
        )
        # region_adj:      [B, 5, 5]
        # region_adj_norm: [B, 5, 5]

        # ---- 5. Region GCN: [B, 5, 64] + [B, 5, 5] -> [B, 5, 64] ----
        region_embeddings = self.region_gcn(region_features, region_adj_norm)
        # region_embeddings: [B, 5, 64]

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
            "region_adj_norm": region_adj_norm,          # [B, 5, 5]
            "region_embeddings": region_embeddings,      # [B, 5, 64]
        }


# =========================================================
# Full backbone
# =========================================================

class NodeRelativeEmotionAdapter(nn.Module):
    """
    用 DE 相对特征调制 node_encoder 输出的通道级节点特征，
    构造情绪分支使用的相对时序节点特征。

    输入:
        node_features: [B, C, node_dim]
        de_input:      [B, C, de_in_dim]
                       当 de_num_bands=5 时，de_input = concat(de_abs, de_rel, de_z)，维度为 15

    输出:
        node_features_emotion: [B, C, node_dim]
    """

    def __init__(
            self,
            node_dim: int = 64,
            de_in_dim: int = 15,
            hidden_dim: int = 64,
            dropout: float = 0.2,
            init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.de_proj = nn.Sequential(
            nn.Linear(de_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Sigmoid(),
        )

        self.delta = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )

        self.norm = nn.LayerNorm(node_dim)
        self.res_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, node_features: torch.Tensor, de_input: torch.Tensor) -> torch.Tensor:
        de_ctx = self.de_proj(de_input)
        joint = torch.cat([node_features, de_ctx], dim=-1)
        gate = self.gate(joint)
        delta = self.delta(joint)
        out = node_features + self.res_scale * gate * delta
        out = self.norm(out)
        return out


class BrainGraphBackbone(nn.Module):
    """
    双分支 EEG backbone (PMG 版)。

    分支 1：原来的多尺度卷积节点特征分支
        EEG x -> NodeFeatureEncoder/MultiScaleTemporalEncoder -> node_features -> FlattenGraphReadout

    分支 2 (新版 PMG)：Region PLV PMG 脑网络分支
        EEG x -> RawSignalPLVGraphConstructor
              -> RegionPLVPMGEncoder
                  (Local PLV-GCN + Region feature pooling
                   + Region PLV adj + Region GCN
                   + Local/Region readout + Fusion)
              -> z_pmg

    最终：
        z = concat(z_conv, z_pmg) = [B, 128]
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
            self.node_emotion_adapter = NodeRelativeEmotionAdapter(
                node_dim=node_dim,
                de_in_dim=self.de_num_bands * 3,
                hidden_dim=node_dim,
                dropout=dropout,
                init_scale=0.1,
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
        self.conv_readout_emotion = FlattenGraphReadout(
            node_dim=node_dim,
            num_nodes=30,
            hidden_dim=256,
            out_dim=64,
            dropout=dropout,
        )

        self.conv_readout_diag = FlattenGraphReadout(
            node_dim=node_dim,
            num_nodes=30,
            hidden_dim=256,
            out_dim=64,
            dropout=dropout,
        )
        self.conv_readout = self.conv_readout_emotion

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
        # 替代原来的 plv_graph_encoder + plv_readout 组合
        # =====================================================
        self.pmg_encoder = RegionPLVPMGEncoder(
            node_dim=node_dim,
            hidden_dim=node_dim,
            num_nodes=30,
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
        self.dual_head_de_bio = True
        print("[DualHead] emotion uses subject-relative DE/Bio; diagnosis uses absolute DE/Bio")
        print("[ConvDualFlow-B2] z_conv_emotion uses DE-relative modulated node features; z_conv_diag uses absolute node features.")

        # 保留原来的 Hjorth projector 定义，避免外部加载旧代码时报属性缺失；
        # 但当前双分支版本不再把 Hjorth 拼进最终 z。
        self.h_projector = nn.Sequential(
            nn.Linear(90, 16),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _forward_dual_head(
            self,
            x: torch.Tensor,
            de_feat: torch.Tensor,
            subject_de_mu: Optional[torch.Tensor] = None,
            subject_de_std: Optional[torch.Tensor] = None,
            subject_bio_mu: Optional[torch.Tensor] = None,
            subject_bio_std: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        de_rel = None
        de_z = None
        de_input_emotion = de_feat
        de_for_diag = de_feat
        de_for_emotion = de_feat

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
            de_input_emotion = torch.cat([de_feat, de_rel, de_z], dim=-1)
            de_input_emotion = torch.nan_to_num(de_input_emotion, nan=0.0, posinf=0.0, neginf=0.0)
            de_for_emotion = self.de_relative_adapter(de_input_emotion)

        h = self.node_encoder(x)
        node_features_abs = h["h_raw"]
        hjorth = h["hjorth"]
        node_features_diag = node_features_abs
        node_features_emotion = node_features_abs
        if use_de_relative and hasattr(self, "node_emotion_adapter"):
            node_features_emotion = self.node_emotion_adapter(
                node_features_abs,
                de_input_emotion,
            )

        node_contrast_feat = self.node_contrast_projector(node_features_emotion)
        z_conv_emotion, _ = self.conv_readout_emotion(node_features_emotion)
        z_conv_diag, _ = self.conv_readout_diag(node_features_diag)

        power_diag_emotion, band_weight_emotion = self.leranable_diag(de_for_emotion)
        power_diag_diag, band_weight_diag = self.leranable_diag(de_for_diag)

        plv_graph_emotion = self.plv_graph_constructor(
            x,
            diag_values=power_diag_emotion,
        )
        plv_graph_diag = self.plv_graph_constructor(
            x,
            diag_values=power_diag_diag,
        )

        pmg_emotion = self.pmg_encoder(
            node_features=node_features_abs,
            plv_adj_norm=plv_graph_emotion["adj_norm"],
            plv_matrix=plv_graph_emotion["plv_matrix"],
        )
        pmg_diag = self.pmg_encoder(
            node_features=node_features_abs,
            plv_adj_norm=plv_graph_diag["adj_norm"],
            plv_matrix=plv_graph_diag["plv_matrix"],
        )

        z_pmg_emotion = pmg_emotion["z_pmg"]
        z_pmg_diag = pmg_diag["z_pmg"]
        z_core_emotion = torch.cat([z_conv_emotion, z_pmg_emotion], dim=-1)
        z_core_diag = torch.cat([z_conv_diag, z_pmg_diag], dim=-1)

        bio_emotion = {}
        bio_diag = {}
        if self.use_biomarkers and self.biomarker_extractor is not None:
            bio_diag = self.biomarker_extractor(
                x,
                de_feat=de_for_diag,
                hjorth=hjorth,
                plv_matrix=plv_graph_diag["plv_matrix"],
                subject_bio_mu=None,
                subject_bio_std=None,
            )
            use_bio_relative = (
                self.use_subject_relative_bio
                and subject_bio_mu is not None
                and subject_bio_std is not None
            )
            bio_emotion_de = de_for_emotion if use_bio_relative else de_for_diag
            bio_emotion = self.biomarker_extractor(
                x,
                de_feat=bio_emotion_de,
                hjorth=hjorth,
                plv_matrix=plv_graph_emotion["plv_matrix"],
                subject_bio_mu=subject_bio_mu,
                subject_bio_std=subject_bio_std,
            )
            z_bio_emotion = bio_emotion["bio_feat"]
            z_bio_diag = bio_diag["bio_feat"]
            z_emotion = torch.cat([z_core_emotion, z_bio_emotion], dim=-1)
            z_diag = torch.cat([z_core_diag, z_bio_diag], dim=-1)
        else:
            z_bio_emotion = z_core_emotion.new_zeros(z_core_emotion.size(0), 0)
            z_bio_diag = z_core_diag.new_zeros(z_core_diag.size(0), 0)
            z_emotion = z_core_emotion
            z_diag = z_core_diag

        pmg_node_embeddings = pmg_emotion["local_node_embeddings"]

        return {
            "z": z_emotion,
            "z_emotion": z_emotion,
            "z_diag": z_diag,

            "z_core": z_core_emotion,
            "z_core_emotion": z_core_emotion,
            "z_core_diag": z_core_diag,

            "z_bio": z_bio_emotion,
            "z_bio_emotion": z_bio_emotion,
            "z_bio_diag": z_bio_diag,
            "z_conv": z_conv_emotion,
            "z_conv_emotion": z_conv_emotion,
            "z_conv_diag": z_conv_diag,

            "z_pmg": z_pmg_emotion,
            "z_pmg_emotion": z_pmg_emotion,
            "z_pmg_diag": z_pmg_diag,
            "z_plv": z_pmg_emotion,
            "z_or": z_pmg_emotion,

            "node_embeddings": pmg_node_embeddings,
            "pmg_node_embeddings": pmg_node_embeddings,
            "plv_node_embeddings": pmg_node_embeddings,
            "local_node_embeddings": pmg_emotion["local_node_embeddings"],
            "pmg_node_embeddings_diag": pmg_diag["local_node_embeddings"],
            "local_node_embeddings_diag": pmg_diag["local_node_embeddings"],

            "z_local": pmg_emotion["z_local"],
            "z_local_emotion": pmg_emotion["z_local"],
            "z_local_diag": pmg_diag["z_local"],
            "z_region": pmg_emotion["z_region"],
            "z_region_emotion": pmg_emotion["z_region"],
            "z_region_diag": pmg_diag["z_region"],
            "region_features": pmg_emotion["region_features"],
            "region_features_emotion": pmg_emotion["region_features"],
            "region_features_diag": pmg_diag["region_features"],
            "region_adj": pmg_emotion["region_adj"],
            "region_adj_emotion": pmg_emotion["region_adj"],
            "region_adj_diag": pmg_diag["region_adj"],
            "region_adj_norm": pmg_emotion["region_adj_norm"],
            "region_adj_norm_emotion": pmg_emotion["region_adj_norm"],
            "region_adj_norm_diag": pmg_diag["region_adj_norm"],
            "region_embeddings": pmg_emotion["region_embeddings"],
            "region_embeddings_emotion": pmg_emotion["region_embeddings"],
            "region_embeddings_diag": pmg_diag["region_embeddings"],

            "node_features": node_features_emotion,
            "node_features_abs": node_features_abs,
            "node_features_emotion": node_features_emotion,
            "node_features_diag": node_features_diag,
            "node_contrast_feat": node_contrast_feat,

            "adj_norm": plv_graph_emotion["adj_norm"],
            "adj_norm_emotion": plv_graph_emotion["adj_norm"],
            "adj_norm_diag": plv_graph_diag["adj_norm"],
            "adj_masked": plv_graph_emotion["adj_masked"],
            "adj_masked_emotion": plv_graph_emotion["adj_masked"],
            "adj_masked_diag": plv_graph_diag["adj_masked"],
            "adj_dense": plv_graph_emotion["adj_dense"],
            "adj_dense_emotion": plv_graph_emotion["adj_dense"],
            "adj_dense_diag": plv_graph_diag["adj_dense"],
            "scores": plv_graph_emotion["scores"],
            "scores_emotion": plv_graph_emotion["scores"],
            "scores_diag": plv_graph_diag["scores"],
            "plv_adj_norm": plv_graph_emotion["adj_norm"],
            "plv_adj_norm_emotion": plv_graph_emotion["adj_norm"],
            "plv_adj_norm_diag": plv_graph_diag["adj_norm"],
            "plv_adj_masked": plv_graph_emotion["adj_masked"],
            "plv_adj_masked_emotion": plv_graph_emotion["adj_masked"],
            "plv_adj_masked_diag": plv_graph_diag["adj_masked"],
            "plv_adj_dense": plv_graph_emotion["adj_dense"],
            "plv_adj_dense_emotion": plv_graph_emotion["adj_dense"],
            "plv_adj_dense_diag": plv_graph_diag["adj_dense"],
            "plv_matrix": plv_graph_emotion["plv_matrix"],
            "plv_matrix_emotion": plv_graph_emotion["plv_matrix"],
            "plv_matrix_diag": plv_graph_diag["plv_matrix"],

            "bio_feat": z_bio_emotion,
            "bio_raw": bio_diag.get("bio_raw", None),
            "bio_raw_abs": bio_diag.get("bio_raw", None),
            "bio_raw_rel": bio_emotion.get("bio_raw_rel", None),
            "bio_raw_z": bio_emotion.get("bio_raw_z", None),
            "bio_input": bio_emotion.get("bio_input", None),
            "bio_input_emotion": bio_emotion.get("bio_input", None),
            "bio_input_diag": bio_diag.get("bio_input", None),
            "bio_freq_feat": bio_emotion.get("bio_freq_feat", None),
            "bio_freq_feat_emotion": bio_emotion.get("bio_freq_feat", None),
            "bio_freq_feat_diag": bio_diag.get("bio_freq_feat", None),
            "bio_asym_feat": bio_emotion.get("bio_asym_feat", None),
            "bio_asym_feat_emotion": bio_emotion.get("bio_asym_feat", None),
            "bio_asym_feat_diag": bio_diag.get("bio_asym_feat", None),
            "bio_hjorth_feat": bio_emotion.get("bio_hjorth_feat", None),
            "bio_hjorth_feat_emotion": bio_emotion.get("bio_hjorth_feat", None),
            "bio_hjorth_feat_diag": bio_diag.get("bio_hjorth_feat", None),
            "bio_plv_feat": bio_emotion.get("bio_plv_feat", None),
            "bio_plv_feat_emotion": bio_emotion.get("bio_plv_feat", None),
            "bio_plv_feat_diag": bio_diag.get("bio_plv_feat", None),
            "bio_nonlinear_feat": bio_emotion.get("bio_nonlinear_feat", None),
            "bio_nonlinear_feat_emotion": bio_emotion.get("bio_nonlinear_feat", None),
            "bio_nonlinear_feat_diag": bio_diag.get("bio_nonlinear_feat", None),

            "power_diag_values": power_diag_emotion,
            "power_diag_emotion": power_diag_emotion,
            "power_diag_diag": power_diag_diag,
            "de_band_weight": band_weight_emotion,
            "de_band_weight_emotion": band_weight_emotion,
            "de_band_weight_diag": band_weight_diag,
            "de_rel": de_rel,
            "de_z": de_z,
            "de_input": de_input_emotion,
            "de_input_emotion": de_input_emotion,
            "de_input_diag": de_feat,
            "de_for_model": de_for_emotion,
            "de_for_emotion": de_for_emotion,
            "de_for_diag": de_for_diag,
            "hjorth": hjorth,
        }

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
        return self._forward_dual_head(
            x=x,
            de_feat=de_feat,
            subject_de_mu=subject_de_mu,
            subject_de_std=subject_de_std,
            subject_bio_mu=subject_bio_mu,
            subject_bio_std=subject_bio_std,
        )

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
        # 4. 分支 2 (新版 PMG)：PLV 构图 + PMG 编码器
        #    Local PLV-GCN + Region PLV-GCN + Fusion
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
            "region_embeddings": pmg_dict["region_embeddings"],  # [B, 5, 64]

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

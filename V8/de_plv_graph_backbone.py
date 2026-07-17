"""Five-band DE + PLV window encoder used by V8 Experiment A.

The module deliberately contains no raw-EEG path and no trial-level temporal
aggregation.  Each DE band is paired with the PLV graph from the same band,
encoded by Chebyshev graph convolutions, and fused with band attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChebGraphConv(nn.Module):
    """Batched Chebyshev graph convolution with residual normalization."""

    def __init__(self, in_dim: int, out_dim: int, order: int = 3, dropout: float = 0.3,
                 eps: float = 1e-6) -> None:
        super().__init__()
        if order < 1:
            raise ValueError("Chebyshev order must be at least one")
        self.order = int(order)
        self.eps = float(eps)
        self.transforms = nn.ModuleList(nn.Linear(in_dim, out_dim, bias=False) for _ in range(self.order))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.residual = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def sanitize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim != 3 or adjacency.shape[-1] != adjacency.shape[-2]:
            raise ValueError(f"adjacency must be [B,N,N], got {tuple(adjacency.shape)}")
        adjacency = torch.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        adjacency = 0.5 * (adjacency + adjacency.transpose(-1, -2))
        eye = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
        return adjacency * (1.0 - eye)

    def scaled_laplacian(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Return L_tilde=2L/lambda_max-I with lambda_max=2 for normalized L."""
        adjacency = self.sanitize_adjacency(adjacency)
        eye = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
        with_loop = adjacency + eye
        degree = with_loop.sum(dim=-1).clamp_min(self.eps)
        inv_sqrt = degree.rsqrt()
        normalized = inv_sqrt.unsqueeze(-1) * with_loop * inv_sqrt.unsqueeze(-2)
        laplacian = eye - normalized
        scaled = laplacian - eye
        return torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, *, precomputed: bool = False) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"node features must be [B,N,F], got {tuple(x.shape)}")
        if adjacency.shape[:2] != x.shape[:2] or adjacency.shape[-1] != x.shape[1]:
            raise ValueError(f"node/graph mismatch: x={tuple(x.shape)}, adjacency={tuple(adjacency.shape)}")
        scaled = adjacency if precomputed else self.scaled_laplacian(adjacency)
        terms = [x]
        if self.order > 1:
            terms.append(torch.bmm(scaled, x))
        for _ in range(2, self.order):
            terms.append(2.0 * torch.bmm(scaled, terms[-1]) - terms[-2])
        out = sum(layer(term) for layer, term in zip(self.transforms, terms)) + self.bias
        out = self.norm(out + self.residual(x))
        return self.dropout(self.activation(out))


class BandGraphEncoder(nn.Module):
    """Encode one band graph using stacked ChebGraphConv and mean+max readout."""

    def __init__(self, input_dim: int = 1, graph_hidden_dim: int = 64, band_embed_dim: int = 64,
                 cheb_order: int = 3, num_graph_layers: int = 2, dropout: float = 0.3,
                 eps: float = 1e-6) -> None:
        super().__init__()
        if num_graph_layers < 1:
            raise ValueError("num_graph_layers must be at least one")
        dims = [int(input_dim)] + [int(graph_hidden_dim)] * int(num_graph_layers)
        self.layers = nn.ModuleList(
            ChebGraphConv(dims[i], dims[i + 1], cheb_order, dropout, eps)
            for i in range(num_graph_layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(graph_hidden_dim * 2, band_embed_dim),
            nn.LayerNorm(band_embed_dim), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        scaled = self.layers[0].scaled_laplacian(adjacency)
        h = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        for layer in self.layers:
            h = layer(h, scaled, precomputed=True)
        pooled = torch.cat((h.mean(dim=1), h.max(dim=1).values), dim=-1)
        return self.readout(pooled)


class DEPLVGraphBackbone(nn.Module):
    """Encode a batch of windows from aligned five-band DE and PLV features."""

    def __init__(
        self,
        num_nodes: int = 30,
        num_bands: int = 5,
        graph_hidden_dim: int = 64,
        band_embed_dim: int = 64,
        window_embed_dim: int = 128,
        cheb_order: int = 3,
        num_graph_layers: int = 2,
        dropout: float = 0.3,
        use_subject_relative_de: bool = True,
        relative_eps: float = 1e-6,
        share_band_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_bands = int(num_bands)
        self.band_embed_dim = int(band_embed_dim)
        self.relative_eps = float(relative_eps)
        self.use_subject_relative_de = bool(use_subject_relative_de)
        self.share_band_encoder = bool(share_band_encoder)
        encoder_kwargs = dict(input_dim=1, graph_hidden_dim=graph_hidden_dim,
                              band_embed_dim=band_embed_dim, cheb_order=cheb_order,
                              num_graph_layers=num_graph_layers, dropout=dropout,
                              eps=relative_eps)
        if self.share_band_encoder:
            self.band_encoder = BandGraphEncoder(**encoder_kwargs)
        else:
            self.band_encoders = nn.ModuleList(BandGraphEncoder(**encoder_kwargs) for _ in range(self.num_bands))
        self.emotion_band_attention = self._attention(band_embed_dim, dropout)
        self.diagnosis_band_attention = self._attention(band_embed_dim, dropout)
        self.emotion_projection = self._projection(band_embed_dim, window_embed_dim, dropout)
        self.diagnosis_projection = self._projection(band_embed_dim, window_embed_dim, dropout)
        self.out_dim = int(window_embed_dim)
        self.core_dim = int(window_embed_dim)
        self.use_biomarkers = False

    @staticmethod
    def _attention(dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Dropout(dropout), nn.Linear(dim, 1))

    @staticmethod
    def _projection(in_dim: int, out_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout))

    def _validate(self, de_feat: torch.Tensor, plv_feat: torch.Tensor) -> None:
        expected_de = (self.num_nodes, self.num_bands)
        expected_plv = (self.num_bands, self.num_nodes, self.num_nodes)
        if de_feat.ndim != 3 or tuple(de_feat.shape[1:]) != expected_de:
            raise ValueError(f"de_feat must be [M,{self.num_nodes},{self.num_bands}], got {tuple(de_feat.shape)}")
        if plv_feat.ndim != 4 or tuple(plv_feat.shape[1:]) != expected_plv:
            raise ValueError(
                f"plv_feat must be [M,{self.num_bands},{self.num_nodes},{self.num_nodes}], "
                f"got {tuple(plv_feat.shape)}"
            )
        if de_feat.shape[0] != plv_feat.shape[0]:
            raise ValueError(f"DE/PLV window counts differ: {de_feat.shape[0]} vs {plv_feat.shape[0]}")

    def _encode_bands(self, de: torch.Tensor, plv: torch.Tensor) -> torch.Tensor:
        # [M,N,F] -> [M,F,N,1], then flatten M and F for the shared encoder.
        nodes = de.permute(0, 2, 1).unsqueeze(-1).contiguous()
        if self.share_band_encoder:
            m = nodes.shape[0]
            encoded = self.band_encoder(
                nodes.reshape(m * self.num_bands, self.num_nodes, 1),
                plv.reshape(m * self.num_bands, self.num_nodes, self.num_nodes),
            )
            return encoded.reshape(m, self.num_bands, self.band_embed_dim)
        return torch.stack(
            [self.band_encoders[band](nodes[:, band], plv[:, band]) for band in range(self.num_bands)],
            dim=1,
        )

    @staticmethod
    def _fuse(bands: torch.Tensor, attention: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.softmax(attention(bands).squeeze(-1), dim=1)
        fused = torch.sum(weight.unsqueeze(-1) * bands, dim=1)
        return fused, weight

    def forward(
        self,
        de_feat: torch.Tensor,
        plv_feat: torch.Tensor,
        subject_de_mu: torch.Tensor | None = None,
        subject_de_std: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        self._validate(de_feat, plv_feat)
        de_abs = torch.nan_to_num(de_feat, nan=0.0, posinf=0.0, neginf=0.0)
        plv = ChebGraphConv.sanitize_adjacency(plv_feat.flatten(0, 1)).reshape_as(plv_feat)
        de_rel = de_abs
        if self.use_subject_relative_de and subject_de_mu is not None and subject_de_std is not None:
            if subject_de_mu.shape != de_abs.shape or subject_de_std.shape != de_abs.shape:
                raise ValueError(
                    "subject DE baseline shape mismatch: "
                    f"de={tuple(de_abs.shape)}, mu={tuple(subject_de_mu.shape)}, std={tuple(subject_de_std.shape)}"
                )
            mu = subject_de_mu.to(device=de_abs.device, dtype=de_abs.dtype)
            std = subject_de_std.to(device=de_abs.device, dtype=de_abs.dtype).clamp_min(self.relative_eps)
            de_rel = torch.nan_to_num((de_abs - mu) / std, nan=0.0, posinf=0.0, neginf=0.0)

        bands_emotion = self._encode_bands(de_rel, plv)
        bands_diagnosis = self._encode_bands(de_abs, plv)
        fused_emotion, attention_emotion = self._fuse(bands_emotion, self.emotion_band_attention)
        fused_diagnosis, attention_diagnosis = self._fuse(bands_diagnosis, self.diagnosis_band_attention)
        z_emotion = self.emotion_projection(fused_emotion)
        z_diag = self.diagnosis_projection(fused_diagnosis)
        return {
            "z": z_emotion, "z_emotion": z_emotion, "z_diag": z_diag,
            "frequency_attention_emotion": attention_emotion,
            "frequency_attention_diag": attention_diagnosis,
            "band_embeddings": bands_emotion,
            "band_embeddings_emotion": bands_emotion,
            "band_embeddings_diag": bands_diagnosis,
            "de_abs": de_abs, "de_rel": de_rel,
        }

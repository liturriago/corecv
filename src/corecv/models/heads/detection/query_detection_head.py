"""Query-based detection head (RT-DETR / D-FINE style).

Implements a transformer-decoder detection head with learnable object queries
that attend to multi-scale feature maps produced by the backbone or neck.
Each query directly regresses a bounding box and produces class logits,
eliminating the need for non-maximum suppression (NMS) at inference time.

Architecture overview::

    features: [f_stride4, f_stride8, f_stride16, f_stride32]
      |
      v  (flatten + project to d_model per level)
    multi_scale_features: (B, S_total, d_model)
      |
      v
    TransformerDecoder (num_layers x DecoderLayer)
      |
      v
    query_features: (B, num_queries, d_model)
      |
      +──> cls_head ──> (B, num_queries, num_classes)
      |
      └──> reg_head ──> (B, num_queries, 4)   — (cx, cy, w, h) normalised

During training, bipartite matching (Hungarian) is used to align predictions
to ground-truth boxes.  During inference the top-K predictions by
classification score are returned directly (NMS-free).

The head dynamically inspects ``FeatureInfo.channels`` to build per-level
input projections, making it transparent to backbone and neck changes.

Example:
    >>> from corecv.core.contract import FeatureInfo
    >>> from corecv.models.heads.detection.query_detection_head import (
    ...     QueryDetectionHead,
    ... )
    >>> fi = FeatureInfo(
    ...     channels={"stride4": 256, "stride8": 512,
    ...               "stride16": 1024, "stride32": 2048},
    ...     strides={"stride4": 4, "stride8": 8,
    ...              "stride16": 16, "stride32": 32},
    ... )
    >>> head = QueryDetectionHead(
    ...     feature_info=fi,
    ...     num_classes=80,
    ...     d_model=256,
    ...     num_queries=300,
    ...     num_decoder_layers=6,
    ...     num_heads=8,
    ... )
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from corecv.core.contract import FeatureInfo
from corecv.core.registry import register_head

# ---------------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """Multi-layer perceptron with GELU activation.

    Args:
        in_features: Input dimension.
        hidden_features: Hidden layer dimension.
        out_features: Output dimension.
        num_layers: Number of linear layers (>= 2).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_features
        for i in range(num_layers - 1):
            next_dim = hidden_features if i == 0 else hidden_features
            layers.append(nn.Linear(current, next_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = next_dim
        layers.append(nn.Linear(current, out_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(*, in_features)``.

        Returns:
            Output tensor of shape ``(*, out_features)``.
        """
        return self.layers(x)


class _MSFeatureProjector(nn.Module):
    """Project and concatenate multi-scale features into a flat sequence.

    Each feature level is spatially flattened and linearly projected to
    ``d_model``, then all levels are concatenated along the sequence
    dimension.

    Args:
        in_channels_per_level: List of channel counts per feature level.
        d_model: Target embedding dimension.
    """

    def __init__(
        self,
        in_channels_per_level: list[int],
        d_model: int,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(ch, d_model, kernel_size=1)
                for ch in in_channels_per_level
            ]
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        """Project and flatten multi-scale features.

        Args:
            features: List of feature tensors, one per level, each of
                shape ``(B, C_l, H_l, W_l)``.

        Returns:
            Concatenated sequence of shape ``(B, S_total, d_model)`` where
            ``S_total = sum(H_l * W_l)`` across all levels.
        """
        projected: list[Tensor] = []
        for level_feat, proj in zip(features, self.projections, strict=True):
            # (B, C_l, H_l, W_l) -> (B, d_model, H_l, W_l)
            projected_feat = proj(level_feat)
            # (B, d_model, H_l, W_l) -> (B, H_l*W_l, d_model)
            projected.append(projected_feat.flatten(2).transpose(1, 2))
        return torch.cat(projected, dim=1)  # (B, S_total, d_model)


class _TransformerDecoderLayer(nn.Module):
    """Single decoder layer with cross-attention to multi-scale features.

    Pre-norm Transformer decoder layer: self-attention -> cross-attention
    (to multi-scale features) -> FFN.

    Args:
        d_model: Embedding dimension.
        num_heads: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention to multi-scale features
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of a single decoder layer.

        Args:
            tgt: Query features of shape ``(B, N_q, d_model)``.
            memory: Multi-scale features of shape ``(B, S, d_model)``.
            tgt_mask: Optional self-attention mask.
            memory_mask: Optional cross-attention mask.

        Returns:
            Updated query features of shape ``(B, N_q, d_model)``.
        """
        # Self-attention with pre-norm
        q = self.norm1(tgt)
        tgt = tgt + self.dropout1(
            self.self_attn(q, q, q, attn_mask=tgt_mask, need_weights=False)[0]
        )

        # Cross-attention with pre-norm
        q2 = self.norm2(tgt)
        tgt = tgt + self.dropout2(
            self.cross_attn(
                q2, memory, memory, attn_mask=memory_mask, need_weights=False,
            )[0]
        )

        # Feed-forward with pre-norm
        ffn_out = self.linear2(
            self.dropout3(self.activation(self.linear1(self.norm3(tgt))))
        )
        tgt = tgt + self.dropout4(ffn_out)

        return tgt


# ---------------------------------------------------------------------------
# Main head
# ---------------------------------------------------------------------------


@register_head("query_detection")
class QueryDetectionHead(nn.Module):
    """RT-DETR / D-FINE style query-based detection head.

    Learns a set of fixed object queries that attend to multi-scale features
    via a transformer decoder.  Each query directly regresses a normalised
    ``(cx, cy, w, h)`` bounding box and produces class logits, enabling
    NMS-free inference.

    The head dynamically inspects ``FeatureInfo.channels`` to construct
    per-level input projections, making it transparent to backbone and neck
    architecture changes.

    Args:
        feature_info: Feature metadata from the backbone or neck.  Per-level
            projections are built from ``feature_info.channels``.
        num_classes: Number of foreground object classes (excluding
            background).
        d_model: Transformer hidden dimension.  Default ``256``.
        num_queries: Number of learnable object queries.  Default ``300``.
        num_decoder_layers: Depth of the transformer decoder.  Default ``6``.
        num_heads: Number of attention heads in the decoder.  Default ``8``.
        dim_feedforward: FFN hidden dimension in the decoder.  Default
            ``2048``.
        dropout: Dropout probability.  Default ``0.1``.
        num_reg_fcs: Number of FC layers in the box regression MLP.
            Default ``3``.
        return_intermediate: If ``True``, forward returns decoder
            intermediate outputs (useful for auxiliary losses).  Default
            ``False``.
    """

    def __init__(  # noqa: PLR0913
        self,
        feature_info: FeatureInfo,
        num_classes: int,
        d_model: int = 256,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_reg_fcs: int = 3,
        return_intermediate: bool = False,
    ) -> None:
        """Initialise the query-based detection head.

        Builds multi-scale feature projections, learnable object queries, a
        transformer decoder stack, and classification / regression heads.
        """
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.return_intermediate = return_intermediate

        # --- Feature-level metadata (preserves insertion order) ---
        self._level_names: list[str] = list(feature_info.channels.keys())
        self._level_channels: list[int] = [
            feature_info.channels[k] for k in self._level_names
        ]
        self._level_strides: list[int] = [
            feature_info.strides[k] for k in self._level_names
        ]

        # --- Learnable object queries ---
        self.query_embed = nn.Embedding(num_queries, d_model)

        # --- Multi-scale feature projection ---
        self.ms_projector = _MSFeatureProjector(
            in_channels_per_level=self._level_channels,
            d_model=d_model,
        )

        # --- Positional encoding for queries ---
        # Learned positional bias added to queries at each decoder layer.
        self.query_pos_embed = nn.Embedding(num_queries, d_model)

        # --- Transformer decoder ---
        self.decoder_layers = nn.ModuleList(
            [
                _TransformerDecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(d_model)

        # --- Classification head ---
        self.cls_head = _MLP(
            in_features=d_model,
            hidden_features=d_model,
            out_features=num_classes,
            num_layers=3,
            dropout=dropout,
        )

        # --- Box regression head ---
        self.reg_head = _MLP(
            in_features=d_model,
            hidden_features=d_model,
            out_features=4,
            num_layers=num_reg_fcs,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise learnable parameters."""
        nn.init.uniform_(self.query_embed.weight, -0.08, 0.08)
        nn.init.uniform_(self.query_pos_embed.weight, -0.08, 0.08)

        # Xavier init for linear layers in the heads and decoder
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def num_levels(self) -> int:
        """Return the number of feature levels."""
        return len(self._level_names)

    @property
    def level_names(self) -> list[str]:
        """Return the ordered list of feature level names."""
        return list(self._level_names)

    @property
    def level_strides(self) -> list[int]:
        """Return the ordered list of per-level stride values."""
        return list(self._level_strides)

    def forward(
        self,
        features: list[Tensor],
    ) -> dict[str, Tensor | list[Tensor]]:
        """Run query-based detection on multi-scale features.

        Args:
            features: List of feature tensors from the backbone or neck,
                one per level, ordered from finest to coarsest spatial
                resolution.  ``len(features)`` must equal
                ``self.num_levels``.

        Returns:
            Dictionary with keys:
                - ``"cls_logits"``: ``(B, num_queries, num_classes)``
                - ``"pred_boxes"``: ``(B, num_queries, 4)`` normalised
                  ``(cx, cy, w, h)`` in ``[0, 1]`` range.

            If ``return_intermediate`` is ``True``, the dictionary also
            contains:
                - ``"intermediate_cls"``: list of ``(B, N_q, num_classes)``
                - ``"intermediate_reg"``: list of ``(B, N_q, 4)``
        """
        if len(features) != self.num_levels:
            msg = (
                f"Expected {self.num_levels} feature levels, "
                f"got {len(features)}."
            )
            raise ValueError(msg)

        B = features[0].shape[0]

        # Flatten and project multi-scale features: (B, S, d_model)
        memory = self.ms_projector(features)

        # Initialise queries: (B, num_queries, d_model)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        query_pos = self.query_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Decode
        intermediate_cls: list[Tensor] = []
        intermediate_reg: list[Tensor] = []

        for layer in self.decoder_layers:
            queries = layer(queries + query_pos, memory)
            if self.return_intermediate:
                intermediate_cls.append(self.cls_head(queries))
                intermediate_reg.append(self.reg_head(queries).sigmoid())

        queries = self.decoder_norm(queries)  # (B, N_q, d_model)

        # Final predictions
        cls_logits = self.cls_head(queries)          # (B, N_q, num_classes)
        pred_boxes = self.reg_head(queries).sigmoid()  # (B, N_q, 4) in [0, 1]

        result: dict[str, Tensor | list[Tensor]] = {
            "cls_logits": cls_logits,
            "pred_boxes": pred_boxes,
        }

        if self.return_intermediate:
            result["intermediate_cls"] = intermediate_cls
            result["intermediate_reg"] = intermediate_reg

        return result

    def get_reference_points(
        self,
        features: list[Tensor],
    ) -> Tensor:
        """Generate normalised reference points for each spatial location.

        Produces a grid of ``cx, cy`` coordinates in ``[0, 1]`` for each
        feature level.  This is useful for downstream box refinement (e.g.
        D-FINE style iterative decoding).

        Args:
            features: Feature tensors (used only for spatial dimensions).

        Returns:
            Reference points of shape ``(B, S_total, 2)`` where
            ``S_total = sum(H_l * W_l)``.
        """
        reference_points: list[Tensor] = []
        for feat in features:
            H, W = feat.shape[2], feat.shape[3]
            # Generate normalised cy, cx grid
            cy = (torch.arange(H, device=feat.device, dtype=feat.dtype) + 0.5) / H
            cx = (torch.arange(W, device=feat.device, dtype=feat.dtype) + 0.5) / W
            grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")
            # (2, H, W) -> (H*W, 2)
            grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            # Broadcast to batch: (1, H*W, 2)
            reference_points.append(grid.unsqueeze(0))

        # (1, S_total, 2) -> (B, S_total, 2)
        ref = torch.cat(reference_points, dim=1)
        return ref.expand(features[0].shape[0], -1, -1)

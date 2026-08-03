"""Swin Transformer backbone for CoreCV.

Provides Swin Transformer variants (Tiny, Small, Base) as hierarchical
vision transformer feature extractors with ``FeatureInfo`` metadata.

Swin Transformer uses shifted windows for efficient self-attention and
produces hierarchical features via patch merging at each stage.
Features are extracted at four stages with strides 4x, 8x, 16x, 32x.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torchvision.models import (
    Swin_B_Weights,
    Swin_S_Weights,
    Swin_T_Weights,
    swin_b,
    swin_s,
    swin_t,
)

from corecv.models.backbones.base import BaseBackbone, FeatureInfo

# Type alias for supported Swin Transformer variants.
SwinVariant = Literal["swin_t", "swin_s", "swin_b"]

# Maps variant name -> (constructor, default weights factory).
_SWIN_REGISTRY: dict[str, tuple[type, type]] = {
    "swin_t": (swin_t, Swin_T_Weights),
    "swin_s": (swin_s, Swin_S_Weights),
    "swin_b": (swin_b, Swin_B_Weights),
}

# Number of hierarchical stages in Swin Transformer.
_NUM_SWIN_STAGES = 4

# Dimensionality constants for tensor shape handling.
_DIM_3D = 3  # (B, N, C) format from attention
_DIM_4D = 4  # (B, H, W, C) or (B, C, H, W) format


def _to_nchw(x: Tensor) -> Tensor:
    """Convert Swin Transformer output to (B, C, H, W) channel-first format.

    TorchVision Swin stages may output either:
    - (B, H*W, C) -- flattened sequence format (3D)
    - (B, H, W, C) -- spatial format (4D, channel-last)

    This helper normalizes both to (B, C, H, W).

    Args:
        x: Feature tensor from a Swin stage.

    Returns:
        Tensor in (B, C, H, W) format.
    """
    if x.dim() == _DIM_3D:
        b, n, c = x.shape
        h = w = int(n**0.5)
        return x.transpose(1, 2).reshape(b, c, h, w)
    if x.dim() == _DIM_4D and x.shape[-1] != x.shape[1]:
        # (B, H, W, C) -> (B, C, H, W)
        return x.permute(0, 3, 1, 2)
    return x


class SwinTransformerBackbone(BaseBackbone):
    """Swin Transformer multi-scale feature extractor.

    Wraps a TorchVision ``Swin Transformer`` model and exposes four
    feature levels (C2-C5) extracted after each hierarchical stage.

    Note:
        TorchVision Swin models return a single flattened feature by
        default.  This backbone hooks into the internal stages to
        extract multi-scale features.

    Example:
        >>> import torch
        >>> from corecv.models.backbones.swin import SwinTransformerBackbone
        >>> backbone = SwinTransformerBackbone("swin_t", pretrained=False)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> features, info = backbone(x)
        >>> [f.shape for f in features]
        [torch.Size([2, 96, 56, 56]), torch.Size([2, 192, 28, 28]),
         torch.Size([2, 384, 14, 14]), torch.Size([2, 768, 7, 7])]
    """

    def __init__(
        self,
        variant: SwinVariant = "swin_t",
        *,
        pretrained: bool = False,
    ) -> None:
        """Initialize the Swin Transformer backbone.

        Args:
            variant: Swin variant name. One of ``swin_t``, ``swin_s``,
                ``swin_b``.
            pretrained: If ``True``, load ImageNet-1K pretrained weights.
        """
        if variant not in _SWIN_REGISTRY:
            msg = f"Unknown Swin variant: {variant!r}. Choose from {list(_SWIN_REGISTRY)}"
            raise ValueError(msg)

        constructor, weights_cls = _SWIN_REGISTRY[variant]
        weights = weights_cls.DEFAULT if pretrained else None
        model = constructor(weights=weights)

        # Extract stage/merge components for channel inference.
        _stem = model.features[0]
        _stages = nn.ModuleList(
            [
                model.features[1],
                model.features[3],
                model.features[5],
                model.features[7],
            ],
        )
        _merges = nn.ModuleList(
            [
                model.features[2],
                model.features[4],
                model.features[6],
            ],
        )

        # Infer channel dimensions via dummy forward pass.
        channels = self._infer_channels_static(_stem, _stages, _merges)

        feature_info = FeatureInfo(
            channels=channels,
            strides=[4, 8, 16, 32],
            names=["C2", "C3", "C4", "C5"],
        )
        super().__init__(feature_info=feature_info)

        # Assign extracted components as submodules after super().__init__.
        self._stem = _stem
        self._stages = _stages
        self._merges = _merges

        # Safely extract norm layers (handle models without norms attribute).
        _norm_list: list[nn.Module] = []
        _model_norms = getattr(model, "norms", None)
        for idx in range(_NUM_SWIN_STAGES):
            if _model_norms is not None and len(_model_norms) > idx:
                _norm_list.append(_model_norms[idx])
            else:
                _norm_list.append(nn.Identity())
        self._norms = nn.ModuleList(_norm_list)

    @staticmethod
    def _infer_channels_static(
        stem: nn.Module,
        stages: nn.ModuleList,
        merges: nn.ModuleList,
    ) -> list[int]:
        """Infer output channel counts via a dummy forward pass.

        Args:
            stem: Patch embedding stem module.
            stages: List of Swin Transformer stage modules.
            merges: List of PatchMerging modules.

        Returns:
            List of four channel counts for stages 0-3.
        """
        dummy = torch.randn(1, 3, 224, 224)
        x = stem(dummy)

        channels: list[int] = []
        last_merge_idx = _NUM_SWIN_STAGES - 1
        for stage_idx in range(_NUM_SWIN_STAGES):
            x = stages[stage_idx](x)

            # Read channel count (last dim in both NHWC and NCHW formats).
            channels.append(x.shape[-1])
            if stage_idx < last_merge_idx:
                x = merges[stage_idx](x)

        return channels

    def forward(self, x: Tensor) -> tuple[list[Tensor], FeatureInfo]:
        """Extract multi-scale features from the input tensor.

        Args:
            x: Input image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Tuple of ``(features, feature_info)`` where *features* is a
            list of four feature tensors with strides 4x, 8x, 16x, 32x,
            and *feature_info* contains channel/stride metadata.
        """
        x = self._stem(x)

        features: list[Tensor] = []
        last_merge_idx = _NUM_SWIN_STAGES - 1
        for stage_idx in range(_NUM_SWIN_STAGES):
            x = self._stages[stage_idx](x)

            # Convert to (B, C, H, W) channel-first for downstream compatibility.
            feat_nchw = _to_nchw(x)
            features.append(feat_nchw)

            if stage_idx < last_merge_idx:
                x = self._merges[stage_idx](x)

        return features, self.feature_info
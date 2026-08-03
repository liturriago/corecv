"""ConvNeXt backbone for CoreCV.

Provides ConvNeXt variants (Tiny, Small, Base, Large) as modernized
ConvNet feature extractors with ``FeatureInfo`` metadata.

ConvNeXt uses a hierarchical design with four stages, each consisting
of a downsampling (patchify) layer followed by multiple ConvNeXt blocks.
Features are extracted after each stage with strides 4x, 8x, 16x, 32x.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_base,
    convnext_small,
    convnext_tiny,
)

from corecv.models.backbones.base import BaseBackbone, FeatureInfo

# Type alias for supported ConvNeXt variants.
ConvNeXtVariant = Literal["convnext_tiny", "convnext_small", "convnext_base"]

# Maps variant name -> (constructor, default weights factory).
_CONVNEXT_REGISTRY: dict[str, tuple[type, type]] = {
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights),
    "convnext_base": (convnext_base, ConvNeXt_Base_Weights),
}

# ConvNeXt features layout (torchvision >= 0.15):
#   [0] = Stem (Conv2dNormActivation)          -> stride 4
#   [1] = Stage 0 blocks (Sequential)          -> stride 4  (no downsample)
#   [2] = Stage 1 downsample + blocks          -> stride 8
#   [3] = Stage 1 blocks continued             -> stride 8
#   [4] = Stage 2 downsample + blocks          -> stride 16
#   [5] = Stage 2 blocks continued             -> stride 16
#   [6] = Stage 3 downsample + blocks          -> stride 32
#   [7] = Stage 3 blocks continued             -> stride 32
# We extract features after the last block of each stage: indices [1, 3, 5, 7].
_EXTRACT_INDICES: list[int] = [1, 3, 5, 7]


class ConvNeXtBackbone(BaseBackbone):
    """ConvNeXt multi-scale feature extractor.

    Wraps a TorchVision ``ConvNeXt`` model and exposes four feature levels
    (C2-C5) extracted after each of the four hierarchical stages.

    Example:
        >>> import torch
        >>> from corecv.models.backbones.convnext import ConvNeXtBackbone
        >>> backbone = ConvNeXtBackbone("convnext_tiny", pretrained=False)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> features, info = backbone(x)
        >>> [f.shape for f in features]
        [torch.Size([2, 96, 56, 56]), torch.Size([2, 192, 28, 28]),
         torch.Size([2, 384, 14, 14]), torch.Size([2, 768, 7, 7])]
    """

    def __init__(
        self,
        variant: ConvNeXtVariant = "convnext_tiny",
        *,
        pretrained: bool = False,
    ) -> None:
        """Initialize the ConvNeXt backbone.

        Args:
            variant: ConvNeXt variant name. One of ``convnext_tiny``,
                ``convnext_small``, ``convnext_base``.
            pretrained: If ``True``, load ImageNet-1K pretrained weights.
        """
        if variant not in _CONVNEXT_REGISTRY:
            msg = f"Unknown ConvNeXt variant: {variant!r}. Choose from {list(_CONVNEXT_REGISTRY)}"
            raise ValueError(msg)

        constructor, weights_cls = _CONVNEXT_REGISTRY[variant]
        weights = weights_cls.DEFAULT if pretrained else None
        model = constructor(weights=weights)

        # Infer channel dimensions via dummy forward pass on model features.
        channels = self._infer_channels_static(model.features)

        feature_info = FeatureInfo(
            channels=channels,
            strides=[4, 8, 16, 32],
            names=["C2", "C3", "C4", "C5"],
        )
        super().__init__(feature_info=feature_info)

        # Assign component as submodule after super().__init__.
        self.features: nn.Sequential = model.features

    @staticmethod
    def _infer_channels_static(features: nn.Sequential) -> list[int]:
        """Infer output channel counts via a dummy forward pass.

        Args:
            features: The ConvNeXt features sequential module.

        Returns:
            List of four channel counts for each stage.
        """
        dummy = torch.randn(1, 3, 224, 224)
        x = dummy
        for stage_idx in range(len(features)):
            x = features[stage_idx](x)
        # Extract channels at the end of each stage.
        channels: list[int] = []
        dummy2 = torch.randn(1, 3, 224, 224)
        x2 = dummy2
        for stage_idx in range(len(features)):
            x2 = features[stage_idx](x2)
            if stage_idx in _EXTRACT_INDICES:
                channels.append(x2.shape[1])
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
        features: list[Tensor] = []
        for stage_idx in range(len(self.features)):
            x = self.features[stage_idx](x)
            if stage_idx in _EXTRACT_INDICES:
                features.append(x)

        return features, self.feature_info

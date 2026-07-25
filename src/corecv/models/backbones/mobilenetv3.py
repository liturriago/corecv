"""MobileNetV3 backbone for CoreCV.

Provides MobileNetV3-Small and MobileNetV3-Large variants as lightweight
feature extractors with ``FeatureInfo`` metadata.

MobileNetV3 uses an inverted-residual architecture with squeeze-and-excitation
blocks.  Features are extracted at three stages with approximate strides
8x, 16x, and 32x.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)

from corecv.models.backbones.base import BaseBackbone, FeatureInfo

# Type alias for supported MobileNetV3 variants.
MobileNetV3Variant = Literal["mobilenetv3_large", "mobilenetv3_small"]

# Maps variant name -> (constructor, default weights factory).
_MOBILENET_REGISTRY: dict[str, tuple[type, type]] = {
    "mobilenetv3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights),
    "mobilenetv3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights),
}


class MobileNetV3Backbone(BaseBackbone):
    """MobileNetV3 multi-scale feature extractor.

    Wraps a TorchVision ``MobileNetV3`` model and exposes features at
    three key stages (approximately strides 8x, 16x, 32x).

    The specific block indices where features are extracted are chosen to
    match common detection/segmentation conventions:

    - **Large**: blocks 3, 8, 14 (before downsampling transitions).
    - **Small**: blocks 1, 3, 8.

    Example:
        >>> import torch
        >>> from corecv.models.backbones.mobilenetv3 import MobileNetV3Backbone
        >>> backbone = MobileNetV3Backbone("mobilenetv3_large", pretrained=False)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> features, info = backbone(x)
        >>> [f.shape for f in features]
        [torch.Size([2, 40, 28, 28]), torch.Size([2, 96, 14, 14]),
         torch.Size([2, 960, 7, 7])]
    """

    def __init__(
        self,
        variant: MobileNetV3Variant = "mobilenetv3_large",
        *,
        pretrained: bool = False,
    ) -> None:
        """Initialize the MobileNetV3 backbone.

        Args:
            variant: MobileNetV3 variant name. One of ``mobilenetv3_large``,
                ``mobilenetv3_small``.
            pretrained: If ``True``, load ImageNet-1K pretrained weights.
        """
        if variant not in _MOBILENET_REGISTRY:
            msg = (
                f"Unknown MobileNetV3 variant: {variant!r}. Choose from {list(_MOBILENET_REGISTRY)}"
            )
            raise ValueError(msg)

        constructor, weights_cls = _MOBILENET_REGISTRY[variant]
        weights = weights_cls.DEFAULT if pretrained else None
        model = constructor(weights=weights)

        # Determine extraction indices based on variant.
        if variant == "mobilenetv3_large":
            _extract_indices: list[int] = [3, 8, 14]
            _strides: list[int] = [8, 16, 32]
            _names: list[str] = ["C3", "C4", "C5"]
        else:
            _extract_indices: list[int] = [1, 3, 8]
            _strides: list[int] = [8, 16, 32]
            _names: list[str] = ["C3", "C4", "C5"]

        # Infer channel dimensions via dummy forward pass on model features.
        channels = self._infer_channels_static(model.features, _extract_indices)

        feature_info = FeatureInfo(
            channels=channels,
            strides=_strides,
            names=_names,
        )
        super().__init__(feature_info=feature_info)

        # Assign components as submodules after super().__init__.
        self.features: nn.Sequential = model.features
        self._extract_indices = _extract_indices

    @staticmethod
    def _infer_channels_static(
        features: nn.Sequential,
        extract_indices: list[int],
    ) -> list[int]:
        """Infer output channel counts via a dummy forward pass.

        Args:
            features: The MobileNetV3 features sequential module.
            extract_indices: Block indices at which to extract features.

        Returns:
            List of channel counts at each extraction point.
        """
        dummy = torch.randn(1, 3, 224, 224)
        channels: list[int] = []
        x = dummy
        for idx, block in enumerate(features):
            x = block(x)
            if idx in extract_indices:
                channels.append(x.shape[1])
        return channels

    def forward(self, x: Tensor) -> tuple[list[Tensor], FeatureInfo]:
        """Extract multi-scale features from the input tensor.

        Args:
            x: Input image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Tuple of ``(features, feature_info)`` where *features* is a
            list of feature tensors and *feature_info* contains metadata.
        """
        features: list[Tensor] = []
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in self._extract_indices:
                features.append(x)

        return features, self.feature_info


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    for variant in ("mobilenetv3_large", "mobilenetv3_small"):
        backbone = MobileNetV3Backbone(variant=variant, pretrained=False)
        backbone.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features, info = backbone(dummy_input)

        print(f"--- {variant} ---")  # noqa: T201
        for i, (feat, ch, stride) in enumerate(
            zip(features, info.channels, info.strides, strict=True),
        ):
            print(  # noqa: T201
                f"  Level {i} ({info.names[i]}): "
                f"channels={ch}, stride={stride}, "
                f"shape={feat.shape}",
            )

        assert [f.shape[1] for f in features] == info.channels, "Channel mismatch!"  # noqa: S101
    print("\nAll MobileNetV3 backbone tests passed.")  # noqa: T201

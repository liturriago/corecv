"""ResNet backbone with multi-scale feature extraction.

Wraps :mod:`torchvision.models` ResNet variants (18, 34, 50, 101) and
extracts intermediate feature maps from ``layer1`` through ``layer4`` at
strides 4, 8, 16, and 32 respectively.

Extracted feature levels (verified against torchvision output shapes):

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 15 15 15 15

   * - Model
     - Level
     - Stride
     - ResNet-18
     - ResNet-34
     - ResNet-50
     - ResNet-101
   * - layer1
     - stride4
     - 4
     - 64
     - 64
     - 256
     - 256
   * - layer2
     - stride8
     - 8
     - 128
     - 128
     - 512
     - 512
   * - layer3
     - stride16
     - 16
     - 256
     - 256
     - 1024
     - 1024
   * - layer4
     - stride32
     - 32
     - 512
     - 512
     - 2048
     - 2048

Example:
    >>> from corecv.models.backbones.resnet import ResNet50Backbone
    >>> backbone = ResNet50Backbone(pretrained=False)
    >>> backbone.feature_info.channels
    {'stride4': 256, 'stride8': 512, 'stride16': 1024, 'stride32': 2048}
"""

from __future__ import annotations

from typing import Any

import torch
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
)

from corecv.core.contract import BaseBackbone, FeatureInfo
from corecv.core.registry import register_backbone

# ---------------------------------------------------------------------------
# Channel configuration per ResNet variant
# ---------------------------------------------------------------------------

_RESNET_VARIANTS: dict[str, dict[str, Any]] = {
    "resnet18": {
        "factory": resnet18,
        "weights": ResNet18_Weights,
        "channels": {"stride4": 64, "stride8": 128, "stride16": 256, "stride32": 512},
    },
    "resnet34": {
        "factory": resnet34,
        "weights": ResNet34_Weights,
        "channels": {"stride4": 64, "stride8": 128, "stride16": 256, "stride32": 512},
    },
    "resnet50": {
        "factory": resnet50,
        "weights": ResNet50_Weights,
        "channels": {"stride4": 256, "stride8": 512, "stride16": 1024, "stride32": 2048},
    },
    "resnet101": {
        "factory": resnet101,
        "weights": ResNet101_Weights,
        "channels": {"stride4": 256, "stride8": 512, "stride16": 1024, "stride32": 2048},
    },
}


class _ResNetBackbone(BaseBackbone):
    """Shared base for all ResNet backbone variants.

    Runs the stem (conv1 -> bn1 -> relu -> maxpool) and then captures the
    outputs of ``layer1`` through ``layer4``.

    This class should not be instantiated directly; use the variant-specific
    subclasses (:class:`ResNet18Backbone`, etc.).
    """

    _variant_key: str

    def __init__(self, pretrained: bool = True, **kwargs: object) -> None:
        """Initialise the ResNet backbone.

        Args:
            pretrained: If ``True``, load ImageNet-1K pretrained weights.
            **kwargs: Additional keyword arguments forwarded to the
                underlying ResNet factory function.
        """
        super().__init__()
        cfg = _RESNET_VARIANTS[self._variant_key]
        weights = cfg["weights"].IMAGENET1K_V1 if pretrained else None
        self._model = cfg["factory"](weights=weights, **kwargs)

    @property
    def feature_info(self) -> FeatureInfo:
        """Return channel and stride metadata for all extracted feature levels.

        Returns:
            A :class:`FeatureInfo` with keys ``stride4``, ``stride8``,
            ``stride16``, and ``stride32``.
        """
        cfg = _RESNET_VARIANTS[self._variant_key]
        return FeatureInfo(
            channels=dict(cfg["channels"]),
            strides={
                "stride4": 4,
                "stride8": 8,
                "stride16": 16,
                "stride32": 32,
            },
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features from the ResNet backbone.

        Runs the stem (conv1 -> bn1 -> relu -> maxpool) followed by the
        four residual layers.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            A list of four feature tensors from layer1 through layer4 at
            strides 4, 8, 16, and 32 respectively.
        """
        m = self._model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        c1 = m.layer1(x)
        c2 = m.layer2(c1)
        c3 = m.layer3(c2)
        c4 = m.layer4(c3)
        return [c1, c2, c3, c4]


@register_backbone("resnet18")
class ResNet18Backbone(_ResNetBackbone):
    """ResNet-18 backbone.

    Feature levels:
        - ``stride4``: 64 channels
        - ``stride8``: 128 channels
        - ``stride16``: 256 channels
        - ``stride32``: 512 channels
    """

    _variant_key = "resnet18"


@register_backbone("resnet34")
class ResNet34Backbone(_ResNetBackbone):
    """ResNet-34 backbone.

    Feature levels:
        - ``stride4``: 64 channels
        - ``stride8``: 128 channels
        - ``stride16``: 256 channels
        - ``stride32``: 512 channels
    """

    _variant_key = "resnet34"


@register_backbone("resnet50")
class ResNet50Backbone(_ResNetBackbone):
    """ResNet-50 backbone.

    Feature levels:
        - ``stride4``: 256 channels
        - ``stride8``: 512 channels
        - ``stride16``: 1024 channels
        - ``stride32``: 2048 channels
    """

    _variant_key = "resnet50"


@register_backbone("resnet101")
class ResNet101Backbone(_ResNetBackbone):
    """ResNet-101 backbone.

    Feature levels:
        - ``stride4``: 256 channels
        - ``stride8``: 512 channels
        - ``stride16``: 1024 channels
        - ``stride32``: 2048 channels
    """

    _variant_key = "resnet101"

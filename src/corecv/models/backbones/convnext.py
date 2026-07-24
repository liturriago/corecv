"""ConvNeXt backbone with multi-scale feature extraction.

Wraps :mod:`torchvision.models` ConvNeXt variants (Tiny, Small, Base, Large)
and extracts intermediate feature maps at four spatial scales.  Each
ConvNeXt ``features`` Sequential contains eight :class:`torch.nn.Sequential`
stages paired into four resolution groups; the last stage of each group is
used as the extracted feature level.

Extracted feature levels (verified against torchvision output shapes,
224x224 input):

.. list-table::
   :header-rows: 1
   :widths: 16 12 10 14 14 14 14

   * - Level
     - Index
     - Stride
     - Tiny
     - Small
     - Base
     - Large
   * - ``stride4``
     - 1
     - 4
     - 96
     - 96
     - 128
     - 192
   * - ``stride8``
     - 3
     - 8
     - 192
     - 192
     - 256
     - 384
   * - ``stride16``
     - 5
     - 16
     - 384
     - 384
     - 512
     - 768
   * - ``stride32``
     - 7
     - 32
     - 768
     - 768
     - 1024
     - 1536

Example:
    >>> from corecv.models.backbones.convnext import ConvNeXtTinyBackbone
    >>> backbone = ConvNeXtTinyBackbone(pretrained=False)
    >>> backbone.feature_info.channels
    {'stride4': 96, 'stride8': 192, 'stride16': 384, 'stride32': 768}
"""

from __future__ import annotations

from typing import Any

import torch
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
)

from corecv.core.contract import BaseBackbone, FeatureInfo
from corecv.core.registry import register_backbone

# ---------------------------------------------------------------------------
# Channel configuration per ConvNeXt variant
# ---------------------------------------------------------------------------

_CONVNEXT_VARIANTS: dict[str, dict[str, Any]] = {
    "convnext_tiny": {
        "factory": convnext_tiny,
        "weights": ConvNeXt_Tiny_Weights,
        "channels": {
            "stride4": 96,
            "stride8": 192,
            "stride16": 384,
            "stride32": 768,
        },
    },
    "convnext_small": {
        "factory": convnext_small,
        "weights": ConvNeXt_Small_Weights,
        "channels": {
            "stride4": 96,
            "stride8": 192,
            "stride16": 384,
            "stride32": 768,
        },
    },
    "convnext_base": {
        "factory": convnext_base,
        "weights": ConvNeXt_Base_Weights,
        "channels": {
            "stride4": 128,
            "stride8": 256,
            "stride16": 512,
            "stride32": 1024,
        },
    },
    "convnext_large": {
        "factory": convnext_large,
        "weights": ConvNeXt_Large_Weights,
        "channels": {
            "stride4": 192,
            "stride8": 384,
            "stride16": 768,
            "stride32": 1536,
        },
    },
}

# Feature extraction indices: last stage of each resolution group.
_CONVNEXT_FEATURE_INDICES: list[int] = [1, 3, 5, 7]


class _ConvNeXtBackbone(BaseBackbone):
    """Shared base for all ConvNeXt backbone variants.

    Iterates through ``features`` :class:`torch.nn.Sequential` and collects
    outputs at indices 1, 3, 5, and 7 (the last block of each resolution
    group).

    This class should not be instantiated directly; use the variant-specific
    subclasses.
    """

    _variant_key: str

    def __init__(self, pretrained: bool = True, **kwargs: object) -> None:
        """Initialise the ConvNeXt backbone.

        Args:
            pretrained: If ``True``, load ImageNet-1K pretrained weights.
            **kwargs: Additional keyword arguments forwarded to the
                underlying ``convnext_*`` factory.
        """
        super().__init__()
        cfg = _CONVNEXT_VARIANTS[self._variant_key]
        weights = cfg["weights"].IMAGENET1K_V1 if pretrained else None
        self._model = cfg["factory"](weights=weights, **kwargs)

    @property
    def feature_info(self) -> FeatureInfo:
        """Return channel and stride metadata for all extracted feature levels.

        Returns:
            A :class:`FeatureInfo` with keys ``stride4``, ``stride8``,
            ``stride16``, and ``stride32``.
        """
        cfg = _CONVNEXT_VARIANTS[self._variant_key]
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
        """Extract multi-scale features from the ConvNeXt backbone.

        Runs the input through ``features`` Sequential and collects the
        output at the last block of each resolution group.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            A list of four feature tensors at strides 4, 8, 16, and 32.
        """
        features: list[torch.Tensor] = []
        target_set: set[int] = set(_CONVNEXT_FEATURE_INDICES)

        for i, layer in enumerate(self._model.features):
            x = layer(x)
            if i in target_set:
                features.append(x)

        return features


@register_backbone("convnext_tiny")
class ConvNeXtTinyBackbone(_ConvNeXtBackbone):
    """ConvNeXt-Tiny backbone.

    Feature levels:
        - ``stride4``: 96 channels
        - ``stride8``: 192 channels
        - ``stride16``: 384 channels
        - ``stride32``: 768 channels
    """

    _variant_key = "convnext_tiny"


@register_backbone("convnext_small")
class ConvNeXtSmallBackbone(_ConvNeXtBackbone):
    """ConvNeXt-Small backbone.

    Feature levels:
        - ``stride4``: 96 channels
        - ``stride8``: 192 channels
        - ``stride16``: 384 channels
        - ``stride32``: 768 channels
    """

    _variant_key = "convnext_small"


@register_backbone("convnext_base")
class ConvNeXtBaseBackbone(_ConvNeXtBackbone):
    """ConvNeXt-Base backbone.

    Feature levels:
        - ``stride4``: 128 channels
        - ``stride8``: 256 channels
        - ``stride16``: 512 channels
        - ``stride32``: 1024 channels
    """

    _variant_key = "convnext_base"


@register_backbone("convnext_large")
class ConvNeXtLargeBackbone(_ConvNeXtBackbone):
    """ConvNeXt-Large backbone.

    Feature levels:
        - ``stride4``: 192 channels
        - ``stride8``: 384 channels
        - ``stride16``: 768 channels
        - ``stride32``: 1536 channels
    """

    _variant_key = "convnext_large"

"""MobileNetV3 backbone with multi-scale feature extraction.

Wraps :mod:`torchvision.models.mobilenet_v3_small` and
:mod:`torchvision.models.mobilenet_v3_large` to expose four feature levels
at strides 4, 8, 16, and 32.  Feature maps are extracted at the last block
of each spatial resolution stage.

Extracted feature levels (verified against torchvision output shapes):

- **MobileNetV3-Small** (``features`` Sequential, 224x224 input):

  ============ ===== ======= =========================
  Level        Index Spatial Channel Count
  ============ ===== ======= =========================
  ``stride4``  1     56x56   16
  ``stride8``  3     28x28   24
  ``stride16`` 8     14x14   48
  ``stride32`` 12    7x7     576
  ============ ===== ======= =========================

- **MobileNetV3-Large** (``features`` Sequential, 224x224 input):

  ============ ===== ======= =========================
  Level        Index Spatial Channel Count
  ============ ===== ======= =========================
  ``stride4``  3     56x56   24
  ``stride8``  6     28x28   40
  ``stride16`` 12    14x14   112
  ``stride32`` 16    7x7     960
  ============ ===== ======= =========================

Example:
    >>> from corecv.models.backbones.mobilenetv3 import MobileNetV3SmallBackbone
    >>> backbone = MobileNetV3SmallBackbone(pretrained=False)
    >>> backbone.feature_info.channels
    {'stride4': 16, 'stride8': 24, 'stride16': 48, 'stride32': 576}
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)

from corecv.core.contract import BaseBackbone, FeatureInfo
from corecv.core.registry import register_backbone


class _MobileNetV3Backbone(BaseBackbone):
    """Shared base for MobileNetV3-Small and MobileNetV3-Large backbones.

    Iterates through the ``features`` :class:`torch.nn.Sequential` and
    collects outputs at specific block indices that correspond to spatial
    resolution boundaries.

    This class should not be instantiated directly; use
    :class:`MobileNetV3SmallBackbone` or :class:`MobileNetV3LargeBackbone`.
    """

    # Subclasses override these.
    _model_factory: Any
    _weights_enum: Any
    _feature_indices: dict[str, int]
    _feature_channels: dict[str, int]

    def __init__(self, pretrained: bool = True, **kwargs: object) -> None:
        """Initialise the MobileNetV3 backbone.

        Args:
            pretrained: If ``True``, load ImageNet-1K pretrained weights.
            **kwargs: Additional keyword arguments forwarded to the
                underlying ``mobilenet_v3_*`` factory.
        """
        super().__init__()
        weights = self._weights_enum.IMAGENET1K_V1 if pretrained else None
        self._model = self._model_factory(weights=weights, **kwargs)

    @property
    def feature_info(self) -> FeatureInfo:
        """Return channel and stride metadata for all extracted feature levels.

        Returns:
            A :class:`FeatureInfo` with keys ``stride4``, ``stride8``,
            ``stride16``, and ``stride32``.
        """
        return FeatureInfo(
            channels=dict(self._feature_channels),
            strides={
                "stride4": 4,
                "stride8": 8,
                "stride16": 16,
                "stride32": 32,
            },
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features from the MobileNetV3 backbone.

        Runs the input through ``features`` Sequential and collects the
        output at each target index.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            A list of four feature tensors at strides 4, 8, 16, and 32,
            ordered from finest to coarsest spatial resolution.
        """
        target_indices: Sequence[int] = sorted(self._feature_indices.values())
        features: list[torch.Tensor] = []
        target_set: set[int] = set(target_indices)

        for i, layer in enumerate(self._model.features):
            x = layer(x)
            if i in target_set:
                features.append(x)

        return features


@register_backbone("mobilenet_v3_small")
class MobileNetV3SmallBackbone(_MobileNetV3Backbone):
    """MobileNetV3-Small backbone with four-level feature extraction.

    Wraps :func:`torchvision.models.mobilenet_v3_small`.  Feature maps are
    extracted from ``features`` Sequential at indices 1, 3, 8, and 12,
    corresponding to strides 4, 8, 16, and 32 respectively.

    Feature levels:
        - ``stride4``: 16 channels, 56x56 spatial (index 1)
        - ``stride8``: 24 channels, 28x28 spatial (index 3)
        - ``stride16``: 48 channels, 14x14 spatial (index 8)
        - ``stride32``: 576 channels, 7x7 spatial (index 12)
    """

    _model_factory = staticmethod(mobilenet_v3_small)
    _weights_enum = MobileNet_V3_Small_Weights
    _feature_indices: dict[str, int] = {
        "stride4": 1,
        "stride8": 3,
        "stride16": 8,
        "stride32": 12,
    }
    _feature_channels: dict[str, int] = {
        "stride4": 16,
        "stride8": 24,
        "stride16": 48,
        "stride32": 576,
    }


@register_backbone("mobilenet_v3_large")
class MobileNetV3LargeBackbone(_MobileNetV3Backbone):
    """MobileNetV3-Large backbone with four-level feature extraction.

    Wraps :func:`torchvision.models.mobilenet_v3_large`.  Feature maps are
    extracted from ``features`` Sequential at indices 3, 6, 12, and 16,
    corresponding to strides 4, 8, 16, and 32 respectively.

    Feature levels:
        - ``stride4``: 24 channels, 56x56 spatial (index 3)
        - ``stride8``: 40 channels, 28x28 spatial (index 6)
        - ``stride16``: 112 channels, 14x14 spatial (index 12)
        - ``stride32``: 960 channels, 7x7 spatial (index 16)
    """

    _model_factory = staticmethod(mobilenet_v3_large)
    _weights_enum = MobileNet_V3_Large_Weights
    _feature_indices: dict[str, int] = {
        "stride4": 3,
        "stride8": 6,
        "stride16": 12,
        "stride32": 16,
    }
    _feature_channels: dict[str, int] = {
        "stride4": 24,
        "stride8": 40,
        "stride16": 112,
        "stride32": 960,
    }

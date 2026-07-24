"""Linear classification head for image-level classification.

Provides :class:`LinearClassificationHead`, a simple head that consumes the
coarsest feature map from a backbone or neck and produces per-class logits
via global average pooling and a 1x1 convolution (equivalent to a linear
layer for 1x1 spatial input).

The head dynamically adapts to the upstream
:class:`~corecv.core.contract.FeatureInfo` metadata and handles both
``dict`` and ``list``-style feature map inputs.

Example:
    >>> from corecv.core.contract import FeatureInfo
    >>> from corecv.models.heads.classification import LinearClassificationHead
    >>> fi = FeatureInfo(
    ...     channels={"level0": 64, "level1": 128, "level2": 256},
    ...     strides={"level0": 4, "level1": 8, "level2": 16},
    ... )
    >>> head = LinearClassificationHead(feature_info=fi, num_classes=10)
    >>> import torch
    >>> feats = [torch.randn(1, c, 56 // s, 56 // s)
    ...         for c, s in zip((64, 128, 256), (4, 8, 16))]
    >>> logits = head(feats)
    >>> logits.shape
    torch.Size([1, 10])
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from corecv.core.contract import FeatureInfo
from corecv.core.registry import register_head


@register_head("linear_classification")
class LinearClassificationHead(nn.Module):
    """Image-level classification head with global average pooling.

    Consumes the coarsest (last) feature map from a backbone or neck,
    applies adaptive average pooling to ``1x1``, and passes through a
    1x1 convolution (equivalent to ``nn.Linear`` on a 1x1 spatial map)
    to produce per-class logits.

    A 1x1 convolution is used instead of ``nn.Linear`` to ensure
    compatibility with ``device='meta'`` shape propagation.

    The head dynamically inspects the :class:`FeatureInfo` to determine
    the input channel count, making it compatible with any backbone
    that implements :class:`~corecv.core.contract.BaseBackbone`.

    Args:
        feature_info: Metadata from the upstream backbone, used to
            determine input channel count from the coarsest feature level.
        num_classes: Number of output classes.
        dropout: Optional dropout probability applied before the
            classifier (default ``0.0`` i.e. no dropout).
    """

    def __init__(
        self,
        feature_info: FeatureInfo,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the classification head.

        Args:
            feature_info: Backbone feature metadata.
            num_classes: Number of output classes.
            dropout: Dropout probability (default 0.0).
        """
        super().__init__()
        self.num_classes = num_classes

        # Sort levels by stride (ascending = finest first).
        sorted_levels: list[str] = sorted(
            feature_info.strides.keys(),
            key=lambda k: feature_info.strides[k],  # type: ignore[arg-type]
        )
        if not sorted_levels:
            msg = "feature_info must contain at least one feature level."
            raise ValueError(msg)

        self._feature_info = feature_info
        self._sorted_levels = sorted_levels
        # Use the coarsest (last) level for classification.
        coarsest_level: str = sorted_levels[-1]
        in_channels: int = feature_info.channels[coarsest_level]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        # 1x1 conv equivalent to nn.Linear(in_channels, num_classes).
        self.fc = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(
        self, features: dict[str, Tensor] | Sequence[Tensor]
    ) -> Tensor:
        """Produce classification logits from multi-scale feature maps.

        Args:
            features: Feature maps from a backbone or neck.  Can be a
                ``dict`` of level-name -> tensor or a ``Sequence``
                (list/tuple) of tensors ordered from finest to coarsest.

        Returns:
            Logit tensor of shape ``(batch_size, num_classes)``.
        """
        if isinstance(features, dict):
            # Dict: pick the last level in sorted (by stride) order.
            x: Tensor = features[self._sorted_levels[-1]]
        else:
            # Sequence: last element is the coarsest.
            x = features[-1]

        x = self.pool(x)  # (B, C, 1, 1)
        x = self.fc(x)    # (B, num_classes, 1, 1)
        x = self.dropout(x)
        return x.flatten(1)  # (B, num_classes)

    @property
    def feature_info(self) -> FeatureInfo:
        """Return the feature metadata used at construction time.

        Returns:
            The :class:`FeatureInfo` instance passed to ``__init__``.
        """
        return self._feature_info  # type: ignore[attr-defined]

"""Classification head for CoreCV.

Implements a standard classification head with global average pooling
and a linear classifier that dynamically consumes ``FeatureInfo``
metadata from backbones.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

_SPATIAL_DIMS = 4


class ClassificationHead(nn.Module):
    """Global average pooling + linear classification head.

    Accepts feature tensors of shape ``(B, C, H, W)`` (spatial) or
    ``(B, C)`` (already pooled) and produces class logits of shape
    ``(B, num_classes)``.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        """Initialize the classification head.

        Args:
            in_channels: Number of input feature channels from the
                backbone or neck.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=in_channels, out_features=num_classes)

    def forward(self, features: Tensor) -> Tensor:
        """Run the forward pass.

        Args:
            features: Input feature tensor of shape ``(B, C, H, W)``
                (spatial feature map) or ``(B, C)`` (pre-pooled vector).

        Returns:
            Class logits of shape ``(B, num_classes)``.
        """
        if features.dim() == _SPATIAL_DIMS:
            features = self.gap(features).flatten(start_dim=1)
        return self.fc(features)
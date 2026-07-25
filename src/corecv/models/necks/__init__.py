"""Necks subpackage for CoreCV.

This subpackage contains feature fusion modules (necks) that combine
multi-scale features from backbones before passing them to task-specific
heads. Supported architectures:

- **FPN**: Feature Pyramid Network for top-down feature fusion.
- **PANet**: Path Aggregation Network with bottom-up augmentation.
- **BiFPN**: Bidirectional Feature Pyramid Network with weighted fusion.
"""

from __future__ import annotations

from corecv.models.necks.bifpn import BiFPN
from corecv.models.necks.fpn import FPN
from corecv.models.necks.panet import PANet

__all__ = [
    "FPN",
    "BiFPN",
    "PANet",
]

"""Heads subpackage for CoreCV.

This subpackage contains task-specific prediction heads that consume
multi-scale features from backbones (via optional necks).  Supported
heads:

- **Classification**: Global average pooling + linear classifier.
- **Segmentation**: DeepLabV3+ and ResUNetDecoder for semantic segmentation.
- **Detection**: Anchor-free dual-head (O2M + O2O) detection head.
"""

from __future__ import annotations

from corecv.models.heads.classification import ClassificationHead
from corecv.models.heads.segmentation import DeepLabV3PlusHead, ResUNetDecoder

__all__ = [
    "ClassificationHead",
    "DeepLabV3PlusHead",
    "ResUNetDecoder",
]

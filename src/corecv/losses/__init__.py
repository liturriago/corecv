"""Loss functions for classification, segmentation, and detection tasks."""

from corecv.losses.classification import (
    ClassificationCrossEntropyLoss,
    ClassificationFocalLoss,
)
from corecv.losses.detection import DualHeadDetectionLoss
from corecv.losses.segmentation import (
    SegmentationCrossEntropyLoss,
    SegmentationDiceLoss,
    SegmentationFocalLoss,
)

__all__ = [
    "ClassificationCrossEntropyLoss",
    "ClassificationFocalLoss",
    "DualHeadDetectionLoss",
    "SegmentationCrossEntropyLoss",
    "SegmentationDiceLoss",
    "SegmentationFocalLoss",
]

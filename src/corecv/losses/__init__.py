"""Loss functions for classification, segmentation, and detection tasks."""

from corecv.losses.loss_classification import ClassificationCrossEntropyLoss, ClassificationFocalLoss
from corecv.losses.loss_detection import DualHeadDetectionLoss
from corecv.losses.loss_segmentation import DiceLoss, SegmentationCrossEntropyLoss, SegmentationFocalLoss

__all__ = [
    "ClassificationCrossEntropyLoss",
    "ClassificationFocalLoss",
    "DiceLoss",
    "DualHeadDetectionLoss",
    "SegmentationCrossEntropyLoss",
    "SegmentationFocalLoss",
]

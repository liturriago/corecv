"""Loss functions for classification, segmentation, and detection tasks."""

from corecv.losses.loss_classification import ClassificationCrossEntropyLoss, ClassificationFocalLoss

__all__ = [
    "ClassificationCrossEntropyLoss",
    "ClassificationFocalLoss",
]

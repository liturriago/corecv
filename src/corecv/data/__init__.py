"""Data loading and preprocessing modules for CoreCV."""

from corecv.data.classification import (
    ClassificationDataset,
    create_classification_dataloader,
)
from corecv.data.segmentation import (
    SegmentationDataset,
    create_segmentation_dataloader,
)

__all__ = [
    "ClassificationDataset",
    "SegmentationDataset",
    "create_classification_dataloader",
    "create_segmentation_dataloader",
]

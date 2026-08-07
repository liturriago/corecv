"""Data loading and preprocessing modules for CoreCV."""

from corecv.data.classification import (
    ClassificationDataset,
    create_classification_dataloader,
)
from corecv.data.detection import (
    DetectionDataset,
    create_detection_dataloader,
)
from corecv.data.segmentation import (
    SegmentationDataset,
    create_segmentation_dataloader,
)

__all__ = [
    "ClassificationDataset",
    "DetectionDataset",
    "SegmentationDataset",
    "create_classification_dataloader",
    "create_detection_dataloader",
    "create_segmentation_dataloader",
]

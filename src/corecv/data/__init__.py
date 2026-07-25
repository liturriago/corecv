"""Data loading and preprocessing modules for CoreCV."""

from corecv.data.data_classification import ClassificationDataset
from corecv.data.data_detection import DetectionDataset
from corecv.data.data_segmentation import SegmentationDataset

__all__ = ["ClassificationDataset", "DetectionDataset", "SegmentationDataset"]

"""Evaluation metrics for classification, segmentation, and detection tasks."""

from corecv.metrics.classification import ClassificationMetrics
from corecv.metrics.detection import DetectionMetrics
from corecv.metrics.segmentation import SegmentationMetrics

__all__ = ["ClassificationMetrics", "DetectionMetrics", "SegmentationMetrics"]

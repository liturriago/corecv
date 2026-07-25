"""Evaluation metrics for classification, segmentation, and detection tasks."""

from corecv.metrics.metric_classification import ClassificationMetrics
from corecv.metrics.metric_detection import DetectionMetrics
from corecv.metrics.metric_segmentation import SegmentationMetrics

__all__ = ["ClassificationMetrics", "DetectionMetrics", "SegmentationMetrics"]

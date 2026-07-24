"""Data module for CoreCV.

Provides coordinated data transforms for classification, segmentation,
and detection tasks, wrapping Albumentations with a clean, typed API,
as well as dataset implementations for common vision tasks.

Example:
    >>> from corecv.data import build_transforms, DetectionTransformConfig
    >>> config = DetectionTransformConfig(image_size=(640, 640))
    >>> transform = build_transforms(config)
"""

from corecv.data.datasets import DetectionDataset, SegmentationDataset
from corecv.data.transforms import (
    BaseTransformConfig,
    ClassificationTransformConfig,
    CoordinatedTransform,
    DetectionTransformConfig,
    SegmentationTransformConfig,
    TransformOutput,
    build_transforms,
)

__all__ = [
    "BaseTransformConfig",
    "ClassificationTransformConfig",
    "CoordinatedTransform",
    "DetectionTransformConfig",
    "DetectionDataset",
    "SegmentationDataset",
    "SegmentationTransformConfig",
    "TransformOutput",
    "build_transforms",
]

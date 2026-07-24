"""Configuration module for CoreCV.

Provides polymorphic :func:`dataclasses.dataclass`-based configuration
schemas for classification, segmentation, and detection tasks, and a
:func:`load_config` factory function.
"""

from corecv.config.schemas import (
    BaseTaskConfig,
    ClassificationConfig,
    DetectionConfig,
    SegmentationConfig,
    load_config,
)

__all__ = [
    "BaseTaskConfig",
    "ClassificationConfig",
    "DetectionConfig",
    "SegmentationConfig",
    "load_config",
]

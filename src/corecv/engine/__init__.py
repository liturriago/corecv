"""CoreCV Engine Module.

Provides training engines for classification, segmentation, and detection
tasks with full GPU-native metric computation and history tracking.
"""

from __future__ import annotations

from corecv.engine.train import (
    ClassificationTrainer,
    DetectionTrainer,
    SegmentationTrainer,
)

__all__ = [
    "ClassificationTrainer",
    "DetectionTrainer",
    "SegmentationTrainer",
]

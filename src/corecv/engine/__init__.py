"""CoreCV Engine Module.

Provides training and evaluation engines for classification, segmentation,
and detection tasks with full GPU-native metric computation and history tracking.
"""

from __future__ import annotations

from corecv.engine.evaluation import (
    ClassificationEvaluator,
    DetectionEvaluator,
    SegmentationEvaluator,
)
from corecv.engine.train import (
    ClassificationTrainer,
    DetectionTrainer,
    SegmentationTrainer,
)

__all__ = [
    "ClassificationEvaluator",
    "ClassificationTrainer",
    "DetectionEvaluator",
    "DetectionTrainer",
    "SegmentationEvaluator",
    "SegmentationTrainer",
]

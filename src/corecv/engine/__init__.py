"""CoreCV Engine Module.

Provides training, evaluation, and prediction engines for classification,
segmentation, and detection tasks with GPU-native metric computation and
history tracking.
"""

from __future__ import annotations

from corecv.engine.predict import ImagePredictor, PredictionResult
from corecv.engine.train import Trainer

__all__ = [
    "ImagePredictor",
    "PredictionResult",
    "Trainer",
]

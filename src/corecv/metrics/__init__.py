"""GPU-native metrics engine for CoreCV.

All metrics in this package compute exclusively on VRAM during
``update()`` — no ``.item()`` calls, no ``.cpu()`` transfers, no
CPU-GPU synchronisations.  Only ``compute()`` performs the final scalar
reductions and returns a plain Python ``dict``.

Submodules:

* :mod:`~corecv.metrics.classification` — Accumulator-based
  classification metrics (accuracy, top-k, precision, recall, F1).
* :mod:`~corecv.metrics.segmentation` — Accumulator-based segmentation
  metrics (mIoU, pixel accuracy, Dice).
* :mod:`~corecv.metrics.detection` — Vectorised mAP@50 and mAP@50:95
  without ``pycocotools``, using ``torchvision.ops.box_iou``.

Example:
    >>> from corecv.metrics import (
    ...     ClassificationMetrics,
    ...     SegmentationMetrics,
    ...     DetectionMetrics,
    ... )
"""

from corecv.metrics.classification import ClassificationMetrics
from corecv.metrics.detection import DetectionMetrics
from corecv.metrics.segmentation import SegmentationMetrics

__all__: list[str] = [
    "ClassificationMetrics",
    "SegmentationMetrics",
    "DetectionMetrics",
]

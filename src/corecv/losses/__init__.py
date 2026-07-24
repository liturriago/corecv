"""GPU-native loss functions for CoreCV.

All losses in this package are implemented as pure vectorised PyTorch
operations with **zero** CPU-GPU synchronisations during forward and
backward passes.  No Python for-loops, no ``.item()`` calls, no
``.cpu()`` transfers — everything stays on VRAM.

Submodules:

* :mod:`~corecv.losses.classification` — Focal loss, label-smoothing CE.
* :mod:`~corecv.losses.segmentation` — Dice loss, combined CE + Dice.
* :mod:`~corecv.losses.detection` — GIoU, CIoU, QualityFocalLoss,
  VarifocalLoss.
* :mod:`~corecv.losses.assigners` — HungarianMatcher, SetCriterion for
  query-based detection (RT-DETR / D-FINE).

Example:
    >>> from corecv.losses import FocalLoss, DiceLoss, GIoULoss
    >>> focal = FocalLoss(alpha=0.25, gamma=2.0)
    >>> dice  = DiceLoss(smooth=1.0)
    >>> giou  = GIoULoss(reduction="mean")
"""

from corecv.losses.assigners import (
    HungarianMatcher,
    SetCriterion,
    TaskAlignedAssigner,
)
from corecv.losses.classification import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
)
from corecv.losses.detection import (
    CIoULoss,
    GIoULoss,
    QualityFocalLoss,
    VarifocalLoss,
)
from corecv.losses.segmentation import (
    CombinedSegmentationLoss,
    DiceLoss,
)

__all__: list[str] = [
    # Classification
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    # Segmentation
    "DiceLoss",
    "CombinedSegmentationLoss",
    # Detection
    "GIoULoss",
    "CIoULoss",
    "QualityFocalLoss",
    "VarifocalLoss",
    # Set-based assignment / criterion
    "HungarianMatcher",
    "SetCriterion",
    # Dynamic assignment
    "TaskAlignedAssigner",
]

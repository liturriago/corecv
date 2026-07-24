"""GPU-native assignment strategies for detection losses.

Provides :class:`HungarianMatcher` for bipartite 1-to-1 matching used in
DETR-style (RT-DETR / D-FINE) query-based detection heads, and
:class:`SetCriterion` that consumes the matcher output to compute focal
classification, L1 bounding-box, and GIoU losses — all as pure vectorised
PyTorch operations with **zero** CPU-GPU synchronisations during loss
computation.

Also provides :class:`TaskAlignedAssigner` for dynamic top-k sample
assignment used in FCOS/TOOD-style anchor-free detection heads with
decoupled classification and regression branches.

Example:
    >>> from corecv.losses.assigners import HungarianMatcher, SetCriterion
    >>> matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    >>> criterion = SetCriterion(num_classes=80, matcher=matcher)

    >>> from corecv.losses.assigners import TaskAlignedAssigner
    >>> assigner = TaskAlignedAssigner(num_classes=80, topk=13)
"""

from corecv.losses.assigners.hungarian import HungarianMatcher, SetCriterion
from corecv.losses.assigners.tal import TaskAlignedAssigner

__all__: list[str] = [
    "HungarianMatcher",
    "SetCriterion",
    "TaskAlignedAssigner",
]

"""Detection head modules for CoreCV.

Provides :class:`~corecv.core.contract.FeatureInfo`-aware detection heads
for anchor-free and query-based object detection.

Available heads
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 30

   * - Class
     - Registry Key
     - Description
   * - :class:`DecoupledAnchorFreeHead`
     - ``decoupled_anchor_free``
     - FCOS/YOLOX-style pure-conv head with decoupled cls/reg branches.
   * - :class:`QueryDetectionHead`
     - ``query_detection``
     - RT-DETR/D-FINE-style transformer-decoder head with learnable queries
       and NMS-free inference.

Example:
    >>> from corecv.models.heads.detection import (
    ...     DecoupledAnchorFreeHead,
    ...     QueryDetectionHead,
    ... )
"""

from corecv.models.heads.detection.decoupled_anchor_free import (
    DecoupledAnchorFreeHead,
)
from corecv.models.heads.detection.query_detection_head import QueryDetectionHead

__all__ = [
    "DecoupledAnchorFreeHead",
    "QueryDetectionHead",
]

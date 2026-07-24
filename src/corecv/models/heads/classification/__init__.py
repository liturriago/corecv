"""Classification head modules for CoreCV.

Provides a simple linear classification head that consumes the coarsest
feature level from a backbone or neck and produces per-class logits.

Available classification heads
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Class
     - Registry Key
     - Description
   * - :class:`LinearClassificationHead`
     - ``linear_classification``
     - Global average pooling + fully-connected layer for image-level
       classification.

Example:
    >>> from corecv.models.heads.classification import LinearClassificationHead
    >>> from corecv.core.contract import FeatureInfo
    >>> fi = FeatureInfo(channels={"feat": 2048}, strides={"feat": 32})
    >>> head = LinearClassificationHead(feature_info=fi, num_classes=1000)
"""

from corecv.models.heads.classification.linear_head import LinearClassificationHead

__all__ = [
    "LinearClassificationHead",
]

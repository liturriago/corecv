"""Head modules for CoreCV.

Provides :class:`~corecv.core.contract.FeatureInfo`-aware head modules for
classification, segmentation, and object detection tasks.  All heads
dynamically adapt to backbone metadata, making them backbone-agnostic.

Available heads
---------------

Classification
    * :class:`~corecv.models.heads.classification.LinearClassificationHead`
      (``linear_classification``) -- global average pooling + FC layer.

Segmentation
    * :class:`~corecv.models.heads.segmentation.ResUNetDecoder`
      (``resunet_decoder``) -- U-Net style decoder with residual blocks.
    * :class:`~corecv.models.heads.segmentation.ASPPDecoder`
      (``aspp_decoder``) -- DeepLabV3+ style decoder with ASPP module.

Detection
    * :class:`~corecv.models.heads.detection.DecoupledAnchorFreeHead`
      (``decoupled_anchor_free``) -- FCOS/YOLOX-style pure-conv head.
    * :class:`~corecv.models.heads.detection.QueryDetectionHead`
      (``query_detection``) -- RT-DETR/D-FINE-style transformer decoder
      head with learnable queries.
"""

from corecv.models.heads.classification import LinearClassificationHead
from corecv.models.heads.detection import DecoupledAnchorFreeHead, QueryDetectionHead
from corecv.models.heads.segmentation import ASPPDecoder, ResUNetDecoder

__all__ = [
    "LinearClassificationHead",
    "ResUNetDecoder",
    "ASPPDecoder",
    "DecoupledAnchorFreeHead",
    "QueryDetectionHead",
]

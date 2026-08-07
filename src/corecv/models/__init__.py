"""CoreCV Models Module.

This package provides SOTA computer vision models for classification,
detection, and segmentation tasks. It includes:

- **Backbones**: Feature extractors with FeatureInfo metadata, including
  TorchVision models (ResNet, MobileNetV3, ConvNeXt, Swin Transformer) and a
  from-scratch CSP pyramid backbone.
- **Necks**: Feature pyramid networks (FPN, PANet, BiFPN, CSPPANet) for
  multi-scale feature fusion.
- **Heads**: Task-specific heads for classification, detection (anchor-free
  dual-head), and segmentation (DeepLabV3+, ResUNetDecoder).
- **Model factories**: High-level model builders for each task.
"""

from __future__ import annotations

from corecv.models.backbones import (
    BackboneName,
    BaseBackbone,
    ConvNeXtBackbone,
    CSPPyramidBackbone,
    FeatureInfo,
    MobileNetV3Backbone,
    ResNetBackbone,
    SwinTransformerBackbone,
    create_backbone,
)
from corecv.models.classification import ClassificationModel, create_classification_model
from corecv.models.detection import DetectionModel, create_detection_model
from corecv.models.heads import (
    ClassificationHead,
    DeepLabV3PlusHead,
    DetectionHead,
    ResUNetDecoder,
)
from corecv.models.necks import FPN, BiFPN, CSPPANet, PANet
from corecv.models.segmentation import SegmentationModel, create_segmentation_model

__all__ = [
    "FPN",
    "BackboneName",
    "BaseBackbone",
    "BiFPN",
    "CSPPANet",
    "CSPPyramidBackbone",
    "ClassificationHead",
    "ClassificationModel",
    "ConvNeXtBackbone",
    "DeepLabV3PlusHead",
    "DetectionHead",
    "DetectionModel",
    "FeatureInfo",
    "MobileNetV3Backbone",
    "PANet",
    "ResNetBackbone",
    "ResUNetDecoder",
    "SegmentationModel",
    "SwinTransformerBackbone",
    "create_backbone",
    "create_classification_model",
    "create_detection_model",
    "create_segmentation_model",
]

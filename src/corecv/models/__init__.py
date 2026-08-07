"""CoreCV Models Module.

This package provides SOTA computer vision models for classification,
detection, and segmentation tasks. It includes:

- **Backbones**: TorchVision-based feature extractors (ResNet, MobileNetV3,
  ConvNeXt, Swin Transformer) with FeatureInfo metadata.
- **Necks**: Feature pyramid networks (FPN, PANet, BiFPN) for multi-scale
  feature fusion.
- **Heads**: Task-specific heads for classification, detection (anchor-free),
  and segmentation (DeepLabV3+, ResUNetDecoder).
- **Model factories**: High-level model builders for each task.
"""

from __future__ import annotations

from corecv.models.backbones import (
    BackboneName,
    BaseBackbone,
    ConvNeXtBackbone,
    FeatureInfo,
    MobileNetV3Backbone,
    ResNetBackbone,
    SwinTransformerBackbone,
    create_backbone,
)
from corecv.models.classification import ClassificationModel, create_classification_model
from corecv.models.heads import (
    ClassificationHead,
    DeepLabV3PlusHead,
    ResUNetDecoder,
)
from corecv.models.necks import FPN, BiFPN, PANet
from corecv.models.segmentation import SegmentationModel, create_segmentation_model

__all__ = [
    "FPN",
    "BackboneName",
    "BaseBackbone",
    "BiFPN",
    "ClassificationHead",
    "ClassificationModel",
    "ConvNeXtBackbone",
    "DeepLabV3PlusHead",
    "FeatureInfo",
    "MobileNetV3Backbone",
    "PANet",
    "ResNetBackbone",
    "ResUNetDecoder",
    "SegmentationModel",
    "SwinTransformerBackbone",
    "create_backbone",
    "create_classification_model",
    "create_segmentation_model",
]

"""Models module for CoreCV.

Provides backbone, neck, and head implementations for vision tasks.
All backbones conform to the :class:`~corecv.core.contract.BaseBackbone`
interface and expose :class:`~corecv.core.contract.FeatureInfo` metadata.
"""

from corecv.models.backbones import (
    ConvNeXtBaseBackbone,
    ConvNeXtLargeBackbone,
    ConvNeXtSmallBackbone,
    ConvNeXtTinyBackbone,
    MobileNetV3LargeBackbone,
    MobileNetV3SmallBackbone,
    ResNet18Backbone,
    ResNet34Backbone,
    ResNet50Backbone,
    ResNet101Backbone,
    ViTBaseBackbone,
    ViTSmallBackbone,
    ViTTinyBackbone,
)
from corecv.models.detector import CoreObjectDetector
from corecv.models.heads import (
    ASPPDecoder,
    DecoupledAnchorFreeHead,
    LinearClassificationHead,
    QueryDetectionHead,
    ResUNetDecoder,
)
from corecv.models.necks import FPN, PANet

__all__ = [
    # MobileNetV3
    "MobileNetV3SmallBackbone",
    "MobileNetV3LargeBackbone",
    # ResNet
    "ResNet18Backbone",
    "ResNet34Backbone",
    "ResNet50Backbone",
    "ResNet101Backbone",
    # ConvNeXt
    "ConvNeXtTinyBackbone",
    "ConvNeXtSmallBackbone",
    "ConvNeXtBaseBackbone",
    "ConvNeXtLargeBackbone",
    # ViT
    "ViTTinyBackbone",
    "ViTSmallBackbone",
    "ViTBaseBackbone",
    # Heads
    "LinearClassificationHead",
    "ResUNetDecoder",
    "ASPPDecoder",
    "DecoupledAnchorFreeHead",
    "QueryDetectionHead",
    # Necks
    "FPN",
    "PANet",
    # Detector
    "CoreObjectDetector",
]

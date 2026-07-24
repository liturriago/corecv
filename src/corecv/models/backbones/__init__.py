"""Backbone modules for CoreCV.

Provides :class:`~corecv.core.contract.BaseBackbone` implementations for
MobileNetV3, ResNet, ConvNeXt, and Vision Transformer architectures.  All
backbones expose :class:`~corecv.core.contract.FeatureInfo` metadata and
are registered in :data:`~corecv.core.registry.BACKBONE_REGISTRY`.

Backbone catalogue
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 15

   * - Class
     - Registry Key
     - Feature Levels
     - Strides
   * - :class:`MobileNetV3SmallBackbone`
     - ``mobilenet_v3_small``
     - 4
     - 4, 8, 16, 32
   * - :class:`MobileNetV3LargeBackbone`
     - ``mobilenet_v3_large``
     - 4
     - 4, 8, 16, 32
   * - :class:`ResNet18Backbone`
     - ``resnet18``
     - 4
     - 4, 8, 16, 32
   * - :class:`ResNet34Backbone`
     - ``resnet34``
     - 4
     - 4, 8, 16, 32
   * - :class:`ResNet50Backbone`
     - ``resnet50``
     - 4
     - 4, 8, 16, 32
   * - :class:`ResNet101Backbone`
     - ``resnet101``
     - 4
     - 4, 8, 16, 32
   * - :class:`ConvNeXtTinyBackbone`
     - ``convnext_tiny``
     - 4
     - 4, 8, 16, 32
   * - :class:`ConvNeXtSmallBackbone`
     - ``convnext_small``
     - 4
     - 4, 8, 16, 32
   * - :class:`ConvNeXtBaseBackbone`
     - ``convnext_base``
     - 4
     - 4, 8, 16, 32
   * - :class:`ConvNeXtLargeBackbone`
     - ``convnext_large``
     - 4
     - 4, 8, 16, 32
   * - :class:`ViTTinyBackbone`
     - ``vit_tiny``
     - 3
     - 8, 16, 32
   * - :class:`ViTSmallBackbone`
     - ``vit_small``
     - 3
     - 8, 16, 32
   * - :class:`ViTBaseBackbone`
     - ``vit_base``
     - 3
     - 8, 16, 32

Example:
    >>> from corecv.models.backbones import ResNet50Backbone
    >>> backbone = ResNet50Backbone(pretrained=True)
    >>> list(backbone.feature_info.channels.values())
    [256, 512, 1024, 2048]
"""

from corecv.models.backbones.convnext import (
    ConvNeXtBaseBackbone,
    ConvNeXtLargeBackbone,
    ConvNeXtSmallBackbone,
    ConvNeXtTinyBackbone,
)
from corecv.models.backbones.mobilenetv3 import (
    MobileNetV3LargeBackbone,
    MobileNetV3SmallBackbone,
)
from corecv.models.backbones.resnet import (
    ResNet18Backbone,
    ResNet34Backbone,
    ResNet50Backbone,
    ResNet101Backbone,
)
from corecv.models.backbones.vit import (
    SimplePyramidAdapter,
    ViTBaseBackbone,
    ViTSmallBackbone,
    ViTTinyBackbone,
)

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
    # Adapter (public for downstream composition)
    "SimplePyramidAdapter",
]

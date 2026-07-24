"""Segmentation head modules for CoreCV.

Provides decoder implementations for semantic segmentation tasks.  All heads
dynamically adapt to backbone
:class:`~corecv.core.contract.FeatureInfo` metadata and are registered in
:func:`~corecv.core.registry.HEAD_REGISTRY`.

Available segmentation heads
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Class
     - Registry Key
     - Description
   * - :class:`ResUNetDecoder`
     - ``resunet_decoder``
     - U-Net style decoder with residual blocks and skip connections
   * - :class:`ASPPDecoder`
     - ``aspp_decoder``
     - DeepLabV3+ style decoder with ASPP context module

Example:
    >>> from corecv.models.backbones.resnet import ResNet50Backbone
    >>> from corecv.models.heads.segmentation import ASPPDecoder
    >>> backbone = ResNet50Backbone(pretrained=False)
    >>> decoder = ASPPDecoder(
    ...     feature_info=backbone.feature_info,
    ...     out_channels=256,
    ...     num_classes=21,
    ... )
    >>> feats = backbone(torch.randn(1, 3, 224, 224))
    >>> logits = decoder(feats)
"""

from corecv.models.heads.segmentation.aspp_decoder import ASPPDecoder
from corecv.models.heads.segmentation.resunet_decoder import ResUNetDecoder

__all__ = [
    "ResUNetDecoder",
    "ASPPDecoder",
]

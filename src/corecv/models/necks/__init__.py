"""Neck modules for CoreCV.

Provides feature-pyramid necks that sit between a :class:`BaseBackbone`
and task-specific heads.  All necks consume
:class:`~corecv.core.contract.FeatureInfo` metadata and dynamically
construct their internal convolutions to be backbone-agnostic.

Available necks
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Class
     - Registry Key
     - Description
   * - :class:`FPN`
     - ``fpn``
     - Feature Pyramid Network (Lin et al., 2017) -- top-down pathway
       with lateral connections.
   * - :class:`PANet`
     - ``panet``
     - Path Aggregation Network (Liu et al., 2018) -- FPN plus
       bottom-up path augmentation.

Example:
    >>> from corecv.models.backbones.resnet import ResNet50Backbone
    >>> from corecv.models.necks import FPN
    >>> backbone = ResNet50Backbone(pretrained=False)
    >>> neck = FPN(feature_info=backbone.feature_info, out_channels=256)
    >>> features = backbone(torch.randn(1, 3, 224, 224))
    >>> pyramid = neck(features)
"""

from corecv.models.necks.fpn import FPN
from corecv.models.necks.panet import PANet

__all__ = [
    "FPN",
    "PANet",
]

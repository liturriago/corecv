"""CoreObjectDetector: End-to-end object detection model.

Wraps a backbone, optional neck, and detection head into a single
:class:`nn.Module` for end-to-end object detection.  The detector is
backbone/neck/head-agnostic -- any combination of components that conform
to the CoreCV contracts can be wired together.

Example:
    >>> from corecv.core.contract import FeatureInfo
    >>> from corecv.models.backbones.resnet import ResNet50Backbone
    >>> from corecv.models.necks.fpn import FPN
    >>> from corecv.models.heads.detection import DecoupledAnchorFreeHead
    >>> from corecv.models.detector import CoreObjectDetector
    >>> backbone = ResNet50Backbone(pretrained=False)
    >>> fi = backbone.feature_info
    >>> neck = FPN(feature_info=fi, out_channels=256)
    >>> head = DecoupledAnchorFreeHead(feature_info=fi, num_classes=80)
    >>> detector = CoreObjectDetector(
    ...     backbone=backbone, neck=neck, head=head,
    ... )
    >>> import torch
    >>> out = detector(torch.randn(1, 3, 224, 224))
    >>> isinstance(out, dict)
    True
"""

from __future__ import annotations

import torch
from torch import nn


class CoreObjectDetector(nn.Module):
    """End-to-end object detection model.

    Composes a backbone, optional neck, and detection head into a single
    module.  The forward pass flows through each component sequentially:

    ``input -> backbone -> (neck) -> head -> output``

    Args:
        backbone: Feature extractor implementing
            :class:`~corecv.core.contract.BaseBackbone`.
        neck: Optional feature pyramid network (FPN, PANet, etc.).
            Pass ``None`` to skip.
        head: Detection head producing class logits and box predictions.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module | None,
        head: nn.Module,
    ) -> None:
        """Initialise the detector.

        Args:
            backbone: Feature extractor.
            neck: Optional neck (``None`` to skip).
            head: Detection head.
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x: torch.Tensor) -> object:
        """Run the full detection pipeline.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Output from the detection head (typically a ``dict`` containing
            ``"cls_logits"`` and ``"pred_boxes"`` or ``"reg_pred"``).
        """
        features = self.backbone(x)
        if self.neck is not None:
            features = self.neck(features)
        return self.head(features)

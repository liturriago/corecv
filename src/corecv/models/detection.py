"""Detection model builders for CoreCV.

This module provides factory functions and wrappers for building anchor-free
object detection models using configurable backbones, necks, and a dual-head
detection head.

Supported head:
    DetectionHead (anchor-free, dual one-to-many/one-to-one)

Typical usage::

    from corecv.models.detection import create_detection_model

    model = create_detection_model("csp_nano", num_classes=80)
    (preds_o2m, preds_o2o) = model(images)  # each a (logits, boxes) tuple
"""

from __future__ import annotations

from torch import Tensor, nn

from corecv.models.backbones import BackboneName, create_backbone
from corecv.models.heads.detection import DetectionHead
from corecv.models.necks import NeckName, create_neck


class DetectionModel(nn.Module):
    """End-to-end anchor-free dual-head detection model.

    Combines a backbone feature extractor with an optional feature-fusion
    neck and a dense detection head. The head predicts class logits and box
    distances for every grid cell of the feature pyramid and exposes both a
    one-to-many and a one-to-one prediction set, the latter enabling NMS-free
    inference.

    Architecture::

        Image (B, 3, H, W)
            |
        Backbone  ->  [C3, C4, C5]
            |
        [Neck]    ->  [P3, P4, P5]  (optional)
            |
        Head (dual O2M/O2O detection)
            |
        ((o2m_logits, o2m_boxes), (o2o_logits, o2o_boxes))

    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        neck: nn.Module | None = None,
    ) -> None:
        """Initialize the detection model.

        Args:
            backbone: Backbone feature extractor that returns
                ``(features, feature_info)``.
            head: Detection head producing the dual prediction tuples.
            neck: Optional neck module that fuses multi-scale features.

        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(
        self,
        x: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """Run the full detection pipeline.

        Args:
            x: Input images of shape ``(B, 3, H, W)``.

        Returns:
            Tuple of ``(preds_one2many, preds_one2one)`` where each element
            is a ``(logits, boxes)`` tuple produced by the detection head.

        """
        features, _feature_info = self.backbone(x)
        if self.neck is not None:
            features = self.neck(features)
        return self.head(features)


def create_detection_model(
    backbone_name: BackboneName,
    num_classes: int,
    *,
    pretrained: bool = False,
    neck: NeckName | None = None,
    neck_out_channels: int = 256,
) -> DetectionModel:
    """Create a detection model with the specified backbone and neck.

    Args:
        backbone_name: Name of the backbone variant to use.
        num_classes: Number of output classes.
        pretrained: If ``True``, load ImageNet-1K pretrained backbone weights.
        neck: Optional neck name for feature fusion.  One of ``fpn``,
            ``panet``, ``bifpn``, ``csppanet``, or ``None`` to skip the neck.
        neck_out_channels: Output channel dimension for the neck module.
            Ignored when *neck* is ``None``.

    Returns:
        A fully assembled :class:`DetectionModel`.

    Raises:
        ValueError: If *neck* is not a recognized neck name.

    Example:
        >>> import torch
        >>> from corecv.models.detection import create_detection_model
        >>> model = create_detection_model("csp_nano", num_classes=10)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> (o2m_logits, o2m_boxes), (o2o_logits, o2o_boxes) = model(x)
        >>> o2m_logits.shape
        torch.Size([2, 1029, 10])

    """
    # Step 1: Build the backbone.
    backbone = create_backbone(backbone_name, pretrained=pretrained)
    feature_info = backbone.feature_info

    # Step 2: Build the optional neck.
    neck_module: nn.Module | None = None
    if neck is not None:
        neck_module = create_neck(
            name=neck,
            in_channels=feature_info.channels,
            out_channels=neck_out_channels,
        )
        head_in_channels: list[int] = [neck_out_channels] * feature_info.num_levels
    else:
        head_in_channels = feature_info.channels

    # Step 3: Build the detection head.
    head = DetectionHead(
        in_channels=head_in_channels,
        num_classes=num_classes,
        strides=tuple(feature_info.strides),
    )

    return DetectionModel(backbone=backbone, head=head, neck=neck_module)

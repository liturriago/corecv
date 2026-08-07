"""Segmentation model builders for CoreCV.

This module provides factory functions and wrappers for building
semantic segmentation models using configurable backbones, necks, and
decoder heads.

Supported decoders:
    deeplabv3plus, resunet

Typical usage::

    from corecv.models.segmentation import create_segmentation_model

    model = create_segmentation_model("resnet50", num_classes=21, pretrained=True)
    logits = model(images)  # (B, 21, H, W)
"""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from corecv.models.backbones import BackboneName, create_backbone
from corecv.models.heads.segmentation import DeepLabV3PlusHead, ResUNetDecoder
from corecv.models.necks import NeckName, create_neck

# Union of all supported decoder names.
DecoderName = Literal["deeplabv3plus", "resunet"]


class SegmentationModel(nn.Module):
    """End-to-end semantic segmentation model.

    Combines a backbone feature extractor, an optional feature-fusion neck,
    and a decoder head that produces dense per-pixel class logits.

    Architecture::

        Image (B, 3, H, W)
            |
        Backbone  ->  [C2, C3, C4, C5]
            |
        [Neck]    ->  [P2, P3, P4, P5]  (optional)
            |
        Head (decoder + classifier)
            |
        Logits (B, num_classes, H, W)
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        neck: nn.Module | None = None,
    ) -> None:
        """Initialize the segmentation model.

        Args:
            backbone: Backbone feature extractor that returns
                ``(features, feature_info)``.
            head: Decoder head producing ``(B, num_classes, H, W)`` logits
                from the multi-scale feature list.
            neck: Optional neck module that fuses multi-scale features.

        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        """Run the full segmentation pipeline.

        Args:
            x: Input images of shape ``(B, 3, H, W)``.

        Returns:
            Per-pixel class logits of shape ``(B, num_classes, H, W)``.

        """
        features, _feature_info = self.backbone(x)
        if self.neck is not None:
            features = self.neck(features)
        return self.head(features, input_size=x.shape[2:])


def create_segmentation_model(
    backbone_name: BackboneName,
    num_classes: int,
    *,
    pretrained: bool = False,
    neck: NeckName | None = None,
    neck_out_channels: int = 256,
    decoder: DecoderName = "deeplabv3plus",
) -> SegmentationModel:
    """Create a segmentation model with the specified backbone and decoder.

    Args:
        backbone_name: Name of the backbone variant to use.
        num_classes: Number of output classes.
        pretrained: If ``True``, load ImageNet-1K pretrained backbone weights.
        neck: Optional neck name for feature fusion.  One of ``fpn``,
            ``panet``, ``bifpn``, or ``None`` to skip the neck.
        neck_out_channels: Output channel dimension for the neck module.
            Ignored when *neck* is ``None``.
        decoder: Decoder head name. One of ``deeplabv3plus`` or ``resunet``.

    Returns:
        A fully assembled :class:`SegmentationModel`.

    Raises:
        ValueError: If *decoder* is not a recognized decoder name.

    Example:
        >>> import torch
        >>> from corecv.models.segmentation import create_segmentation_model
        >>> model = create_segmentation_model("resnet18", num_classes=21)
        >>> x = torch.randn(2, 3, 128, 128)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 21, 128, 128])

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
        head_channels: list[int] = [neck_out_channels] * feature_info.num_levels
    else:
        head_channels = feature_info.channels

    # Step 3: Build the decoder head.
    if decoder == "deeplabv3plus":
        head = DeepLabV3PlusHead(in_channels=head_channels, num_classes=num_classes)
    elif decoder == "resunet":
        head = ResUNetDecoder(in_channels=head_channels, num_classes=num_classes)
    else:
        msg = f"Unknown decoder: {decoder!r}. Choose from 'deeplabv3plus' or 'resunet'"
        raise ValueError(msg)

    return SegmentationModel(backbone=backbone, head=head, neck=neck_module)

"""Segmentation model builders for CoreCV.

This module provides factory functions and wrappers for building
semantic segmentation models using configurable backbones, necks,
and segmentation heads (DeepLabV3+ and ResUNetDecoder).

Supported backbones:
    resnet18, resnet34, resnet50, resnet101, resnet152,
    mobilenetv3_large, mobilenetv3_small,
    convnext_tiny, convnext_small, convnext_base,
    swin_t, swin_s, swin_b

Supported necks: fpn, panet, bifpn
Supported heads: resunet, deeplabv3plus

Typical usage::

    from corecv.models.model_segmentation import create_segmentation_model

    model = create_segmentation_model(
        "resnet50", neck_name="fpn", head_name="deeplabv3plus", num_classes=21,
    )
    logits = model(images)  # (B, 21, H, W)
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from corecv.models.backbones import BackboneName, create_backbone
from corecv.models.heads.segmentation import DeepLabV3PlusHead, ResUNetDecoder
from corecv.models.necks.bifpn import BiFPN
from corecv.models.necks.fpn import FPN
from corecv.models.necks.panet import PANet

# Type aliases for supported neck and head names.
NeckName = Literal["fpn", "panet", "bifpn"]
HeadName = Literal["resunet", "deeplabv3plus"]

# Maps neck name to neck class.
_NECK_BUILDERS: dict[str, type] = {
    "fpn": FPN,
    "panet": PANet,
    "bifpn": BiFPN,
}

# Maps head name to head class.
_HEAD_BUILDERS: dict[str, type] = {
    "resunet": ResUNetDecoder,
    "deeplabv3plus": DeepLabV3PlusHead,
}


class SegmentationModel(nn.Module):
    """End-to-end semantic segmentation model.

    Combines a backbone feature extractor with a feature-fusion neck
    and a segmentation decoder head.  The backbone produces multi-scale
    features; the neck unifies channel dimensions; the head produces
    per-pixel logits at the finest backbone resolution.

    Architecture::

        Image (B, 3, H, W)
            |
        Backbone  ->  [C2, C3, C4, C5]
            |
        Neck       ->  [P2, P3, P4, P5]
            |
        Head       ->  Logits (B, num_classes, H_feat, W_feat)

    Note:
        The output spatial dimensions match the finest backbone feature
        level (typically stride 4x).  For full-resolution output, apply
        ``F.interpolate(logits, size=(H, W), mode="bilinear")`` externally.

    Example:
        >>> import torch
        >>> from corecv.models.model_segmentation import create_segmentation_model
        >>> model = create_segmentation_model(
        ...     "resnet50", "fpn", "deeplabv3plus", num_classes=21,
        ... )
        >>> x = torch.randn(2, 3, 224, 224)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 21, 56, 56])
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
    ) -> None:
        """Initialize the segmentation model.

        Args:
            backbone: Backbone feature extractor that returns
                ``(features, feature_info)``.
            neck: Neck module that fuses multi-scale features and
                produces a list of feature tensors with unified channels.
            head: Segmentation head that produces per-pixel logits.
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
            Segmentation logits of shape ``(B, num_classes, H_feat, W_feat)``
            where ``(H_feat, W_feat)`` is the spatial size of the finest
            backbone feature level.
        """
        # Extract multi-scale features from backbone.
        features, _feature_info = self.backbone(x)

        # Fuse features through the neck.
        neck_features = self.neck(features)

        # Decode to per-pixel segmentation logits.
        return self.head(neck_features)


def create_segmentation_model(  # noqa: PLR0913
    backbone_name: BackboneName,
    neck_name: NeckName,
    head_name: HeadName,
    num_classes: int,
    *,
    pretrained: bool = False,
    neck_out_channels: int = 256,
    aspp_dilations: list[int] | None = None,
) -> SegmentationModel:
    """Create a segmentation model with the specified components.

    Args:
        backbone_name: Name of the backbone variant to use.
        neck_name: Name of the neck module.  One of ``fpn``, ``panet``,
            ``bifpn``.
        head_name: Name of the segmentation head.  One of ``resunet``,
            ``deeplabv3plus``.
        num_classes: Number of segmentation classes (including background
            if applicable).
        pretrained: If ``True``, load ImageNet-1K pretrained backbone weights.
        neck_out_channels: Output channel dimension for the neck module.
        aspp_dilations: Dilation rates for DeepLabV3+ ASPP module.  Only
            used when *head_name* is ``deeplabv3plus``.  Defaults to
            ``[6, 12, 18]``.

    Returns:
        A fully assembled :class:`SegmentationModel`.

    Raises:
        ValueError: If *neck_name* or *head_name* is not recognized.

    Example:
        >>> model = create_segmentation_model(
        ...     "resnet50", "fpn", "deeplabv3plus", num_classes=21,
        ... )
        >>> model.backbone.feature_info.channels
        [256, 512, 1024, 2048]
    """
    # Step 1: Build the backbone.
    backbone = create_backbone(backbone_name, pretrained=pretrained)
    feature_info = backbone.feature_info

    # Step 2: Build the neck.
    if neck_name not in _NECK_BUILDERS:
        msg = f"Unknown neck: {neck_name!r}. Choose from {list(_NECK_BUILDERS)}"
        raise ValueError(msg)

    neck_cls = _NECK_BUILDERS[neck_name]
    neck = neck_cls(
        in_channels=feature_info.channels,
        out_channels=neck_out_channels,
    )

    # Step 3: Build the segmentation head.
    # After the neck, all feature levels have neck_out_channels.
    neck_channels = [neck_out_channels] * feature_info.num_levels

    if head_name not in _HEAD_BUILDERS:
        msg = f"Unknown head: {head_name!r}. Choose from {list(_HEAD_BUILDERS)}"
        raise ValueError(msg)

    head_cls = _HEAD_BUILDERS[head_name]

    if head_name == "deeplabv3plus":
        head = head_cls(
            in_channels=neck_channels,
            num_classes=num_classes,
            aspp_dilations=aspp_dilations,
        )
    else:
        head = head_cls(
            in_channels=neck_channels,
            num_classes=num_classes,
        )

    return SegmentationModel(backbone=backbone, neck=neck, head=head)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    _batch_size = 2
    _num_classes = 21
    _img_size = 224

    print("=== Segmentation Model Tests ===\n")  # noqa: T201

    # Test representative backbone + neck + head combinations.
    _test_configs: list[tuple[BackboneName, NeckName, HeadName]] = [
        ("resnet18", "fpn", "resunet"),
        ("resnet50", "fpn", "deeplabv3plus"),
        ("resnet50", "panet", "resunet"),
        ("resnet50", "bifpn", "deeplabv3plus"),
        ("mobilenetv3_large", "fpn", "resunet"),
        ("mobilenetv3_small", "panet", "deeplabv3plus"),
        ("convnext_tiny", "fpn", "resunet"),
        ("convnext_small", "bifpn", "deeplabv3plus"),
        ("swin_t", "fpn", "resunet"),
        ("swin_t", "panet", "deeplabv3plus"),
    ]

    for backbone_name, neck_name, head_name in _test_configs:
        model = create_segmentation_model(
            backbone_name=backbone_name,
            neck_name=neck_name,
            head_name=head_name,
            num_classes=_num_classes,
            pretrained=False,
        )
        model.eval()

        dummy_input = torch.randn(_batch_size, 3, _img_size, _img_size)
        with torch.no_grad():
            logits = model(dummy_input)

        assert logits.shape[0] == _batch_size, f"Batch mismatch: {logits.shape}"  # noqa: S101
        assert logits.shape[1] == _num_classes, f"Class mismatch: {logits.shape}"  # noqa: S101
        print(  # noqa: T201
            f"  {backbone_name:25s} + {neck_name:6s} + {head_name:13s} "
            f"-> logits {logits.shape}  OK",
        )

    print("\nAll segmentation model tests passed.")  # noqa: T201

"""Classification model builders for CoreCV.

This module provides factory functions and wrappers for building
image classification models using configurable backbones and heads.

Supported backbones:
    resnet18, resnet34, resnet50, resnet101, resnet152,
    mobilenetv3_large, mobilenetv3_small,
    convnext_tiny, convnext_small, convnext_base,
    swin_t, swin_s, swin_b

Typical usage::

    from corecv.models.model_classification import create_classification_model

    model = create_classification_model("resnet50", num_classes=1000, pretrained=True)
    logits = model(images)  # (B, 1000)
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from corecv.models.backbones import BackboneName, create_backbone
from corecv.models.heads.classification import ClassificationHead
from corecv.models.necks.bifpn import BiFPN
from corecv.models.necks.fpn import FPN
from corecv.models.necks.panet import PANet

# Type alias for neck options.
NeckName = Literal["fpn", "panet", "bifpn"]

# Maps neck name to neck class.
_NECK_BUILDERS: dict[str, type] = {
    "fpn": FPN,
    "panet": PANet,
    "bifpn": BiFPN,
}


class ClassificationModel(nn.Module):
    """End-to-end image classification model.

    Combines a backbone feature extractor with an optional feature-fusion
    neck and a classification head.  The backbone produces multi-scale
    features; the head operates on the finest-level feature (or the
    neck output if a neck is provided).

    Architecture::

        Image (B, 3, H, W)
            |
        Backbone  ->  [C2, C3, C4, C5]
            |
        [Neck]    ->  [P2, P3, P4, P5]  (optional)
            |
        Head (uses finest level only)
            |
        Logits (B, num_classes)

    Example:
        >>> import torch
        >>> from corecv.models.model_classification import create_classification_model
        >>> model = create_classification_model("resnet50", num_classes=10)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 10])
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        neck: nn.Module | None = None,
    ) -> None:
        """Initialize the classification model.

        Args:
            backbone: Backbone feature extractor that returns
                ``(features, feature_info)``.
            head: Classification head that produces ``(B, num_classes)``
                logits from a single feature tensor.
            neck: Optional neck module that fuses multi-scale features.
                When ``None``, the finest backbone feature is used directly.
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        """Run the full classification pipeline.

        Args:
            x: Input images of shape ``(B, 3, H, W)``.

        Returns:
            Class logits of shape ``(B, num_classes)``.
        """
        # Extract multi-scale features from backbone.
        features, _feature_info = self.backbone(x)

        # Optionally fuse features through a neck.
        if self.neck is not None:
            features = self.neck(features)

        # Classification head: use the finest-level (index 0) feature.
        return self.head(features[0])


def create_classification_model(
    backbone_name: BackboneName,
    num_classes: int,
    *,
    pretrained: bool = False,
    neck: NeckName | None = None,
    neck_out_channels: int = 256,
) -> ClassificationModel:
    """Create a classification model with the specified backbone and neck.

    Args:
        backbone_name: Name of the backbone variant to use.
        num_classes: Number of output classes.
        pretrained: If ``True``, load ImageNet-1K pretrained backbone weights.
        neck: Optional neck name for feature fusion.  One of ``fpn``,
            ``panet``, ``bifpn``, or ``None`` to skip the neck.
        neck_out_channels: Output channel dimension for the neck module.
            Ignored when *neck* is ``None``.

    Returns:
        A fully assembled :class:`ClassificationModel`.

    Raises:
        ValueError: If *neck* is not a recognized neck name.

    Example:
        >>> model = create_classification_model(
        ...     "swin_t", num_classes=100, pretrained=False, neck="fpn",
        ... )
        >>> model.backbone.feature_info.channels
        [96, 192, 384, 768]
    """
    # Step 1: Build the backbone.
    backbone = create_backbone(backbone_name, pretrained=pretrained)
    feature_info = backbone.feature_info

    # Step 2: Build the optional neck.
    neck_module: nn.Module | None = None
    head_in_channels: int = feature_info.channels[0]

    if neck is not None:
        if neck not in _NECK_BUILDERS:
            msg = f"Unknown neck: {neck!r}. Choose from {list(_NECK_BUILDERS)}"
            raise ValueError(msg)

        neck_cls = _NECK_BUILDERS[neck]
        neck_module = neck_cls(
            in_channels=feature_info.channels,
            out_channels=neck_out_channels,
        )
        head_in_channels = neck_out_channels

    # Step 3: Build the classification head.
    head = ClassificationHead(
        in_channels=head_in_channels,
        num_classes=num_classes,
    )

    return ClassificationModel(backbone=backbone, head=head, neck=neck_module)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    _batch_size = 2
    _num_classes = 10
    _img_size = 224

    print("=== Classification Model Tests ===\n")  # noqa: T201

    # Test all backbone variants without neck.
    _backbone_names: list[BackboneName] = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "mobilenetv3_large",
        "mobilenetv3_small",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "swin_t",
        "swin_s",
        "swin_b",
    ]

    for name in _backbone_names:
        model = create_classification_model(
            backbone_name=name,
            num_classes=_num_classes,
            pretrained=False,
        )
        model.eval()

        dummy_input = torch.randn(_batch_size, 3, _img_size, _img_size)
        with torch.no_grad():
            logits = model(dummy_input)

        assert logits.shape == (_batch_size, _num_classes), (  # noqa: S101
            f"Expected ({_batch_size}, {_num_classes}), got {logits.shape}"
        )
        print(  # noqa: T201
            f"  {name:25s} -> logits {logits.shape}  OK",
        )

    # Test with FPN neck.
    print("\n--- With FPN neck ---")  # noqa: T201
    model_fpn = create_classification_model(
        backbone_name="resnet50",
        num_classes=_num_classes,
        pretrained=False,
        neck="fpn",
        neck_out_channels=256,
    )
    model_fpn.eval()

    dummy_input = torch.randn(_batch_size, 3, _img_size, _img_size)
    with torch.no_grad():
        logits_fpn = model_fpn(dummy_input)

    assert logits_fpn.shape == (_batch_size, _num_classes)  # noqa: S101
    print(f"  resnet50 + FPN -> logits {logits_fpn.shape}  OK")  # noqa: T201

    # Test with PANet neck.
    print("\n--- With PANet neck ---")  # noqa: T201
    model_panet = create_classification_model(
        backbone_name="convnext_tiny",
        num_classes=_num_classes,
        pretrained=False,
        neck="panet",
        neck_out_channels=256,
    )
    model_panet.eval()

    with torch.no_grad():
        logits_panet = model_panet(dummy_input)

    assert logits_panet.shape == (_batch_size, _num_classes)  # noqa: S101
    print(f"  convnext_tiny + PANet -> logits {logits_panet.shape}  OK")  # noqa: T201

    print("\nAll classification model tests passed.")  # noqa: T201

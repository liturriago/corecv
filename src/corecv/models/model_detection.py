"""Detection model builders for CoreCV.

This module provides factory functions and wrappers for building
anchor-free object detection models using configurable backbones,
necks, and dual-head detection heads (One-to-Many + One-to-One).

The model output is fully compatible with ``DualHeadDetectionLoss``
in ``corecv.losses.loss_detection``.

Supported backbones:
    resnet18, resnet34, resnet50, resnet101, resnet152,
    mobilenetv3_large, mobilenetv3_small,
    convnext_tiny, convnext_small, convnext_base,
    swin_t, swin_s, swin_b

Supported necks: fpn, panet, bifpn

Typical usage::

    from corecv.models.model_detection import create_detection_model
    from corecv.losses.loss_detection import DualHeadDetectionLoss

    model = create_detection_model("resnet50", "fpn", num_classes=80)
    preds_o2m, preds_o2o = model(images)

    loss_fn = DualHeadDetectionLoss(num_classes=80)
    losses = loss_fn(preds_o2m, preds_o2o, targets)
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from corecv.models.backbones import BackboneName, create_backbone
from corecv.models.heads.detection import AnchorFreeDetectionHead
from corecv.models.necks.bifpn import BiFPN
from corecv.models.necks.fpn import FPN
from corecv.models.necks.panet import PANet

# Type alias for supported neck names.
NeckName = Literal["fpn", "panet", "bifpn"]

# Maps neck name to neck class.
_NECK_BUILDERS: dict[str, type] = {
    "fpn": FPN,
    "panet": PANet,
    "bifpn": BiFPN,
}


class DetectionModel(nn.Module):
    """End-to-end anchor-free detection model with dual heads.

    Combines a backbone feature extractor with a feature-fusion neck
    and an anchor-free dual-head detector that produces both
    One-to-Many (O2M) and One-to-One (O2O) predictions.

    Architecture::

        Image (B, 3, H, W)
            |
        Backbone  ->  [C2, C3, C4, C5]
            |
        Neck       ->  [P2, P3, P4, P5]
            |
        Head       ->  (preds_o2m, preds_o2o)

    Output format (compatible with ``DualHeadDetectionLoss``)::

        preds_o2m = (pred_logits_o2m, pred_boxes_o2m)
        preds_o2o = (pred_logits_o2o, pred_boxes_o2o)

        pred_logits: (B, N_anchors, num_classes)  -- raw logits
        pred_boxes:  (B, N_anchors, 4)            -- [x1, y1, x2, y2]

    Example:
        >>> import torch
        >>> from corecv.models.model_detection import create_detection_model
        >>> model = create_detection_model("resnet50", "fpn", num_classes=80)
        >>> x = torch.randn(2, 3, 640, 640)
        >>> (logits_o2m, boxes_o2m), (logits_o2o, boxes_o2o) = model(x)
        >>> logits_o2m.shape, boxes_o2m.shape
        (torch.Size([2, 25200, 80]), torch.Size([2, 25200, 4]))
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
    ) -> None:
        """Initialize the detection model.

        Args:
            backbone: Backbone feature extractor that returns
                ``(features, feature_info)``.
            neck: Neck module that fuses multi-scale features and
                produces a list of feature tensors with unified channels.
            head: Anchor-free detection head that produces dual-head
                predictions.
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
            Tuple of two prediction tuples:

            - ``(pred_logits_o2m, pred_boxes_o2m)`` for the O2M head.
            - ``(pred_logits_o2o, pred_boxes_o2o)`` for the O2O head.

            Each ``pred_logits`` has shape ``(B, N_anchors, num_classes)``
            and each ``pred_boxes`` has shape ``(B, N_anchors, 4)`` in
            ``[x1, y1, x2, y2]`` format.
        """
        # Extract multi-scale features from backbone.
        features, _feature_info = self.backbone(x)

        # Fuse features through the neck.
        neck_features = self.neck(features)

        # Run the dual-head detection head.
        return self.head(neck_features)


def create_detection_model(  # noqa: PLR0913
    backbone_name: BackboneName,
    neck_name: NeckName,
    num_classes: int,
    *,
    pretrained: bool = False,
    neck_out_channels: int = 256,
    head_num_convs: int = 4,
) -> DetectionModel:
    """Create a detection model with the specified backbone and neck.

    Args:
        backbone_name: Name of the backbone variant to use.
        neck_name: Name of the neck module.  One of ``fpn``, ``panet``,
            ``bifpn``.
        num_classes: Number of foreground object classes.
        pretrained: If ``True``, load ImageNet-1K pretrained backbone weights.
        neck_out_channels: Output channel dimension for the neck module.
        head_num_convs: Number of ``Conv-BN-ReLU`` layers in the detection
            head's shared feature tower.

    Returns:
        A fully assembled :class:`DetectionModel`.

    Raises:
        ValueError: If *neck_name* is not a recognized neck name.

    Example:
        >>> model = create_detection_model("resnet50", "fpn", num_classes=80)
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

    # Step 3: Build the detection head.
    # After the neck, all feature levels have neck_out_channels.
    neck_channels = [neck_out_channels] * feature_info.num_levels

    head = AnchorFreeDetectionHead(
        in_channels=neck_channels,
        num_classes=num_classes,
        num_convs=head_num_convs,
    )

    return DetectionModel(backbone=backbone, neck=neck, head=head)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    _batch_size = 2
    _num_classes = 80
    _img_size = 640

    print("=== Detection Model Tests ===\n")  # noqa: T201

    # Test representative backbone + neck combinations.
    _test_configs: list[tuple[BackboneName, NeckName]] = [
        ("resnet18", "fpn"),
        ("resnet50", "fpn"),
        ("resnet50", "panet"),
        ("resnet50", "bifpn"),
        ("mobilenetv3_large", "fpn"),
        ("mobilenetv3_small", "panet"),
        ("convnext_tiny", "fpn"),
        ("convnext_small", "bifpn"),
        ("swin_t", "fpn"),
        ("swin_t", "panet"),
    ]

    for backbone_name, neck_name in _test_configs:
        model = create_detection_model(
            backbone_name=backbone_name,
            neck_name=neck_name,
            num_classes=_num_classes,
            pretrained=False,
        )
        model.eval()

        dummy_input = torch.randn(_batch_size, 3, _img_size, _img_size)
        with torch.no_grad():
            (logits_o2m, boxes_o2m), (logits_o2o, boxes_o2o) = model(dummy_input)

        assert logits_o2m.shape == (  # noqa: S101
            _batch_size,
            logits_o2m.shape[1],
            _num_classes,
        ), f"O2M logits shape mismatch: {logits_o2m.shape}"
        assert boxes_o2m.shape == (  # noqa: S101
            _batch_size,
            boxes_o2m.shape[1],
            4,
        ), f"O2M boxes shape mismatch: {boxes_o2m.shape}"
        assert logits_o2m.shape == logits_o2o.shape, "O2M/O2O logits mismatch"  # noqa: S101
        assert boxes_o2m.shape == boxes_o2o.shape, "O2M/O2O boxes mismatch"  # noqa: S101

        print(  # noqa: T201
            f"  {backbone_name:25s} + {neck_name:6s} "
            f"-> logits {logits_o2m.shape}, boxes {boxes_o2m.shape}  OK",
        )

    # Verify compatibility with DualHeadDetectionLoss.
    print("\n--- Loss compatibility check ---")  # noqa: T201
    from corecv.losses.loss_detection import DualHeadDetectionLoss

    model = create_detection_model("resnet50", "fpn", num_classes=_num_classes)
    model.train()

    dummy_images = torch.randn(_batch_size, 3, _img_size, _img_size)
    preds_o2m, preds_o2o = model(dummy_images)

    # Create dummy targets: [batch_idx, class_id, x1, y1, x2, y2] in normalized [0, 1] range
    dummy_targets = torch.tensor(
        [
            [0, 5, 0.1, 0.1, 0.3, 0.3],
            [1, 10, 0.2, 0.2, 0.4, 0.4],
        ],
        dtype=torch.float32,
    )

    loss_fn = DualHeadDetectionLoss(num_classes=_num_classes)
    losses = loss_fn(preds_o2m, preds_o2o, dummy_targets)

    print(f"  Loss keys: {list(losses.keys())}")  # noqa: T201
    print(f"  Total loss: {losses['loss_total'].item():.4f}")  # noqa: T201
    assert "loss_total" in losses, "Missing loss_total key"  # noqa: S101

    print("\nAll detection model tests passed.")  # noqa: T201

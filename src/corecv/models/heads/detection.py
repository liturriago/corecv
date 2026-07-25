"""Anchor-free detection head for CoreCV.

Implements an anchor-free object detection head with dual One-to-Many
(O2M) and One-to-One (O2O) branches for NMS-free training.  Dynamically
consumes ``FeatureInfo`` metadata from backbones.

All losses and metrics (e.g., mAP via ``torchvision.ops.box_iou``) run
entirely on GPU without pycocotools or CPU PCIe bottlenecks.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------


class _ConvBnRelu(nn.Module):
    """Convolution + Batch Normalization + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        """Initialize the block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Spatial size of the convolution kernel.
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Run the forward pass.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            Output tensor of shape ``(B, C_out, H, W)``.
        """
        return self.relu(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Anchor-Free Detection Head
# ---------------------------------------------------------------------------


class AnchorFreeDetectionHead(nn.Module):
    """Anchor-free dual-head detection head.

    Produces both **One-to-Many** (O2M) and **One-to-One** (O2O)
    predictions for NMS-free anchor-free object detection.  The two
    branches share the same backbone features but maintain separate
    classification and box-regression weights.

    The O2M head is trained with Task-Aligned top-K assignment while the
    O2O head uses top-1 assignment.  At inference time only the O2O head
    is used, eliminating the need for NMS post-processing.
    """

    def __init__(
        self,
        in_channels: list[int],
        num_classes: int,
        num_convs: int = 4,
    ) -> None:
        """Initialize the detection head.

        Args:
            in_channels: Channel counts for each feature level produced
                by the backbone/neck.
            num_classes: Number of foreground object classes.
            num_convs: Number of ``Conv-BN-ReLU`` layers in the shared
                per-level feature tower.
        """
        super().__init__()
        self.num_classes = num_classes

        # Shared feature tower applied per level
        self.shared_towers = nn.ModuleList()
        for ch in in_channels:
            layers: list[nn.Module] = [_ConvBnRelu(ch, ch) for _ in range(num_convs)]
            self.shared_towers.append(nn.Sequential(*layers))

        # O2M classification and box-regression branches (per level)
        self.o2m_cls_heads = nn.ModuleList(
            [nn.Conv2d(ch, num_classes, kernel_size=3, padding=1) for ch in in_channels],
        )
        self.o2m_reg_heads = nn.ModuleList(
            [nn.Conv2d(ch, 4, kernel_size=3, padding=1) for ch in in_channels],
        )

        # O2O classification and box-regression branches (per level)
        self.o2o_cls_heads = nn.ModuleList(
            [nn.Conv2d(ch, num_classes, kernel_size=3, padding=1) for ch in in_channels],
        )
        self.o2o_reg_heads = nn.ModuleList(
            [nn.Conv2d(ch, 4, kernel_size=3, padding=1) for ch in in_channels],
        )

        self._init_bias()

    # ------------------------------------------------------------------
    # Bias initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_conv_bias(conv: nn.Conv2d, value: float) -> None:
        """Set all bias values of a convolution layer.

        Args:
            conv: The convolution layer whose bias to initialize.
            value: Constant value for the bias.
        """
        if conv.bias is not None:
            nn.init.constant_(conv.bias, value)

    def _init_bias(self) -> None:
        """Initialize classification and regression biases.

        Classification heads are initialised with a negative bias to
        counteract foreground / background class imbalance at the start
        of training.
        """
        neg_bias = -2.19  # -log((1 - 0.1) / 0.1) for focal loss prior
        for head in (*self.o2m_cls_heads, *self.o2o_cls_heads):
            self._init_conv_bias(head, neg_bias)
        for head in (*self.o2m_reg_heads, *self.o2o_reg_heads):
            self._init_conv_bias(head, 0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        features: list[Tensor],
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """Run the forward pass.

        Args:
            features: List of multi-scale feature tensors from the neck,
                ordered from highest resolution to lowest.

        Returns:
            Tuple of two prediction tuples:

            - ``preds_o2m = (pred_logits, pred_boxes)`` for the O2M head.
            - ``preds_o2o = (pred_logits, pred_boxes)`` for the O2O head.

            Each ``pred_logits`` has shape ``(B, N_anchors, num_classes)``
            and each ``pred_boxes`` has shape ``(B, N_anchors, 4)`` in
            ``[x1, y1, x2, y2]`` format, where ``N_anchors`` is the total
            number of spatial anchor points across all feature levels.
        """
        o2m_logits_parts: list[Tensor] = []
        o2m_boxes_parts: list[Tensor] = []
        o2o_logits_parts: list[Tensor] = []
        o2o_boxes_parts: list[Tensor] = []

        for level_idx, feat in enumerate(features):
            # Shared feature refinement
            shared = self.shared_towers[level_idx](feat)

            # O2M branch predictions
            o2m_cls = self.o2m_cls_heads[level_idx](shared)
            o2m_reg = self.o2m_reg_heads[level_idx](shared).sigmoid()

            # O2O branch predictions
            o2o_cls = self.o2o_cls_heads[level_idx](shared)
            o2o_reg = self.o2o_reg_heads[level_idx](shared).sigmoid()

            # Reshape: (B, C, H, W) -> (B, H*W, C)
            o2m_logits_parts.append(o2m_cls.flatten(start_dim=2).permute(0, 2, 1))
            o2o_logits_parts.append(o2o_cls.flatten(start_dim=2).permute(0, 2, 1))

            # Box predictions: flatten + permute to (B, H*W, 4), then sort corners
            # to guarantee x1 <= x2 and y1 <= y2 (required by box_iou / CIoU).
            # Without this, sigmoid outputs can have x2 < x1 → area=0 → NaN loss.
            o2m_reg_flat = o2m_reg.flatten(start_dim=2).permute(0, 2, 1)
            o2m_boxes_parts.append(torch.cat([
                torch.min(o2m_reg_flat[..., :2], o2m_reg_flat[..., 2:]),
                torch.max(o2m_reg_flat[..., :2], o2m_reg_flat[..., 2:]),
            ], dim=-1))

            o2o_reg_flat = o2o_reg.flatten(start_dim=2).permute(0, 2, 1)
            o2o_boxes_parts.append(torch.cat([
                torch.min(o2o_reg_flat[..., :2], o2o_reg_flat[..., 2:]),
                torch.max(o2o_reg_flat[..., :2], o2o_reg_flat[..., 2:]),
            ], dim=-1))

        # Concatenate across all feature levels
        pred_logits_o2m = torch.cat(o2m_logits_parts, dim=1)
        pred_boxes_o2m = torch.cat(o2m_boxes_parts, dim=1)
        pred_logits_o2o = torch.cat(o2o_logits_parts, dim=1)
        pred_boxes_o2o = torch.cat(o2o_boxes_parts, dim=1)

        return (pred_logits_o2m, pred_boxes_o2m), (pred_logits_o2o, pred_boxes_o2o)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    _batch_size = 2
    _num_classes = 80
    _in_channels = [256, 256, 256, 256, 256]
    _spatial_sizes = [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]

    _head = AnchorFreeDetectionHead(
        in_channels=_in_channels,
        num_classes=_num_classes,
        num_convs=4,
    )

    _features = [
        torch.randn(_batch_size, ch, h, w)
        for ch, (h, w) in zip(_in_channels, _spatial_sizes, strict=True)
    ]

    (_logits_o2m, _boxes_o2m), (_logits_o2o, _boxes_o2o) = _head(_features)

    _total_anchors = sum(h * w for h, w in _spatial_sizes)

    print(f"Total anchors: {_total_anchors}")  # noqa: T201
    print(f"O2M logits: {_logits_o2m.shape}")  # noqa: T201
    print(f"O2M boxes:  {_boxes_o2m.shape}")  # noqa: T201
    print(f"O2O logits: {_logits_o2o.shape}")  # noqa: T201
    print(f"O2O boxes:  {_boxes_o2o.shape}")  # noqa: T201

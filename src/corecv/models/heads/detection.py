"""Anchor-free dual-head detection head for CoreCV.

Implements a dense anchor-free detection head that predicts, for every grid
cell of the input feature pyramid, per-class logits and the distances from
the cell center to the four edges of the enclosing box.

Two structurally identical copies of the box/class branches are trained in
parallel:

- A **one-to-many** head that supervises the backbone with a rich assignment
  of anchors per object.
- A **one-to-one** head that operates on *detached* features and predicts a
  single anchor per object, producing non-duplicated detections that make
  NMS-free (end-to-end) inference viable.

Reference:
    Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection",
    ICCV 2019.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from corecv.models.backbones.csp_pyramid import ConvBlock

# Box output channels per anchor: [left, top, right, bottom].
_BOX_COORDS = 4

# Grid-cell offset used to center anchor points within each cell.
_CELL_OFFSET = 0.5

# Reference input resolution used to compute the class-bias prior.
_REFERENCE_SIZE = 640

# Prior number of objects expected per grid cell at the reference size.
_PRIOR_OBJECTS = 5.0

# Box-bias constant that starts the regression with small boxes.
_INIT_BOX_BIAS = 2.0

# Default pixel stride of each feature level, ordered finest to coarsest.
_DEFAULT_STRIDES: tuple[int, ...] = (8, 16, 32)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution block.

    Applies a depthwise convolution followed by a pointwise convolution,
    each followed by batch norm and SiLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        """Initialize the depthwise-separable convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size of the depthwise convolution.
            stride: Stride of the depthwise convolution.

        """
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Run the depthwise-separable convolution.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Output tensor of shape ``(B, C_out, H, W)``.

        """
        x = self.act1(self.bn1(self.depthwise(x)))
        return self.act2(self.bn2(self.pointwise(x)))


class DetectionHead(nn.Module):
    """Anchor-free dual-head detection head.

    For each feature level, a box branch stacks two 3x3 convolutions followed
    by a pointwise projection to the four box distances, while a class branch
    stacks depthwise-separable convolutions followed by a pointwise projection
    to the class logits. Both branches exist twice: the ``cv2``/``cv3``
    one-to-many pair and the ``one2one_cv2``/``one2one_cv3`` one-to-one pair,
    which consumes detached features.

    The head flattens all levels into a single anchor set, decodes the box
    distances into ``(x1, y1, x2, y2)`` image coordinates, and returns the
    two prediction tuples for training and inference.

    Example:
        >>> import torch
        >>> from corecv.models.heads.detection import DetectionHead
        >>> head = DetectionHead(in_channels=[128, 128, 128], num_classes=5)
        >>> feats = [torch.randn(2, c, h, w) for c, h, w in
        ...          [(128, 16, 16), (128, 8, 8), (128, 4, 4)]]
        >>> (o2m_logits, o2m_boxes), (o2o_logits, o2o_boxes) = head(feats)
        >>> o2m_logits.shape
        torch.Size([2, 336, 5])
        >>> o2m_boxes.shape
        torch.Size([2, 336, 4])

    """

    def __init__(
        self,
        in_channels: list[int],
        num_classes: int,
        *,
        strides: tuple[int, ...] = _DEFAULT_STRIDES,
        reg_max: int = 1,
    ) -> None:
        """Initialize the detection head.

        Args:
            in_channels: Channel dimensions of each input feature level,
                ordered from finest to coarsest.
            num_classes: Number of output classes.
            strides: Pixel stride of each feature level relative to the
                input image, ordered finest to coarsest.
            reg_max: Number of regression bins. Only ``1`` (direct distance
                regression) is supported.

        Raises:
            ValueError: If the *strides* length does not match the number of
                feature levels, or if *reg_max* is not ``1``.

        """
        super().__init__()
        if len(strides) != len(in_channels):
            msg = (
                f"strides length {len(strides)} does not match in_channels "
                f"length {len(in_channels)}"
            )
            raise ValueError(msg)
        if reg_max != 1:
            msg = f"only reg_max=1 (direct distance regression) is supported, got {reg_max}"
            raise ValueError(msg)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max
        box_channels = _BOX_COORDS * reg_max

        # One-to-many branches (dense supervision of the backbone).
        self.cv2 = nn.ModuleList(
            [self._make_box_branch(in_ch, box_channels) for in_ch in in_channels],
        )
        self.cv3 = nn.ModuleList(
            [self._make_class_branch(in_ch, num_classes) for in_ch in in_channels],
        )

        # One-to-one branches (single anchor per object, detached features).
        self.one2one_cv2 = nn.ModuleList(
            [self._make_box_branch(in_ch, box_channels) for in_ch in in_channels],
        )
        self.one2one_cv3 = nn.ModuleList(
            [self._make_class_branch(in_ch, num_classes) for in_ch in in_channels],
        )

        self._init_weights()

    @staticmethod
    def _make_box_branch(in_channels: int, box_channels: int) -> nn.Sequential:
        """Build the box-distance branch for a single feature level."""
        return nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3),
            ConvBlock(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, box_channels, kernel_size=1),
        )

    @staticmethod
    def _make_class_branch(in_channels: int, num_classes: int) -> nn.Sequential:
        """Build the class-logit branch for a single feature level."""
        return nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels),
            DepthwiseSeparableConv(in_channels, in_channels),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def _init_weights(self) -> None:
        """Initialize convolution weights and the box/class bias priors."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Bias priors per level: constant box bias and a class-logit prior
        # that reflects the expected number of objects per grid cell.
        for level, stride in enumerate(self.strides):
            class_bias = math.log(
                _PRIOR_OBJECTS / (self.num_classes * (_REFERENCE_SIZE / stride) ** 2),
            )
            nn.init.constant_(self.cv2[level][-1].bias, _INIT_BOX_BIAS)
            nn.init.constant_(self.one2one_cv2[level][-1].bias, _INIT_BOX_BIAS)
            nn.init.constant_(self.cv3[level][-1].bias, class_bias)
            nn.init.constant_(self.one2one_cv3[level][-1].bias, class_bias)

    @staticmethod
    def _make_anchors(
        spatial_shapes: list[tuple[int, int]],
        strides: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Generate anchor points and strides for a feature pyramid.

        Args:
            spatial_shapes: Spatial ``(H, W)`` size of each feature level,
                ordered from finest to coarsest.
            strides: Pixel stride of each level relative to the input image.
            device: Device for the generated tensors.
            dtype: Dtype for the generated tensors.

        Returns:
            Tuple of ``(anchor_points, stride_tensor)`` with shapes
            ``(A, 2)`` and ``(A, 1)``, where ``A`` is the total number of
            anchor points across all levels.

        """
        points: list[Tensor] = []
        stride_tensors: list[Tensor] = []
        for (height, width), stride in zip(spatial_shapes, strides, strict=True):
            rows, cols = torch.meshgrid(
                torch.arange(height, device=device, dtype=dtype),
                torch.arange(width, device=device, dtype=dtype),
                indexing="xy",
            )
            cell_points = torch.stack([cols, rows], dim=-1).reshape(-1, 2)
            cell_points = cell_points + _CELL_OFFSET
            points.append(cell_points * stride)
            stride_tensors.append(
                torch.full(
                    (height * width, 1),
                    float(stride),
                    device=device,
                    dtype=dtype,
                ),
            )
        return torch.cat(points, dim=0), torch.cat(stride_tensors, dim=0)

    @staticmethod
    def _dist2bbox(points: Tensor, distances: Tensor) -> Tensor:
        """Decode box distances into boxes around anchor points.

        Args:
            points: Anchor points of shape ``(A, 2)`` in image coordinates.
            distances: Predicted ``[left, top, right, bottom]`` distances of
                shape ``(B, A, 4)`` in pixels.

        Returns:
            Boxes of shape ``(B, A, 4)`` in ``(x1, y1, x2, y2)`` format.

        """
        x1 = points[:, 0] - distances[..., 0]
        y1 = points[:, 1] - distances[..., 1]
        x2 = points[:, 0] + distances[..., 2]
        y2 = points[:, 1] + distances[..., 3]
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def forward(
        self,
        features: list[Tensor],
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """Run the dual detection head on multi-scale features.

        Args:
            features: List of multi-scale feature tensors ordered from
                finest to coarsest, each with shape ``(B, C_i, H_i, W_i)``.

        Returns:
            Tuple of ``(preds_one2many, preds_one2one)`` where each element
            is a ``(logits, boxes)`` tuple with shapes ``(B, A, num_classes)``
            and ``(B, A, 4)``. Boxes are decoded to ``(x1, y1, x2, y2)``
            image coordinates and ``A`` is the total number of anchors.

        """
        o2m_logits: list[Tensor] = []
        o2m_distances: list[Tensor] = []
        o2o_logits: list[Tensor] = []
        o2o_distances: list[Tensor] = []
        spatial_shapes: list[tuple[int, int]] = []

        for level, feat in enumerate(features):
            batch_size = feat.shape[0]
            height, width = feat.shape[2:]
            spatial_shapes.append((height, width))
            num_points = height * width

            detached = feat.detach()
            o2m_box = self.cv2[level](feat)
            o2m_cls = self.cv3[level](feat)
            o2o_box = self.one2one_cv2[level](detached)
            o2o_cls = self.one2one_cv3[level](detached)

            o2m_box = o2m_box.view(batch_size, _BOX_COORDS, num_points).transpose(1, 2)
            o2m_cls = o2m_cls.view(batch_size, self.num_classes, num_points).transpose(1, 2)
            o2o_box = o2o_box.view(batch_size, _BOX_COORDS, num_points).transpose(1, 2)
            o2o_cls = o2o_cls.view(batch_size, self.num_classes, num_points).transpose(1, 2)

            o2m_logits.append(o2m_cls)
            o2m_distances.append(o2m_box)
            o2o_logits.append(o2o_cls)
            o2o_distances.append(o2o_box)

        anchor_points, stride_tensor = self._make_anchors(
            spatial_shapes,
            self.strides,
            device=features[0].device,
            dtype=features[0].dtype,
        )

        o2m_logits = torch.cat(o2m_logits, dim=1)
        o2o_logits = torch.cat(o2o_logits, dim=1)
        o2m_boxes = self._dist2bbox(anchor_points, torch.cat(o2m_distances, dim=1) * stride_tensor)
        o2o_boxes = self._dist2bbox(anchor_points, torch.cat(o2o_distances, dim=1) * stride_tensor)

        self.anchor_points = anchor_points
        self.stride_tensor = stride_tensor

        return (o2m_logits, o2m_boxes), (o2o_logits, o2o_boxes)

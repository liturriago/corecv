"""Decoupled anchor-free detection head (FCOS / YOLOX style).

Implements a pure-convolution detection head with separate classification
and regression branches per feature level, fully aware of per-level strides
from the upstream :class:`~corecv.core.contract.FeatureInfo`.  The head is
lightweight and edge-friendly: every sub-module is a standard ``Conv2d`` +
``BatchNorm2d`` + activation stack with 1x1 output projections.

Architecture overview::

    For each feature level (e.g. stride4, stride8, stride16, stride32):

        feat_l ──> SharedConvStack(num_convs) ──> cls_branch ──> cls_out
                      │
                      └────────────────────────> reg_branch ──> reg_out

    cls_out : (B, num_classes, H_l, W_l)   — classification logits
    reg_out : (B, 4, H_l, W_l)             — bounding box regression
              (l, t, r, b distances from each cell centre)

    During training, a centerness target is computed per location to
    re-weight the regression loss; at inference the centerness score is
    multiplied into the classification score.

The head dynamically inspects ``FeatureInfo.channels`` and
``FeatureInfo.stride`` to build the correct number of per-level stacks
with matched input channel counts, making it transparent to backbone and
neck architecture changes.

Example:
    >>> from corecv.core.contract import FeatureInfo
    >>> from corecv.models.heads.detection.decoupled_anchor_free import (
    ...     DecoupledAnchorFreeHead,
    ... )
    >>> fi = FeatureInfo(
    ...     channels={"stride4": 256, "stride8": 512,
    ...               "stride16": 1024, "stride32": 2048},
    ...     strides={"stride4": 4, "stride8": 8,
    ...              "stride16": 16, "stride32": 32},
    ... )
    >>> head = DecoupledAnchorFreeHead(
    ...     feature_info=fi,
    ...     num_classes=80,
    ...     feat_channels=256,
    ...     num_convs=4,
    ... )
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from corecv.core.contract import FeatureInfo
from corecv.core.registry import register_head

# ---------------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------------


class _ConvBnAct(nn.Module):
    """Convolution + BatchNorm + activation block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Activated feature tensor of shape ``(B, out_channels, H, W)``.
        """
        return self.act(self.bn(self.conv(x)))


class _FeatureLevelHead(nn.Module):
    """Per-level classification and regression branch pair.

    Builds a shared ``ConvStack`` of ``_ConvBnAct`` blocks followed by two
    decoupled 1x1-conv output heads for classification and regression.

    Args:
        in_channels: Input channel count for this feature level.
        feat_channels: Intermediate channel count inside the conv stack.
        num_classes: Number of foreground classes.
        num_convs: Number of 3x3 convolution blocks in the shared stack.
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        num_classes: int,
        num_convs: int,
    ) -> None:
        super().__init__()

        # Shared conv stack
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(num_convs):
            layers.append(_ConvBnAct(current_channels, feat_channels))
            current_channels = feat_channels
        self.shared_convs = nn.Sequential(*layers)

        # Classification branch: 3x3 conv + 1x1 conv
        self.cls_branch = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, num_classes, kernel_size=1),
        )

        # Regression branch: 3x3 conv + 1x1 conv (l, t, r, b)
        self.reg_branch = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 4, kernel_size=1),
        )

        # Centerness branch: 3x3 conv + 1x1 conv
        self.crt_branch = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise conv and BN weights using prior-biased init.

        Classification biases are initialised to ``-log((1 - p) / p)``
        with ``p = 0.01`` to stabilise early-stage training.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Prior-biased classification bias
        prior_prob = 0.01
        bias_value = -torch.log(
            torch.tensor((1.0 - prior_prob) / prior_prob, dtype=torch.float32)
        )
        self.cls_branch[-1].bias.data.fill_(bias_value.item())

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run the per-level head.

        Args:
            x: Feature tensor of shape ``(B, in_channels, H, W)``.

        Returns:
            A tuple of:
                - ``cls_logits``: ``(B, num_classes, H, W)``
                - ``reg_pred``: ``(B, 4, H, W)``
                - ``centerness``: ``(B, 1, H, W)``
        """
        feat = self.shared_convs(x)
        return self.cls_branch(feat), self.reg_branch(feat), self.crt_branch(feat)


# ---------------------------------------------------------------------------
# Main head
# ---------------------------------------------------------------------------


@register_head("decoupled_anchor_free")
class DecoupledAnchorFreeHead(nn.Module):
    """FCOS/YOLOX-style decoupled anchor-free detection head.

    Dynamically adapts to an upstream :class:`FeatureInfo` by constructing
    per-level branches whose input channels and count match the feature
    map metadata.  The head is fully stride-aware: a per-level ``stride``
    buffer is registered so that downstream loss and post-processing can
    map spatial predictions back to the original image coordinates.

    Args:
        feature_info: Feature metadata from the backbone or neck.  The head
            builds one branch per key in ``feature_info.channels``.
        num_classes: Number of foreground object classes (excluding
            background).
        feat_channels: Intermediate channel dimension inside each per-level
            conv stack.  Default ``256``.
        num_convs: Number of 3x3 ``Conv-BN-ReLU`` blocks in the shared
            portion of each branch.  Default ``4``.
    """

    def __init__(
        self,
        feature_info: FeatureInfo,
        num_classes: int,
        feat_channels: int = 256,
        num_convs: int = 4,
    ) -> None:
        """Initialise the decoupled anchor-free detection head.

        Builds per-level branches matching the ``FeatureInfo`` metadata and
        registers per-level stride buffers for coordinate mapping.
        """
        super().__init__()

        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self._feature_info = feature_info

        # Preserve insertion order so levels go finest -> coarsest
        self._level_names: list[str] = list(feature_info.channels.keys())
        self._level_channels: list[int] = [
            feature_info.channels[k] for k in self._level_names
        ]
        self._level_strides: list[int] = [
            feature_info.strides[k] for k in self._level_names
        ]

        # Per-level branches
        self.level_heads = nn.ModuleList(
            [
                _FeatureLevelHead(
                    in_channels=ch,
                    feat_channels=feat_channels,
                    num_classes=num_classes,
                    num_convs=num_convs,
                )
                for ch in self._level_channels
            ]
        )

        # Register stride buffers (non-learnable, move with device)
        for name, stride in zip(self._level_names, self._level_strides, strict=True):
            self.register_buffer(
                f"stride_{name}",
                torch.tensor(stride, dtype=torch.int64, requires_grad=False),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply initialisation to all ``_FeatureLevelHead`` modules."""
        # _FeatureLevelHead._init_weights already handles per-module init.

    @property
    def num_levels(self) -> int:
        """Return the number of feature levels this head operates on."""
        return len(self._level_names)

    @property
    def level_names(self) -> list[str]:
        """Return the ordered list of feature level names."""
        return list(self._level_names)

    @property
    def level_strides(self) -> list[int]:
        """Return the ordered list of per-level stride values."""
        return list(self._level_strides)

    def forward(
        self,
        features: list[Tensor],
    ) -> dict[str, list[Tensor]]:
        """Run detection on all feature levels.

        Args:
            features: List of feature tensors from the backbone or neck,
                one per level, ordered from finest to coarsest spatial
                resolution.  ``len(features)`` must equal
                ``self.num_levels``.

        Returns:
            Dictionary with keys:
                - ``"cls_logits"``: list of ``(B, num_classes, H_l, W_l)``
                - ``"reg_pred"``: list of ``(B, 4, H_l, W_l)``
                - ``"centerness"``: list of ``(B, 1, H_l, W_l)``
        """
        if len(features) != self.num_levels:
            msg = (
                f"Expected {self.num_levels} feature levels, "
                f"got {len(features)}."
            )
            raise ValueError(msg)

        cls_logits: list[Tensor] = []
        reg_pred: list[Tensor] = []
        centerness: list[Tensor] = []

        for feat, head in zip(features, self.level_heads, strict=True):
            cls, reg, crt = head(feat)
            cls_logits.append(cls)
            reg_pred.append(reg)
            centerness.append(crt)

        return {
            "cls_logits": cls_logits,
            "reg_pred": reg_pred,
            "centerness": centerness,
        }

    def get_stride_tensors(self, device: torch.device) -> list[Tensor]:
        """Return per-level stride tensors on the given device.

        Used by downstream loss modules and post-processors that need
        to map per-cell spatial predictions back to image coordinates.

        Args:
            device: Target device for the stride tensors.

        Returns:
            List of scalar ``int64`` tensors, one per level.
        """
        return [
            getattr(self, f"stride_{name}").to(device=device)
            for name in self._level_names
        ]

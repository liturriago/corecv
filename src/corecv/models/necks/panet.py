"""Path Aggregation Network (PANet) neck implementation.

Implements the PANet architecture from "Path Aggregation Network for
Instance Segmentation" (Liu et al., 2018).  PANet extends FPN by adding
a **bottom-up path augmentation** on top of the FPN's top-down pyramid,
providing a shorter information path from low-level features to the
deepest layers and improving feature fusion across all scales.

Architecture
------------

Given backbone feature levels ``[C2, C3, C4, C5]`` at strides
``[4, 8, 16, 32]``:

1. **FPN top-down pathway** (identical to :class:`~corecv.models.necks.fpn.FPN`)
   produces ``[P2, P3, P4, P5]``.
2. **Bottom-up path augmentation** merges FPN outputs from shallowest to
   deepest via stride-2 convolutions for spatial downsampling and
   element-wise addition, producing ``[N2, N3, N4, N5]``.
3. **Output 3x3 convolutions** are applied after each bottom-up merge.

The result is ``[N2, N3, N4, N5]`` -- one feature map per backbone level,
all with ``out_channels`` channels.

Dynamic Channel Projection
--------------------------

Like :class:`~corecv.models.necks.fpn.FPN`, the PANet accepts any
:class:`~corecv.core.contract.FeatureInfo` and dynamically instantiates
all 1x1 and 3x3 convolutions based on backbone metadata.

Example:
    >>> from corecv.models.backbones.resnet import ResNet50Backbone
    >>> from corecv.models.necks.panet import PANet
    >>> backbone = ResNet50Backbone(pretrained=False)
    >>> neck = PANet(feature_info=backbone.feature_info, out_channels=256)
    >>> features = backbone(torch.randn(1, 3, 224, 224))
    >>> pyramid = neck(features)
    >>> tuple(f.shape for f in pyramid)
    (torch.Size([1, 256, 56, 56]), torch.Size([1, 256, 28, 28]),
     torch.Size([1, 256, 14, 14]), torch.Size([1, 256, 7, 7]))
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from corecv.core.contract import FeatureInfo
from corecv.core.registry import register_neck


def _sorted_levels(
    feature_info: FeatureInfo,
) -> list[tuple[str, int, int]]:
    """Sort feature levels by stride in ascending order.

    Args:
        feature_info: Backbone feature metadata.

    Returns:
        A list of ``(level_name, stride, in_channels)`` tuples sorted
        from smallest to largest stride.
    """
    pairs = [
        (name, feature_info.strides[name], feature_info.channels[name])
        for name in feature_info.channels
    ]
    return sorted(pairs, key=lambda x: x[1])


@register_neck("panet")
class PANet(nn.Module):
    """Path Aggregation Network neck with dynamic channel projection.

    Combines a top-down FPN pathway with a bottom-up path augmentation,
    yielding richer multi-scale features for detection and segmentation
    heads.

    Attributes:
        out_channels: Uniform channel count for all output feature levels.
        levels: Sorted list of ``(level_name, stride, in_channels)`` tuples.
        lateral_convs: 1x1 convolutions for FPN top-down lateral connections.
        fpn_convs: 3x3 convolutions after FPN top-down merges.
        panet_reduce_convs: 3x3 stride-2 convolutions for bottom-up spatial
            downsampling between adjacent PANet levels.
        panet_lateral_convs: 1x1 convolutions that project FPN outputs into
            the PANet bottom-up pathway.
        panet_convs: 3x3 convolutions applied after each bottom-up addition.
    """

    def __init__(
        self,
        feature_info: FeatureInfo,
        out_channels: int = 256,
    ) -> None:
        """Initialise the PANet neck.

        Args:
            feature_info: Metadata describing the backbone's feature levels.
                Used to determine the number of levels and their input
                channel counts for all convolution construction.
            out_channels: Output channel dimension for every pyramid level.
                All lateral and projection convolutions target this width.
        """
        super().__init__()
        self.out_channels = out_channels
        self.levels = _sorted_levels(feature_info)
        num_levels = len(self.levels)

        # -----------------------------------------------------------------
        # FPN top-down pathway (same as FPN class)
        # -----------------------------------------------------------------
        lateral_convs: list[tuple[str, nn.Module]] = []
        fpn_convs: list[tuple[str, nn.Module]] = []

        for name, _stride, in_ch in self.levels:
            lateral_convs.append(
                (
                    name,
                    nn.Conv2d(in_ch, out_channels, kernel_size=1),
                )
            )
            fpn_convs.append(
                (
                    name,
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                )
            )

        self.lateral_convs = nn.ModuleDict(OrderedDict(lateral_convs))
        self.fpn_convs = nn.ModuleDict(OrderedDict(fpn_convs))

        # -----------------------------------------------------------------
        # Bottom-up path augmentation
        # -----------------------------------------------------------------

        # 1x1 lateral convolutions: project FPN outputs into PANet pathway
        panet_lateral_convs: list[tuple[str, nn.Module]] = []

        # 3x3 stride-2 convolutions: downsample from level i to level i+1
        panet_reduce_convs: list[tuple[str, nn.Module]] = []

        # 3x3 output convolutions: applied after each bottom-up addition
        panet_convs: list[tuple[str, nn.Module]] = []

        for _i, (name, _stride, _in_ch) in enumerate(self.levels):
            panet_lateral_convs.append(
                (
                    name,
                    nn.Conv2d(out_channels, out_channels, kernel_size=1),
                )
            )
            panet_convs.append(
                (
                    name,
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                )
            )

        # Stride-2 conv from level i -> level i+1 exists for all but the
        # deepest level (deepest level is the starting point of the
        # bottom-up path and requires no downsampling).
        for i in range(num_levels - 1):
            src_name = self.levels[i][0]
            panet_reduce_convs.append(
                (
                    f"{src_name}_reduce",
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                )
            )

        self.panet_lateral_convs = nn.ModuleDict(OrderedDict(panet_lateral_convs))
        self.panet_reduce_convs = nn.ModuleDict(OrderedDict(panet_reduce_convs))
        self.panet_convs = nn.ModuleDict(OrderedDict(panet_convs))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Apply FPN top-down then PANet bottom-up pathways.

        Args:
            features: Sequence of backbone feature tensors **in stride-
                ascending order** (highest resolution first).  The length
                must match the number of levels in the
                :class:`FeatureInfo` provided at construction time.

        Returns:
            A list of feature tensors in stride-ascending order (smallest
            stride first), each with shape
            ``(B, out_channels, H_i, W_i)``.

        Raises:
            ValueError: If the number of input features does not match
                the number of registered backbone levels.
        """
        if len(features) != len(self.levels):
            msg = (
                f"Expected {len(self.levels)} feature maps "
                f"(one per backbone level), received {len(features)}."
            )
            raise ValueError(msg)

        num_levels = len(self.levels)

        # -----------------------------------------------------------------
        # Stage 1 -- FPN top-down pathway
        # -----------------------------------------------------------------

        # Lateral projections to uniform channel dimension
        laterals: list[torch.Tensor] = []
        for i, (name, _stride, _in_ch) in enumerate(self.levels):
            laterals.append(self.lateral_convs[name](features[i]))

        # Top-down merge: deepest -> shallowest
        for i in range(num_levels - 1, 0, -1):
            target_h = laterals[i - 1].shape[-2]
            target_w = laterals[i - 1].shape[-1]
            upsampled = F.interpolate(
                laterals[i],
                size=(target_h, target_w),
                mode="nearest",
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # FPN output convolutions
        fpn_out: list[torch.Tensor] = []
        for i, (name, _stride, _in_ch) in enumerate(self.levels):
            fpn_out.append(self.fpn_convs[name](laterals[i]))

        # -----------------------------------------------------------------
        # Stage 2 -- Bottom-up path augmentation
        # -----------------------------------------------------------------

        # Project each FPN output into the PANet pathway
        panet_features: list[torch.Tensor] = []
        for i, (name, _stride, _in_ch) in enumerate(self.levels):
            panet_features.append(self.panet_lateral_convs[name](fpn_out[i]))

        # Bottom-up merge: shallowest -> deepest
        # The deepest level (index num_levels - 1) is the starting point
        # and only needs its lateral projection (already done above).
        for i in range(1, num_levels):
            src_name = self.levels[i - 1][0]
            downsampled = self.panet_reduce_convs[f"{src_name}_reduce"](
                panet_features[i - 1],
            )
            panet_features[i] = panet_features[i] + downsampled

        # PANet output convolutions
        outputs: list[torch.Tensor] = []
        for i, (name, _stride, _in_ch) in enumerate(self.levels):
            outputs.append(self.panet_convs[name](panet_features[i]))

        return outputs

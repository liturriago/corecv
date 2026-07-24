"""Feature Pyramid Network (FPN) neck implementation.

Implements the FPN architecture from "Feature Pyramid Networks for Object
Detection" (Lin et al., 2017). The FPN augments a backbone's multi-scale
feature maps with a top-down pathway and lateral connections, producing
pyramid feature maps at every scale with a uniform channel dimension.

Architecture
------------

Given backbone feature levels ``[C2, C3, C4, C5]`` at strides
``[4, 8, 16, 32]``:

1. **Lateral 1x1 convolutions** project each level to ``out_channels``.
2. **Top-down pathway** merges coarse features into finer scales via
   nearest-neighbour upsampling and element-wise addition.
3. **Output 3x3 convolutions** reduce aliasing artifacts from upsampling.

The result is ``[P2, P3, P4, P5]`` -- one feature map per backbone level,
all with ``out_channels`` channels.

Dynamic Channel Projection
--------------------------

The neck accepts any :class:`~corecv.core.contract.FeatureInfo` and
automatically instantiates the correct number of 1x1 lateral convolutions
with input channels derived from the backbone metadata. This means the
same :class:`FPN` class works with ResNet, MobileNetV3, ConvNeXt, ViT,
or any future backbone that implements :class:`BaseBackbone`.

Example:
    >>> from corecv.models.backbones.resnet import ResNet50Backbone
    >>> from corecv.models.necks.fpn import FPN
    >>> backbone = ResNet50Backbone(pretrained=False)
    >>> neck = FPN(feature_info=backbone.feature_info, out_channels=256)
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


@register_neck("fpn")
class FPN(nn.Module):
    """Feature Pyramid Network neck with dynamic channel projection.

    Produces a multi-scale feature pyramid from a backbone's intermediate
    feature maps.  The number of lateral convolutions, their input channels,
    and the interpolation targets are all derived at construction time from
    the provided :class:`FeatureInfo`, making this neck backbone-agnostic.

    Attributes:
        out_channels: Uniform channel count for all output feature levels.
        levels: Sorted list of ``(level_name, stride, in_channels)`` tuples.
        lateral_convs: Module mapping level name -> 1x1 conv for channel
            alignment in the top-down pathway.
        fpn_convs: Module mapping level name -> 3x3 conv applied after the
            top-down merge to reduce upsampling aliasing.
    """

    def __init__(
        self,
        feature_info: FeatureInfo,
        out_channels: int = 256,
    ) -> None:
        """Initialise the FPN neck.

        Args:
            feature_info: Metadata describing the backbone's feature levels.
                Used to determine the number of levels and their input
                channel counts for lateral convolution construction.
            out_channels: Output channel dimension for every pyramid level.
                All lateral 1x1 convolutions project to this width.
        """
        super().__init__()
        self.out_channels = out_channels
        self.levels = _sorted_levels(feature_info)

        # 1x1 lateral convolutions: project backbone channels -> out_channels
        lateral_convs: list[tuple[str, nn.Module]] = []
        # 3x3 output convolutions: suppress upsampling aliasing
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Apply the FPN top-down pathway to backbone features.

        Args:
            features: Sequence of backbone feature tensors **in stride-
                ascending order** (i.e. highest resolution first).  The
                length must match the number of levels in the
                :class:`FeatureInfo` provided at construction time.

        Returns:
            A list of pyramid feature tensors in stride-ascending order
            (smallest stride first), each with shape
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

        # Step 1 -- Lateral projections (1x1 convs) to uniform channel dim
        laterals: list[torch.Tensor] = []
        for i, (name, _stride, _in_ch) in enumerate(self.levels):
            laterals.append(self.lateral_convs[name](features[i]))

        # Step 2 -- Top-down pathway (deepest -> shallowest)
        # Start from the deepest level (last element) and propagate
        # coarser features into finer scales via nearest-neighbour upsample
        # and element-wise addition.
        for i in range(len(self.levels) - 1, 0, -1):
            target_h = laterals[i - 1].shape[-2]
            target_w = laterals[i - 1].shape[-1]
            upsampled = F.interpolate(
                laterals[i],
                size=(target_h, target_w),
                mode="nearest",
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Step 3 -- Output 3x3 convolutions to reduce aliasing
        outputs: list[torch.Tensor] = []
        for i, (name, _stride, _in_ch) in enumerate(self.levels):
            outputs.append(self.fpn_convs[name](laterals[i]))

        return outputs

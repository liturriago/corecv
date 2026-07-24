"""ASPPDecoder: DeepLabV3+ style segmentation decoder with ASPP module.

Implements a decoder that uses Atrous Spatial Pyramid Pooling (ASPP) on the
coarsest feature level to capture multi-scale context, followed by a simple
lightweight decoder that fuses low-level features from the encoder for
boundary refinement.

The decoder dynamically adapts its input projection layers based on
:class:`~corecv.core.contract.FeatureInfo` metadata, making it compatible
with any backbone that conforms to the
:class:`~corecv.core.contract.BaseBackbone` contract.

Architecture overview::

    Encoder features (sorted finest -> coarsest):
        f_stride4  (B, C1, H/4,  W/4)   -- low-level skip
        f_stride8  (B, C2, H/8,  W/8)
        f_stride16 (B, C3, H/16, W/16)
        f_stride32 (B, C4, H/32, W/32)  -- high-level input
            |
            v
        ASPP on f_stride32 (rates 6, 12, 18) + global pooling
            |
            v
        1x1 conv -> (B, 256, H/32, W/32)
            |
            v
        4x upsample -> (B, 256, H/8, W/8)
            |
            v
        cat(low-level f_stride8, projected) -> 1x1 conv -> Refine
            |
            v
        4x upsample -> (B, 256, H/2, W/2)
            |
            v
        1x1 Conv -> (B, num_classes, H, W)

Example:
    >>> from corecv.models.backbones.resnet import ResNet50Backbone
    >>> from corecv.models.heads.segmentation.aspp_decoder import ASPPDecoder
    >>> backbone = ResNet50Backbone(pretrained=False)
    >>> fi = backbone.feature_info
    >>> decoder = ASPPDecoder(
    ...     feature_info=fi, out_channels=256, num_classes=21
    ... )
    >>> feats = backbone(torch.randn(1, 3, 224, 224))
    >>> logits = decoder(feats)  # (1, 21, 224, 224)
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from corecv.core.contract import FeatureInfo
from corecv.core.registry import register_head

# Minimum number of feature levels required by the decoder.
_MIN_LEVELS: int = 2

# Fixed channel dimension for low-level feature projection in the decoder.
_LOW_LEVEL_CHANNELS: int = 48


# ---------------------------------------------------------------------------
# ASPP (Atrous Spatial Pyramid Pooling)
# ---------------------------------------------------------------------------


class _ASPPConv(nn.Module):
    """Single branch of the ASPP module: 3x3 atrous convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        dilation: Atrous (dilated) convolution rate.
    """

    def __init__(
        self, in_channels: int, out_channels: int, dilation: int
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply atrous convolution -> BN -> ReLU.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Output tensor of shape ``(B, C_out, H, W)``.
        """
        return self.relu(self.bn(self.conv(x)))


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module.

    Captures multi-scale contextual information by applying multiple parallel
    atrous convolutions with different dilation rates, plus a global average
    pooling branch.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels per branch.
        atrous_rates: Sequence of dilation rates for the parallel branches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates: Sequence[int] = (6, 12, 18),
    ) -> None:
        super().__init__()
        # Branch 1: 1x1 convolution (local context).
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Branches 2+: 3x3 atrous convolutions with different rates.
        self.branches = nn.ModuleList(
            [_ASPPConv(in_channels, out_channels, rate) for rate in atrous_rates]
        )

        # Global average pooling branch.
        # Note: No BatchNorm here to avoid training failures with
        # batch_size=1, since AdaptiveAvgPool2d(1) produces a 1x1 spatial
        # output where BN requires > 1 value per channel.
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Project concatenated features.
        num_branches = 1 + len(atrous_rates) + 1  # 1x1 + atrous + global
        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels * num_branches,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply ASPP and project concatenated features.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Output tensor of shape ``(B, out_channels, H, W)``.
        """
        h, w = x.shape[2], x.shape[3]

        # Collect branch outputs.
        parts: list[Tensor] = [self.branch1(x)]
        for branch in self.branches:
            parts.append(branch(x))

        # Global pooling branch: upsample to match spatial dims.
        gp = self.global_pool(x)
        gp = nn.functional.interpolate(
            gp, size=(h, w), mode="bilinear", align_corners=False
        )
        parts.append(gp)

        # Concatenate and project.
        return self.project(torch.cat(parts, dim=1))


# ---------------------------------------------------------------------------
# Lightweight decoder
# ---------------------------------------------------------------------------


class _DecoderBlock(nn.Module):
    """Lightweight decoder block: upsample + concat low-level + refine.

    Args:
        in_channels: Channels of the high-level feature being upsampled.
        low_level_channels: Channels of the low-level skip connection.
        out_channels: Output channels after fusion.
    """

    def __init__(
        self,
        in_channels: int,
        low_level_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        # 4x upsample via transposed convolution.
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Project low-level features to a fixed channel dimension.
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(
                low_level_channels, _LOW_LEVEL_CHANNELS,
                kernel_size=1, bias=False,
            ),
            nn.BatchNorm2d(_LOW_LEVEL_CHANNELS),
            nn.ReLU(inplace=True),
        )
        # Refine after concatenation.
        refine_in = out_channels + _LOW_LEVEL_CHANNELS
        self.refine = nn.Sequential(
            nn.Conv2d(refine_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, low_level: Tensor) -> Tensor:
        """Upsample high-level features, fuse with low-level, and refine.

        Args:
            x: High-level feature tensor of shape ``(B, C, H, W)``.
            low_level: Low-level skip tensor of shape
                ``(B, C_low, 2H, 2W)``.

        Returns:
            Refined tensor of shape ``(B, out_channels, 2H, 2W)``.
        """
        x = self.up(x)
        # Handle spatial mismatch (e.g. odd input sizes).
        diff_h = low_level.shape[2] - x.shape[2]
        diff_w = low_level.shape[3] - x.shape[3]
        x = nn.functional.pad(
            x,
            [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
        )
        low = self.low_level_proj(low_level)
        x = torch.cat([x, low], dim=1)
        return self.refine(x)


# ---------------------------------------------------------------------------
# Main decoder
# ---------------------------------------------------------------------------


@register_head("aspp_decoder")
class ASPPDecoder(nn.Module):
    """DeepLabV3+ style segmentation decoder with ASPP and lightweight decoder.

    Applies ASPP (Atrous Spatial Pyramid Pooling) to the coarsest feature
    level for multi-scale context aggregation, then fuses with low-level
    encoder features through a lightweight decoder path.

    Input projections are created dynamically from
    :class:`~corecv.core.contract.FeatureInfo` metadata, making this
    decoder compatible with any backbone that exposes multi-scale features.

    For backbones with 4 levels (stride 4, 8, 16, 32):
        - ASPP is applied to stride-32 features.
        - Low-level features come from stride-8.
        - Decoder upsamples 32->8 (4x) then 8->2 (4x) for 8x total.

    For backbones with 3 levels (stride 8, 16, 32):
        - ASPP is applied to stride-32 features.
        - Low-level features come from stride-8 (the finest available).
        - Decoder upsamples 32->8 (4x) then 8->2 (4x) for 8x total.

    Args:
        feature_info: Feature metadata from the backbone or neck, containing
            channel counts and strides for each feature level.
        out_channels: Internal channel dimension throughout the decoder.
        num_classes: Number of output segmentation classes.
        atrous_rates: Dilation rates for the ASPP module.
        dropout: Dropout probability in the ASPP and classification head.

    Raises:
        ValueError: If ``feature_info`` contains fewer than 2 feature levels.
    """

    def __init__(
        self,
        feature_info: FeatureInfo,
        out_channels: int = 256,
        num_classes: int = 21,
        atrous_rates: Sequence[int] = (6, 12, 18),
        dropout: float = 0.5,
    ) -> None:
        """Initialise the ASPP decoder.

        Args:
            feature_info: Feature metadata from the backbone or neck.
            out_channels: Internal channel dimension.
            num_classes: Number of segmentation classes.
            atrous_rates: ASPP dilation rates.
            dropout: Dropout probability.
        """
        super().__init__()
        self.num_classes = num_classes

        # ------------------------------------------------------------------
        # Sort feature levels by stride (ascending = finest first).
        # ------------------------------------------------------------------
        sorted_levels = sorted(
            feature_info.strides.keys(),
            key=lambda k: feature_info.strides[k],
        )
        if len(sorted_levels) < _MIN_LEVELS:
            msg = (
                f"ASPPDecoder requires at least {_MIN_LEVELS} feature "
                f"levels, got {len(sorted_levels)}."
            )
            raise ValueError(msg)

        self._sorted_levels = sorted_levels
        in_channels_list = [
            feature_info.channels[lvl] for lvl in sorted_levels
        ]

        # ------------------------------------------------------------------
        # Input projection: map each backbone feature to out_channels.
        # ------------------------------------------------------------------
        self.input_projections = nn.ModuleDict(
            {
                lvl: nn.Sequential(
                    nn.Conv2d(ch, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for lvl, ch in zip(
                    sorted_levels, in_channels_list, strict=True
                )
            }
        )

        # ------------------------------------------------------------------
        # ASPP on the coarsest level.
        # ------------------------------------------------------------------
        self.aspp = _ASPPModule(
            in_channels=out_channels,
            out_channels=out_channels,
            atrous_rates=atrous_rates,
        )

        # ------------------------------------------------------------------
        # Decoder path: two stages of upsample + low-level fusion.
        # ------------------------------------------------------------------
        # Stage 1: ASPP output (coarsest) -> upsample to second-finest.
        self.decoder_stage1 = _DecoderBlock(
            in_channels=out_channels,
            low_level_channels=out_channels,
            out_channels=out_channels,
        )

        # Stage 2: upsample to finest resolution.
        self.decoder_stage2 = _DecoderBlock(
            in_channels=out_channels,
            low_level_channels=out_channels,
            out_channels=out_channels,
        )

        # ------------------------------------------------------------------
        # Classification head.
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, num_classes, kernel_size=1),
        )

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        """Forward pass through the ASPP decoder.

        Args:
            features: Ordered sequence of feature tensors from the backbone,
                sorted by ascending stride (finest to coarsest).  Each
                tensor has shape ``(B, C_i, H_i, W_i)``.

        Returns:
            Segmentation logits of shape ``(B, num_classes, H, W)``
            where ``H`` and ``W`` are approximately 2x the finest input
            feature's spatial dimensions (or 4x if a single decoder stage
            suffices).

        Raises:
            ValueError: If the number of features does not match the number
                of levels in ``feature_info``.
        """
        if len(features) != len(self._sorted_levels):
            msg = (
                f"Expected {len(self._sorted_levels)} feature maps, "
                f"got {len(features)}."
            )
            raise ValueError(msg)

        # ------------------------------------------------------------------
        # Step 1: Project all encoder features.
        # ------------------------------------------------------------------
        projected: OrderedDict[str, Tensor] = OrderedDict()
        for lvl, feat in zip(
            self._sorted_levels, features, strict=True
        ):
            projected[lvl] = self.input_projections[lvl](feat)

        # ------------------------------------------------------------------
        # Step 2: ASPP on the coarsest level.
        # ------------------------------------------------------------------
        coarsest_lvl = self._sorted_levels[-1]
        x = self.aspp(projected[coarsest_lvl])

        # ------------------------------------------------------------------
        # Step 3: Decoder stage 1 -- upsample to second-finest resolution
        # with low-level skip connection.
        # ------------------------------------------------------------------
        second_finest_lvl = self._sorted_levels[1]
        x = self.decoder_stage1(x, projected[second_finest_lvl])

        # ------------------------------------------------------------------
        # Step 4: Decoder stage 2 -- upsample to finest resolution.
        # ------------------------------------------------------------------
        finest_lvl = self._sorted_levels[0]
        x = self.decoder_stage2(x, projected[finest_lvl])

        # ------------------------------------------------------------------
        # Step 5: Classification head.
        # ------------------------------------------------------------------
        return self.classifier(x)

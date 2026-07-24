"""ResUNetDecoder: U-Net style segmentation decoder with residual blocks.

Implements a decoder that consumes multi-scale feature maps from a backbone
(or neck) and progressively upsamples them using transposed convolutions,
fusing encoder features via skip connections at matching spatial resolutions.

The decoder dynamically adapts its input projection layers based on
:class:`~corecv.core.contract.FeatureInfo` metadata, making it compatible
with any backbone that conforms to the
:class:`~corecv.core.contract.BaseBackbone` contract (ResNet, MobileNetV3,
ConvNeXt, ViT with adapter, etc.).

Architecture overview::

    Encoder features (sorted finest -> coarsest):
        f_stride4  (B, C1, H/4,  W/4)   -- skip connection
        f_stride8  (B, C2, H/8,  W/8)   -- skip connection
        f_stride16 (B, C3, H/16, W/16)  -- skip connection
        f_stride32 (B, C4, H/32, W/32)  -- bottleneck
            |
            v
        ASPP / Bottleneck processing
            |
            v
        Up block 32->16  + cat(skip16)  -> ResBlock -> f16'
        Up block 16->8   + cat(skip8)   -> ResBlock -> f8'
        Up block 8->4    + cat(skip4)   -> ResBlock -> f4'
            |
            v
        1x1 Conv -> (B, num_classes, H, W)

Example:
    >>> from corecv.models.backbones.resnet import ResNet50Backbone
    >>> from corecv.models.heads.segmentation.resunet_decoder import ResUNetDecoder
    >>> backbone = ResNet50Backbone(pretrained=False)
    >>> fi = backbone.feature_info
    >>> decoder = ResUNetDecoder(
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


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ConvBnReLU(nn.Module):
    """Convolution -> BatchNorm -> ReLU triple.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply conv -> BN -> ReLU.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            Output tensor of shape ``(B, C_out, H', W')``.
        """
        return self.relu(self.bn(self.conv(x)))


class _ResidualBlock(nn.Module):
    """Two-layer residual block with pre-activation style (BN-ReLU-Conv).

    Args:
        channels: Number of input and output channels (must be equal).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block1 = _ConvBnReLU(channels, channels, kernel_size=3, padding=1)
        self.block2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Output tensor of the same shape.
        """
        residual = x
        out = self.block1(x)
        out = self.block2(out)
        return self.relu(out + residual)


class _UpBlock(nn.Module):
    """Upsample block: transposed convolution followed by concatenation.

    Used in the decoder path to double the spatial resolution and fuse with
    a skip-connection feature map from the encoder.  A residual block
    refines the fused output.

    Args:
        in_channels: Channels of the feature being upsampled.
        skip_channels: Channels of the skip-connection feature map.
        out_channels: Output channels after concatenation and projection.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        # Transposed convolution for 2x spatial upsampling.
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        # 1x1 projection after concatenation to reduce channel dimension.
        concat_ch = in_channels // 2 + skip_channels
        self.proj = _ConvBnReLU(
            concat_ch, out_channels, kernel_size=1, padding=0
        )
        # Residual refinement block.
        self.res_block = _ResidualBlock(out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """Upsample, concatenate skip connection, and refine.

        Args:
            x: Feature tensor to upsample, shape ``(B, C, H, W)``.
            skip: Skip-connection tensor from the encoder,
                shape ``(B, C_skip, 2*H, 2*W)``.

        Returns:
            Refined feature tensor of shape ``(B, out_channels, 2*H, 2*W)``.
        """
        x = self.up(x)
        # Pad if spatial dimensions don't match exactly (odd input sizes).
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        x = nn.functional.pad(
            x,
            [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
        )
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        return self.res_block(x)


# ---------------------------------------------------------------------------
# Main decoder
# ---------------------------------------------------------------------------


@register_head("resunet_decoder")
class ResUNetDecoder(nn.Module):
    """U-Net style segmentation decoder with residual blocks and skip connections.

    Consumes an ordered list of multi-scale feature maps (finest to
    coarsest) and produces per-pixel class logits at the original input
    resolution.  Input projections are created dynamically from
    :class:`~corecv.core.contract.FeatureInfo` metadata.

    The decoder expects features sorted by ascending stride (finest first).
    For backbones with 4 levels (stride 4, 8, 16, 32), the full decoder
    path is used.  For backbones with 3 levels (stride 8, 16, 32), the
    stride-4 stage is skipped.

    Args:
        feature_info: Feature metadata from the backbone or neck, containing
            channel counts and strides for each feature level.
        out_channels: Number of channels in the bottleneck and decoder
            internal feature maps.
        num_classes: Number of output segmentation classes.
        dropout: Dropout probability applied to the final classification
            head (default ``0.1``).

    Raises:
        ValueError: If ``feature_info`` contains fewer than 2 feature levels.
    """

    def __init__(
        self,
        feature_info: FeatureInfo,
        out_channels: int = 256,
        num_classes: int = 21,
        dropout: float = 0.1,
    ) -> None:
        """Initialise the ResUNet decoder.

        Args:
            feature_info: Feature metadata from the backbone or neck.
            out_channels: Internal channel dimension.
            num_classes: Number of segmentation classes.
            dropout: Dropout probability in the classification head.
        """
        super().__init__()
        self.num_classes = num_classes

        # ------------------------------------------------------------------
        # Determine the sorted feature levels and their channel counts.
        # ------------------------------------------------------------------
        # Sort by stride (ascending = finest first).
        sorted_levels = sorted(
            feature_info.strides.keys(),
            key=lambda k: feature_info.strides[k],
        )
        if len(sorted_levels) < _MIN_LEVELS:
            msg = (
                f"ResUNetDecoder requires at least {_MIN_LEVELS} feature "
                f"levels, got {len(sorted_levels)}."
            )
            raise ValueError(msg)

        self._sorted_levels = sorted_levels
        in_channels_list = [
            feature_info.channels[lvl] for lvl in sorted_levels
        ]

        # ------------------------------------------------------------------
        # Input projection: map each backbone feature to common channel dim.
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
        # Bottleneck: residual blocks at the coarsest scale.
        # ------------------------------------------------------------------
        self.bottleneck = nn.Sequential(
            _ResidualBlock(out_channels),
            _ResidualBlock(out_channels),
        )

        # ------------------------------------------------------------------
        # Decoder path: up-blocks from coarsest to finest.
        # ------------------------------------------------------------------
        num_stages = len(sorted_levels) - 1
        self.decoder_stages = nn.ModuleList()

        for _i in range(num_stages):
            # _i=0: upsample from coarsest to second-coarsest.
            # _i=1: upsample from second-coarsest to third-coarsest, etc.
            # All projected features share the same channel dim.
            self.decoder_stages.append(
                _UpBlock(
                    in_channels=out_channels,
                    skip_channels=out_channels,
                    out_channels=out_channels,
                )
            )

        # ------------------------------------------------------------------
        # Final classification head: 1x1 conv + optional dropout.
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, num_classes, kernel_size=1),
        )

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        """Forward pass through the ResUNet decoder.

        Args:
            features: Ordered sequence of feature tensors from the backbone,
                sorted by ascending stride (finest to coarsest).  Each
                tensor has shape ``(B, C_i, H_i, W_i)``.

        Returns:
            Segmentation logits of shape ``(B, num_classes, H, W)``
            where ``H`` and ``W`` match the spatial dimensions of the
            finest-scale input feature.

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
        # Step 1: Project all encoder features to common channel dimension.
        # ------------------------------------------------------------------
        projected: OrderedDict[str, Tensor] = OrderedDict()
        for lvl, feat in zip(
            self._sorted_levels, features, strict=True
        ):
            projected[lvl] = self.input_projections[lvl](feat)

        # ------------------------------------------------------------------
        # Step 2: Bottleneck at the coarsest scale.
        # ------------------------------------------------------------------
        coarsest_lvl = self._sorted_levels[-1]
        x = self.bottleneck(projected[coarsest_lvl])

        # ------------------------------------------------------------------
        # Step 3: Decoder path with skip connections (coarsest -> finest).
        # ------------------------------------------------------------------
        for i, stage in enumerate(self.decoder_stages):
            skip_idx = len(self._sorted_levels) - 2 - i
            skip_lvl = self._sorted_levels[skip_idx]
            x = stage(x, projected[skip_lvl])

        # ------------------------------------------------------------------
        # Step 4: Classification head.
        # ------------------------------------------------------------------
        return self.classifier(x)

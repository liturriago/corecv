"""Segmentation heads for CoreCV.

Implements DeepLabV3+ and ResUNetDecoder heads for semantic segmentation.
Both heads dynamically consume ``FeatureInfo`` metadata from backbones
to configure multi-scale feature processing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class _ConvBnRelu(nn.Module):
    """Convolution + Batch Normalization + ReLU block.

    A lightweight building block used throughout both segmentation decoders.
    """

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
# ResUNet Decoder
# ---------------------------------------------------------------------------


class _ResUNetDecoderBlock(nn.Module):
    """Single decoder stage for the ResUNet architecture.

    Upsamples the deep feature via transposed convolution, concatenates
    with the skip connection, and applies two ``Conv-BN-ReLU`` blocks.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        """Initialize the decoder block.

        Args:
            in_channels: Number of channels from the deeper level.
            skip_channels: Number of channels in the skip connection.
            out_channels: Number of output channels after refinement.
        """
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv1 = _ConvBnRelu(in_channels // 2 + skip_channels, out_channels)
        self.conv2 = _ConvBnRelu(out_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """Run the forward pass.

        Args:
            x: Deep feature tensor of shape ``(B, C_deep, H_d, W_d)``.
            skip: Skip-connection tensor of shape ``(B, C_skip, H_s, W_s)``.

        Returns:
            Refined feature tensor of shape ``(B, C_out, H_s, W_s)``.
        """
        x = self.upsample(x)
        # Handle spatial size mismatch between upsampled and skip tensors
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class ResUNetDecoder(nn.Module):
    """ResUNet decoder for semantic segmentation.

    Takes multi-scale features ordered from highest to lowest resolution
    and progressively upsamples them with skip connections to produce
    per-pixel segmentation logits at the original feature resolution.
    """

    def __init__(self, in_channels: list[int], num_classes: int) -> None:
        """Initialize the ResUNet decoder.

        Args:
            in_channels: Channel counts for each scale level, ordered from
                highest resolution (index 0) to lowest resolution (last).
            num_classes: Number of segmentation classes.
        """
        super().__init__()
        num_levels = len(in_channels)

        # Build decoder stages from deepest to shallowest
        self.decoder_blocks = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder_blocks.append(
                _ResUNetDecoderBlock(
                    in_channels=in_channels[idx],
                    skip_channels=in_channels[idx - 1],
                    out_channels=in_channels[idx - 1],
                ),
            )

        self.segmentation_head = nn.Conv2d(in_channels[0], num_classes, kernel_size=1)

    def forward(self, features: list[Tensor]) -> Tensor:
        """Run the forward pass.

        Args:
            features: List of multi-scale feature tensors ordered from
                highest resolution (index 0) to lowest resolution (last).

        Returns:
            Segmentation logits of shape ``(B, num_classes, H, W)``
            where ``(H, W)`` matches the highest-resolution input feature.
        """
        x = features[-1]
        for block_idx, block in enumerate(self.decoder_blocks):
            skip_idx = len(features) - 2 - block_idx
            x = block(x, features[skip_idx])
        return self.segmentation_head(x)


# ---------------------------------------------------------------------------
# DeepLabV3+ Head
# ---------------------------------------------------------------------------


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module.

    Applies parallel atrous (dilated) convolutions at multiple rates
    together with image-level global pooling to capture multi-scale
    context.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: list[int],
    ) -> None:
        """Initialize the ASPP module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels for each branch.
            dilations: Dilation rates for the parallel atrous convolutions.
        """
        super().__init__()

        # 1x1 convolution branch
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Parallel 3x3 atrous convolution branches
        self.branches_atrous = nn.ModuleList()
        for dilation in dilations:
            self.branches_atrous.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
            )

        # Image-level global pooling branch
        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Projection after concatenation
        concat_channels = out_channels * (2 + len(dilations))
        self.project = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x: Tensor) -> Tensor:
        """Run the forward pass.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            ASPP output tensor of shape ``(B, C_out, H, W)``.
        """
        spatial_size = x.shape[2:]

        out_1x1 = self.branch_1x1(x)
        out_atrous = [branch(x) for branch in self.branches_atrous]

        out_pool = self.branch_pool(x)
        out_pool = F.interpolate(
            out_pool,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )

        concatenated = torch.cat([out_1x1, *out_atrous, out_pool], dim=1)
        return self.dropout(self.project(concatenated))


class DeepLabV3PlusHead(nn.Module):
    """DeepLabV3+ head for semantic segmentation.

    Combines Atrous Spatial Pyramid Pooling (ASPP) applied to the
    deepest feature level with a lightweight decoder that fuses
    high-level semantics with low-level detail via skip connections.
    """

    def __init__(
        self,
        in_channels: list[int],
        num_classes: int,
        aspp_dilations: list[int] | None = None,
    ) -> None:
        """Initialize the DeepLabV3+ head.

        Args:
            in_channels: Channel counts for each scale level, ordered from
                highest resolution (index 0) to lowest resolution (last).
            num_classes: Number of segmentation classes.
            aspp_dilations: Dilation rates for the ASPP module.  Defaults
                to ``[6, 12, 18]`` when not provided.
        """
        super().__init__()
        if aspp_dilations is None:
            aspp_dilations = [6, 12, 18]

        low_level_channels = in_channels[0]
        high_level_channels = in_channels[-1]

        # ASPP encoder on the deepest feature level
        self.aspp = _ASPPModule(
            in_channels=high_level_channels,
            out_channels=high_level_channels,
            dilations=aspp_dilations,
        )

        # Low-level feature projection (1x1 reduction)
        low_level_reduced = max(low_level_channels // 4, 48)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_reduced, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_reduced),
            nn.ReLU(inplace=True),
        )

        # Decoder refinement
        decoder_in = high_level_channels + low_level_reduced
        self.decoder = nn.Sequential(
            _ConvBnRelu(decoder_in, 256),
            _ConvBnRelu(256, 256),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        """Run the forward pass.

        Args:
            features: List of multi-scale feature tensors ordered from
                highest resolution (index 0) to lowest resolution (last).

        Returns:
            Segmentation logits of shape ``(B, num_classes, H, W)``
            where ``(H, W)`` matches the highest-resolution input feature.
        """
        low_level = features[0]
        high_level = features[-1]

        # Encode: ASPP on deepest features
        aspp_out = self.aspp(high_level)

        # Upsample ASPP output to low-level spatial resolution
        aspp_out = F.interpolate(
            aspp_out,
            size=low_level.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Project low-level features
        low_level_out = self.low_level_conv(low_level)

        # Merge ASPP and low-level features, then refine
        decoder_input = torch.cat([aspp_out, low_level_out], dim=1)
        return self.decoder(decoder_input)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    _batch_size = 2
    _num_classes = 21

    # ---- ResUNet Decoder ----
    _resunet_ch = [64, 128, 256, 512]
    _resunet_feats = [
        torch.randn(_batch_size, 64, 64, 64),
        torch.randn(_batch_size, 128, 32, 32),
        torch.randn(_batch_size, 256, 16, 16),
        torch.randn(_batch_size, 512, 8, 8),
    ]
    _resunet_head = ResUNetDecoder(in_channels=_resunet_ch, num_classes=_num_classes)
    _resunet_out = _resunet_head(_resunet_feats)
    print(f"ResUNet in_channels: {_resunet_ch}")  # noqa: T201
    print(f"ResUNet output:      {_resunet_out.shape}")  # noqa: T201

    # ---- DeepLabV3+ Head ----
    _deeplab_ch = [256, 512, 1024, 2048]
    _deeplab_feats = [
        torch.randn(_batch_size, 256, 64, 64),
        torch.randn(_batch_size, 512, 32, 32),
        torch.randn(_batch_size, 1024, 16, 16),
        torch.randn(_batch_size, 2048, 8, 8),
    ]
    _deeplab_head = DeepLabV3PlusHead(in_channels=_deeplab_ch, num_classes=_num_classes)
    _deeplab_out = _deeplab_head(_deeplab_feats)
    print(f"DeepLabV3+ in_channels: {_deeplab_ch}")  # noqa: T201
    print(f"DeepLabV3+ output:      {_deeplab_out.shape}")  # noqa: T201

"""Segmentation heads for CoreCV.

Provides decoder heads that map multi-scale backbone features to dense
per-pixel class logits:

- **DeepLabV3PlusHead**: Encoder-decoder head with an Atrous Spatial
  Pyramid Pooling (ASPP) module and low-level feature fusion.
- **ResUNetDecoder**: U-Net style decoder with skip connections.

Reference:
    Chen et al., "Encoder-Decoder with Atrous Separable Convolution for
    Semantic Image Segmentation", ECCV 2018.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

_MIN_LEVELS = 2


def _init_weights(module: nn.Module) -> None:
    """Initialize convolution and batch-norm weights."""
    for sub in module.modules():
        if isinstance(sub, nn.Conv2d):
            nn.init.kaiming_uniform_(sub.weight, nonlinearity="relu")
            if sub.bias is not None:
                nn.init.zeros_(sub.bias)
        elif isinstance(sub, nn.BatchNorm2d):
            nn.init.ones_(sub.weight)
            nn.init.zeros_(sub.bias)


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module.

    Applies parallel atrous convolutions with increasing dilation rates plus
    a global-average-pooling branch, then fuses them with a 1x1 projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        dilations: tuple[int, ...] = (1, 6, 12, 18),
    ) -> None:
        """Initialize the ASPP module.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output channels after fusion.
            dilations: Dilation rates for the parallel atrous convolutions.

        """
        super().__init__()
        self.convs = nn.ModuleList(
            [
                self._make_branch(in_channels, out_channels, dilation=dilation)
                for dilation in dilations
            ],
        )
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(
                len(dilations) * out_channels + out_channels,
                out_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        _init_weights(self)

    @staticmethod
    def _make_branch(in_channels: int, out_channels: int, dilation: int) -> nn.Sequential:
        """Build a single ASPP convolution branch."""
        if dilation == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            )
        return nn.Sequential(conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, features: Tensor) -> Tensor:
        """Apply parallel atrous convolutions and fuse the results.

        Args:
            features: Input feature tensor of shape ``(B, C, H, W)``.

        Returns:
            Fused feature tensor of shape ``(B, out_channels, H, W)``.

        """
        height, width = features.shape[2:]
        branches = [conv(features) for conv in self.convs]

        pooled = self.pool_branch(features)
        pooled = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
        branches.append(pooled)

        return self.project(torch.cat(branches, dim=1))


class DeepLabV3PlusHead(nn.Module):
    """DeepLabV3+ encoder-decoder head.

    Applies ASPP on the coarsest feature, fuses it with a projected low-level
    feature via a skip connection, and produces per-pixel class logits
    ``(B, num_classes, H, W)``. When *input_size* is provided, logits are
    upsampled to that resolution.
    """

    def __init__(
        self,
        in_channels: list[int],
        num_classes: int,
        *,
        aspp_channels: int = 256,
        low_level_channels: int = 48,
        low_level_index: int = 0,
        aspp_level: int = -1,
    ) -> None:
        """Initialize the DeepLabV3+ head.

        Args:
            in_channels: Channel dimensions of each feature level, ordered
                from finest to coarsest.
            num_classes: Number of output classes.
            aspp_channels: Output channels of the ASPP module.
            low_level_channels: Channels of the projected low-level feature.
            low_level_index: Index of the low-level feature used for the
                decoder skip connection (0 = finest).
            aspp_level: Index of the feature consumed by ASPP (defaults to
                the coarsest level).

        Raises:
            ValueError: If fewer than two levels are provided or if
                *aspp_level* and *low_level_index* select the same level.

        """
        super().__init__()
        if len(in_channels) < _MIN_LEVELS:
            msg = (
                f"DeepLabV3PlusHead requires at least {_MIN_LEVELS} levels, "
                f"got {len(in_channels)}"
            )
            raise ValueError(msg)

        aspp_level = aspp_level % len(in_channels)
        if aspp_level == low_level_index:
            msg = "aspp_level and low_level_index must select different levels"
            raise ValueError(msg)

        self.aspp_level = aspp_level
        self.low_level_index = low_level_index
        self.aspp = _ASPP(in_channels[aspp_level], aspp_channels)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(in_channels[low_level_index], low_level_channels, kernel_size=1),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )
        self.last_conv = nn.Sequential(
            nn.Conv2d(aspp_channels + low_level_channels, aspp_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_channels, aspp_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(aspp_channels, num_classes, kernel_size=1)
        _init_weights(self)

    def forward(self, features: list[Tensor], input_size: tuple[int, int] | None = None) -> Tensor:
        """Run the DeepLabV3+ decoder on multi-scale features.

        Args:
            features: List of multi-scale feature tensors ordered from finest
                to coarsest.
            input_size: Optional ``(H, W)`` target size for the output logits.

        Returns:
            Per-pixel class logits of shape ``(B, num_classes, H, W)``.

        """
        coarse = features[self.aspp_level]
        low_level = features[self.low_level_index]

        x = self.aspp(coarse)
        x = F.interpolate(x, size=low_level.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, self.low_level_conv(low_level)], dim=1)
        x = self.last_conv(x)

        logits = self.classifier(x)
        if input_size is not None:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


class ResUNetDecoder(nn.Module):
    """U-Net style decoder head with skip connections.

    Progressively upsamples the coarsest feature and fuses it with encoder
    skip connections at every finer level, then produces per-pixel class
    logits ``(B, num_classes, H, W)``. When *input_size* is provided, logits
    are upsampled to that resolution.
    """

    def __init__(
        self,
        in_channels: list[int],
        num_classes: int,
        decoder_channels: int = 256,
    ) -> None:
        """Initialize the ResUNet decoder.

        Args:
            in_channels: Channel dimensions of each feature level, ordered
                from finest to coarsest.
            num_classes: Number of output classes.
            decoder_channels: Channel dimension of the decoder stages.

        Raises:
            ValueError: If fewer than two levels are provided.

        """
        super().__init__()
        if len(in_channels) < _MIN_LEVELS:
            msg = f"ResUNetDecoder requires at least {_MIN_LEVELS} levels, got {len(in_channels)}"
            raise ValueError(msg)

        # Skip connections are consumed from second-coarsest to finest.
        skip_channels = in_channels[-2::-1]
        block_inputs = [in_channels[-1] + skip_channels[0]]
        block_inputs.extend(decoder_channels + c for c in skip_channels[1:])

        self.blocks = nn.ModuleList(
            [self._make_block(in_c, decoder_channels) for in_c in block_inputs],
        )
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        _init_weights(self)

    @staticmethod
    def _make_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Build a double 3x3 convolution decoder block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[Tensor], input_size: tuple[int, int] | None = None) -> Tensor:
        """Run the U-Net decoder on multi-scale features.

        Args:
            features: List of multi-scale feature tensors ordered from finest
                to coarsest.
            input_size: Optional ``(H, W)`` target size for the output logits.

        Returns:
            Per-pixel class logits of shape ``(B, num_classes, H, W)``.

        """
        x = features[-1]
        skips = features[-2::-1]

        for block, skip in zip(self.blocks, skips, strict=True):
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        logits = self.classifier(x)
        if input_size is not None:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits

"""Path Aggregation Network (PANet) neck for CoreCV.

Extends FPN with an additional bottom-up path augmentation for enhanced
feature aggregation across scales. Consumes multi-scale features from
backbones and produces enriched feature maps.

Reference:
    Liu et al., "Path Aggregation Network for Instance Segmentation", CVPR 2018.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PANet(nn.Module):
    """Path Aggregation Network with bidirectional feature fusion.

    Augments the standard FPN top-down pathway with an additional bottom-up
    path that propagates fine-grained spatial information from lower levels
    to higher levels. Each output level is processed by a 3x3 convolution.

    Architecture::

        Input:  C1 -----> C2 -----> C3 -----> C4
                  |         |         |         |
                  v         v         v         v
                1x1       1x1       1x1       1x1    (lateral convs)
                  |         |         |         |
                  +<--------+<--------+         |    (top-down: upsample + add)
                  |         |         |         |
                  v         v         v         v
                3x3       3x3       3x3       3x3    (output convs)
                  |         |         |         |
                  +-------->+-------->+-------->+    (bottom-down: downsample + add)
                  |         |         |         |
                  v         v         v         v
                3x3       3x3       3x3       3x3    (final convs)
                  |         |         |         |
                P1        P2        P3        P4

    Example:
        >>> import torch
        >>> from corecv.models.necks.panet import PANet
        >>> neck = PANet(in_channels=[64, 128, 256, 512], out_channels=256)
        >>> feats = [torch.randn(2, c, h, w) for c, h, w in
        ...          [(64, 64, 64), (128, 32, 32), (256, 16, 16), (512, 8, 8)]]
        >>> outputs = neck(feats)
        >>> [o.shape for o in outputs]
        [torch.Size([2, 256, 64, 64]), torch.Size([2, 256, 32, 32]),
         torch.Size([2, 256, 16, 16]), torch.Size([2, 256, 8, 8])]
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 256,
    ) -> None:
        """Initialize the Path Aggregation Network.

        Args:
            in_channels: Channel dimensions of each input feature level,
                ordered from finest (highest resolution) to coarsest.
            out_channels: Number of channels in each output feature map.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_levels = len(in_channels)

        # Lateral 1x1 convolutions: project backbone features to out_channels
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels],
        )

        # Top-down output 3x3 convolutions (after top-down fusion)
        self.topdown_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in range(num_levels)
            ],
        )

        # Bottom-up 3x3 convolutions (after bottom-up fusion)
        self.bottomup_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in range(num_levels)
            ],
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolution weights with Kaiming uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Run bidirectional feature aggregation on multi-scale features.

        Args:
            features: List of multi-scale feature tensors ordered from finest
                (highest spatial resolution) to coarsest, each with shape
                ``(B, C_i, H_i, W_i)``.

        Returns:
            List of feature tensors with unified channel dimension
            ``(B, out_channels, H_i, W_i)``, same order as input.
        """
        num_levels = len(features)

        # Step 1: Apply lateral 1x1 convolutions
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features, strict=True)]

        # Step 2: Top-down pathway (coarse -> fine)
        for idx in range(num_levels - 1, 0, -1):
            upsampled = nn.functional.interpolate(
                laterals[idx],
                size=laterals[idx - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[idx - 1] = laterals[idx - 1] + upsampled

        # Step 3: Apply top-down output 3x3 convolutions
        topdown_outs = [conv(lat) for conv, lat in zip(self.topdown_convs, laterals, strict=True)]

        # Step 4: Bottom-up pathway (fine -> coarse)
        bottomup = [topdown_outs[0]]
        for idx in range(1, num_levels):
            # Downsample the finer level to match the spatial size of the coarser level
            downsampled = nn.functional.max_pool2d(
                bottomup[idx - 1],
                kernel_size=2,
                stride=2,
            )
            # Ensure spatial alignment (handles odd-sized feature maps)
            if downsampled.shape[2:] != topdown_outs[idx].shape[2:]:
                downsampled = nn.functional.interpolate(
                    downsampled,
                    size=topdown_outs[idx].shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            merged = topdown_outs[idx] + downsampled
            bottomup.append(merged)

        # Step 5: Apply bottom-up output 3x3 convolutions
        return [conv(feat) for conv, feat in zip(self.bottomup_convs, bottomup, strict=True)]


if __name__ == "__main__":
    torch.manual_seed(42)

    # Simulate ResNet-style backbone features: C2=64, C3=128, C4=256, C5=512
    batch_size = 2
    in_channels_list = [64, 128, 256, 512]
    out_channels = 256
    spatial_sizes = [(64, 64), (32, 32), (16, 16), (8, 8)]

    dummy_features = [
        torch.randn(batch_size, c, h, w)
        for c, (h, w) in zip(in_channels_list, spatial_sizes, strict=True)
    ]

    panet = PANet(in_channels=in_channels_list, out_channels=out_channels)
    panet.eval()

    with torch.no_grad():
        outputs = panet(dummy_features)

    print("--- PANet Forward Pass ---")  # noqa: T201
    for i, (inp, out) in enumerate(zip(dummy_features, outputs, strict=True)):
        print(  # noqa: T201
            f"  Level {i}: input {inp.shape} -> output {out.shape}",
        )

    # Verify output channels
    assert all(o.shape[1] == out_channels for o in outputs), "Channel mismatch!"  # noqa: S101
    # Verify spatial sizes are preserved
    for inp, out in zip(dummy_features, outputs, strict=True):
        assert inp.shape[2:] == out.shape[2:], "Spatial size mismatch!"  # noqa: S101
    print("All assertions passed.")  # noqa: T201

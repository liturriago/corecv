"""Feature Pyramid Network (FPN) neck for CoreCV.

Implements a standard top-down FPN that consumes multi-scale features from
backbones and produces feature maps with unified channel dimensions for
detection and segmentation heads.

Reference:
    Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FPN(nn.Module):
    """Feature Pyramid Network with top-down pathway and lateral connections.

    Accepts a list of multi-scale feature maps ordered from finest to coarsest
    resolution (e.g., ``[C1, C2, C3, C4]``) and produces a list of feature
    maps with unified channel dimensions (``[P1, P2, P3, P4]``).

    The top-down pathway merges coarser features into finer ones via
    element-wise addition after 1x1 channel projection (lateral connections),
    followed by a 3x3 convolution to reduce aliasing artifacts.

    Example:
        >>> import torch
        >>> from corecv.models.necks.fpn import FPN
        >>> neck = FPN(in_channels=[64, 128, 256, 512], out_channels=256)
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
        """Initialize the Feature Pyramid Network.

        Args:
            in_channels: Channel dimensions of each input feature level,
                ordered from finest (highest resolution) to coarsest.
            out_channels: Number of channels in each output feature map.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_levels = len(in_channels)

        # Lateral 1x1 convolutions: project each backbone feature to out_channels
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels],
        )

        # Output 3x3 convolutions: reduce aliasing after top-down merge
        self.output_convs = nn.ModuleList(
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
        """Run the top-down feature pyramid on multi-scale backbone features.

        Args:
            features: List of multi-scale feature tensors ordered from finest
                (highest spatial resolution) to coarsest, each with shape
                ``(B, C_i, H_i, W_i)``.

        Returns:
            List of feature tensors with unified channel dimension
            ``(B, out_channels, H_i, W_i)``, same order as input.
        """
        # Step 1: Apply lateral 1x1 convolutions to all levels
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features, strict=True)]

        # Step 2: Top-down pathway — upsample coarser features and merge
        for idx in range(len(laterals) - 1, 0, -1):
            # Upsample the coarser level to match the spatial size of the finer level
            upsampled = nn.functional.interpolate(
                laterals[idx],
                size=laterals[idx - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[idx - 1] = laterals[idx - 1] + upsampled

        # Step 3: Apply output 3x3 convolutions to all merged levels
        return [conv(lat) for conv, lat in zip(self.output_convs, laterals, strict=True)]


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

    fpn = FPN(in_channels=in_channels_list, out_channels=out_channels)
    fpn.eval()

    with torch.no_grad():
        outputs = fpn(dummy_features)

    print("--- FPN Forward Pass ---")  # noqa: T201
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

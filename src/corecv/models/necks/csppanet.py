"""Bidirectional pyramid neck with cross-stage partial refinement for CoreCV.

Implements a feature pyramid fusion network that combines a top-down
(FPN-style) and a bottom-up (PAN-style) pathway. Multi-scale features are
merged by channel concatenation instead of addition, and each fused level is
refined with a cross-stage partial (CSP) block.

Reference:
    Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017.
    Liu et al., "Path Aggregation Network for Instance Segmentation", CVPR 2018.
    Wang et al., "CSPNet: A New Backbone that can Enhance Learning Capability
    of CNN", CVPR 2020.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from corecv.models.backbones.csp_pyramid import ConvBlock, CSPBlock

# Default number of CSP refinement repeats per fused level.
_DEFAULT_REPEATS = 2

# Channel expansion factor of the CSP refinement blocks.
_CSP_EXPANSION = 0.5


class CSPPANet(nn.Module):
    """Bidirectional pyramid network with concatenation and CSP refinement.

    Accepts a list of multi-scale feature maps ordered from finest to coarsest
    resolution (e.g., ``[C3, C4, C5]``) and produces a list of feature maps
    with unified channel dimensions (``[P3, P4, P5]``).

    The **top-down** pathway up-samples coarser features by nearest neighbor
    and concatenates them with the finer levels, enriching high-resolution
    maps with deep semantic context. The **bottom-up** pathway down-samples
    fine features with a strided convolution and concatenates them with the
    top-down outputs, restoring precise localization in coarse maps. Each
    concatenation is fused with a CSP block.

    Example:
        >>> import torch
        >>> from corecv.models.necks.csppanet import CSPPANet
        >>> neck = CSPPANet(in_channels=[64, 128, 256], out_channels=128)
        >>> feats = [torch.randn(2, c, h, w) for c, h, w in
        ...          [(64, 32, 32), (128, 16, 16), (256, 8, 8)]]
        >>> outputs = neck(feats)
        >>> [o.shape[1] for o in outputs]
        [128, 128, 128]
        >>> outputs[0].shape
        torch.Size([2, 128, 32, 32])

    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 256,
        repeats: int = _DEFAULT_REPEATS,
        *,
        use_c3k: bool = False,
    ) -> None:
        """Initialize the bidirectional pyramid network.

        Args:
            in_channels: Channel dimensions of each input feature level,
                ordered from finest (highest resolution) to coarsest.
            out_channels: Number of channels in all output feature maps.
            repeats: Number of CSP refinement repeats per fused level.
            use_c3k: Whether the CSP refinement blocks use the
                three-convolution bottleneck instead of the two-convolution
                bottleneck.

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.repeats = repeats
        num_levels = len(in_channels)

        # Lateral 1x1 convolutions: project each backbone feature to out_channels.
        self.lateral_convs = nn.ModuleList(
            [ConvBlock(in_ch, out_channels, kernel_size=1) for in_ch in in_channels],
        )

        # Top-down refinement blocks (one per non-coarsest level).
        self.td_blocks = nn.ModuleList(
            [
                CSPBlock(
                    out_channels * 2,
                    out_channels,
                    repeats=repeats,
                    expansion=_CSP_EXPANSION,
                    use_c3k=use_c3k,
                )
                for _ in range(num_levels - 1)
            ],
        )

        # Bottom-up downsampling and refinement blocks (one per non-finest level).
        self.down_convs = nn.ModuleList(
            [
                ConvBlock(out_channels, out_channels, kernel_size=3, stride=2)
                for _ in range(num_levels - 1)
            ],
        )
        self.bu_blocks = nn.ModuleList(
            [
                CSPBlock(
                    out_channels * 2,
                    out_channels,
                    repeats=repeats,
                    expansion=_CSP_EXPANSION,
                    use_c3k=use_c3k,
                )
                for _ in range(num_levels - 1)
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
        """Run bidirectional feature fusion on multi-scale backbone features.

        Args:
            features: List of multi-scale feature tensors ordered from finest
                (highest spatial resolution) to coarsest, each with shape
                ``(B, C_i, H_i, W_i)``.

        Returns:
            List of feature tensors with unified channel dimension
            ``(B, out_channels, H_i, W_i)``, same order as input.

        """
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features, strict=True)]
        num_levels = len(laterals)

        # Top-down pathway: fuse the coarsest feature into finer levels.
        td: list[Tensor] = [laterals[-1]]
        for idx in range(num_levels - 2, -1, -1):
            upsampled = nn.functional.interpolate(
                td[-1],
                size=laterals[idx].shape[2:],
                mode="nearest",
            )
            fused = torch.cat([laterals[idx], upsampled], dim=1)
            td.append(self.td_blocks[idx](fused))
        td.reverse()

        # Bottom-up pathway: propagate fine localization into coarser levels.
        bu: list[Tensor] = [td[0]]
        for idx in range(1, num_levels):
            downsampled = self.down_convs[idx - 1](bu[-1])
            if downsampled.shape[2:] != td[idx].shape[2:]:
                downsampled = nn.functional.interpolate(
                    downsampled,
                    size=td[idx].shape[2:],
                    mode="nearest",
                )
            fused = torch.cat([td[idx], downsampled], dim=1)
            bu.append(self.bu_blocks[idx - 1](fused))

        return bu


if __name__ == "__main__":
    torch.manual_seed(42)

    # Simulate pyramid backbone features: P3=64, P4=128, P5=256
    batch_size = 2
    in_channels_list = [64, 128, 256]
    out_channels = 128
    spatial_sizes = [(32, 32), (16, 16), (8, 8)]

    dummy_features = [
        torch.randn(batch_size, c, h, w)
        for c, (h, w) in zip(in_channels_list, spatial_sizes, strict=True)
    ]

    neck = CSPPANet(in_channels=in_channels_list, out_channels=out_channels, repeats=2)
    neck.eval()

    with torch.no_grad():
        outputs = neck(dummy_features)

    print("--- CSPPANet Forward Pass ---")  # noqa: T201
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

"""Bidirectional Feature Pyramid Network (BiFPN) neck for CoreCV.

Implements weighted bidirectional feature fusion with fast normalized scaling.
Consumes multi-scale features from backbones and produces enriched feature
maps with learnable fusion weights.

Reference:
    Tan et al., "EfficientDet: Scalable and Efficient Object Detection", CVPR 2020.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FastNormalizedFusion(nn.Module):
    """Fast normalized feature fusion with learnable weights.

    Applies weighted sum of input features using fast normalized weights::

        out = sum(w_i * feat_i) / (sum(w_i) + eps)

    where weights are clamped to ``[0, inf)`` and normalized by their sum.
    This is more efficient than softmax-based fusion while maintaining
    training stability.
    """

    def __init__(self, num_inputs: int, eps: float = 1e-4) -> None:
        """Initialize fast normalized fusion.

        Args:
            num_inputs: Number of input features to fuse.
            eps: Small constant for numerical stability in normalization.
        """
        super().__init__()
        self.eps = eps
        # Learnable fusion weights — initialized to 1.0
        self.weight = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))

    def forward(self, inputs: list[Tensor]) -> Tensor:
        """Fuse multiple input tensors with learnable normalized weights.

        All inputs must have the same spatial dimensions. If they differ,
        the first input's spatial size is used as the reference and others
        are resized via bilinear interpolation.

        Args:
            inputs: List of tensors with shape ``(B, C, H, W)``.

        Returns:
            Fused tensor with shape ``(B, C, H_ref, W_ref)``.
        """
        # Ensure all inputs have the same spatial size
        ref_size = inputs[0].shape[2:]
        aligned: list[Tensor] = []
        for feat in inputs:
            resized = feat
            if feat.shape[2:] != ref_size:
                resized = nn.functional.interpolate(
                    feat,
                    size=ref_size,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(resized)

        # Fast normalized fusion: w_i * x_i / (sum(w_j) + eps)
        w = torch.clamp(self.weight, min=0.0)
        w_sum = w.sum() + self.eps
        stacked = torch.stack(aligned, dim=0)  # (num_inputs, B, C, H, W)
        # Reshape weights for broadcasting: (num_inputs, 1, 1, 1, 1)
        w_broadcast = (w / w_sum).view(-1, 1, 1, 1, 1)
        return (stacked * w_broadcast).sum(dim=0)


class BiFPNBlock(nn.Module):
    """Single BiFPN block with bidirectional feature fusion.

    Implements one iteration of the BiFPN fusion pattern:

    - **Top-down path** (level ``n-2`` to ``0``): Fuses current and upsampled
      coarser features via ``FastNormalizedFusion``.
    - **Bottom-up path** (level ``1`` to ``n-1``): Fuses current, downsampled
      finer, and original input features via ``FastNormalizedFusion``.
    - Each fused feature is passed through a depthwise-separable 3x3 convolution.

    Note:
        The intermediate features at each level use ``out_channels``.
    """

    def __init__(
        self,
        num_levels: int,
        out_channels: int,
    ) -> None:
        """Initialize a single BiFPN block.

        Args:
            num_levels: Number of input/output feature levels.
            out_channels: Channel dimension for all intermediate and output
                feature maps.
        """
        super().__init__()
        self.num_levels = num_levels
        self.out_channels = out_channels

        # Top-down fusion nodes: each node fuses 2 inputs (current + upsampled coarser)
        # Level 0 in top-down has no coarser input, so it's skipped.
        self.td_fusions = nn.ModuleList(
            [FastNormalizedFusion(num_inputs=2) for _ in range(num_levels - 1)],
        )

        # Bottom-up fusion nodes: each node fuses 3 inputs (current + downsampled finer + original)
        self.bu_fusions = nn.ModuleList(
            [FastNormalizedFusion(num_inputs=3) for _ in range(num_levels - 1)],
        )

        # Depthwise-separable 3x3 convolutions for each level in both paths
        self.td_convs = nn.ModuleList(
            [self._make_depthwise_sep_conv(out_channels) for _ in range(num_levels - 1)],
        )
        self.bu_convs = nn.ModuleList(
            [self._make_depthwise_sep_conv(out_channels) for _ in range(num_levels - 1)],
        )

    @staticmethod
    def _make_depthwise_sep_conv(channels: int) -> nn.Sequential:
        """Create a depthwise-separable 3x3 convolution block.

        Args:
            channels: Number of input and output channels.

        Returns:
            Sequential block with depthwise 3x3 conv + batch norm + ReLU,
            followed by pointwise 1x1 conv + batch norm + ReLU.
        """
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Run one BiFPN bidirectional fusion pass.

        Args:
            features: List of ``num_levels`` feature tensors, each with shape
                ``(B, out_channels, H_i, W_i)``.

        Returns:
            List of ``num_levels`` enhanced feature tensors with the same
            shapes as the input.
        """
        n = self.num_levels

        # ---- Top-down path (coarsest -> finest) ----
        # Level n-1 (coarsest) stays unchanged in the top-down path
        td: list[Tensor] = [features[-1]]
        for i in range(n - 2, -1, -1):
            # Upsample the previous (coarser) top-down feature to current level's size
            upsampled = nn.functional.interpolate(
                td[-1],
                size=features[i].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            # Fuse: current backbone feature + upsampled coarser feature
            idx = n - 2 - i  # index into td_fusions / td_convs (0 for finest)
            fused = self.td_fusions[idx]([features[i], upsampled])
            td.append(self.td_convs[idx](fused))

        # Reverse so td[0] = finest, td[n-1] = coarsest
        td.reverse()

        # ---- Bottom-up path (finest -> coarsest) ----
        # Level 0 (finest) stays unchanged in the bottom-up path
        bu: list[Tensor] = [td[0]]
        for i in range(1, n):
            # Downsample the previous (finer) bottom-up feature to current level's size
            downsampled = nn.functional.max_pool2d(bu[-1], kernel_size=2, stride=2)
            if downsampled.shape[2:] != td[i].shape[2:]:
                downsampled = nn.functional.interpolate(
                    downsampled,
                    size=td[i].shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            # Fuse: current top-down feature + downsampled finer + original backbone feature
            fused = self.bu_fusions[i - 1]([td[i], downsampled, features[i]])
            bu.append(self.bu_convs[i - 1](fused))

        return bu


class BiFPN(nn.Module):
    """Bidirectional Feature Pyramid Network with weighted feature fusion.

    Stacks multiple ``BiFPNBlock`` iterations for progressively refined
    multi-scale feature fusion. Uses fast normalized fusion weights that
    are learned end-to-end during training.

    Example:
        >>> import torch
        >>> from corecv.models.necks.bifpn import BiFPN
        >>> neck = BiFPN(in_channels=[64, 128, 256, 512], out_channels=160, num_repeats=3)
        >>> feats = [torch.randn(2, c, h, w) for c, h, w in
        ...          [(64, 64, 64), (128, 32, 32), (256, 16, 16), (512, 8, 8)]]
        >>> outputs = neck(feats)
        >>> [o.shape for o in outputs]
        [torch.Size([2, 160, 64, 64]), torch.Size([2, 160, 32, 32]),
         torch.Size([2, 160, 16, 16]), torch.Size([2, 160, 8, 8])]
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 160,
        num_repeats: int = 1,
    ) -> None:
        """Initialize the Bidirectional Feature Pyramid Network.

        Args:
            in_channels: Channel dimensions of each input feature level,
                ordered from finest (highest resolution) to coarsest.
            out_channels: Number of channels in all intermediate and output
                feature maps.
            num_repeats: Number of BiFPN block repetitions for iterative
                refinement. Defaults to 1.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_repeats = num_repeats
        num_levels = len(in_channels)

        # Channel projection from backbone channels to unified out_channels
        self.input_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for in_ch in in_channels
            ],
        )

        # Stack of BiFPN blocks
        self.blocks = nn.ModuleList(
            [
                BiFPNBlock(num_levels=num_levels, out_channels=out_channels)
                for _ in range(num_repeats)
            ],
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolution weights with Kaiming uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        """Run iterative bidirectional feature fusion.

        Args:
            features: List of multi-scale feature tensors ordered from finest
                (highest spatial resolution) to coarsest, each with shape
                ``(B, C_i, H_i, W_i)``.

        Returns:
            List of feature tensors with unified channel dimension
            ``(B, out_channels, H_i, W_i)``, same order as input.
        """
        # Project all backbone features to unified channels
        projected = [conv(feat) for conv, feat in zip(self.input_convs, features, strict=True)]

        # Run through stacked BiFPN blocks
        out = projected
        for block in self.blocks:
            out = block(out)

        return out


if __name__ == "__main__":
    torch.manual_seed(42)

    # Simulate ResNet-style backbone features: C2=64, C3=128, C4=256, C5=512
    batch_size = 2
    in_channels_list = [64, 128, 256, 512]
    out_channels = 160
    spatial_sizes = [(64, 64), (32, 32), (16, 16), (8, 8)]

    dummy_features = [
        torch.randn(batch_size, c, h, w)
        for c, (h, w) in zip(in_channels_list, spatial_sizes, strict=True)
    ]

    # Test with default num_repeats=1
    bifpn_single = BiFPN(in_channels=in_channels_list, out_channels=out_channels, num_repeats=1)
    bifpn_single.eval()

    with torch.no_grad():
        outputs_single = bifpn_single(dummy_features)

    print("--- BiFPN Forward Pass (num_repeats=1) ---")  # noqa: T201
    for i, (inp, out) in enumerate(zip(dummy_features, outputs_single, strict=True)):
        print(  # noqa: T201
            f"  Level {i}: input {inp.shape} -> output {out.shape}",
        )

    # Test with num_repeats=3
    bifpn_multi = BiFPN(in_channels=in_channels_list, out_channels=out_channels, num_repeats=3)
    bifpn_multi.eval()

    with torch.no_grad():
        outputs_multi = bifpn_multi(dummy_features)

    print("\n--- BiFPN Forward Pass (num_repeats=3) ---")  # noqa: T201
    for i, (inp, out) in enumerate(zip(dummy_features, outputs_multi, strict=True)):
        print(  # noqa: T201
            f"  Level {i}: input {inp.shape} -> output {out.shape}",
        )

    # Verify output channels
    assert all(o.shape[1] == out_channels for o in outputs_single), "Channel mismatch (single)!"  # noqa: S101
    assert all(o.shape[1] == out_channels for o in outputs_multi), "Channel mismatch (multi)!"  # noqa: S101
    # Verify spatial sizes are preserved
    for inp, out in zip(dummy_features, outputs_single, strict=True):
        assert inp.shape[2:] == out.shape[2:], "Spatial size mismatch (single)!"  # noqa: S101
    for inp, out in zip(dummy_features, outputs_multi, strict=True):
        assert inp.shape[2:] == out.shape[2:], "Spatial size mismatch (multi)!"  # noqa: S101
    print("\nAll assertions passed.")  # noqa: T201

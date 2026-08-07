"""CSP pyramid backbone for CoreCV.

Provides from-scratch hierarchical pyramid backbone variants (nano, small,
medium, large, xlarge) as multi-scale feature extractors with ``FeatureInfo``
metadata.

The backbone downsamples the input five times to build three feature levels
with strides 8x, 16x, and 32x. It is composed of convolution stems,
cross-stage-partial (CSP) blocks, a fast spatial pyramid pooling module, and
a positional self-attention stage at the deepest level.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from corecv.models.backbones.base import BaseBackbone, FeatureInfo

# Type alias for supported CSP pyramid variants.
PyramidVariant = Literal[
    "csp_nano",
    "csp_small",
    "csp_medium",
    "csp_large",
    "csp_xlarge",
]

# Minimum scaled channel count after width scaling.
_MIN_CHANNELS = 8

# Multiple of 8 that scaled channel counts are rounded to.
_CHANNEL_MULTIPLE = 8

# Minimum number of block repeats per stage.
_MIN_REPEATS = 1

# Width at and above which the three-convolution bottleneck is used.
_FULL_SCALE_WIDTH = 1.0

# Fraction of output channels allocated to each CSP branch.
_BRANCH_EXPANSION = 0.5

# Number of branches a CSP module splits its channels into.
_BRANCH_COUNT = 2

# Factor used to halve channel counts and compute convolution padding.
_HALVING_FACTOR = 2

# Number of pooled feature maps concatenated by the spatial pyramid pooling.
_SPPF_OUTPUTS = 4

# Default max-pooling kernel size of the spatial pyramid pooling module.
_SPPF_POOL_KERNEL = 5

# Default number of attention heads in the positional self-attention block.
_NUM_ATTENTION_HEADS = 4

# Default key/value dimension ratio of the positional self-attention block.
_ATTENTION_RATIO = 0.5

# Expansion factor of the FFN inside the positional self-attention block.
_FFN_EXPANSION = 2

# Scaling factors per variant: (depth, width, max_channels).
_SCALES: dict[str, tuple[float, float, int]] = {
    "csp_nano": (0.50, 0.25, 1024),
    "csp_small": (0.50, 0.50, 1024),
    "csp_medium": (0.50, 1.00, 512),
    "csp_large": (1.00, 1.00, 512),
    "csp_xlarge": (1.00, 1.50, 768),
}

# Base channel layout (width 1.0) for the two stems and the three outputs.
_BASE_CHANNELS: list[int] = [64, 128, 256, 512, 1024]

# Base block repeats (depth 1.0) for the four CSP stages.
_BASE_REPEATS: list[int] = [2, 4, 4, 4]

# Base block repeats (depth 1.0) for the positional self-attention stage.
_BASE_ATTENTION_REPEATS: int = 4


def _scaled_channels(
    base: list[int],
    width: float,
    max_channels: int,
) -> list[int]:
    """Apply width scaling and a channel cap to a base channel layout.

    Args:
        base: Base channel counts at width 1.0.
        width: Width scaling factor.
        max_channels: Upper bound on the scaled channel counts.

    Returns:
        Scaled channel counts rounded to a multiple of 8.

    """
    return [
        min(
            max(round(ch * width / _CHANNEL_MULTIPLE) * _CHANNEL_MULTIPLE, _MIN_CHANNELS),
            max_channels,
        )
        for ch in base
    ]


def _scaled_repeats(base: list[int], depth: float) -> list[int]:
    """Apply depth scaling to a base repeat layout.

    Args:
        base: Base repeat counts at depth 1.0.
        depth: Depth scaling factor.

    Returns:
        Scaled repeat counts, at least one per stage.

    """
    return [max(round(r * depth), _MIN_REPEATS) for r in base]


class ConvBlock(nn.Module):
    """Convolutional block: 2D convolution followed by batch norm and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
    ) -> None:
        """Initialize the convolutional block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size of the convolution.
            stride: Stride of the convolution.

        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // _HALVING_FACTOR,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Run the convolution, batch norm, and activation.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Activated output tensor of shape ``(B, C_out, H_out, W_out)``.

        """
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Residual bottleneck of two stacked 3x3 convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expansion: float = _BRANCH_EXPANSION,
        shortcut: bool = True,
    ) -> None:
        """Initialize the residual bottleneck.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            expansion: Fraction of *out_channels* used for the hidden state.
            shortcut: Whether to add the residual connection when the input
                and output channel counts match.

        """
        super().__init__()
        hidden = round(out_channels * expansion)
        self.cv1 = ConvBlock(in_channels, hidden, kernel_size=3)
        self.cv2 = ConvBlock(hidden, out_channels, kernel_size=3)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Run the two convolutions and add the residual when enabled.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Output tensor of shape ``(B, C_out, H, W)``.

        """
        out = self.cv2(self.cv1(x))
        return x + out if self.shortcut else out


class C3kBottleneck(nn.Module):
    """Residual three-convolution bottleneck for higher capacity.

    Composes three convolutions (1x1, 3x3, 1x1) with a residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expansion: float = _BRANCH_EXPANSION,
        shortcut: bool = True,
    ) -> None:
        """Initialize the three-convolution bottleneck.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            expansion: Fraction of *out_channels* used for the hidden state.
            shortcut: Whether to add the residual connection when the input
                and output channel counts match.

        """
        super().__init__()
        hidden = round(out_channels * expansion)
        self.cv1 = ConvBlock(in_channels, hidden, kernel_size=1)
        self.cv2 = ConvBlock(hidden, hidden, kernel_size=3)
        self.cv3 = ConvBlock(hidden, out_channels, kernel_size=1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Run the three convolutions and add the residual when enabled.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Output tensor of shape ``(B, C_out, H, W)``.

        """
        out = self.cv3(self.cv2(self.cv1(x)))
        return x + out if self.shortcut else out


class SPPF(nn.Module):
    """Fast spatial pyramid pooling module.

    Pools the input at three progressively larger receptive fields by
    chaining max-pooling layers, concatenates the pooled maps with the
    original, and fuses them with a pointwise convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = _SPPF_POOL_KERNEL,
    ) -> None:
        """Initialize the spatial pyramid pooling module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size of the chained max-pooling layers.

        """
        super().__init__()
        hidden = in_channels // _HALVING_FACTOR
        self.cv1 = ConvBlock(in_channels, hidden, kernel_size=1)
        self.cv2 = ConvBlock(hidden * _SPPF_OUTPUTS, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // _HALVING_FACTOR,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pool at three scales, concatenate, and fuse.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Fused tensor of shape ``(B, out_channels, H, W)``.

        """
        y = self.cv1(x)
        y1 = self.pool(y)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([y, y1, y2, y3], dim=1))


class PSABlock(nn.Module):
    """Positional self-attention block.

    Computes multi-head self-attention over spatial positions within a
    reduced attention dimension with a depthwise-convolutional positional
    encoding, followed by a two-layer pointwise FFN. Both the attention and
    the FFN use residual shortcuts.
    """

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = _NUM_ATTENTION_HEADS,
        attn_ratio: float = _ATTENTION_RATIO,
    ) -> None:
        """Initialize the positional self-attention block.

        Args:
            channels: Number of input and output channels.
            num_heads: Number of attention heads.
            attn_ratio: Fraction of *channels* used for the key/value
                dimension.

        """
        super().__init__()
        key_dim = max(round(channels * attn_ratio / num_heads) * num_heads, num_heads)
        self.num_heads = num_heads
        self.head_dim = key_dim // num_heads
        self.key_dim = key_dim
        self.scale = self.head_dim**-0.5

        self.q = ConvBlock(channels, key_dim, kernel_size=1)
        self.k = ConvBlock(channels, key_dim, kernel_size=1)
        self.v = ConvBlock(channels, key_dim, kernel_size=1)
        self.out = ConvBlock(key_dim, channels, kernel_size=1)
        self.pos = nn.Conv2d(
            key_dim,
            key_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=key_dim,
            bias=False,
        )

        self.ffn1 = ConvBlock(channels, channels * _FFN_EXPANSION, kernel_size=1)
        self.ffn2 = nn.Conv2d(channels * _FFN_EXPANSION, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention and the FFN with residual shortcuts.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Output tensor of shape ``(B, C, H, W)``.

        """
        b, _c, h, w = x.shape
        q = (
            self.q(x)
            .view(b, self.num_heads, self.head_dim, h * w)
            .transpose(1, 2)
        )
        k = (
            self.k(x)
            .view(b, self.num_heads, self.head_dim, h * w)
            .transpose(1, 2)
        )
        v = self.v(x)
        v = v + self.pos(v)
        v = (
            v.view(b, self.num_heads, self.head_dim, h * w)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        scores = torch.softmax(scores, dim=-1)
        out = torch.matmul(scores, v)
        out = out.transpose(2, 3).reshape(b, self.key_dim, h, w)
        out = self.out(out)

        x = x + out
        return x + self.ffn2(self.ffn1(x))


class CSPBlock(nn.Module):
    """Cross-stage-partial block.

    Splits the expanded channels into two branches, passes one branch through
    a stack of bottleneck modules, concatenates both branches, and fuses
    them with a pointwise convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repeats: int = 1,
        *,
        expansion: float = _BRANCH_EXPANSION,
        use_c3k: bool = False,
    ) -> None:
        """Initialize the cross-stage-partial block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            repeats: Number of bottleneck modules in the deep branch.
            expansion: Fraction of *out_channels* used for each branch.
            use_c3k: Whether to use the three-convolution bottleneck in the
                deep branch instead of the two-convolution bottleneck.

        """
        super().__init__()
        branch = max(round(out_channels * expansion), _MIN_CHANNELS)
        mid = branch * _BRANCH_COUNT
        self.cv1 = ConvBlock(in_channels, mid, kernel_size=1)
        self.cv2 = ConvBlock(mid, out_channels, kernel_size=1)
        bottleneck = C3kBottleneck if use_c3k else Bottleneck
        self.blocks = nn.ModuleList(
            [bottleneck(branch, branch, expansion=expansion) for _ in range(repeats)],
        )

    def forward(self, x: Tensor) -> Tensor:
        """Process the two branches and fuse them.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            Output tensor of shape ``(B, out_channels, H, W)``.

        """
        y = self.cv1(x)
        short, deep = y.chunk(_BRANCH_COUNT, dim=1)
        for block in self.blocks:
            deep = block(deep)
        return self.cv2(torch.cat([short, deep], dim=1))


class C2PSA(nn.Module):
    """CSP module with positional self-attention in the deep branch.

    Splits the expanded channels into two branches, passes one branch through
    a stack of positional self-attention blocks, concatenates both branches,
    and fuses them with a pointwise convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repeats: int = 1,
        *,
        num_heads: int = _NUM_ATTENTION_HEADS,
        attn_ratio: float = _ATTENTION_RATIO,
    ) -> None:
        """Initialize the CSP attention module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            repeats: Number of self-attention blocks in the deep branch.
            num_heads: Number of attention heads per block.
            attn_ratio: Key/value dimension ratio per block.

        """
        super().__init__()
        branch = max(round(out_channels * _BRANCH_EXPANSION), _MIN_CHANNELS)
        mid = branch * _BRANCH_COUNT
        self.cv1 = ConvBlock(in_channels, mid, kernel_size=1)
        self.cv2 = ConvBlock(mid, out_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                PSABlock(branch, num_heads=num_heads, attn_ratio=attn_ratio)
                for _ in range(repeats)
            ],
        )

    def forward(self, x: Tensor) -> Tensor:
        """Process the two branches and fuse them.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            Output tensor of shape ``(B, out_channels, H, W)``.

        """
        y = self.cv1(x)
        short, deep = y.chunk(_BRANCH_COUNT, dim=1)
        for block in self.blocks:
            deep = block(deep)
        return self.cv2(torch.cat([short, deep], dim=1))


class CSPPyramidBackbone(BaseBackbone):
    """CSP pyramid multi-scale feature extractor.

    Builds a from-scratch hierarchical backbone that exposes three feature
    levels (strides 8x, 16x, 32x) from convolution stems, CSP blocks, a fast
    spatial pyramid pooling module, and a positional self-attention stage.

    Example:
        >>> import torch
        >>> from corecv.models.backbones.csp_pyramid import CSPPyramidBackbone
        >>> backbone = CSPPyramidBackbone("csp_nano")
        >>> x = torch.randn(2, 3, 224, 224)
        >>> features, info = backbone(x)
        >>> info.channels
        [64, 128, 256]

    """

    def __init__(
        self,
        variant: PyramidVariant = "csp_nano",
        *,
        pretrained: bool = False,
    ) -> None:
        """Initialize the CSP pyramid backbone.

        Args:
            variant: CSP pyramid variant name. One of ``csp_nano``,
                ``csp_small``, ``csp_medium``, ``csp_large``,
                ``csp_xlarge``.
            pretrained: If ``True``, load pretrained weights. Pretrained
                weights are not available for this backbone family.

        Raises:
            ValueError: If *variant* is not recognized or if *pretrained*
                is requested.

        """
        if variant not in _SCALES:
            msg = f"Unknown CSP pyramid variant: {variant!r}. Choose from {list(_SCALES)}"
            raise ValueError(msg)
        if pretrained:
            msg = f"pretrained weights are not available for variant {variant!r}"
            raise ValueError(msg)

        depth, width, max_channels = _SCALES[variant]
        stem1, stem2, p3, p4, p5 = _scaled_channels(_BASE_CHANNELS, width, max_channels)
        stage_repeats = _scaled_repeats(_BASE_REPEATS, depth)
        attention_repeats = max(round(_BASE_ATTENTION_REPEATS * depth), _MIN_REPEATS)
        use_c3k = width >= _FULL_SCALE_WIDTH

        feature_info = FeatureInfo(
            channels=[p3, p4, p5],
            strides=[8, 16, 32],
            names=["C3", "C4", "C5"],
        )
        super().__init__(feature_info=feature_info)

        r1, r2, r3, r4 = stage_repeats
        self.stem = nn.Sequential(
            ConvBlock(3, stem1, kernel_size=3, stride=2),
            ConvBlock(stem1, stem2, kernel_size=3, stride=2),
            CSPBlock(stem2, stem2, repeats=r1, use_c3k=use_c3k),
        )
        self.stage1 = nn.Sequential(
            ConvBlock(stem2, p3, kernel_size=3, stride=2),
            CSPBlock(p3, p3, repeats=r2, use_c3k=use_c3k),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(p3, p4, kernel_size=3, stride=2),
            CSPBlock(p4, p4, repeats=r3, use_c3k=use_c3k),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(p4, p5, kernel_size=3, stride=2),
            CSPBlock(p5, p5, repeats=r4, use_c3k=use_c3k),
            SPPF(p5, p5),
            C2PSA(p5, p5, repeats=attention_repeats),
        )

    def forward(self, x: Tensor) -> tuple[list[Tensor], FeatureInfo]:
        """Extract multi-scale features from the input tensor.

        Args:
            x: Input image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Tuple of ``(features, feature_info)`` where *features* is a
            list of three feature tensors with strides 8x, 16x, and 32x
            relative to the input, and *feature_info* contains channel and
            stride metadata.

        """
        x = self.stem(x)
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        return [p3, p4, p5], self.feature_info

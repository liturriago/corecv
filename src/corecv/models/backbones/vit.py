"""Vision Transformer (ViT) backbone with multi-scale pyramid adapter.

Implements ViT-Tiny, ViT-Small, and ViT-Base variants from scratch to
enable intermediate feature extraction.  A :class:`SimplePyramidAdapter`
converts the flat patch-token sequence output into a multi-scale feature
pyramid with strides 8, 16, and 32, making the ViT transparently comply
with the :class:`~corecv.core.contract.BaseBackbone` contract.

Architecture overview::

    Image (B, 3, 224, 224)
      |
      v
    PatchEmbedding -> (B, N, embed_dim)   N = (H/P)*(W/P) = 14*14 = 196
      |
      v
    TransformerBlock x depth
      |
      v
    LayerNorm -> (B, 1+N, embed_dim)
      |
      v  (drop CLS token)
    patch_tokens: (B, N, embed_dim)
      |
      v
    SimplePyramidAdapter -> [f8, f16, f32]
      - f8:  (B, embed_dim, 28, 28)  stride  8
      - f16: (B, embed_dim, 14, 14)  stride 16
      - f32: (B, embed_dim,  7,  7)  stride 32

Variant configurations:

.. list-table::
   :header-rows: 1

   * - Variant
     - embed_dim
     - depth
     - num_heads
     - mlp_ratio
   * - ViT-Tiny
     - 192
     - 12
     - 3
     - 4.0
   * - ViT-Small
     - 384
     - 12
     - 6
     - 4.0
   * - ViT-Base
     - 768
     - 12
     - 12
     - 4.0

Example:
    >>> from corecv.models.backbones.vit import ViTBaseBackbone
    >>> backbone = ViTBaseBackbone(pretrained=False)
    >>> backbone.feature_info.channels
    {'stride8': 768, 'stride16': 768, 'stride32': 768}
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn

from corecv.core.contract import BaseBackbone, FeatureInfo
from corecv.core.registry import register_backbone

# ---------------------------------------------------------------------------
# ViT variant configurations
# ---------------------------------------------------------------------------

_VIT_VARIANTS: dict[str, dict[str, Any]] = {
    "vit_tiny": {
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "vit_small": {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "vit_base": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
    },
}


# ---------------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------------


class _PatchEmbedding(nn.Module):
    """Convert an image into a sequence of patch embeddings.

    Uses a single strided convolution to project non-overlapping patches
    into the embedding dimension.

    Args:
        img_size: Input image spatial size (assumed square).
        patch_size: Patch side length.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension per patch.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image to patch embeddings.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Patch embeddings of shape ``(B, num_patches, embed_dim)``.
        """
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # (B, embed_dim, H/P, W/P) -> (B, embed_dim, N)
        x = x.flatten(2)
        # (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.transpose(1, 2)
        return x


class _MLP(nn.Module):
    """Feed-forward network with GELU activation.

    Args:
        in_features: Input feature dimension.
        hidden_features: Hidden layer dimension.
        drop: Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape ``(*, in_features)``.

        Returns:
            Output tensor of the same shape.
        """
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer block with multi-head self-attention.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        drop: Dropout probability.
        attn_drop: Attention dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer block.

        Args:
            x: Input tensor of shape ``(B, N, embed_dim)``.

        Returns:
            Output tensor of the same shape.
        """
        # Pre-norm self-attention with residual connection.
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # Pre-norm MLP with residual connection.
        x = x + self.mlp(self.norm2(x))
        return x


class SimplePyramidAdapter(nn.Module):
    """Convert flat ViT patch tokens into a multi-scale feature pyramid.

    Takes the 2D-reshaped patch tokens from the ViT encoder and produces
    three feature maps at strides 8, 16, and 32 using transposed convolution
    (upsample), pointwise projection (identity), and strided convolution
    (downsample).

    For a 224x224 input with patch_size=16, the spatial grid is 14x14.
    The adapter produces:
        - stride 8: 28x28 via transposed convolution (2x upsample)
        - stride 16: 14x14 via pointwise projection
        - stride 32: 7x7 via strided convolution (2x downsample)

    Args:
        in_channels: Number of input channels (ViT embed_dim).
        out_channels: Number of output channels for each pyramid level.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialise the pyramid adapter.

        Args:
            in_channels: Number of input channels (ViT embed_dim).
            out_channels: Number of output channels for each pyramid level.
        """
        super().__init__()
        self.out_channels = out_channels

        # Stride 8: upsample 14x14 -> 28x28
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Stride 16: identity projection at 14x14
        self.lateral = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Stride 32: downsample 14x14 -> 7x7
        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        grid_size: int,
    ) -> list[torch.Tensor]:
        """Reshape patch tokens to 2D and create the feature pyramid.

        Args:
            patch_tokens: Patch token sequence of shape
                ``(B, num_patches, in_channels)``.
            grid_size: Spatial side length of the patch grid
                (``num_patches = grid_size ** 2``).

        Returns:
            A list of three feature tensors at strides 8, 16, and 32.
        """
        B, _n, C = patch_tokens.shape
        # Reshape to spatial: (B, C, grid_size, grid_size)
        feat = patch_tokens.transpose(1, 2).reshape(B, C, grid_size, grid_size)

        f8 = self.up(feat)          # (B, out_c, 2*grid, 2*grid)
        f16 = self.lateral(feat)    # (B, out_c, grid, grid)
        f32 = self.down(feat)       # (B, out_c, grid//2, grid//2)

        return [f8, f16, f32]


# ---------------------------------------------------------------------------
# ViT backbone
# ---------------------------------------------------------------------------


class _ViTBackbone(BaseBackbone):
    """Shared base for ViT backbone variants.

    Implements the full Vision Transformer pipeline: patch embedding,
    positional embedding, Transformer encoder blocks, final layer norm,
    and a :class:`SimplePyramidAdapter` that produces multi-scale features.

    This class should not be instantiated directly; use the variant-specific
    subclasses.
    """

    _variant_key: str

    def __init__(  # noqa: PLR0913
        self,
        pretrained: bool = False,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        """Initialise the ViT backbone.

        Args:
            pretrained: Whether to load pretrained weights.  Currently
                unused (always ``False``) as the custom ViT implementation
                does not ship pretrained checkpoints yet.
            img_size: Input image spatial size (assumed square).
            patch_size: Patch side length.
            in_channels: Number of input image channels.
            drop: Dropout probability in Transformer blocks.
            attn_drop: Attention dropout probability.
        """
        super().__init__()
        if pretrained:
            msg = (
                "Pretrained weights are not yet available for the custom "
                "ViT backbone. Initialising with random weights."
            )
            warnings.warn(msg, stacklevel=2)
        cfg = _VIT_VARIANTS[self._variant_key]
        embed_dim: int = cfg["embed_dim"]
        depth: int = cfg["depth"]
        num_heads: int = cfg["num_heads"]
        mlp_ratio: float = cfg["mlp_ratio"]

        self.patch_embed = _PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable CLS token and positional embeddings.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer encoder blocks.
        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Multi-scale pyramid adapter (stride 8, 16, 32).
        self.adapter = SimplePyramidAdapter(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights using truncated normal and constant biases."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def feature_info(self) -> FeatureInfo:
        """Return channel and stride metadata for the adapter output levels.

        All three levels (stride 8, 16, 32) have the same channel count,
        equal to the ViT ``embed_dim``.

        Returns:
            A :class:`FeatureInfo` with keys ``stride8``, ``stride16``,
            and ``stride32``.
        """
        cfg = _VIT_VARIANTS[self._variant_key]
        ch = cfg["embed_dim"]
        return FeatureInfo(
            channels={"stride8": ch, "stride16": ch, "stride32": ch},
            strides={"stride8": 8, "stride16": 16, "stride32": 32},
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass: patch embed -> Transformer -> pyramid adapter.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            A list of three feature tensors at strides 8, 16, and 32.
        """
        B = x.shape[0]

        # Patch embedding: (B, 3, H, W) -> (B, N, embed_dim)
        x = self.patch_embed(x)

        # Prepend CLS token and add positional embedding.
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        # Transformer encoder.
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Drop CLS token, keep patch tokens.
        patch_tokens = x[:, 1:]

        # Derive spatial grid size from sequence length.
        grid_size = int(patch_tokens.shape[1] ** 0.5)

        # Create multi-scale feature pyramid.
        return self.adapter(patch_tokens, grid_size)


@register_backbone("vit_tiny")
class ViTTinyBackbone(_ViTBackbone):
    """ViT-Tiny backbone (embed_dim=192, depth=12, heads=3).

    Feature levels (via :class:`SimplePyramidAdapter`):
        - ``stride8``: 192 channels, 28x28 spatial
        - ``stride16``: 192 channels, 14x14 spatial
        - ``stride32``: 192 channels, 7x7 spatial
    """

    _variant_key = "vit_tiny"


@register_backbone("vit_small")
class ViTSmallBackbone(_ViTBackbone):
    """ViT-Small backbone (embed_dim=384, depth=12, heads=6).

    Feature levels (via :class:`SimplePyramidAdapter`):
        - ``stride8``: 384 channels, 28x28 spatial
        - ``stride16``: 384 channels, 14x14 spatial
        - ``stride32``: 384 channels, 7x7 spatial
    """

    _variant_key = "vit_small"


@register_backbone("vit_base")
class ViTBaseBackbone(_ViTBackbone):
    """ViT-Base backbone (embed_dim=768, depth=12, heads=12).

    Feature levels (via :class:`SimplePyramidAdapter`):
        - ``stride8``: 768 channels, 28x28 spatial
        - ``stride16``: 768 channels, 14x14 spatial
        - ``stride32``: 768 channels, 7x7 spatial
    """

    _variant_key = "vit_base"

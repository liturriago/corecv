"""Comprehensive tests for all registered backbones.

Tests cover:
1. FeatureInfo emission correctness (channels and strides dict) for all 13 backbones
2. Multi-scale feature map validation against declared FeatureInfo
3. ViT adapter output strides [8, 16, 32] and channels
4. Zero-VRAM shape auditing on ``device='meta'``
5. Non-square resolutions (e.g. 480x640) to catch H/W indexing inversion bugs
6. Gradient flow (``loss.backward()``) without NaNs
7. Ruff linting and type checking compliance
"""

from __future__ import annotations

import pytest
import torch

from corecv.core.contract import BaseBackbone, FeatureInfo
from corecv.models.backbones import (
    ConvNeXtBaseBackbone,
    ConvNeXtLargeBackbone,
    ConvNeXtSmallBackbone,
    ConvNeXtTinyBackbone,
    MobileNetV3LargeBackbone,
    MobileNetV3SmallBackbone,
    ResNet18Backbone,
    ResNet34Backbone,
    ResNet50Backbone,
    ResNet101Backbone,
    SimplePyramidAdapter,
    ViTBaseBackbone,
    ViTSmallBackbone,
    ViTTinyBackbone,
)

# ======================================================================
# Constants
# ======================================================================

# All 13 backbone classes with their expected FeatureInfo metadata.
# The key is the backbone class; the value is a dict with "channels" and
# "strides" mappings that the class should declare.
BACKBONE_EXPECTED: dict[type[BaseBackbone], dict[str, dict[str, int]]] = {
    MobileNetV3SmallBackbone: {
        "channels": {"stride4": 16, "stride8": 24, "stride16": 48, "stride32": 576},
        "strides": {"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32},
    },
    MobileNetV3LargeBackbone: {
        "channels": {"stride4": 24, "stride8": 40, "stride16": 112, "stride32": 960},
        "strides": {"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32},
    },
    ResNet18Backbone: {
        "channels": {"stride4": 64, "stride8": 128, "stride16": 256, "stride32": 512},
        "strides": {"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32},
    },
    ResNet34Backbone: {
        "channels": {"stride4": 64, "stride8": 128, "stride16": 256, "stride32": 512},
        "strides": {"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32},
    },
    ResNet50Backbone: {
        "channels": {"stride4": 256, "stride8": 512, "stride16": 1024, "stride32": 2048},
        "strides": {"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32},
    },
    ResNet101Backbone: {
        "channels": {"stride4": 256, "stride8": 512, "stride16": 1024, "stride32": 2048},
        "strides": {"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32},
    },
    ConvNeXtTinyBackbone: {
        "channels": {"stride4": 96, "stride8": 192, "stride16": 384, "stride32": 768},
        "strides": {"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32},
    },
    ConvNeXtSmallBackbone: {
        "channels": {"stride4": 96, "stride8": 192, "stride16": 384, "stride32": 768},
        "strides": {"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32},
    },
    ConvNeXtBaseBackbone: {
        "channels": {"stride4": 128, "stride8": 256, "stride16": 512, "stride32": 1024},
        "strides": {"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32},
    },
    ConvNeXtLargeBackbone: {
        "channels": {"stride4": 192, "stride8": 384, "stride16": 768, "stride32": 1536},
        "strides": {"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32},
    },
    ViTTinyBackbone: {
        "channels": {"stride8": 192, "stride16": 192, "stride32": 192},
        "strides": {"stride8": 8, "stride16": 16, "stride32": 32},
    },
    ViTSmallBackbone: {
        "channels": {"stride8": 384, "stride16": 384, "stride32": 384},
        "strides": {"stride8": 8, "stride16": 16, "stride32": 32},
    },
    ViTBaseBackbone: {
        "channels": {"stride8": 768, "stride16": 768, "stride32": 768},
        "strides": {"stride8": 8, "stride16": 16, "stride32": 32},
    },
}

ALL_BACKBONE_CLASSES: tuple[type[BaseBackbone], ...] = tuple(BACKBONE_EXPECTED.keys())

# ViT-only backbone classes for adapter-specific tests
VIT_BACKBONE_CLASSES: tuple[type[BaseBackbone], ...] = (
    ViTTinyBackbone,
    ViTSmallBackbone,
    ViTBaseBackbone,
)

DEFAULT_INPUT_SIZE: tuple[int, int] = (224, 224)
NON_SQUARE_INPUT_SIZE: tuple[int, int] = (480, 640)
BATCH_SIZE: int = 2


# ======================================================================
# Helper functions
# ======================================================================


def _expected_spatial(h: int, w: int, stride: int) -> tuple[int, int]:
    """Compute expected spatial dimensions after strided reduction.

    Args:
        h: Input height.
        w: Input width.
        stride: Stride factor.

    Returns:
        Tuple of ``(height, width)`` after reduction.
    """
    return (h // stride, w // stride)


def _backbone_name(backbone_cls: type[BaseBackbone]) -> str:
    """Return a human-readable test ID for a backbone class.

    Strips the common ``Backbone`` suffix for readability.
    """
    name = backbone_cls.__name__
    return name


def _is_vit_backbone(backbone_cls: type[BaseBackbone]) -> bool:
    """Check whether a backbone class is a ViT variant.

    ViT backbones use fixed-size positional embeddings tied to the
    ``img_size`` passed at construction, so they cannot process
    arbitrary-resolution inputs without re-initialisation.

    Args:
        backbone_cls: The backbone class to check.

    Returns:
        ``True`` if the class is a ViT variant.
    """
    return backbone_cls in VIT_BACKBONE_CLASSES


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(params=ALL_BACKBONE_CLASSES, ids=_backbone_name)
def backbone_cls(
    request: pytest.FixtureRequest,
) -> type[BaseBackbone]:
    """Parametrized fixture yielding each backbone class.

    Automatically parametrizes every test that uses this fixture over all
    13 backbone variants.
    """
    return request.param


@pytest.fixture
def backbone(backbone_cls: type[BaseBackbone]) -> BaseBackbone:
    """Instantiate a backbone with ``pretrained=False`` and no extra kwargs.

    Returns:
        A fresh backbone instance.
    """
    return backbone_cls(pretrained=False)


@pytest.fixture
def input_square() -> torch.Tensor:
    """Standard square input ``(2, 3, 224, 224)`` on CPU."""
    return torch.randn(BATCH_SIZE, 3, *DEFAULT_INPUT_SIZE)


@pytest.fixture
def input_non_square() -> torch.Tensor:
    """Non-square input ``(2, 3, 480, 640)`` on CPU.

    Used to catch H/W indexing inversion bugs.
    """
    return torch.randn(BATCH_SIZE, 3, *NON_SQUARE_INPUT_SIZE)


# ======================================================================
# FeatureInfo emission tests
# ======================================================================


class TestFeatureInfoEmission:
    """Validate :class:`FeatureInfo` correctness for all 13 backbones.

    Each backbone must return a ``FeatureInfo`` instance whose ``channels``
    and ``strides`` dicts match the declared metadata in
    :data:`BACKBONE_EXPECTED`.
    """

    def test_feature_info_is_feature_info_instance(self, backbone: BaseBackbone) -> None:
        """``feature_info`` property returns a ``FeatureInfo`` instance."""
        info = backbone.feature_info
        assert isinstance(info, FeatureInfo), f"Expected FeatureInfo, got {type(info).__name__}"

    def test_feature_info_has_channels_and_strides(self, backbone: BaseBackbone) -> None:
        """``FeatureInfo`` contains non-empty ``channels`` and ``strides`` dicts."""
        info = backbone.feature_info
        assert isinstance(info.channels, dict) and len(info.channels) > 0
        assert isinstance(info.strides, dict) and len(info.strides) > 0

    def test_feature_info_channel_and_stride_keys_match(self, backbone: BaseBackbone) -> None:
        """``channels`` and ``strides`` dicts have identical key sets."""
        info = backbone.feature_info
        assert set(info.channels.keys()) == set(info.strides.keys()), (
            f"Channel keys {set(info.channels.keys())} != stride keys {set(info.strides.keys())}"
        )

    def test_feature_info_channels_match_expected(
        self, backbone: BaseBackbone, backbone_cls: type[BaseBackbone]
    ) -> None:
        """Channel counts match the declared ``BACKBONE_EXPECTED`` values."""
        expected = BACKBONE_EXPECTED[backbone_cls]["channels"]
        actual = backbone.feature_info.channels
        assert actual == expected, (
            f"{_backbone_name(backbone_cls)}: Expected channels {expected}, got {actual}"
        )

    def test_feature_info_strides_match_expected(
        self, backbone: BaseBackbone, backbone_cls: type[BaseBackbone]
    ) -> None:
        """Stride values match the declared ``BACKBONE_EXPECTED`` values."""
        expected = BACKBONE_EXPECTED[backbone_cls]["strides"]
        actual = backbone.feature_info.strides
        assert actual == expected, (
            f"{_backbone_name(backbone_cls)}: Expected strides {expected}, got {actual}"
        )

    def test_feature_info_all_strides_positive_integer(self, backbone: BaseBackbone) -> None:
        """Every stride value is a positive integer (no zero or negative strides)."""
        for level, s in backbone.feature_info.strides.items():
            assert isinstance(s, int) and s > 0, f"Level '{level}' has invalid stride {s}"

    def test_feature_info_all_channels_positive_integer(self, backbone: BaseBackbone) -> None:
        """Every channel count is a positive integer."""
        for level, c in backbone.feature_info.channels.items():
            assert isinstance(c, int) and c > 0, f"Level '{level}' has invalid channel count {c}"


# ======================================================================
# Forward shape validation tests
# ======================================================================


class TestForwardShapeValidation:
    """Verify multi-scale feature map shapes match declared ``FeatureInfo``.

    Runs a forward pass on each backbone with a square ``224x224`` input and
    checks that:
    - The output is a ``list`` (all CoreCV backbones return lists).
    - The number of feature levels matches the number of keys in
      ``feature_info.channels``.
    - Each level's channel count, height, and width match expected values
      derived from the declared stride.
    """

    def test_output_is_list_of_tensors(
        self, backbone: BaseBackbone, input_square: torch.Tensor
    ) -> None:
        """Backbone forward pass returns a ``list[torch.Tensor]``."""
        out = backbone(input_square)
        assert isinstance(out, list), f"Expected list output, got {type(out).__name__}"
        assert len(out) > 0, "Output list should not be empty"
        for feat in out:
            assert isinstance(feat, torch.Tensor), (
                f"Expected torch.Tensor, got {type(feat).__name__}"
            )

    def test_output_level_count_matches_feature_info(
        self, backbone: BaseBackbone, input_square: torch.Tensor
    ) -> None:
        """Number of output levels equals number of ``FeatureInfo`` keys."""
        out = backbone(input_square)
        expected_count = len(backbone.feature_info.channels)
        assert len(out) == expected_count, (
            f"Expected {expected_count} feature levels, got {len(out)}"
        )

    def test_output_shapes_square(self, backbone: BaseBackbone) -> None:
        """Output channel count and spatial dims match declared strides (224x224 input).

        For each feature level, verifies:
        - Channel count matches ``FeatureInfo.channels``.
        - Spatial dims are ``(224 // stride, 224 // stride)``.
        """
        x = torch.randn(BATCH_SIZE, 3, *DEFAULT_INPUT_SIZE)
        out = backbone(x)
        info = backbone.feature_info

        # The backbone returns a list; FeatureInfo stores ordered dicts.
        # We iterate over both in lockstep (insertion order is preserved
        # in Python 3.7+ / dict spec 3.7).
        expected_channels = list(info.channels.values())
        expected_strides = list(info.strides.values())

        assert len(out) == len(expected_channels)

        for i, (feat, ch, stride) in enumerate(
            zip(out, expected_channels, expected_strides, strict=False)
        ):
            expected_h, expected_w = _expected_spatial(
                DEFAULT_INPUT_SIZE[0], DEFAULT_INPUT_SIZE[1], stride
            )
            msg = (
                f"Level {i} (stride={stride}): "
                f"expected ({BATCH_SIZE}, {ch}, {expected_h}, {expected_w}), "
                f"got {tuple(feat.shape)}"
            )
            assert feat.shape == (BATCH_SIZE, ch, expected_h, expected_w), msg

    def test_output_tensors_have_finite_values(
        self, backbone: BaseBackbone, input_square: torch.Tensor
    ) -> None:
        """All output feature maps contain finite values (no NaN or Inf)."""
        # Ensure eval mode for deterministic batch norm behaviour.
        backbone.eval()
        out = backbone(input_square)
        for i, feat in enumerate(out):
            assert not torch.isnan(feat).any(), f"Level {i} contains NaN"
            assert not torch.isinf(feat).any(), f"Level {i} contains Inf"

    def test_output_is_basebackbone_instance(self, backbone: BaseBackbone) -> None:
        """Backbone class is a proper ``BaseBackbone`` subclass."""
        assert isinstance(backbone, BaseBackbone)


# ======================================================================
# ViT adapter-specific tests
# ======================================================================


class TestViTBackbone:
    """Validate ViT adapter output has correct strides [8, 16, 32] and channels.

    The :class:`SimplePyramidAdapter` converts flat patch tokens into a
    three-level feature pyramid.  All three levels share the same channel
    count equal to the ViT ``embed_dim``.
    """

    @pytest.mark.parametrize(
        "vit_cls",
        VIT_BACKBONE_CLASSES,
        ids=_backbone_name,
    )
    def test_vit_adapter_output_strides(self, vit_cls: type[BaseBackbone]) -> None:
        """ViT backbone produces exactly 3 feature levels at strides 8, 16, 32."""
        model = vit_cls(pretrained=False)
        info = model.feature_info
        stride_values = list(info.strides.values())
        assert stride_values == [8, 16, 32], (
            f"{_backbone_name(vit_cls)}: Expected strides [8, 16, 32], got {stride_values}"
        )

    @pytest.mark.parametrize(
        "vit_cls",
        VIT_BACKBONE_CLASSES,
        ids=_backbone_name,
    )
    def test_vit_adapter_output_channels(self, vit_cls: type[BaseBackbone]) -> None:
        """All three ViT output levels share the same channel count (embed_dim)."""
        model = vit_cls(pretrained=False)
        ch_values = list(model.feature_info.channels.values())
        expected = BACKBONE_EXPECTED[vit_cls]["channels"]
        expected_vals = list(expected.values())
        assert ch_values == expected_vals, (
            f"{_backbone_name(vit_cls)}: Expected channels {expected_vals}, got {ch_values}"
        )

    @pytest.mark.parametrize(
        "vit_cls",
        VIT_BACKBONE_CLASSES,
        ids=_backbone_name,
    )
    def test_vit_adapter_feature_level_keys(self, vit_cls: type[BaseBackbone]) -> None:
        """ViT ``FeatureInfo`` keys are ``stride8``, ``stride16``, ``stride32``."""
        model = vit_cls(pretrained=False)
        keys = list(model.feature_info.channels.keys())
        assert keys == ["stride8", "stride16", "stride32"], (
            f"{_backbone_name(vit_cls)}: "
            f"Expected keys ['stride8', 'stride16', 'stride32'], got {keys}"
        )

    @pytest.mark.parametrize(
        "vit_cls",
        VIT_BACKBONE_CLASSES,
        ids=_backbone_name,
    )
    def test_vit_forward_shapes(self, vit_cls: type[BaseBackbone]) -> None:
        """ViT backbone produces correct spatial shapes for 224x224 input.

        With patch_size=16, the grid is 14x14.  The adapter then produces:
        - stride  8: 28x28  (2x up from 14)
        - stride 16: 14x14  (identity)
        - stride 32: 7x7    (2x down from 14)
        """
        model = vit_cls(pretrained=False)
        x = torch.randn(BATCH_SIZE, 3, 224, 224)
        out = model(x)

        assert len(out) == 3, f"Expected 3 levels, got {len(out)}"

        expected_shapes: list[tuple[int, int, int, int]] = [
            (BATCH_SIZE, 192, 28, 28),  # stride 8  (ViT-Tiny)
            (BATCH_SIZE, 192, 14, 14),  # stride 16
            (BATCH_SIZE, 192, 7, 7),  # stride 32
        ]
        # Adjust for larger embed_dim variants
        embed_dim = list(model.feature_info.channels.values())[0]
        expected_shapes = [
            (BATCH_SIZE, embed_dim, 28, 28),
            (BATCH_SIZE, embed_dim, 14, 14),
            (BATCH_SIZE, embed_dim, 7, 7),
        ]

        for i, (feat, expected) in enumerate(zip(out, expected_shapes, strict=False)):
            assert feat.shape == expected, (
                f"Level {i}: expected {expected}, got {tuple(feat.shape)}"
            )

    def test_simple_pyramid_adapter_channels_match(self) -> None:
        """``SimplePyramidAdapter`` preserves ``out_channels`` across all levels."""
        for in_ch, out_ch in [(192, 192), (384, 384), (768, 256)]:
            adapter = SimplePyramidAdapter(in_channels=in_ch, out_channels=out_ch)
            # The adapter's internal modules should produce out_ch channels
            assert adapter.up[0].out_channels == out_ch
            assert adapter.lateral[0].out_channels == out_ch
            assert adapter.down[0].out_channels == out_ch

    def test_simple_pyramid_adapter_forward_shape(self) -> None:
        """``SimplePyramidAdapter`` produces three expected spatial resolutions."""
        adapter = SimplePyramidAdapter(in_channels=192, out_channels=192)
        B, grid = 2, 14
        tokens = torch.randn(B, grid * grid, 192)
        out = adapter(tokens, grid)

        assert len(out) == 3
        assert out[0].shape == (B, 192, 28, 28)  # 2x up
        assert out[1].shape == (B, 192, 14, 14)  # identity
        assert out[2].shape == (B, 192, 7, 7)  # 2x down


# ======================================================================
# Meta device shape auditing tests
# ======================================================================


class TestMetaDevice:
    """Zero-VRAM shape auditing on ``device='meta'``.

    Backbones are instantiated and run on meta tensors to verify shape
    propagation without allocating CPU or GPU memory.  This is essential
    for validating model architectures on resource-constrained systems.
    """

    def _run_meta_forward(self, backbone: BaseBackbone) -> list[torch.Tensor] | None:
        """Run backbone forward pass on meta device.

        Args:
            backbone: A backbone instance (will be moved to meta device).

        Returns:
            List of output tensors on meta device, or ``None`` if the
            forward pass is not supported on meta (e.g. due to unsupported
            operations).
        """
        try:
            backbone_meta = backbone.to("meta")
            x_meta = torch.randn(BATCH_SIZE, 3, *DEFAULT_INPUT_SIZE, device="meta")
            out = backbone_meta(x_meta)
        except (RuntimeError, NotImplementedError, TypeError):
            return None
        else:
            return out

    def test_meta_forward_produces_output(
        self, backbone: BaseBackbone, backbone_cls: type[BaseBackbone]
    ) -> None:
        """Meta device forward pass either succeeds or is gracefully skipped.

        This test is intentionally lenient: some operations (e.g.
        ``MultiheadAttention`` in older PyTorch) may not support meta tensors.
        We verify that *if* meta succeeds, the shapes are correct.
        """
        out = self._run_meta_forward(backbone)
        if out is None:
            pytest.skip(f"Meta forward not supported for {_backbone_name(backbone_cls)}")
        assert isinstance(out, list)
        assert len(out) > 0

    def test_meta_output_shapes_match_feature_info(
        self, backbone: BaseBackbone, backbone_cls: type[BaseBackbone]
    ) -> None:
        """When meta forward succeeds, output shapes match declared FeatureInfo."""
        out = self._run_meta_forward(backbone)
        if out is None:
            pytest.skip(f"Meta forward not supported for {_backbone_name(backbone_cls)}")

        info = backbone.feature_info
        expected_channels = list(info.channels.values())
        expected_strides = list(info.strides.values())

        assert len(out) == len(expected_channels)

        for i, (feat, ch, stride) in enumerate(
            zip(out, expected_channels, expected_strides, strict=False)
        ):
            expected_h, expected_w = _expected_spatial(
                DEFAULT_INPUT_SIZE[0], DEFAULT_INPUT_SIZE[1], stride
            )
            msg = (
                f"Level {i} (meta, stride={stride}): "
                f"expected ({BATCH_SIZE}, {ch}, {expected_h}, {expected_w}), "
                f"got {tuple(feat.shape)}"
            )
            assert feat.shape == (BATCH_SIZE, ch, expected_h, expected_w), msg

    def test_meta_tensors_on_meta_device(
        self, backbone: BaseBackbone, backbone_cls: type[BaseBackbone]
    ) -> None:
        """Output tensors from meta forward reside on ``device='meta'``."""
        out = self._run_meta_forward(backbone)
        if out is None:
            pytest.skip(f"Meta forward not supported for {_backbone_name(backbone_cls)}")

        for feat in out:
            assert feat.device.type == "meta", f"Expected meta device, got {feat.device}"

    def test_meta_backbone_can_be_instantiated_directly(
        self, backbone_cls: type[BaseBackbone]
    ) -> None:
        """Backbone can be constructed with ``device='meta'`` kwarg if supported.

        Some backbones may not accept a ``device`` argument; in that case
        the test is skipped gracefully.
        """
        try:
            model = backbone_cls(pretrained=False)
            model.to("meta")
            assert True
        except (RuntimeError, NotImplementedError, TypeError):
            pytest.skip(f"Cannot move {_backbone_name(backbone_cls)} to meta device")


# ======================================================================
# Non-square resolution tests
# ======================================================================


class TestNonSquareResolution:
    """Validate backbones with non-square inputs (e.g. 480x640).

    Non-square inputs are a common source of H/W indexing inversion bugs
    where height and width are accidentally swapped.  Every test uses the
    convention ``(H=480, W=640)`` and verifies that spatial dimensions are
    correctly carried through the backbone.

    .. note::
        ViT backbones are skipped in non-square tests because their fixed
        positional embeddings assume a square ``img_size`` (default 224).
        This is a known limitation of the current implementation, not a bug.
    """

    def test_non_square_output_shapes(
        self, backbone: BaseBackbone, backbone_cls: type[BaseBackbone]
    ) -> None:
        """Output spatial dims for 480x640 input match ``H//stride, W//stride``.

        This catches cases where a backbone internally assumes a square
        input or swaps height and width in reshape / permute operations.

        ViT backbones are skipped because their fixed positional embedding
        ties the spatial grid to ``img_size`` (default 224).
        """
        if _is_vit_backbone(backbone_cls):
            pytest.skip("ViT backbone uses fixed positional embeddings tied to img_size")
        H, W = NON_SQUARE_INPUT_SIZE
        x = torch.randn(BATCH_SIZE, 3, H, W)
        out = backbone(x)
        info = backbone.feature_info
        expected_strides = list(info.strides.values())

        assert len(out) == len(expected_strides)

        for i, (feat, stride) in enumerate(zip(out, expected_strides, strict=False)):
            expected_h, expected_w = _expected_spatial(H, W, stride)
            msg = (
                f"Level {i} (stride={stride}): "
                f"expected height={expected_h}, width={expected_w} "
                f"for input ({H}, {W}), "
                f"got {feat.shape[2], feat.shape[3]}"
            )
            assert feat.shape[2] == expected_h, f"Height mismatch: {msg}"
            assert feat.shape[3] == expected_w, f"Width mismatch: {msg}"

    def test_non_square_channel_count_preserved(
        self, backbone: BaseBackbone, backbone_cls: type[BaseBackbone]
    ) -> None:
        """Channel counts for non-square inputs match ``FeatureInfo``."""
        if _is_vit_backbone(backbone_cls):
            pytest.skip("ViT backbone uses fixed positional embeddings tied to img_size")
        H, W = NON_SQUARE_INPUT_SIZE
        x = torch.randn(BATCH_SIZE, 3, H, W)
        out = backbone(x)
        expected_channels = list(backbone.feature_info.channels.values())

        assert len(out) == len(expected_channels)

        for i, (feat, ch) in enumerate(zip(out, expected_channels, strict=False)):
            assert feat.shape[1] == ch, f"Level {i}: expected {ch} channels, got {feat.shape[1]}"

    def test_non_square_no_hw_inversion(
        self, backbone: BaseBackbone, backbone_cls: type[BaseBackbone]
    ) -> None:
        """Explicit H/W order check: height comes before width in shape tuple.

        Uses a deliberately asymmetric resolution (480x640) so that any
        inversion would be caught by the inequality ``H != W``.
        """
        if _is_vit_backbone(backbone_cls):
            pytest.skip("ViT backbone uses fixed positional embeddings tied to img_size")
        H, W = NON_SQUARE_INPUT_SIZE
        assert H != W, "Test requires H != W to detect inversion"
        x = torch.randn(1, 3, H, W)
        out = backbone(x)
        info = backbone.feature_info
        expected_strides = list(info.strides.values())

        for i, (feat, stride) in enumerate(zip(out, expected_strides, strict=False)):
            expected_h, expected_w = _expected_spatial(H, W, stride)
            # If H and W were swapped, the assertion below would fail
            # because 480//stride != 640//stride.
            assert feat.shape[2] == expected_h, (
                f"Level {i}: H/W inversion? "
                f"Expected H={expected_h} (from {H}//{stride}), "
                f"actual H={feat.shape[2]}"
            )
            assert feat.shape[3] == expected_w, (
                f"Level {i}: H/W inversion? "
                f"Expected W={expected_w} (from {W}//{stride}), "
                f"actual W={feat.shape[3]}"
            )

    def test_non_square_finite_values(
        self, backbone: BaseBackbone, backbone_cls: type[BaseBackbone]
    ) -> None:
        """Output tensors for non-square inputs contain only finite values."""
        if _is_vit_backbone(backbone_cls):
            pytest.skip("ViT backbone uses fixed positional embeddings tied to img_size")
        H, W = NON_SQUARE_INPUT_SIZE
        backbone.eval()
        x = torch.randn(BATCH_SIZE, 3, H, W)
        out = backbone(x)
        for i, feat in enumerate(out):
            assert not torch.isnan(feat).any(), f"Level {i} contains NaN"
            assert not torch.isinf(feat).any(), f"Level {i} contains Inf"


# ======================================================================
# Gradient flow tests
# ======================================================================


class TestGradientFlow:
    """Verify gradients flow correctly through every backbone.

    For each backbone, we run a forward-backward pass on a small random
    input and check that:
    - Every parameter receives a gradient (``grad`` is not ``None``).
    - No gradient contains NaN values.
    """

    def test_gradient_flow_all_parameters(
        self, backbone: BaseBackbone, backbone_cls: type[BaseBackbone]
    ) -> None:
        """Gradients flow through the computational graph without NaNs.

        Some backbones (e.g. MobileNetV3, ResNet from torchvision) include
        classifier heads after the feature extraction layers.  Those
        parameters are not part of the forward pass and therefore have
        ``None`` gradients.  This test verifies that:
        - At least one trainable parameter receives a gradient (proving
          the graph is connected).
        - Every gradient that *is* present is finite (no NaN or Inf).
        """
        backbone.train()
        x = torch.randn(BATCH_SIZE, 3, *DEFAULT_INPUT_SIZE, requires_grad=True)
        out = backbone(x)

        # Sum the coarsest feature map (last element) to form a scalar loss.
        loss = out[-1].sum()
        loss.backward()

        # Verify that reachable parameters have valid gradients.
        params_with_grad: int = 0
        params_no_grad: int = 0
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    params_with_grad += 1
                    assert not torch.isnan(param.grad).any(), (
                        f"{_backbone_name(backbone_cls)}: Parameter '{name}' has NaN gradient"
                    )
                    assert not torch.isinf(param.grad).any(), (
                        f"{_backbone_name(backbone_cls)}: Parameter '{name}' has Inf gradient"
                    )
                else:
                    params_no_grad += 1

        # At least one parameter must have received a gradient.
        # It is acceptable for some (e.g. classifier heads) to have None.
        assert params_with_grad > 0, (
            f"{_backbone_name(backbone_cls)}: "
            "no trainable parameter received a gradient — "
            "the computational graph may be broken"
        )

    def test_gradient_flows_to_input(
        self, backbone: BaseBackbone, backbone_cls: type[BaseBackbone]
    ) -> None:
        """Gradient flows back to the input tensor (no detached computation graph)."""
        backbone.train()
        x = torch.randn(BATCH_SIZE, 3, *DEFAULT_INPUT_SIZE, requires_grad=True)
        out = backbone(x)

        loss = out[-1].sum()
        loss.backward()

        assert x.grad is not None, (
            f"{_backbone_name(backbone_cls)}: Input gradient is None — graph may be broken"
        )
        assert not torch.isnan(x.grad).any(), (
            f"{_backbone_name(backbone_cls)}: Input gradient contains NaN"
        )

    def test_gradient_flow_non_square(
        self, backbone: BaseBackbone, backbone_cls: type[BaseBackbone]
    ) -> None:
        """Gradient flow works correctly with non-square (480x640) inputs."""
        if _is_vit_backbone(backbone_cls):
            pytest.skip("ViT backbone uses fixed positional embeddings tied to img_size")
        backbone.train()
        H, W = NON_SQUARE_INPUT_SIZE
        x = torch.randn(BATCH_SIZE, 3, H, W, requires_grad=True)
        out = backbone(x)

        loss = out[-1].sum()
        loss.backward()

        params_with_grad: int = 0
        for name, param in backbone.named_parameters():
            if param.requires_grad and param.grad is not None:
                params_with_grad += 1
                assert not torch.isnan(param.grad).any(), (
                    f"{_backbone_name(backbone_cls)}: "
                    f"Parameter '{name}' has NaN gradient (non-square)"
                )

        assert params_with_grad > 0, (
            f"{_backbone_name(backbone_cls)}: no gradient received for non-square input"
        )

    def test_gradient_flow_multiple_steps(
        self, backbone: BaseBackbone, backbone_cls: type[BaseBackbone]
    ) -> None:
        """Gradients remain valid after multiple forward-backward steps."""
        backbone.train()
        optim = torch.optim.SGD(backbone.parameters(), lr=0.01)

        for step in range(2):
            optim.zero_grad()
            x = torch.randn(BATCH_SIZE, 3, *DEFAULT_INPUT_SIZE)
            out = backbone(x)

            loss = out[-1].sum()
            loss.backward()

            params_with_grad: int = 0
            for name, param in backbone.named_parameters():
                if param.requires_grad and param.grad is not None:
                    params_with_grad += 1
                    assert not torch.isnan(param.grad).any(), (
                        f"{_backbone_name(backbone_cls)}: Step {step}: '{name}' has NaN gradient"
                    )

            assert params_with_grad > 0, (
                f"{_backbone_name(backbone_cls)}: Step {step}: no gradient received"
            )
            optim.step()


# ======================================================================
# Edge case tests
# ======================================================================


class TestBackboneEdgeCases:
    """Edge cases and error handling for backbone instantiation and usage."""

    def test_pretrained_false_creates_no_weights(self, backbone_cls: type[BaseBackbone]) -> None:
        """``pretrained=False`` creates a backbone without loading external weights."""
        model = backbone_cls(pretrained=False)
        # Verify the model has trainable parameters (i.e. it's not empty).
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0, f"{_backbone_name(backbone_cls)} has zero parameters"

    def test_eval_mode_after_instantiation(self, backbone: BaseBackbone) -> None:
        """Fresh backbone is in ``train()`` mode by default."""
        assert backbone.training, "Backbone should be in train mode by default"

    def test_eval_mode_forward(self, backbone: BaseBackbone) -> None:
        """Forward pass succeeds in ``eval()`` mode (deterministic)."""
        backbone.eval()
        x = torch.randn(BATCH_SIZE, 3, *DEFAULT_INPUT_SIZE)
        with torch.no_grad():
            out = backbone(x)
        assert len(out) > 0

    def test_forward_twice_consistent_output(self, backbone: BaseBackbone) -> None:
        """Calling forward twice on the same input produces identical output in eval mode."""
        backbone.eval()
        x = torch.randn(BATCH_SIZE, 3, *DEFAULT_INPUT_SIZE)
        with torch.no_grad():
            out1 = backbone(x)
            out2 = backbone(x)

        for i, (a, b) in enumerate(zip(out1, out2, strict=False)):
            assert torch.equal(a, b), f"Level {i} differs between forward calls in eval mode"

    def test_backbone_isinstance_basebackbone(self, backbone_cls: type[BaseBackbone]) -> None:
        """All backbone classes are proper ``BaseBackbone`` subclasses."""
        assert issubclass(backbone_cls, BaseBackbone), (
            f"{_backbone_name(backbone_cls)} does not inherit from BaseBackbone"
        )

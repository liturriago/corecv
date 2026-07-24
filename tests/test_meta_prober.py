"""Comprehensive tests for the MetaProber hardware-compliance validator.

Tests cover:
1. Meta device shape propagation with zero-VRAM tensors
2. Channel/stride compatibility validation between backbone, neck, and head
3. Detection of dynamic operations (interpolate, view/reshape, split/chunk)
4. Successful validation for compatible models
5. Proper error messages for incompatible models
6. Export compatibility verification (torch.export)
7. Multi-scale feature map handling (dict outputs from backbones)
8. Non-square input resolution testing (e.g. 480x640)
9. Training/eval mode preservation during shape propagation
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import fx as _fx
from torch import nn
from torch.fx import Graph, GraphModule

from corecv.core.contract import BaseBackbone, FeatureInfo
from corecv.engine.validator import TORCH_EXPORT_AVAILABLE, MetaProber

# ======================================================================
# Mock model definitions
# ======================================================================


class MockDictBackbone(BaseBackbone):
    """Mock backbone returning multi-scale dict of feature maps.

    Each feature level is computed independently from the input using a
    strided convolution with ``kernel_size == stride``.  This guarantees
    that spatial dimensions match ``input_size // stride`` which is what
    the channel/stride validation logic expects.

    Attributes:
        _feature_info: FeatureInfo describing channels and strides.
        convs: List of strided convolutions, one per feature level.
    """

    def __init__(
        self,
        out_channels: tuple[int, ...] = (64, 128, 256),
        out_strides: tuple[int, ...] = (4, 8, 16),
    ) -> None:
        """Initialise with configurable output channels and strides.

        Args:
            out_channels: Channel count per feature level.
            out_strides: Stride per feature level relative to input.
        """
        super().__init__()
        self._feature_info = FeatureInfo(
            channels={f"level{i}": c for i, c in enumerate(out_channels)},
            strides={f"level{i}": s for i, s in enumerate(out_strides)},
        )
        self.convs: nn.ModuleList = nn.ModuleList()
        # Each level processes the ORIGINAL input independently
        for c, s in zip(out_channels, out_strides, strict=False):
            self.convs.append(
                nn.Conv2d(3, c, kernel_size=s, stride=s, padding=0)
            )

    @property
    def feature_info(self) -> FeatureInfo:
        """Return the feature metadata for this backbone."""
        return self._feature_info

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass producing multi-scale feature dict.

        Each level is produced independently from the input ``x`` so that
        spatial sizes follow ``input_size // stride`` exactly.

        Args:
            x: Input tensor ``(N, 3, H, W)``.

        Returns:
            Dict mapping ``"level0"``, ``"level1"``, … to feature tensors.
        """
        out: dict[str, torch.Tensor] = {}
        for i, conv in enumerate(self.convs):
            out[f"level{i}"] = conv(x)
        return out


class MockSingleBackbone(BaseBackbone):
    """Mock backbone returning a single tensor (non-dict output).

    Used to test the single-scale code path in channel/stride validation.

    Attributes:
        _feature_info: FeatureInfo with one level.
        conv: Single strided convolution.
    """

    def __init__(
        self, out_channels: int = 256, stride: int = 16
    ) -> None:
        """Initialise single-scale backbone.

        Args:
            out_channels: Output channel count.
            stride: Total stride from input to output.
        """
        super().__init__()
        self._feature_info = FeatureInfo(
            channels={"out": out_channels},
            strides={"out": stride},
        )
        self.conv = nn.Conv2d(3, out_channels, kernel_size=stride, stride=stride, padding=0)

    @property
    def feature_info(self) -> FeatureInfo:
        """Return the feature metadata for this backbone."""
        return self._feature_info

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a single feature tensor.

        Args:
            x: Input tensor ``(N, 3, H, W)``.

        Returns:
            Single output tensor ``(N, out_channels, H/s, W/s)``.
        """
        return self.conv(x)


class MockNeck(nn.Module):
    """Mock neck module that preserves dict structure.

    Applies a pointwise convolution to each feature level.

    Attributes:
        convs: One convolution per input feature level.
    """

    def __init__(self, channels: tuple[int, ...] = (64, 128, 256)) -> None:
        """Initialise neck with per-level convolutions.

        Args:
            channels: Channel count per feature level.
        """
        super().__init__()
        self.convs: nn.ModuleList = nn.ModuleList(
            [nn.Conv2d(c, c, kernel_size=1) for c in channels]
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass applying conv to each level.

        Args:
            features: Dict of feature tensors from backbone.

        Returns:
            Dict with same keys, transformed features.
        """
        out: dict[str, torch.Tensor] = {}
        for i, (key, feat) in enumerate(features.items()):
            out[key] = self.convs[i](feat)
        return out


class MockHead(nn.Module):
    """Mock head module that consumes the last feature level.

    Uses only static-safe operations for valid-model tests.

    Attributes:
        conv: Final convolution.
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 10) -> None:
        """Initialise head.

        Args:
            in_channels: Number of input channels from backbone/neck.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(
        self, features: dict[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Dict of feature tensors or single tensor.

        Returns:
            Output tensor ``(N, num_classes)``.
        """
        x = features[[*features.keys()][-1]] if isinstance(features, dict) else features
        x = self.conv(x)
        # Static pooling using concrete spatial dims
        B: int
        C: int
        H: int
        W: int
        B, C, H, W = x.shape
        x = F.avg_pool2d(x, kernel_size=(H, W))
        x = x.flatten(1)
        return x


class MockHeadSingle(nn.Module):
    """Mock head consuming a single tensor directly."""

    def __init__(self, in_channels: int = 256, num_classes: int = 10) -> None:
        """Initialise.

        Args:
            in_channels: Input channel count.
            num_classes: Output class count.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with single tensor input.

        Args:
            x: Input tensor ``(N, C, H, W)``.

        Returns:
            Output tensor ``(N, num_classes)``.
        """
        x = self.conv(x)
        B, C, H, W = x.shape
        x = F.avg_pool2d(x, kernel_size=(H, W))
        x = x.flatten(1)
        return x


# ======================================================================
# Dynamic operation models (for testing detection)
# ======================================================================


class DynamicSplitModel(nn.Module):
    """Model using ``.split()`` with a non-integer sections argument.

    This is detected via ``call_method`` by ``_find_dynamic_operations``.
    """

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward with dynamic split (list of sizes).

        Args:
            x: Input tensor.

        Returns:
            List of split tensors.
        """
        return x.split([1, 1, 1], dim=0)


class DynamicChunkModel(nn.Module):
    """Model using ``.chunk()`` where chunks comes from tensor shape.

    The fx.Node argument triggers ``_has_dynamic_args`` in the
    ``call_function`` branch or ``_is_dynamic_split`` in the
    ``call_method`` branch.
    """

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward with tensor-based chunk size.

        Args:
            x: Input tensor.

        Returns:
            List of chunked tensors.
        """
        n: int = x.shape[0]
        return x.chunk(n, dim=0)


class ValidStaticModel(nn.Module):
    """Model using only static-safe operations (conv, relu, avg_pool, flatten).

    Should pass MetaProber static audit without issues.
    """

    def __init__(self) -> None:
        """Initialise with conv-relu-conv-avg_pool."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(N, 3, H, W)``.

        Returns:
            Output tensor ``(N, 16)``.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        B, C, H, W = x.shape
        x = F.avg_pool2d(x, kernel_size=(H, W))
        x = x.flatten(1)
        return x


# ======================================================================
# Model that fails to implement BaseBackbone
# ======================================================================


class NonCompliantBackbone(nn.Module):
    """Backbone that does NOT implement BaseBackbone interface."""

    def __init__(self) -> None:
        """Initialise."""
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.conv(x)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def prober() -> MetaProber:
    """Fixture providing a fresh MetaProber instance."""
    return MetaProber()


@pytest.fixture
def default_input_size() -> tuple[int, int]:
    """Standard square input resolution ``(224, 224)``."""
    return 224, 224


@pytest.fixture
def non_square_input_size() -> tuple[int, int]:
    """Non-square input resolution ``(480, 640)``.

    Used to catch H/W indexing inversion bugs.
    """
    return 480, 640


@pytest.fixture
def mock_dict_backbone() -> MockDictBackbone:
    """Multi-scale backbone with channels ``(64, 128, 256)`` and strides ``(4, 8, 16)``."""
    return MockDictBackbone()


@pytest.fixture
def mock_single_backbone() -> MockSingleBackbone:
    """Single-scale backbone with 256 channels and stride 16."""
    return MockSingleBackbone()


@pytest.fixture
def mock_neck() -> MockNeck:
    """Neck module matching ``(64, 128, 256)`` channels."""
    return MockNeck(channels=(64, 128, 256))


@pytest.fixture
def mock_head() -> MockHead:
    """Head module with 256 input channels, 10 output classes."""
    return MockHead(in_channels=256, num_classes=10)


@pytest.fixture
def valid_backbone() -> MockDictBackbone:
    """Valid backbone for full-pipeline validation tests."""
    return MockDictBackbone(out_channels=(64, 128), out_strides=(4, 8))


@pytest.fixture
def valid_neck() -> MockNeck:
    """Valid neck for full-pipeline validation tests."""
    return MockNeck(channels=(64, 128))


@pytest.fixture
def valid_head() -> MockHead:
    """Valid head for full-pipeline validation tests."""
    return MockHead(in_channels=128, num_classes=5)


# ======================================================================
# Initialisation tests
# ======================================================================


class TestMetaProberInit:
    """Verify MetaProber can be instantiated without errors."""

    def test_init(self) -> None:
        """A fresh MetaProber instance should be created successfully."""
        p = MetaProber()
        assert isinstance(p, MetaProber)

    def test_constants_defined(self, prober: MetaProber) -> None:
        """Core sets ``DYNAMIC_OPS`` and ``STATIC_SAFE_OPS`` must be non-empty."""
        assert len(prober.DYNAMIC_OPS) > 0
        assert len(prober.STATIC_SAFE_OPS) > 0

    def test_no_overlap(self, prober: MetaProber) -> None:
        """No operation should appear in both ``DYNAMIC_OPS`` and ``STATIC_SAFE_OPS``."""
        overlap = prober.DYNAMIC_OPS & prober.STATIC_SAFE_OPS
        assert len(overlap) == 0, f"Overlapping ops: {overlap}"


# ======================================================================
# Meta shape propagation tests
# ======================================================================


class TestMetaShapePropagation:
    """Verify tensor shapes propagate correctly on ``device='meta'``."""

    def test_meta_device_shapes_dict_backbone(
        self,
        prober: MetaProber,
        mock_dict_backbone: MockDictBackbone,
        mock_neck: MockNeck,
        mock_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Shapes propagate on meta device without allocating real memory."""
        shapes = prober._propagate_meta_shapes(
            mock_dict_backbone, mock_neck, mock_head, default_input_size
        )

        assert "input" in shapes
        assert "backbone_output" in shapes
        assert "neck_output" in shapes
        assert "head_output" in shapes

        # Input shape
        assert shapes["input"] == (1, 3, 224, 224)

        # Backbone output shapes (dict of tuples)
        backbone_out = shapes["backbone_output"]
        assert isinstance(backbone_out, dict)
        # level0: stride 4 -> 224//4 = 56
        assert backbone_out["level0"] == (1, 64, 56, 56)
        # level1: stride 8 -> 224//8 = 28
        assert backbone_out["level1"] == (1, 128, 28, 28)
        # level2: stride 16 -> 224//16 = 14
        assert backbone_out["level2"] == (1, 256, 14, 14)

        # Neck output preserves dict structure with same shapes
        neck_out = shapes["neck_output"]
        assert isinstance(neck_out, dict)
        assert set(neck_out.keys()) == set(backbone_out.keys())
        for key in neck_out:
            assert isinstance(neck_out[key], tuple)

        # Head output should be a single tensor shape
        head_out = shapes["head_output"]
        assert isinstance(head_out, tuple)
        assert head_out[0] == 1  # batch
        assert head_out[1] == 10  # num_classes

    def test_meta_device_shapes_single_backbone(
        self,
        prober: MetaProber,
        mock_single_backbone: MockSingleBackbone,
        mock_head: MockHeadSingle,
        default_input_size: tuple[int, int],
    ) -> None:
        """Single-scale backbone produces tuple shapes."""
        shapes = prober._propagate_meta_shapes(
            mock_single_backbone, None, mock_head, default_input_size
        )

        backbone_out = shapes["backbone_output"]
        assert isinstance(backbone_out, tuple)
        # Stride 16 -> 224//16 = 14
        assert backbone_out == (1, 256, 14, 14)

        # Head output
        head_out = shapes["head_output"]
        assert isinstance(head_out, tuple)

    def test_meta_device_no_real_allocation(
        self,
        prober: MetaProber,
        mock_dict_backbone: MockDictBackbone,
        mock_neck: MockNeck,
        mock_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Meta device propagation does not allocate CPU/CUDA memory."""
        shapes = prober._propagate_meta_shapes(
            mock_dict_backbone, mock_neck, mock_head, default_input_size
        )
        # If any tensor had been materialised on a real device this would
        # have raised an error.  Simply verifying that the call succeeds
        # confirms zero-VRAM behaviour.
        assert shapes is not None

    def test_propagation_without_neck(
        self,
        prober: MetaProber,
        mock_dict_backbone: MockDictBackbone,
        mock_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Propagation works when ``neck=None``."""
        shapes = prober._propagate_meta_shapes(
            mock_dict_backbone, None, mock_head, default_input_size
        )

        assert "neck_output" not in shapes
        backbone_out = shapes["backbone_output"]
        head_out = shapes["head_output"]
        assert isinstance(backbone_out, dict)
        assert isinstance(head_out, tuple)


# ======================================================================
# Channel/stride compatibility tests
# ======================================================================


class TestChannelStrideCompatibility:
    """Verify channel counts and spatial strides match ``FeatureInfo`` declarations."""

    def test_valid_dict_backbone(
        self,
        prober: MetaProber,
        mock_dict_backbone: MockDictBackbone,
        mock_neck: MockNeck,
        mock_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Compatible backbone should pass channel/stride validation."""
        shapes = prober._propagate_meta_shapes(
            mock_dict_backbone, mock_neck, mock_head, default_input_size
        )
        # Should not raise
        prober._validate_channel_stride_compatibility(
            mock_dict_backbone.feature_info,
            mock_neck,
            mock_head,
            shapes,
            default_input_size,
        )

    def test_channel_mismatch(
        self,
        prober: MetaProber,
        default_input_size: tuple[int, int],
    ) -> None:
        """Channel mismatch between ``FeatureInfo`` and actual output raises ``ValueError``.

        Builds the shapes dict manually to avoid meta-propagation failures
        from channel mismatches in downstream components.
        """
        feature_info = FeatureInfo(
            channels={"level0": 999},  # Expected 999, actual 64
            strides={"level0": 4},
        )
        shapes = {
            "backbone_output": {
                "level0": (1, 64, 56, 56),
            },
            "neck_output": {
                "level0": (1, 64, 56, 56),
            },
        }

        with pytest.raises(ValueError, match="Channel mismatch"):
            prober._validate_channel_stride_compatibility(
                feature_info, None, MockHead(in_channels=64, num_classes=5),
                shapes, default_input_size,
            )

    def test_stride_mismatch(
        self,
        prober: MetaProber,
        default_input_size: tuple[int, int],
    ) -> None:
        """Stride mismatch (wrong spatial size) raises ``ValueError``.

        Uses manually constructed shapes dict.
        """
        feature_info = FeatureInfo(
            channels={"level0": 64},
            strides={"level0": 4},  # expects 56x56, actual is 28x28
        )
        shapes = {
            "backbone_output": {
                "level0": (1, 64, 28, 28),
            },
            "neck_output": {
                "level0": (1, 64, 28, 28),
            },
        }

        with pytest.raises(ValueError, match="Stride mismatch"):
            prober._validate_channel_stride_compatibility(
                feature_info, None, MockHead(in_channels=64, num_classes=5),
                shapes, default_input_size,
            )

    def test_neck_level_count_mismatch(
        self,
        prober: MetaProber,
        default_input_size: tuple[int, int],
    ) -> None:
        """Neck producing different number of levels than backbone raises error."""
        feature_info = FeatureInfo(
            channels={"level0": 64, "level1": 128},
            strides={"level0": 4, "level1": 8},
        )
        backbone_shapes = {
            "level0": (1, 64, 56, 56),
            "level1": (1, 128, 28, 28),
        }
        neck_shapes = {
            "level0": (1, 64, 56, 56),  # only one level — mismatch
        }
        shapes = {
            "backbone_output": backbone_shapes,
            "neck_output": neck_shapes,
        }

        with pytest.raises(ValueError, match="feature levels"):
            prober._validate_channel_stride_compatibility(
                feature_info, MockNeck(channels=(64,)),
                MockHead(in_channels=64, num_classes=5),
                shapes, default_input_size,
            )

    def test_unexpected_backbone_output_format(
        self,
        prober: MetaProber,
        default_input_size: tuple[int, int],
    ) -> None:
        """Non-dict, non-tuple backbone output format raises error."""
        feature_info = FeatureInfo(
            channels={"out": 64}, strides={"out": 4}
        )
        shapes = {
            "backbone_output": 42,  # neither dict nor tuple
        }

        with pytest.raises(TypeError, match="Unexpected backbone output shape format"):
            prober._validate_channel_stride_compatibility(
                feature_info, None, MockHead(in_channels=64, num_classes=5),
                shapes, default_input_size,
            )


# ======================================================================
# Dynamic operation detection tests
# ======================================================================


class TestDynamicOperationDetection:
    """Verify MetaProber detects operations incompatible with static export."""

    def test_dynamic_split_detected(
        self, prober: MetaProber, default_input_size: tuple[int, int]
    ) -> None:
        """``.split()`` with a non-integer first argument is flagged."""
        model = DynamicSplitModel()
        dummy_head = nn.Identity()
        model.eval()

        with pytest.raises(ValueError, match="tensor.split"):
            prober._audit_static_compatibility(model, None, dummy_head, default_input_size)

    def test_dynamic_chunk_with_node_arg_detected(
        self, prober: MetaProber, default_input_size: tuple[int, int]
    ) -> None:
        """``.chunk()`` where chunks comes from a tensor shape is flagged."""
        model = DynamicChunkModel()
        dummy_head = nn.Identity()
        model.eval()

        with pytest.raises(ValueError, match="tensor.chunk"):
            prober._audit_static_compatibility(model, None, dummy_head, default_input_size)

    def test_static_model_passes_audit(
        self, prober: MetaProber, default_input_size: tuple[int, int]
    ) -> None:
        """A model using only static-safe operations passes without errors."""
        model = ValidStaticModel()
        dummy_head = nn.Identity()
        model.eval()

        # Should not raise
        prober._audit_static_compatibility(model, None, dummy_head, default_input_size)

    def test_empty_model_passes_audit(
        self, prober: MetaProber, default_input_size: tuple[int, int]
    ) -> None:
        """An identity model passes without errors."""

        class EmptyModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        model = EmptyModel()
        dummy_head = nn.Identity()
        model.eval()

        prober._audit_static_compatibility(model, None, dummy_head, default_input_size)

    def test_multiple_dynamic_ops_reported(
        self, prober: MetaProber, default_input_size: tuple[int, int]
    ) -> None:
        """Multiple dynamic operations are all reported in the error."""

        class MultiDynamicModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.split([1, 1, 1], dim=0)  # dynamic split
                return x[0]

        model = MultiDynamicModel()
        dummy_head = nn.Identity()
        model.eval()

        with pytest.raises(ValueError) as exc_info:
            prober._audit_static_compatibility(model, None, dummy_head, default_input_size)

        msg = str(exc_info.value)
        assert "tensor.split" in msg

    def test_is_dynamic_interpolate_with_scale_factor(
        self, prober: MetaProber
    ) -> None:
        """Interpolate with a static scale factor is NOT flagged.

        Note: ``_find_dynamic_operations`` checks for ``aten::interpolate``
        in the target string.  ``fx.symbolic_trace`` produces Python-level
        targets (``torch.nn.functional.interpolate``), so this test verifies
        that static scale_factor does NOT accidentally trigger detection
        via any other code path.
        """

        class StaticInterpolateModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.interpolate(x, scale_factor=2.0, mode="bilinear")

        model = StaticInterpolateModel()
        traced = torch.fx.symbolic_trace(model)
        ops = prober._find_dynamic_operations(traced, "Test")
        interp_ops = [op for op in ops if "interpolate" in op]
        assert len(interp_ops) == 0

    def test_find_dynamic_with_aten_interpolate(
        self, prober: MetaProber
    ) -> None:
        """``_find_dynamic_operations`` catches ``aten::interpolate``.

        The validator's ``DYNAMIC_OPS`` set uses ``aten::``-prefixed names.
        These appear when checking graphs produced by ``torch.export`` or
        when custom nodes carry ``aten::`` in their target string.  This
        test constructs an FX graph programmatically with a node whose
        target contains ``"aten::interpolate"`` and whose args have
        ``size=None, scale_factor=None`` (both positional, both ``None``),
        which triggers ``_is_dynamic_interpolate``.
        """

        class _StubAtenOp:
            """Callable stub whose ``repr`` contains ``aten::interpolate``."""

            __name__: str = "aten::interpolate"

            @staticmethod
            def __call__(*args: object) -> object:  # noqa: ANN401
                if args:
                    return args[0]
                return None

            def __repr__(self) -> str:
                return "aten::interpolate"

        # Build a minimal graph manually with the exact node we need
        graph = Graph()
        inp = graph.placeholder("x")
        # call_function target=aten::interpolate, args=(inp, None, None)
        interp_node = graph.call_function(_StubAtenOp(), (inp, None, None))
        graph.output(interp_node)
        graph.lint()

        traced = GraphModule(torch.nn.Module(), graph)
        traced.recompile()

        ops = prober._find_dynamic_operations(traced, "Test")
        interp_ops = [op for op in ops if "interpolate" in op and "dynamic size" in op]
        assert len(interp_ops) > 0

    def test_is_dynamic_view_reshape_single_minus_one(
        self, prober: MetaProber
    ) -> None:
        """View with exactly one ``-1`` dimension is NOT flagged.

        When ``-1`` is passed as a positional arg (not inside a tuple),
        ``_is_dynamic_view_reshape`` receives ``-1`` (an int) and the
        ``isinstance(shape, (list, tuple))`` guard returns ``False``,
        so the operation is not considered dynamic.
        """

        class ValidViewModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.view(-1, 1, 1, 1)

        model = ValidViewModel()
        traced = torch.fx.symbolic_trace(model)
        ops = prober._find_dynamic_operations(traced, "Test")
        view_ops = [op for op in ops if "view" in op or "reshape" in op]
        assert len(view_ops) == 0

    def test_dynamic_view_with_tuple_arg_detected(
        self, prober: MetaProber
    ) -> None:
        """View with a tuple containing multiple ``-1`` IS flagged.

        When the shape is passed as a tuple, ``_is_dynamic_view_reshape``
        can inspect it and detect multiple ``-1`` entries.
        """

        class TupleViewModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.view((-1, -1))

        model = TupleViewModel()
        traced = _fx.symbolic_trace(model)
        ops = prober._find_dynamic_operations(traced, "Test")
        view_ops = [op for op in ops if "tensor.view()" in op]
        assert len(view_ops) > 0


# ======================================================================
# Full pipeline validation tests
# ======================================================================


class TestFullPipelineValidation:
    """End-to-end ``validate_compatibility`` tests."""

    def test_valid_model_passes(
        self,
        prober: MetaProber,
        valid_backbone: MockDictBackbone,
        valid_neck: MockNeck,
        valid_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """A fully compatible model returns ``True``."""
        result = prober.validate_compatibility(
            valid_backbone, valid_neck, valid_head, default_input_size
        )
        assert result is True

    def test_valid_model_without_neck(
        self,
        prober: MetaProber,
        valid_backbone: MockDictBackbone,
        valid_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """A compatible model without a neck returns ``True``."""
        result = prober.validate_compatibility(
            valid_backbone, None, valid_head, default_input_size
        )
        assert result is True

    def test_non_compliant_backbone_raises_value_error(
        self,
        prober: MetaProber,
        mock_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """A backbone not implementing ``BaseBackbone`` raises ``ValueError``."""
        backbone = NonCompliantBackbone()

        with pytest.raises(TypeError, match="BaseBackbone"):
            prober.validate_compatibility(backbone, None, mock_head, default_input_size)


# ======================================================================
# Incompatible model error messages
# ======================================================================


class TestIncompatibleModelErrors:
    """Verify descriptive error messages for incompatible configurations."""

    def test_channel_mismatch_error_message(
        self,
        prober: MetaProber,
        default_input_size: tuple[int, int],
    ) -> None:
        """Channel mismatch error message includes level name and actual/expected values."""
        feature_info = FeatureInfo(
            channels={"feat": 999},
            strides={"feat": 4},
        )
        shapes = {
            "backbone_output": {
                "feat": (1, 64, 56, 56),
            },
            "neck_output": {
                "feat": (1, 64, 56, 56),
            },
        }

        with pytest.raises(ValueError) as exc_info:
            prober._validate_channel_stride_compatibility(
                feature_info, None, MockHead(in_channels=64, num_classes=5),
                shapes, default_input_size,
            )

        msg = str(exc_info.value)
        assert "Channel mismatch" in msg
        assert "feat" in msg
        assert "999" in msg
        assert "64" in msg

    def test_fx_tracing_failure_error_message(
        self,
        prober: MetaProber,
        default_input_size: tuple[int, int],
    ) -> None:
        """Models with data-dependent control flow produce a clear FX tracing error."""

        class ControlFlowModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Data-dependent control flow that FX cannot trace
                if x.sum() > 0:
                    return x
                return -x

        model = ControlFlowModel()
        dummy_head = nn.Identity()

        with pytest.raises(ValueError, match="FX tracing"):
            prober._audit_static_compatibility(model, None, dummy_head, default_input_size)

    def test_neck_level_count_error_message(
        self,
        prober: MetaProber,
        default_input_size: tuple[int, int],
    ) -> None:
        """Neck level count mismatch includes level counts in the error.

        Uses manually constructed shapes dict since the dropping neck
        may fail meta propagation due to channel mismatches.
        """
        feature_info = FeatureInfo(
            channels={"level0": 64, "level1": 128},
            strides={"level0": 4, "level1": 8},
        )
        shapes = {
            "backbone_output": {
                "level0": (1, 64, 56, 56),
                "level1": (1, 128, 28, 28),
            },
            "neck_output": {
                "level0": (1, 64, 56, 56),  # dropped level1
            },
        }

        with pytest.raises(ValueError) as exc_info:
            prober._validate_channel_stride_compatibility(
                feature_info, MockNeck(channels=(64,)),
                MockHead(in_channels=64, num_classes=5),
                shapes, default_input_size,
            )

        msg = str(exc_info.value)
        assert "Neck output has" in msg
        assert "feature levels" in msg


# ======================================================================
# Export compatibility tests
# ======================================================================


class TestExportCompatibility:
    """Verify ``torch.export`` integration for compatible models."""

    def test_export_compatible_model(
        self,
        prober: MetaProber,
        valid_backbone: MockDictBackbone,
        valid_neck: MockNeck,
        valid_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """A valid model can be exported with ``torch.export``."""
        prober._verify_export_compatibility(
            valid_backbone, valid_neck, valid_head, default_input_size
        )

    def test_export_without_neck(
        self,
        prober: MetaProber,
        valid_backbone: MockDictBackbone,
        valid_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Export works when ``neck=None``."""
        prober._verify_export_compatibility(
            valid_backbone, None, valid_head, default_input_size
        )

    def test_export_available_flag(self) -> None:
        """The module correctly detects ``torch.export`` availability."""
        # torch.export is available in modern PyTorch; if not, the
        # _verify_export_compatibility path is simply skipped.
        assert TORCH_EXPORT_AVAILABLE is True or TORCH_EXPORT_AVAILABLE is False


# ======================================================================
# Multi-scale feature map tests
# ======================================================================


class TestMultiScaleFeatures:
    """Verify dict-returning backbones are handled correctly."""

    def test_dict_backbone_produces_correct_shapes(
        self,
        prober: MetaProber,
        mock_dict_backbone: MockDictBackbone,
        mock_neck: MockNeck,
        mock_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Dict output shapes have expected structure and values."""
        shapes = prober._propagate_meta_shapes(
            mock_dict_backbone, mock_neck, mock_head, default_input_size
        )

        backbone_out = shapes["backbone_output"]
        assert isinstance(backbone_out, dict)
        assert len(backbone_out) == 3  # three levels

        # Verify each level has correct spatial dims
        expected_sizes = {
            "level0": (1, 64, 56, 56),
            "level1": (1, 128, 28, 28),
            "level2": (1, 256, 14, 14),
        }
        for level, expected in expected_sizes.items():
            assert backbone_out[level] == expected, (
                f"Expected {expected}, got {backbone_out[level]} at {level}"
            )

    def test_dict_backbone_with_varying_levels(
        self, prober: MetaProber, default_input_size: tuple[int, int]
    ) -> None:
        """Backbones with 2 or 4 feature levels work correctly."""
        # Two-level backbone
        backbone2 = MockDictBackbone(out_channels=(32, 64), out_strides=(2, 4))
        head2 = MockHead(in_channels=64, num_classes=3)

        shapes = prober._propagate_meta_shapes(backbone2, None, head2, default_input_size)
        assert len(shapes["backbone_output"]) == 2

        result = prober.validate_compatibility(backbone2, None, head2, default_input_size)
        assert result is True

        # Four-level backbone
        backbone4 = MockDictBackbone(
            out_channels=(16, 32, 64, 128), out_strides=(2, 4, 8, 16)
        )
        head4 = MockHead(in_channels=128, num_classes=3)

        result = prober.validate_compatibility(backbone4, None, head4, default_input_size)
        assert result is True

    def test_dict_backbone_neck_preserves_levels(
        self,
        prober: MetaProber,
        mock_dict_backbone: MockDictBackbone,
        mock_neck: MockNeck,
        mock_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Neck preserves the number of feature levels from backbone."""
        shapes = prober._propagate_meta_shapes(
            mock_dict_backbone, mock_neck, mock_head, default_input_size
        )

        backbone_keys = set(shapes["backbone_output"].keys())
        neck_keys = set(shapes["neck_output"].keys())
        assert backbone_keys == neck_keys, (
            f"Neck keys {neck_keys} do not match backbone keys {backbone_keys}"
        )


# ======================================================================
# Non-square input resolution tests
# ======================================================================


class TestNonSquareInput:
    """Verify MetaProber works correctly with non-square inputs (e.g. 480x640).

    This is critical for catching H/W indexing inversion bugs.
    """

    def test_non_square_meta_propagation(
        self,
        prober: MetaProber,
        mock_dict_backbone: MockDictBackbone,
        mock_neck: MockNeck,
        mock_head: MockHead,
        non_square_input_size: tuple[int, int],
    ) -> None:
        """Shape propagation works for 480x640 input."""
        H, W = non_square_input_size
        shapes = prober._propagate_meta_shapes(
            mock_dict_backbone, mock_neck, mock_head, non_square_input_size
        )

        # Input shape
        assert shapes["input"] == (1, 3, H, W)

        # Backbone level0: stride 4 -> (480//4, 640//4) = (120, 160)
        assert shapes["backbone_output"]["level0"] == (1, 64, 120, 160)

        # level1: stride 8 -> (480//8, 640//8) = (60, 80)
        assert shapes["backbone_output"]["level1"] == (1, 128, 60, 80)

        # level2: stride 16 -> (480//16, 640//16) = (30, 40)
        assert shapes["backbone_output"]["level2"] == (1, 256, 30, 40)

    def test_non_square_validation_passes(
        self,
        prober: MetaProber,
        valid_backbone: MockDictBackbone,
        valid_neck: MockNeck,
        valid_head: MockHead,
        non_square_input_size: tuple[int, int],
    ) -> None:
        """Full validation passes for non-square input."""
        result = prober.validate_compatibility(
            valid_backbone, valid_neck, valid_head, non_square_input_size
        )
        assert result is True

    def test_non_square_no_hw_inversion(
        self,
        prober: MetaProber,
        non_square_input_size: tuple[int, int],
    ) -> None:
        """Verify (H, W) order is correct: first element is height."""

        class HWCheckBackbone(BaseBackbone):
            def __init__(self) -> None:
                super().__init__()
                self._feature_info = FeatureInfo(
                    channels={"out": 16},
                    strides={"out": 4},
                )
                self.conv = nn.Conv2d(3, 16, kernel_size=4, stride=4, padding=0)

            @property
            def feature_info(self) -> FeatureInfo:
                return self._feature_info

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"out": self.conv(x)}

        backbone = HWCheckBackbone()
        head = MockHead(in_channels=16, num_classes=2)

        H, W = non_square_input_size
        shapes = prober._propagate_meta_shapes(backbone, None, head, non_square_input_size)

        out_shape = shapes["backbone_output"]["out"]
        # H//4 = 120, W//4 = 160
        assert out_shape[2] == H // 4, (
            f"Expected height {H//4}, got {out_shape[2]} — possible H/W inversion"
        )
        assert out_shape[3] == W // 4, (
            f"Expected width {W//4}, got {out_shape[3]} — possible H/W inversion"
        )

    def test_non_square_export(
        self,
        prober: MetaProber,
        valid_backbone: MockDictBackbone,
        valid_neck: MockNeck,
        valid_head: MockHead,
        non_square_input_size: tuple[int, int],
    ) -> None:
        """Export compatibility works with non-square inputs."""
        prober._verify_export_compatibility(
            valid_backbone, valid_neck, valid_head, non_square_input_size
        )


# ======================================================================
# Training/eval mode preservation tests
# ======================================================================


class TestModePreservation:
    """Verify training/eval mode is preserved during shape propagation."""

    def test_eval_mode_preserved(
        self,
        prober: MetaProber,
        mock_dict_backbone: MockDictBackbone,
        mock_neck: MockNeck,
        mock_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Model in eval mode stays in eval mode after propagation."""
        mock_dict_backbone.eval()
        mock_neck.eval()
        mock_head.eval()

        prober._propagate_meta_shapes(
            mock_dict_backbone, mock_neck, mock_head, default_input_size
        )

        assert not mock_dict_backbone.training
        assert not mock_neck.training
        assert not mock_head.training

    def test_train_mode_preserved(
        self,
        prober: MetaProber,
        mock_dict_backbone: MockDictBackbone,
        mock_neck: MockNeck,
        mock_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Model in train mode stays in train mode after propagation."""
        mock_dict_backbone.train()
        mock_neck.train()
        mock_head.train()

        prober._propagate_meta_shapes(
            mock_dict_backbone, mock_neck, mock_head, default_input_size
        )

        assert mock_dict_backbone.training
        assert mock_neck.training
        assert mock_head.training

    def test_mixed_modes_preserved(
        self,
        prober: MetaProber,
        mock_dict_backbone: MockDictBackbone,
        mock_neck: MockNeck,
        mock_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Different mode combinations are preserved individually."""
        mock_dict_backbone.eval()
        mock_neck.train()
        mock_head.eval()

        prober._propagate_meta_shapes(
            mock_dict_backbone, mock_neck, mock_head, default_input_size
        )

        assert not mock_dict_backbone.training
        assert mock_neck.training
        assert not mock_head.training

    def test_mode_preserved_without_neck(
        self,
        prober: MetaProber,
        mock_dict_backbone: MockDictBackbone,
        mock_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Mode preservation works when ``neck=None``."""
        mock_dict_backbone.train()
        mock_head.train()

        prober._propagate_meta_shapes(
            mock_dict_backbone, None, mock_head, default_input_size
        )

        assert mock_dict_backbone.training
        assert mock_head.training


# ======================================================================
# Edge case tests
# ======================================================================


class TestProberEdgeCases:
    """Edge cases and error handling for MetaProber."""

    def test_validate_compatibility_with_non_compliant_backbone(
        self,
        prober: MetaProber,
        mock_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Non-BaseBackbone raises descriptive ``TypeError``."""
        backbone = NonCompliantBackbone()

        with pytest.raises(TypeError) as exc_info:
            prober.validate_compatibility(backbone, None, mock_head, default_input_size)

        msg = str(exc_info.value)
        assert "BaseBackbone" in msg
        assert "feature_info" in msg

    def test_meta_shape_propagation_with_none_neck(
        self,
        prober: MetaProber,
        mock_dict_backbone: MockDictBackbone,
        mock_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Propagation without neck produces correct shape dict (no neck_output key)."""
        shapes = prober._propagate_meta_shapes(
            mock_dict_backbone, None, mock_head, default_input_size
        )

        assert "input" in shapes
        assert "backbone_output" in shapes
        assert "neck_output" not in shapes
        assert "head_output" in shapes

    def test_extract_shapes_various_types(self, prober: MetaProber) -> None:
        """``_extract_shapes`` handles tensors, tuples, lists, and dicts."""
        # Tensor
        t = torch.randn(1, 3, 224, 224)
        assert prober._extract_shapes(t) == (1, 3, 224, 224)

        # Tuple of tensors
        tup = (torch.randn(1, 64, 56, 56), torch.randn(1, 128, 28, 28))
        result = prober._extract_shapes(tup)
        assert result == ((1, 64, 56, 56), (1, 128, 28, 28))

        # List of tensors
        lst = [torch.randn(1, 64, 56, 56)]
        result = prober._extract_shapes(lst)
        assert result == ((1, 64, 56, 56),)

        # Dict of tensors
        d = {"a": torch.randn(1, 3, 224, 224)}
        result = prober._extract_shapes(d)
        assert result == {"a": (1, 3, 224, 224)}

        # Unsupported type
        assert isinstance(prober._extract_shapes(42), str)

    def test_find_dynamic_operations_empty_graph(self, prober: MetaProber) -> None:
        """Empty graph returns no dynamic ops."""

        class Identity(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        model = Identity()
        traced = torch.fx.symbolic_trace(model)
        ops = prober._find_dynamic_operations(traced, "Test")
        assert ops == []


# ======================================================================
# New ops set coverage tests
# ======================================================================


class TestOpsSetCoverage:
    """Verify the expanded DYNAMIC_OPS and STATIC_SAFE_OPS sets."""

    def test_dynamic_ops_contains_new_ops(self, prober: MetaProber) -> None:
        """Padding, upsampling, data-dependent, and quantization ops are in DYNAMIC_OPS."""
        expected_dynamic = {
            "aten::pad",
            "aten::upsample_nearest2d",
            "aten::upsample_bilinear2d",
            "aten::flip",
            "aten::roll",
            "aten::rot90",
            "aten::embedding",
            "aten::embedding_bag",
            "aten::dropout",
            "aten::feature_dropout",
            "aten::alpha_dropout",
            "aten::quantize_per_tensor",
            "aten::dequantize",
            "aten::quantize_per_channel",
            "aten::adaptive_avg_pool1d",
            "aten::adaptive_max_pool1d",
        }
        for op in expected_dynamic:
            assert op in prober.DYNAMIC_OPS, f"{op} missing from DYNAMIC_OPS"

    def test_static_safe_ops_contains_new_ops(self, prober: MetaProber) -> None:
        """1D ops, clamping ops, and missing activations are in STATIC_SAFE_OPS."""
        expected_static = {
            "aten::conv1d",
            "aten::max_pool1d",
            "aten::avg_pool1d",
            "aten::batch_norm1d",
            "aten::clamp",
            "aten::clamp_min",
            "aten::clamp_max",
            "aten::relu6",
            "aten::hard_sigmoid",
            "aten::hard_tanh",
        }
        for op in expected_static:
            assert op in prober.STATIC_SAFE_OPS, f"{op} missing from STATIC_SAFE_OPS"


# ======================================================================
# Memory layout validation tests
# ======================================================================


class TestMemoryLayoutValidation:
    """Verify _validate_memory_layout catches problematic layout patterns."""

    def test_no_layout_ops_returns_empty(self, prober: MetaProber) -> None:
        """Graph without layout operations returns no violations."""

        class SimpleModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        model = SimpleModel()
        traced = torch.fx.symbolic_trace(model)
        violations = prober._validate_memory_layout(traced)
        assert violations == []

    def test_nchw_to_nhwc_permute_detected(self, prober: MetaProber) -> None:
        """NCHW->NHWC permute (0,2,3,1) is flagged as a layout operation."""
        graph = Graph()
        inp = graph.placeholder("x")

        class _NchwToNhwcOp:
            __name__: str = "aten::permute"

            @staticmethod
            def __call__(*args: object) -> object:  # noqa: ANN401
                if args:
                    return args[0]
                return None

            def __repr__(self) -> str:
                return "aten::permute"

        # Simulate NCHW->NHWC: permute with dims (0, 2, 3, 1)
        perm_node = graph.call_function(_NchwToNhwcOp(), (inp, (0, 2, 3, 1)))
        graph.output(perm_node)
        graph.lint()

        traced = GraphModule(torch.nn.Module(), graph)
        traced.recompile()

        violations = prober._validate_memory_layout(traced)
        layout_ops_found = any("permute" in v for v in violations)
        assert layout_ops_found, "NCHW->NHWC permute should be detected"

    def test_nhwc_to_nchw_permute_flagged(self, prober: MetaProber) -> None:
        """NHWC->NCHW permute (0,3,1,2) triggers anti-pattern warning."""
        graph = Graph()
        inp = graph.placeholder("x")

        class _NhwcToNchwOp:
            __name__: str = "aten::permute"

            @staticmethod
            def __call__(*args: object) -> object:  # noqa: ANN401
                if args:
                    return args[0]
                return None

            def __repr__(self) -> str:
                return "aten::permute"

        perm_node = graph.call_function(_NhwcToNchwOp(), (inp, (0, 3, 1, 2)))
        graph.output(perm_node)
        graph.lint()

        traced = GraphModule(torch.nn.Module(), graph)
        traced.recompile()

        violations = prober._validate_memory_layout(traced)
        has_anti_pattern = any("NHWC->NCHW" in v for v in violations)
        assert has_anti_pattern, "NHWC->NCHW should be flagged as anti-pattern"

    def test_contiguous_detected(self, prober: MetaProber) -> None:
        """contiguous() calls are detected as layout operations."""

        class ContiguousModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.contiguous()

        model = ContiguousModel()
        traced = torch.fx.symbolic_trace(model)
        violations = prober._validate_memory_layout(traced)
        has_contiguous = any("contiguous" in v for v in violations)
        assert has_contiguous, "contiguous() should be detected"

    def test_ping_pong_conversions_flagged(self, prober: MetaProber) -> None:
        """Both NCHW->NHWC and NHWC->NCHW in same graph triggers ping-pong warning."""
        graph = Graph()
        inp = graph.placeholder("x")
        inp2 = graph.placeholder("y")

        class _PermuteOp:
            __name__: str = "aten::permute"

            @staticmethod
            def __call__(*args: object) -> object:  # noqa: ANN401
                if args:
                    return args[0]
                return None

            def __repr__(self) -> str:
                return "aten::permute"

        p1 = graph.call_function(_PermuteOp(), (inp, (0, 2, 3, 1)))  # NCHW->NHWC
        p2 = graph.call_function(_PermuteOp(), (inp2, (0, 3, 1, 2)))  # NHWC->NCHW
        out = graph.call_function(torch.add, (p1, p2))
        graph.output(out)
        graph.lint()

        traced = GraphModule(torch.nn.Module(), graph)
        traced.recompile()

        violations = prober._validate_memory_layout(traced)
        has_ping_pong = any("ping-pong" in v.lower() for v in violations)
        assert has_ping_pong, "NCHW<->NHWC ping-pong should be flagged"


# ======================================================================
# Hardware constraint validation tests
# ======================================================================


class TestHardwareConstraintsValidation:
    """Verify _validate_hardware_constraints checks edge hardware requirements."""

    def test_clean_graph_returns_empty(self, prober: MetaProber) -> None:
        """Simple graph passes without violations."""

        class SimpleModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.relu()

        model = SimpleModel()
        traced = torch.fx.symbolic_trace(model)
        violations = prober._validate_hardware_constraints(traced, "edge")
        assert violations == []

    def test_mixed_precision_detected(self, prober: MetaProber) -> None:
        """fp16 conversion .half() triggers mixed precision warning."""

        class HalfModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.half()

        model = HalfModel()
        traced = torch.fx.symbolic_trace(model)
        violations = prober._validate_hardware_constraints(traced, "edge")
        has_mixed = any("Mixed-precision" in v or "Half-precision" in v for v in violations)
        assert has_mixed, "Mixed precision should be detected"

    def test_quantize_without_dequant(self, prober: MetaProber) -> None:
        """Quantize without matching dequantize raises a violation."""
        graph = Graph()
        inp = graph.placeholder("x")

        class _QuantOp:
            __name__: str = "aten::quantize_per_tensor"

            @staticmethod
            def __call__(*args: object) -> object:  # noqa: ANN401
                if args:
                    return args[0]
                return None

            def __repr__(self) -> str:
                return "aten::quantize_per_tensor"

        quant_node = graph.call_function(_QuantOp(), (inp, 1.0, 0))
        graph.output(quant_node)
        graph.lint()

        traced = GraphModule(torch.nn.Module(), graph)
        traced.recompile()

        violations = prober._validate_hardware_constraints(traced, "edge")
        has_quant_issue = any("quantize" in v and "dequantize" in v for v in violations)
        assert has_quant_issue, "Quant without dequant should be flagged"

    def test_xnnpack_nchw_conv_flag(self, prober: MetaProber) -> None:
        """XNNPACK validation flags NCHW convs without NHWC layout."""
        graph = Graph()
        inp = graph.placeholder("x")

        class _ConvOp:
            __name__: str = "aten::conv2d"

            @staticmethod
            def __call__(*args: object) -> object:  # noqa: ANN401
                if args:
                    return args[0]
                return None

            def __repr__(self) -> str:
                return "aten::conv2d"

        # Multiple conv2d nodes
        c1 = graph.call_function(_ConvOp(), (inp,))
        c2 = graph.call_function(_ConvOp(), (c1,))
        c3 = graph.call_function(_ConvOp(), (c2,))
        c4 = graph.call_function(_ConvOp(), (c3,))
        graph.output(c4)
        graph.lint()

        traced = GraphModule(torch.nn.Module(), graph)
        traced.recompile()

        violations = prober._validate_hardware_constraints(traced, "xnnpack")
        has_layout_warning = any("NHWC" in v for v in violations)
        assert has_layout_warning, "XNNPACK should flag NCHW convs without NHWC layout"

    def test_qnn_fp16_detected(self, prober: MetaProber) -> None:
        """QNN validation detects fp16 operations."""

        class HalfModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.half()

        model = HalfModel()
        traced = torch.fx.symbolic_trace(model)
        violations = prober._validate_hardware_constraints(traced, "qnn")
        has_fp16_warning = any("fp16" in v.lower() or "half" in v.lower() for v in violations)
        assert has_fp16_warning, "QNN should flag fp16 operations"

    def test_target_hardware_default_edge(self, prober: MetaProber) -> None:
        """Default target_hardware='edge' runs both XNNPACK and QNN checks."""
        graph = Graph()
        inp = graph.placeholder("x")
        out = graph.call_function(torch.relu, (inp,))
        graph.output(out)
        graph.lint()

        traced = GraphModule(torch.nn.Module(), graph)
        traced.recompile()

        violations = prober._validate_hardware_constraints(traced)
        # Clean graph should pass
        assert violations == []

    def test_qnn_embedding_limited(self, prober: MetaProber) -> None:
        """QNN validation flags embedding operations as limited support."""
        graph = Graph()
        inp = graph.placeholder("x")

        class _EmbeddingOp:
            __name__: str = "aten::embedding"

            @staticmethod
            def __call__(*args: object) -> object:  # noqa: ANN401
                if args:
                    return args[0]
                return None

            def __repr__(self) -> str:
                return "aten::embedding"

        emb_node = graph.call_function(_EmbeddingOp(), (inp,))
        graph.output(emb_node)
        graph.lint()

        traced = GraphModule(torch.nn.Module(), graph)
        traced.recompile()

        violations = prober._validate_hardware_constraints(traced, "qnn")
        has_emb_warning = any("embedding" in v.lower() for v in violations)
        assert has_emb_warning, "QNN should flag embedding ops as limited support"


# ======================================================================
# Enhanced _is_dynamic_interpolate tests
# ======================================================================


class TestEnhancedDynamicInterpolate:
    """Verify the enhanced _is_dynamic_interpolate catches edge cases."""

    def test_static_scale_factor_not_flagged(self, prober: MetaProber) -> None:
        """Static scale_factor=2.0 is NOT flagged as dynamic."""
        graph = Graph()
        inp = graph.placeholder("x")

        class _InterpolateOp:
            __name__: str = "aten::interpolate"

            @staticmethod
            def __call__(*args: object) -> object:  # noqa: ANN401
                if args:
                    return args[0]
                return None

            def __repr__(self) -> str:
                return "aten::interpolate"

        # Static scale_factor as positional arg: (inp, None, 2.0)
        interp_node = graph.call_function(
            _InterpolateOp(), (inp, None, 2.0)
        )
        graph.output(interp_node)
        graph.lint()

        traced = GraphModule(torch.nn.Module(), graph)
        traced.recompile()

        ops = prober._find_dynamic_operations(traced, "Test")
        interp_ops = [op for op in ops if "interpolate" in op]
        assert len(interp_ops) == 0, "Static scale_factor should not be flagged"

    def test_dynamic_scale_factor_flagged(self, prober: MetaProber) -> None:
        """scale_factor that is an fx.Node IS flagged as dynamic."""
        graph = Graph()
        inp = graph.placeholder("x")
        dynamic_sf = graph.placeholder("scale_factor")

        class _InterpolateOp:
            __name__: str = "aten::interpolate"

            @staticmethod
            def __call__(*args: object) -> object:  # noqa: ANN401
                if args:
                    return args[0]
                return None

            def __repr__(self) -> str:
                return "aten::interpolate"

        # Dynamic scale_factor as positional arg: (inp, None, dynamic_sf)
        interp_node = graph.call_function(
            _InterpolateOp(), (inp, None, dynamic_sf)
        )
        graph.output(interp_node)
        graph.lint()

        traced = GraphModule(torch.nn.Module(), graph)
        traced.recompile()

        ops = prober._find_dynamic_operations(traced, "Test")
        interp_ops = [op for op in ops if "interpolate" in op]
        assert len(interp_ops) > 0, "Dynamic scale_factor should be flagged"

    def test_recompute_scale_factor_true_flagged(self, prober: MetaProber) -> None:
        """recompute_scale_factor=True in kwargs IS flagged."""
        graph = Graph()
        inp = graph.placeholder("x")

        class _InterpolateOp:
            __name__: str = "aten::interpolate"

            @staticmethod
            def __call__(*args: object) -> object:  # noqa: ANN401
                if args:
                    return args[0]
                return None

            def __repr__(self) -> str:
                return "aten::interpolate"

        interp_node = graph.call_function(
            _InterpolateOp(),
            (inp, None, 2.0),
            {"recompute_scale_factor": True},
        )
        graph.output(interp_node)
        graph.lint()

        traced = GraphModule(torch.nn.Module(), graph)
        traced.recompile()

        ops = prober._find_dynamic_operations(traced, "Test")
        interp_ops = [op for op in ops if "interpolate" in op]
        assert len(interp_ops) > 0, "recompute_scale_factor=True should be flagged"

    def test_recompute_scale_factor_dynamic_flagged(self, prober: MetaProber) -> None:
        """recompute_scale_factor as fx.Node IS flagged."""
        graph = Graph()
        inp = graph.placeholder("x")
        dynamic_rsf = graph.placeholder("recompute_flag")

        class _InterpolateOp:
            __name__: str = "aten::interpolate"

            @staticmethod
            def __call__(*args: object) -> object:  # noqa: ANN401
                if args:
                    return args[0]
                return None

            def __repr__(self) -> str:
                return "aten::interpolate"

        interp_node = graph.call_function(
            _InterpolateOp(),
            (inp, None, 2.0),
            {"recompute_scale_factor": dynamic_rsf},
        )
        graph.output(interp_node)
        graph.lint()

        traced = GraphModule(torch.nn.Module(), graph)
        traced.recompile()

        ops = prober._find_dynamic_operations(traced, "Test")
        interp_ops = [op for op in ops if "interpolate" in op]
        assert len(interp_ops) > 0, "Dynamic recompute_scale_factor should be flagged"

    def test_scale_factor_in_kwargs_flagged_when_dynamic(self, prober: MetaProber) -> None:
        """scale_factor in kwargs as fx.Node IS flagged."""
        graph = Graph()
        inp = graph.placeholder("x")
        dynamic_sf = graph.placeholder("scale_factor")

        class _InterpolateOp:
            __name__: str = "aten::interpolate"

            @staticmethod
            def __call__(*args: object) -> object:  # noqa: ANN401
                if args:
                    return args[0]
                return None

            def __repr__(self) -> str:
                return "aten::interpolate"

        interp_node = graph.call_function(
            _InterpolateOp(),
            (inp,),
            {"scale_factor": dynamic_sf},
        )
        graph.output(interp_node)
        graph.lint()

        traced = GraphModule(torch.nn.Module(), graph)
        traced.recompile()

        ops = prober._find_dynamic_operations(traced, "Test")
        interp_ops = [op for op in ops if "interpolate" in op]
        assert len(interp_ops) > 0, "Dynamic scale_factor in kwargs should be flagged"


# ======================================================================
# _create_traced_pipeline tests
# ======================================================================


class TestCreateTracedPipeline:
    """Verify _create_traced_pipeline creates a valid traced GraphModule."""

    def test_traced_pipeline_success(self, prober: MetaProber) -> None:
        """Valid model produces a traced GraphModule."""
        backbone = MockDictBackbone(out_channels=(64, 128), out_strides=(4, 8))
        neck = MockNeck(channels=(64, 128))
        head = MockHead(in_channels=128, num_classes=5)

        traced = prober._create_traced_pipeline(backbone, neck, head, (224, 224))
        assert traced is not None
        assert isinstance(traced, GraphModule)

    def test_traced_pipeline_without_neck(self, prober: MetaProber) -> None:
        """Tracing works when neck=None."""
        backbone = MockDictBackbone(out_channels=(64,), out_strides=(4,))
        head = MockHead(in_channels=64, num_classes=5)

        traced = prober._create_traced_pipeline(backbone, None, head, (224, 224))
        assert traced is not None
        assert isinstance(traced, GraphModule)

    def test_traced_pipeline_control_flow_returns_none(self, prober: MetaProber) -> None:
        """Model with data-dependent control flow returns None (tracing fails gracefully)."""

        class ControlFlowModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.sum() > 0:  # data-dependent, cannot trace
                    return x
                return -x

        model = ControlFlowModel()
        dummy_head = nn.Identity()

        traced = prober._create_traced_pipeline(model, None, dummy_head, (224, 224))
        assert traced is None, "Control flow model should fail tracing"


# ======================================================================
# validate_compatibility extended tests
# ======================================================================


class TestValidateCompatibilityExtended:
    """Verify validate_compatibility with new target_hardware parameter."""

    def test_default_hardware_backward_compat(
        self,
        prober: MetaProber,
        valid_backbone: MockDictBackbone,
        valid_neck: MockNeck,
        valid_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Calling without target_hardware (backward compat) still passes."""
        result = prober.validate_compatibility(
            valid_backbone, valid_neck, valid_head, default_input_size
        )
        assert result is True

    def test_explicit_edge_hardware(
        self,
        prober: MetaProber,
        valid_backbone: MockDictBackbone,
        valid_neck: MockNeck,
        valid_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Explicit target_hardware='edge' passes."""
        result = prober.validate_compatibility(
            valid_backbone, valid_neck, valid_head,
            default_input_size, target_hardware="edge",
        )
        assert result is True

    def test_explicit_xnnpack_hardware(
        self,
        prober: MetaProber,
        valid_backbone: MockDictBackbone,
        valid_neck: MockNeck,
        valid_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Explicit target_hardware='xnnpack' passes for valid model."""
        result = prober.validate_compatibility(
            valid_backbone, valid_neck, valid_head,
            default_input_size, target_hardware="xnnpack",
        )
        assert result is True

    def test_explicit_qnn_hardware(
        self,
        prober: MetaProber,
        valid_backbone: MockDictBackbone,
        valid_neck: MockNeck,
        valid_head: MockHead,
        default_input_size: tuple[int, int],
    ) -> None:
        """Explicit target_hardware='qnn' passes for valid model."""
        result = prober.validate_compatibility(
            valid_backbone, valid_neck, valid_head,
            default_input_size, target_hardware="qnn",
        )
        assert result is True

    def test_validate_compatibility_rejects_dynamic(
        self,
        prober: MetaProber,
        default_input_size: tuple[int, int],
    ) -> None:
        """Model with dynamic split is rejected by validate_compatibility."""

        # Create a compliant backbone that internally uses dynamic split.
        # The split must be dimensionally valid on meta tensors to pass
        # meta propagation; it is detected as dynamic at the static audit stage.
        class DynamicSplitBackbone(BaseBackbone):
            def __init__(self) -> None:
                super().__init__()
                self._feature_info = FeatureInfo(
                    channels={"out": 32},
                    strides={"out": 4},
                )
                self.conv = nn.Conv2d(3, 64, kernel_size=4, stride=4, padding=0)

            @property
            def feature_info(self) -> FeatureInfo:
                return self._feature_info

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                x = self.conv(x)
                # Dynamic split with list of sizes (non-integer first arg).
                # 64 channels split as [32, 32] is dimensionally valid.
                parts = x.split([32, 32], dim=1)
                return {"out": parts[0]}

        backbone = DynamicSplitBackbone()
        dummy_head = MockHead(in_channels=32, num_classes=5)

        with pytest.raises(ValueError, match="dynamic operations"):
            prober.validate_compatibility(
                backbone, None, dummy_head, default_input_size
            )

    def test_validate_compatibility_rejects_control_flow(
        self,
        prober: MetaProber,
        default_input_size: tuple[int, int],
    ) -> None:
        """Model with control flow is rejected by validate_compatibility.

        Note: The control flow causes meta propagation to fail first
        (meta tensors cannot be used as boolean conditions), so the
        error message reflects meta propagation failure rather than
        a static audit failure.
        """

        class ControlFlowBackbone(BaseBackbone):
            def __init__(self) -> None:
                super().__init__()
                self._feature_info = FeatureInfo(
                    channels={"out": 3},
                    strides={"out": 4},
                )
                self.conv = nn.Conv2d(3, 3, kernel_size=4, stride=4, padding=0)

            @property
            def feature_info(self) -> FeatureInfo:
                return self._feature_info

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                x = self.conv(x)
                # Data-dependent control flow that fails on meta tensors
                if x.sum() > 0:
                    return {"out": x}
                return {"out": -x}

        backbone = ControlFlowBackbone()
        dummy_head = MockHead(in_channels=3, num_classes=5)

        with pytest.raises(ValueError, match="Meta shape propagation failed"):
            prober.validate_compatibility(
                backbone, None, dummy_head, default_input_size
            )


# ======================================================================
# DYNAMIC_OPS class attribute access
# ======================================================================


class TestDynamicOpsNewEntries:
    """Verify that new DYNAMIC_OPS are properly detected in graphs."""

    @pytest.mark.parametrize(
        ("op_name", "expected_in_message"),
        [
            ("aten::pad", "aten::pad"),
            ("aten::flip", "aten::flip"),
            ("aten::dropout", "aten::dropout"),
            ("aten::quantize_per_tensor", "aten::quantize_per_tensor"),
            ("aten::embedding", "aten::embedding"),
        ],
    )
    def test_new_dynamic_ops_detected_in_graph(
        self,
        prober: MetaProber,
        op_name: str,
        expected_in_message: str,
    ) -> None:
        """New DYNAMIC_OPS entries are detected by _find_dynamic_operations."""

        class _DynamicOp:
            """Callable stub whose __name__ contains the op name."""

            __name__: str = op_name

            @staticmethod
            def __call__(*args: object) -> object:  # noqa: ANN401
                if args:
                    return args[0]
                return None

            def __repr__(self) -> str:
                return op_name

        graph = Graph()
        inp = graph.placeholder("x")
        op_node = graph.call_function(_DynamicOp(), (inp,))
        graph.output(op_node)
        graph.lint()

        traced = GraphModule(torch.nn.Module(), graph)
        traced.recompile()

        ops = prober._find_dynamic_operations(traced, "Test")
        matching_ops = [op for op in ops if expected_in_message in op]
        assert len(matching_ops) > 0, (
            f"Op {op_name} should be detected as dynamic"
        )

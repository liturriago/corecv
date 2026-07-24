"""Tests for dynamic Backbone -> Neck -> Head wiring.

Covers:
1. Meta device shape propagation (zero-VRAM) for all head types.
2. Non-square input resolutions (480x640) to catch H/W inversion bugs.
3. Gradient flow verification (loss.backward() without NaNs).
4. CoreObjectDetector end-to-end validation on meta device.
5. Classification, segmentation, and detection pipeline combinations.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from corecv.core.contract import BaseBackbone, FeatureInfo
from corecv.engine.validator import MetaProber
from corecv.models.detector import CoreObjectDetector
from corecv.models.heads.classification import LinearClassificationHead
from corecv.models.heads.detection import (
    DecoupledAnchorFreeHead,
    QueryDetectionHead,
)
from corecv.models.heads.segmentation import (
    ASPPDecoder,
    ResUNetDecoder,
)
from corecv.models.necks.fpn import FPN
from corecv.models.necks.panet import PANet

# ======================================================================
# Mock backbone producing a list of tensors (stride-ascending order)
# ======================================================================


class MockListBackbone(BaseBackbone):
    """Mock backbone that returns feature maps as a list of tensors.

    Real backbones (ResNet, MobileNetV3, etc.) output their intermediate
    feature maps as a list/tuple ordered from finest (smallest stride) to
    coarsest (largest stride).  This mock replicates that contract for
    testing neck and head wiring without a full backbone.

    Each level is computed via a strided convolution applied to the input,
    so spatial dimensions follow ``input_size // stride`` exactly.

    Attributes:
        _feature_info: FeatureInfo describing channels and strides.
        convs: Strided convolutions, one per feature level.
        sorted_levels: Level names sorted by stride (ascending).
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
        level_names: list[str] = [f"level{s}" for s in out_strides]
        self._feature_info = FeatureInfo(
            channels=dict(zip(level_names, out_channels, strict=True)),
            strides=dict(zip(level_names, out_strides, strict=True)),
        )
        self.sorted_levels: list[str] = level_names

        self.convs: nn.ModuleList = nn.ModuleList()
        for c, s in zip(out_channels, out_strides, strict=True):
            self.convs.append(
                nn.Conv2d(3, c, kernel_size=s, stride=s, padding=0)
            )

    @property
    def feature_info(self) -> FeatureInfo:
        """Return the feature metadata for this backbone."""
        return self._feature_info

    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward pass producing multi-scale feature list.

        Args:
            x: Input tensor ``(N, 3, H, W)``.

        Returns:
            List of feature tensors ordered from finest to coarsest
            resolution (ascending stride).
        """
        return [conv(x) for conv in self.convs]


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def prober() -> MetaProber:
    """Fixture providing a fresh MetaProber instance."""
    return MetaProber()


@pytest.fixture
def square_input() -> tuple[int, int]:
    """Standard square input resolution ``(224, 224)``."""
    return 224, 224


@pytest.fixture
def non_square_input() -> tuple[int, int]:
    """Non-square input resolution ``(480, 640)``.

    Used to catch H/W indexing inversion bugs.
    """
    return 480, 640


@pytest.fixture
def backbone_3level() -> MockListBackbone:
    """Three-level backbone with channels ``(64, 128, 256)`` and strides ``(4, 8, 16)``."""
    return MockListBackbone(out_channels=(64, 128, 256), out_strides=(4, 8, 16))


@pytest.fixture
def backbone_4level() -> MockListBackbone:
    """Four-level backbone with channels ``(64, 128, 256, 512)`` and strides ``(4, 8, 16, 32)``."""
    return MockListBackbone(out_channels=(64, 128, 256, 512), out_strides=(4, 8, 16, 32))


@pytest.fixture
def backbone_2level() -> MockListBackbone:
    """Two-level backbone with channels ``(64, 128)`` and strides ``(4, 8)``."""
    return MockListBackbone(out_channels=(64, 128), out_strides=(4, 8))


@pytest.fixture
def fpn_256(backbone_3level: MockListBackbone) -> FPN:
    """FPN neck with 256 output channels matching the 3-level backbone."""
    return FPN(feature_info=backbone_3level.feature_info, out_channels=256)


@pytest.fixture
def panet_256(backbone_3level: MockListBackbone) -> PANet:
    """PANet neck with 256 output channels matching the 3-level backbone."""
    return PANet(feature_info=backbone_3level.feature_info, out_channels=256)


def _neck_feature_info(
    feature_info: FeatureInfo,
    out_channels: int,
) -> FeatureInfo:
    """Build a ``FeatureInfo`` matching a neck's uniform output channels.

    After a neck like FPN or PANet, all feature levels have the same
    channel count (``out_channels``).  This helper creates a compatible
    ``FeatureInfo`` for downstream heads.

    Args:
        feature_info: The original backbone ``FeatureInfo``.
        out_channels: The neck's output channel count.

    Returns:
        A new ``FeatureInfo`` with all levels set to ``out_channels``
        but preserving the original strides.
    """
    return FeatureInfo(
        channels=dict.fromkeys(feature_info.channels, out_channels),
        strides=dict(feature_info.strides),
    )


# ======================================================================
# Test helpers
# ======================================================================


def _check_finite_gradients(model: nn.Module, batch_size: int = 2) -> None:
    """Verify all parameters have finite gradients after a backward pass.

    Performs a single forward-backward step using random input and a
    dummy loss (sum of all output elements).

    Args:
        model: The model to test.
        batch_size: Batch size for the input tensor.  Uses ``2`` by
            default to avoid BatchNorm issues with single-sample batches.

    Raises:
        AssertionError: If any parameter gradient contains NaN or Inf.
    """
    model.train()
    x: Tensor = torch.randn(batch_size, 3, 64, 64)
    output = model(x)

    if isinstance(output, dict):
        flat_vals: list[Tensor] = []
        for v in output.values():
            if isinstance(v, Tensor):
                flat_vals.append(v.sum())
            elif isinstance(v, (list, tuple)):
                flat_vals.extend(vi.sum() for vi in v)
        loss = torch.stack(flat_vals).sum() if flat_vals else torch.tensor(0.0)
    elif isinstance(output, (list, tuple)):
        loss = torch.stack([o.sum() for o in output]).sum()
    else:
        loss = output.sum()

    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        msg = f"Gradient for {name} contains NaN or Inf."
        assert torch.isfinite(param.grad).all(), msg


def _propagate_meta(
    backbone: nn.Module,
    neck: nn.Module | None,
    head: nn.Module,
    input_size: tuple[int, int],
) -> dict[str, object]:
    """Manually propagate shapes on meta device (no MetaProber dependency).

    Moves all modules to ``device='meta'`` before forward to enable
    zero-VRAM shape propagation even for modules whose parameters
    would normally reside on CPU (e.g. ``nn.Linear``).

    Args:
        backbone: Backbone module.
        neck: Optional neck module.
        head: Head module.
        input_size: Input image size (H, W).

    Returns:
        Dictionary with keys ``"backbone_output"``, ``"neck_output"``
        (if neck is not ``None``), and ``"head_output"`` containing
        shape tuples or nested shape structures.
    """
    h, w = input_size
    device = torch.device("meta")

    # Move all modules to meta device for zero-VRAM shape propagation.
    backbone.to(device)
    if neck is not None:
        neck.to(device)
    head.to(device)

    shapes: dict[str, object] = {}
    with torch.no_grad():
        dummy = torch.randn(1, 3, h, w, device=device)
        backbone_out = backbone(dummy)
        shapes["backbone_output"] = _extract_shapes(backbone_out)

        neck_input = backbone_out
        if neck is not None:
            neck_out = neck(neck_input)
            shapes["neck_output"] = _extract_shapes(neck_out)
            neck_input = neck_out

        head_out = head(neck_input)
    shapes["head_output"] = _extract_shapes(head_out)

    return shapes


def _extract_shapes(output: object) -> tuple | dict | str:
    """Extract shapes from model output (tensor, tuple, list, or dict).

    Args:
        output: Model output of any supported type.

    Returns:
        Shape information preserving the output structure.
    """
    if isinstance(output, Tensor):
        return tuple(output.shape)
    if isinstance(output, (tuple, list)):
        return tuple(_extract_shapes(o) for o in output)
    if isinstance(output, dict):
        return {k: _extract_shapes(v) for k, v in output.items()}
    return str(type(output))


# ======================================================================
# Meta device shape propagation tests
# ======================================================================


class TestMetaDeviceWiring:
    """Verify Backbone -> Neck -> Head wiring on device='meta'.

    All tests use zero-VRAM meta tensors to check shape propagation
    without allocating real memory.
    """

    def test_classification_backbone_to_head(
        self,
        backbone_3level: MockListBackbone,
        square_input: tuple[int, int],
    ) -> None:
        """Backbone -> LinearClassificationHead propagates shapes correctly."""
        head = LinearClassificationHead(
            feature_info=backbone_3level.feature_info, num_classes=10,
        )
        shapes = _propagate_meta(backbone_3level, None, head, square_input)

        backbone_out = shapes["backbone_output"]
        assert isinstance(backbone_out, tuple)
        assert len(backbone_out) == 3

        head_out = shapes["head_output"]
        assert isinstance(head_out, tuple)
        assert head_out == (1, 10)

    def test_classification_via_prober(
        self,
        prober: MetaProber,
        backbone_3level: MockListBackbone,
        square_input: tuple[int, int],
    ) -> None:
        """MetaProber validates classification pipeline successfully."""
        head = LinearClassificationHead(
            feature_info=backbone_3level.feature_info, num_classes=10,
        )
        result = prober.validate_compatibility(
            backbone_3level, None, head, square_input,
        )
        assert result is True

    def test_segmentation_fpn_resunet(
        self,
        backbone_3level: MockListBackbone,
        fpn_256: FPN,
        square_input: tuple[int, int],
    ) -> None:
        """Backbone -> FPN -> ResUNetDecoder propagates shapes on meta."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = ResUNetDecoder(
            feature_info=neck_fi,
            out_channels=256,
            num_classes=21,
        )
        shapes = _propagate_meta(backbone_3level, fpn_256, head, square_input)

        neck_out = shapes["neck_output"]
        assert isinstance(neck_out, tuple)
        assert len(neck_out) == 3
        for level_shape in neck_out:
            assert isinstance(level_shape, tuple)
            assert level_shape[1] == 256

        head_out = shapes["head_output"]
        assert isinstance(head_out, tuple)
        assert head_out[1] == 21
        assert head_out[2] == 56

    def test_segmentation_panet_aspp(
        self,
        backbone_3level: MockListBackbone,
        panet_256: PANet,
        square_input: tuple[int, int],
    ) -> None:
        """Backbone -> PANet -> ASPPDecoder propagates shapes on meta."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = ASPPDecoder(
            feature_info=neck_fi,
            out_channels=256,
            num_classes=21,
        )
        shapes = _propagate_meta(backbone_3level, panet_256, head, square_input)

        neck_out = shapes["neck_output"]
        assert isinstance(neck_out, tuple)
        assert len(neck_out) == 3
        for level_shape in neck_out:
            assert isinstance(level_shape, tuple)
            assert level_shape[1] == 256

        head_out = shapes["head_output"]
        assert isinstance(head_out, tuple)
        assert head_out[1] == 21

    def test_detection_fpn_anchor_free(
        self,
        backbone_3level: MockListBackbone,
        fpn_256: FPN,
        square_input: tuple[int, int],
    ) -> None:
        """Backbone -> FPN -> DecoupledAnchorFreeHead on meta."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = DecoupledAnchorFreeHead(
            feature_info=neck_fi,
            num_classes=80,
            feat_channels=256,
            num_convs=2,
        )
        shapes = _propagate_meta(backbone_3level, fpn_256, head, square_input)

        neck_out = shapes["neck_output"]
        assert isinstance(neck_out, tuple)
        assert len(neck_out) == 3

        head_out = shapes["head_output"]
        assert isinstance(head_out, dict)
        assert "cls_logits" in head_out
        assert "reg_pred" in head_out
        assert "centerness" in head_out

        cls_shapes = head_out["cls_logits"]
        assert isinstance(cls_shapes, tuple) and len(cls_shapes) == 3
        for cls_shape in cls_shapes:
            assert isinstance(cls_shape, tuple)
            assert cls_shape[1] == 80

    def test_detection_panet_query(
        self,
        backbone_3level: MockListBackbone,
        panet_256: PANet,
        square_input: tuple[int, int],
    ) -> None:
        """Backbone -> PANet -> QueryDetectionHead on meta."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = QueryDetectionHead(
            feature_info=neck_fi,
            num_classes=80,
            d_model=64,
            num_queries=10,
            num_decoder_layers=1,
            num_heads=4,
        )
        shapes = _propagate_meta(backbone_3level, panet_256, head, square_input)

        neck_out = shapes["neck_output"]
        assert isinstance(neck_out, tuple)
        assert len(neck_out) == 3

        head_out = shapes["head_output"]
        assert isinstance(head_out, dict)
        assert "cls_logits" in head_out
        assert "pred_boxes" in head_out

        cls_shape = head_out["cls_logits"]
        assert isinstance(cls_shape, tuple)
        assert cls_shape == (1, 10, 80)

        box_shape = head_out["pred_boxes"]
        assert isinstance(box_shape, tuple)
        assert box_shape == (1, 10, 4)

    def test_segmentation_4level_fpn_resunet(
        self,
        backbone_4level: MockListBackbone,
        square_input: tuple[int, int],
    ) -> None:
        """4-level backbone with FPN + ResUNetDecoder works on meta."""
        neck = FPN(feature_info=backbone_4level.feature_info, out_channels=128)
        neck_fi = _neck_feature_info(backbone_4level.feature_info, 128)
        head = ResUNetDecoder(
            feature_info=neck_fi,
            out_channels=128,
            num_classes=10,
        )
        shapes = _propagate_meta(backbone_4level, neck, head, square_input)

        neck_out = shapes["neck_output"]
        assert isinstance(neck_out, tuple) and len(neck_out) == 4

        head_out = shapes["head_output"]
        assert isinstance(head_out, tuple)
        assert head_out[1] == 10

    def test_detection_4level_panet_anchor_free(
        self,
        backbone_4level: MockListBackbone,
        square_input: tuple[int, int],
    ) -> None:
        """4-level backbone with PANet + DecoupledAnchorFreeHead on meta."""
        neck = PANet(feature_info=backbone_4level.feature_info, out_channels=128)
        neck_fi = _neck_feature_info(backbone_4level.feature_info, 128)
        head = DecoupledAnchorFreeHead(
            feature_info=neck_fi,
            num_classes=80,
            feat_channels=128,
            num_convs=2,
        )
        shapes = _propagate_meta(backbone_4level, neck, head, square_input)

        neck_out = shapes["neck_output"]
        assert isinstance(neck_out, tuple) and len(neck_out) == 4

        head_out = shapes["head_output"]
        assert isinstance(head_out, dict)
        assert len(head_out["cls_logits"]) == 4


# ======================================================================
# Non-square input resolution tests
# ======================================================================


class TestNonSquareResolution:
    """Verify correct shape propagation for non-square inputs (480x640).

    This is critical for catching H/W indexing inversion bugs where
    height and width are accidentally swapped.
    """

    def test_classification_non_square(
        self,
        backbone_3level: MockListBackbone,
        non_square_input: tuple[int, int],
    ) -> None:
        """Classification head works with 480x640 input."""
        head = LinearClassificationHead(
            feature_info=backbone_3level.feature_info, num_classes=10,
        )
        shapes = _propagate_meta(backbone_3level, None, head, non_square_input)

        backbone_out = shapes["backbone_output"]
        assert isinstance(backbone_out, tuple)

        head_out = shapes["head_output"]
        assert isinstance(head_out, tuple)
        assert head_out == (1, 10)

    def test_segmentation_fpn_resunet_non_square(
        self,
        backbone_3level: MockListBackbone,
        fpn_256: FPN,
        non_square_input: tuple[int, int],
    ) -> None:
        """Seg pipeline handles 480x640 without H/W inversion."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = ResUNetDecoder(
            feature_info=neck_fi,
            out_channels=256,
            num_classes=21,
        )
        shapes = _propagate_meta(backbone_3level, fpn_256, head, non_square_input)

        neck_out = shapes["neck_output"]
        assert isinstance(neck_out, tuple)

        H, W = non_square_input
        finest_shape = neck_out[0]
        assert isinstance(finest_shape, tuple)
        assert finest_shape[2] == H // 4
        assert finest_shape[3] == W // 4

        coarsest_shape = neck_out[2]
        assert isinstance(coarsest_shape, tuple)
        assert coarsest_shape[2] == H // 16
        assert coarsest_shape[3] == W // 16

    def test_segmentation_panet_aspp_non_square(
        self,
        backbone_3level: MockListBackbone,
        panet_256: PANet,
        non_square_input: tuple[int, int],
    ) -> None:
        """PANet + ASPPDecoder handles 480x640 without H/W inversion."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = ASPPDecoder(
            feature_info=neck_fi,
            out_channels=256,
            num_classes=21,
        )
        shapes = _propagate_meta(backbone_3level, panet_256, head, non_square_input)

        neck_out = shapes["neck_output"]
        assert isinstance(neck_out, tuple)

        H, W = non_square_input
        finest_shape = neck_out[0]
        assert isinstance(finest_shape, tuple)
        assert finest_shape[2] == H // 4
        assert finest_shape[3] == W // 4

    def test_detection_fpn_anchor_free_non_square(
        self,
        backbone_3level: MockListBackbone,
        fpn_256: FPN,
        non_square_input: tuple[int, int],
    ) -> None:
        """FPN + DecoupledAnchorFreeHead handles 480x640 input."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = DecoupledAnchorFreeHead(
            feature_info=neck_fi,
            num_classes=80,
            feat_channels=256,
            num_convs=2,
        )
        shapes = _propagate_meta(backbone_3level, fpn_256, head, non_square_input)

        neck_out = shapes["neck_output"]
        assert isinstance(neck_out, tuple)

        head_out = shapes["head_output"]
        assert isinstance(head_out, dict)
        cls_shapes = head_out["cls_logits"]
        assert isinstance(cls_shapes, tuple)
        H, W = non_square_input
        for i, cls_shape in enumerate(cls_shapes):
            assert isinstance(cls_shape, tuple)
            stride = 4 * (2**i)
            assert cls_shape[2] == H // stride
            assert cls_shape[3] == W // stride

    def test_detection_panet_query_non_square(
        self,
        backbone_3level: MockListBackbone,
        panet_256: PANet,
        non_square_input: tuple[int, int],
    ) -> None:
        """PANet + QueryDetectionHead handles 480x640 input."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = QueryDetectionHead(
            feature_info=neck_fi,
            num_classes=80,
            d_model=64,
            num_queries=10,
            num_decoder_layers=1,
            num_heads=4,
        )
        shapes = _propagate_meta(backbone_3level, panet_256, head, non_square_input)

        neck_out = shapes["neck_output"]
        assert isinstance(neck_out, tuple)

        head_out = shapes["head_output"]
        assert isinstance(head_out, dict)
        assert head_out["cls_logits"] == (1, 10, 80)
        assert head_out["pred_boxes"] == (1, 10, 4)

    def test_hw_inversion_guards_non_square(
        self,
        non_square_input: tuple[int, int],
    ) -> None:
        """Verify H/W ordering is correct for 480x640 input."""
        H, W = non_square_input

        class HWCheckBackbone(BaseBackbone):
            """Backbone that verifies H/W ordering."""

            def __init__(self) -> None:
                """Initialise with single strided conv."""
                super().__init__()
                self._feature_info = FeatureInfo(
                    channels={"out": 16},
                    strides={"out": 4},
                )
                self.conv = nn.Conv2d(3, 16, kernel_size=4, stride=4, padding=0)

            @property
            def feature_info(self) -> FeatureInfo:
                """Return feature metadata."""
                return self._feature_info

            def forward(self, x: Tensor) -> dict[str, Tensor]:
                """Forward returning dict with single level."""
                return {"out": self.conv(x)}

        backbone = HWCheckBackbone()
        head = LinearClassificationHead(
            feature_info=backbone.feature_info, num_classes=2,
        )
        shapes = _propagate_meta(backbone, None, head, non_square_input)

        out_shape = shapes["backbone_output"]
        assert isinstance(out_shape, dict)
        out = out_shape["out"]
        assert isinstance(out, tuple)
        assert out[2] == H // 4, (
            f"Expected height {H//4}, got {out[2]} -- possible H/W inversion"
        )
        assert out[3] == W // 4, (
            f"Expected width {W//4}, got {out[3]} -- possible H/W inversion"
        )


# ======================================================================
# Gradient flow tests
# ======================================================================


class TestGradientFlow:
    """Verify loss.backward() produces finite gradients for all pipelines.

    Each test creates the full pipeline, performs a forward-backward pass
    with ``batch_size=2`` (to avoid BatchNorm issues with single-sample
    batches), and checks that all parameter gradients are finite (no NaN
    or Inf).
    """

    def test_classification_gradient_flow(
        self,
        backbone_3level: MockListBackbone,
    ) -> None:
        """Classification pipeline has finite gradients."""
        head = LinearClassificationHead(
            feature_info=backbone_3level.feature_info, num_classes=10,
        )

        class ClsPipeline(nn.Module):
            """Backbone -> LinearClassificationHead pipeline."""

            def __init__(
                self,
                backbone: nn.Module,
                head: nn.Module,
            ) -> None:
                """Initialise.

                Args:
                    backbone: Backbone module.
                    head: Classification head.
                """
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x: Tensor) -> Tensor:
                """Forward pass.

                Args:
                    x: Input tensor.

                Returns:
                    Classification logits.
                """
                features = self.backbone(x)
                return self.head(features)

        model = ClsPipeline(backbone_3level, head)
        _check_finite_gradients(model)

    def test_segmentation_fpn_resunet_gradient_flow(
        self,
        backbone_3level: MockListBackbone,
        fpn_256: FPN,
    ) -> None:
        """FPN + ResUNetDecoder has finite gradients."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = ResUNetDecoder(
            feature_info=neck_fi,
            out_channels=256,
            num_classes=21,
        )

        class SegPipeline(nn.Module):
            """Backbone -> FPN -> ResUNetDecoder pipeline."""

            def __init__(
                self,
                backbone: nn.Module,
                neck: nn.Module,
                head: nn.Module,
            ) -> None:
                """Initialise.

                Args:
                    backbone: Backbone module.
                    neck: Neck module.
                    head: Segmentation head.
                """
                super().__init__()
                self.backbone = backbone
                self.neck = neck
                self.head = head

            def forward(self, x: Tensor) -> Tensor:
                """Forward pass.

                Args:
                    x: Input tensor.

                Returns:
                    Segmentation logits.
                """
                features = self.backbone(x)
                features = self.neck(features)
                return self.head(features)

        model = SegPipeline(backbone_3level, fpn_256, head)
        _check_finite_gradients(model)

    def test_segmentation_panet_aspp_gradient_flow(
        self,
        backbone_3level: MockListBackbone,
        panet_256: PANet,
    ) -> None:
        """PANet + ASPPDecoder has finite gradients."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = ASPPDecoder(
            feature_info=neck_fi,
            out_channels=256,
            num_classes=21,
        )

        class SegPipeline(nn.Module):
            """Backbone -> PANet -> ASPPDecoder pipeline."""

            def __init__(
                self,
                backbone: nn.Module,
                neck: nn.Module,
                head: nn.Module,
            ) -> None:
                """Initialise.

                Args:
                    backbone: Backbone module.
                    neck: Neck module.
                    head: Segmentation head.
                """
                super().__init__()
                self.backbone = backbone
                self.neck = neck
                self.head = head

            def forward(self, x: Tensor) -> Tensor:
                """Forward pass.

                Args:
                    x: Input tensor.

                Returns:
                    Segmentation logits.
                """
                features = self.backbone(x)
                features = self.neck(features)
                return self.head(features)

        model = SegPipeline(backbone_3level, panet_256, head)
        _check_finite_gradients(model)

    def test_detection_fpn_anchor_free_gradient_flow(
        self,
        backbone_3level: MockListBackbone,
        fpn_256: FPN,
    ) -> None:
        """FPN + DecoupledAnchorFreeHead has finite gradients."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = DecoupledAnchorFreeHead(
            feature_info=neck_fi,
            num_classes=80,
            feat_channels=256,
            num_convs=2,
        )

        class DetPipeline(nn.Module):
            """Backbone -> FPN -> DecoupledAnchorFreeHead pipeline."""

            def __init__(
                self,
                backbone: nn.Module,
                neck: nn.Module,
                head: nn.Module,
            ) -> None:
                """Initialise.

                Args:
                    backbone: Backbone module.
                    neck: Neck module.
                    head: Detection head.
                """
                super().__init__()
                self.backbone = backbone
                self.neck = neck
                self.head = head

            def forward(self, x: Tensor) -> dict[str, list[Tensor]]:
                """Forward pass.

                Args:
                    x: Input tensor.

                Returns:
                    Detection head output dict.
                """
                features = self.backbone(x)
                features = self.neck(features)
                return self.head(features)

        model = DetPipeline(backbone_3level, fpn_256, head)
        _check_finite_gradients(model)

    def test_detection_panet_query_gradient_flow(
        self,
        backbone_3level: MockListBackbone,
        panet_256: PANet,
    ) -> None:
        """PANet + QueryDetectionHead has finite gradients."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = QueryDetectionHead(
            feature_info=neck_fi,
            num_classes=80,
            d_model=64,
            num_queries=10,
            num_decoder_layers=1,
            num_heads=4,
        )

        class DetPipeline(nn.Module):
            """Backbone -> PANet -> QueryDetectionHead pipeline."""

            def __init__(
                self,
                backbone: nn.Module,
                neck: nn.Module,
                head: nn.Module,
            ) -> None:
                """Initialise.

                Args:
                    backbone: Backbone module.
                    neck: Neck module.
                    head: Detection head.
                """
                super().__init__()
                self.backbone = backbone
                self.neck = neck
                self.head = head

            def forward(self, x: Tensor) -> dict[str, Tensor | list[Tensor]]:
                """Forward pass.

                Args:
                    x: Input tensor.

                Returns:
                    Detection head output dict.
                """
                features = self.backbone(x)
                features = self.neck(features)
                return self.head(features)

        model = DetPipeline(backbone_3level, panet_256, head)
        _check_finite_gradients(model)


# ======================================================================
# CoreObjectDetector end-to-end tests
# ======================================================================


class TestCoreObjectDetector:
    """End-to-end tests for CoreObjectDetector on meta device."""

    def test_detector_init(
        self,
        backbone_3level: MockListBackbone,
        fpn_256: FPN,
    ) -> None:
        """CoreObjectDetector initialises without errors."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = DecoupledAnchorFreeHead(
            feature_info=neck_fi,
            num_classes=80,
            feat_channels=256,
            num_convs=2,
        )
        detector = CoreObjectDetector(
            backbone=backbone_3level, neck=fpn_256, head=head,
        )
        assert isinstance(detector, CoreObjectDetector)
        assert detector.backbone is backbone_3level
        assert detector.neck is fpn_256
        assert detector.head is head

    def test_detector_meta_propagation(
        self,
        backbone_3level: MockListBackbone,
        fpn_256: FPN,
        square_input: tuple[int, int],
    ) -> None:
        """CoreObjectDetector propagates shapes on meta device."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = DecoupledAnchorFreeHead(
            feature_info=neck_fi,
            num_classes=80,
            feat_channels=256,
            num_convs=2,
        )
        detector = CoreObjectDetector(
            backbone=backbone_3level, neck=fpn_256, head=head,
        )

        h, w = square_input
        device = torch.device("meta")
        detector.to(device)
        with torch.no_grad():
            out = detector(torch.randn(1, 3, h, w, device=device))

        assert isinstance(out, dict)
        assert "cls_logits" in out
        assert "reg_pred" in out
        assert "centerness" in out

        cls_logits = out["cls_logits"]
        assert isinstance(cls_logits, (list, tuple))
        assert len(cls_logits) == 3

        for cls_tensor in cls_logits:
            assert cls_tensor.shape[1] == 80

    def test_detector_without_neck(
        self,
        backbone_3level: MockListBackbone,
        square_input: tuple[int, int],
    ) -> None:
        """CoreObjectDetector works without a neck."""
        head = DecoupledAnchorFreeHead(
            feature_info=backbone_3level.feature_info,
            num_classes=80,
            feat_channels=256,
            num_convs=2,
        )
        detector = CoreObjectDetector(
            backbone=backbone_3level, neck=None, head=head,
        )

        h, w = square_input
        device = torch.device("meta")
        detector.to(device)
        with torch.no_grad():
            out = detector(torch.randn(1, 3, h, w, device=device))

        assert isinstance(out, dict)
        assert "cls_logits" in out

    def test_detector_with_query_detection_head(
        self,
        backbone_3level: MockListBackbone,
        panet_256: PANet,
        square_input: tuple[int, int],
    ) -> None:
        """CoreObjectDetector with QueryDetectionHead on meta."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = QueryDetectionHead(
            feature_info=neck_fi,
            num_classes=80,
            d_model=64,
            num_queries=10,
            num_decoder_layers=1,
            num_heads=4,
        )
        detector = CoreObjectDetector(
            backbone=backbone_3level, neck=panet_256, head=head,
        )

        h, w = square_input
        device = torch.device("meta")
        detector.to(device)
        with torch.no_grad():
            out = detector(torch.randn(1, 3, h, w, device=device))

        assert isinstance(out, dict)
        assert "cls_logits" in out
        assert "pred_boxes" in out
        assert out["cls_logits"].shape == (1, 10, 80)
        assert out["pred_boxes"].shape == (1, 10, 4)

    def test_detector_gradient_flow(
        self,
        backbone_3level: MockListBackbone,
        fpn_256: FPN,
    ) -> None:
        """CoreObjectDetector has finite gradients."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = DecoupledAnchorFreeHead(
            feature_info=neck_fi,
            num_classes=80,
            feat_channels=256,
            num_convs=2,
        )
        detector = CoreObjectDetector(
            backbone=backbone_3level, neck=fpn_256, head=head,
        )
        _check_finite_gradients(detector)

    def test_detector_non_square(
        self,
        backbone_3level: MockListBackbone,
        fpn_256: FPN,
        non_square_input: tuple[int, int],
    ) -> None:
        """CoreObjectDetector handles 480x640 input on meta."""
        neck_fi = _neck_feature_info(backbone_3level.feature_info, 256)
        head = DecoupledAnchorFreeHead(
            feature_info=neck_fi,
            num_classes=80,
            feat_channels=256,
            num_convs=2,
        )
        detector = CoreObjectDetector(
            backbone=backbone_3level, neck=fpn_256, head=head,
        )

        H, W = non_square_input
        device = torch.device("meta")
        detector.to(device)
        with torch.no_grad():
            out = detector(torch.randn(1, 3, H, W, device=device))

        assert isinstance(out, dict)
        cls_logits = out["cls_logits"]
        assert isinstance(cls_logits, (list, tuple))
        assert cls_logits[0].shape[2] == H // 4
        assert cls_logits[0].shape[3] == W // 4
        assert cls_logits[0].shape[2] < cls_logits[0].shape[3]


# ======================================================================
# Edge case: backbone with insufficient levels
# ======================================================================


class TestBackboneLevelEdgeCases:
    """Test behaviour with backbones that have too few or too many levels."""

    def test_resunet_requires_min_2_levels(
        self,
    ) -> None:
        """ResUNetDecoder raises ValueError for single-level backbone."""
        fi = FeatureInfo(channels={"out": 64}, strides={"out": 4})
        with pytest.raises(ValueError, match="at least 2"):
            ResUNetDecoder(feature_info=fi, out_channels=128, num_classes=10)

    def test_aspp_requires_min_2_levels(
        self,
    ) -> None:
        """ASPPDecoder raises ValueError for single-level backbone."""
        fi = FeatureInfo(channels={"out": 64}, strides={"out": 4})
        with pytest.raises(ValueError, match="at least 2"):
            ASPPDecoder(feature_info=fi, out_channels=128, num_classes=10)

    def test_fpn_works_with_varying_levels(
        self,
        backbone_2level: MockListBackbone,
    ) -> None:
        """FPN works with 2-level backbone on meta."""
        neck = FPN(feature_info=backbone_2level.feature_info, out_channels=128)
        head = LinearClassificationHead(
            feature_info=backbone_2level.feature_info, num_classes=5,
        )
        shapes = _propagate_meta(backbone_2level, neck, head, (224, 224))
        neck_out = shapes["neck_output"]
        assert isinstance(neck_out, tuple) and len(neck_out) == 2

        head_out = shapes["head_output"]
        assert isinstance(head_out, tuple)
        assert head_out == (1, 5)

    def test_detection_no_neck(
        self,
        backbone_3level: MockListBackbone,
        square_input: tuple[int, int],
    ) -> None:
        """Detection head works directly on backbone output (no neck)."""
        head = DecoupledAnchorFreeHead(
            feature_info=backbone_3level.feature_info,
            num_classes=80,
            feat_channels=256,
            num_convs=2,
        )
        shapes = _propagate_meta(backbone_3level, None, head, square_input)
        head_out = shapes["head_output"]
        assert isinstance(head_out, dict)
        assert len(head_out["cls_logits"]) == 3

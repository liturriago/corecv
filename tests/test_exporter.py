"""Comprehensive tests for the CoreExporter end-to-end export pipeline.

Tests cover:
1. Classification, segmentation, and detection model export
2. TargetRewriter integration (edge rewrites, server no-rewrite, original intact)
3. MetaProber validation (pass, fail, fallback)
4. ONNX-specific (opset versions, dynamic axes, output names, checker)
5. ExecuTorch-specific (file creation, dynamic shapes, graph breaks)
6. Error handling (invalid target, opset, hardware, input shape)
7. Edge cases (batch_size=1, non-square, large input)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from corecv.core.contract import BaseBackbone, FeatureInfo
from corecv.engine import CoreExporter
from corecv.engine.exporter import (
    TORCH_EXPORT_AVAILABLE,
    ValidationResult,
    _infer_output_names,
    _ONNXCompatModel,
)
from corecv.models.heads.detection.decoupled_anchor_free import (
    DecoupledAnchorFreeHead,
)
from corecv.models.heads.segmentation.resunet_decoder import ResUNetDecoder
from corecv.models.necks.fpn import FPN

# ======================================================================
# Test model definitions — synthetic CoreCV-like models (no real weights)
# ======================================================================


class SimpleClassificationBackbone(BaseBackbone):
    """Minimal backbone producing a list of feature maps for classification.

    Each level is produced independently from the input via a strided conv
    with ``kernel_size == stride``, guaranteeing spatial sizes match the
    declared ``FeatureInfo`` strides.
    """

    def __init__(self, out_channels: int = 1280) -> None:
        """Initialise with independent strided convs per level.

        Args:
            out_channels: Channel count at the coarsest level.
        """
        super().__init__()
        self._feature_info = FeatureInfo(
            channels={
                "stride4": 64,
                "stride8": 128,
                "stride16": 256,
                "stride32": out_channels,
            },
            strides={"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32},
        )
        # Each level processes the input independently so spatial sizes
        # are exactly input_size // stride (padding=0, kernel=stride).
        self.conv4 = nn.Conv2d(3, 64, kernel_size=4, stride=4, padding=0)
        self.conv8 = nn.Conv2d(3, 128, kernel_size=8, stride=8, padding=0)
        self.conv16 = nn.Conv2d(3, 256, kernel_size=16, stride=16, padding=0)
        self.conv32 = nn.Conv2d(3, out_channels, kernel_size=32, stride=32, padding=0)

    @property
    def feature_info(self) -> FeatureInfo:
        """Return feature metadata."""
        return self._feature_info

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass producing four feature levels.

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            List of feature tensors (finest to coarsest).
        """
        return [
            self.conv4(x),
            self.conv8(x),
            self.conv16(x),
            self.conv32(x),
        ]


class SimpleSegmentationBackbone(BaseBackbone):
    """Minimal backbone producing 4 feature levels for segmentation.

    Each level is a strided conv from the input with
    ``kernel_size == stride`` for exact spatial match with metadata.
    """

    def __init__(self) -> None:
        """Initialise with independent strided convs per level."""
        super().__init__()
        self._feature_info = FeatureInfo(
            channels={
                "stride4": 32,
                "stride8": 64,
                "stride16": 128,
                "stride32": 256,
            },
            strides={"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32},
        )
        self.conv4 = nn.Conv2d(3, 32, kernel_size=4, stride=4, padding=0)
        self.conv8 = nn.Conv2d(3, 64, kernel_size=8, stride=8, padding=0)
        self.conv16 = nn.Conv2d(3, 128, kernel_size=16, stride=16, padding=0)
        self.conv32 = nn.Conv2d(3, 256, kernel_size=32, stride=32, padding=0)

    @property
    def feature_info(self) -> FeatureInfo:
        """Return feature metadata."""
        return self._feature_info

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            List of four feature tensors.
        """
        return [
            self.conv4(x),
            self.conv8(x),
            self.conv16(x),
            self.conv32(x),
        ]


class SimpleClassificationHead(nn.Module):
    """Minimal classification head: GAP + 1x1 Conv (meta-device-safe).

    Uses a 1x1 convolution (equivalent to ``nn.Linear`` on a 1x1 spatial
    map) to ensure compatibility with ``device='meta'`` shape propagation
    (``nn.Linear`` does not support meta device inputs).
    """

    def __init__(self, in_channels: int = 1280, num_classes: int = 1000) -> None:
        """Initialise head.

        Args:
            in_channels: Input channel count.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 1x1 conv instead of Linear for meta-device compatibility
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of feature tensors from backbone.

        Returns:
            Class logits ``(B, num_classes)``.
        """
        x = features[-1]  # coarsest level
        x = self.pool(x)  # (B, C, 1, 1)
        x = self.conv(x)  # (B, num_classes, 1, 1)
        return x.flatten(1)  # (B, num_classes)


class ClassificationModel(nn.Module):
    """Synthetic classification model: backbone + classification head.

    Produces a single ``(B, num_classes)`` tensor.
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """Initialise model.

        Args:
            num_classes: Number of output classes.
        """
        super().__init__()
        self.backbone = SimpleClassificationBackbone(out_channels=1280)
        self.head = SimpleClassificationHead(
            in_channels=1280, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            Logits ``(B, num_classes)``.
        """
        features = self.backbone(x)
        return self.head(features)


class SegmentationModel(nn.Module):
    """Synthetic segmentation model: backbone + FPN + ResUNetDecoder.

    Produces a ``(B, C, H, W)`` output.
    """

    def __init__(self, num_classes: int = 21) -> None:
        """Initialise segmentation model.

        Args:
            num_classes: Number of segmentation classes.
        """
        super().__init__()
        self.backbone = SimpleSegmentationBackbone()
        self._neck_channels = 128

        # FPN maps all backbone levels to common channel count.
        # The head must be initialised with the FPN channel metadata.
        fpn_feature_info = FeatureInfo(
            channels=dict.fromkeys(
                self.backbone.feature_info.channels, self._neck_channels
            ),
            strides=dict(self.backbone.feature_info.strides),
        )

        self.neck = FPN(
            feature_info=self.backbone.feature_info,
            out_channels=self._neck_channels,
        )

        self.head = ResUNetDecoder(
            feature_info=fpn_feature_info,
            out_channels=self._neck_channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing segmentation logits.

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            Segmentation logits ``(B, num_classes, H, W)``.
        """
        features = self.backbone(x)
        features = self.neck(features)
        return self.head(features)


class DetectionModel(nn.Module):
    """Synthetic detection model: backbone + FPN + DecoupledAnchorFreeHead.

    Produces a ``dict`` with lists of per-level tensors.
    """

    def __init__(self, num_classes: int = 80) -> None:
        """Initialise detection model.

        Args:
            num_classes: Number of detection classes.
        """
        super().__init__()
        self.backbone = SimpleSegmentationBackbone()
        self._neck_channels = 128

        # FPN maps all backbone levels to common channel count.
        # The detection head must be initialised with FPN metadata.
        fpn_feature_info = FeatureInfo(
            channels=dict.fromkeys(
                self.backbone.feature_info.channels, self._neck_channels
            ),
            strides=dict(self.backbone.feature_info.strides),
        )

        self.neck = FPN(
            feature_info=self.backbone.feature_info,
            out_channels=self._neck_channels,
        )

        self.head = DecoupledAnchorFreeHead(
            feature_info=fpn_feature_info,
            num_classes=num_classes,
            feat_channels=self._neck_channels,
            num_convs=2,
        )

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """Forward pass producing detection dict output.

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            Dict with ``cls_logits``, ``reg_pred``, ``centerness`` lists.
        """
        features = self.backbone(x)
        features = self.neck(features)
        return self.head(features)


# ======================================================================
# Model with GELU activation (for edge rewrite verification)
# ======================================================================


class GELUClassificationModel(nn.Module):
    """Classification model that uses nn.GELU — used to test edge rewrites.

    Will be rewritten (GELU -> ReLU) when ``target_hardware='edge'``.
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """Initialise model with GELU activations.

        Args:
            num_classes: Number of output classes.
        """
        super().__init__()
        self.backbone = SimpleClassificationBackbone(out_channels=1280)
        # Add a GELU activation after the coarsest level conv
        original_conv32 = self.backbone.conv32
        self.backbone.conv32 = nn.Sequential(
            original_conv32,
            nn.GELU(),
        )
        self.head = SimpleClassificationHead(
            in_channels=1280, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            Logits ``(B, num_classes)``.
        """
        features = self.backbone(x)
        return self.head(features)


# ======================================================================
# Incompatible model — uses dynamic operations that MetaProber rejects
# ======================================================================


class DynamicBackbone(BaseBackbone):
    """Backbone that uses a dynamic ``.split()`` call in its forward.

    MetaProber will detect ``tensor.split()`` with a non-integer
    (list) argument and flag it as an incompatible dynamic operation.
    """

    def __init__(self, out_channels: int = 1280) -> None:
        """Initialise.

        Args:
            out_channels: Channel count at the coarsest level.
        """
        super().__init__()
        self._feature_info = FeatureInfo(
            channels={
                "stride4": 64,
                "stride8": 128,
                "stride16": 256,
                "stride32": out_channels,
            },
            strides={"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32},
        )
        self.conv4 = nn.Conv2d(3, 64, kernel_size=4, stride=4, padding=0)
        self.conv8 = nn.Conv2d(3, 128, kernel_size=8, stride=8, padding=0)
        self.conv16 = nn.Conv2d(3, 256, kernel_size=16, stride=16, padding=0)
        self.conv32 = nn.Conv2d(3, out_channels, kernel_size=32, stride=32, padding=0)

    @property
    def feature_info(self) -> FeatureInfo:
        """Return feature metadata."""
        return self._feature_info

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass with a dynamic ``.split()`` that MetaProber flags.

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            List of feature tensors.
        """
        f4 = self.conv4(x)
        f8 = self.conv8(x)
        f16 = self.conv16(x)
        f32 = self.conv32(x)
        # Add a dynamic split that MetaProber detects
        # Using a list of split sizes (non-integer) is flagged as dynamic
        parts = f32.split([f32.shape[1] // 2, f32.shape[1] - f32.shape[1] // 2], dim=1)
        f32 = parts[0]
        return [f4, f8, f16, f32]


class IncompatibleModel(nn.Module):
    """Model whose backbone contains a dynamic operation.

    This should fail MetaProber validation.
    """

    def __init__(self, num_classes: int = 1000) -> None:
        """Initialise.

        Args:
            num_classes: Number of output classes.
        """
        super().__init__()
        self.backbone = DynamicBackbone(out_channels=1280)
        self.head = SimpleClassificationHead(
            in_channels=1280, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Logits.
        """
        features = self.backbone(x)
        return self.head(features)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def temp_dir() -> str:
    """Fixture providing a temporary directory for export outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)


@pytest.fixture
def classification_model() -> ClassificationModel:
    """Fixture providing a synthetic classification model."""
    return ClassificationModel(num_classes=1000).eval()


@pytest.fixture
def segmentation_model() -> SegmentationModel:
    """Fixture providing a synthetic segmentation model."""
    return SegmentationModel(num_classes=21).eval()


@pytest.fixture
def detection_model() -> DetectionModel:
    """Fixture providing a synthetic detection model."""
    return DetectionModel(num_classes=80).eval()


@pytest.fixture
def gelu_classification_model() -> GELUClassificationModel:
    """Fixture providing a classification model with GELU activation."""
    return GELUClassificationModel(num_classes=1000).eval()


@pytest.fixture
def incompatible_model() -> IncompatibleModel:
    """Fixture providing a model with dynamic operations."""
    return IncompatibleModel(num_classes=1000).eval()


@pytest.fixture
def exporter_config() -> dict:
    """Fixture providing a base exporter configuration."""
    return {
        "target": "both",
        "opset_version": 18,
        "target_hardware": "edge",
        "input_shape": (1, 3, 224, 224),
    }


# ======================================================================
# Helpers
# ======================================================================


def _onnx_check(onnx_path: str) -> None:
    """Validate an ONNX file with ``onnx.checker.check_model``.

    Args:
        onnx_path: Path to the ONNX file.
    """
    try:
        import onnx  # noqa: PLC0415

        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
    except ImportError:
        pytest.skip("onnx package not installed; skipping ONNX checker validation.")


def _executorch_check(pte_path: str) -> None:
    """Quick sanity check on an ExecuTorch .pte file.

    Verifies the file exists and has nonzero size.  Full validation
    requires the ``executorch`` package which is not always available.

    Args:
        pte_path: Path to the .pte file.
    """
    p = Path(pte_path)
    assert p.exists(), f"ExecuTorch file does not exist: {pte_path}"
    assert p.stat().st_size > 0, f"ExecuTorch file is empty: {pte_path}"


def _check_export_results(
    results: dict[str, str],
    expected_keys: list[str],
) -> None:
    """Shared assertions for export results.

    Args:
        results: Dict returned by ``CoreExporter.run_export()``.
        expected_keys: Expected keys in the results dict.
    """
    for key in expected_keys:
        assert key in results, f"Missing key '{key}' in export results"
        path = results[key]
        assert Path(path).exists(), f"File does not exist: {path}"
        assert Path(path).stat().st_size > 0, f"File is empty: {path}"

    if "onnx" in expected_keys:
        _onnx_check(results["onnx"])

    if "executorch" in expected_keys:
        if not TORCH_EXPORT_AVAILABLE:
            pytest.skip("torch.export not available; skipping ExecuTorch checks.")
        _executorch_check(results["executorch"])


# ======================================================================
# 1. Classification Model Export
# ======================================================================


class TestClassificationExport:
    """End-to-end export of a classification model."""

    def test_classification_export(
        self,
        classification_model: ClassificationModel,
        temp_dir: str,
    ) -> None:
        """Classification model exports to ONNX and ExecuTorch successfully."""
        exporter = CoreExporter(
            model=classification_model,
            target="both",
            opset_version=18,
            target_hardware="edge",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        results = exporter.run_export()

        _check_export_results(results, ["onnx", "executorch"])

    def test_classification_output_shape_preserved(
        self,
        classification_model: ClassificationModel,
    ) -> None:
        """Classification model output shape is (B, 1000)."""
        # Verify the model itself produces correct shape
        dummy = torch.randn(1, 3, 224, 224)
        output = classification_model(dummy)
        assert output.shape == (1, 1000), (
            f"Expected (1, 1000), got {output.shape}"
        )

    def test_classification_onnx_only(
        self,
        classification_model: ClassificationModel,
        temp_dir: str,
    ) -> None:
        """Classification model exports to ONNX only when target='onnx'."""
        exporter = CoreExporter(
            model=classification_model,
            target="onnx",
            opset_version=17,
            target_hardware="server",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        results = exporter.run_export()

        assert list(results.keys()) == ["onnx"]
        _check_export_results(results, ["onnx"])

    def test_classification_executorch_only(
        self,
        classification_model: ClassificationModel,
        temp_dir: str,
    ) -> None:
        """Classification model exports to ExecuTorch when target='executorch'."""
        if not TORCH_EXPORT_AVAILABLE:
            pytest.skip("torch.export not available.")

        exporter = CoreExporter(
            model=classification_model,
            target="executorch",
            opset_version=18,
            target_hardware="server",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        results = exporter.run_export()

        assert list(results.keys()) == ["executorch"]
        _check_export_results(results, ["executorch"])


# ======================================================================
# 2. Segmentation Model Export
# ======================================================================


class TestSegmentationExport:
    """End-to-end export of a segmentation model."""

    def test_segmentation_export(
        self,
        segmentation_model: SegmentationModel,
        temp_dir: str,
    ) -> None:
        """Segmentation model exports to ONNX and ExecuTorch successfully."""
        exporter = CoreExporter(
            model=segmentation_model,
            target="both",
            opset_version=18,
            target_hardware="server",
            input_shape=(1, 3, 512, 512),
            output_dir=temp_dir,
        )
        results = exporter.run_export()

        _check_export_results(results, ["onnx", "executorch"])

    def test_segmentation_output_shape_preserved(
        self,
        segmentation_model: SegmentationModel,
    ) -> None:
        """Segmentation model output shape is (B, C, H, W) with spatial dims."""
        dummy = torch.randn(1, 3, 512, 512)
        output = segmentation_model(dummy)
        # ResUNetDecoder upsamples to the finest feature level's resolution,
        # which is stride 4 from input: 512/4 = 128
        assert output.ndim == 4, f"Expected 4D output, got {output.ndim}D"
        assert output.shape[0] == 1, f"Expected batch 1, got {output.shape[0]}"
        assert output.shape[1] == 21, f"Expected 21 classes, got {output.shape[1]}"
        assert output.shape[2] > 0, "Height should be positive"
        assert output.shape[3] > 0, "Width should be positive"


# ======================================================================
# 3. Detection Model Export
# ======================================================================


class TestDetectionExport:
    """End-to-end export of a detection model with structured dict output."""

    def test_detection_export(
        self,
        detection_model: DetectionModel,
        temp_dir: str,
    ) -> None:
        """Detection model exports with dict output handled correctly."""
        exporter = CoreExporter(
            model=detection_model,
            target="both",
            opset_version=18,
            target_hardware="server",
            input_shape=(1, 3, 640, 640),
            output_dir=temp_dir,
        )
        results = exporter.run_export()

        _check_export_results(results, ["onnx", "executorch"])

    def test_detection_dict_output(
        self,
        detection_model: DetectionModel,
    ) -> None:
        """Detection model produces dict with lists per level."""
        dummy = torch.randn(1, 3, 640, 640)
        output = detection_model(dummy)

        assert isinstance(output, dict), f"Expected dict, got {type(output)}"
        assert "cls_logits" in output
        assert "reg_pred" in output
        assert "centerness" in output
        assert len(output["cls_logits"]) == 4, "Expected 4 feature levels"

    def test_onnx_compat_model_flattens(
        self,
        detection_model: DetectionModel,
    ) -> None:
        """_ONNXCompatModel flattens dict output to tuple."""
        wrapped = _ONNXCompatModel(detection_model)
        dummy = torch.randn(1, 3, 640, 640)
        output = wrapped(dummy)

        assert isinstance(output, tuple), f"Expected tuple, got {type(output)}"
        # 4 levels * 3 outputs (cls, reg, centerness) = 12
        assert len(output) == 12, f"Expected 12 tensors, got {len(output)}"

    def test_detection_onnx_output_count(
        self,
        detection_model: DetectionModel,
        temp_dir: str,
    ) -> None:
        """ONNX export of detection model produces expected number of outputs."""
        exporter = CoreExporter(
            model=detection_model,
            target="onnx",
            opset_version=18,
            target_hardware="server",
            input_shape=(1, 3, 640, 640),
            output_dir=temp_dir,
        )
        results = exporter.run_export()

        onnx_path = results["onnx"]
        try:
            import onnx  # noqa: PLC0415

            model = onnx.load(onnx_path)
            # 4 levels * 3 outputs (cls_logits, reg_pred, centerness) = 12
            assert len(model.graph.output) == 12, (
                f"Expected 12 ONNX outputs, got {len(model.graph.output)}"
            )
        except ImportError:
            pytest.skip("onnx package not installed; skipping output count check.")

    def test_executorch_detection(
        self,
        detection_model: DetectionModel,
        temp_dir: str,
    ) -> None:
        """ExecuTorch export works for detection models."""
        if not TORCH_EXPORT_AVAILABLE:
            pytest.skip("torch.export not available.")

        exporter = CoreExporter(
            model=detection_model,
            target="executorch",
            opset_version=18,
            target_hardware="server",
            input_shape=(1, 3, 640, 640),
            output_dir=temp_dir,
        )
        results = exporter.run_export()

        _check_export_results(results, ["executorch"])


# ======================================================================
# 4. TargetRewriter Integration
# ======================================================================


class TestTargetRewriterIntegration:
    """Verify CoreExporter correctly integrates TargetRewriter."""

    def test_edge_rewrite_applied(
        self,
        gelu_classification_model: GELUClassificationModel,
        temp_dir: str,
    ) -> None:
        """When target_hardware='edge', GELU->ReLU and SiLU->Hardswish are applied."""
        exporter = CoreExporter(
            model=gelu_classification_model,
            target="onnx",
            opset_version=18,
            target_hardware="edge",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        rewritten = exporter.rewrite_model()

        # Verify GELU was replaced with ReLU
        has_gelu = False
        has_relu = False
        for node in rewritten.graph.nodes:
            if node.op == "call_module":
                mod = rewritten.get_submodule(node.target)
                if isinstance(mod, nn.GELU):
                    has_gelu = True
                if isinstance(mod, nn.ReLU):
                    has_relu = True

        assert not has_gelu, "GELU should be replaced in edge-rewritten model"
        assert has_relu, "ReLU should be present after edge rewrite"
        assert isinstance(rewritten, torch.fx.GraphModule)

    def test_server_no_rewrite(
        self,
        gelu_classification_model: GELUClassificationModel,
        temp_dir: str,
    ) -> None:
        """When target_hardware='server', model is deep-copied but unchanged."""
        exporter = CoreExporter(
            model=gelu_classification_model,
            target="onnx",
            opset_version=18,
            target_hardware="server",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        rewritten = exporter.rewrite_model()

        # GELU should still be present (no rewrite for server)
        has_gelu = any(
            isinstance(m, nn.GELU) for m in rewritten.modules()
        )
        assert has_gelu, "GELU should remain in server mode (no rewrite)"
        # Should NOT be a GraphModule — server mode returns a plain deep copy
        assert not isinstance(rewritten, torch.fx.GraphModule), (
            "Server mode should return a plain nn.Module, not GraphModule"
        )

    def test_original_model_unchanged(
        self,
        gelu_classification_model: GELUClassificationModel,
        temp_dir: str,
    ) -> None:
        """Deep copy is used — original model is not mutated after rewrite."""
        # Snapshot the original model's GELU presence
        orig_has_gelu = any(
            isinstance(m, nn.GELU) for m in gelu_classification_model.modules()
        )
        assert orig_has_gelu, "Original model should have GELU"

        exporter = CoreExporter(
            model=gelu_classification_model,
            target="both",
            opset_version=18,
            target_hardware="edge",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        exporter.run_export()

        # Original model must still have GELU
        orig_still_has_gelu = any(
            isinstance(m, nn.GELU) for m in gelu_classification_model.modules()
        )
        assert orig_still_has_gelu, (
            "Original model should be unchanged after export"
        )

    def test_forward_pass_after_rewrite(
        self,
        gelu_classification_model: GELUClassificationModel,
        temp_dir: str,
    ) -> None:
        """Rewritten model still produces valid output."""
        exporter = CoreExporter(
            model=gelu_classification_model,
            target="onnx",
            opset_version=18,
            target_hardware="edge",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        rewritten = exporter.rewrite_model()

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = rewritten(dummy)

        assert output.shape == (1, 1000), (
            f"Expected (1, 1000), got {output.shape}"
        )
        assert not torch.isnan(output).any(), "Output should not contain NaN"


# ======================================================================
# 5. MetaProber Validation
# ======================================================================


class TestMetaProberValidation:
    """Verify CoreExporter integrates MetaProber correctly."""

    def test_validation_passes(
        self,
        classification_model: ClassificationModel,
        temp_dir: str,
    ) -> None:
        """Compatible model passes MetaProber validation."""
        exporter = CoreExporter(
            model=classification_model,
            target="onnx",
            opset_version=18,
            target_hardware="edge",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        rewritten = exporter.rewrite_model()
        result = exporter.validate_compatibility(rewritten)

        assert result.passed, (
            f"Validation should pass, got errors: {result.errors}"
        )
        assert len(result.details) > 0, "Should have validation details"

    def test_validation_fails(
        self,
        incompatible_model: IncompatibleModel,
        temp_dir: str,
    ) -> None:
        """Incompatible model (dynamic operations) fails validation."""
        exporter = CoreExporter(
            model=incompatible_model,
            target="onnx",
            opset_version=18,
            target_hardware="edge",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        rewritten = exporter.rewrite_model()
        result = exporter.validate_compatibility(rewritten)

        assert not result.passed, "Validation should fail for dynamic model"
        assert len(result.errors) > 0, "Should have validation errors"

    def test_validation_fails_raises_in_run_export(
        self,
        incompatible_model: IncompatibleModel,
        temp_dir: str,
    ) -> None:
        """run_export raises RuntimeError when validation fails."""
        exporter = CoreExporter(
            model=incompatible_model,
            target="onnx",
            opset_version=18,
            target_hardware="edge",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        with pytest.raises(RuntimeError, match="validation failed"):
            exporter.run_export()

    def test_fallback_to_original_model(
        self,
        classification_model: ClassificationModel,
        temp_dir: str,
    ) -> None:
        """FX-traced model validation falls back to original model.

        After FX symbolic trace, submodule type info is erased.  The
        exporter should gracefully fall back to the original model stored
        at construction time.
        """
        exporter = CoreExporter(
            model=classification_model,
            target="onnx",
            opset_version=18,
            target_hardware="edge",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        # Simulate a model that lost BaseBackbone type info (FX-traced)
        traced = torch.fx.symbolic_trace(classification_model)
        result = exporter.validate_compatibility(traced)

        # Should pass via fallback to original model
        assert result.passed, (
            f"Fallback validation should pass, got errors: {result.errors}"
        )


# ======================================================================
# 6. ONNX-Specific Tests
# ======================================================================


class TestONNXSpecific:
    """ONNX export-specific functionality."""

    def test_opset_version_17(
        self,
        classification_model: ClassificationModel,
        temp_dir: str,
    ) -> None:
        """OPSet version 17 works for ONNX export."""
        exporter = CoreExporter(
            model=classification_model,
            target="onnx",
            opset_version=17,
            target_hardware="server",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        results = exporter.run_export()

        _check_export_results(results, ["onnx"])

    def test_opset_version_18(
        self,
        classification_model: ClassificationModel,
        temp_dir: str,
    ) -> None:
        """OPSet version 18 works for ONNX export."""
        exporter = CoreExporter(
            model=classification_model,
            target="onnx",
            opset_version=18,
            target_hardware="server",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        results = exporter.run_export()

        _check_export_results(results, ["onnx"])

    def test_dynamic_axes(
        self,
        classification_model: ClassificationModel,
        temp_dir: str,
    ) -> None:
        """Dynamic axes for height/width dimensions work correctly.

        Note: The batch dimension is NOT dynamic when the input has
        ``batch_size=1`` because the model specialized the batch dimension
        to a constant during tracing.  Only spatial dimensions (H, W) are
        marked dynamic in that case.
        """
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
        }
        exporter = CoreExporter(
            model=classification_model,
            target="onnx",
            opset_version=18,
            target_hardware="server",
            input_shape=(1, 3, 224, 224),
            dynamic_axes=dynamic_axes,
            output_dir=temp_dir,
        )
        results = exporter.run_export()

        _check_export_results(results, ["onnx"])

        # Verify dynamic axes are present in the ONNX graph
        onnx_path = results["onnx"]
        try:
            import onnx  # noqa: PLC0415

            model = onnx.load(onnx_path)
            graph = model.graph

            # Check that spatial dimensions (H, W) are dynamic.
            # Batch dimension (index 0) cannot be dynamic when the input
            # batch size is 1 because the model specialized it.
            input_info = graph.input[0]
            input_type = input_info.type.tensor_type

            # Height (index 2) should be dynamic
            height_dim = input_type.shape.dim[2]
            assert height_dim.dim_param != "", (
                "Expected dynamic height dim, got static"
            )

            # Width (index 3) should be dynamic
            width_dim = input_type.shape.dim[3]
            assert width_dim.dim_param != "", (
                "Expected dynamic width dim, got static"
            )
        except ImportError:
            pytest.skip("onnx package not installed; skipping dynamic axes check.")

    def test_output_names_auto_inferred(
        self,
        classification_model: ClassificationModel,
    ) -> None:
        """Output names are correctly auto-inferred for classification."""
        dummy = torch.randn(1, 3, 224, 224)
        names = _infer_output_names(classification_model, dummy)
        assert names == ["output"], (
            f"Expected ['output'], got {names}"
        )

    def test_output_names_detection(
        self,
        detection_model: DetectionModel,
    ) -> None:
        """Output names are correctly auto-inferred for detection dict."""
        dummy = torch.randn(1, 3, 640, 640)
        names = _infer_output_names(detection_model, dummy)
        # Detection model returns dict of lists -> each list element gets a suffix
        expected = [
            "centerness.output_0", "centerness.output_1",
            "centerness.output_2", "centerness.output_3",
            "cls_logits.output_0", "cls_logits.output_1",
            "cls_logits.output_2", "cls_logits.output_3",
            "reg_pred.output_0", "reg_pred.output_1",
            "reg_pred.output_2", "reg_pred.output_3",
        ]
        assert names == expected, (
            f"Expected {expected}, got {names}"
        )

    def test_onnx_model_valid(
        self,
        classification_model: ClassificationModel,
        temp_dir: str,
    ) -> None:
        """Exported ONNX model passes ``onnx.checker.check_model()``."""
        try:
            import onnx  # noqa: PLC0415
        except ImportError:
            pytest.skip("onnx package not installed.")

        exporter = CoreExporter(
            model=classification_model,
            target="onnx",
            opset_version=18,
            target_hardware="server",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        results = exporter.run_export()

        onnx_model = onnx.load(results["onnx"])
        onnx.checker.check_model(onnx_model)
        # If we get here, the model is valid
        assert True, "ONNX model passed checker validation"


# ======================================================================
# 7. ExecuTorch-Specific Tests
# ======================================================================


class TestExecuTorchSpecific:
    """ExecuTorch export-specific functionality."""

    def test_executorch_file_exists(
        self,
        classification_model: ClassificationModel,
        temp_dir: str,
    ) -> None:
        """ExecuTorch export creates a .pte file."""
        if not TORCH_EXPORT_AVAILABLE:
            pytest.skip("torch.export not available.")

        exporter = CoreExporter(
            model=classification_model,
            target="executorch",
            opset_version=18,
            target_hardware="server",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        results = exporter.run_export()

        pte_path = Path(results["executorch"])
        assert pte_path.exists(), ".pte file should exist"
        assert pte_path.suffix == ".pte", f"Expected .pte suffix, got {pte_path.suffix}"

    def test_dynamic_shapes(
        self,
        classification_model: ClassificationModel,
        temp_dir: str,
    ) -> None:
        """Dynamic shapes for batch/H/W work with ExecuTorch export."""
        if not TORCH_EXPORT_AVAILABLE:
            pytest.skip("torch.export not available.")

        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
        }
        exporter = CoreExporter(
            model=classification_model,
            target="executorch",
            opset_version=18,
            target_hardware="server",
            input_shape=(1, 3, 224, 224),
            dynamic_axes=dynamic_axes,
            output_dir=temp_dir,
        )
        results = exporter.run_export()

        _check_export_results(results, ["executorch"])

    def test_no_graph_breaks(
        self,
        classification_model: ClassificationModel,
    ) -> None:
        """torch.export.export succeeds without graph breaks."""
        if not TORCH_EXPORT_AVAILABLE:
            pytest.skip("torch.export not available.")

        from torch.export import export as torch_export  # noqa: PLC0415

        dummy = torch.randn(1, 3, 224, 224)
        # This should not raise
        exported = torch_export(classification_model, (dummy,))
        assert exported is not None, "ExportedProgram should not be None"


# ======================================================================
# 8. Error Handling
# ======================================================================


class TestErrorHandling:
    """Verify CoreExporter raises appropriate errors for invalid inputs."""

    def test_invalid_target(
        self,
        classification_model: ClassificationModel,
    ) -> None:
        """Invalid target raises ValueError."""
        with pytest.raises(ValueError, match="Invalid target"):
            CoreExporter(
                model=classification_model,
                target="tflite",  # invalid
                opset_version=18,
                target_hardware="server",
                input_shape=(1, 3, 224, 224),
            )

    def test_invalid_opset(
        self,
        classification_model: ClassificationModel,
    ) -> None:
        """Invalid opset_version raises ValueError."""
        with pytest.raises(ValueError, match="Invalid opset_version"):
            CoreExporter(
                model=classification_model,
                target="onnx",
                opset_version=15,  # not 17 or 18
                target_hardware="server",
                input_shape=(1, 3, 224, 224),
            )

    def test_invalid_hardware(
        self,
        classification_model: ClassificationModel,
    ) -> None:
        """Invalid target_hardware raises ValueError."""
        with pytest.raises(ValueError, match="Invalid target_hardware"):
            CoreExporter(
                model=classification_model,
                target="onnx",
                opset_version=18,
                target_hardware="mobile",  # invalid
                input_shape=(1, 3, 224, 224),
            )

    def test_invalid_input_shape(
        self,
        classification_model: ClassificationModel,
    ) -> None:
        """Invalid input_shape (wrong ndim) raises ValueError."""
        with pytest.raises(ValueError, match="input_shape"):
            CoreExporter(
                model=classification_model,
                target="onnx",
                opset_version=18,
                target_hardware="server",
                input_shape=(1, 3, 224),  # 3D instead of 4D
            )

    def test_empty_input_shape(
        self,
        classification_model: ClassificationModel,
    ) -> None:
        """Empty input_shape raises ValueError."""
        with pytest.raises(ValueError, match="input_shape"):
            CoreExporter(
                model=classification_model,
                target="onnx",
                opset_version=18,
                target_hardware="server",
                input_shape=(),  # empty
            )


# ======================================================================
# 9. Edge Cases
# ======================================================================


class TestEdgeCases:
    """Edge case tests for the CoreExporter."""

    def test_batch_size_1(
        self,
        classification_model: ClassificationModel,
        temp_dir: str,
    ) -> None:
        """Export works with batch size 1."""
        exporter = CoreExporter(
            model=classification_model,
            target="both",
            opset_version=18,
            target_hardware="server",
            input_shape=(1, 3, 224, 224),
            output_dir=temp_dir,
        )
        results = exporter.run_export()
        _check_export_results(results, ["onnx", "executorch"])

    def test_non_square_input(
        self,
        classification_model: ClassificationModel,
        temp_dir: str,
    ) -> None:
        """Export works with non-square (480x640) input.

        This catches potential H/W indexing inversion bugs.
        """
        exporter = CoreExporter(
            model=classification_model,
            target="both",
            opset_version=18,
            target_hardware="server",
            input_shape=(1, 3, 480, 640),
            output_dir=temp_dir,
        )

        # Verify the model runs with non-square input first
        dummy = torch.randn(1, 3, 480, 640)
        output = classification_model(dummy)
        # For stride-32 coarsest level: 480/32=15, 640/32=20
        # But with our SimpleClassificationBackbone: stem (stride 2) gives 240x320,
        # then layer1 (stride 2) 120x160, layer2 (stride 2) 60x80, layer3 (stride 2) 30x40
        # So the last feature is 30x40, not 15x20. That's fine, the GAP flattens it.
        assert output.shape == (1, 1000), (
            f"Expected (1, 1000), got {output.shape}"
        )

        results = exporter.run_export()
        _check_export_results(results, ["onnx", "executorch"])

    def test_large_input(
        self,
        classification_model: ClassificationModel,
        temp_dir: str,
    ) -> None:
        """Export works with large (1024x1024) input."""
        exporter = CoreExporter(
            model=classification_model,
            target="onnx",
            opset_version=18,
            target_hardware="server",
            input_shape=(1, 3, 1024, 1024),
            output_dir=temp_dir,
        )

        # Verify forward pass
        dummy = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            output = classification_model(dummy)
        assert output.shape == (1, 1000), (
            f"Expected (1, 1000), got {output.shape}"
        )

        results = exporter.run_export()
        _check_export_results(results, ["onnx"])


# ======================================================================
# Additional validation / compatibility tests
# ======================================================================


class TestValidationResult:
    """Verify ValidationResult behaves correctly."""

    def test_validation_result_bool(self) -> None:
        """ValidationResult __bool__ returns passed status."""
        passed = ValidationResult(passed=True, details=["ok"])
        assert bool(passed) is True

        failed = ValidationResult(passed=False, errors=["bad"])
        assert bool(failed) is False

    def test_validation_result_str(self) -> None:
        """ValidationResult __str__ returns meaningful summary."""
        passed = ValidationResult(passed=True, details=["ok"])
        assert "PASSED" in str(passed)

        failed = ValidationResult(passed=False, errors=["bad"])
        assert "FAILED" in str(failed)
        assert "bad" in str(failed)


class TestONNXCompatModel:
    """Verify _ONNXCompatModel wrapper behaviour."""

    def test_flat_tensor_wrapped(self) -> None:
        """Single tensor output is wrapped in a 1-element tuple."""
        model = nn.Linear(10, 5)
        wrapped = _ONNXCompatModel(model)
        dummy = torch.randn(2, 10)
        output = wrapped(dummy)
        assert isinstance(output, tuple)
        assert len(output) == 1
        assert output[0].shape == (2, 5)

    def test_dict_flattened(self) -> None:
        """Dict output is flattened deterministically."""

        class DictModel(nn.Module):
            """Model returning a dict."""

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: D102
                return {"b": x * 2, "a": x}

        wrapped = _ONNXCompatModel(DictModel())
        dummy = torch.randn(2, 3)
        output = wrapped(dummy)
        assert isinstance(output, tuple)
        assert len(output) == 2
        # Sorted keys: "a" then "b"
        assert torch.equal(output[0], dummy)
        assert torch.equal(output[1], dummy * 2)

    def test_list_flattened(self) -> None:
        """List output is flattened."""

        class ListModel(nn.Module):
            """Model returning a list."""

            def forward(self, x: torch.Tensor) -> list[torch.Tensor]:  # noqa: D102
                return [x, x * 2]

        wrapped = _ONNXCompatModel(ListModel())
        dummy = torch.randn(2, 3)
        output = wrapped(dummy)
        assert isinstance(output, tuple)
        assert len(output) == 2

    def test_nested_flattened(self) -> None:
        """Nested dict/list output is flattened."""

        class NestedModel(nn.Module):
            """Model returning nested dict/list."""

            def forward(self, x: torch.Tensor) -> dict:  # noqa: D102
                return {"out": {"cls": x, "reg": x * 2}}

        wrapped = _ONNXCompatModel(NestedModel())
        dummy = torch.randn(2, 3)
        output = wrapped(dummy)
        assert isinstance(output, tuple)
        assert len(output) == 2

    def test_unsupported_type_raises(self) -> None:
        """Unsupported output type raises TypeError."""

        class BadModel(nn.Module):
            """Model returning a string."""

            def forward(self, x: torch.Tensor) -> str:  # noqa: D102, ARG002
                return "string_output"

        wrapped = _ONNXCompatModel(BadModel())
        with pytest.raises(TypeError, match="Unsupported output type"):
            wrapped(torch.randn(2, 3))


# ======================================================================
# Fixture interaction tests
# ======================================================================


class TestFixtures:
    """Verify test fixtures produce valid models."""

    def test_classification_model_output(
        self,
        classification_model: ClassificationModel,
    ) -> None:
        """Classification model produces correct output shape."""
        dummy = torch.randn(1, 3, 224, 224)
        output = classification_model(dummy)
        assert output.shape == (1, 1000)

    def test_segmentation_model_output(
        self,
        segmentation_model: SegmentationModel,
    ) -> None:
        """Segmentation model produces correct output shape."""
        dummy = torch.randn(1, 3, 512, 512)
        output = segmentation_model(dummy)
        # ResUNetDecoder upsamples to finest feature level (stride 4):
        # 512/4 = 128
        assert output.shape == (1, 21, 128, 128), (
            f"Expected (1, 21, 128, 128), got {output.shape}"
        )

    def test_detection_model_output(
        self,
        detection_model: DetectionModel,
    ) -> None:
        """Detection model produces correct dict output."""
        dummy = torch.randn(1, 3, 640, 640)
        output = detection_model(dummy)
        assert isinstance(output, dict)
        for key in ("cls_logits", "reg_pred", "centerness"):
            assert key in output
            assert len(output[key]) == 4  # 4 feature levels

    def test_gelu_model_has_gelu(
        self,
        gelu_classification_model: GELUClassificationModel,
    ) -> None:
        """GELU model fixture actually contains GELU."""
        has_gelu = any(
            isinstance(m, nn.GELU) for m in gelu_classification_model.modules()
        )
        assert has_gelu, "GELU fixture should contain nn.GELU"

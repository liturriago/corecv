"""Comprehensive tests for the config module (schemas and load_config).

Tests cover polymorphic dataclass loading, strict validation, type checking,
non-square ``input_size`` support, frozen dataclass immutability, and default
value preservation.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import pytest

from corecv.config import (
    BaseTaskConfig,
    ClassificationConfig,
    DetectionConfig,
    SegmentationConfig,
    load_config,
)

# ======================================================================
# Helper fixtures
# ======================================================================


class TestConfigLoading:
    """Valid YAML file and dict loading for all three task types."""

    # ------------------------------------------------------------------
    # YAML file loading
    # ------------------------------------------------------------------

    def test_load_classification_from_yaml(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Load a ClassificationConfig from a YAML file."""
        content: str = (
            "task: classification\n"
            "model_name: mobilenet_v3_large\n"
            "num_classes: 1000\n"
        )
        cfg_path: pathlib.Path = tmp_path / "classification.yaml"
        cfg_path.write_text(content, encoding="utf-8")

        config: BaseTaskConfig = load_config(str(cfg_path))

        assert isinstance(config, ClassificationConfig)
        assert config.task == "classification"
        assert config.model_name == "mobilenet_v3_large"
        assert config.num_classes == 1000
        # Default values
        assert config.pretrained is True
        assert config.input_size == (224, 224)
        assert config.head_type == "linear"

    def test_load_segmentation_from_yaml(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Load a SegmentationConfig from a YAML file."""
        content: str = (
            "task: segmentation\n"
            "model_name: resnet50\n"
            "num_classes: 21\n"
            "head_type: deeplabv3plus\n"
        )
        cfg_path: pathlib.Path = tmp_path / "segmentation.yaml"
        cfg_path.write_text(content, encoding="utf-8")

        config: BaseTaskConfig = load_config(str(cfg_path))

        assert isinstance(config, SegmentationConfig)
        assert config.task == "segmentation"
        assert config.model_name == "resnet50"
        assert config.num_classes == 21
        assert config.head_type == "deeplabv3plus"
        # Defaults
        assert config.input_size == (224, 224)
        assert config.decoder_channels == 256

    def test_load_detection_from_yaml(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Load a DetectionConfig from a YAML file."""
        content: str = (
            "task: detection\n"
            "model_name: mobilenet_v3_large\n"
            "num_classes: 80\n"
            "head_type: retinanet\n"
        )
        cfg_path: pathlib.Path = tmp_path / "detection.yaml"
        cfg_path.write_text(content, encoding="utf-8")

        config: BaseTaskConfig = load_config(str(cfg_path))

        assert isinstance(config, DetectionConfig)
        assert config.task == "detection"
        assert config.model_name == "mobilenet_v3_large"
        assert config.num_classes == 80
        assert config.head_type == "retinanet"
        # Defaults
        assert config.input_size == (224, 224)
        assert config.score_thresh == 0.05

    def test_load_yaml_with_all_fields(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Load a config from YAML including every optional field."""
        content: str = (
            "task: classification\n"
            "model_name: resnet50\n"
            "num_classes: 10\n"
            "pretrained: false\n"
            "target_hardware: edge\n"
            "mixed_precision: false\n"
            "head_type: mlp\n"
            "dropout: 0.5\n"
            "label_smoothing: 0.1\n"
        )
        cfg_path: pathlib.Path = tmp_path / "full.yaml"
        cfg_path.write_text(content, encoding="utf-8")

        config: BaseTaskConfig = load_config(str(cfg_path))

        assert isinstance(config, ClassificationConfig)
        assert config.model_name == "resnet50"
        assert config.num_classes == 10
        assert config.pretrained is False
        assert config.target_hardware == "edge"
        assert config.mixed_precision is False
        assert config.head_type == "mlp"
        assert config.dropout == 0.5
        assert config.label_smoothing == 0.1

    # ------------------------------------------------------------------
    # Dict loading
    # ------------------------------------------------------------------

    def test_load_classification_from_dict(self) -> None:
        """Load a ClassificationConfig from a native dict (no file I/O)."""
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
            "num_classes": 1000,
        }
        config: BaseTaskConfig = load_config(raw)

        assert isinstance(config, ClassificationConfig)
        assert config.task == "classification"
        assert config.model_name == "mobilenet_v3_large"
        assert config.num_classes == 1000

    def test_load_segmentation_from_dict(self) -> None:
        """Load a SegmentationConfig from a native dict."""
        raw: dict[str, Any] = {
            "task": "segmentation",
            "model_name": "resnet50",
            "num_classes": 21,
            "head_type": "unet",
        }
        config: BaseTaskConfig = load_config(raw)

        assert isinstance(config, SegmentationConfig)
        assert config.task == "segmentation"
        assert config.head_type == "unet"

    def test_load_detection_from_dict(self) -> None:
        """Load a DetectionConfig from a native dict."""
        raw: dict[str, Any] = {
            "task": "detection",
            "model_name": "mobilenet_v3_large",
            "num_classes": 80,
            "head_type": "fcos",
        }
        config: BaseTaskConfig = load_config(raw)

        assert isinstance(config, DetectionConfig)
        assert config.task == "detection"
        assert config.head_type == "fcos"

    def test_task_discrimination(self) -> None:
        """Verify the ``task`` field correctly discriminates the dataclass type."""
        class_cfg: BaseTaskConfig = load_config(
            {"task": "classification", "model_name": "m", "num_classes": 5}
        )
        seg_cfg: BaseTaskConfig = load_config(
            {"task": "segmentation", "model_name": "m", "num_classes": 5}
        )
        det_cfg: BaseTaskConfig = load_config(
            {"task": "detection", "model_name": "m", "num_classes": 5}
        )

        assert isinstance(class_cfg, ClassificationConfig)
        assert isinstance(seg_cfg, SegmentationConfig)
        assert isinstance(det_cfg, DetectionConfig)


class TestConfigValidation:
    """Strict validation — missing fields, unknown task, extra fields, types."""

    def test_missing_num_classes(self) -> None:
        """Omitting required ``num_classes`` raises ``ValueError``."""
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
        }
        with pytest.raises(ValueError, match="num_classes"):
            load_config(raw)

    def test_missing_model_name(self) -> None:
        """Omitting required ``model_name`` raises ``ValueError``."""
        raw: dict[str, Any] = {
            "task": "classification",
            "num_classes": 10,
        }
        with pytest.raises(ValueError, match="model_name"):
            load_config(raw)

    def test_missing_required_field_displays_name(self) -> None:
        """Error message includes the missing field name."""
        raw: dict[str, Any] = {
            "task": "segmentation",
            # model_name is missing
            "num_classes": 21,
        }
        with pytest.raises(ValueError) as exc_info:
            load_config(raw)
        assert "model_name" in str(exc_info.value)

    def test_unknown_task(self) -> None:
        """Using an unknown task value raises ``ValueError`` with valid tasks."""
        raw: dict[str, Any] = {
            "task": "unknown",
            "model_name": "resnet50",
            "num_classes": 10,
        }
        with pytest.raises(ValueError) as exc_info:
            load_config(raw)
        msg: str = str(exc_info.value)
        assert "unknown" in msg
        assert "classification" in msg
        assert "segmentation" in msg
        assert "detection" in msg

    def test_extra_unrecognised_field(self) -> None:
        """Extra unrecognised fields raise ``ValueError`` listing the field."""
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
            "num_classes": 1000,
            "unknown_field": 123,
        }
        with pytest.raises(ValueError, match="unknown_field"):
            load_config(raw)

    def test_multiple_extra_fields(self) -> None:
        """Multiple extra fields — the first encountered raises ``ValueError``."""
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
            "num_classes": 10,
            "foo": 1,
            "bar": 2,
        }
        with pytest.raises(ValueError) as exc_info:
            load_config(raw)
        msg: str = str(exc_info.value)
        assert "Unrecognised field" in msg
        # The validator raises on the first unrecognised field encountered
        assert "foo" in msg

    def test_type_mismatch_num_classes_string(
        self,
    ) -> None:
        """String instead of int for ``num_classes`` raises ``ValueError``."""
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
            "num_classes": "ten",
        }
        with pytest.raises(ValueError, match="num_classes"):
            load_config(raw)

    def test_type_mismatch_input_size_wrong_length(
        self,
    ) -> None:
        """``input_size`` with wrong tuple length raises ``ValueError``."""
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
            "num_classes": 10,
            "input_size": (224,),  # wrong length
        }
        with pytest.raises(ValueError, match="input_size"):
            load_config(raw)

    def test_input_size_list_accepted_for_yaml_compat(
        self,
    ) -> None:
        """YAML list ``[224, 224]`` for ``input_size`` is accepted.

        YAML cannot serialize tuples, so lists are accepted for tuple-typed
        fields to ensure round-trip compatibility with YAML configs.
        """
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
            "num_classes": 10,
            "input_size": [224, 224],  # list, accepted for YAML compat
        }
        config = load_config(raw)
        # Value is stored as-is (list), but validation passes.
        assert list(config.input_size) == [224, 224]

    def test_type_mismatch_pretrained_string(
        self,
    ) -> None:
        """String instead of bool for ``pretrained`` raises ``ValueError``."""
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
            "num_classes": 10,
            "pretrained": "yes",
        }
        with pytest.raises(ValueError, match="pretrained"):
            load_config(raw)

    def test_type_mismatch_dropout_string(
        self,
    ) -> None:
        """String instead of float for ``dropout`` raises ``ValueError``."""
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
            "num_classes": 10,
            "dropout": "zero",
        }
        with pytest.raises(ValueError, match="dropout"):
            load_config(raw)


class TestInputSize:
    """Non-square ``input_size`` support and H/W ordering correctness."""

    def test_non_square_tuple(self) -> None:
        """``(480, 640)`` with distinct height and width is accepted."""
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "resnet50",
            "num_classes": 10,
            "input_size": (480, 640),
        }
        config: BaseTaskConfig = load_config(raw)
        assert config.input_size == (480, 640)

    def test_no_hw_inversion_classification(
        self,
    ) -> None:
        """Verify (H, W) order in ClassificationConfig: first element is height."""
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
            "num_classes": 10,
            "input_size": (480, 640),
        }
        config: ClassificationConfig = load_config(raw)  # type: ignore[assignment]
        h: int
        w: int
        h, w = config.input_size
        assert h == 480, f"Expected height=480, got h={h}"
        assert w == 640, f"Expected width=640, got w={w}"

    def test_no_hw_inversion_segmentation(
        self,
    ) -> None:
        """Verify (H, W) order in SegmentationConfig."""
        raw: dict[str, Any] = {
            "task": "segmentation",
            "model_name": "resnet50",
            "num_classes": 21,
            "input_size": (480, 640),
        }
        config: SegmentationConfig = load_config(raw)  # type: ignore[assignment]
        h, w = config.input_size
        assert h == 480
        assert w == 640

    def test_no_hw_inversion_detection(
        self,
    ) -> None:
        """Verify (H, W) order in DetectionConfig."""
        raw: dict[str, Any] = {
            "task": "detection",
            "model_name": "mobilenet_v3_large",
            "num_classes": 80,
            "input_size": (480, 640),
        }
        config: DetectionConfig = load_config(raw)  # type: ignore[assignment]
        h, w = config.input_size
        assert h == 480
        assert w == 640

    def test_square_input_size_default(self) -> None:
        """Default ``input_size`` is ``(224, 224)`` (square)."""
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
            "num_classes": 1000,
        }
        config: ClassificationConfig = load_config(raw)  # type: ignore[assignment]
        assert config.input_size == (224, 224)


class TestDataclassBehavior:
    """Frozen dataclass immutability and default value preservation."""

    def test_frozen_immutability(self) -> None:
        """Modifying a field on a frozen dataclass.

        Raises:
            dataclasses.FrozenInstanceError: on attribute assignment.
        """
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
            "num_classes": 1000,
        }
        config: ClassificationConfig = load_config(raw)  # type: ignore[assignment]

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.model_name = "resnet50"  # type: ignore[misc]

    def test_frozen_immutability_new_field(self) -> None:
        """Assigning a new attribute on a frozen dataclass also raises."""
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
            "num_classes": 1000,
        }
        config: ClassificationConfig = load_config(raw)  # type: ignore[assignment]

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.new_field = "value"  # type: ignore[misc]

    def test_defaults_pretrained(self) -> None:
        """``pretrained`` defaults to ``True`` when not provided."""
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
            "num_classes": 1000,
        }
        config: ClassificationConfig = load_config(raw)  # type: ignore[assignment]
        assert config.pretrained is True

    def test_defaults_mixed_precision(self) -> None:
        """``mixed_precision`` defaults to ``True`` when not provided."""
        raw: dict[str, Any] = {
            "task": "segmentation",
            "model_name": "resnet50",
            "num_classes": 21,
        }
        config: SegmentationConfig = load_config(raw)  # type: ignore[assignment]
        assert config.mixed_precision is True

    def test_defaults_target_hardware(self) -> None:
        r"""``target_hardware`` defaults to ``"gpu"`` when not provided."""
        raw: dict[str, Any] = {
            "task": "detection",
            "model_name": "mobilenet_v3_large",
            "num_classes": 80,
        }
        config: DetectionConfig = load_config(raw)  # type: ignore[assignment]
        assert config.target_hardware == "gpu"

    def test_classification_defaults(self) -> None:
        """Classification-specific fields default correctly."""
        raw: dict[str, Any] = {
            "task": "classification",
            "model_name": "mobilenet_v3_large",
            "num_classes": 1000,
        }
        config: ClassificationConfig = load_config(raw)  # type: ignore[assignment]
        assert config.head_type == "linear"
        assert config.dropout == 0.0
        assert config.label_smoothing == 0.0

    def test_segmentation_defaults(self) -> None:
        """Segmentation-specific fields default correctly."""
        raw: dict[str, Any] = {
            "task": "segmentation",
            "model_name": "resnet50",
            "num_classes": 21,
        }
        config: SegmentationConfig = load_config(raw)  # type: ignore[assignment]
        assert config.backbone_out_indices == (0, 1, 2, 3)
        assert config.decoder_channels == 256
        assert config.align_corners is False
        assert config.ignore_index == 255

    def test_detection_defaults(self) -> None:
        """Detection-specific fields default correctly."""
        raw: dict[str, Any] = {
            "task": "detection",
            "model_name": "mobilenet_v3_large",
            "num_classes": 80,
        }
        config: DetectionConfig = load_config(raw)  # type: ignore[assignment]
        assert config.num_anchors == 9
        assert config.score_thresh == 0.05
        assert config.nms_thresh == 0.5
        assert config.detections_per_img == 100
        assert config.box_coder_weights == (1.0, 1.0, 1.0, 1.0)

    def test_segmentation_default_backbone_out_indices(self) -> None:
        """``backbone_out_indices`` default is ``(0, 1, 2, 3)``."""
        raw: dict[str, Any] = {
            "task": "segmentation",
            "model_name": "resnet50",
            "num_classes": 21,
        }
        config: SegmentationConfig = load_config(raw)  # type: ignore[assignment]
        assert config.backbone_out_indices == (0, 1, 2, 3)

    def test_detection_default_anchor_sizes(self) -> None:
        """``anchor_sizes`` default structure is correct."""
        raw: dict[str, Any] = {
            "task": "detection",
            "model_name": "mobilenet_v3_large",
            "num_classes": 80,
        }
        config: DetectionConfig = load_config(raw)  # type: ignore[assignment]
        assert isinstance(config.anchor_sizes, tuple)
        assert len(config.anchor_sizes) == 5
        assert config.anchor_sizes[0] == (32,)

    def test_yaml_file_not_found(self) -> None:
        """Loading a non-existent YAML file raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_invalid_type_passed(self) -> None:
        """Passing an invalid type (e.g. ``int``) raises ``TypeError``."""
        with pytest.raises(TypeError):
            load_config(42)  # type: ignore[arg-type]

    def test_yaml_non_mapping_content(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """YAML file that does not contain a top-level mapping.

        Raises:
            TypeError: when the YAML root is not a dict.
        """
        cfg_path: pathlib.Path = tmp_path / "list.yaml"
        cfg_path.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(TypeError):
            load_config(str(cfg_path))


class TestCheckType:
    """Direct tests for the ``_check_type`` helper (via ``schemas`` module)."""

    def test_check_type_via_validation(self) -> None:
        """Exercise type checking through ``load_config`` validation paths."""
        # Valid bool
        cfg: BaseTaskConfig = load_config(
            {
                "task": "classification",
                "model_name": "m",
                "num_classes": 10,
                "pretrained": False,
            }
        )
        assert cfg.pretrained is False

        # Valid int
        cfg2: BaseTaskConfig = load_config(
            {
                "task": "classification",
                "model_name": "m",
                "num_classes": 42,
            }
        )
        assert cfg2.num_classes == 42

        # Wrong bool type
        with pytest.raises(ValueError, match="pretrained"):
            load_config(
                {
                    "task": "classification",
                    "model_name": "m",
                    "num_classes": 10,
                    "pretrained": 1,  # int, not bool
                }
            )

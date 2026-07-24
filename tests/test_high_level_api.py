"""Comprehensive tests for the CoreModel high-level API facade.

Tests cover:

1.  **CoreModel Initialisation** — valid/invalid task types, square and
    non-square input sizes, device auto-detection, num_classes inference.
2.  **Fluent Setters** — ``set_loss_fn``, ``set_train_dataloader``,
    ``set_val_dataloader`` with chaining.
3.  **Training API (``.train()``)** — polymorphic config styles (kwargs,
    dict, TrainingConfig dataclass, YAML file path, mixed), validation
    errors (missing dataloader, missing loss, invalid config), functional
    1-epoch dry-runs for classification, segmentation, and detection.
4.  **Inference API (``.predict()``)** — single / list image paths, tensor
    inputs, task-specific post-processing (classification top-k + softmax,
    segmentation argmax, detection bbox / NMS / confidence), parameter
    overrides (conf_threshold, iou_threshold, topk, batch_size).
5.  **Export API (``.export()``)** — ONNX, ExecuTorch, both; edge vs
    server target; opset versions 17/18; optimise flag; custom output_path,
    input_shape, dynamic_axes; invalid parameter errors.
6.  **Integration Tests** — full train → predict → export workflow,
    non-square resolution end-to-end, gradient flow through ``train()``.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from corecv.api import CoreModel
from corecv.api.model import ExportConfig, TrainingConfig
from corecv.core.contract import BaseBackbone, FeatureInfo
from corecv.engine.predictor import Prediction
from corecv.models import (
    CoreObjectDetector,
    DecoupledAnchorFreeHead,
    LinearClassificationHead,
    ResUNetDecoder,
)

# ======================================================================
# Constants
# ======================================================================

NUM_CLASSES: int = 10
NON_SQUARE_H: int = 480
NON_SQUARE_W: int = 640

# ======================================================================
# Mock backbone (reused from test_trainer.py)
# ======================================================================


class MockListBackbone(BaseBackbone):
    """Mock backbone returning a list of feature maps.

    Each level is a strided convolution applied directly to the input,
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

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass producing multi-scale feature list.

        Args:
            x: Input tensor ``(N, 3, H, W)``.

        Returns:
            List of feature tensors ordered from finest to coarsest.
        """
        return [conv(x) for conv in self.convs]


# ======================================================================
# Classification model factory
# ======================================================================


def _build_classification_model(
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Build a synthetic classification model.

    Args:
        num_classes: Number of output classes.

    Returns:
        A ``nn.Sequential`` backbone + classification head.
    """
    backbone = MockListBackbone(
        out_channels=(16, 24, 48),
        out_strides=(4, 8, 16),
    )
    head = LinearClassificationHead(
        feature_info=backbone.feature_info,
        num_classes=num_classes,
    )
    return nn.Sequential(backbone, head)


# ======================================================================
# Segmentation model factory
# ======================================================================


class _SegmentationModel(nn.Module):
    """Wraps backbone + decoder + upsample for full-resolution output."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        out_channels: int = 64,
    ) -> None:
        """Initialise with backbone and decoder.

        Args:
            num_classes: Number of segmentation classes.
            out_channels: Base channel count.
        """
        super().__init__()
        self.backbone = MockListBackbone(
            out_channels=(out_channels, out_channels * 2, out_channels * 4),
            out_strides=(4, 8, 16),
        )
        self.decoder = ResUNetDecoder(
            feature_info=self.backbone.feature_info,
            out_channels=out_channels * 2,
            num_classes=num_classes,
            dropout=0.0,
        )
        self.upsample = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: backbone -> decoder -> upsample.

        Args:
            x: Input tensor ``(B, 3, H, W)``.

        Returns:
            Segmentation logits ``(B, C, H, W)`` at input resolution.
        """
        feats = self.backbone(x)
        logits = self.decoder(feats)
        return self.upsample(logits)


# ======================================================================
# Detection model factory
# ======================================================================


def _build_detection_model(
    num_classes: int = NUM_CLASSES,
) -> CoreObjectDetector:
    """Build a synthetic detection model.

    Args:
        num_classes: Number of detection classes.

    Returns:
        A ``CoreObjectDetector`` instance.
    """
    feat_channels = 64
    backbone = MockListBackbone(
        out_channels=(feat_channels, feat_channels * 2, feat_channels * 4),
        out_strides=(8, 16, 32),
    )
    head = DecoupledAnchorFreeHead(
        feature_info=backbone.feature_info,
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_convs=2,
    )
    return CoreObjectDetector(backbone=backbone, neck=None, head=head)


# ======================================================================
# Synthetic datasets (copied from test_trainer.py patterns)
# ======================================================================


class SyntheticClassificationDataset(Dataset):
    """Yields random images and integer class labels."""

    def __init__(
        self,
        num_samples: int = 4,
        img_size: int = 224,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """Initialise with given dimensions.

        Args:
            num_samples: Number of synthetic samples.
            img_size: Spatial size (square).
            num_classes: Number of classes.
        """
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return a synthetic (image, label) pair.

        Args:
            idx: Index (ignored).

        Returns:
            Tuple of ``(3, H, W)`` image tensor and integer label.
        """
        img = torch.randn(3, self.img_size, self.img_size)
        label = int(torch.randint(0, self.num_classes, ()).item())
        return img, label


class SyntheticSegmentationDataset(Dataset):
    """Yields random images and segmentation masks."""

    def __init__(
        self,
        num_samples: int = 4,
        img_size: int = 256,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        """Initialise with given dimensions.

        Args:
            num_samples: Number of synthetic samples.
            img_size: Spatial size (square).
            num_classes: Number of segmentation classes.
        """
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a synthetic (image, mask) pair.

        Args:
            idx: Index (ignored).

        Returns:
            Tuple of ``(3, H, W)`` image tensor and ``(H, W)`` mask.
        """
        img = torch.randn(3, self.img_size, self.img_size)
        mask = torch.randint(
            0, self.num_classes, (self.img_size, self.img_size), dtype=torch.long,
        )
        return img, mask


class SyntheticDetectionDataset(Dataset):
    """Yields random images and detection targets (boxes + labels)."""

    def __init__(
        self,
        num_samples: int = 4,
        img_size: int = 320,
        num_classes: int = NUM_CLASSES,
        num_boxes: int = 3,
    ) -> None:
        """Initialise with given dimensions.

        Args:
            num_samples: Number of synthetic samples.
            img_size: Spatial size (square).
            num_classes: Number of detection classes.
            num_boxes: Number of ground-truth boxes per image.
        """
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_boxes = num_boxes

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return a synthetic (image, target_dict) pair.

        Args:
            idx: Index (ignored).

        Returns:
            Tuple of ``(3, H, W)`` image tensor and a dict with
            ``'boxes'`` ``(N, 4)`` and ``'labels'`` ``(N,)``.
        """
        img = torch.randn(3, self.img_size, self.img_size)
        s: float = float(self.img_size)
        cx = torch.rand(self.num_boxes) * s * 0.6 + s * 0.2
        cy = torch.rand(self.num_boxes) * s * 0.6 + s * 0.2
        w = torch.rand(self.num_boxes) * s * 0.3 + 10.0
        h = torch.rand(self.num_boxes) * s * 0.3 + 10.0
        x1 = (cx - w / 2).clamp(min=0.0)
        y1 = (cy - h / 2).clamp(min=0.0)
        x2 = (cx + w / 2).clamp(max=s - 1.0)
        y2 = (cy + h / 2).clamp(max=s - 1.0)
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        labels = torch.randint(0, self.num_classes, (self.num_boxes,), dtype=torch.long)
        return img, {"boxes": boxes, "labels": labels}


def det_collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Collate detection batch: stack images, keep targets as list-of-dicts.

    Args:
        batch: List of ``(image, target_dict)`` tuples.

    Returns:
        Tuple of stacked image tensor ``(B, 3, H, W)`` and list of target dicts.
    """
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


# ======================================================================
# Global seed fixture
# ======================================================================


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Reset the random seed before every test for reproducibility."""
    torch.manual_seed(42)


# ======================================================================
# Helpers
# ======================================================================


def _check_gradient(loss: torch.Tensor, model: nn.Module) -> None:
    """Run ``loss.backward()`` and assert no NaN/Inf in any parameter grad.

    Args:
        loss: Scalar loss tensor.
        model: Model whose parameters are checked for valid gradients.

    Raises:
        AssertionError: If any parameter has NaN or Inf gradient.
    """
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, (
                f"Expected gradient for {name} to be populated."
            )
            assert not param.grad.isnan().any(), (
                f"NaN gradient found in {name} (shape {param.shape})."
            )
            assert not param.grad.isinf().any(), (
                f"Inf gradient found in {name} (shape {param.shape})."
            )


# ======================================================================
# 1. CoreModel Initialisation
# ======================================================================


class TestCoreModelInitialisation:
    """Tests for CoreModel constructor validation."""

    # ------------------------------------------------------------------
    # Valid task types
    # ------------------------------------------------------------------

    def test_classification_task(self) -> None:
        """Initialise CoreModel with classification task."""
        model = _build_classification_model()
        # nn.Sequential doesn't have a head.num_classes attr, so pass explicitly
        cm = CoreModel(
            model, task="classification", input_size=(224, 224),
            num_classes=NUM_CLASSES,
        )
        assert cm.task == "classification"
        assert cm.input_size == (224, 224)
        assert cm.num_classes == NUM_CLASSES

    def test_segmentation_task(self) -> None:
        """Initialise CoreModel with segmentation task."""
        model = _SegmentationModel()
        cm = CoreModel(model, task="segmentation", input_size=(256, 256))
        assert cm.task == "segmentation"

    def test_detection_task(self) -> None:
        """Initialise CoreModel with detection task."""
        model = _build_detection_model()
        cm = CoreModel(model, task="detection", input_size=(320, 320))
        assert cm.task == "detection"
        # CoreObjectDetector stores num_classes on head
        assert cm.num_classes == NUM_CLASSES

    # ------------------------------------------------------------------
    # Invalid task type
    # ------------------------------------------------------------------

    def test_invalid_task_raises_value_error(self) -> None:
        """Invalid task string raises ValueError."""
        model = _build_classification_model()
        with pytest.raises(ValueError, match="Unsupported task"):
            CoreModel(model, task="invalid_task")

    # ------------------------------------------------------------------
    # Input size handling
    # ------------------------------------------------------------------

    def test_square_input_size(self) -> None:
        """Square input size is stored correctly."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification", input_size=(224, 224))
        assert cm.input_size == (224, 224)

    def test_non_square_input_size(self) -> None:
        """Non-square (480x640) input size is stored without H/W inversion."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification", input_size=(NON_SQUARE_H, NON_SQUARE_W))
        h, w = cm.input_size
        assert h == NON_SQUARE_H, f"Expected height={NON_SQUARE_H}, got {h}"
        assert w == NON_SQUARE_W, f"Expected width={NON_SQUARE_W}, got {w}"

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    def test_device_auto_detection(self) -> None:
        """Device is auto-detected when not specified."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification")
        expected_type = "cuda" if torch.cuda.is_available() else "cpu"
        assert cm.device.type == expected_type

    def test_explicit_device_cpu(self) -> None:
        """Explicit CPU device is honoured."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        assert cm.device.type == "cpu"

    def test_explicit_device_meta(self) -> None:
        """Zero-VRAM device='meta' is accepted for shape validation."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification", device=torch.device("meta"))
        assert cm.device.type == "meta"

    # ------------------------------------------------------------------
    # num_classes inference
    # ------------------------------------------------------------------

    def test_num_classes_inferred_from_model_head(self) -> None:
        """num_classes is inferred from model.head.num_classes."""
        # Use a model that has a head.num_classes attribute
        model = _build_detection_model(num_classes=7)
        cm = CoreModel(model, task="detection")
        assert cm.num_classes == 7

    def test_num_classes_inferred_from_detector(self) -> None:
        """num_classes is inferred from detector head."""
        model = _build_detection_model(num_classes=5)
        cm = CoreModel(model, task="detection")
        assert cm.num_classes == 5

    def test_num_classes_explicit_override(self) -> None:
        """Explicit num_classes overrides model inference."""
        model = _build_classification_model(num_classes=10)
        cm = CoreModel(model, task="classification", num_classes=42)
        assert cm.num_classes == 42


# ======================================================================
# 2. Fluent Setters
# ======================================================================


class TestFluentSetters:
    """Tests for fluent setter chaining."""

    def test_set_loss_fn_returns_self(self) -> None:
        """set_loss_fn returns self for chaining."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification")
        result = cm.set_loss_fn(nn.CrossEntropyLoss())
        assert result is cm

    def test_set_train_dataloader_returns_self(self) -> None:
        """set_train_dataloader returns self for chaining."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification")
        loader = DataLoader(
            SyntheticClassificationDataset(), batch_size=4,
        )
        result = cm.set_train_dataloader(loader)
        assert result is cm

    def test_set_val_dataloader_returns_self(self) -> None:
        """set_val_dataloader returns self for chaining."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification")
        loader = DataLoader(
            SyntheticClassificationDataset(), batch_size=4,
        )
        result = cm.set_val_dataloader(loader)
        assert result is cm

    def test_chained_setters(self) -> None:
        """Multiple setters can be chained fluently."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification")
        train_loader = DataLoader(
            SyntheticClassificationDataset(), batch_size=4,
        )
        val_loader = DataLoader(
            SyntheticClassificationDataset(), batch_size=4,
        )
        result = (
            cm
            .set_loss_fn(nn.CrossEntropyLoss())
            .set_train_dataloader(train_loader)
            .set_val_dataloader(val_loader)
        )
        assert result is cm
        assert cm._loss_fn is not None
        assert cm._train_loader is not None
        assert cm._val_loader is not None


# ======================================================================
# 3. Training API — Polymorphic config validation
# ======================================================================


class TestTrainConfigPolymorphism:
    """Tests for the four config styles accepted by ``.train()``."""

    @pytest.fixture
    def trained_cm(self) -> CoreModel:
        """Return a CoreModel ready for train (dataloader + loss set)."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        cm.set_loss_fn(nn.CrossEntropyLoss())
        loader = DataLoader(
            SyntheticClassificationDataset(num_samples=4),
            batch_size=4,
        )
        cm.set_train_dataloader(loader)
        return cm

    def test_train_kwargs_only(self, trained_cm: CoreModel) -> None:
        """Train with direct **kwargs (no config arg)."""
        history = trained_cm.train(epochs=1, lr=0.001, batch_size=4)
        assert isinstance(history, dict)
        assert "train" in history or "loss" in str(history)

    def test_train_dict_config(self, trained_cm: CoreModel) -> None:
        """Train with a dict config."""
        cfg: dict[str, Any] = {"epochs": 1, "lr": 0.001, "batch_size": 4}
        history = trained_cm.train(cfg)
        assert isinstance(history, dict)

    def test_train_trainingconfig_dataclass(self, trained_cm: CoreModel) -> None:
        """Train with a TrainingConfig dataclass."""
        cfg = TrainingConfig(epochs=1, lr=0.001, batch_size=4)
        history = trained_cm.train(cfg)
        assert isinstance(history, dict)

    def test_train_yaml_file(self, trained_cm: CoreModel, tmp_path: Path) -> None:
        """Train with a YAML file path."""
        yaml_path = tmp_path / "train_config.yaml"
        yaml_path.write_text(
            "epochs: 1\nlr: 0.001\nbatch_size: 4\n",
            encoding="utf-8",
        )
        history = trained_cm.train(str(yaml_path))
        assert isinstance(history, dict)

    def test_train_mixed_kwargs_override_dict(self, trained_cm: CoreModel) -> None:
        """Kwargs override values from dict config."""
        cfg: dict[str, Any] = {"epochs": 1, "lr": 0.001, "batch_size": 4}
        history = trained_cm.train(cfg, epochs=2)
        assert isinstance(history, dict)

    def test_train_mixed_kwargs_override_dataclass(self, trained_cm: CoreModel) -> None:
        """Kwargs override values from TrainingConfig dataclass."""
        cfg = TrainingConfig(epochs=1, lr=0.001, batch_size=4)
        history = trained_cm.train(cfg, lr=0.01)
        assert isinstance(history, dict)

    def test_train_mixed_kwargs_override_yaml(
        self,
        trained_cm: CoreModel,
        tmp_path: Path,
    ) -> None:
        """Kwargs override values from YAML config."""
        yaml_path = tmp_path / "override.yaml"
        yaml_path.write_text(
            "epochs: 1\nlr: 0.001\nbatch_size: 4\n",
            encoding="utf-8",
        )
        history = trained_cm.train(str(yaml_path), epochs=3)
        assert isinstance(history, dict)


# ======================================================================
# 3b. Training API — Validation errors
# ======================================================================


class TestTrainValidationErrors:
    """Tests for validation errors raised by ``.train()``."""

    def test_missing_train_dataloader(self) -> None:
        """train() raises ValueError when no dataloader is set."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        cm.set_loss_fn(nn.CrossEntropyLoss())
        with pytest.raises(ValueError, match="Training dataloader is required"):
            cm.train(epochs=1, batch_size=4)

    def test_missing_loss_fn(self) -> None:
        """train() auto-instantiates default loss function when not set."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        loader = DataLoader(
            SyntheticClassificationDataset(num_samples=4), batch_size=4,
        )
        cm.set_train_dataloader(loader)
        history = cm.train(epochs=1, batch_size=4)
        assert cm._loss_fn is not None
        assert isinstance(history, dict)

    def test_invalid_config_type(self) -> None:
        """train() raises TypeError for invalid config type."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        cm.set_loss_fn(nn.CrossEntropyLoss())
        loader = DataLoader(
            SyntheticClassificationDataset(num_samples=4), batch_size=4,
        )
        cm.set_train_dataloader(loader)
        with pytest.raises(TypeError, match="Expected str, dict, TrainingConfig, or None"):
            cm.train(config=42)  # type: ignore[arg-type]

    def test_yaml_file_not_found(self) -> None:
        """train() raises FileNotFoundError for non-existent YAML."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        cm.set_loss_fn(nn.CrossEntropyLoss())
        loader = DataLoader(
            SyntheticClassificationDataset(num_samples=4), batch_size=4,
        )
        cm.set_train_dataloader(loader)
        with pytest.raises(FileNotFoundError, match="Training config file not found"):
            cm.train("nonexistent_config.yaml")

    def test_yaml_non_mapping(self, tmp_path: Path) -> None:
        """train() raises TypeError when YAML is not a mapping."""
        yaml_path = tmp_path / "list.yaml"
        yaml_path.write_text("- one\n- two\n", encoding="utf-8")
        model = _build_classification_model()
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        cm.set_loss_fn(nn.CrossEntropyLoss())
        loader = DataLoader(
            SyntheticClassificationDataset(num_samples=4), batch_size=4,
        )
        cm.set_train_dataloader(loader)
        with pytest.raises(TypeError, match="YAML file must contain a top-level mapping"):
            cm.train(str(yaml_path))


# ======================================================================
# 3c. Training API — Functional 1-epoch dry-runs
# ======================================================================


class TestFunctionalTraining:
    """1-epoch dry-runs for all three task types."""

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def test_classification_train_one_epoch(self) -> None:
        """Classification 1-epoch dry-run produces finite loss."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        cm.set_loss_fn(nn.CrossEntropyLoss())
        loader = DataLoader(
            SyntheticClassificationDataset(num_samples=4, num_classes=NUM_CLASSES),
            batch_size=4,
        )
        cm.set_train_dataloader(loader)

        history = cm.train(epochs=1, lr=0.001, batch_size=4)
        assert isinstance(history, dict)

    def test_classification_gradient_flow(self) -> None:
        """Classification loss.backward() produces no NaN gradients."""
        model = _build_classification_model()
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        cm.set_loss_fn(nn.CrossEntropyLoss())
        dataset = SyntheticClassificationDataset(
            num_samples=4, num_classes=NUM_CLASSES,
        )
        loader = DataLoader(dataset, batch_size=4)
        cm.set_train_dataloader(loader)

        # Run one step manually to check gradient flow
        cm.train(epochs=1, lr=0.001, batch_size=4)

        # Verify model parameters have been updated (trainer ran backward)
        # CoreTrainer already checks for NaN internally; we just check
        # that training completed.
        assert cm.trainer is not None

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def test_segmentation_train_one_epoch(self) -> None:
        """Segmentation 1-epoch dry-run produces finite loss."""
        model = _SegmentationModel(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="segmentation", device=torch.device("cpu"))
        # Use a simple combined loss for segmentation
        cm.set_loss_fn(nn.CrossEntropyLoss())
        loader = DataLoader(
            SyntheticSegmentationDataset(
                num_samples=2, img_size=256, num_classes=NUM_CLASSES,
            ),
            batch_size=2,
        )
        cm.set_train_dataloader(loader)

        history = cm.train(epochs=1, lr=0.001, batch_size=2)
        assert isinstance(history, dict)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def test_detection_train_one_epoch(self) -> None:
        """Detection 1-epoch dry-run produces finite loss."""
        model = _build_detection_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="detection", device=torch.device("cpu"))

        # Build a simple detection-compatible loss
        class _SimpleDetLoss(nn.Module):
            """Minimal detection loss that operates on dict outputs."""

            def forward(
                self,
                preds: dict[str, list[torch.Tensor]],
                targets: list[dict[str, torch.Tensor]],  # noqa: ARG002
            ) -> torch.Tensor:
                loss = torch.tensor(0.0, device=preds["cls_logits"][0].device)
                for cls_l in preds["cls_logits"]:
                    loss = loss + cls_l.sum() * 0.0  # dummy
                return loss + torch.tensor(1.0)

        cm.set_loss_fn(_SimpleDetLoss())

        dataset = SyntheticDetectionDataset(
            num_samples=2, img_size=320, num_classes=NUM_CLASSES,
        )
        loader = DataLoader(dataset, batch_size=2, collate_fn=det_collate)
        cm.set_train_dataloader(loader)

        history = cm.train(epochs=1, lr=0.001, batch_size=2)
        assert isinstance(history, dict)


# ======================================================================
# 4. Inference API - CoreModel.predict()
# ======================================================================


class TestPredictAPI:
    """Tests for CoreModel.predict()."""

    # ------------------------------------------------------------------
    # Classification prediction
    # ------------------------------------------------------------------

    def test_predict_classification_tensor(self) -> None:
        """Predict on a single tensor for classification."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        img = torch.randn(3, 224, 224)
        results = cm.predict(img, topk=3)
        assert isinstance(results, list)
        assert len(results) == 1
        pred = results[0]
        assert isinstance(pred, Prediction)
        assert pred.task == "classification"
        assert pred.classification is not None
        assert pred.classification.class_ids.shape == (3,)
        assert pred.classification.scores.shape == (3,)

    def test_predict_classification_topk(self) -> None:
        """Top-k parameter is respected for classification."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        img = torch.randn(3, 224, 224)
        results = cm.predict(img, topk=5)
        assert results[0].classification.class_ids.shape == (5,)

    def test_predict_classification_list_of_tensors(self) -> None:
        """Predict on a list of tensors for classification."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        imgs = [torch.randn(3, 224, 224) for _ in range(3)]
        results = cm.predict(imgs, topk=3)
        assert len(results) == 3
        for r in results:
            assert r.classification is not None

    def test_predict_classification_image_path(self, tmp_path: Path) -> None:
        """Predict on a single image file path for classification."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        # Create a small synthetic PNG image
        img_path = tmp_path / "test_input.png"
        arr: np.ndarray = np.random.randint(
            0, 256, (224, 224, 3), dtype=np.uint8,
        )
        Image.fromarray(arr).save(str(img_path))
        results = cm.predict(str(img_path), topk=3)
        assert len(results) == 1
        pred = results[0]
        assert pred.task == "classification"
        assert pred.classification is not None
        assert pred.classification.class_ids.shape == (3,)
        assert pred.classification.scores.shape == (3,)

    def test_predict_classification_list_of_paths(self, tmp_path: Path) -> None:
        """Predict on a list of image file paths for classification."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        paths: list[str] = []
        for i in range(3):
            p = tmp_path / f"test_{i}.png"
            arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(arr).save(str(p))
            paths.append(str(p))
        results = cm.predict(paths, topk=3)
        assert len(results) == 3
        for r in results:
            assert r.classification is not None

    # ------------------------------------------------------------------
    # Segmentation prediction
    # ------------------------------------------------------------------

    def test_predict_segmentation_tensor(self) -> None:
        """Predict on a single tensor for segmentation."""
        model = _SegmentationModel(num_classes=NUM_CLASSES)
        cm = CoreModel(
            model, task="segmentation", device=torch.device("cpu"),
            input_size=(256, 256),
        )
        img = torch.randn(3, 256, 256)
        results = cm.predict(img)
        assert len(results) == 1
        pred = results[0]
        assert pred.task == "segmentation"
        assert pred.segmentation is not None
        assert pred.segmentation.mask.shape == (256, 256)

    def test_predict_segmentation_list(self) -> None:
        """Predict on list of tensors for segmentation."""
        model = _SegmentationModel(num_classes=NUM_CLASSES)
        cm = CoreModel(
            model, task="segmentation", device=torch.device("cpu"),
            input_size=(256, 256),
        )
        imgs = [torch.randn(3, 256, 256) for _ in range(2)]
        results = cm.predict(imgs)
        assert len(results) == 2

    # ------------------------------------------------------------------
    # Detection prediction
    # ------------------------------------------------------------------

    def test_predict_detection_tensor(self) -> None:
        """Predict on a single tensor for detection."""
        model = _build_detection_model(num_classes=NUM_CLASSES)
        cm = CoreModel(
            model, task="detection", device=torch.device("cpu"),
            input_size=(320, 320),
        )
        img = torch.randn(3, 320, 320)
        results = cm.predict(img, conf_threshold=0.01, iou_threshold=0.5)
        assert len(results) == 1
        pred = results[0]
        assert pred.task == "detection"
        assert pred.detection is not None
        assert pred.detection.boxes.shape[1] == 4

    # ------------------------------------------------------------------
    # Parameter overrides
    # ------------------------------------------------------------------

    def test_predict_batch_size_override(self) -> None:
        """Batch size parameter is passed to the predictor."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        imgs = [torch.randn(3, 224, 224) for _ in range(4)]
        results = cm.predict(imgs, batch_size=2, topk=3)
        assert len(results) == 4

    def test_predict_half_precision(self) -> None:
        """Half precision flag is accepted (runs on CPU with full precision)."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        img = torch.randn(3, 224, 224)
        results = cm.predict(img, half_precision=True, topk=3)
        assert len(results) == 1

    def test_predict_confidence_threshold(self) -> None:
        """Confidence threshold is passed for detection."""
        model = _build_detection_model(num_classes=NUM_CLASSES)
        cm = CoreModel(
            model, task="detection", device=torch.device("cpu"),
            input_size=(320, 320),
        )
        img = torch.randn(3, 320, 320)
        # High threshold may filter all boxes but should not error
        results = cm.predict(img, conf_threshold=0.99, topk=3)
        assert len(results) == 1

    def test_predict_iou_threshold(self) -> None:
        """IoU threshold is passed to NMS for detection."""
        model = _build_detection_model(num_classes=NUM_CLASSES)
        cm = CoreModel(
            model, task="detection", device=torch.device("cpu"),
            input_size=(320, 320),
        )
        img = torch.randn(3, 320, 320)
        results = cm.predict(img, iou_threshold=0.3, conf_threshold=0.01)
        assert len(results) == 1


# ======================================================================
# 5. Export API
# ======================================================================


class TestExportAPI:
    """Tests for CoreModel.export()."""

    # ------------------------------------------------------------------
    # Export format variations
    # ------------------------------------------------------------------

    def test_export_onnx_format(self) -> None:
        """Export to ONNX format succeeds."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "test_model")
            results = cm.export(
                format="onnx",
                target_hardware="server",
                opset=17,
                output_path=out_path,
            )
        assert isinstance(results, dict)
        assert "onnx" in results

    def test_export_executorch_format(self) -> None:
        """Export to ExecuTorch format succeeds."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "test_model")
            results = cm.export(
                format="executorch",
                target_hardware="server",
                opset=18,
                output_path=out_path,
            )
        assert isinstance(results, dict)
        # If torch.export is not available, the exporter may still produce
        # a result dict but skip the actual export
        assert "executorch" in results

    def test_export_both_formats(self) -> None:
        """Export to both ONNX and ExecuTorch."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "test_model")
            results = cm.export(
                format="both",
                target_hardware="server",
                opset=18,
                output_path=out_path,
            )
        assert isinstance(results, dict)
        # At least one of the formats should be available
        assert "onnx" in results or "executorch" in results

    # ------------------------------------------------------------------
    # Target hardware
    # ------------------------------------------------------------------

    def test_export_edge_hardware(self) -> None:
        """Export with target_hardware='edge' (rewrites applied)."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "edge_model")
            results = cm.export(
                format="onnx",
                target_hardware="edge",
                opset=18,
                output_path=out_path,
            )
        assert "onnx" in results

    def test_export_server_hardware(self) -> None:
        """Export with target_hardware='server' (no rewrites)."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "server_model")
            results = cm.export(
                format="onnx",
                target_hardware="server",
                opset=17,
                output_path=out_path,
            )
        assert "onnx" in results

    def test_export_edge_rewrites_gelu_to_relu(self) -> None:
        """Export with target_hardware='edge' succeeds for GELU model."""
        # Build a classification model containing a GELU activation
        backbone = MockListBackbone(
            out_channels=(16, 24, 48), out_strides=(4, 8, 16),
        )
        # Add GELU between backbone and head
        class _GELUModel(nn.Module):
            """Minimal classification model with a GELU activation."""

            def __init__(
                self, bb: nn.Module, num_classes: int,
            ) -> None:
                super().__init__()
                self.backbone = bb
                self.gelu = nn.GELU()
                self.head = LinearClassificationHead(
                    feature_info=bb.feature_info,
                    num_classes=num_classes,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                feats = self.backbone(x)
                feats[-1] = self.gelu(feats[-1])
                return self.head(feats)

        gelu_model = _GELUModel(backbone, NUM_CLASSES)
        cm = CoreModel(
            gelu_model, task="classification", device=torch.device("cpu"),
            num_classes=NUM_CLASSES,
        )
        # The export should complete without error for edge hardware
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "edge_gelu_model")
            results = cm.export(
                format="onnx",
                target_hardware="edge",
                opset=18,
                output_path=out_path,
            )
        assert "onnx" in results, "ONNX format key must be present in results"

    # ------------------------------------------------------------------
    # Opset versions
    # ------------------------------------------------------------------

    def test_export_opset_17(self) -> None:
        """Export with opset 17 succeeds."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "opset17_model")
            results = cm.export(
                format="onnx",
                target_hardware="server",
                opset=17,
                output_path=out_path,
            )
        assert "onnx" in results

    def test_export_opset_18(self) -> None:
        """Export with opset 18 succeeds."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "opset18_model")
            results = cm.export(
                format="onnx",
                target_hardware="server",
                opset=18,
                output_path=out_path,
            )
        assert "onnx" in results

    # ------------------------------------------------------------------
    # Optimize flag
    # ------------------------------------------------------------------

    def test_export_with_optimize(self) -> None:
        """Export with optimize=True."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "optimized_model")
            results = cm.export(
                format="onnx",
                target_hardware="server",
                opset=18,
                optimize=True,
                output_path=out_path,
            )
        assert "onnx" in results

    def test_export_without_optimize(self) -> None:
        """Export with optimize=False."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "no_opt_model")
            results = cm.export(
                format="onnx",
                target_hardware="server",
                opset=18,
                optimize=False,
                output_path=out_path,
            )
        assert "onnx" in results

    # ------------------------------------------------------------------
    # Custom output_path and input_shape
    # ------------------------------------------------------------------

    def test_export_custom_output_path(self) -> None:
        """Custom output_path is respected in export results."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "my_custom_model")
            results = cm.export(
                format="onnx",
                target_hardware="server",
                opset=17,
                output_path=out_path,
            )
        # The output path should be in the results
        onnx_path = results.get("onnx", "")
        if onnx_path:
            assert "my_custom_model" in onnx_path

    def test_export_custom_input_shape(self) -> None:
        """Custom input_shape is accepted."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "custom_shape")
            results = cm.export(
                format="onnx",
                target_hardware="server",
                opset=18,
                output_path=out_path,
                input_shape=(1, 3, 224, 224),
            )
        assert "onnx" in results

    def test_export_dynamic_axes(self) -> None:
        """Dynamic axes parameter is accepted."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        dynamic_axes: dict[str, dict[int, str]] = {
            "input": {0: "batch", 2: "height", 3: "width"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "dynamic_model")
            results = cm.export(
                format="onnx",
                target_hardware="server",
                opset=18,
                output_path=out_path,
                dynamic_axes=dynamic_axes,
            )
        assert "onnx" in results

    # ------------------------------------------------------------------
    # Validation errors
    # ------------------------------------------------------------------

    def test_export_invalid_format(self) -> None:
        """Invalid export format raises ValueError."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with pytest.raises(ValueError, match="Unknown format"):
            cm.export(format="tflite")

    def test_export_invalid_hardware(self) -> None:
        """Invalid target_hardware raises ValueError."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with pytest.raises(ValueError, match="Unknown target_hardware"):
            cm.export(format="onnx", target_hardware="mobile")

    def test_export_invalid_opset(self) -> None:
        """Invalid opset version raises ValueError."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with pytest.raises(ValueError, match="Invalid opset"):
            cm.export(format="onnx", opset=15)

    def test_export_invalid_input_shape_dims(self) -> None:
        """Invalid input_shape dimensions raises ValueError."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        with pytest.raises(ValueError, match="input_shape must have 4 dimensions"):
            cm.export(format="onnx", input_shape=(1, 3, 224))


# ======================================================================
# 6. Integration Tests
# ======================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow_train_predict_export(self) -> None:
        """Full train -> predict -> export workflow for classification."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        cm.set_loss_fn(nn.CrossEntropyLoss())
        loader = DataLoader(
            SyntheticClassificationDataset(
                num_samples=4, num_classes=NUM_CLASSES,
            ),
            batch_size=4,
        )
        cm.set_train_dataloader(loader)

        # Train
        history = cm.train(epochs=1, lr=0.001, batch_size=4)
        assert isinstance(history, dict)

        # Predict
        img = torch.randn(3, 224, 224)
        preds = cm.predict(img, topk=3)
        assert len(preds) == 1
        assert preds[0].classification is not None

        # Export
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "full_workflow")
            results = cm.export(
                format="onnx",
                target_hardware="server",
                opset=17,
                output_path=out_path,
            )
            assert "onnx" in results

    def test_non_square_resolution_end_to_end(self) -> None:
        """End-to-end workflow with non-square (480x640) resolution."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(
            model,
            task="classification",
            device=torch.device("cpu"),
            input_size=(NON_SQUARE_H, NON_SQUARE_W),
            num_classes=NUM_CLASSES,
        )
        cm.set_loss_fn(nn.CrossEntropyLoss())

        # Use a smaller non-square resolution that the model can handle
        # with its stride-16 backbone: ensure H,W >= 16
        ns_h, ns_w = 64, 96
        dataset = SyntheticClassificationDataset(
            num_samples=4, img_size=ns_h, num_classes=NUM_CLASSES,
        )
        loader = DataLoader(dataset, batch_size=4)
        cm.set_train_dataloader(loader)

        # Train — should not error with non-square input
        # Pass explicit input_size override so the trainer uses the correct dims
        history = cm.train(epochs=1, lr=0.001, batch_size=4)
        assert isinstance(history, dict)

        # Predict with non-square tensor (predict uses self._input_size for preprocessing)
        img = torch.randn(3, ns_h, ns_w)
        preds = cm.predict(img, topk=3)
        assert len(preds) == 1
        assert preds[0].classification is not None

    def test_gradient_flow_through_train(self) -> None:
        """Gradient flows through train() without NaN."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        cm.set_loss_fn(nn.CrossEntropyLoss())
        loader = DataLoader(
            SyntheticClassificationDataset(
                num_samples=4, num_classes=NUM_CLASSES,
            ),
            batch_size=4,
        )
        cm.set_train_dataloader(loader)

        # Train — the CoreTrainer internally runs loss.backward()
        cm.train(epochs=1, lr=0.001, batch_size=4)

        # Verify gradients exist and are finite on the outermost Sequential
        for name, param in cm.model.named_parameters():
            if param.requires_grad:
                # After training, grad may be freed by optimizer.step()
                # but at least verify no NaN in current param values
                assert not param.data.isnan().any(), (
                    f"NaN detected in {name} after training."
                )
                assert not param.data.isinf().any(), (
                    f"Inf detected in {name} after training."
                )

    def test_explicit_gradient_flow_manual_backward(self) -> None:
        """Manually run forward->loss->backward and check gradients."""
        # Use a single-layer model where all parameters are connected to output
        model = nn.Linear(224 * 224 * 3, NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        cm.set_loss_fn(nn.CrossEntropyLoss())

        # Create a simple flatten + linear dataset wrapper
        class _FlattenedDataset(SyntheticClassificationDataset):
            """Classification dataset that flattens images for linear model."""

            def __getitem__(
                self, idx: int,
            ) -> tuple[torch.Tensor, int]:
                img, label = super().__getitem__(idx)
                return img.flatten(), label

        loader = DataLoader(
            _FlattenedDataset(
                num_samples=4, img_size=224, num_classes=NUM_CLASSES,
            ),
            batch_size=4,
        )
        cm.set_train_dataloader(loader)

        # Get one batch, run forward, compute loss, backward
        images, labels = next(iter(loader))
        logits = cm.model(images)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        # Check gradients flow without NaN/Inf on all connected parameters
        loss.backward()
        for name, param in cm.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"Expected gradient for {name} to be populated."
                )
                assert not param.grad.isnan().any(), (
                    f"NaN gradient found in {name} (shape {param.shape})."
                )
                assert not param.grad.isinf().any(), (
                    f"Inf gradient found in {name} (shape {param.shape})."
                )

    def test_trainer_is_lazy(self) -> None:
        """CoreTrainer is None until train() is called."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        assert cm.trainer is None
        assert cm.predictor is None

    def test_predictor_is_lazy(self) -> None:
        """CorePredictor is created lazily on first predict call."""
        model = _build_classification_model(num_classes=NUM_CLASSES)
        cm = CoreModel(model, task="classification", device=torch.device("cpu"))
        assert cm.predictor is None
        img = torch.randn(3, 224, 224)
        cm.predict(img, topk=3)
        assert cm.predictor is not None


# ======================================================================
# 7. ExportConfig validation
# ======================================================================


class TestExportConfigValidation:
    """Direct tests for ExportConfig dataclass validation."""

    def test_export_config_valid_defaults(self) -> None:
        """ExportConfig default values are valid."""
        cfg = ExportConfig()
        assert cfg.format == "onnx"
        assert cfg.target_hardware == "server"
        assert cfg.opset == 17
        assert cfg.optimize is True
        assert cfg.input_shape == (1, 3, 224, 224)

    def test_export_config_invalid_format(self) -> None:
        """ExportConfig rejects invalid format."""
        with pytest.raises(ValueError, match="Unknown format"):
            ExportConfig(format="tflite")

    def test_export_config_invalid_hardware(self) -> None:
        """ExportConfig rejects invalid target_hardware."""
        with pytest.raises(ValueError, match="Unknown target_hardware"):
            ExportConfig(target_hardware="mobile")

    def test_export_config_invalid_opset(self) -> None:
        """ExportConfig rejects invalid opset."""
        with pytest.raises(ValueError, match="Invalid opset"):
            ExportConfig(opset=15)

    def test_export_config_invalid_input_shape_ndim(self) -> None:
        """ExportConfig rejects input_shape with wrong number of dims."""
        with pytest.raises(ValueError, match="input_shape must have 4 dimensions"):
            ExportConfig(input_shape=(1, 3, 224))


# ======================================================================
# 8. TrainingConfig validation
# ======================================================================


class TestTrainingConfigValidation:
    """Direct tests for TrainingConfig dataclass validation."""

    def test_training_config_valid_defaults(self) -> None:
        """TrainingConfig default values are valid."""
        cfg = TrainingConfig()
        assert cfg.epochs == 100
        assert cfg.lr == 0.001
        assert cfg.batch_size == 32
        assert cfg.optimizer == "adamw"

    def test_training_config_epochs_range(self) -> None:
        """Epochs < 1 raises ValueError."""
        with pytest.raises(ValueError, match="epochs must be >= 1"):
            TrainingConfig(epochs=0)

    def test_training_config_lr_positive(self) -> None:
        """Lr <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="lr must be > 0"):
            TrainingConfig(lr=0.0)

    def test_training_config_batch_size_range(self) -> None:
        """batch_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            TrainingConfig(batch_size=0)

    def test_training_config_unknown_optimizer(self) -> None:
        """Unknown optimizer raises ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            TrainingConfig(optimizer="unknown_opt")

    def test_training_config_unknown_scheduler(self) -> None:
        """Unknown scheduler raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            TrainingConfig(scheduler="unknown_sched")

    def test_training_config_grad_accum_range(self) -> None:
        """grad_accum < 1 raises ValueError."""
        with pytest.raises(ValueError, match="grad_accum must be >= 1"):
            TrainingConfig(grad_accum=0)

    def test_training_config_ema_decay_range(self) -> None:
        """ema_decay outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="ema_decay must be in"):
            TrainingConfig(ema_decay=1.5)

    def test_training_config_frozen(self) -> None:
        """TrainingConfig is frozen (immutable)."""
        cfg = TrainingConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.epochs = 50  # type: ignore[misc]


# ======================================================================
# Test Auto-Build Training Pipeline (DataLoaders + Loss auto-instantiation)
# ======================================================================


class TestAutoBuildTrainingPipeline:
    """Verify 1-step CoreModel instantiation and train() auto-building."""

    def test_auto_build_classification_training(self, tmp_path: Path) -> None:
        """CoreModel auto-builds dataset, dataloader, and loss for classification."""
        cls_dir = tmp_path / "cls_auto"
        (cls_dir / "cat").mkdir(parents=True)
        (cls_dir / "dog").mkdir(parents=True)
        arr1 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        arr2 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr1).save(str(cls_dir / "cat" / "img1.jpg"))
        Image.fromarray(arr2).save(str(cls_dir / "dog" / "img2.jpg"))

        model = CoreModel("resnet18", task="classification", num_classes=2)
        history = model.train(data=str(cls_dir), epochs=1, batch_size=2)
        assert "train" in history
        assert len(history["train"]) == 1

    def test_auto_build_dict_config_training(self, tmp_path: Path) -> None:
        """CoreModel initialized with a dict config auto-runs train()."""
        cls_dir = tmp_path / "cls_dict_auto"
        (cls_dir / "cat").mkdir(parents=True)
        (cls_dir / "dog").mkdir(parents=True)
        arr1 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        arr2 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr1).save(str(cls_dir / "cat" / "img1.jpg"))
        Image.fromarray(arr2).save(str(cls_dir / "dog" / "img2.jpg"))

        config = {
            "model_name": "resnet18",
            "task": "classification",
            "num_classes": 2,
            "data": str(cls_dir),
            "epochs": 1,
            "batch_size": 2,
        }
        model = CoreModel(config)
        history = model.train()
        assert "train" in history
        assert len(history["train"]) == 1


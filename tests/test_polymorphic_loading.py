"""Tests for polymorphic loading in CoreModel.

Validates:
- CoreModel("weights.pt") → checkpoint loading
- CoreModel("config.yaml") → YAML config loading
- CoreModel(nn.Module) → backward compatibility
- CoreModel.from_pretrained("path/to/best.pt") → classmethod
- model.predict(..., weights="weights.pt") → on-the-fly weight loading
- model.export(..., weights="weights.pt") → on-the-fly weight loading before export
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from corecv.api import CoreModel
from corecv.models.backbones.resnet import ResNet18Backbone
from corecv.models.heads.classification import LinearClassificationHead

# ======================================================================
# Test Fixtures & Helpers
# ======================================================================


class MockModel(nn.Module):
    """Simple mock model for testing."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize the mock model."""
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(16, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and head."""
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


def _create_dummy_checkpoint(
    path: Path,
    model: nn.Module,
    model_config: dict | None = None,
) -> None:
    """Create a dummy checkpoint file for testing."""
    checkpoint = {
        "epoch": 10,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": None,
        "ema_state_dict": {},
        "scaler_state_dict": {},
        "metrics": {"train": [], "val": []},
    }
    if model_config is not None:
        checkpoint["model_config"] = model_config
    torch.save(checkpoint, path)


def _create_dummy_yaml_config(path: Path, task: str = "classification") -> None:
    """Create a dummy YAML config file for testing."""
    config: dict[str, object] = {
        "task": task,
        "model_name": "resnet18",
        "num_classes": 5,
        "head_type": "linear_classification",
        "pretrained": False,
        "dropout": 0.0,
        "label_smoothing": 0.0,
        "input_size": [224, 224],
    }
    with path.open("w") as f:
        yaml.dump(config, f)


# ======================================================================
# Test Classes
# ======================================================================


class TestPolymorphicInstantiation:
    """Test CoreModel instantiation with different argument types."""

    def test_coremodel_from_nn_module(self) -> None:
        """CoreModel(nn.Module) - backward compatibility."""
        model = MockModel(num_classes=10)
        coremodel = CoreModel(model, task="classification", input_size=(224, 224))
        assert isinstance(coremodel.model, nn.Module)
        assert coremodel.task == "classification"

    def test_coremodel_from_checkpoint_path(self, tmp_path: Path) -> None:
        """CoreModel('weights.pt') - loads from checkpoint."""
        model = MockModel(num_classes=5)
        checkpoint_path = tmp_path / "weights.pt"
        _create_dummy_checkpoint(
            checkpoint_path,
            model,
            model_config={
                "task": "classification",
                "model_name": "resnet18",
                "num_classes": 5,
                "backbone": "resnet18",
                "head": "linear",
                "input_size": [224, 224],
            },
        )

        coremodel = CoreModel(str(checkpoint_path), task="classification")
        assert isinstance(coremodel.model, nn.Module)
        assert coremodel.task == "classification"
        assert coremodel.num_classes == 5

    def test_coremodel_from_yaml_config_path(self, tmp_path: Path) -> None:
        """CoreModel('config.yaml') - loads from YAML config."""
        config_path = tmp_path / "config.yaml"
        _create_dummy_yaml_config(config_path, task="classification")

        coremodel = CoreModel(str(config_path), task="classification")
        assert isinstance(coremodel.model, nn.Module)
        assert coremodel.task == "classification"

    def test_coremodel_from_pathlib_path(self, tmp_path: Path) -> None:
        """CoreModel(Path('weights.pt')) - accepts pathlib.Path."""
        model = MockModel(num_classes=3)
        checkpoint_path = tmp_path / "weights.pt"
        _create_dummy_checkpoint(
            checkpoint_path,
            model,
            model_config={"task": "classification", "model_name": "resnet18", "num_classes": 3},
        )

        coremodel = CoreModel(checkpoint_path, task="classification")
        assert coremodel.num_classes == 3


class TestFromPretrainedClassmethod:
    """Test CoreModel.from_pretrained() classmethod."""

    def test_from_pretrained_loads_checkpoint(self, tmp_path: Path) -> None:
        """from_pretrained() loads checkpoint and rebuilds architecture."""
        model = MockModel(num_classes=7)
        checkpoint_path = tmp_path / "best.pt"
        _create_dummy_checkpoint(
            checkpoint_path,
            model,
            model_config={
                "task": "classification",
                "model_name": "resnet18",
                "num_classes": 7,
                "backbone": "resnet18",
                "head": "linear",
                "input_size": [224, 224],
            },
        )

        coremodel = CoreModel.from_pretrained(str(checkpoint_path))
        assert isinstance(coremodel, CoreModel)
        assert coremodel.task == "classification"
        assert coremodel.num_classes == 7

    def test_from_pretrained_raises_on_missing_model_config(self, tmp_path: Path) -> None:
        """from_pretrained() raises KeyError if model_config missing."""
        model = MockModel(num_classes=10)
        checkpoint_path = tmp_path / "no_config.pt"
        _create_dummy_checkpoint(checkpoint_path, model, model_config=None)

        with pytest.raises(KeyError, match="model_config"):
            CoreModel.from_pretrained(str(checkpoint_path))

    def test_from_pretrained_raises_on_nonexistent_file(self) -> None:
        """from_pretrained() raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            CoreModel.from_pretrained("nonexistent.pt")

    def test_from_pretrained_device_argument(self, tmp_path: Path) -> None:
        """from_pretrained() respects device argument."""
        model = MockModel(num_classes=4)
        checkpoint_path = tmp_path / "best.pt"
        _create_dummy_checkpoint(
            checkpoint_path,
            model,
            model_config={"task": "classification", "model_name": "resnet18", "num_classes": 4},
        )

        coremodel = CoreModel.from_pretrained(str(checkpoint_path), device=torch.device("cpu"))
        assert coremodel.device.type == "cpu"


class TestPredictWithWeights:
    """Test model.predict(source, weights=...) on-the-fly loading."""

    def test_predict_weights_loads_checkpoint(self, tmp_path: Path) -> None:
        """predict(weights=...) loads checkpoint before inference."""
        model = MockModel(num_classes=3)
        checkpoint_path = tmp_path / "weights.pt"
        _create_dummy_checkpoint(
            checkpoint_path,
            model,
            model_config={"task": "classification", "num_classes": 3, "input_size": [224, 224]},
        )

        coremodel = CoreModel(model, task="classification", input_size=(224, 224))
        source = torch.randn(3, 224, 224)

        preds = coremodel.predict(source, weights=str(checkpoint_path))
        assert len(preds) == 1
        assert hasattr(preds[0], "classification")

    def test_predict_weights_overrides_existing_weights(self, tmp_path: Path) -> None:
        """predict(weights=...) overrides model weights temporarily."""
        model1 = MockModel(num_classes=5)
        model2 = MockModel(num_classes=5)

        checkpoint_path = tmp_path / "weights.pt"
        _create_dummy_checkpoint(
            checkpoint_path,
            model2,
            model_config={"task": "classification", "num_classes": 5, "input_size": [224, 224]},
        )

        coremodel = CoreModel(model1, task="classification", input_size=(224, 224))
        source = torch.randn(3, 224, 224)

        # Should not raise even though model1 != model2 architecture
        preds = coremodel.predict(source, weights=str(checkpoint_path))
        assert len(preds) == 1

    def test_predict_without_weights_uses_current_model(self) -> None:
        """predict() without weights uses current model weights."""
        model = MockModel(num_classes=4)
        coremodel = CoreModel(model, task="classification", input_size=(224, 224))
        source = torch.randn(3, 224, 224)

        preds = coremodel.predict(source)
        assert len(preds) == 1


class TestExportWithWeights:
    """Test model.export(..., weights=...) on-the-fly loading."""

    def test_export_weights_loads_checkpoint(self, tmp_path: Path) -> None:
        """export(weights=...) loads checkpoint before export."""
        model = MockModel(num_classes=3)
        checkpoint_path = tmp_path / "weights.pt"
        _create_dummy_checkpoint(
            checkpoint_path,
            model,
            model_config={"task": "classification", "num_classes": 3, "input_size": [224, 224]},
        )

        coremodel = CoreModel(model, task="classification", input_size=(224, 224))

        # Export to ONNX (will fail if ONNX not installed, but should attempt load)
        with patch("corecv.api.model.CoreExporter") as mock_exporter_class:
            mock_exporter = MagicMock()
            mock_exporter.run_export.return_value = {"onnx": str(tmp_path / "model.onnx")}
            mock_exporter_class.return_value = mock_exporter

            coremodel.export(format="onnx", weights=str(checkpoint_path))

            # Verify exporter was called with the loaded model
            mock_exporter_class.assert_called_once()

    def test_export_without_weights_uses_current_model(self, tmp_path: Path) -> None:
        """export() without weights uses current model."""
        model = MockModel(num_classes=4)
        coremodel = CoreModel(model, task="classification", input_size=(224, 224))

        with patch("corecv.api.model.CoreExporter") as mock_exporter_class:
            mock_exporter = MagicMock()
            mock_exporter.run_export.return_value = {"onnx": str(tmp_path / "model.onnx")}
            mock_exporter_class.return_value = mock_exporter

            coremodel.export(format="onnx")
            mock_exporter_class.assert_called_once()


class TestBackwardCompatibility:
    """Test that existing CoreModel(nn.Module) usage still works."""

    def test_existing_api_still_works(self) -> None:
        """Original CoreModel(nn.Module, task=..., input_size=...) works."""
        model = MockModel(num_classes=10)
        coremodel = CoreModel(model, task="classification", input_size=(224, 224))

        assert coremodel.model is model
        assert coremodel.task == "classification"
        assert coremodel.input_size == (224, 224)

    def test_fluent_setters_still_work(self) -> None:
        """Fluent setters (set_loss_fn, set_train_dataloader, etc.) still work."""
        model = MockModel(num_classes=5)
        coremodel = CoreModel(model, task="classification", input_size=(224, 224))

        loss_fn = nn.CrossEntropyLoss()
        dataset = TensorDataset(
            torch.randn(4, 3, 224, 224),
            torch.randint(0, 5, (4,)),
        )
        train_loader = DataLoader(dataset, batch_size=2)

        coremodel.set_loss_fn(loss_fn).set_train_dataloader(train_loader)
        assert coremodel._loss_fn is loss_fn
        assert coremodel._train_loader is train_loader

    def test_train_kwargs_style_still_works(self) -> None:
        """model.train(epochs=..., lr=..., ...) still works."""
        model = MockModel(num_classes=3)
        coremodel = CoreModel(model, task="classification", input_size=(64, 64))

        loss_fn = nn.CrossEntropyLoss()
        dataset = TensorDataset(
            torch.randn(8, 3, 64, 64),
            torch.randint(0, 3, (8,)),
        )
        train_loader = DataLoader(dataset, batch_size=4)
        coremodel.set_loss_fn(loss_fn).set_train_dataloader(train_loader)

        # Should run without error (1 epoch, fast)
        history = coremodel.train(epochs=1, lr=0.01, batch_size=4, device="cpu")
        assert "train" in history
        assert "val" in history


class TestErrorHandling:
    """Test error handling for polymorphic loading."""

    def test_invalid_extension_raises(self) -> None:
        """CoreModel('file.txt') raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model file extension"):
            CoreModel("file.txt", task="classification")

    def test_nonexistent_checkpoint_raises(self) -> None:
        """CoreModel('nonexistent.pt') raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CoreModel("nonexistent.pt", task="classification")

    def test_nonexistent_yaml_raises(self) -> None:
        """CoreModel('nonexistent.yaml') raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CoreModel("nonexistent.yaml", task="classification")

    def test_predict_invalid_weights_path_raises(self) -> None:
        """predict(weights='nonexistent.pt') raises FileNotFoundError."""
        model = MockModel(num_classes=3)
        coremodel = CoreModel(model, task="classification", input_size=(224, 224))

        with pytest.raises(FileNotFoundError):
            coremodel.predict(torch.randn(1, 3, 224, 224), weights="nonexistent.pt")

    def test_export_invalid_weights_path_raises(self) -> None:
        """export(weights='nonexistent.pt') raises FileNotFoundError."""
        model = MockModel(num_classes=3)
        coremodel = CoreModel(model, task="classification", input_size=(224, 224))

        with pytest.raises(FileNotFoundError):
            coremodel.export(format="onnx", weights="nonexistent.pt")


class TestCheckpointConfigExtraction:
    """Test that model_config is properly extracted and used."""

    def test_checkpoint_model_config_contains_task(self, tmp_path: Path) -> None:
        """Checkpoint model_config contains task info."""
        model = MockModel(num_classes=5)
        checkpoint_path = tmp_path / "weights.pt"
        config = {"task": "classification", "model_name": "resnet18", "num_classes": 5}
        _create_dummy_checkpoint(checkpoint_path, model, model_config=config)

        coremodel = CoreModel(str(checkpoint_path), task="classification")
        assert coremodel.task == "classification"

    def test_checkpoint_model_config_contains_num_classes(self, tmp_path: Path) -> None:
        """Checkpoint model_config num_classes overrides default."""
        model = MockModel(num_classes=10)
        checkpoint_path = tmp_path / "weights.pt"
        config = {"task": "classification", "model_name": "resnet18", "num_classes": 7}
        _create_dummy_checkpoint(checkpoint_path, model, model_config=config)

        coremodel = CoreModel(str(checkpoint_path), task="classification")
        assert coremodel.num_classes == 7

    def test_explicit_num_classes_overrides_checkpoint(self, tmp_path: Path) -> None:
        """Explicit num_classes parameter overrides checkpoint config."""
        model = MockModel(num_classes=10)
        checkpoint_path = tmp_path / "weights.pt"
        config = {"task": "classification", "model_name": "resnet18", "num_classes": 7}
        _create_dummy_checkpoint(checkpoint_path, model, model_config=config)

        coremodel = CoreModel(str(checkpoint_path), task="classification", num_classes=12)
        assert coremodel.num_classes == 12


class TestIntegrationWithRealBackbone:
    """Integration test with real registered backbone (ResNet18)."""

    def test_from_pretrained_with_resnet18(self, tmp_path: Path) -> None:
        """from_pretrained works with ResNet18 backbone from registry."""
        # Build a real model using registry
        backbone = ResNet18Backbone(pretrained=False)
        head = LinearClassificationHead(feature_info=backbone.feature_info, num_classes=4)
        model = nn.Sequential(backbone, head)
        model.num_classes = 4

        checkpoint_path = tmp_path / "resnet.pt"
        _create_dummy_checkpoint(
            checkpoint_path,
            model,
            model_config={
                "task": "classification",
                "model_name": "resnet18",
                "num_classes": 4,
                "backbone": "resnet18",
                "head": "linear",
                "input_size": [224, 224],
            },
        )

        coremodel = CoreModel.from_pretrained(str(checkpoint_path))
        assert coremodel.task == "classification"
        assert coremodel.num_classes == 4

        # Forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = coremodel.model(x)
        assert out.shape == (1, 4)


# ======================================================================
# Pytest Configuration
# ======================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

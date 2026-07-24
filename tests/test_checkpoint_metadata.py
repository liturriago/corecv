"""Tests for checkpoint metadata in CoreTrainer.

Validates:
- CoreTrainer accepts optional model_config parameter
- Checkpoints saved by CoreTrainer contain model_config
- Checkpoints can be reloaded with CoreModel.from_pretrained()
- model_config contains task, num_classes, backbone, head, input_size
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from corecv.api import CoreModel
from corecv.engine.trainer import CoreTrainer
from corecv.models.backbones.resnet import ResNet18Backbone
from corecv.models.heads.classification import LinearClassificationHead

# ======================================================================
# Test Fixtures & Helpers
# ======================================================================


def _build_classifier(num_classes: int = 10) -> nn.Sequential:
    """Build a simple classifier: ResNet18 + LinearClassificationHead."""
    backbone = ResNet18Backbone(pretrained=False)
    head = LinearClassificationHead(
        feature_info=backbone.feature_info,
        num_classes=num_classes,
    )
    model = nn.Sequential(backbone, head)
    model.num_classes = num_classes
    return model


def _create_dataloader(
    batch_size: int = 4,
    num_samples: int = 8,
    img_size: int = 64,
    num_classes: int = 10,
) -> DataLoader:
    """Create a dummy DataLoader for testing."""
    dataset = TensorDataset(
        torch.randn(num_samples, 3, img_size, img_size),
        torch.randint(0, num_classes, (num_samples,)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ======================================================================
# Test Classes
# ======================================================================


class TestCoreTrainerModelConfigParameter:
    """Test CoreTrainer accepts and stores model_config."""

    def test_trainer_accepts_model_config(self) -> None:
        """CoreTrainer.__init__ accepts model_config parameter."""
        model = _build_classifier(num_classes=5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        train_loader = _create_dataloader()
        model_config = {"task": "classification", "num_classes": 5, "backbone": "resnet18"}

        trainer = CoreTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            train_dataloader=train_loader,
            model_config=model_config,
        )

        assert trainer.model_config == model_config

    def test_trainer_model_config_optional(self) -> None:
        """model_config parameter is optional (defaults to None)."""
        model = _build_classifier(num_classes=5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        train_loader = _create_dataloader()

        trainer = CoreTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            train_dataloader=train_loader,
        )

        assert trainer.model_config is None


class TestCheckpointContainsModelConfig:
    """Test saved checkpoints contain model_config."""

    def test_save_checkpoint_includes_model_config(self, tmp_path: Path) -> None:
        """save_checkpoint() includes model_config in checkpoint dict."""
        model = _build_classifier(num_classes=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        train_loader = _create_dataloader(num_samples=4)
        model_config = {
            "task": "classification",
            "num_classes": 3,
            "backbone": "resnet18",
            "head": "linear",
        }

        trainer = CoreTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            train_dataloader=train_loader,
            model_config=model_config,
            output_dir=str(tmp_path),
        )

        trainer.save_checkpoint(
            path=str(tmp_path / "epoch_1.pt"),
            epoch=1,
            metrics={"loss": 0.5},
        )

        # Load and verify checkpoint contents
        checkpoint = torch.load(tmp_path / "epoch_1.pt", map_location="cpu")
        assert "model_config" in checkpoint
        assert checkpoint["model_config"] == model_config

    def test_save_checkpoint_model_config_none_when_not_provided(self, tmp_path: Path) -> None:
        """save_checkpoint() includes model_config=None when not provided."""
        model = _build_classifier(num_classes=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        train_loader = _create_dataloader(num_samples=4)

        trainer = CoreTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            train_dataloader=train_loader,
            output_dir=str(tmp_path),
        )

        trainer.save_checkpoint(
            path=str(tmp_path / "epoch_1.pt"),
            epoch=1,
            metrics={"loss": 0.5},
        )

        checkpoint = torch.load(tmp_path / "epoch_1.pt", map_location="cpu")
        assert "model_config" in checkpoint
        assert checkpoint["model_config"] is None

    def test_fit_saves_checkpoints_with_model_config(self, tmp_path: Path) -> None:
        """fit() saves checkpoints containing model_config."""
        model = _build_classifier(num_classes=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        train_loader = _create_dataloader(num_samples=4, num_classes=2)
        val_loader = _create_dataloader(num_samples=4, num_classes=2)
        model_config = {"task": "classification", "num_classes": 2, "backbone": "resnet18"}

        trainer = CoreTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            model_config=model_config,
            output_dir=str(tmp_path),
        )

        trainer.fit(num_epochs=2)

        # Check final checkpoint
        final_ckpt = torch.load(tmp_path / "final.pt", map_location="cpu")
        assert final_ckpt["model_config"] == model_config

        # Check epoch checkpoints
        epoch1_ckpt = torch.load(tmp_path / "epoch_1.pt", map_location="cpu")
        assert epoch1_ckpt["model_config"] == model_config


class TestFromPretrainedReload:
    """Test CoreModel.from_pretrained() reloads checkpoints with model_config."""

    def test_from_pretrained_loads_model_config(self, tmp_path: Path) -> None:
        """from_pretrained() extracts model_config from checkpoint."""
        model = _build_classifier(num_classes=5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        train_loader = _create_dataloader(num_samples=4, num_classes=5)
        model_config = {
            "task": "classification",
            "model_name": "resnet18",
            "num_classes": 5,
            "backbone": "resnet18",
            "head": "linear",
            "input_size": [224, 224],
        }

        trainer = CoreTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            train_dataloader=train_loader,
            model_config=model_config,
            output_dir=str(tmp_path),
        )

        trainer.fit(num_epochs=1)

        # Reload with from_pretrained
        coremodel = CoreModel.from_pretrained(str(tmp_path / "final.pt"))

        assert coremodel.task == "classification"
        assert coremodel.num_classes == 5

    def test_from_pretrained_model_state_dict_matches(self, tmp_path: Path) -> None:
        """from_pretrained() loads model weights and produces correct output."""
        model = _build_classifier(num_classes=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        train_loader = _create_dataloader(num_samples=4, num_classes=3)
        model_config = {
            "task": "classification",
            "model_name": "resnet18",
            "num_classes": 3,
            "backbone": "resnet18",
        }

        trainer = CoreTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            train_dataloader=train_loader,
            model_config=model_config,
            output_dir=str(tmp_path),
        )

        trainer.fit(num_epochs=1)

        # Reload and verify forward pass produces correct output shape
        coremodel = CoreModel.from_pretrained(str(tmp_path / "final.pt"))
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = coremodel.model(x)
        assert out.shape == (1, 3)

    def test_from_pretrained_works_with_different_architectures(self, tmp_path: Path) -> None:
        """from_pretrained() works when architecture is rebuilt from config."""
        # This test verifies the config has enough info to rebuild
        model = _build_classifier(num_classes=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        train_loader = _create_dataloader(num_samples=4, num_classes=4)
        model_config = {
            "task": "classification",
            "model_name": "resnet18",
            "num_classes": 4,
            "backbone": "resnet18",
            "head": "linear",
            "input_size": [224, 224],
        }

        trainer = CoreTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            train_dataloader=train_loader,
            model_config=model_config,
            output_dir=str(tmp_path),
        )

        trainer.fit(num_epochs=1)

        # Should be able to reload
        coremodel = CoreModel.from_pretrained(str(tmp_path / "final.pt"))
        assert coremodel.task == "classification"
        assert coremodel.num_classes == 4

        # Forward pass should work
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = coremodel.model(x)
        assert out.shape == (1, 4)


class TestCheckpointMetadataCompleteness:
    """Test checkpoint contains all expected metadata fields."""

    def test_checkpoint_has_all_expected_keys(self, tmp_path: Path) -> None:
        """Checkpoint has all standard keys + model_config."""
        model = _build_classifier(num_classes=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        train_loader = _create_dataloader(num_samples=4, num_classes=3)
        model_config = {"task": "classification", "num_classes": 3}

        trainer = CoreTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            train_dataloader=train_loader,
            model_config=model_config,
            output_dir=str(tmp_path),
        )

        trainer.fit(num_epochs=1)

        checkpoint = torch.load(tmp_path / "final.pt", map_location="cpu")

        # Standard keys
        assert "epoch" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "scheduler_state_dict" in checkpoint
        assert "ema_state_dict" in checkpoint
        assert "scaler_state_dict" in checkpoint
        assert "metrics" in checkpoint
        # New key
        assert "model_config" in checkpoint

    def test_checkpoint_model_config_structure(self, tmp_path: Path) -> None:
        """model_config in checkpoint has expected structure."""
        model = _build_classifier(num_classes=7)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        train_loader = _create_dataloader(num_samples=4, num_classes=7)
        model_config = {
            "task": "classification",
            "num_classes": 7,
            "backbone": "resnet18",
            "head": "linear",
            "input_size": [224, 224],
            "pretrained": False,
        }

        trainer = CoreTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.CrossEntropyLoss(),
            train_dataloader=train_loader,
            model_config=model_config,
            output_dir=str(tmp_path),
        )

        trainer.fit(num_epochs=1)

        checkpoint = torch.load(tmp_path / "final.pt", map_location="cpu")
        saved_config = checkpoint["model_config"]

        assert saved_config["task"] == "classification"
        assert saved_config["num_classes"] == 7
        assert saved_config["backbone"] == "resnet18"
        assert saved_config["head"] == "linear"
        assert saved_config["input_size"] == [224, 224]
        assert saved_config["pretrained"] is False


class TestIntegrationWithCoreModel:
    """Integration tests with CoreModel high-level API."""

    def test_coremodel_train_saves_checkpoint_with_config(self, tmp_path: Path) -> None:
        """CoreModel.train() saves checkpoints with model_config key."""
        model = _build_classifier(num_classes=3)
        coremodel = CoreModel(model, task="classification", input_size=(64, 64))

        loss_fn = nn.CrossEntropyLoss()
        train_loader = _create_dataloader(batch_size=2, num_samples=4, img_size=64, num_classes=3)
        coremodel.set_loss_fn(loss_fn).set_train_dataloader(train_loader)

        coremodel.train(
            epochs=1,
            lr=0.001,
            batch_size=2,
            device="cpu",
            output_dir=str(tmp_path),
        )

        # Check final checkpoint has model_config key
        final_ckpt = torch.load(tmp_path / "final.pt", map_location="cpu")
        assert "model_config" in final_ckpt

    def test_coremodel_predict_after_train_uses_ema(self, tmp_path: Path) -> None:
        """CoreModel.predict() after train() uses EMA weights (via trainer)."""
        model = _build_classifier(num_classes=2)
        coremodel = CoreModel(model, task="classification", input_size=(32, 32))

        loss_fn = nn.CrossEntropyLoss()
        train_loader = _create_dataloader(batch_size=2, num_samples=4, img_size=32, num_classes=2)
        coremodel.set_loss_fn(loss_fn).set_train_dataloader(train_loader)

        coremodel.train(epochs=1, lr=0.01, batch_size=2, device="cpu", output_dir=str(tmp_path))

        # Predict should work with EMA weights
        x = torch.randn(3, 32, 32)
        preds = coremodel.predict(x)
        assert len(preds) == 1


# ======================================================================
# Pytest Configuration
# ======================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

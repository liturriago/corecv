"""Comprehensive tests for the CoreCV CLI interface.

Tests cover:

1. **Help commands** — ``--help`` for the app and all subcommands.
2. **Train command** — required/optional args, error cases, config delegation.
3. **Predict command** — required/optional args, source types, error cases.
4. **Export command** — format/hardware/opset options, error cases, delegation.
5. **Integration** — Verify CLI commands delegate correctly to CoreModel API.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from typer.testing import CliRunner

from corecv.api.model import CoreModel
from corecv.cli.main import app

# ======================================================================
# Constants
# ======================================================================

NUM_CLASSES: int = 10

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def runner() -> CliRunner:
    """Return a CliRunner for invoking the Typer app."""
    return CliRunner()


@pytest.fixture
def mock_model_path(tmp_path: Path) -> Path:
    """Create a dummy model file that exists on disk for Typer path validation."""
    model_path = tmp_path / "model.pkl"
    model_path.write_text("dummy", encoding="utf-8")
    return model_path


@pytest.fixture
def mock_model() -> MagicMock:
    """Return a mock CoreModel instance with default return values."""
    model = MagicMock(spec=CoreModel)
    model.task = "classification"
    model.num_classes = NUM_CLASSES
    model.train.return_value = {"train": [{"loss": 0.5}]}
    model.predict.return_value = []
    model.export.return_value = {"onnx": "/tmp/model.onnx"}
    return model


@pytest.fixture(autouse=True)
def patch_load_model(mock_model: MagicMock):
    """Patch _load_model to return a mock, avoiding pickle I/O in tests."""
    with patch("corecv.cli.main._load_model", return_value=mock_model) as mock:
        yield mock


# ======================================================================
# 1. Help commands
# ======================================================================


class TestHelpCommands:
    """Tests for --help output on the app and all subcommands."""

    def test_app_help(self, runner: CliRunner) -> None:
        """CoreCV --help displays app-level help with commands list."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "CoreCV" in result.stdout
        assert "train" in result.stdout
        assert "predict" in result.stdout
        assert "export" in result.stdout

    def test_train_help(self, runner: CliRunner) -> None:
        """CoreCV train --help displays train command help."""
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--epochs" in result.stdout
        assert "--lr" in result.stdout

    def test_predict_help(self, runner: CliRunner) -> None:
        """CoreCV predict --help displays predict command help."""
        result = runner.invoke(app, ["predict", "--help"])
        assert result.exit_code == 0
        assert "--conf" in result.stdout
        assert "--topk" in result.stdout

    def test_export_help(self, runner: CliRunner) -> None:
        """CoreCV export --help displays export command help."""
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.stdout
        assert "--opset" in result.stdout

    def test_no_args_shows_help(self, runner: CliRunner) -> None:
        """CoreCV with no args shows help text."""
        result = runner.invoke(app, [])
        # Typer raises SystemExit(2) for no_args_is_help; verify help text
        # is displayed regardless.
        assert "CoreCV" in result.stdout
        assert "train" in result.stdout
        assert "predict" in result.stdout
        assert "export" in result.stdout


# ======================================================================
# 2. Train command
# ======================================================================


class TestTrainCommand:
    """Tests for the train subcommand."""

    def test_train_minimal(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Train with only the required MODEL_PATH argument."""
        result = runner.invoke(app, ["train", str(mock_model_path)])
        assert result.exit_code == 0
        mock_model.train.assert_called_once()

    def test_train_epochs(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Train with --epochs specified and verify the config."""
        result = runner.invoke(
            app, ["train", str(mock_model_path), "--epochs", "50"],
        )
        assert result.exit_code == 0
        _, kwargs = mock_model.train.call_args
        assert kwargs["config"].epochs == 50

    def test_train_lr(
        self,
        runner: CliRunner,
        mock_model_path: Path,
    ) -> None:
        """Train with --lr specified."""
        result = runner.invoke(
            app, ["train", str(mock_model_path), "--lr", "0.01"],
        )
        assert result.exit_code == 0

    def test_train_batch_size(
        self,
        runner: CliRunner,
        mock_model_path: Path,
    ) -> None:
        """Train with --batch-size specified."""
        result = runner.invoke(
            app, ["train", str(mock_model_path), "-b", "64"],
        )
        assert result.exit_code == 0

    def test_train_optimizer(
        self,
        runner: CliRunner,
        mock_model_path: Path,
    ) -> None:
        """Train with --optimizer specified."""
        result = runner.invoke(
            app, ["train", str(mock_model_path), "--optimizer", "adam"],
        )
        assert result.exit_code == 0

    def test_train_scheduler_none(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """String 'none' scheduler is converted to None in config."""
        result = runner.invoke(
            app, ["train", str(mock_model_path), "--scheduler", "none"],
        )
        assert result.exit_code == 0
        _, kwargs = mock_model.train.call_args
        assert kwargs["config"].scheduler is None

    def test_train_no_amp(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Train with --no-amp sets amp to False."""
        result = runner.invoke(
            app, ["train", str(mock_model_path), "--no-amp"],
        )
        assert result.exit_code == 0
        _, kwargs = mock_model.train.call_args
        assert not kwargs["config"].amp

    def test_train_all_options(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Train with all optional arguments and verify config."""
        result = runner.invoke(app, [
            "train", str(mock_model_path),
            "--epochs", "10",
            "--lr", "0.001",
            "--batch-size", "16",
            "--optimizer", "adamw",
            "--scheduler", "cosine",
            "--amp",
            "--grad-accum", "4",
            "--ema-decay", "0.99",
            "--output-dir", "./runs",
            "--device", "cpu",
            "--verbose",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.train.call_args
        config = kwargs["config"]
        assert config.epochs == 10
        assert config.lr == 0.001
        assert config.batch_size == 16
        assert config.optimizer == "adamw"
        assert config.scheduler == "cosine"
        assert config.amp
        assert config.grad_accum == 4
        assert config.device == "cpu"

    def test_train_verbose(
        self,
        runner: CliRunner,
        mock_model_path: Path,
    ) -> None:
        """Train with --verbose flag succeeds."""
        result = runner.invoke(
            app, ["train", str(mock_model_path), "--verbose"],
        )
        assert result.exit_code == 0

    def test_train_missing_model_path(
        self,
        runner: CliRunner,
    ) -> None:
        """Train without MODEL_PATH exits with error (missing required arg)."""
        result = runner.invoke(app, ["train"])
        assert result.exit_code != 0

    def test_train_model_not_found(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Train with non-existent model file shows Typer validation error."""
        missing = tmp_path / "nonexistent.pkl"
        result = runner.invoke(app, ["train", str(missing)])
        # Typer (via Click) rejects non-existent paths with exit code 2
        assert result.exit_code == 2

    def test_train_invalid_epochs_type(
        self,
        runner: CliRunner,
        mock_model_path: Path,
    ) -> None:
        """Train with non-integer --epochs raises Typer error."""
        result = runner.invoke(
            app, ["train", str(mock_model_path), "--epochs", "abc"],
        )
        assert result.exit_code != 0


# ======================================================================
# 3. Predict command
# ======================================================================


class TestPredictCommand:
    """Tests for the predict subcommand."""

    def test_predict_minimal(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Predict with required args only."""
        result = runner.invoke(
            app, ["predict", str(mock_model_path), "image.jpg"],
        )
        assert result.exit_code == 0
        mock_model.predict.assert_called_once()

    def test_predict_conf_threshold(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Predict with --conf threshold."""
        result = runner.invoke(app, [
            "predict", str(mock_model_path), "image.jpg",
            "--conf", "0.5",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.predict.call_args
        assert kwargs["conf_threshold"] == 0.5

    def test_predict_iou_threshold(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Predict with --iou threshold."""
        result = runner.invoke(app, [
            "predict", str(mock_model_path), "image.jpg",
            "--iou", "0.45",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.predict.call_args
        assert kwargs["iou_threshold"] == 0.45

    def test_predict_topk(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Predict with --topk."""
        result = runner.invoke(app, [
            "predict", str(mock_model_path), "image.jpg",
            "--topk", "5",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.predict.call_args
        assert kwargs["topk"] == 5

    def test_predict_half(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Predict with --half precision flag."""
        result = runner.invoke(app, [
            "predict", str(mock_model_path), "image.jpg",
            "--half",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.predict.call_args
        assert kwargs["half_precision"] is True

    def test_predict_compile(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Predict with --compile flag."""
        result = runner.invoke(app, [
            "predict", str(mock_model_path), "image.jpg",
            "--compile",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.predict.call_args
        assert kwargs["compile_model"] is True

    def test_predict_batch_size(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Predict with --batch-size."""
        result = runner.invoke(app, [
            "predict", str(mock_model_path), "image.jpg",
            "--batch-size", "16",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.predict.call_args
        assert kwargs["batch_size"] == 16

    def test_predict_output_json(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        tmp_path: Path,
    ) -> None:
        """Predict with --output saves predictions to JSON file."""
        output_file = tmp_path / "predictions.json"
        result = runner.invoke(app, [
            "predict", str(mock_model_path), "image.jpg",
            "--output", str(output_file),
        ])
        assert result.exit_code == 0
        assert output_file.exists()
        data = json.loads(output_file.read_text(encoding="utf-8"))
        assert isinstance(data, list)

    def test_predict_all_options(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Predict with all optional arguments."""
        result = runner.invoke(app, [
            "predict", str(mock_model_path), "image.jpg",
            "--conf", "0.25",
            "--iou", "0.5",
            "--topk", "10",
            "--half",
            "--compile",
            "--batch-size", "32",
            "--verbose",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.predict.call_args
        assert kwargs["conf_threshold"] == 0.25
        assert kwargs["iou_threshold"] == 0.5
        assert kwargs["topk"] == 10
        assert kwargs["half_precision"] is True
        assert kwargs["compile_model"] is True
        assert kwargs["batch_size"] == 32

    def test_predict_verbose(
        self,
        runner: CliRunner,
        mock_model_path: Path,
    ) -> None:
        """Predict with --verbose flag succeeds."""
        result = runner.invoke(app, [
            "predict", str(mock_model_path), "image.jpg",
            "--verbose",
        ])
        assert result.exit_code == 0

    def test_predict_missing_source(
        self,
        runner: CliRunner,
        mock_model_path: Path,
    ) -> None:
        """Predict without SOURCE argument exits with error."""
        result = runner.invoke(app, ["predict", str(mock_model_path)])
        assert result.exit_code != 0

    def test_predict_missing_model_path(
        self,
        runner: CliRunner,
    ) -> None:
        """Predict without MODEL_PATH exits with error."""
        result = runner.invoke(app, ["predict"])
        assert result.exit_code != 0

    def test_predict_directory_source(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        tmp_path: Path,
    ) -> None:
        """Predict with a directory path as source."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        result = runner.invoke(app, [
            "predict", str(mock_model_path), str(img_dir),
        ])
        assert result.exit_code == 0


# ======================================================================
# 4. Export command
# ======================================================================


class TestExportCommand:
    """Tests for the export subcommand."""

    def test_export_minimal(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Export with required args only."""
        result = runner.invoke(app, ["export", str(mock_model_path)])
        assert result.exit_code == 0
        mock_model.export.assert_called_once()

    def test_export_format_onnx(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Export with --format onnx."""
        result = runner.invoke(app, [
            "export", str(mock_model_path), "--format", "onnx",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.export.call_args
        assert kwargs["format"] == "onnx"

    def test_export_format_executorch(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Export with --format executorch."""
        result = runner.invoke(app, [
            "export", str(mock_model_path), "--format", "executorch",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.export.call_args
        assert kwargs["format"] == "executorch"

    def test_export_format_both(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Export with --format both."""
        result = runner.invoke(app, [
            "export", str(mock_model_path), "--format", "both",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.export.call_args
        assert kwargs["format"] == "both"

    def test_export_hardware_edge(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Export with --target-hardware edge."""
        result = runner.invoke(app, [
            "export", str(mock_model_path), "--target-hardware", "edge",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.export.call_args
        assert kwargs["target_hardware"] == "edge"

    def test_export_hardware_server(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Export with --target-hardware server."""
        result = runner.invoke(app, [
            "export", str(mock_model_path), "--target-hardware", "server",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.export.call_args
        assert kwargs["target_hardware"] == "server"

    def test_export_opset_17(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Export with --opset 17."""
        result = runner.invoke(app, [
            "export", str(mock_model_path), "--opset", "17",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.export.call_args
        assert kwargs["opset"] == 17

    def test_export_opset_18(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Export with --opset 18."""
        result = runner.invoke(app, [
            "export", str(mock_model_path), "--opset", "18",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.export.call_args
        assert kwargs["opset"] == 18

    def test_export_no_optimize(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Export with --no-optimize."""
        result = runner.invoke(app, [
            "export", str(mock_model_path), "--no-optimize",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.export.call_args
        assert kwargs["optimize"] is False

    def test_export_input_shape(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Export with --input-shape as comma-separated values."""
        result = runner.invoke(app, [
            "export", str(mock_model_path),
            "--input-shape", "1,3,224,224",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.export.call_args
        assert kwargs["input_shape"] == (1, 3, 224, 224)

    def test_export_dynamic_axes(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Export with --dynamic-axes flag builds dynamic axes dict."""
        result = runner.invoke(app, [
            "export", str(mock_model_path), "--dynamic-axes",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.export.call_args
        assert kwargs["dynamic_axes"] is not None
        assert "input" in kwargs["dynamic_axes"]

    def test_export_custom_output(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Export with --output path."""
        result = runner.invoke(app, [
            "export", str(mock_model_path), "--output", "./custom_export",
        ])
        assert result.exit_code == 0
        _, kwargs = mock_model.export.call_args
        assert kwargs["output_path"] == "./custom_export"

    def test_export_verbose(
        self,
        runner: CliRunner,
        mock_model_path: Path,
    ) -> None:
        """Export with --verbose flag succeeds."""
        result = runner.invoke(app, [
            "export", str(mock_model_path), "--verbose",
        ])
        assert result.exit_code == 0

    def test_export_missing_model_path(
        self,
        runner: CliRunner,
    ) -> None:
        """Export without MODEL_PATH exits with error."""
        result = runner.invoke(app, ["export"])
        assert result.exit_code != 0


# ======================================================================
# 5. Integration — Command delegation to CoreModel
# ======================================================================


class TestCommandDelegation:
    """Verify CLI commands delegate correctly to CoreModel API methods."""

    def test_train_delegates_to_model_train(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Train command calls CoreModel.train with TrainingConfig."""
        result = runner.invoke(app, [
            "train", str(mock_model_path),
            "--epochs", "5",
            "--lr", "0.01",
        ])
        assert result.exit_code == 0
        mock_model.train.assert_called_once()
        _, kwargs = mock_model.train.call_args
        assert "config" in kwargs
        assert kwargs["config"].epochs == 5
        assert kwargs["config"].lr == 0.01

    def test_predict_delegates_to_model_predict(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Predict command calls CoreModel.predict with correct args."""
        result = runner.invoke(app, [
            "predict", str(mock_model_path), "test_image.jpg",
            "--topk", "3",
        ])
        assert result.exit_code == 0
        mock_model.predict.assert_called_once()
        _, kwargs = mock_model.predict.call_args
        assert kwargs["source"] == "test_image.jpg"
        assert kwargs["topk"] == 3

    def test_export_delegates_to_model_export(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Export command calls CoreModel.export with correct args."""
        result = runner.invoke(app, [
            "export", str(mock_model_path),
            "--format", "onnx",
            "--opset", "18",
        ])
        assert result.exit_code == 0
        mock_model.export.assert_called_once()
        _, kwargs = mock_model.export.call_args
        assert kwargs["format"] == "onnx"
        assert kwargs["opset"] == 18

    def test_predict_with_pt_source(
        self,
        runner: CliRunner,
        mock_model_path: Path,
        mock_model: MagicMock,
    ) -> None:
        """Predict resolves .pt files via _resolve_source."""
        dummy_tensor = torch.randn(1, 3, 224, 224)
        with patch(
            "corecv.cli.main._resolve_source",
            return_value=dummy_tensor,
        ) as mock_resolve:
            result = runner.invoke(app, [
                "predict", str(mock_model_path), "data.pt",
            ])
            assert result.exit_code == 0
            mock_resolve.assert_called_once_with("data.pt")
            _, kwargs = mock_model.predict.call_args
            assert kwargs["source"] is dummy_tensor

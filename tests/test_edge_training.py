"""Tests for edge-hardware training pipeline.

Verifies that when ``CoreModel.train()`` is called with
``target_hardware="edge"``, the :class:`TargetRewriter.rewrite_for_edge()`
is applied to the model **before** the optimizer is instantiated, so that
optimizer parameter groups point to the rewritten graph modules.

Requirements covered:
1.  Create a simple model with GELU/SiLU activations (edge-incompatible).
2.  Call ``model.train()`` with ``target_hardware="edge"``.
3.  Verify ``TargetRewriter`` was applied before optimizer creation:
    - GELU/SiLU modules are replaced with ReLU/Hardswish in the graph.
    - Optimizer parameters reference the rewritten modules (same object
      identity).
    - Training actually runs without errors (finite loss, no NaN/Inf).
4.  Test that ``target_hardware="server"`` does **not** apply rewrites.
"""

from __future__ import annotations

import tempfile

import pytest
import torch
from torch import nn
from torch.fx import GraphModule
from torch.utils.data import DataLoader, Dataset

from corecv.api import CoreModel
from corecv.engine.rewriter import TargetRewriter

# ======================================================================
# Constants
# ======================================================================

NUM_CLASSES: int = 10

# ======================================================================
# GELU / SiLU classification models (traceable by ``fx.symbolic_trace``)
# ======================================================================


class GeluClassifier(nn.Module):
    """Simple classification model with an ``nn.GELU`` activation.

    Architecture: Conv2d -> GELU -> Conv2d -> AdaptiveAvgPool -> Linear.
    This is traceable by ``torch.fx.symbolic_trace`` and contains an
    edge-incompatible activation.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        """Initialise layers.

        Args:
            num_classes: Number of output classes.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(N, 3, H, W)``.

        Returns:
            Classification logits ``(N, num_classes)``.
        """
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class SiluClassifier(nn.Module):
    """Simple classification model with an ``nn.SiLU`` activation.

    Architecture: Conv2d -> SiLU -> Conv2d -> AdaptiveAvgPool -> Linear.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        """Initialise layers.

        Args:
            num_classes: Number of output classes.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(N, 3, H, W)``.

        Returns:
            Classification logits ``(N, num_classes)``.
        """
        x = self.conv1(x)
        x = self.silu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class MixedActivationClassifier(nn.Module):
    """Model with both GELU and SiLU activations.

    Used to test that both edge-incompatible activations are replaced
    simultaneously.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        """Initialise layers.

        Args:
            num_classes: Number of output classes.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.silu = nn.SiLU()
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(N, 3, H, W)``.

        Returns:
            Classification logits ``(N, num_classes)``.
        """
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.silu(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# ======================================================================
# Synthetic dataset
# ======================================================================


class SyntheticClassificationDataset(Dataset):
    """Yields random images and integer class labels."""

    def __init__(
        self,
        num_samples: int = 4,
        img_size: int = 32,
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
            idx: Index (ignored, data is random).

        Returns:
            Tuple of ``(3, H, W)`` image tensor and integer label.
        """
        img = torch.randn(3, self.img_size, self.img_size)
        label = int(torch.randint(0, self.num_classes, ()).item())
        return img, label


# ======================================================================
# Helpers
# ======================================================================


def _check_no_gelu_silu_in_graph(model: nn.Module) -> None:
    """Assert that the model graph contains no GELU or SiLU module calls.

    Args:
        model: A ``GraphModule`` whose graph is inspected.

    Raises:
        AssertionError: If any ``call_module`` node targets a GELU or SiLU
            module, or if the model is not a ``GraphModule``.
    """
    assert isinstance(model, GraphModule), (
        f"Expected GraphModule after edge rewrite, got {type(model).__name__}"
    )

    for node in model.graph.nodes:
        if node.op == "call_module":
            try:
                mod = model.get_submodule(node.target)
            except AttributeError:
                continue
            if isinstance(mod, (nn.GELU, nn.SiLU)):
                pytest.fail(
                    f"Edge-incompatible activation {type(mod).__name__} "
                    f"found in rewritten graph at node '{node.name}'"
                )


def _check_edge_rewrites_applied(model: nn.Module) -> None:
    """Assert that edge-friendly replacements exist in the graph.

    Args:
        model: A ``GraphModule`` that has been rewritten for edge.

    Raises:
        AssertionError: If ReLU or Hardswish modules are not found, or
            the model is not a ``GraphModule``.
    """
    assert isinstance(model, GraphModule), (
        f"Expected GraphModule after edge rewrite, got {type(model).__name__}"
    )

    has_relu = False
    has_hardswish = False
    for node in model.graph.nodes:
        if node.op == "call_module":
            try:
                mod = model.get_submodule(node.target)
            except AttributeError:
                continue
            if isinstance(mod, nn.ReLU):
                has_relu = True
            if isinstance(mod, nn.Hardswish):
                has_hardswish = True

    # At least one edge-friendly replacement should be present
    assert has_relu or has_hardswish, (
        "No edge-friendly activation (ReLU or Hardswish) found in rewritten graph"
    )


def _verify_optimizer_params_match_model(
    core_model: CoreModel,
) -> None:
    """Assert that the trainer's optimizer params reference the model's params.

    This is the central verification that ``TargetRewriter`` was applied
    **before** the optimiser was built:
    ``optimizer.param_groups[0]['params']`` must contain the same Python
    objects as ``model.parameters()``.

    Args:
        core_model: A ``CoreModel`` instance after ``.train()`` has been
            called.

    Raises:
        AssertionError: If any parameter object differs.
    """
    assert core_model.trainer is not None, (
        "No trainer available — did train() complete?"
    )

    opt_params: list[torch.Tensor] = list(
        core_model.trainer.optimizer.param_groups[0]["params"]
    )
    model_params: list[torch.Tensor] = list(core_model.model.parameters())

    assert len(opt_params) == len(model_params), (
        f"Optimizer tracks {len(opt_params)} parameters but model has "
        f"{len(model_params)}"
    )

    for i, (opt_p, model_p) in enumerate(zip(opt_params, model_params, strict=True)):
        assert opt_p is model_p, (
            f"Parameter index {i} differs: optimizer param at "
            f"id={id(opt_p)} vs model param at id={id(model_p)}. "
            "This means the optimizer was built before the rewrite."
        )


def _make_core_model(
    model: nn.Module,
    num_classes: int = NUM_CLASSES,
    img_size: int = 32,
    num_samples: int = 4,
) -> CoreModel:
    """Build a fully-configured ``CoreModel`` ready for training.

    Args:
        model: The PyTorch model to wrap.
        num_classes: Number of classification classes.
        img_size: Spatial size of synthetic images.
        num_samples: Number of synthetic samples in the dataset.

    Returns:
        A ``CoreModel`` with a loss function, train dataloader, and
        CPU device set.
    """
    core = CoreModel(
        model,
        task="classification",
        device=torch.device("cpu"),
        num_classes=num_classes,
    )
    core.set_loss_fn(nn.CrossEntropyLoss())

    dataset = SyntheticClassificationDataset(
        num_samples=num_samples,
        img_size=img_size,
        num_classes=num_classes,
    )
    loader = DataLoader(dataset, batch_size=num_samples)
    core.set_train_dataloader(loader)

    return core


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Reset the random seed before every test for reproducibility."""
    torch.manual_seed(42)


# ======================================================================
# Tests: target_hardware "edge"
# ======================================================================


class TestEdgeTrainingActivationsReplaced:
    """Verify that ``target_hardware="edge"`` replaces GELU/SiLU."""

    def test_gelu_replaced_after_edge_training(self) -> None:
        """``nn.GELU`` is replaced with ``nn.ReLU`` after edge training."""
        core = _make_core_model(GeluClassifier())
        core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="edge",
            output_dir=tempfile.mkdtemp(),
        )

        rewritten = core.model
        _check_no_gelu_silu_in_graph(rewritten)
        _check_edge_rewrites_applied(rewritten)

    def test_silu_replaced_after_edge_training(self) -> None:
        """``nn.SiLU`` is replaced with ``nn.Hardswish`` after edge training."""
        core = _make_core_model(SiluClassifier())
        core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="edge",
            output_dir=tempfile.mkdtemp(),
        )

        rewritten = core.model
        _check_no_gelu_silu_in_graph(rewritten)

        # Verify Hardswish is present (SiLU -> Hardswish)
        has_hardswish = any(
            node.op == "call_module"
            and isinstance(
                rewritten.get_submodule(node.target), nn.Hardswish
            )
            for node in rewritten.graph.nodes
        )
        assert has_hardswish, (
            "nn.Hardswish should be present after replacing nn.SiLU"
        )

    def test_mixed_activations_replaced(self) -> None:
        """Both GELU and SiLU are replaced when present in the same model."""
        core = _make_core_model(MixedActivationClassifier())
        core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="edge",
            output_dir=tempfile.mkdtemp(),
        )

        rewritten = core.model
        _check_no_gelu_silu_in_graph(rewritten)
        _check_edge_rewrites_applied(rewritten)

        # Both ReLU and Hardswish should be present
        has_relu = False
        has_hardswish = False
        for node in rewritten.graph.nodes:
            if node.op == "call_module":
                mod = rewritten.get_submodule(node.target)
                if isinstance(mod, nn.ReLU):
                    has_relu = True
                if isinstance(mod, nn.Hardswish):
                    has_hardswish = True

        assert has_relu, "ReLU should be present (replaced GELU)"
        assert has_hardswish, "Hardswish should be present (replaced SiLU)"

    def test_output_is_graph_module(self) -> None:
        """After edge training, the model is a ``GraphModule``."""
        core = _make_core_model(GeluClassifier())
        core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="edge",
            output_dir=tempfile.mkdtemp(),
        )

        assert isinstance(core.model, GraphModule), (
            "Model should be a GraphModule after edge rewrite"
        )


class TestEdgeTrainingOptimizerParams:
    """Verify that the optimizer references rewritten model parameters."""

    def test_optimizer_params_match_rewritten_model_gelu(self) -> None:
        """Optimizer params are the same objects as rewritten GELU model params."""
        core = _make_core_model(GeluClassifier())
        core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="edge",
            output_dir=tempfile.mkdtemp(),
        )

        _verify_optimizer_params_match_model(core)

    def test_optimizer_params_match_rewritten_model_silu(self) -> None:
        """Optimizer params are the same objects as rewritten SiLU model params."""
        core = _make_core_model(SiluClassifier())
        core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="edge",
            output_dir=tempfile.mkdtemp(),
        )

        _verify_optimizer_params_match_model(core)

    def test_optimizer_params_match_rewritten_model_mixed(self) -> None:
        """Optimizer params match rewritten model for mixed-activation model."""
        core = _make_core_model(MixedActivationClassifier())
        core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="edge",
            output_dir=tempfile.mkdtemp(),
        )

        _verify_optimizer_params_match_model(core)

    def test_optimizer_param_groups_have_correct_length(self) -> None:
        """Number of optimizer params equals number of model params after rewrite."""
        core = _make_core_model(GeluClassifier())
        core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="edge",
            output_dir=tempfile.mkdtemp(),
        )

        opt_len = len(core.trainer.optimizer.param_groups[0]["params"])  # type: ignore[union-attr]
        model_len = len(list(core.model.parameters()))
        assert opt_len == model_len, (
            f"Optimizer manages {opt_len} params but model has {model_len}"
        )


class TestEdgeTrainingActuallyRuns:
    """Verify that edge training completes without errors and produces valid results."""

    def test_edge_training_produces_finite_loss(self) -> None:
        """Training with ``target_hardware='edge'`` produces finite loss."""
        core = _make_core_model(GeluClassifier())
        history = core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="edge",
            output_dir=tempfile.mkdtemp(),
        )

        assert isinstance(history, dict), "train() must return a dict"
        assert "train" in history, "History must contain 'train' key"
        train_metrics = history["train"]
        assert len(train_metrics) == 1, "Expected metrics for 1 epoch"
        loss_val = train_metrics[0].get("loss", float("inf"))
        assert torch.isfinite(torch.tensor(loss_val)), (
            f"Training loss must be finite, got {loss_val}"
        )

    def test_edge_training_non_square_resolution(self) -> None:
        """Edge training works with non-square (64x96) inputs to catch H/W bugs."""
        core = _make_core_model(
            GeluClassifier(),
            img_size=64,
            num_samples=4,
        )
        # Override input_size to be non-square
        core._input_size = (64, 96)  # noqa: SLF001

        # Create a non-square dataset
        class _NonSquareDataset(SyntheticClassificationDataset):
            """Dataset with non-square images."""

            def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
                img = torch.randn(3, 64, 96)
                label = int(torch.randint(0, self.num_classes, ()).item())
                return img, label

        dataset = _NonSquareDataset(
            num_samples=4, img_size=64, num_classes=NUM_CLASSES,
        )
        loader = DataLoader(dataset, batch_size=4)
        core.set_train_dataloader(loader)

        history = core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="edge",
            output_dir=tempfile.mkdtemp(),
        )

        train_metrics = history["train"][0]
        loss_val = train_metrics.get("loss", float("inf"))
        assert torch.isfinite(torch.tensor(loss_val)), (
            f"Non-square training loss must be finite, got {loss_val}"
        )

    def test_edge_training_no_nan_in_parameters(self) -> None:
        """After edge training, model parameters contain no NaN or Inf values."""
        core = _make_core_model(GeluClassifier())
        core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="edge",
            output_dir=tempfile.mkdtemp(),
        )

        for name, param in core.model.named_parameters():
            assert not param.data.isnan().any(), (
                f"NaN detected in parameter '{name}' after edge training"
            )
            assert not param.data.isinf().any(), (
                f"Inf detected in parameter '{name}' after edge training"
            )

    def test_edge_training_with_zero_vram_meta(self) -> None:
        """Edge rewrite + optimizer creation works on ``device='meta'`` (no training loop)."""
        model = GeluClassifier()
        core = CoreModel(
            model,
            task="classification",
            device=torch.device("meta"),
            num_classes=NUM_CLASSES,
        )
        core.set_loss_fn(nn.CrossEntropyLoss())
        dataset = SyntheticClassificationDataset(
            num_samples=4, img_size=32, num_classes=NUM_CLASSES,
        )
        loader = DataLoader(dataset, batch_size=4)
        core.set_train_dataloader(loader)

        # Apply rewrite manually (simulating what train() does before the loop)
        rewritten = TargetRewriter().rewrite_for_edge(core._model)
        core._model = rewritten

        # Build optimizer manually (simulating the pre-training pipeline)
        class _FakeCfg:
            """Minimal config surrogate for _build_optimizer."""
            optimizer: str = "adamw"
            lr: float = 0.01

        optim = core._build_optimizer(_FakeCfg())  # type: ignore[arg-type]

        # Verify the model is rewritten and optimizer params match
        assert isinstance(core.model, GraphModule), (
            "Model should be a GraphModule after edge rewrite on meta"
        )
        _check_no_gelu_silu_in_graph(core.model)
        _check_edge_rewrites_applied(core.model)

        opt_params = list(optim.param_groups[0]["params"])
        model_params = list(core.model.parameters())
        assert len(opt_params) == len(model_params)
        for i, (opt_p, model_p) in enumerate(
            zip(opt_params, model_params, strict=True)
        ):
            assert opt_p is model_p, (
                f"Parameter {i} object mismatch on meta device"
            )

    def test_edge_training_gradient_flow(self) -> None:
        """Gradients flow through the rewritten edge model without NaN."""
        core = _make_core_model(GeluClassifier())
        core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="edge",
            output_dir=tempfile.mkdtemp(),
        )

        # Manually run a forward-backward to verify gradient flow
        model = core.model
        model.train()
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient after backward"
            )
            assert not param.grad.isnan().any(), (
                f"NaN gradient in '{name}' after backward"
            )
            assert not param.grad.isinf().any(), (
                f"Inf gradient in '{name}' after backward"
            )


# ======================================================================
# Tests: target_hardware "server"  (no rewrites)
# ======================================================================


class TestServerTrainingNoRewrites:
    """Verify that ``target_hardware='server'`` does NOT apply rewrites."""

    def test_server_training_preserves_gelu(self) -> None:
        """``nn.GELU`` is preserved when training with ``target_hardware='server'``."""
        core = _make_core_model(GeluClassifier())
        core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="server",
            output_dir=tempfile.mkdtemp(),
        )

        model = core.model
        # Model should NOT be a GraphModule (no rewrite applied)
        assert not isinstance(model, GraphModule), (
            "Model should NOT be a GraphModule with server hardware"
        )

        # GELU should still be present
        has_gelu = any(
            isinstance(mod, nn.GELU)
            for mod in model.modules()
        )
        assert has_gelu, "GELU should still be present with server hardware"

    def test_server_training_preserves_silu(self) -> None:
        """``nn.SiLU`` is preserved when training with ``target_hardware='server'``."""
        core = _make_core_model(SiluClassifier())
        core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="server",
            output_dir=tempfile.mkdtemp(),
        )

        model = core.model
        assert not isinstance(model, GraphModule), (
            "Model should NOT be a GraphModule with server hardware"
        )

        has_silu = any(
            isinstance(mod, nn.SiLU)
            for mod in model.modules()
        )
        assert has_silu, "SiLU should still be present with server hardware"

    def test_server_training_preserves_mixed_activations(self) -> None:
        """Both GELU and SiLU are preserved with ``target_hardware='server'``."""
        core = _make_core_model(MixedActivationClassifier())
        core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="server",
            output_dir=tempfile.mkdtemp(),
        )

        model = core.model
        module_types = {type(mod) for mod in model.modules()}
        assert nn.GELU in module_types, "GELU should be present with server"
        assert nn.SiLU in module_types, "SiLU should be present with server"
        assert nn.ReLU not in module_types, (
            "ReLU should NOT be introduced with server hardware"
        )
        assert nn.Hardswish not in module_types, (
            "Hardswish should NOT be introduced with server hardware"
        )

    def test_server_training_optimizer_params_match_model(self) -> None:
        """With server hardware, optimizer params match the *original* model."""
        core = _make_core_model(GeluClassifier())
        core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="server",
            output_dir=tempfile.mkdtemp(),
        )

        # The model is not rewritten, but the optimizer should still point
        # to the correct parameters
        opt_params = list(
            core.trainer.optimizer.param_groups[0]["params"]  # type: ignore[union-attr]
        )
        model_params = list(core.model.parameters())

        assert len(opt_params) == len(model_params)
        for i, (opt_p, model_p) in enumerate(
            zip(opt_params, model_params, strict=True)
        ):
            assert opt_p is model_p, (
                f"Parameter {i} object mismatch with server hardware"
            )

    def test_server_training_produces_finite_loss(self) -> None:
        """Server training produces finite loss (baseline comparison)."""
        core = _make_core_model(GeluClassifier())
        history = core.train(
            epochs=1,
            lr=0.01,
            batch_size=4,
            target_hardware="server",
            output_dir=tempfile.mkdtemp(),
        )

        train_metrics = history["train"][0]
        loss_val = train_metrics.get("loss", float("inf"))
        assert torch.isfinite(torch.tensor(loss_val)), (
            f"Server training loss must be finite, got {loss_val}"
        )


# ======================================================================
# Tests: zero-VRAM with ``device='meta'``
# ======================================================================


class TestEdgeTrainingMetaDevice:
    """Edge training configuration on ``device='meta'`` (zero-VRAM)."""

    def test_edge_rewrite_on_meta_device(self) -> None:
        """``TargetRewriter.rewrite_for_edge`` works on a ``meta``-device model."""
        model = GeluClassifier().to("meta")
        rewriter = TargetRewriter()
        rewritten = rewriter.rewrite_for_edge(model)

        assert isinstance(rewritten, GraphModule)
        # Verify replacements were applied
        _check_no_gelu_silu_in_graph(rewritten)
        _check_edge_rewrites_applied(rewritten)

    def test_edge_train_on_meta_device_no_error(self) -> None:
        """Config resolution + rewrite + optimizer creation works on ``device='meta'``."""
        model = GeluClassifier()
        core = CoreModel(
            model,
            task="classification",
            device=torch.device("meta"),
            num_classes=NUM_CLASSES,
        )
        core.set_loss_fn(nn.CrossEntropyLoss())
        core.set_train_dataloader(
            DataLoader(
                SyntheticClassificationDataset(
                    num_samples=4, img_size=32, num_classes=NUM_CLASSES,
                ),
                batch_size=4,
            ),
        )

        # Apply rewrite and build optimizer manually for meta
        rewritten = TargetRewriter().rewrite_for_edge(core._model)
        core._model = rewritten

        class _FakeCfg:
            """Minimal config surrogate for _build_optimizer."""
            optimizer: str = "adamw"
            lr: float = 0.01

        optim = core._build_optimizer(_FakeCfg())  # type: ignore[arg-type]

        # Verify the model is a GraphModule and rewrites were applied
        assert isinstance(core.model, GraphModule)
        _check_no_gelu_silu_in_graph(core.model)
        _check_edge_rewrites_applied(core.model)

        # Verify optimizer param identity
        for opt_p, model_p in zip(
            optim.param_groups[0]["params"],
            core.model.parameters(),
            strict=True,
        ):
            assert opt_p is model_p


# ======================================================================
# Tests: direct TargetRewriter + optimizer integration
# ======================================================================


class TestDirectRewriterOptimizerIntegration:
    """Direct verification that ``TargetRewriter`` runs before optimiser creation.

    These tests bypass ``CoreModel`` and directly call the rewriter +
    optimiser pattern that ``CoreModel.train()`` follows internally.
    """

    def test_optimizer_after_rewrite_references_graph_module_params(self) -> None:
        """An optimizer created after ``rewrite_for_edge`` uses rewritten params."""
        model = GeluClassifier()
        rewriter = TargetRewriter()
        rewritten = rewriter.rewrite_for_edge(model)

        optim = torch.optim.SGD(rewritten.parameters(), lr=0.01)

        opt_params = list(optim.param_groups[0]["params"])
        model_params = list(rewritten.parameters())

        assert len(opt_params) == len(model_params)
        for i, (opt_p, model_p) in enumerate(
            zip(opt_params, model_params, strict=True)
        ):
            assert opt_p is model_p, (
                f"Parameter {i} should be the same object after rewrite"
            )

    def test_optimizer_before_rewrite_uses_original_params(self) -> None:
        """An optimizer created BEFORE rewrite references the *original* params.

        This is a negative test: if the optimiser is created first, its
        params will point to the old (unrewritten) module.  This
        demonstrates why rewrite-before-optimiser is essential.
        """
        model = GeluClassifier()
        original_params = list(model.parameters())

        # Create optimiser BEFORE rewrite (wrong order — params already captured)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)

        # Now apply rewrite (too late — optimiser already captured old params)
        TargetRewriter().rewrite_for_edge(model)

        # The optimiser still points to the original (unrewritten) params.
        # This is the BUG that the rewrite-before-optimiser order prevents.
        opt_params = list(optim.param_groups[0]["params"])
        assert len(opt_params) == len(original_params), (
            "Optimiser should track original parameter count"
        )
        for opt_p, orig_p in zip(opt_params, original_params, strict=True):
            assert opt_p is orig_p, (
                "Optimiser should reference original params when created before rewrite"
            )

    def test_rewrite_before_optimiser_identity_tracking(self) -> None:
        """Demonstrate that rewrite-before-optimiser yields correct identity.

        This test explicitly checks that after a correct rewrite-first
        pattern, the number of parameter groups and their identities are
        consistent between the rewritten model and the optimiser.
        """
        model = GeluClassifier()
        rewriter = TargetRewriter()
        rewritten = rewriter.rewrite_for_edge(model)

        # Capture rewritten parameter IDs
        rewritten_param_ids = {id(p) for p in rewritten.parameters()}

        # Create optimiser AFTER rewrite (correct order)
        optim = torch.optim.SGD(rewritten.parameters(), lr=0.01)
        opt_param_ids = {id(p) for p in optim.param_groups[0]["params"]}

        assert rewritten_param_ids == opt_param_ids, (
            "All rewritten model parameter IDs must be present in the optimiser"
        )


# ======================================================================
# Tests: non-square resolution (H/W indexing)
# ======================================================================


class TestEdgeTrainingNonSquare:
    """Edge training with non-square input resolutions."""

    def test_edge_rewrite_non_square_forward(self) -> None:
        """Rewritten model produces correct output shape on non-square (64x96) input."""
        model = GeluClassifier()
        rewriter = TargetRewriter()
        rewritten = rewriter.rewrite_for_edge(model)

        x = torch.randn(2, 3, 64, 96)
        output = rewritten(x)

        assert output.shape == (2, NUM_CLASSES), (
            f"Expected (2, {NUM_CLASSES}), got {output.shape}"
        )
        assert not torch.isnan(output).any(), "Output should not contain NaN"
        assert not torch.isinf(output).any(), "Output should not contain Inf"

    def test_edge_rewrite_non_square_gradient(self) -> None:
        """Gradients flow correctly through rewritten model on non-square input."""
        model = GeluClassifier().train()
        rewriter = TargetRewriter()
        rewritten = rewriter.rewrite_for_edge(model)
        rewritten.train()

        x = torch.randn(2, 3, 64, 96)
        output = rewritten(x)
        loss = output.sum()
        loss.backward()

        for name, param in rewritten.named_parameters():
            assert param.grad is not None, (
                f"No gradient for '{name}' on non-square input"
            )
            assert not param.grad.isnan().any(), (
                f"NaN gradient for '{name}' on non-square input"
            )


# ======================================================================
# Tests: error handling
# ======================================================================


class TestEdgeTrainingErrors:
    """Error handling for edge training configuration."""

    def test_invalid_target_hardware_raises_value_error(self) -> None:
        """An invalid ``target_hardware`` value raises ``ValueError``."""
        model = GeluClassifier()
        core = CoreModel(
            model,
            task="classification",
            device=torch.device("cpu"),
            num_classes=NUM_CLASSES,
        )

        with pytest.raises(
            ValueError,
            match="Unknown target_hardware",
        ):
            core.train(epochs=1, target_hardware="invalid_hw")  # type: ignore[arg-type]

    def test_edge_training_with_unsupported_task_raises(self) -> None:
        """Edge training on an unsupported task raises ``ValueError``."""
        with pytest.raises(ValueError, match="Unsupported task"):
            CoreModel(
                GeluClassifier(),
                task="unsupported_task",  # type: ignore[arg-type]
            )

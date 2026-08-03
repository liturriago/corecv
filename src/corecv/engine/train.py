"""Training engines for classification, segmentation, and detection tasks.

Provides :class:`Trainer` for managing end-to-end training loops with
GPU-native metric computation, optional LR scheduling, and per-epoch
history tracking.

Example::

    from corecv.engine.train import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_classes=10,
    )
    history = trainer.fit(num_epochs=10)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from corecv.metrics.metric_classification import ClassificationMetrics

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _batch_to_device(
    batch: dict[str, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    """Move all tensor values in a batch dictionary to the target device.

    Non-tensor values are passed through unchanged.

    Args:
        batch: Dictionary mapping string keys to tensors or other values.
        device: Target device for tensor transfer.

    Returns:
        A new dictionary with all tensor values placed on *device*.
    """
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Trainer for computer vision models.

    Manages the training and validation loops for models,
    including forward pass, loss computation, backpropagation, metric
    accumulation, and optional learning rate scheduling.

    Attributes:
        task: Task to train (classification, segmentation, detection).
        model: The model moved to the target device.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        loss_fn: Loss function accepting ``(logits, labels)``.
        optimizer: Optimizer for parameter updates.
        num_classes: Number of classes.
        device: Target device for computation.
        scheduler: Optional learning rate scheduler.
    """

    def __init__(  # noqa: PLR0913
        self,
        task: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        num_classes: int,
        device: str | torch.device = "cuda",
        scheduler: LRScheduler | None = None,
        use_amp: bool = False,
    ) -> None:
        """Initialize the trainer.

        Args:
            task: Task to train (classification, segmentation, detection).
            model: Model mapping ``(B, 3, H, W)`` to the task-specific output format.
            train_loader: Training data loader yielding dicts with
                ``"images"`` and ``"labels"`` keys.
            val_loader: Validation data loader yielding dicts with
                ``"images"`` and ``"labels"`` keys.
            loss_fn: Loss function accepting ``(logits, labels)`` and
                returning a scalar loss tensor.
            optimizer: Optimizer instance for gradient-based updates.
            num_classes: Number of target classes.
            device: Target device string or :class:`torch.device`.
            scheduler: Optional LR scheduler stepped after each epoch.
            use_amp: Whether to enable Automatic Mixed Precision (AMP).
        """
        self.task = task
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)
        
        if task == "classification":
            self._train_metrics = ClassificationMetrics(num_classes=num_classes)
            self._val_metrics = ClassificationMetrics(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown task: {task}")

    def train_epoch(self) -> dict[str, float]:
        """Execute a single training epoch.

        Returns:
            Dictionary of training metrics for this epoch.  Always
            contains ``"loss"`` plus all keys returned by
            :meth:`Metrics.compute`.
        """
        self.model.train()
        self._train_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        for raw_batch in self.train_loader:
            batch = _batch_to_device(raw_batch, self.device)
            images = batch["images"]
            labels = batch["labels"]

            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp, device_type=self.device.type):
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self._train_metrics.update(logits.detach(), labels)
            total_loss += loss.item()
            num_batches += 1

        metrics = self._train_metrics.compute()
        metrics["loss"] = total_loss / max(num_batches, 1)
        return metrics

    def validate(self) -> dict[str, float]:
        """Run validation and compute metrics.

        Returns:
            Dictionary of validation metrics.  Always contains ``"loss"``
            plus all keys returned by :meth:`Metrics.compute`.
        """
        self.model.eval()
        self._val_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for raw_batch in self.val_loader:
                batch = _batch_to_device(raw_batch, self.device)
                images = batch["images"]
                labels = batch["labels"]

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, labels)

                self._val_metrics.update(logits, labels)
                total_loss += loss.item()
                num_batches += 1

        metrics = self._val_metrics.compute()
        metrics["loss"] = total_loss / max(num_batches, 1)
        return metrics

    def fit(self, num_epochs: int) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            num_epochs: Total number of training epochs to execute.

        Returns:
            Dictionary mapping metric names (prefixed with ``"train_"``
            or ``"val_"``) to lists of per-epoch float values.
        """
        history: dict[str, list[float]] = defaultdict(list)

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            for key, value in train_metrics.items():
                history[f"train_{key}"].append(value)
            for key, value in val_metrics.items():
                history[f"val_{key}"].append(value)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f",
                epoch,
                num_epochs,
                train_metrics["loss"],
                val_metrics["loss"],
            )

            self._train_metrics.print_results("Train")
            self._val_metrics.print_results("Val")

        self.cleanup()
        return dict(history)

    def cleanup(self) -> None:
        """Free GPU VRAM by moving the model to CPU and clearing PyTorch cache."""
        logger.info("Cleaning up GPU VRAM...")
        if hasattr(self, "model"):
            self.model.to("cpu")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("GPU VRAM cleared.")

"""Training engines for classification, segmentation, and detection tasks.

Provides :class:`Trainer` for managing end-to-end training loops with
GPU-native metric computation, optional LR scheduling, and per-epoch
history tracking.

Example::

    from corecv.engine.train import Trainer

    trainer = Trainer(
        task="classification",
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
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from corecv.metrics.classification import ClassificationMetrics
from corecv.metrics.segmentation import SegmentationMetrics

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
        val_loader: Optional DataLoader for validation data. When ``None``,
            the monitored metric is tracked on the training split instead.
        loss_fn: Loss function accepting ``(logits, targets)``.
        optimizer: Optimizer for parameter updates.
        num_classes: Number of classes.
        device: Target device for computation.
        scheduler: Optional learning rate scheduler.
        target_key: Batch key holding the ground-truth targets
            (``"labels"`` for classification, ``"masks"`` for segmentation).
        monitor: Metric tracked for best-model selection and early stopping.
        mode: Whether to minimize (``"min"``) or maximize (``"max"``) the
            monitored metric.
        patience: Number of epochs without improvement before early stopping.
        checkpoint_path: Optional path where the best model state dict is
            saved whenever the monitored metric improves.
        best_score: Best monitored score reached during :meth:`fit`.
        best_epoch: Epoch at which *best_score* was achieved.

    """

    def __init__(  # noqa: PLR0913
        self,
        task: str,
        model: nn.Module,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        num_classes: int,
        device: str | torch.device = "cuda",
        scheduler: LRScheduler | None = None,
        use_amp: bool = False,
        val_loader: DataLoader | None = None,
        patience: int | None = None,
        monitor: str | None = None,
        mode: Literal["min", "max"] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            task: Task to train. One of ``classification`` or
                ``segmentation``.
            model: Model mapping ``(B, 3, H, W)`` to the task-specific
                output format.
            train_loader: Training data loader yielding dicts with
                ``"images"`` plus a task-specific target key (``"labels"``
                for classification, ``"masks"`` for segmentation).
            loss_fn: Loss function accepting ``(logits, targets)`` and
                returning a scalar loss tensor.
            optimizer: Optimizer instance for gradient-based updates.
            num_classes: Number of target classes.
            device: Target device string or :class:`torch.device`.
            scheduler: Optional LR scheduler stepped after each epoch.
            use_amp: Whether to enable Automatic Mixed Precision (AMP).
            val_loader: Optional validation data loader yielding dicts with
                ``"images"`` plus a task-specific target key. When ``None``
                (default), the monitored metric is tracked on the training
                metrics instead.
            patience: Number of consecutive epochs without improvement of the
                monitored metric before early stopping. When ``None`` (default),
                early stopping is disabled.
            monitor: Metric to track for best-model selection and early
                stopping. Defaults to ``top1_acc`` for classification and
                ``dice`` for segmentation. Monitored on the validation split
                when a *val_loader* is provided, and on the training split
                otherwise.
            mode: Whether to minimize (``"min"``) or maximize (``"max"``) the
                monitored metric. Defaults to ``min`` for ``"loss"`` and
                ``max`` otherwise.
            checkpoint_path: Optional path where the best model state dict is
                saved whenever the monitored metric improves. When ``None``
                (default), no checkpoint is saved.

        Raises:
            ValueError: If *task* is not a supported task.

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
            self.target_key = "labels"
            default_monitor = "top1_acc"
        elif task == "segmentation":
            self._train_metrics = SegmentationMetrics(num_classes=num_classes)
            self._val_metrics = SegmentationMetrics(num_classes=num_classes)
            self.target_key = "masks"
            default_monitor = "dice"
        else:
            raise ValueError(f"Unknown task: {task}")

        self.monitor = monitor if monitor is not None else default_monitor
        self.mode = mode if mode is not None else ("min" if self.monitor == "loss" else "max")
        self.patience = patience
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.best_score: float | None = None
        self.best_epoch: int = 0

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
            targets = batch[self.target_key]

            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp, device_type=self.device.type):
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self._train_metrics.update(logits.detach(), targets)
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

        Raises:
            ValueError: If no *val_loader* was provided to the :class:`Trainer`.

        """
        if self.val_loader is None:
            msg = "validate() requires a val_loader; none was provided to the Trainer"
            raise ValueError(msg)

        self.model.eval()
        self._val_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for raw_batch in self.val_loader:
                batch = _batch_to_device(raw_batch, self.device)
                images = batch["images"]
                targets = batch[self.target_key]

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, targets)

                self._val_metrics.update(logits, targets)
                total_loss += loss.item()
                num_batches += 1

        metrics = self._val_metrics.compute()
        metrics["loss"] = total_loss / max(num_batches, 1)
        return metrics

    def fit(self, num_epochs: int) -> dict[str, list[float]]:
        """Run the full training loop with best-model tracking and early stopping.

        Tracks the monitored metric (``self.monitor``), saves the best model
        state dict to ``checkpoint_path`` whenever it improves, and stops
        early after ``patience`` epochs without improvement. The metric is
        monitored on the validation split when a *val_loader* is provided,
        and on the training split otherwise.

        Args:
            num_epochs: Total number of training epochs to execute.

        Returns:
            Dictionary mapping metric names (prefixed with ``"train_"``
            or ``"val_"``) to lists of per-epoch float values. When no
            *val_loader* is provided, only ``train_*`` keys are present.

        """
        history: dict[str, list[float]] = defaultdict(list)
        has_val = self.val_loader is not None
        self.best_score = float("inf") if self.mode == "min" else float("-inf")
        self.epochs_without_improvement = 0

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate() if has_val else {}

            if self.scheduler is not None:
                self.scheduler.step()

            monitored = val_metrics if has_val else train_metrics
            stop = self._update_best_model(monitored, epoch, "val" if has_val else "train")

            for key, value in train_metrics.items():
                history[f"train_{key}"].append(value)
            for key, value in val_metrics.items():
                history[f"val_{key}"].append(value)

            log_msg = f"Epoch {epoch}/{num_epochs} | train_loss={train_metrics['loss']:.4f}"
            if has_val:
                log_msg += f" val_loss={val_metrics['loss']:.4f}"
            logger.info(log_msg)

            self._train_metrics.print_results("Train")
            if has_val:
                self._val_metrics.print_results("Val")

            if stop:
                break

        self.cleanup()
        return dict(history)

    def _update_best_model(
        self,
        metrics: dict[str, float],
        epoch: int,
        split: str,
    ) -> bool:
        """Track the best model and signal whether early stopping should trigger.

        Compares the current epoch's monitored metric against the best score
        seen so far. On improvement, resets the no-improvement counter and
        saves the model state dict to ``checkpoint_path``. Otherwise
        increments the counter and returns ``True`` once it reaches
        ``patience``.

        Args:
            metrics: Metrics of the monitored split for the current epoch.
            epoch: Current epoch number.
            split: Name of the monitored split (``"train"`` or ``"val"``).

        Returns:
            ``True`` if training should stop early, ``False`` otherwise.

        Raises:
            ValueError: If the monitored metric is not present in *metrics*.

        """
        if self.monitor not in metrics:
            msg = (
                f"monitor metric {self.monitor!r} not found in {split} "
                f"metrics: {sorted(metrics)}"
            )
            raise ValueError(msg)

        current = metrics[self.monitor]
        improved = current > self.best_score if self.mode == "max" else current < self.best_score
        if improved:
            self.best_score = current
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            if self.checkpoint_path is not None:
                torch.save(self.model.state_dict(), self.checkpoint_path)
                logger.info(
                    "Saved best model to %s (%s_%s=%.4f)",
                    self.checkpoint_path,
                    split,
                    self.monitor,
                    current,
                )
            return False

        self.epochs_without_improvement += 1
        if self.patience is not None and self.epochs_without_improvement >= self.patience:
            logger.info(
                "Early stopping after %d epochs without improvement (patience=%d)",
                epoch,
                self.patience,
            )
            return True
        return False

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

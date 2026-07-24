"""Unified training loop coordinator for CoreCV models.

Provides the :class:`CoreTrainer` class that orchestrates the complete
training loop with support for:

- **AMP**: Automatic Mixed Precision via ``torch.amp.autocast``
- **Gradient Accumulation**: Scale loss by ``1 / accum_steps``,
  call ``optimizer.step()`` every ``accum_steps``
- **Gradient Clipping**: ``torch.nn.utils.clip_grad_norm_`` before step
- **EMA**: Exponential Moving Average of model weights
- **Checkpoint**: Save / load with optimizer, scheduler, epoch state
- **Metrics**: Integration with ``corecv.metrics`` objects
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_MIN_BATCH_ELEMENTS = 2

# ======================================================================
# EMA context manager
# ======================================================================


class EMAContext:
    """Context manager that temporarily applies EMA weights to the model.

    Used internally by :meth:`CoreTrainer.model_ema` to provide a
    ``with``-compatible interface for inference with EMA-averaged weights.

    Example::

        with trainer.model_ema:
            output = trainer.model(inputs)
    """

    def __init__(self, trainer: CoreTrainer) -> None:
        """Store a reference to the parent trainer.

        Args:
            trainer: The :class:`CoreTrainer` instance.
        """
        self._trainer = trainer
        self._saved_state: dict[str, torch.Tensor] = {}

    def __enter__(self) -> nn.Module:
        """Save original weights and load EMA weights into the model.

        Returns:
            The model with EMA weights applied.
        """
        model = self._trainer.model
        ema_params = self._trainer._ema_params  # noqa: SLF001

        for name, param in model.named_parameters():
            if name in ema_params:
                self._saved_state[name] = param.data.clone()

        for name, param in model.named_parameters():
            if name in ema_params:
                param.data.copy_(ema_params[name])

        return model

    def __exit__(self, *args: object) -> None:
        """Restore original weights from before the context."""
        model = self._trainer.model
        for name, param in model.named_parameters():
            if name in self._saved_state:
                param.data.copy_(self._saved_state[name])
        self._saved_state.clear()


# ======================================================================
# CoreTrainer
# ======================================================================


class CoreTrainer:
    """Unified training engine for CoreCV models.

    Coordinates the complete training loop with support for:

    - **Automatic Mixed Precision (AMP)** via ``torch.amp.autocast``
    - **Gradient accumulation**
    - **Gradient clipping**
    - **Exponential Moving Average (EMA)** of model weights
    - **Checkpoint** save/load with optimizer, scheduler, epoch state
    - **Metrics** integration with ``corecv.metrics`` objects

    Args:
        model: The CoreCV model to train (``nn.Module``).
        optimizer: PyTorch optimizer.
        loss_fn: Loss function (callable taking ``preds``, ``targets``
            and returning a scalar loss tensor).
        train_dataloader: Training :class:`DataLoader`.
        val_dataloader: Optional validation :class:`DataLoader`.
        device: :class:`torch.device` for training (``'cuda'`` or ``'cpu'``).
            If ``None``, auto-detects ``'cuda'`` when available.
        gradient_accumulation_steps: Number of steps to accumulate
            gradients before each optimizer update.  Default ``1``.
        max_grad_norm: Max norm for gradient clipping.  ``None`` disables
            clipping.  Default ``1.0``.
        use_amp: Enable Automatic Mixed Precision.  Defaults to ``True``
            when ``device`` is CUDA, ``False`` otherwise.
        amp_dtype: AMP computation dtype.  Default ``torch.float16``.
        ema_decay: EMA decay factor.  ``None`` disables EMA.
            Default ``0.9999``.
        ema_start_epoch: Epoch at which to start updating EMA weights
            (allows EMA to begin after a warmup period).  Default ``0``.
        scheduler: Optional LR scheduler.
        scheduler_interval: When to step the scheduler.  One of ``'step'``
            (after each optimizer step) or ``'epoch'`` (after each epoch).
            Default ``'epoch'``.
        log_interval: Log training metrics every ``log_interval`` optimizer
            steps.  Default ``50``.
        output_dir: Directory for checkpoints.  Default ``'./checkpoints'``.
        train_metrics: Optional metrics object (from ``corecv.metrics``)
            with ``update(preds, targets)`` and ``compute()`` methods.
            Accumulated during training.
        val_metrics: Optional metrics object (from ``corecv.metrics``)
            with ``update(preds, targets)`` and ``compute()`` methods.
            Accumulated during validation.
        model_config: Optional dictionary containing the model configuration
            (e.g. architecture hyperparameters).  Stored in checkpoints
            under the ``'model_config'`` key for reproducibility.
            Default ``None``.

    Example::

        >>> trainer = CoreTrainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     loss_fn=nn.CrossEntropyLoss(),
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     device=torch.device("cuda"),
        ... )
        >>> history = trainer.fit(num_epochs=100)
        >>> with trainer.model_ema:
        ...     output = trainer.model(test_inputs)
    """

    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: object,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        device: torch.device | None = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float | None = 1.0,
        use_amp: bool | None = None,
        amp_dtype: torch.dtype = torch.float16,
        ema_decay: float | None = 0.9999,
        ema_start_epoch: int = 0,
        scheduler: object | None = None,
        scheduler_interval: str = "epoch",
        log_interval: int = 50,
        output_dir: str = "./checkpoints",
        train_metrics: nn.Module | None = None,
        val_metrics: nn.Module | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the CoreTrainer with model, optimizer, and training configuration."""
        # ---- Device ----------------------------------------------------
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device: torch.device = device

        # ---- AMP -------------------------------------------------------
        if use_amp is None:
            use_amp = device.type == "cuda"
        self.use_amp: bool = use_amp
        self.amp_dtype: torch.dtype = amp_dtype

        # ---- Core components -------------------------------------------
        self.model: nn.Module = model.to(self.device)
        self.optimizer: torch.optim.Optimizer = optimizer
        self.loss_fn: object = loss_fn
        self.train_dataloader: DataLoader = train_dataloader
        self.val_dataloader: DataLoader | None = val_dataloader

        # ---- Training hyperparameters ----------------------------------
        self.gradient_accumulation_steps: int = int(gradient_accumulation_steps)
        self.max_grad_norm: float | None = max_grad_norm

        self.ema_decay: float | None = ema_decay
        self.ema_start_epoch: int = int(ema_start_epoch)

        self.scheduler: object | None = scheduler
        if scheduler_interval not in ("step", "epoch"):
            msg = (
                f"scheduler_interval must be 'step' or 'epoch', "
                f"got {scheduler_interval!r}"
            )
            raise ValueError(msg)
        self.scheduler_interval: str = scheduler_interval

        self.log_interval: int = int(log_interval)
        self.output_dir: Path = Path(output_dir)

        # ---- Metrics ---------------------------------------------------
        self.train_metrics: nn.Module | None = train_metrics
        self.val_metrics: nn.Module | None = val_metrics

        # ---- Model config ----------------------------------------------
        self.model_config: dict[str, Any] | None = model_config

        # ---- AMP gradient scaler ---------------------------------------
        self.scaler: GradScaler = GradScaler(device=device.type, enabled=use_amp)

        # ---- EMA shadow parameters -------------------------------------
        self._ema_params: dict[str, torch.Tensor] = {}
        if self.ema_decay is not None:
            self._init_ema()

        # ---- Output directory ------------------------------------------
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Public API
    # ==================================================================

    def train_one_epoch(self, epoch: int) -> dict[str, Any]:
        """Run one training epoch.

        Iterates over ``train_dataloader``, computes forward / backward,
        accumulates gradients, applies gradient clipping, updates weights,
        and optionally updates EMA and scheduler (if
        ``scheduler_interval == 'step'``).

        Args:
            epoch: Current epoch number (1-indexed, used for logging and
                EMA start condition).

        Returns:
            Dictionary of training metrics for the epoch, including at
            minimum ``'loss'`` and ``'lr'``, plus any metrics from
            ``train_metrics.compute()``.

        Raises:
            ValueError: If ``train_dataloader`` is empty.
        """
        self.model.train()

        num_batches: int = len(self.train_dataloader)
        if num_batches == 0:
            msg = "train_dataloader is empty"
            raise ValueError(msg)

        total_loss: float = 0.0
        log_loss: float = 0.0
        log_count: int = 0
        current_lr: float = self._get_current_lr()
        optimizer_steps: int = 0

        if self.train_metrics is not None:
            self.train_metrics.reset()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_dataloader):
            # --- Unpack batch -------------------------------------------
            inputs, targets = self._unpack_batch(batch)
            inputs = inputs.to(self.device)
            targets = self._targets_to_device(targets)

            # --- Forward (AMP autocast) ---------------------------------
            with autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

            # --- Backward (scale for accumulation) ----------------------
            scaled_loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(scaled_loss).backward()

            # --- Accumulators for logging / epoch stats -----------------
            loss_item: float = loss.item()
            total_loss += loss_item
            log_loss += loss_item
            log_count += 1

            # --- Optimizer step at accumulation boundary ----------------
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self._optimizer_step(epoch)
                optimizer_steps += 1
                current_lr = self._get_current_lr()

                # Logging
                if optimizer_steps % self.log_interval == 0:
                    avg_loss: float = log_loss / max(log_count, 1)
                    logger.info(
                        "Epoch [%d] Step [%d/%d]  Loss: %.4f  LR: %.6f",
                        epoch,
                        batch_idx + 1,
                        num_batches,
                        avg_loss,
                        current_lr,
                    )
                    log_loss = 0.0
                    log_count = 0

            # --- Training metrics ---------------------------------------
            if self.train_metrics is not None:
                self.train_metrics.update(outputs.detach(), targets)

        # Handle remaining gradients when accumulation boundary not reached
        last_batch: int = batch_idx + 1  # noqa: F841  # safe after non-empty loop
        if last_batch % self.gradient_accumulation_steps != 0:
            self._optimizer_step(epoch)
            current_lr = self._get_current_lr()

        # ---- Epoch-level metrics ---------------------------------------
        avg_epoch_loss: float = total_loss / num_batches
        metrics: dict[str, Any] = {
            "loss": avg_epoch_loss,
            "lr": current_lr,
        }

        if self.train_metrics is not None:
            train_results: dict[str, Any] = self.train_metrics.compute()
            for k, v in train_results.items():
                metrics[k] = v.item() if isinstance(v, torch.Tensor) else v

        return metrics

    def validate(
        self,
        epoch: int,
        use_ema: bool = False,
    ) -> dict[str, Any]:
        """Run validation on ``val_dataloader``.

        Args:
            epoch: Current epoch number (used for logging).
            use_ema: If ``True``, temporarily apply EMA weights for the
                validation run.  Default ``False``.

        Returns:
            Dictionary of validation metrics, including at least
            ``'val_loss'``, plus any metrics from ``val_metrics.compute()``.

        Raises:
            RuntimeError: If ``val_dataloader`` is ``None``.
        """
        if self.val_dataloader is None:
            msg = "val_dataloader is not configured"
            raise RuntimeError(msg)

        # Optionally apply EMA weights
        ema_ctx: EMAContext | None = None
        if use_ema and self.ema_decay is not None:
            ema_ctx = EMAContext(self)
            ema_ctx.__enter__()

        self.model.eval()

        total_loss: float = 0.0
        num_batches: int = len(self.val_dataloader)

        if num_batches == 0:
            if ema_ctx is not None:
                ema_ctx.__exit__()
            return {"val_loss": 0.0}

        if self.val_metrics is not None:
            self.val_metrics.reset()

        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, targets = self._unpack_batch(batch)
                inputs = inputs.to(self.device)
                targets = self._targets_to_device(targets)

                with autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()

                if self.val_metrics is not None:
                    self.val_metrics.update(outputs, targets)

        # Restore original weights if EMA was applied
        if ema_ctx is not None:
            ema_ctx.__exit__()

        avg_val_loss: float = total_loss / num_batches
        metrics: dict[str, Any] = {
            "val_loss": avg_val_loss,
        }

        if self.val_metrics is not None:
            val_results: dict[str, Any] = self.val_metrics.compute()
            for k, v in val_results.items():
                metrics[k] = v.item() if isinstance(v, torch.Tensor) else v

        logger.info("Epoch [%d]  Validation Loss: %.4f", epoch, avg_val_loss)
        return metrics

    def fit(self, num_epochs: int) -> dict[str, list]:
        """Run the complete training loop for a given number of epochs.

        For each epoch:

        1. Calls :meth:`train_one_epoch`
        2. Calls :meth:`validate` (if ``val_dataloader`` is available)
        3. Steps the epoch-based scheduler (if configured with
           ``scheduler_interval='epoch'``)
        4. Saves a checkpoint

        Args:
            num_epochs: Number of epochs to train.

        Returns:
            History dictionary with keys ``'train'`` and ``'val'``, each
            containing a list of per-epoch metric dictionaries.
        """
        history: dict[str, list] = {
            "train": [],
            "val": [],
        }

        for epoch in range(1, num_epochs + 1):
            # ---- Training ----------------------------------------------
            train_metrics: dict[str, Any] = self.train_one_epoch(epoch)
            history["train"].append(train_metrics)

            # ---- Validation --------------------------------------------
            if self.val_dataloader is not None:
                val_metrics: dict[str, Any] = self.validate(epoch, use_ema=False)
                history["val"].append(val_metrics)
            else:
                history["val"].append({"val_loss": 0.0})

            # ---- Epoch-based scheduler step ----------------------------
            if (
                self.scheduler is not None
                and self.scheduler_interval == "epoch"
            ):
                self.scheduler.step()

            # ---- Print Epoch Summary -----------------------------------
            parts: list[str] = [f"Epoch {epoch:3d}/{num_epochs:3d}"]
            # Train metrics
            for k, v in train_metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k}: {v:.4f}")
                else:
                    parts.append(f"{k}: {v}")
            # Validation metrics
            val_dict: dict[str, Any] = history["val"][-1]
            for k, v in val_dict.items():
                if isinstance(v, float):
                    parts.append(f"{k}: {v:.4f}")
                else:
                    parts.append(f"{k}: {v}")
            print(" | ".join(parts))  # noqa: T201

            # ---- Save checkpoint ---------------------------------------
            self.save_checkpoint(
                path=str(self.output_dir / f"epoch_{epoch}.pt"),
                epoch=epoch,
                metrics={
                    "train": train_metrics,
                    "val": history["val"][-1],
                },
            )

        # ---- Final checkpoint ------------------------------------------
        self.save_checkpoint(
            path=str(self.output_dir / "final.pt"),
            epoch=num_epochs,
            metrics=history,
        )

        return history

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        """Save a training checkpoint to disk.

        The checkpoint dictionary contains the following keys:

        - ``epoch`` — Current epoch number.
        - ``model_state_dict`` — Model parameters.
        - ``optimizer_state_dict`` — Optimizer state.
        - ``scheduler_state_dict`` — Scheduler state (``None`` if not set).
        - ``ema_state_dict`` — EMA shadow parameters.
        - ``scaler_state_dict`` — AMP gradient scaler state.
        - ``metrics`` — User-supplied metrics dictionary.
        - ``model_config`` — Model configuration dictionary (``None`` if
          not provided at initialisation).

        Args:
            path: File path for the checkpoint.
            epoch: Current epoch number.
            metrics: Dictionary of metrics to store in the checkpoint
                (e.g. loss, accuracy, etc.).
        """
        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "ema_state_dict": self._ema_params,
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": metrics,
            "model_config": self.model_config,
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_ema: bool = True,
    ) -> dict[str, Any]:
        """Load a training checkpoint from disk.

        Args:
            path: File path of the checkpoint.
            load_optimizer: If ``True``, restore the optimizer state dict.
                Default ``True``.
            load_scheduler: If ``True``, restore the scheduler state dict.
                Default ``True``.
            load_ema: If ``True``, restore the EMA parameter state.
                Default ``True``.

        Returns:
            The full checkpoint dictionary (contains at least ``'epoch'``,
            ``'model_state_dict'``, and ``'metrics'``).  May also contain
            ``'model_config'`` if the checkpoint was saved by a trainer
            that was initialised with a ``model_config``.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the checkpoint is missing a required key.
        """
        checkpoint: dict[str, Any] = torch.load(path, map_location=self.device)

        if "model_state_dict" not in checkpoint:
            msg = f"Checkpoint at {path} is missing 'model_state_dict'"
            raise RuntimeError(msg)

        # ---- Load model state ------------------------------------------
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # ---- Load optimizer state --------------------------------------
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # ---- Load scheduler state --------------------------------------
        if load_scheduler and self.scheduler is not None:
            sched_state = checkpoint.get("scheduler_state_dict")
            if sched_state is not None:
                self.scheduler.load_state_dict(sched_state)

        # ---- Load EMA state --------------------------------------------
        if load_ema and "ema_state_dict" in checkpoint:
            self._ema_params = checkpoint["ema_state_dict"]

        # ---- Load scaler state -----------------------------------------
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        epoch: int = checkpoint.get("epoch", 0)
        logger.info("Checkpoint loaded from %s (epoch %d)", path, epoch)

        return checkpoint

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def model_ema(self) -> EMAContext:
        """Get a context manager that temporarily applies EMA weights.

        Use this for inference with EMA-averaged weights::

            with trainer.model_ema:
                output = trainer.model(input)

        The EMA weights are applied to the model upon entering the
        ``with`` block and automatically restored upon exit.

        Returns:
            An :class:`EMAContext` context manager.
        """
        return EMAContext(self)

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _init_ema(self) -> None:
        """Initialise EMA shadow parameters from the model's current weights.

        Creates a detached copy of each trainable parameter's data.
        """
        self._ema_params = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self._ema_params[name] = param.data.clone().detach()

    def _update_ema(self) -> None:
        """Update EMA shadow parameters after an optimizer step.

        Implements::

            shadow = decay * shadow + (1 - decay) * param.data

        for each trainable parameter of the model.
        """
        if self.ema_decay is None:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self._ema_params:
                    self._ema_params[name] = (
                        self.ema_decay * self._ema_params[name]
                        + (1.0 - self.ema_decay) * param.data
                    )

    def _optimizer_step(self, epoch: int) -> None:
        """Perform a single optimiser step with gradient clipping and EMA.

        Args:
            epoch: Current epoch (used for EMA start condition).
        """
        # Gradient clipping
        if self.max_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

        # Optimizer step (AMP-aware)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        # EMA update
        if self.ema_decay is not None and epoch >= self.ema_start_epoch:
            self._update_ema()

        # Step-based scheduler
        if self.scheduler is not None and self.scheduler_interval == "step":
            self.scheduler.step()

    def _get_current_lr(self) -> float:
        """Return the current learning rate of the first parameter group.

        Returns:
            Learning rate as a float, or ``0.0`` if there are no parameter
            groups.
        """
        if self.optimizer.param_groups:
            return float(self.optimizer.param_groups[0]["lr"])
        return 0.0

    @staticmethod
    def _unpack_batch(
        batch: object,
    ) -> tuple[torch.Tensor, object]:
        """Unpack a batch from the dataloader.

        Supports the following formats:

        - ``(inputs, targets)`` tuple (or extended tuples with extra items)
        - ``(inputs, targets, ...)``
        - ``{'inputs': ..., 'targets': ...}`` dict
        - ``{'image': ..., 'label': ...}`` dict
        - Single tensor (unsupervised — targets set to ``None``)

        Args:
            batch: A single batch produced by the :class:`DataLoader`.

        Returns:
            Tuple ``(inputs, targets)``.
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) >= _MIN_BATCH_ELEMENTS:
                return batch[0], batch[1]
            return batch[0], None
        if isinstance(batch, dict):
            inputs: torch.Tensor = batch.get("inputs") or batch.get("image")
            targets: object = batch.get("targets") or batch.get("label")
            return inputs, targets
        return batch, None

    def _targets_to_device(self, targets: object) -> object:
        """Move target data to the training device.

        Handles:

        - :class:`torch.Tensor`
        - :class:`list` of tensors
        - :class:`tuple` of tensors
        - :class:`dict` of tensors / scalars

        Args:
            targets: Target data in any of the supported formats.

        Returns:
            Targets moved to ``self.device``, preserving the original
            container type.
        """
        if targets is None:
            return None
        if isinstance(targets, torch.Tensor):
            return targets.to(self.device)
        if isinstance(targets, (list, tuple)):
            return type(targets)(
                self._targets_to_device(t) for t in targets
            )
        if isinstance(targets, dict):
            return {k: self._targets_to_device(v) for k, v in targets.items()}
        # Scalar or non-tensor — return as-is
        return targets

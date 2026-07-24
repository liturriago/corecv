"""CoreModel facade — high-level unified API for CoreCV models.

Provides the :class:`CoreModel` class that wraps any CoreCV model and exposes
a unified API for training, inference, and model export.  All configuration
parameters are strictly validated against the schemas defined in
:mod:`corecv.config.schemas` and local training/export dataclasses.

Internal delegation:

* :class:`CoreTrainer` (``corecv.engine.trainer``) — training loop
* :class:`CorePredictor` (``corecv.engine.predictor``) — inference
* :class:`CoreExporter` (``corecv.engine.exporter``) — export pipeline
* :class:`TargetRewriter` (``corecv.engine.rewriter``) — graph rewriting
* :class:`MetaProber` (``corecv.engine.validator``) — zero-VRAM validation

Example::

    >>> import torch
    >>> from corecv.api import CoreModel
    >>>
    >>> model = CoreModel(my_model, task="classification", input_size=(224, 224))
    >>>
    >>> # Train with keyword arguments
    >>> history = model.train(epochs=10, lr=0.001, batch_size=64)
    >>>
    >>> # Predict with a source image
    >>> preds = model.predict("image.jpg", topk=5)
    >>>
    >>> # Export to ONNX with edge-hardware rewrites
    >>> paths = model.export(format="onnx", target_hardware="edge")
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

import torch
from torch import nn
from torch.utils.data import DataLoader

from corecv.config.schemas import load_config
from corecv.core.registry import get_backbone, get_head, get_neck
from corecv.engine.exporter import CoreExporter
from corecv.engine.predictor import CorePredictor, Prediction
from corecv.engine.rewriter import TargetRewriter
from corecv.engine.trainer import CoreTrainer
from corecv.models.detector import CoreObjectDetector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SUPPORTED_TASKS: frozenset[str] = frozenset({
    "classification", "segmentation", "detection",
})
_VALID_OPTIMIZERS: frozenset[str] = frozenset({"adamw", "adam", "sgd"})
_VALID_SCHEDULERS: frozenset[str] = frozenset({"cosine", "step", "none", None})  # type: ignore[assignment]
_VALID_EXPORT_FORMATS: frozenset[str] = frozenset({"onnx", "executorch", "both"})
_VALID_HARDWARE_TARGETS: frozenset[str] = frozenset({"edge", "server"})
_VALID_OPSET_VERSIONS: frozenset[int] = frozenset({17, 18})
_INPUT_SHAPE_NDIM: int = 4

_DEFAULT_INPUT_SIZE: tuple[int, int] = (224, 224)
_DEFAULT_EPOCHS: int = 100
_DEFAULT_LR: float = 0.001
_DEFAULT_BATCH_SIZE: int = 32
_DEFAULT_OPTIMIZER: str = "adamw"
_DEFAULT_OPSET: int = 17

# -- Error message constants (TRY003) --
_ERR_EPOCHS_RANGE: str = "epochs must be >= 1, got {}"
_ERR_LR_POSITIVE: str = "lr must be > 0, got {}"
_ERR_BATCH_SIZE_RANGE: str = "batch_size must be >= 1, got {}"
_ERR_UNKNOWN_OPTIMIZER: str = "Unknown optimizer {!r}. Valid options: {}"
_ERR_UNKNOWN_SCHEDULER: str = "Unknown scheduler {!r}. Valid options: {}"
_ERR_GRAD_ACCUM_RANGE: str = "grad_accum must be >= 1, got {}"
_ERR_EMA_DECAY_RANGE: str = "ema_decay must be in (0, 1), got {}"
_ERR_UNKNOWN_FORMAT: str = "Unknown format {!r}. Valid options: {}"
_ERR_UNKNOWN_HARDWARE: str = "Unknown target_hardware {!r}. Valid options: {}"
_ERR_INVALID_OPSET: str = "Invalid opset {}. Valid options: {}"
_ERR_INPUT_SHAPE_DIMS: str = "input_shape must have 4 dimensions (B, C, H, W), got {}"
_ERR_UNSUPPORTED_TASK: str = "Unsupported task {!r}. Must be one of {}"
_ERR_MISSING_DATALOADER: str = (
    "Training dataloader is required."
    " Call set_train_dataloader() before train()."
)
_ERR_MISSING_LOSS_FN: str = "Loss function is required. Call set_loss_fn() before train()."
_ERR_CONFIG_TYPE: str = "Expected str, dict, TrainingConfig, or None, got {}"
_ERR_YAML_NOT_MAPPING: str = "YAML file must contain a top-level mapping, got {}"
_ERR_TRAIN_CONFIG_NOT_FOUND: str = "Training config file not found: {}"
_ERR_MODEL_EXTENSION: str = (
    "Unsupported model file extension {!r}. Expected .pt, .pth, .yaml, or .yml."
)
_ERR_MODEL_TYPE: str = "Expected nn.Module, str, Path, or dict, got {!r}"
_ERR_CHECKPOINT_NOT_FOUND: str = "Checkpoint file not found: {}"
_ERR_WEIGHTS_NOT_FOUND: str = "Weights file not found: {}"
_ERR_CHECKPOINT_NO_CONFIG: str = (
    "Checkpoint {} is missing 'model_config' key. Available keys: {}"
)
_ERR_INPUT_SIZE_DIM: int = 2
_ERR_CHECKPOINT_NO_CONFIG_BARE: str = (
    "Checkpoint {} is missing 'model_config'. Use from_pretrained()"
    " or pass an nn.Module directly."
)
_ERR_CONFIG_FILE_NOT_FOUND: str = "Configuration file not found: {}"
_ERR_MISSING_TASK_IN_CONFIG: str = (
    "Config must contain a 'task' field"
    " (classification, segmentation, or detection)"
)
_ERR_MISSING_MODEL_NAME: str = (
    "Config must contain a 'model_name' field"
    " (e.g. resnet50, mobilenet_v3_large)"
)
_ERR_UNKNOWN_TASK_IN_CONFIG: str = (
    "Unknown task {!r}. Supported: classification, segmentation, detection"
)
_ERR_INVALID_STATE_DICT: str = "Could not extract a valid state_dict from {}"


# ======================================================================
# TrainingConfig  (validated dataclass)
# ======================================================================


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    """Validated training hyperparameter configuration.

    All fields are validated in ``__post_init__``.

    Attributes:
        epochs: Number of training epochs.  Must be ``>= 1``.
        lr: Learning rate.  Must be ``> 0``.
        batch_size: Batch size per device.  Must be ``>= 1``.
        optimizer: Optimizer name.  One of ``"adamw"``, ``"adam"``,
            ``"sgd"``.
        scheduler: Scheduler name.  One of ``"cosine"``, ``"step"``,
            ``"none"``, or ``None``.
        amp: Enable automatic mixed precision.  Default ``True``.
        grad_accum: Gradient accumulation steps.  Must be ``>= 1``.
        clip_grad: Max gradient norm for clipping.  ``None`` disables
            clipping.  Default ``1.0``.
        ema: Enable exponential moving average.  Default ``True``.
        ema_decay: EMA decay factor.  Must be in ``(0, 1)``.
            Default ``0.9999``.
        device: Target device string (e.g. ``"cuda"``, ``"cpu"``).
            If ``None``, auto-detected.
        output_dir: Directory for checkpoints and logs.
        target_hardware: Hardware profile.  ``"edge"`` applies
            activation rewrites (GELU -> ReLU, SiLU -> Hardswish)
            and LayerNorm collapses before the optimiser is built;
            ``"server"`` (default) skips rewrites.
    """

    epochs: int = _DEFAULT_EPOCHS
    lr: float = _DEFAULT_LR
    batch_size: int = _DEFAULT_BATCH_SIZE
    optimizer: str = _DEFAULT_OPTIMIZER
    scheduler: str | None = None
    amp: bool = True
    grad_accum: int = 1
    clip_grad: float | None = 1.0
    ema: bool = True
    ema_decay: float = 0.9999
    device: str | None = None
    output_dir: str = "./checkpoints"
    target_hardware: str = "server"

    def __post_init__(self) -> None:
        """Validate training configuration fields."""
        hw = self.target_hardware.lower()
        object.__setattr__(self, "target_hardware", hw)
        if hw not in _VALID_HARDWARE_TARGETS:
            raise ValueError(
                _ERR_UNKNOWN_HARDWARE.format(self.target_hardware, sorted(_VALID_HARDWARE_TARGETS))
            )
        if self.epochs < 1:
            raise ValueError(_ERR_EPOCHS_RANGE.format(self.epochs))
        if self.lr <= 0.0:
            raise ValueError(_ERR_LR_POSITIVE.format(self.lr))
        if self.batch_size < 1:
            raise ValueError(_ERR_BATCH_SIZE_RANGE.format(self.batch_size))
        if self.optimizer not in _VALID_OPTIMIZERS:
            raise ValueError(
                _ERR_UNKNOWN_OPTIMIZER.format(self.optimizer, sorted(_VALID_OPTIMIZERS))
            )
        if self.scheduler not in _VALID_SCHEDULERS:
            raise ValueError(
                _ERR_UNKNOWN_SCHEDULER.format(self.scheduler, sorted(_VALID_SCHEDULERS))
            )
        if self.grad_accum < 1:
            raise ValueError(
                _ERR_GRAD_ACCUM_RANGE.format(self.grad_accum)
            )
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError(
                _ERR_EMA_DECAY_RANGE.format(self.ema_decay)
            )


# ======================================================================
# ExportConfig  (validated dataclass)
# ======================================================================


@dataclass(frozen=True, kw_only=True)
class ExportConfig:
    """Validated export configuration.

    All fields are validated in ``__post_init__``.

    Attributes:
        format: Export format.  One of ``"onnx"``, ``"executorch"``,
            ``"both"``.
        target_hardware: Target hardware profile.  ``"edge"`` applies
            activation rewrites (GELU -> ReLU, SiLU -> Hardswish);
            ``"server"`` skips rewrites.
        opset: ONNX opset version.  Must be ``17`` or ``18``.
        optimize: Apply additional graph optimisations (rewrites,
            layout folding, delegate passes).  Default ``True``.
        output_path: Explicit output file path.  If ``None``, a
            timestamped path is generated in a temporary directory.
        input_shape: Input tensor shape ``(B, C, H, W)``.
            Defaults to ``(1, 3, 224, 224)``.
        dynamic_axes: ONNX-style dynamic axes dictionary, e.g.
            ``{"input": {0: "batch", 2: "height", 3: "width"}}``.
    """

    format: str = "onnx"
    target_hardware: str = "server"
    opset: int = _DEFAULT_OPSET
    optimize: bool = True
    output_path: str | None = None
    input_shape: tuple[int, ...] = (1, 3, *_DEFAULT_INPUT_SIZE)
    dynamic_axes: dict[str, dict[int, str]] | None = None

    def __post_init__(self) -> None:
        """Validate export configuration fields."""
        if self.format not in _VALID_EXPORT_FORMATS:
            raise ValueError(
                _ERR_UNKNOWN_FORMAT.format(self.format, sorted(_VALID_EXPORT_FORMATS))
            )
        hw = self.target_hardware.lower()
        object.__setattr__(self, "target_hardware", hw)
        if hw not in _VALID_HARDWARE_TARGETS:
            raise ValueError(
                _ERR_UNKNOWN_HARDWARE.format(self.target_hardware, sorted(_VALID_HARDWARE_TARGETS))
            )
        if self.opset not in _VALID_OPSET_VERSIONS:
            raise ValueError(
                _ERR_INVALID_OPSET.format(self.opset, sorted(_VALID_OPSET_VERSIONS))
            )
        if len(self.input_shape) != _INPUT_SHAPE_NDIM:
            raise ValueError(
                _ERR_INPUT_SHAPE_DIMS.format(len(self.input_shape))
            )


# ======================================================================
# CoreModel
# ======================================================================


class CoreModel:
    """High-level unified API facade for CoreCV models.

    Wraps any CoreCV model (classification, segmentation, or detection)
    and exposes a single entry point for training, inference, and export
    through delegation to specialised engines.

    Engines are created **lazily** — no heavyweight initialisation happens
    until the corresponding method is called.

    Args:
        model: One of:

            * ``nn.Module`` — a pre-built CoreCV model.
            * ``str`` / ``Path`` — path to a ``.pt`` / ``.pth`` checkpoint,
              a ``.yaml`` / ``.yml`` configuration file, or a plain
              registered backbone name (e.g. ``"resnet18"``).
            * ``dict`` — raw configuration dictionary containing at
              minimum ``model_name``.
        task: Task type.  One of ``"classification"``,
            ``"segmentation"``, or ``"detection"``.
        input_size: Input image dimensions ``(height, width)``.
            Default ``(224, 224)``.
        device: Target :class:`torch.device`.  If ``None``, auto-detects
            CUDA when available.
        num_classes: Number of output classes.  If ``None``, inferred
            from the model (via ``model.head.num_classes``) when possible.

    Example:
        >>> import torch
        >>> from corecv.api import CoreModel
        >>> from corecv.models import CoreObjectDetector
        >>>
        >>> detector = CoreObjectDetector(...)
        >>> model = CoreModel(detector, task="detection", input_size=(640, 640))
        >>>
        >>> # Fluent configuration
        >>> (model
        ...  .set_loss_fn(torch.nn.CrossEntropyLoss())
        ...  .set_train_dataloader(train_loader)
        ...  .set_val_dataloader(val_loader))
        >>>
        >>> # Train
        >>> history = model.train(epochs=50, lr=0.001, batch_size=16)
        >>>
        >>> # Predict
        >>> results = model.predict("test.jpg", conf_threshold=0.5)
        >>>
        >>> # Export
        >>> paths = model.export(format="onnx", target_hardware="edge")
    """

    def __init__(  # noqa: PLR0913, PLR0912
        self,
        model: nn.Module | str | Path | dict[str, Any],
        task: Literal["classification", "segmentation", "detection"],
        input_size: tuple[int, int] = _DEFAULT_INPUT_SIZE,
        device: torch.device | None = None,
        num_classes: int | None = None,
        pretrained: bool = True,
        neck: str | None = None,
        head: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialise the CoreModel facade.

        Args:
            model: A CoreCV model (``nn.Module``), a path to a ``.pt`` /
                ``.pth`` checkpoint, a path to a ``.yaml`` / ``.yml``
                configuration file, a plain registered backbone name
                (e.g. ``"resnet18"``), or a raw configuration ``dict``
                containing at minimum ``model_name``.
            task: Task type discriminant.
            input_size: Input ``(height, width)``.
            device: Target device (auto-detected if ``None``).
            num_classes: Number of output classes (inferred if ``None``).
            pretrained: Whether to load pretrained backbone weights. Only used
                when *model* is a plain backbone name string or a configuration
                ``dict``.  Default ``True``.
            neck: Registered neck name (e.g. ``"fpn"``, ``"panet"``).  Only used
                when building from a backbone name or ``dict``.  If ``None``, the
                task-specific default is used.
            head: Registered head name (e.g. ``"linear_classification"``,
                ``"decoupled_anchor_free"``).  Only used when building from a
                backbone name or ``dict``.  If ``None``, the task-specific
                default is used.
            **kwargs: Additional configuration entries forwarded to component
                constructors via dynamic kwarg inspection.  Only used when
                building from a backbone name or ``dict``.

        Raises:
            ValueError: If ``task`` is not a supported value, or if
                *model* has an unsupported file extension.
            TypeError: If *model* is not an ``nn.Module``, ``str``,
                ``Path``, or ``dict``.
            FileNotFoundError: If a file path does not exist.
        """
        if task not in _SUPPORTED_TASKS:
            raise ValueError(
                _ERR_UNSUPPORTED_TASK.format(task, sorted(_SUPPORTED_TASKS))
            )

        # --- Polymorphic model loading --------------------------------
        if isinstance(model, dict):
            merged = {**model, "task": task, "pretrained": pretrained}
            if num_classes is not None:
                merged["num_classes"] = num_classes
            if neck is not None:
                merged["neck_type"] = neck
            if head is not None:
                merged["head_type"] = head
            merged.update(kwargs)
            self._model = self._build_model_from_config(merged)
        elif isinstance(model, (str, Path)):
            path = Path(model)
            ext = path.suffix.lower()
            if ext in (".pt", ".pth"):
                # Load from checkpoint — _load_from_checkpoint sets self._model
                checkpoint_config = self._load_from_checkpoint(path)
                # Allow checkpoint metadata to fill optional fields
                num_classes = (
                    num_classes
                    if num_classes is not None
                    else checkpoint_config.get("num_classes")
                )
            elif ext in (".yaml", ".yml"):
                # Build model from YAML configuration
                self._load_from_config(path)
            elif ext == "":
                # Plain backbone name — build dynamically
                config: dict[str, Any] = {
                    "model_name": str(model),
                    "task": task,
                    "num_classes": num_classes or 1000,
                    "pretrained": pretrained,
                }
                if neck is not None:
                    config["neck_type"] = neck
                if head is not None:
                    config["head_type"] = head
                config.update(kwargs)
                self._model = self._build_model_from_config(config)
            else:
                raise ValueError(
                    _ERR_MODEL_EXTENSION.format(ext)
                )
        elif isinstance(model, nn.Module):
            self._model = model
        else:
            raise TypeError(
                _ERR_MODEL_TYPE.format(type(model).__name__)
            )

        # --- Core attributes -------------------------------------------
        self._task: str = task
        self._input_size: tuple[int, int] = (
            int(input_size[0]), int(input_size[1])
        )
        self._device: torch.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._num_classes: int | None = (
            num_classes if num_classes is not None else self._infer_num_classes()
        )

        # --- Optional components (must be set before training) ---------
        self._loss_fn: object | None = None
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None

        # --- Lazy engines ----------------------------------------------
        self._trainer: CoreTrainer | None = None
        self._predictor: CorePredictor | None = None
        self._exporter: CoreExporter | None = None

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def model(self) -> nn.Module:
        """Return the wrapped CoreCV model."""
        return self._model

    @property
    def task(self) -> str:
        """Return the task type.

        One of ``"classification"``, ``"segmentation"``, or ``"detection"``.
        """
        return self._task

    @property
    def input_size(self) -> tuple[int, int]:
        """Return the input ``(height, width)`` used for preprocessing."""
        return self._input_size

    @property
    def device(self) -> torch.device:
        """Return the target :class:`torch.device`."""
        return self._device

    @property
    def num_classes(self) -> int | None:
        """Return the number of output classes, or ``None`` if unknown."""
        return self._num_classes

    @property
    def trainer(self) -> CoreTrainer | None:
        """Return the internal :class:`CoreTrainer` instance.

        ``None`` until :meth:`train` is called.
        """
        return self._trainer

    @property
    def predictor(self) -> CorePredictor | None:
        """Return the internal :class:`CorePredictor` instance.

        ``None`` until :meth:`predict` is called.
        """
        return self._predictor

    # ==================================================================
    # from_pretrained  (factory classmethod)
    # ==================================================================

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: torch.device | None = None,
    ) -> CoreModel:
        """Load a pretrained model from a checkpoint file.

        The checkpoint must contain a ``model_config`` key (a dictionary
        describing the architecture) and a ``model_state_dict`` key
        containing the trained weights.  The architecture is rebuilt from
        ``model_config``, the weights are loaded, and a fully-configured
        :class:`CoreModel` instance is returned.

        Args:
            path: Path to a ``.pt`` or ``.pth`` checkpoint file.
            device: Target device.  If ``None``, auto-detected.

        Returns:
            A :class:`CoreModel` wrapping the reconstructed model.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            KeyError: If the checkpoint is missing ``model_config``.
            RuntimeError: If the checkpoint cannot be loaded.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(
                _ERR_CHECKPOINT_NOT_FOUND.format(path_obj)
            )

        checkpoint: dict[str, Any] = torch.load(
            str(path_obj), map_location="cpu", weights_only=False
        )

        model_config: dict[str, Any] | None = checkpoint.get("model_config")
        if model_config is None:
            raise KeyError(
                _ERR_CHECKPOINT_NO_CONFIG.format(
                    path_obj, list(checkpoint.keys())
                )
            )

        # Rebuild architecture from config
        model = cls._build_model_from_config(model_config)

        # Load state dict
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict"),
        )
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
            logger.info(
                "Loaded state_dict into model built from config (strict=False)"
            )

        # Infer metadata
        task = model_config.get("task", "classification")
        input_size = model_config.get(
            "input_size",
            (
                model_config.get("input_height", _DEFAULT_INPUT_SIZE[0]),
                model_config.get("input_width", _DEFAULT_INPUT_SIZE[1]),
            ),
        )
        if isinstance(input_size, (list, tuple)) and len(input_size) == _ERR_INPUT_SIZE_DIM:  # noqa: PLR2004
            input_size = (int(input_size[0]), int(input_size[1]))
        else:
            input_size = _DEFAULT_INPUT_SIZE

        num_classes = model_config.get("num_classes")

        return cls(
            model=model,
            task=task,  # type: ignore[arg-type]
            input_size=input_size,
            device=device,
            num_classes=num_classes,
        )

    # ==================================================================
    # Fluent setters
    # ==================================================================

    def set_loss_fn(self, loss_fn: object) -> CoreModel:
        """Set the loss function for training.

        Args:
            loss_fn: A callable ``loss_fn(preds, targets) -> Tensor``.

        Returns:
            ``self`` for chaining.
        """
        self._loss_fn = loss_fn
        return self

    def set_train_dataloader(self, loader: DataLoader) -> CoreModel:
        """Set the training :class:`DataLoader`.

        Args:
            loader: Training data loader.

        Returns:
            ``self`` for chaining.
        """
        self._train_loader = loader
        return self

    def set_val_dataloader(self, loader: DataLoader) -> CoreModel:
        """Set the validation :class:`DataLoader`.

        Args:
            loader: Validation data loader.

        Returns:
            ``self`` for chaining.
        """
        self._val_loader = loader
        return self

    # ==================================================================
    # Training
    # ==================================================================

    @overload
    def train(  # type: ignore[misc]
        self,
        config: str,
        *,
        target_hardware: str = "server",
        **kwargs: object,
    ) -> dict[str, list]: ...

    @overload
    def train(
        self,
        config: dict[str, Any],
        *,
        target_hardware: str = "server",
        **kwargs: object,
    ) -> dict[str, list]: ...

    @overload
    def train(
        self,
        config: TrainingConfig,
        *,
        target_hardware: str = "server",
        **kwargs: object,
    ) -> dict[str, list]: ...

    @overload
    def train(
        self,
        *,
        target_hardware: str = "server",
        **kwargs: object,
    ) -> dict[str, list]: ...

    def train(
        self,
        config: str | dict[str, Any] | TrainingConfig | None = None,
        *,
        target_hardware: str = "server",
        **kwargs: object,
    ) -> dict[str, list]:
        """Train the model with the given configuration.

        Accepts a polymorphic ``config`` argument:

        * ``str`` — Path to a ``.yaml`` configuration file.
        * ``dict`` — Configuration dictionary.
        * :class:`TrainingConfig` — A validated dataclass instance.
        * ``None`` — All parameters are provided via ``**kwargs``.

        In all cases, keyword arguments take precedence over values in
        ``config`` (when both are present).

        **Required pre-conditions** (must be set before calling ``train``):

        * A train :class:`DataLoader` via :meth:`set_train_dataloader`
          or passed via ``config``.
        * A loss function via :meth:`set_loss_fn`.

        Args:
            config: Path to a ``.yaml`` file, a ``dict``, or a
                :class:`TrainingConfig` instance.  ``None`` means all
                parameters come from ``**kwargs``.
            target_hardware: Hardware profile.  ``"edge"`` applies
                activation rewrites (GELU -> ReLU, SiLU -> Hardswish)
                and LayerNorm collapses **before** building the
                optimiser.  ``"server"`` (default) skips rewrites.
            **kwargs: Additional or overriding training hyperparameters.
                See :class:`TrainingConfig` for supported keys.

        Returns:
            A history dictionary with keys ``"train"`` and ``"val"``,
            each containing a list of per-epoch metric dictionaries.

        Raises:
            ValueError: If configuration validation fails or required
                components are missing.
            RuntimeError: If the training engine encounters an error.

        Example::

            >>> # Keyword arguments
            >>> history = model.train(epochs=10, lr=0.001, batch_size=64)

            >>> # Dictionary
            >>> history = model.train({"epochs": 10, "lr": 0.001})

            >>> # YAML file
            >>> history = model.train("configs/train.yaml")

            >>> # Dataclass
            >>> cfg = TrainingConfig(epochs=10, lr=0.001)
            >>> history = model.train(cfg)

            >>> # Mixed (kwargs override dict)
            >>> history = model.train({"epochs": 10}, batch_size=128)
        """
        # --- Resolve and validate training config -----------------------
        train_cfg: TrainingConfig = self._resolve_train_config(
            config, target_hardware=target_hardware, **kwargs
        )

        # --- Check required components ----------------------------------
        if self._train_loader is None:
            raise ValueError(_ERR_MISSING_DATALOADER)
        if self._loss_fn is None:
            raise ValueError(_ERR_MISSING_LOSS_FN)

        # --- Resolve device ---------------------------------------------
        device: torch.device = (
            torch.device(train_cfg.device)
            if train_cfg.device is not None
            else self._device
        )

        # --- Rewrite graph for edge hardware (if requested) ------------
        if train_cfg.target_hardware == "edge":
            logger.info(
                "Applying edge-hardware graph rewrites (GELU -> ReLU,"
                " SiLU -> Hardswish, LayerNorm collapses)"
            )
            self._model = TargetRewriter().rewrite_for_edge(self._model)

        # --- Build optimiser and scheduler ------------------------------
        optimiser: torch.optim.Optimizer = self._build_optimizer(train_cfg)
        scheduler: object | None = self._build_scheduler(train_cfg, optimiser)

        # --- Create CoreTrainer -----------------------------------------
        self._trainer = CoreTrainer(
            model=self._model,
            optimizer=optimiser,
            loss_fn=self._loss_fn,
            train_dataloader=self._train_loader,
            val_dataloader=self._val_loader,
            device=device,
            gradient_accumulation_steps=train_cfg.grad_accum,
            max_grad_norm=train_cfg.clip_grad,
            use_amp=train_cfg.amp,
            ema_decay=train_cfg.ema_decay if train_cfg.ema else None,
            scheduler=scheduler,
            output_dir=train_cfg.output_dir,
        )

        # --- Run training loop ------------------------------------------
        logger.info(
            "Starting training for %d epochs (task=%s, device=%s, lr=%.6f)",
            train_cfg.epochs,
            self._task,
            device.type,
            train_cfg.lr,
        )
        history: dict[str, list] = self._trainer.fit(
            num_epochs=train_cfg.epochs,
        )
        logger.info("Training completed.")
        return history

    # ==================================================================
    # Inference / Prediction
    # ==================================================================

    def predict(  # noqa: PLR0913
        self,
        source: (
            str | Path | torch.Tensor | list[str | Path | torch.Tensor]
        ),
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        topk: int | None = None,
        half_precision: bool = False,
        compile_model: bool = False,
        batch_size: int = 8,
        weights: str | Path | None = None,
    ) -> list[Prediction]:
        """Run inference on one or more images.

        Delegates to :class:`CorePredictor` which handles preprocessing,
        GPU-native post-processing, and batching.

        Args:
            source: Input source — a single image path, a list of paths,
                a ``torch.Tensor``, or a list of tensors.
            conf_threshold: Minimum confidence score for detection
                predictions.  ``None`` uses the predictor default (0.25).
            iou_threshold: IoU threshold for NMS in detection.
                ``None`` uses the predictor default (0.45).
            topk: Number of top predictions for classification.
                ``None`` uses the predictor default (5).
            half_precision: Enable FP16 inference via ``torch.autocast``.
                Default ``False``.
            compile_model: Enable ``torch.compile`` on the model (requires
                PyTorch >= 2.0).  Default ``False``.
            batch_size: Maximum batch size for list / folder inference.
                Default ``8``.
            weights: Optional path to a ``.pt`` / ``.pth`` checkpoint.
                If provided, the model weights are loaded from this file
                before prediction.  This is useful for swapping weights
                without rebuilding the :class:`CoreModel`.

        Returns:
            A list of :class:`Prediction` objects, one per input image.

        Example::

            >>> # Single image
            >>> preds = model.predict("photo.jpg", topk=5)
            >>> print(preds[0].classification.class_ids)

            >>> # Batch of tensors
            >>> batch = torch.randn(4, 3, 224, 224)
            >>> preds = model.predict(list(batch))

            >>> # Load weights before prediction
            >>> preds = model.predict("photo.jpg", weights="best.pt")
        """
        # Load weights if provided
        if weights is not None:
            self._load_weights(weights)

        # Use provided thresholds or fall back to defaults
        _conf: float = conf_threshold if conf_threshold is not None else 0.25
        _iou: float = iou_threshold if iou_threshold is not None else 0.45
        _topk: int = topk if topk is not None else 5

        # Build predictor lazily or with updated params
        if self._predictor is None or self._needs_predictor_rebuild(
            _conf, _iou, _topk, half_precision, compile_model, batch_size,
        ):
            self._predictor = self._build_predictor(
                conf_threshold=_conf,
                iou_threshold=_iou,
                topk=_topk,
                half_precision=half_precision,
                compile_model=compile_model,
                batch_size=batch_size,
            )

        # Run prediction
        return self._predictor.predict(source)

    # ==================================================================
    # Export
    # ==================================================================

    def export(  # noqa: PLR0913
        self,
        format: str = "onnx",
        target_hardware: str = "server",
        opset: int = _DEFAULT_OPSET,
        optimize: bool = True,
        output_path: str | None = None,
        input_shape: tuple[int, ...] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        weights: str | Path | None = None,
    ) -> dict[str, str]:
        """Export the model to ONNX and/or ExecuTorch format.

        Delegates to :class:`CoreExporter` which internally uses
        :class:`TargetRewriter` (for edge-hardware graph rewrites)
        and :class:`MetaProber` (for zero-VRAM shape validation).

        The export pipeline is:

        1. **Rewrite** — When ``target_hardware='edge'``, applies
           activation replacements (GELU -> ReLU, SiLU -> Hardswish)
           and collapses redundant layout permutations.
        2. **Validate** — Runs shape propagation on ``device='meta'``
           and audits the graph for dynamic operations.
        3. **Export** — Serialises to ``.onnx`` and/or ``.pte``.

        Args:
            format: Export target.  One of ``"onnx"``, ``"executorch"``,
                ``"both"``.
            target_hardware: Hardware profile.  ``"edge"`` applies
                activation rewrites; ``"server"`` skips them.
            opset: ONNX opset version (``17`` or ``18``).
            optimize: When ``True``, enables TargetRewriter graph
                optimisations and XNNPACK delegate (for ExecuTorch).
            output_path: Explicit output file path.  For ``"both"``
                format, this is used as a prefix (e.g. ``"model"``
                produces ``"model.onnx"`` and ``"model.pte"``).
                If ``None``, a timestamped path is auto-generated.
            input_shape: Input tensor shape ``(B, C, H, W)``.
                Defaults to ``(1, 3, H, W)`` using ``self.input_size``.
            dynamic_axes: ONNX-style dynamic axes dictionary.
                ``None`` means static shapes.
            weights: Optional path to a ``.pt`` / ``.pth`` checkpoint.
                If provided, the model weights are loaded from this file
                before export.  The architecture must already be set
                (e.g. via config or a previous checkpoint load).

        Returns:
            A dictionary mapping format names to file paths, e.g.
            ``{"onnx": "/path/to/model.onnx"}``.

        Raises:
            ValueError: If any parameter is invalid.
            RuntimeError: If validation or export fails.

        Example::

            >>> # ONNX for server
            >>> paths = model.export(format="onnx", target_hardware="server")
            >>> paths["onnx"]
            '.../model_20260723_021130.onnx'

            >>> # ExecuTorch for edge with rewrites
            >>> paths = model.export(
            ...     format="executorch",
            ...     target_hardware="edge",
            ...     opset=18,
            ... )
            >>> paths["executorch"]
            '.../model_20260723_021131.pte'

            >>> # Both formats
            >>> paths = model.export(format="both")
            >>> list(paths.keys())
            ['onnx', 'executorch']

            >>> # Load weights before export
            >>> paths = model.export(format="onnx", weights="best.pt")
        """
        # --- Load weights if provided -----------------------------------
        if weights is not None:
            self._load_weights(weights)

        # --- Validate export config -------------------------------------
        export_cfg = self._resolve_export_config(
            format=format,
            target_hardware=target_hardware,
            opset=opset,
            optimize=optimize,
            output_path=output_path,
            input_shape=input_shape,
            dynamic_axes=dynamic_axes,
        )

        # --- Build CoreExporter -----------------------------------------
        ex: CoreExporter = CoreExporter(
            model=self._model,
            target=export_cfg.format,
            opset_version=export_cfg.opset,
            target_hardware=export_cfg.target_hardware,
            input_shape=export_cfg.input_shape,
            dynamic_axes=export_cfg.dynamic_axes,
            output_dir=str(Path(export_cfg.output_path).parent)
            if export_cfg.output_path
            else tempfile.mkdtemp(prefix="corecv_export_"),
        )

        # --- Run export pipeline ----------------------------------------
        logger.info(
            "Starting export (format=%s, hardware=%s, opset=%d, optimize=%s)",
            export_cfg.format,
            export_cfg.target_hardware,
            export_cfg.opset,
            export_cfg.optimize,
        )

        results: dict[str, str] = ex.run_export()

        # --- Rename if explicit output_path was provided ----------------
        if export_cfg.output_path is not None:
            results = self._rename_export_outputs(results, export_cfg.output_path)

        logger.info("Export completed: %s", results)
        return results

    # ==================================================================
    # Internal: config resolution & validation
    # ==================================================================

    @staticmethod
    def _resolve_train_config(
        config: str | dict[str, Any] | TrainingConfig | None,
        **kwargs: object,
    ) -> TrainingConfig:
        """Resolve and merge a polymorphic training configuration.

        Args:
            config: YAML path, dict, TrainingConfig, or ``None``.
            **kwargs: Overriding keyword arguments.  May include
                ``target_hardware`` and other :class:`TrainingConfig` fields.

        Returns:
            A validated :class:`TrainingConfig` instance.

        Raises:
            ValueError: If the configuration is invalid.
            FileNotFoundError: If a YAML path does not exist.
        """
        base: dict[str, Any] = {}

        if isinstance(config, TrainingConfig):
            # Start from dataclass fields, then override with kwargs
            base = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
        elif isinstance(config, str):
            # Load YAML file
            import yaml  # noqa: PLC0415

            path: Path = Path(config)
            if not path.exists():
                raise FileNotFoundError(
                    _ERR_TRAIN_CONFIG_NOT_FOUND.format(path)
                )
            with path.open("r", encoding="utf-8") as f:
                raw: Any = yaml.safe_load(f)
            if not isinstance(raw, dict):
                raise TypeError(
                    _ERR_YAML_NOT_MAPPING.format(type(raw).__name__)
                )
            base = raw
        elif isinstance(config, dict):
            base = dict(config)
        elif config is not None:
            raise TypeError(
                _ERR_CONFIG_TYPE.format(type(config).__name__)
            )

        # Merge kwargs on top (kwargs take precedence)
        base.update(kwargs)  # type: ignore[arg-type]

        # Validate and return
        return TrainingConfig(**base)

    @staticmethod
    def _resolve_export_config(  # noqa: PLR0913
        format: str = "onnx",
        target_hardware: str = "server",
        opset: int = _DEFAULT_OPSET,
        optimize: bool = True,
        output_path: str | None = None,
        input_shape: tuple[int, ...] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
    ) -> ExportConfig:
        """Resolve and validate export configuration.

        Args:
            format: Export format.
            target_hardware: Hardware profile.
            opset: ONNX opset version.
            optimize: Enable optimisations.
            output_path: Explicit output path.
            input_shape: Input tensor shape.
            dynamic_axes: Dynamic axes mapping.

        Returns:
            A validated :class:`ExportConfig` instance.
        """
        return ExportConfig(
            format=format,
            target_hardware=target_hardware,
            opset=opset,
            optimize=optimize,
            output_path=output_path,
            input_shape=input_shape or (1, 3, *_DEFAULT_INPUT_SIZE),
            dynamic_axes=dynamic_axes,
        )

    # ==================================================================
    # Internal: engine builders
    # ==================================================================

    def _build_predictor(  # noqa: PLR0913
        self,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        topk: int = 5,
        half_precision: bool = False,
        compile_model: bool = False,
        batch_size: int = 8,
    ) -> CorePredictor:
        """Build a :class:`CorePredictor` for this model.

        Args:
            conf_threshold: Detection confidence threshold.
            iou_threshold: NMS IoU threshold.
            topk: Classification top-k.
            half_precision: Enable FP16.
            compile_model: Enable ``torch.compile``.
            batch_size: Max batch size.

        Returns:
            A configured :class:`CorePredictor`.
        """
        return CorePredictor(
            model=self._model,
            task=self._task,  # type: ignore[arg-type]
            input_size=self._input_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            topk=topk,
            half_precision=half_precision,
            compile_model=compile_model,
            batch_size=batch_size,
            num_classes=self._num_classes,
        )

    def _build_optimizer(
        self,
        cfg: TrainingConfig,
    ) -> torch.optim.Optimizer:
        """Build a PyTorch optimiser from training configuration.

        Args:
            cfg: Validated training configuration.

        Returns:
            A configured :class:`torch.optim.Optimizer`.

        Raises:
            ValueError: If the optimiser type is unknown.
        """
        params: list[torch.Tensor] = list(self._model.parameters())

        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr)
        if cfg.optimizer == "adam":
            return torch.optim.Adam(params, lr=cfg.lr)
        if cfg.optimizer == "sgd":
            return torch.optim.SGD(
                params, lr=cfg.lr, momentum=0.9, weight_decay=1e-4,
            )

        raise ValueError(_ERR_UNKNOWN_OPTIMIZER.format(cfg.optimizer, sorted(_VALID_OPTIMIZERS)))

    def _build_scheduler(
        self,
        cfg: TrainingConfig,
        optimizer: torch.optim.Optimizer,
    ) -> object | None:
        """Build an LR scheduler from training configuration.

        Args:
            cfg: Validated training configuration.
            optimizer: The optimiser to schedule.

        Returns:
            A PyTorch LR scheduler, or ``None``.
        """
        if cfg.scheduler is None or cfg.scheduler == "none":
            return None

        if cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs,
            )
        if cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1,
            )

        raise ValueError(_ERR_UNKNOWN_SCHEDULER.format(cfg.scheduler, sorted(_VALID_SCHEDULERS)))

    # -- Internal helpers -------------------------------------------------

    def _load_from_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load model weights and architecture from a checkpoint file.

        The checkpoint is expected to contain:
        * ``model_config`` (:class:`dict`) — architecture parameters used
          to rebuild the model via :meth:`_build_model_from_config`.
        * ``model_state_dict`` or ``state_dict`` — the trained weights.

        This method sets ``self._model`` to the rebuilt model with loaded
        weights and returns the ``model_config`` dictionary so that callers
        (including :meth:`__init__`) can extract metadata such as
        ``num_classes``.

        Args:
            path: Path to a ``.pt`` or ``.pth`` checkpoint file.

        Returns:
            The ``model_config`` dictionary extracted from the checkpoint.

        Raises:
            FileNotFoundError: If the path does not exist.
            RuntimeError: If loading fails.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(
                _ERR_CHECKPOINT_NOT_FOUND.format(path_obj)
            )

        checkpoint: dict[str, Any] = torch.load(
            str(path_obj), map_location="cpu", weights_only=False
        )

        model_config: dict[str, Any] = checkpoint.get("model_config", {})
        if model_config:
            self._model = self._build_model_from_config(model_config)
        else:
            # Fallback: assume checkpoint contains a full model
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                # We can't rebuild without config, so raise
                raise KeyError(
                    _ERR_CHECKPOINT_NO_CONFIG_BARE.format(path_obj)
                )
            # Bare state_dict — wrap as-is (caller must set self._model)
            self._model = nn.Module()

        state_dict = checkpoint.get(
            "model_state_dict", checkpoint.get("state_dict")
        )
        if state_dict is not None:
            load_result = self._model.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                logger.warning(
                    "Missing keys in state_dict load: %s", load_result.missing_keys
                )
            if load_result.unexpected_keys:
                logger.warning(
                    "Unexpected keys in state_dict load: %s",
                    load_result.unexpected_keys,
                )

        return model_config

    def _load_from_config(self, path: str | Path) -> None:
        """Build a model from a YAML configuration file.

        Loads the YAML file using
        :func:`~corecv.config.schemas.load_config`, validates it against
        the task-specific schema, and builds the model via
        :meth:`_build_model_from_config`.

        The constructed model is stored in ``self._model``.

        Args:
            path: Path to a ``.yaml`` or ``.yml`` configuration file.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the configuration is invalid.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(
                _ERR_CONFIG_FILE_NOT_FOUND.format(path_obj)
            )

        task_config = load_config(str(path_obj))
        raw: dict[str, Any] = dataclasses.asdict(task_config)
        self._model = self._build_model_from_config(raw)

    @staticmethod
    def _filter_kwargs_for_signature(
        target_cls: type,
        config: dict[str, Any],
        extra_keys: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Filter *config* to keys accepted by *target_cls* constructor.

        Uses :func:`inspect.signature` to introspect *target_cls* and
        retain only the entries whose keys match formal parameter names.
        If the constructor accepts ``**kwargs``, all config entries are
        forwarded.

        Args:
            target_cls: The class whose constructor signature is inspected.
            config: Source dictionary of potential keyword arguments.
            extra_keys: Mandatory key-value pairs (e.g. ``feature_info``)
                that are always included in the result.

        Returns:
            A filtered dictionary suitable for ``target_cls(**result)``.
        """
        sig = inspect.signature(target_cls)
        params = sig.parameters

        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )

        if has_var_keyword:
            filtered = dict(config)
        else:
            accepted = {
                name
                for name, p in params.items()
                if p.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            }
            filtered = {k: v for k, v in config.items() if k in accepted}

        if extra_keys:
            filtered.update(extra_keys)

        return filtered

    @staticmethod
    def _build_model_from_config(config: dict[str, Any]) -> nn.Module:
        """Build a model from a configuration dictionary.

        Uses the CoreCV registries (backbone, neck, head) to resolve
        components by name and wire them together according to the
        ``task`` field.

        Supported tasks:
        * ``"classification"`` — backbone + classification head
        * ``"segmentation"`` — backbone + segmentation decoder head
        * ``"detection"`` — backbone + neck + detection head wrapped in
          :class:`~corecv.models.detector.CoreObjectDetector`

        Args:
            config: Dictionary containing at minimum ``task``,
                ``model_name``, and ``num_classes``.  Additional fields
                are passed to component constructors.

        Returns:
            A fully assembled :class:`nn.Module`.

        Raises:
            ValueError: If ``task`` or ``model_name`` are unknown or
                missing.
            KeyError: If a required component is not found in its
                registry.
        """
        task: str = config.get("task", "")
        model_name: str = config.get("model_name", "")
        num_classes: int = config.get("num_classes", 1000)
        pretrained: bool = config.get("pretrained", True)

        if not task:
            raise ValueError(_ERR_MISSING_TASK_IN_CONFIG)
        if not model_name:
            raise ValueError(_ERR_MISSING_MODEL_NAME)

        # --- Resolve and build backbone ---------------------------------
        backbone_cls = get_backbone(model_name)
        backbone: nn.Module = backbone_cls(pretrained=pretrained)

        # Build task-specific model
        if task == "classification":
            head_type: str = config.get("head_type", "linear_classification")
            head_cls = get_head(head_type)
            head_kwargs = CoreModel._filter_kwargs_for_signature(
                head_cls,
                config,
                extra_keys={
                    "feature_info": backbone.feature_info,  # type: ignore[attr-defined]
                    "num_classes": num_classes,
                },
            )
            head = head_cls(**head_kwargs)
            model = nn.Sequential(OrderedDict([
                ("backbone", backbone),
                ("head", head),
            ]))

        elif task == "segmentation":
            head_type = config.get("head_type", "aspp_decoder")
            head_cls = get_head(head_type)
            decoder_channels: int = config.get("decoder_channels", 256)
            head_kwargs = CoreModel._filter_kwargs_for_signature(
                head_cls,
                config,
                extra_keys={
                    "feature_info": backbone.feature_info,  # type: ignore[attr-defined]
                    "out_channels": decoder_channels,
                    "num_classes": num_classes,
                },
            )
            head = head_cls(**head_kwargs)
            model = nn.Sequential(OrderedDict([
                ("backbone", backbone),
                ("head", head),
            ]))

        elif task == "detection":
            neck_type: str = config.get("neck_type", "fpn")
            head_type = config.get("head_type", "decoupled_anchor_free")
            neck_channels: int = config.get(
                "neck_channels", config.get("out_channels", 256),
            )
            neck_cls = get_neck(neck_type)
            neck_kwargs = CoreModel._filter_kwargs_for_signature(
                neck_cls,
                config,
                extra_keys={
                    "feature_info": backbone.feature_info,  # type: ignore[attr-defined]
                    "out_channels": neck_channels,
                },
            )
            neck = neck_cls(**neck_kwargs)
            head_cls = get_head(head_type)
            head_kwargs = CoreModel._filter_kwargs_for_signature(
                head_cls,
                config,
                extra_keys={
                    "feature_info": backbone.feature_info,  # type: ignore[attr-defined]
                    "num_classes": num_classes,
                },
            )
            head = head_cls(**head_kwargs)
            model = CoreObjectDetector(
                backbone=backbone, neck=neck, head=head,
            )

        else:
            raise ValueError(
                _ERR_UNKNOWN_TASK_IN_CONFIG.format(task)
            )

        return model

    def _load_weights(self, path: str | Path) -> None:
        """Load a checkpoint's ``state_dict`` into the current model.

        This is a lightweight weight-loading helper that does **not**
        rebuild the model architecture — the architecture must already
        match the checkpoint.  It is used by :meth:`predict` and
        :meth:`export` when the ``weights`` parameter is provided.

        Args:
            path: Path to a ``.pt`` or ``.pth`` checkpoint file.

        Raises:
            FileNotFoundError: If the path does not exist.
            RuntimeError: If the state dict cannot be loaded.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(
                _ERR_WEIGHTS_NOT_FOUND.format(path_obj)
            )

        checkpoint: dict[str, Any] = torch.load(
            str(path_obj), map_location="cpu", weights_only=False
        )

        state_dict = checkpoint.get(
            "model_state_dict", checkpoint.get("state_dict", checkpoint)
        )
        if isinstance(state_dict, dict) and all(
            isinstance(k, str) for k in state_dict
        ):
            load_result = self._model.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                logger.warning(
                    "Missing keys loading weights: %s", load_result.missing_keys
                )
            if load_result.unexpected_keys:
                logger.warning(
                    "Unexpected keys loading weights: %s",
                    load_result.unexpected_keys,
                )
            logger.info("Loaded weights from %s", path_obj)
        else:
            raise RuntimeError(
                _ERR_INVALID_STATE_DICT.format(path_obj)
            )

    def _infer_num_classes(self) -> int | None:
        """Attempt to infer the number of classes from the model head.

        Looks for a ``head.num_classes`` attribute, then falls back to
        ``model.num_classes``.

        Returns:
            The number of classes, or ``None`` if inference fails.
        """
        head = getattr(self._model, "head", None)
        if head is not None and hasattr(head, "num_classes"):
            return int(head.num_classes)
        if hasattr(self._model, "num_classes"):
            return int(self._model.num_classes)
        return None

    def _needs_predictor_rebuild(  # noqa: PLR0913
        self,
        conf_threshold: float,
        iou_threshold: float,
        topk: int,
        half_precision: bool,
        _compile_model: bool,
        batch_size: int,
    ) -> bool:
        """Check whether the cached predictor needs to be rebuilt.

        Compares the desired parameters against the existing predictor's
        attributes.

        Args:
            conf_threshold: Desired confidence threshold.
            iou_threshold: Desired IoU threshold.
            topk: Desired top-k.
            half_precision: Desired FP16 flag.
            _compile_model: Desired compile flag (unused, triggers rebuild
                whenever the predictor is None).
            batch_size: Desired batch size.

        Returns:
            ``True`` if a rebuild is needed.
        """
        if self._predictor is None:
            return True

        p = self._predictor
        return bool(
            p.conf_threshold != conf_threshold
            or p.iou_threshold != iou_threshold
            or p.topk != topk
            or p.half_precision != half_precision
            or p.batch_size != batch_size
        )

    @staticmethod
    def _rename_export_outputs(
        results: dict[str, str],
        output_path: str,
    ) -> dict[str, str]:
        """Rename export output files to a user-specified path.

        For ``"both"`` format where two files are produced, the
        ``output_path`` serves as a stem.

        Args:
            results: Map of format -> current file path.
            output_path: Desired output path or stem.

        Returns:
            Updated dictionary with renamed paths.
        """
        renamed: dict[str, str] = {}
        out = Path(output_path)

        if len(results) == 1:
            # Single output: use the path directly (add extension if needed)
            fmt = next(iter(results.keys()))
            ext = ".onnx" if fmt == "onnx" else ".pte"
            target = out if out.suffix else out.with_suffix(ext)
            renamed[fmt] = str(target)
        else:
            # Multiple outputs: use output_path as a stem
            stem = out.stem if out.suffix else out.name
            parent = out.parent
            for fmt, current_path in results.items():
                ext = Path(current_path).suffix
                renamed[fmt] = str(parent / f"{stem}{ext}")

        return renamed

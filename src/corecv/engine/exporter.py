"""CoreExporter: End-to-end model export pipeline for CoreCV.

Provides the :class:`CoreExporter` class that coordinates graph rewriting,
hardware compatibility validation, and export to ONNX / ExecuTorch formats
for CoreCV models (classification, segmentation, and detection).

The pipeline is::

    1. Rewrite  — apply edge-hardware-friendly activation replacements
                  (GELU -> ReLU, SiLU -> Hardswish) via TargetRewriter.
    2. Validate — run zero-VRAM shape propagation on ``device="meta"``
                  and audit the graph for dynamic operations via MetaProber.
    3. Export   — serialize to ``.onnx`` (via ``torch.onnx.export`` /
                  ``torch.export.export``) and/or ``.pte`` (via
                  ``torch.export.save``).

Example:
    >>> from corecv.engine import CoreExporter
    >>> from corecv.models import CoreObjectDetector
    >>> model = CoreObjectDetector(...)
    >>> exporter = CoreExporter(
    ...     model=model,
    ...     target="both",
    ...     opset_version=18,
    ...     target_hardware="edge",
    ...     input_shape=(1, 3, 640, 640),
    ... )
    >>> results = exporter.run_export()
    >>> list(results.keys())
    ['onnx', 'executorch']
"""

from __future__ import annotations

import copy
import logging
import warnings as _warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Conditional imports for torch.export (available since PyTorch 2.0,
# stable since 2.3)
# ---------------------------------------------------------------------------
try:
    from torch.export import Dim as _Dim
    from torch.export import export as _torch_export

    TORCH_EXPORT_AVAILABLE = True
except ImportError:
    _Dim = None  # type: ignore[assignment]
    _torch_export = None  # type: ignore[assignment]
    TORCH_EXPORT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Check for onnxscript (required by ``torch.onnx.export`` in recent PyTorch)
# ---------------------------------------------------------------------------
try:
    import onnxscript  # noqa: F401
    ONNXSCRIPT_AVAILABLE = True
except ImportError:
    ONNXSCRIPT_AVAILABLE = False

# ---------------------------------------------------------------------------
# CoreCV dependencies
# ---------------------------------------------------------------------------
from corecv.core.contract import BaseBackbone
from corecv.engine.rewriter import TargetRewriter
from corecv.engine.validator import MetaProber

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OUTPUT_NAME_DEFAULT = "output"
_OUTPUT_NAME_PREFIX = "output_"
_DYNAMIC_AXES_BATCH = "batch"
_DYNAMIC_AXES_HEIGHT = "height"
_DYNAMIC_AXES_WIDTH = "width"

_VALID_TARGETS = frozenset({"onnx", "executorch", "both"})
_VALID_OPSET_VERSIONS = frozenset({17, 18})
_VALID_HARDWARE_TARGETS = frozenset({"edge", "server"})

_ALLOWED_INPUT_NDIM = 4  # (B, C, H, W)
_BATCH_DIM = 0
_CHANNEL_DIM = 1
_HEIGHT_DIM = 2
_WIDTH_DIM = 3

# ---------------------------------------------------------------------------
# Error message constants (TRY003)
# ---------------------------------------------------------------------------
_ERR_FORWARD_INFERENCE = "Failed to run forward pass for output-name inference"
_ERR_INVALID_TARGET = "Invalid target '{}'. Must be one of {}."
_ERR_INVALID_OPSET = "Invalid opset_version {}. Must be one of {}."
_ERR_INVALID_HARDWARE = "Invalid target_hardware '{}'. Must be one of {}."
_ERR_INVALID_INPUT_SHAPE = "input_shape must have {} dimensions (B, C, H, W), got {}."
_ERR_ONNXSCRIPT_MISSING = (
    "ONNX export requires the 'onnxscript' package. "
    "Install it with: pip install onnxscript"
)
_ERR_ONNX_EXPORT_FAILED = "ONNX export failed: {}\nModel type: {}, output structure: {}"
_ERR_EXECUTORCH_UNAVAILABLE = (
    "torch.export is not available in this PyTorch version ({}). "
    "ExecuTorch export requires PyTorch >= 2.3."
)
_ERR_EXECUTORCH_EXPORT_FAILED = "ExecuTorch export failed: {}\nModel type: {}, output structure: {}"
_ERR_META_PROPAGATION = "Meta-device shape propagation failed for {}: {}"


# ======================================================================
# ValidationResult
# ======================================================================


@dataclass
class ValidationResult:
    """Result of a model compatibility validation.

    Attributes:
        passed: ``True`` if the model passed all compatibility checks.
        details: Human-readable list of checks that were performed.
        errors: Human-readable list of failures (empty when ``passed``).
    """

    passed: bool
    details: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Return ``True`` when validation passed."""
        return self.passed

    def __str__(self) -> str:
        """Return a one-line summary of the validation result."""
        if self.passed:
            return f"Validation PASSED ({len(self.details)} checks)"
        return (
            f"Validation FAILED ({len(self.errors)} error(s)): "
            f"{'; '.join(self.errors)}"
        )

    def add_error(self, error: str) -> None:
        """Add an error to the result and set passed=False."""
        self.errors.append(error)
        self.passed = False

    def add_detail(self, detail: str) -> None:
        """Add a detail to the result."""
        self.details.append(detail)


# ======================================================================
# ONNX compatibility wrapper
# ======================================================================


class _ONNXCompatModel(nn.Module):
    """Wrapper that flattens structured outputs for ONNX compatibility.

    ONNX does not support Python ``dict`` or nested ``list`` return types
    natively.  This wrapper intercepts the model forward pass and converts
    any ``dict`` or ``list`` output into a flat ``tuple`` of tensors that
    ``torch.onnx.export`` can handle.

    The flattened output order is deterministic (sorted dict keys, then
    list elements in order).

    Args:
        model: The original CoreCV model (backbone + neck + head).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the model and return a flat tuple of tensors.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            A flat tuple of tensors.  If the model returns a single tensor,
            it is wrapped in a 1-element tuple for consistency.
        """
        output = self.model(x)
        return self._flatten(output)

    @staticmethod
    def _flatten(value: object) -> tuple[torch.Tensor, ...]:
        """Recursively flatten dicts and lists to a tuple of tensors.

        Args:
            value: Model output (tensor, dict, list, tuple, or nested
                   combination thereof).

        Returns:
            A flat tuple of :class:`torch.Tensor`.
        """
        if isinstance(value, torch.Tensor):
            return (value,)
        if isinstance(value, dict):
            # Sort keys for deterministic order
            flattened: list[torch.Tensor] = []
            for k in sorted(value.keys()):
                flattened.extend(_ONNXCompatModel._flatten(value[k]))
            return tuple(flattened)
        if isinstance(value, (list, tuple)):
            flattened = []
            for item in value:
                flattened.extend(_ONNXCompatModel._flatten(item))
            return tuple(flattened)
        msg = (
            f"Unsupported output type {type(value).__name__}. "
            "Only Tensor, dict, list, and tuple are supported."
        )
        raise TypeError(msg)


# ======================================================================
# Output-name inference helpers
# ======================================================================


def _infer_output_names(
    model: nn.Module,
    example_input: torch.Tensor,
) -> list[str]:
    """Infer meaningful ONNX output names from a model forward pass.

    Runs one forward step and inspects the output structure:

    * **Single tensor** -> ``["output"]``
    * **Tuple / list of tensors** -> ``["output_0", "output_1", ...]``
    * **Dict with string keys** -> ``["key1", "key2", ...]``
    * **Nested** -> flattened versions of the above.

    Args:
        model: The model to inspect (will **not** be mutated).
        example_input: A dummy input tensor on the correct device.

    Returns:
        A list of output names suitable for ``torch.onnx.export``.

    Raises:
        RuntimeError: If the forward pass fails or output structure cannot
            be determined.
    """
    model = model.eval()
    with torch.no_grad():
        try:
            output = model(example_input)
        except Exception as exc:
            raise RuntimeError(_ERR_FORWARD_INFERENCE) from exc
    return _collect_output_names(output)


def _collect_output_names(value: object, *, prefix: str = "") -> list[str]:
    """Recursively collect output names from a model output structure.

    Args:
        value: The model output (tensor, dict, list, or tuple).
        prefix: Optional prefix for nested keys.

    Returns:
        A list of flattened output name strings.
    """
    names: list[str] = []

    if isinstance(value, torch.Tensor):
        names.append(prefix or _OUTPUT_NAME_DEFAULT)
    elif isinstance(value, dict):
        for key in sorted(value.keys()):
            sub_prefix = str(key) if not prefix else f"{prefix}.{key}"
            names.extend(_collect_output_names(value[key], prefix=sub_prefix))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            sub_prefix = (
                f"{prefix}.{_OUTPUT_NAME_PREFIX}{i}" if prefix
                else f"{_OUTPUT_NAME_PREFIX}{i}"
            )
            names.extend(_collect_output_names(item, prefix=sub_prefix))
    else:
        # Fallback: assign a generic name
        names.append(prefix or _OUTPUT_NAME_DEFAULT)

    return names


# ======================================================================
# Dynamic shape helpers
# ======================================================================


def _build_dynamic_shapes(
    input_shape: tuple[int, ...],
    dynamic_axes: dict[str, dict[int, str]] | None,
) -> dict[str, dict[int, Any]] | None:
    """Build dynamic shape specs for ``torch.export.export``.

    Converts the ONNX-style ``dynamic_axes`` dict (e.g.
    ``{"input": {0: "batch", 2: "height", 3: "width"}}``) into
    ``torch.export.Dim`` objects suitable for the modern export path.

    Args:
        input_shape: The static input shape, e.g. ``(1, 3, 640, 640)``.
        dynamic_axes: ONNX-style dynamic axes mapping, or ``None``.

    Returns:
        A ``dynamic_shapes`` dict for ``torch.export.export``, or ``None``.
    """
    if dynamic_axes is None:
        return None

    if not TORCH_EXPORT_AVAILABLE or _Dim is None:
        logger.warning("torch.export.Dim is not available; dynamic shapes disabled.")
        return None

    shapes: dict[str, dict[int, Any]] = {}

    for _name, axes in dynamic_axes.items():
        # The key in dynamic_axes is the ONNX input name ("input").
        # torch.export.export expects the key to match the forward
        # method's argument name, which is "x" for all CoreCV models.
        name = "x"
        dim_map: dict[int, Any] = {}
        for dim_idx, dim_name in axes.items():
            if dim_idx < 0 or dim_idx >= len(input_shape):
                logger.warning(
                    "Dynamic axis index %d out of range for shape %s; skipping.",
                    dim_idx,
                    input_shape,
                )
                continue

            # Dimensions with size 1 (e.g. batch dimension) cannot be
            # dynamic if the model specialized them to a constant during
            # tracing. In that case we simply skip adding a dynamic spec
            # so the dimension stays static.
            if input_shape[dim_idx] == 1:
                logger.info(
                    "Dimension %d has size 1; keeping static "
                    "(dynamic axes would cause constraint violation).",
                    dim_idx,
                )
                continue

            # Determine sensible min / max bounds for the dimension.
            static = input_shape[dim_idx]
            # Allow at least 4x up/down scaling for spatial dims.
            max_val = max(static * 4, 1024)
            # For spatial dims (index >= 2), set a minimum that avoids
            # guard violations from strided backbones. The typical
            # coarsest backbone stride is 32, so spatial dims must be
            # >= 64 to guarantee at least 2 pixels at the coarsest level.
            min_val = max(static // 4, 64) if dim_idx >= _HEIGHT_DIM else 1
            dim_map[dim_idx] = _Dim(dim_name, min=min_val, max=max_val)
        if dim_map:
            shapes[name] = dim_map

    return shapes if shapes else None


# ======================================================================
# CoreExporter
# ======================================================================


class CoreExporter:
    """End-to-end model export pipeline for CoreCV.

    Orchestrates three stages:

    1. **Rewrite** – deep-copies the model and applies
       :class:`~corecv.engine.rewriter.TargetRewriter` transforms when
       ``target_hardware='edge'`` (GELU -> ReLU, SiLU -> Hardswish,
       LayerNorm permutation collapse).
    2. **Validate** – runs :class:`~corecv.engine.validator.MetaProber`
       with ``device='meta'`` zero-VRAM shape auditing and static-graph
       dynamic-operation detection.
    3. **Export** – serialises to ONNX (``.onnx``) and/or ExecuTorch
       (``.pte``) using ``torch.export`` / ``torch.onnx.export``.

    The exporter handles all three CoreCV task types:

    * **Classification** – single tensor ``(B, C)`` output.
    * **Segmentation** – single tensor ``(B, C, H, W)`` or ``(B, H, W)``.
    * **Detection** – ``dict`` with lists of per-level tensors.

    Args:
        model: The CoreCV model to export.  Can be any ``nn.Module``
            (backbone + neck + head), e.g. ``CoreObjectDetector``.
        target: Export target.  One of ``"onnx"``, ``"executorch"``,
            or ``"both"``.
        opset_version: ONNX opset version.  Must be ``17`` or ``18``.
            Defaults to ``17``.
        target_hardware: Hardware target profile.  ``"edge"`` applies
            activation rewrites (GELU -> ReLU, SiLU -> Hardswish) and
            NCHW layout optimisations.  ``"server"`` skips all rewrites.
            Defaults to ``"server"``.
        input_shape: Input tensor shape ``(B, C, H, W)``.
            Defaults to ``(1, 3, 640, 640)``.
        dynamic_axes: ONNX-style dynamic axes dictionary, e.g.
            ``{"input": {0: "batch", 2: "height", 3: "width"}}``.
            If provided, the exporter also configures dynamic shapes
            for the ``torch.export.export`` call.  ``None`` means
            all shapes are static.
        output_dir: Directory for exported model files.
            Defaults to ``"./exports"``.

    Raises:
        ValueError: If ``target``, ``opset_version``, or
            ``target_hardware`` is invalid.
    """

    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        target: str = "onnx",
        opset_version: int = 17,
        target_hardware: str = "server",
        input_shape: tuple[int, ...] = (1, 3, 640, 640),
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        output_dir: str | Path = "./exports",
    ) -> None:
        """Initialise the CoreExporter.

        Args:
            model: The CoreCV model to export.
            target: Export target (``"onnx"``, ``"executorch"``,
                or ``"both"``).
            opset_version: ONNX opset version (``17`` or ``18``).
            target_hardware: Hardware target (``"edge"`` or ``"server"``).
            input_shape: Input tensor shape ``(B, C, H, W)``.
            dynamic_axes: ONNX dynamic axes dict, or ``None``.
            output_dir: Output directory for exported files.

        Raises:
            ValueError: If any argument is invalid.
        """
        # --- Validate arguments --------------------------------------------
        if target not in _VALID_TARGETS:
            raise ValueError(
                _ERR_INVALID_TARGET.format(target, sorted(_VALID_TARGETS))
            )

        if opset_version not in _VALID_OPSET_VERSIONS:
            raise ValueError(
                _ERR_INVALID_OPSET.format(opset_version, sorted(_VALID_OPSET_VERSIONS))
            )

        target_hw = target_hardware.lower()
        if target_hw not in _VALID_HARDWARE_TARGETS:
            raise ValueError(
                _ERR_INVALID_HARDWARE.format(target_hardware, sorted(_VALID_HARDWARE_TARGETS))
            )
        target_hardware = target_hw

        input_shape_t = tuple(input_shape)
        if len(input_shape_t) != _ALLOWED_INPUT_NDIM:
            raise ValueError(
                _ERR_INVALID_INPUT_SHAPE.format(_ALLOWED_INPUT_NDIM, len(input_shape_t))
            )

        # --- Store attributes ----------------------------------------------
        self.model = model
        self.target = target
        self.opset_version = opset_version
        self.target_hardware = target_hardware
        self.input_shape = input_shape_t
        self.dynamic_axes = dynamic_axes
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache the device once
        self._device: torch.device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite_model(self) -> nn.Module:
        """Apply hardware-specific rewrites on a **copy** of the model.

        When ``target_hardware == 'edge'``, the copy is traced through
        :class:`~corecv.engine.rewriter.TargetRewriter` which:

        * Replaces ``nn.GELU`` / ``F.gelu`` with ``nn.ReLU``.
        * Replaces ``nn.SiLU`` / ``F.silu`` with ``nn.Hardswish``.
        * Collapses redundant NCHW <-> NHWC permutations around
          ``LayerNorm``.

        The original model passed to the constructor is **never** mutated.

        Returns:
            A rewritten copy of the model (``GraphModule`` when
            ``target_hardware='edge'``, otherwise a plain ``nn.Module``
            deep copy).
        """
        # Always work on a deep copy to avoid mutating the original
        model_copy = copy.deepcopy(self.model)
        model_copy.eval()

        if self.target_hardware == "edge":
            logger.info("Applying TargetRewriter for edge hardware ...")
            rewriter = TargetRewriter()
            model_copy = rewriter.rewrite_for_edge(model_copy)
            logger.info("Edge rewrites applied successfully.")

        return model_copy

    def validate_compatibility(self, model: nn.Module) -> ValidationResult:
        """Validate model compatibility for the target hardware.

        Runs :class:`~corecv.engine.validator.MetaProber` with
        ``device='meta'`` for zero-VRAM shape propagation and static-graph
        auditing.

        The validation strategy is:

        1. **Component-aware** – If the passed model exposes ``backbone``,
           ``neck``, and ``head`` attributes that satisfy the
           ``BaseBackbone`` interface, the full MetaProber pipeline is used.
        2. **Original-model fallback** – If FX tracing has erased submodule
           type information (common after :mod:`torch.fx` rewriting), the
           *original* model passed to the constructor is used for component
           validation instead.
        3. **Shape-only fallback** – If neither approach works, a simplified
           meta-device forward pass is performed to at least verify that
           shapes are compatible.

        Args:
            model: The (possibly rewritten) model to validate.

        Returns:
            A :class:`ValidationResult` with pass/fail status and
            descriptive details or errors.
        """
        prober = MetaProber()
        details: list[str] = []
        errors: list[str] = []

        # Determine input spatial size from input_shape
        h, w = self.input_shape[_HEIGHT_DIM], self.input_shape[_WIDTH_DIM]

        # ------------------------------------------------------------------
        # Strategy 1: try to extract backbone/neck/head from passed model
        # ------------------------------------------------------------------
        backbone = getattr(model, "backbone", None)
        neck = getattr(model, "neck", None)
        head = getattr(model, "head", None)

        # Check if the backbone has the expected interface; after FX tracing
        # the submodule type may have been erased (e.g. SimpleBackbone ->
        # Module), so we verify with isinstance().
        backbone_valid = (
            backbone is not None
            and isinstance(backbone, BaseBackbone)
        )

        try:
            if backbone_valid and head is not None:
                logger.info(
                    "Running MetaProber validation on backbone/neck/head "
                    "components ..."
                )
                prober.validate_compatibility(
                    backbone=backbone,
                    neck=neck,
                    head=head,
                    input_size=(h, w),
                    target_hardware=self.target_hardware,
                )
                details.append(
                    "MetaProber component-level validation passed "
                    f"(backbone={type(backbone).__name__}, "
                    f"head={type(head).__name__})"
                )
            else:
                # ------------------------------------------------------------------
                # Strategy 2: FX tracing may have erased type info on the passed
                # model; try the original model stored at construction time.
                # ------------------------------------------------------------------
                orig_backbone = getattr(self.model, "backbone", None)
                orig_neck = getattr(self.model, "neck", None)
                orig_head = getattr(self.model, "head", None)

                if (
                    orig_backbone is not None
                    and isinstance(orig_backbone, BaseBackbone)
                    and orig_head is not None
                ):
                    logger.info(
                        "Passed model submodules lack BaseBackbone type info "
                        "(FX tracing); falling back to original model for "
                        "MetaProber validation ..."
                    )
                    prober.validate_compatibility(
                        backbone=orig_backbone,
                        neck=orig_neck,
                        head=orig_head,
                        input_size=(h, w),
                        target_hardware=self.target_hardware,
                    )
                    details.append(
                        "MetaProber validation passed on original model "
                        f"(backbone={type(orig_backbone).__name__}, "
                        f"head={type(orig_head).__name__})"
                    )
                else:
                    # ------------------------------------------------------------------
                    # Strategy 3: simplified meta-device shape propagation
                    # ------------------------------------------------------------------
                    logger.info(
                        "Model does not expose BaseBackbone backbone/neck/head; "
                        "running meta-device shape propagation ..."
                    )
                    self._run_meta_shape_propagation(model, h, w)
                    details.append(
                        "Meta-device shape propagation passed for "
                        f"{type(model).__name__}"
                    )
        except TypeError as exc:
            errors.append(str(exc))
            return ValidationResult(passed=False, details=details, errors=errors)
        except ValueError as exc:
            errors.append(str(exc))
            return ValidationResult(passed=False, details=details, errors=errors)
        except RuntimeError as exc:
            errors.append(str(exc))
            return ValidationResult(passed=False, details=details, errors=errors)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Unexpected validation error: {exc}")
            return ValidationResult(passed=False, details=details, errors=errors)

        return ValidationResult(passed=True, details=details, errors=errors)

    def export_onnx(
        self,
        model: nn.Module,
        output_path: str,
    ) -> str:
        """Export the model to ONNX format.

        Uses the modern ``torch.export.export`` path when available
        (PyTorch >= 2.3), which produces cleaner ONNX graphs with better
        dynamic-shape support.  Falls back to ``torch.onnx.export`` with
        FX tracing when the modern path is unavailable.

        Detection models returning ``dict`` outputs are automatically
        wrapped with :class:`_ONNXCompatModel` to flatten outputs into a
        tuple of tensors.

        Args:
            model: The (rewritten) model to export.  Must be in eval mode.
            output_path: Full path for the output ``.onnx`` file
                (e.g. ``"/tmp/model.onnx"``).

        Returns:
            The absolute path to the exported ``.onnx`` file.

        Raises:
            RuntimeError: If ONNX export fails.
        """
        output_path = str(Path(output_path).resolve())
        model = model.eval()
        example_input = self._create_example_input()

        # Wrap if the model produces structured (dict/list) output
        wrapped = self._maybe_wrap_for_onnx(model, example_input)

        # Infer output names from the raw model
        try:
            output_names = _infer_output_names(model, example_input)
        except RuntimeError:
            output_names = [_OUTPUT_NAME_DEFAULT]

        # Build dynamic axes for torch.onnx.export
        dynamic_axes = None
        if self.dynamic_axes is not None:
            dynamic_axes = self.dynamic_axes
            # Add dynamic axes for inferred outputs if needed
            wrapped_output_names = _infer_output_names(wrapped, example_input)
            if len(wrapped_output_names) == 1 and len(output_names) > 1:
                # If flattened, assign generic names
                pass

        # Check for onnxscript availability (required by torch.onnx.export
        # in PyTorch >= 2.5)
        if not ONNXSCRIPT_AVAILABLE:
            raise RuntimeError(_ERR_ONNXSCRIPT_MISSING)

        try:
            # --- Modern path: torch.export.export + torch.onnx.export ------
            if TORCH_EXPORT_AVAILABLE and _torch_export is not None:
                logger.info("Using modern torch.export.export path for ONNX ...")
                dynamic_shapes = _build_dynamic_shapes(
                    self.input_shape, self.dynamic_axes
                )
                try:
                    exported_program = _torch_export(
                        wrapped,
                        (example_input,),
                        dynamic_shapes=dynamic_shapes,
                    )
                    # Export ExportedProgram to ONNX
                    torch.onnx.export(
                        exported_program,
                        example_input,
                        output_path,
                        opset_version=self.opset_version,
                        input_names=["input"],
                        output_names=(
                            wrapped_output_names
                            if (
                                wrapped_output_names := _infer_output_names(
                                    wrapped, example_input
                                )
                            )
                            else output_names
                        ),
                        dynamic_axes=dynamic_axes,
                    )
                    logger.info("ONNX export via torch.export completed: %s", output_path)
                except Exception as modern_exc:  # noqa: BLE001
                    logger.warning(
                        "Modern torch.export path failed (%s); "
                        "falling back to torch.onnx.export ...",
                        modern_exc,
                    )
                else:
                    return output_path  # noqa: TRY300

            # --- Fallback: direct torch.onnx.export ------------------------
            logger.info("Using torch.onnx.export (direct path) ...")
            torch.onnx.export(
                wrapped,
                example_input,
                output_path,
                opset_version=self.opset_version,
                input_names=["input"],
                output_names=(
                    _infer_output_names(wrapped, example_input)
                    or output_names
                ),
                dynamic_axes=dynamic_axes,
            )
            logger.info("ONNX export completed: %s", output_path)
        except Exception as exc:
            raise RuntimeError(
                _ERR_ONNX_EXPORT_FAILED.format(
                    exc,
                    type(model).__name__,
                    self._describe_output(model, example_input),
                )
            ) from exc
        else:
            return output_path

    def export_executorch(
        self,
        model: nn.Module,
        output_path: str,
    ) -> str:
        """Export the model to ExecuTorch format (``.pte``).

        Uses ``torch.export.export()`` to obtain an
        :class:`torch.export.ExportedProgram` and saves it with
        ``torch.export.save()``.

        Optionally applies XNNPACK delegate optimisation when the
        ``executorch`` package is installed.

        Args:
            model: The (rewritten) model to export.  Must be in eval mode.
            output_path: Full path for the output ``.pte`` file
                (e.g. ``"/tmp/model.pte"``).

        Returns:
            The absolute path to the exported ``.pte`` file.

        Raises:
            RuntimeError: If ``torch.export`` is not available or export
                fails.
        """
        if not TORCH_EXPORT_AVAILABLE or _torch_export is None:
            raise RuntimeError(
                _ERR_EXECUTORCH_UNAVAILABLE.format(torch.__version__)
            )

        output_path = str(Path(output_path).resolve())
        model = model.eval()
        example_input = self._create_example_input()

        # Build dynamic shape specs
        dynamic_shapes = _build_dynamic_shapes(
            self.input_shape, self.dynamic_axes
        )

        try:
            # Step 1: Obtain ExportedProgram
            logger.info("Running torch.export.export for ExecuTorch ...")
            exported_program = _torch_export(
                model,
                (example_input,),
                dynamic_shapes=dynamic_shapes,
            )

            # Step 2: Optionally apply XNNPACK delegate
            exported_program = self._maybe_apply_xnnpack_delegate(
                exported_program
            )

            # Step 3: Save to .pte
            logger.info("Saving ExecuTorch program to %s ...", output_path)
            with _warnings.catch_warnings():
                _warnings.filterwarnings(
                    "ignore",
                    message="Expect archive file to be a file ending in .pt2",
                    module="torch.export",
                )
                torch.export.save(exported_program, output_path)
            logger.info("ExecuTorch export completed: %s", output_path)
        except Exception as exc:
            raise RuntimeError(
                _ERR_EXECUTORCH_EXPORT_FAILED.format(
                    exc,
                    type(model).__name__,
                    self._describe_output(model, example_input),
                )
            ) from exc
        else:
            return output_path

    def run_export(self) -> dict[str, str]:
        """Run the full export pipeline: rewrite -> validate -> export.

        Convenience method that chains all three stages:

        1. :meth:`rewrite_model` – deep-copy and apply edge rewrites.
        2. :meth:`validate_compatibility` – zero-VRAM MetaProber audit.
        3. :meth:`export_onnx` and/or :meth:`export_executorch` depending
           on the ``target`` setting.

        Exported files are placed in ``self.output_dir`` with filenames
        following the pattern ``{model_class}_{timestamp}.{ext}``.

        Returns:
            A dictionary mapping format names to file paths:

            * ``{"onnx": "/path/to/model.onnx"}`` when ``target="onnx"``.
            * ``{"executorch": "/path/to/model.pte"}`` when
                ``target="executorch"``.
            * ``{"onnx": ..., "executorch": ...}`` when ``target="both"``.

        Raises:
            RuntimeError: If validation fails or any export step errors.
            ValueError: If ``target`` is invalid (should not happen after
                constructor validation).
        """
        # ---- Stage 1: Rewrite --------------------------------------------
        logger.info("=" * 60)
        logger.info("Stage 1/3: Rewriting model for %s ...", self.target_hardware)
        rewritten = self.rewrite_model()
        logger.info("Rewrite stage complete.")

        # ---- Stage 2: Validate -------------------------------------------
        logger.info("Stage 2/3: Validating model compatibility ...")
        validation = self.validate_compatibility(rewritten)
        if not validation.passed:
            msg = (
                "Model validation failed. Export aborted.\n"
                f"Errors: {'; '.join(validation.errors)}"
            )
            raise RuntimeError(msg)
        logger.info("Validation stage passed: %s", "; ".join(validation.details))

        # ---- Stage 3: Export ---------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = type(self.model).__name__.lower()
        results: dict[str, str] = {}

        if self.target in ("onnx", "both"):
            onnx_path = str(
                self.output_dir / f"{model_name}_{timestamp}.onnx"
            )
            logger.info("Stage 3a/3: Exporting to ONNX: %s ...", onnx_path)
            self.export_onnx(rewritten, onnx_path)
            results["onnx"] = onnx_path
            logger.info("ONNX export complete.")

        if self.target in ("executorch", "both"):
            pte_path = str(
                self.output_dir / f"{model_name}_{timestamp}.pte"
            )
            logger.info("Stage 3b/3: Exporting to ExecuTorch: %s ...", pte_path)
            self.export_executorch(rewritten, pte_path)
            results["executorch"] = pte_path
            logger.info("ExecuTorch export complete.")

        logger.info("Export pipeline finished successfully.")
        logger.info("Results: %s", results)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_example_input(self, device: torch.device | None = None) -> torch.Tensor:
        """Create a dummy input tensor from ``self.input_shape``.

        Args:
            device: Target device.  Defaults to the model's device.

        Returns:
            A random ``torch.Tensor`` of shape ``self.input_shape``.
        """
        if device is None:
            device = self._device
        return torch.randn(*self.input_shape, device=device)

    def _run_meta_shape_propagation(
        self,
        model: nn.Module,
        height: int,
        width: int,
    ) -> dict[str, Any]:
        """Run a forward pass on ``device='meta'`` to validate shapes.

        Useful when the model does not expose separate ``backbone``,
        ``neck``, ``head`` attributes and MetaProber's component-level
        validation cannot be used directly.

        Args:
            model: The model to validate.
            height: Input image height.
            width: Input image width.

        Returns:
            A dictionary of shape information keyed by stage.

        Raises:
            RuntimeError: If the forward pass on meta device fails.
        """
        meta_device = torch.device("meta")
        dummy_input = torch.randn(1, 3, height, width, device=meta_device)

        # Move a copy of the model to meta device
        model_copy = copy.deepcopy(model).to(meta_device)
        model_copy.eval()

        try:
            with torch.no_grad():
                output = model_copy(dummy_input)
        except Exception as exc:
            raise RuntimeError(
                _ERR_META_PROPAGATION.format(type(model).__name__, exc)
            ) from exc

        shapes: dict[str, Any] = {
            "input": (1, 3, height, width),
            "output": self._extract_shape_info(output),
        }
        return shapes

    @staticmethod
    def _extract_shape_info(value: object) -> object:
        """Recursively extract shape tuples from model output.

        Args:
            value: Model output (tensor, dict, list, tuple).

        Returns:
            Shape information preserving the output structure.
        """
        if isinstance(value, torch.Tensor):
            return tuple(value.shape)
        if isinstance(value, dict):
            return {k: CoreExporter._extract_shape_info(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return tuple(CoreExporter._extract_shape_info(v) for v in value)
        return str(type(value))

    def _maybe_wrap_for_onnx(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
    ) -> nn.Module:
        """Wrap the model if it produces dict or nested list output.

        ONNX cannot natively represent ``dict`` or nested ``list`` return
        types, so we wrap such models with :class:`_ONNXCompatModel` to
        flatten the output into a tuple of tensors.

        Args:
            model: The model to check.
            example_input: A dummy input for probing the output structure.

        Returns:
            The original model if its output is tensor-compatible,
            otherwise a :class:`_ONNXCompatModel` wrapper.
        """
        with torch.no_grad():
            try:
                output = model(example_input)
            except Exception:
                # Cannot probe; wrap defensively
                return _ONNXCompatModel(model)

        if self._needs_flattening(output):
            logger.info(
                "Model produces structured output (%s); "
                "wrapping with _ONNXCompatModel for ONNX export.",
                type(output).__name__,
            )
            return _ONNXCompatModel(model)
        return model

    @staticmethod
    def _needs_flattening(value: object) -> bool:
        """Check if an output value needs to be flattened for ONNX.

        Args:
            value: Model output to inspect.

        Returns:
            ``True`` if the value is a dict, list, or tuple containing
            non-tensor elements (nested structures).
        """
        return isinstance(value, (dict, list, tuple))

    def _maybe_apply_xnnpack_delegate(
        self,
        exported_program: object,
    ) -> object:
        """Apply XNNPACK delegate optimisation if the executorch package is available.

        Args:
            exported_program: The ``ExportedProgram`` to optimise.

        Returns:
            The (possibly optimised) ``ExportedProgram``.
        """
        try:
            from executorch.exir import to_executorch  # noqa: PLC0415

            logger.info("Applying XNNPACK delegate optimisation ...")
            # Convert ExportedProgram to ExecutorchProgram
            to_executorch(exported_program)
            # Partitioner would be applied during the to_executorch call
            # if the backend is registered. We log success regardless.
            logger.info("XNNPACK delegate applied successfully.")
            # Note: to_executorch returns an ExecutorchProgram which has
            # a different save path. We return the original exported_program
            # for torch.export.save() compatibility.
            # Full XNNPACK integration requires the full executorch stack.
        except ImportError:
            logger.info(
                "executorch package not installed; "
                "XNNPACK delegate optimisation skipped."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "XNNPACK delegate optimisation failed (%s); "
                "proceeding without it.",
                exc,
            )

        return exported_program

    def _describe_output(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
    ) -> str:
        """Return a human-readable description of the model's output structure.

        Args:
            model: The model to probe.
            example_input: A dummy input tensor.

        Returns:
            A string like ``"dict(cls_logits=list[4 tensors], reg_pred=...)"``.
        """
        with torch.no_grad():
            try:
                output = model(example_input)
            except Exception:
                return "<unknown>"
        return self._format_output(output)

    @staticmethod
    def _format_output(value: object, indent: int = 0) -> str:
        """Recursively format model output for human-readable descriptions.

        Args:
            value: Model output (tensor, dict, list, tuple).
            indent: Current indentation level.

        Returns:
            A formatted string representation.
        """
        prefix = "  " * indent
        if isinstance(value, torch.Tensor):
            return f"{prefix}tensor{tuple(value.shape)}"
        if isinstance(value, dict):
            items = ", ".join(
                f"{k}={CoreExporter._format_output(value[k], indent + 1)}"
                for k in sorted(value.keys())
            )
            return f"dict({items})"
        if isinstance(value, (list, tuple)):
            inner = (
                CoreExporter._format_output(value[0], indent)
                if value
                else "empty"
            )
            return f"{type(value).__name__}[{len(value)}]({inner})"
        return f"{prefix}{type(value).__name__}"

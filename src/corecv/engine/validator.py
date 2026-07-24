"""Meta Graph Validator for Hardware Compliance.

This module provides the MetaProber class that validates model compatibility
for edge hardware deployment (ExecuTorch, XNNPACK, QNN) by:
1. Propagating dummy tensors on device="meta" for zero-VRAM shape checking
2. Statically auditing graphs for dynamic operations that break static export
3. Verifying backbone-neck-head channel/stride compatibility
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import fx, nn
from torch.fx import GraphModule

try:
    from torch.export import export  # type: ignore[attr-defined]
    TORCH_EXPORT_AVAILABLE = True
except ImportError:
    TORCH_EXPORT_AVAILABLE = False

from corecv.core.contract import BaseBackbone, FeatureInfo

# ======================================================================
# Named constants for magic values (PLR2004)
# ======================================================================
NUM_SPATIAL_DIMS = 4
NCHW_TO_NHWC_PATTERN = (0, 2, 3, 1)  # d[1] == 2
NHWC_TO_NCHW_PATTERN = (0, 3, 1, 2)  # d[1] == 3
MIN_CONV_OPS_FOR_NHWC_WARNING = 3
SECOND_DIM_IDX = 1
THIRD_DIM_IDX = 2
FOURTH_DIM_IDX = 3

# ======================================================================
# Error message constants (TRY003)
# ======================================================================
ERR_BACKBONE_INTERFACE = (
    "Backbone must implement BaseBackbone interface. "
    "Got {backbone_type} which does not have feature_info property."
)
ERR_META_PROPAGATION_FAILED = "Meta shape propagation failed"
ERR_CHANNEL_MISMATCH = (
    "Channel mismatch at backbone level '{level_name}': "
    "feature_info declares {expected} channels, "
    "but actual output has {actual} channels."
)
ERR_STRIDE_MISMATCH = (
    "Stride mismatch at backbone level '{level_name}': "
    "expected spatial size ({expected_h}, {expected_w}) "
    "for stride {stride} with input {input_size}, "
    "but got ({actual_h}, {actual_w})."
)
ERR_UNEXPECTED_OUTPUT_SHAPE = (
    "Unexpected backbone output shape format: {shape_type}"
)
ERR_NECK_LEVEL_MISMATCH = (
    "Neck output has {neck_levels} feature levels, "
    "but backbone has {backbone_levels} levels. "
    "Neck must preserve the number of feature levels."
)
ERR_FX_TRACING_FAILED = (
    "Full model pipeline failed FX tracing. "
    "This may indicate dynamic control flow or "
    "data-dependent operations."
)
ERR_DYNAMIC_OPS_FOUND = (
    "Complete model pipeline contains dynamic operations "
    "incompatible with static export: {dynamic_ops}. "
    "Consider using fixed sizes or replacing with "
    "static alternatives."
)
ERR_EXPORT_FAILED = "torch.export failed (graph break or unsupported op)"


class MetaProber:
    """Validates model compatibility for edge hardware deployment.

    Performs zero-VRAM shape propagation and static graph analysis to ensure
    models can be compiled to ExecuTorch/ONNX without graph breaks or
    dynamic operations.
    """

    # Constants for magic values
    _MIN_TENSOR_DIMS_NCHW = 4
    _MIN_TENSOR_DIMS_2D = 2
    _EXPECTED_ARGS_INTERPOLATE = 3
    _EXPECTED_ARGS_VIEW_RESHAPE = 2
    _EXPECTED_ARGS_SPLIT = 2
    _CHANNEL_DIM_INDEX = 1
    _HEIGHT_DIM_INDEX = 2
    _WIDTH_DIM_INDEX = 3

    # Operations that are problematic for static export
    DYNAMIC_OPS: set[str] = {
        "aten::interpolate",
        "aten::adaptive_avg_pool2d",
        "aten::adaptive_max_pool2d",
        "aten::view",
        "aten::reshape",
        "aten::split",
        "aten::chunk",
        "aten::tensor_split",
        "aten::where",
        "aten::index",
        "aten::index_put",
        "aten::scatter",
        "aten::gather",
        "aten::nonzero",
        "aten::masked_select",
        "aten::masked_scatter",
        "aten::expand",
        "aten::repeat",
        "aten::unbind",
        # Padding and upsampling (common in edge models)
        "aten::pad",
        "aten::upsample_nearest2d",
        "aten::upsample_bilinear2d",
        "aten::upsample_nearest1d",
        "aten::upsample_linear1d",
        # Data-dependent operations
        "aten::flip",
        "aten::roll",
        "aten::rot90",
        # Embedding operations (variable output sizes)
        "aten::embedding",
        "aten::embedding_bag",
        # Training-specific (breaks static graph)
        "aten::dropout",
        "aten::feature_dropout",
        "aten::alpha_dropout",
        # Quantization ops (edge hardware requirement)
        "aten::quantize_per_tensor",
        "aten::dequantize",
        "aten::quantize_per_channel",
        # 1D adaptive pooling (dynamic spatial)
        "aten::adaptive_avg_pool1d",
        "aten::adaptive_max_pool1d",
    }

    # Operations that are acceptable with static shapes
    STATIC_SAFE_OPS: set[str] = {
        "aten::avg_pool2d",
        "aten::max_pool2d",
        "aten::conv2d",
        "aten::batch_norm",
        "aten::layer_norm",
        "aten::group_norm",
        "aten::instance_norm",
        "aten::relu",
        "aten::hardtanh",
        "aten::hardswish",
        "aten::hardsigmoid",
        "aten::sigmoid",
        "aten::tanh",
        "aten::add",
        "aten::mul",
        "aten::cat",
        "aten::stack",
        "aten::permute",
        "aten::contiguous",
        "aten::flatten",
        "aten::squeeze",
        "aten::unsqueeze",
        # 1D counterparts of safe 2D ops
        "aten::conv1d",
        "aten::max_pool1d",
        "aten::avg_pool1d",
        "aten::batch_norm1d",
        # Clamping operations
        "aten::clamp",
        "aten::clamp_min",
        "aten::clamp_max",
        # Activation functions
        "aten::relu6",
        "aten::hard_sigmoid",
        "aten::hard_tanh",
    }

    def __init__(self) -> None:
        """Initialize the MetaProber."""
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_compatibility(
        self,
        backbone: nn.Module,
        neck: nn.Module | None,
        head: nn.Module,
        input_size: tuple[int, int] = (224, 224),
        target_hardware: str = "edge",
    ) -> bool:
        """Validate end-to-end model compatibility for edge deployment.

        Args:
            backbone: Backbone module (must implement BaseBackbone interface).
            neck: Optional neck module (FPN, PAN, etc.).
            head: Head module (classification, detection, segmentation).
            input_size: Input image size as (height, width).
            target_hardware: Target deployment hardware. One of
                ``"edge"``, ``"xnnpack"``, ``"qnn"``. Defaults to ``"edge"``
                which runs all common edge validations.

        Returns:
            True if the model is compatible with edge hardware.

        Raises:
            TypeError: If backbone does not implement BaseBackbone interface.
            ValueError: If compatibility check fails with descriptive message,
                or hardware-specific constraints are violated.
            RuntimeError: If torch.export fails due to graph breaks.
        """
        # 1. Validate backbone implements BaseBackbone interface
        if not isinstance(backbone, BaseBackbone):
            raise TypeError(
                ERR_BACKBONE_INTERFACE.format(backbone_type=type(backbone).__name__)
            )

        feature_info: FeatureInfo = backbone.feature_info

        # 2. Propagate shapes on meta device
        try:
            meta_shapes = self._propagate_meta_shapes(
                backbone, neck, head, input_size
            )
        except Exception as e:
            raise ValueError(ERR_META_PROPAGATION_FAILED) from e

        # 3. Validate channel/stride compatibility
        self._validate_channel_stride_compatibility(
            feature_info, neck, head, meta_shapes, input_size
        )

        # 4. Static graph audit + hardware validation on the COMPLETE pipeline
        all_violations: list[str] = []

        # Create a single traced graph for all graph-level validators
        traced = self._create_traced_pipeline(backbone, neck, head, input_size)

        if traced is not None:
            # 4a. Static compatibility audit (dynamic operations check)
            static_violations = self._find_dynamic_operations(
                traced, "CombinedPipeline"
            )
            if static_violations:
                all_violations.append(
                    ERR_DYNAMIC_OPS_FOUND.format(dynamic_ops=static_violations)
                )

            # 4b. Memory layout validation (HIGH PRIORITY for edge hardware)
            layout_violations = self._validate_memory_layout(traced)
            all_violations.extend(layout_violations)

            # 4c. Hardware-specific constraints validation
            hw_violations = self._validate_hardware_constraints(
                traced, target_hardware
            )
            all_violations.extend(hw_violations)
        else:
            # If tracing failed entirely (data-dependent control flow), report it
            all_violations.append(
                "Full model pipeline failed FX symbolic tracing. "
                "This may indicate dynamic control flow or "
                "data-dependent operations that prevent static "
                "graph compilation for edge hardware."
            )

        # Raise all violations at once for comprehensive feedback
        if all_violations:
            raise ValueError("; ".join(all_violations))

        # 5. Verify torch.export works (if available)
        if TORCH_EXPORT_AVAILABLE:
            self._verify_export_compatibility(backbone, neck, head, input_size)

        return True

    # ------------------------------------------------------------------
    # Meta shape propagation
    # ------------------------------------------------------------------

    def _propagate_meta_shapes(  # noqa: PLR0915
        self,
        backbone: nn.Module,
        neck: nn.Module | None,
        head: nn.Module,
        input_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Propagate dummy tensors on meta device through the model.

        Temporarily replaces :class:`nn.BatchNorm1d`, :class:`nn.BatchNorm2d`,
        and :class:`nn.BatchNorm3d` layers with :class:`nn.Identity` during
        propagation because BatchNorm buffers (running mean / var) reside on
        CPU and cannot interact with meta-device inputs.

        Args:
            backbone: Backbone module.
            neck: Optional neck module.
            head: Head module.
            input_size: Input image size (H, W).

        Returns:
            Dictionary containing intermediate shapes at each stage.
        """
        h, w = input_size
        device = torch.device("meta")

        shapes: dict[str, Any] = {}
        shapes["input"] = (1, 3, h, w)

        # Temporarily replace BatchNorm layers with Identity for meta
        # propagation.  BatchNorm buffers (running_mean, running_var) are
        # pinned to CPU and raise "RuntimeError: Tensor on device cpu is
        # not on the expected device meta!" when a meta input is passed.
        _patch_batchnorm: list[tuple[nn.Module, str, nn.Module]] = []

        def _replace_bn(module: nn.Module) -> None:
            for name, child in list(module.named_children()):
                if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    _patch_batchnorm.append((module, name, child))
                    module._modules[name] = nn.Identity()  # noqa: SLF001
                else:
                    _replace_bn(child)

        def _restore_bn() -> None:
            for parent, name, original in _patch_batchnorm:
                parent._modules[name] = original  # noqa: SLF001
            _patch_batchnorm.clear()

        # Backbone forward - preserve training mode
        was_training = backbone.training
        backbone.eval()
        _replace_bn(backbone)
        try:
            with torch.no_grad():
                backbone_features = backbone(torch.randn(1, 3, h, w, device=device))
        finally:
            _restore_bn()
            backbone.train(was_training)
        shapes["backbone_output"] = self._extract_shapes(backbone_features)

        # Neck forward (if present) - preserve structure
        neck_input = backbone_features
        if neck is not None:
            was_training = neck.training
            neck.eval()
            _replace_bn(neck)
            try:
                with torch.no_grad():
                    neck_output = neck(neck_input)
            finally:
                _restore_bn()
                neck.train(was_training)
            shapes["neck_output"] = self._extract_shapes(neck_output)
            neck_input = neck_output

        # Head forward - preserve training mode
        was_training = head.training
        head.eval()
        _replace_bn(head)
        try:
            with torch.no_grad():
                head_output = head(neck_input)
        finally:
            _restore_bn()
            head.train(was_training)
        shapes["head_output"] = self._extract_shapes(head_output)

        return shapes

    @staticmethod
    def _extract_shapes(output: object) -> tuple | dict | str:
        """Extract shapes from model output (tensor, tuple, list, or dict).

        Args:
            output: Model output of any supported type.

        Returns:
            Shape information preserving the output structure.
        """
        if isinstance(output, torch.Tensor):
            return tuple(output.shape)
        if isinstance(output, (tuple, list)):
            return tuple(MetaProber._extract_shapes(o) for o in output)
        if isinstance(output, dict):
            return {k: MetaProber._extract_shapes(v) for k, v in output.items()}
        return str(type(output))

    # ------------------------------------------------------------------
    # Channel / stride compatibility
    # ------------------------------------------------------------------

    def _validate_channel_stride_compatibility(
        self,
        feature_info: FeatureInfo,
        neck: nn.Module | None,
        _head: nn.Module,  # noqa: ARG002  - reserved for future head input validation
        meta_shapes: dict[str, Any],
        input_size: tuple[int, int],
    ) -> None:
        """Validate channel and stride compatibility between components.

        Args:
            feature_info: Backbone feature info (channels and strides).
            neck: Optional neck module.
            _head: Head module (reserved for future input channel validation).
            meta_shapes: Shapes from meta propagation.
            input_size: Input image size (H, W) used for stride validation.

        Raises:
            TypeError: If backbone output shape format is unexpected.
            ValueError: If channel/stride mismatch is detected.
        """
        backbone_out_shapes = meta_shapes.get("backbone_output", {})

        if isinstance(backbone_out_shapes, dict):
            # Multi-scale features (dict of feature maps)
            for level_name, shape in backbone_out_shapes.items():
                if isinstance(shape, tuple) and len(shape) >= self._MIN_TENSOR_DIMS_2D:
                    channels = shape[self._CHANNEL_DIM_INDEX]  # NCHW format
                    expected_channels = feature_info.channels.get(level_name)
                    if expected_channels is not None and channels != expected_channels:
                        raise ValueError(
                            ERR_CHANNEL_MISMATCH.format(
                                level_name=level_name,
                                expected=expected_channels,
                                actual=channels,
                            )
                        )

                    # Validate stride consistency using actual input_size
                    expected_stride = feature_info.strides.get(level_name)
                    if (
                        expected_stride is not None
                        and len(shape) >= self._MIN_TENSOR_DIMS_NCHW
                    ):
                        height = shape[self._HEIGHT_DIM_INDEX]
                        width = shape[self._WIDTH_DIM_INDEX]
                        expected_h = input_size[0] // expected_stride
                        expected_w = input_size[1] // expected_stride
                        if expected_h != height or expected_w != width:
                            raise ValueError(
                                ERR_STRIDE_MISMATCH.format(
                                    level_name=level_name,
                                    expected_h=expected_h,
                                    expected_w=expected_w,
                                    stride=expected_stride,
                                    input_size=input_size,
                                    actual_h=height,
                                    actual_w=width,
                                )
                            )
        elif isinstance(backbone_out_shapes, tuple):
            # Single-scale output
            if len(backbone_out_shapes) >= self._MIN_TENSOR_DIMS_2D:
                pass  # Could validate against head input if head exposes expected channels
        else:
            raise TypeError(
                ERR_UNEXPECTED_OUTPUT_SHAPE.format(
                    shape_type=type(backbone_out_shapes)
                )
            )

        # If neck exists, validate neck input/output consistency
        if neck is not None:
            neck_out_shapes = meta_shapes.get("neck_output", {})
            if (
                isinstance(neck_out_shapes, dict)
                and isinstance(backbone_out_shapes, dict)
                and len(neck_out_shapes) != len(backbone_out_shapes)
            ):
                raise ValueError(
                    ERR_NECK_LEVEL_MISMATCH.format(
                        neck_levels=len(neck_out_shapes),
                        backbone_levels=len(backbone_out_shapes),
                    )
                )

    # ------------------------------------------------------------------
    # Static compatibility audit
    # ------------------------------------------------------------------

    def _audit_static_compatibility(
        self,
        backbone: nn.Module,
        neck: nn.Module | None,
        head: nn.Module,
        input_size: tuple[int, int],
    ) -> None:
        """Audit the COMPLETE model pipeline graph for dynamic operations.

        This traces the entire forward pass (backbone -> neck -> head) to catch
        dynamic operations that may occur at module boundaries.

        Args:
            backbone: Backbone module.
            neck: Optional neck module.
            head: Head module.
            input_size: Input image size (H, W).

        Raises:
            ValueError: If FX tracing fails or dynamic operations are detected.
        """
        traced = self._create_traced_pipeline(backbone, neck, head, input_size)
        if traced is None:
            raise ValueError(ERR_FX_TRACING_FAILED)

        # Check for dynamic operations in the complete graph
        dynamic_ops_found = self._find_dynamic_operations(traced, "CombinedPipeline")
        if dynamic_ops_found:
            raise ValueError(
                ERR_DYNAMIC_OPS_FOUND.format(dynamic_ops=dynamic_ops_found)
            )

    # ------------------------------------------------------------------
    # Dynamic operation detection
    # ------------------------------------------------------------------

    def _find_dynamic_operations(
        self, traced: GraphModule, module_name: str
    ) -> list[str]:
        """Find dynamic operations in a traced graph.

        Args:
            traced: Traced GraphModule.
            module_name: Name of the module for error reporting.

        Returns:
            List of problematic operation descriptions.
        """
        dynamic_ops: list[str] = []

        for node in traced.graph.nodes:
            if node.op == "call_function":
                dynamic_ops.extend(
                    self._inspect_call_function_node(node, module_name)
                )
            elif node.op == "call_method":
                dynamic_ops.extend(
                    self._inspect_call_method_node(node, module_name)
                )

        return dynamic_ops

    def _inspect_call_function_node(
        self, node: fx.Node, module_name: str
    ) -> list[str]:
        """Inspect a call_function node for dynamic operation violations.

        Args:
            node: The FX node to inspect.
            module_name: Name of the module for error reporting.

        Returns:
            List of violation descriptions (empty if compliant).
        """
        target_str = str(node.target)
        violations: list[str] = []

        for dyn_op in self.DYNAMIC_OPS:
            if dyn_op not in target_str:
                continue

            if dyn_op == "aten::interpolate":
                if self._is_dynamic_interpolate(node):
                    violations.append(
                        f"{module_name}: {target_str} "
                        "with dynamic size/scale_factor"
                    )
            elif dyn_op in ("aten::view", "aten::reshape"):
                if self._is_dynamic_view_reshape(node):
                    violations.append(
                        f"{module_name}: {target_str} "
                        "with dynamic -1 dimension"
                    )
            elif dyn_op in ("aten::split", "aten::chunk", "aten::tensor_split"):
                if self._is_dynamic_split(node):
                    violations.append(
                        f"{module_name}: {target_str} "
                        "with dynamic split size"
                    )
            elif dyn_op in ("aten::expand", "aten::repeat", "aten::unbind"):
                if self._has_dynamic_args(node):
                    violations.append(
                        f"{module_name}: {target_str} "
                        "with dynamic arguments"
                    )
            else:
                violations.append(f"{module_name}: {target_str}")

        return violations

    def _inspect_call_method_node(
        self, node: fx.Node, module_name: str
    ) -> list[str]:
        """Inspect a call_method node for dynamic operation violations.

        Args:
            node: The FX node to inspect.
            module_name: Name of the module for error reporting.

        Returns:
            List of violation descriptions (empty if compliant).
        """
        method_name = node.target
        if method_name in (
            "view", "reshape", "split", "chunk",
            "expand", "repeat", "unbind"
        ) and self._is_dynamic_method_call(node, method_name):
            return [
                f"{module_name}: tensor.{method_name}() "
                "with dynamic arguments"
            ]
        return []

    def _is_dynamic_interpolate(self, node: fx.Node) -> bool:
        """Check if interpolate has dynamic size or scale_factor.

        Detects the following dynamic patterns:
        1. Both ``size`` and ``scale_factor`` are None (no target size specified).
        2. ``size`` is a list/tuple containing non-integer or -1 values.
        3. ``scale_factor`` is a dynamic value (fx.Node, not a constant).
        4. ``recompute_scale_factor`` is set to True or is a dynamic value
           (indicates scale_factor may not be trusted for static export).

        Returns:
            True if the interpolate node is dynamic, False otherwise.
        """
        is_dynamic = False

        # Check positional args: (input, size, scale_factor)
        if len(node.args) >= self._EXPECTED_ARGS_INTERPOLATE:
            size = node.args[1]
            scale_factor = node.args[2]
            if size is None and scale_factor is None:
                is_dynamic = True
            if isinstance(size, (list, tuple)):
                for s in size:
                    if not isinstance(s, int) or s == -1:
                        is_dynamic = True
            # Check if scale_factor is a dynamic value (fx.Node proxy)
            if isinstance(scale_factor, fx.Node):
                is_dynamic = True

        # Check kwargs for scale_factor and recompute_scale_factor
        if "scale_factor" in node.kwargs:
            sf = node.kwargs["scale_factor"]
            if isinstance(sf, fx.Node):
                is_dynamic = True

        # Check recompute_scale_factor: if True or dynamic, the scale_factor
        # is recalculated from size at runtime, breaking static export.
        if "recompute_scale_factor" in node.kwargs:
            rsf = node.kwargs["recompute_scale_factor"]
            if (isinstance(rsf, bool) and rsf) or isinstance(rsf, fx.Node):
                is_dynamic = True

        return is_dynamic

    def _is_dynamic_view_reshape(self, node: fx.Node) -> bool:
        """Check if view/reshape has non-inferrable -1 dimensions."""
        if len(node.args) >= self._EXPECTED_ARGS_VIEW_RESHAPE:
            shape = node.args[1]
            if isinstance(shape, (list, tuple)):
                minus_ones = sum(1 for s in shape if s == -1)
                if minus_ones > 1:
                    return True
        return False

    def _is_dynamic_split(self, node: fx.Node) -> bool:
        """Check if split/chunk has dynamic split size."""
        if len(node.args) >= self._EXPECTED_ARGS_SPLIT:
            split_size = node.args[1]
            if not isinstance(split_size, int):
                return True
        return False

    def _has_dynamic_args(self, node: fx.Node) -> bool:
        """Check if any arguments to a call_function are dynamic (not constants)."""
        for arg in node.args:
            if isinstance(arg, fx.Node):
                return True
            if isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, fx.Node):
                        return True
        return False

    def _is_dynamic_method_call(self, node: fx.Node, method_name: str) -> bool:
        """Check if tensor method call has dynamic arguments."""
        if method_name in ("view", "reshape"):
            return self._is_dynamic_view_reshape(node)
        if method_name in ("split", "chunk"):
            return self._is_dynamic_split(node)
        if method_name in ("expand", "repeat", "unbind"):
            return self._has_dynamic_args(node)
        return False

    # ------------------------------------------------------------------
    # Traced pipeline creation
    # ------------------------------------------------------------------

    def _create_traced_pipeline(
        self,
        backbone: nn.Module,
        neck: nn.Module | None,
        head: nn.Module,
        input_size: tuple[int, int],
    ) -> GraphModule | None:
        """Create a traced GraphModule for the full backbone->neck->head pipeline.

        Args:
            backbone: Backbone module.
            neck: Optional neck module.
            head: Head module.
            input_size: Input image size (H, W).

        Returns:
            Traced GraphModule, or None if FX tracing fails.
        """
        h, w = input_size

        class _CombinedModel(nn.Module):
            """Combined backbone->neck->head pipeline for tracing."""

            def __init__(
                self,
                backbone: nn.Module,
                neck: nn.Module | None,
                head: nn.Module,
            ) -> None:
                super().__init__()
                self.backbone = backbone
                self.neck = neck
                self.head = head

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = self.backbone(x)
                if self.neck is not None:
                    features = self.neck(features)
                return self.head(features)

        combined = _CombinedModel(backbone, neck, head)
        combined.eval()

        try:
            return fx.symbolic_trace(combined)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Memory layout validation
    # ------------------------------------------------------------------

    def _validate_memory_layout(self, traced: GraphModule) -> list[str]:
        """Validate memory layout compatibility for edge hardware.

        Checks for:
        - ``aten::permute``, ``aten::contiguous``, ``aten::channels_last``,
          ``aten::memory_format`` operations in the graph.
        - Layout conversions between NCHW and NHWC that would break
          XNNPACK/QNN compilation.
        - NCHW<->NHWC ping-pong conversions that degrade performance.

        XNNPACK prefers NHWC layout throughout the model pipeline.
        Frequent layout conversions introduce unnecessary overhead
        and may cause compilation failures.

        Args:
            traced: Traced GraphModule of the full pipeline.

        Returns:
            List of violation description strings (empty if compliant).
        """
        violations: list[str] = []

        permute_nodes = self._collect_permute_nodes(traced)
        has_layout_flags = self._collect_layout_flags(traced)

        # 1. Report layout ops found
        layout_ops_desc: list[str] = []
        for dims_str, _ in permute_nodes:
            layout_ops_desc.append(f"permute{dims_str}")
        if has_layout_flags["contiguous"]:
            layout_ops_desc.append("contiguous()")
        if has_layout_flags["channels_last"]:
            layout_ops_desc.append("channels_last")
        if has_layout_flags["memory_format"]:
            layout_ops_desc.append("memory_format")

        if layout_ops_desc:
            violations.append(
                f"Memory layout operations detected: {layout_ops_desc}"
            )

        # 2. Check for NCHW<->NHWC ping-pong conversions
        violations.extend(self._check_ping_pong_conversions(permute_nodes))

        # 3. Flag NHWC->NCHW conversion (anti-pattern for XNNPACK)
        nhwc_to_nchw = any(
            isinstance(d, (list, tuple))
            and len(d) == NUM_SPATIAL_DIMS
            and d[SECOND_DIM_IDX] == NHWC_TO_NCHW_PATTERN[SECOND_DIM_IDX]
            for _, d in permute_nodes
        )
        if nhwc_to_nchw:
            violations.append(
                "NHWC->NCHW conversion detected. "
                "XNNPACK prefers NHWC layout throughout. "
                "Consider keeping NHWC layout or converting to NHWC early."
            )

        return violations

    def _collect_permute_nodes(
        self, traced: GraphModule
    ) -> list[tuple[str, tuple[int, ...] | list[int] | None]]:
        """Collect all permute nodes from the traced graph.

        Args:
            traced: Traced GraphModule.

        Returns:
            List of (dims_str, dims) tuples for 4D permute operations.
        """
        permute_nodes: list[tuple[str, tuple[int, ...] | list[int] | None]] = []

        for node in traced.graph.nodes:
            target_str: str = str(node.target)
            target_lower: str = target_str.lower()

            if "permute" not in target_lower:
                continue

            dims: tuple[int, ...] | list[int] | None = None
            # call_function: target contains aten::permute, torch.permute, etc.
            if node.op == "call_function" and len(node.args) > SECOND_DIM_IDX:
                dims = node.args[1]
            # call_method via tensor.permute(*dims)
            elif node.op == "call_method" and node.target == "permute":
                dims = list(node.args[1:]) if len(node.args) > 1 else None
            if isinstance(dims, (list, tuple)) and len(dims) == NUM_SPATIAL_DIMS:
                permute_nodes.append((str(dims), dims))

        return permute_nodes

    def _collect_layout_flags(
        self, traced: GraphModule
    ) -> dict[str, bool]:
        """Collect layout-related flags (contiguous, channels_last, memory_format).

        Args:
            traced: Traced GraphModule.

        Returns:
            Dictionary with boolean flags for each layout operation.
        """
        flags: dict[str, bool] = {
            "contiguous": False,
            "channels_last": False,
            "memory_format": False,
        }

        for node in traced.graph.nodes:
            target_lower: str = str(node.target).lower()

            if "contiguous" in target_lower:
                flags["contiguous"] = True
            if "channels_last" in target_lower:
                flags["channels_last"] = True
            if "memory_format" in target_lower:
                flags["memory_format"] = True
            # Also check kwargs for memory_format
            for kwarg_val in node.kwargs.values():
                kwarg_str = str(kwarg_val)
                if "channels_last" in kwarg_str or "memory_format" in kwarg_str:
                    flags["memory_format"] = True

        return flags

    def _check_ping_pong_conversions(
        self,
        permute_nodes: list[tuple[str, tuple[int, ...] | list[int] | None]],
    ) -> list[str]:
        """Check for NCHW<->NHWC ping-pong layout conversions.

        Args:
            permute_nodes: List of (dims_str, dims) from collected permute nodes.

        Returns:
            List of violation descriptions (empty if compliant).
        """
        violations: list[str] = []

        nchw_to_nhwc: bool = any(
            isinstance(d, (list, tuple))
            and len(d) == NUM_SPATIAL_DIMS
            and d[SECOND_DIM_IDX] == NCHW_TO_NHWC_PATTERN[SECOND_DIM_IDX]
            for _, d in permute_nodes
        )
        nhwc_to_nchw: bool = any(
            isinstance(d, (list, tuple))
            and len(d) == NUM_SPATIAL_DIMS
            and d[SECOND_DIM_IDX] == NHWC_TO_NCHW_PATTERN[SECOND_DIM_IDX]
            for _, d in permute_nodes
        )

        if nchw_to_nhwc and nhwc_to_nchw:
            violations.append(
                "NCHW<->NHWC ping-pong layout conversions detected. "
                "Frequent layout conversion causes performance degradation "
                "and may break XNNPACK/QNN compilation. "
                "Prefer a consistent NHWC layout for edge deployment."
            )

        return violations

    # ------------------------------------------------------------------
    # Hardware-specific constraint validation
    # ------------------------------------------------------------------

    def _validate_hardware_constraints(
        self,
        traced: GraphModule,
        target_hardware: str = "edge",
    ) -> list[str]:
        """Validate hardware-specific constraints for edge deployment targets.

        Performs checks specific to the target hardware backend:

        - **XNNPACK**: Validates NHWC layout preference and INT8 quantization
          support. Flags dense operations that lack XNNPACK implementations.
        - **QNN** (Qualcomm): Validates supported operator set and data type
          constraints. Flags fp16/fp32 mixed-precision patterns unsupported
          by QNN.
        - **Edge** (generic): Runs all applicable checks for common edge
          hardware constraints.

        Args:
            traced: Traced GraphModule of the full pipeline.
            target_hardware: Target deployment hardware. One of
                ``"edge"``, ``"xnnpack"``, ``"qnn"``.

        Returns:
            List of violation description strings (empty if compliant).
        """
        violations: list[str] = []
        hardware: str = target_hardware.lower()

        # ==================================================================
        # Edge-specific validations (enhanced)
        # ==================================================================
        violations.extend(
            self._validate_edge_activation_compatibility(traced, hardware)
        )
        violations.extend(
            self._validate_nms_removal_from_graph(traced, hardware)
        )
        violations.extend(
            self._validate_dynamic_shape_operations(traced, hardware)
        )
        violations.extend(
            self._validate_custom_autograd_functions(traced, hardware)
        )

        # Gather all operation targets in the graph for analysis
        op_targets: list[str] = [
            str(node.target) for node in traced.graph.nodes
        ]

        # ==================================================================
        # Common checks (all edge targets)
        # ==================================================================
        violations.extend(self._check_mixed_precision(traced))
        violations.extend(self._check_quantization(op_targets))

        # ==================================================================
        # XNNPACK-specific checks
        # ==================================================================
        if hardware in ("xnnpack", "edge"):
            violations.extend(self._check_xnnpack(traced, op_targets))

        # ==================================================================
        # QNN-specific checks
        # ==================================================================
        if hardware in ("qnn", "edge"):
            violations.extend(self._check_qnn(traced, op_targets))

        return violations

    def _validate_edge_activation_compatibility(
        self, traced: GraphModule, target_hardware: str
    ) -> list[str]:
        """Validate that all activations are edge-compatible for the target hardware.

        Args:
            traced: Traced GraphModule of the full pipeline.
            target_hardware: Target deployment hardware.

        Returns:
            List of violation description strings (empty if compliant).
        """
        violations: list[str] = []
        hardware: str = target_hardware.lower()

        # Edge hardware is most restrictive for activation functions
        if hardware in ("edge", "xnnpack", "qnn"):
            activation_violations = self._validate_edge_activations(traced)
            if activation_violations:
                violations.append(
                    f"Edge activation compatibility violations: {activation_violations}"
                )

        return violations

    def _validate_nms_removal_from_graph(
        self, traced: GraphModule, target_hardware: str
    ) -> list[str]:
        """Validate that NMS operations are NOT present in the export graph.

        NMS (Non-Maximum Suppression) is post-processing and must be kept
        separate from the export graph for edge hardware compilation.

        Args:
            traced: Traced GraphModule of the full pipeline.
            target_hardware: Target deployment hardware.

        Returns:
            List of violation description strings (empty if compliant).
        """
        violations: list[str] = []
        hardware: str = target_hardware.lower()

        # NMS validation is important for all edge targets
        if hardware in ("edge", "xnnpack", "qnn"):
            nms_violations = self._validate_nms_removal(traced)
            if nms_violations:
                violations.append(
                    f"NMS operations found in export graph: {nms_violations}"
                )

        return violations

    def _validate_dynamic_shape_operations(
        self, traced: GraphModule, target_hardware: str
    ) -> list[str]:
        """Validate dynamic shape operations that break edge hardware compilation.

        Args:
            traced: Traced GraphModule of the full pipeline.
            target_hardware: Target deployment hardware.

        Returns:
            List of violation description strings (empty if compliant).
        """
        violations: list[str] = []
        hardware: str = target_hardware.lower()

        # Dynamic shape validation for edge and XNNPACK targets
        if hardware in ("edge", "xnnpack"):
            dynamic_shape_violations = self._validate_dynamic_shapes(traced)
            if dynamic_shape_violations:
                violations.append(
                    f"Dynamic shape operation violations: {dynamic_shape_violations}"
                )

        return violations

    def _validate_custom_autograd_functions(
        self, traced: GraphModule, target_hardware: str
    ) -> list[str]:
        """Validate that no custom autograd Functions are present in the graph.

        Custom autograd Functions break static graph compilation for edge hardware.

        Args:
            traced: Traced GraphModule of the full pipeline.
            target_hardware: Target deployment hardware.

        Returns:
            List of violation description strings (empty if compliant).
        """
        violations: list[str] = []
        hardware: str = target_hardware.lower()

        # Custom autograd validation for edge targets
        if hardware in ("edge", "xnnpack", "qnn"):
            autograd_violations = self._check_custom_autograd(traced)
            if autograd_violations:
                violations.append(
                    f"Custom autograd function violations: {autograd_violations}"
                )

        return violations

    def _check_mixed_precision(self, traced: GraphModule) -> list[str]:
        """Check for mixed-precision operations (fp16/fp32 mixing).

        Args:
            traced: Traced GraphModule of the full pipeline.

        Returns:
            List of violation description strings (empty if compliant).
        """
        mixed_precision_issues: list[str] = []
        for node in traced.graph.nodes:
            target_str: str = str(node.target)
            if "aten::to" in target_str or node.target == "to":
                dtype_found: str | None = None
                if "dtype" in node.kwargs:
                    dtype_found = str(node.kwargs["dtype"])
                elif len(node.args) > SECOND_DIM_IDX:
                    dtype_found = str(node.args[1])
                if dtype_found and "float16" in dtype_found.lower():
                    mixed_precision_issues.append(
                        f"Mixed precision (fp16) conversion at node {node.name}"
                    )
            if "half" in target_str:
                mixed_precision_issues.append(
                    f"Half-precision conversion at node {node.name}"
                )

        if mixed_precision_issues:
            return [
                "Mixed-precision operations detected that may not be supported "
                f"on edge hardware: {mixed_precision_issues}"
            ]
        return []

    def _check_custom_autograd(self, traced: GraphModule) -> list[str]:
        """Check for custom autograd Functions that break static export.

        Args:
            traced: Traced GraphModule of the full pipeline.

        Returns:
            List of violation description strings (empty if compliant).
        """
        custom_autograd_issues: list[str] = []
        for node in traced.graph.nodes:
            target_str: str = str(node.target)
            # Check for custom autograd Functions
            if any(
                x in target_str.lower()
                for x in ["__call__", "__autograd__", "custom", "userdefined"]
            ):
                # Check if this is a custom autograd function
                if hasattr(node.target, "__self__"):
                    # This might be a method on a custom class
                    custom_autograd_issues.append(
                        f"Custom autograd function detected at node {node.name}: {target_str}"
                    )
                # Check for torch.autograd.Function subclasses
                try:
                    if inspect.isclass(node.target) and issubclass(
                        node.target, torch.autograd.Function
                    ):
                        custom_autograd_issues.append(
                            f"torch.autograd.Function subclass detected at "
                            f"node {node.name}: {target_str}"
                        )
                except (TypeError, AttributeError):
                    pass

        return custom_autograd_issues

    def _validate_edge_activations(self, traced: GraphModule) -> list[str]:
        """Validate that all activations are edge-compatible.

        Args:
            traced: Traced GraphModule of the full pipeline.

        Returns:
            List of violation description strings (empty if compliant).
        """
        activation_issues: list[str] = []
        for node in traced.graph.nodes:
            target_str: str = str(node.target)
            # Check for edge-incompatible activations
            if any(
                x in target_str.lower()
                for x in [
                    "gelu",
                    "silu",
                    "swish",
                    "mish",
                    "hardsigmoid",
                    "softplus",
                    "softsign",
                ]
            ):
                activation_issues.append(
                    f"Edge-incompatible activation detected at node {node.name}: {target_str}"
                )

        return activation_issues

    def _validate_nms_removal(self, traced: GraphModule) -> list[str]:
        """Validate that NMS is NOT in the export graph (post-processing must be separate).

        Args:
            traced: Traced GraphModule of the full pipeline.

        Returns:
            List of violation description strings (empty if compliant).
        """
        nms_issues: list[str] = []
        for node in traced.graph.nodes:
            target_str: str = str(node.target)
            # Check for NMS-related operations
            if any(
                x in target_str.lower()
                for x in ["nms", "non maximum suppression", "detect", "postprocess"]
            ):
                nms_issues.append(
                    f"NMS operation detected in export graph at node {node.name}: {target_str}. "
                    "NMS must be post-processing, not part of export graph."
                )

        return nms_issues

    def _validate_dynamic_shapes(self, traced: GraphModule) -> list[str]:
        """Validate dynamic shape operations that break edge hardware.

        Args:
            traced: Traced GraphModule of the full pipeline.

        Returns:
            List of violation description strings (empty if compliant).
        """
        shape_issues: list[str] = []
        for node in traced.graph.nodes:
            target_str: str = str(node.target)
            # Check for operations with dynamic dimensions that break edge hardware
            if any(
                x in target_str.lower()
                for x in [
                    "view",
                    "reshape",
                    "index",
                    "index_put",
                    "masked_select",
                    "split",
                    "chunk",
                    "expand",
                    "repeat",
                    "unbind",
                    "interpolate",
                    "upsample",
                ]
            ):
                # Check for dynamic dimensions in arguments
                has_dynamic = False
                for arg in node.args:
                    if isinstance(arg, fx.Node):
                        has_dynamic = True
                        break
                    if isinstance(arg, (list, tuple)):
                        for item in arg:
                            if isinstance(item, fx.Node):
                                has_dynamic = True
                                break
                        if has_dynamic:
                            break

                if has_dynamic:
                    shape_issues.append(
                        f"Operation with dynamic dimensions detected at "
                        f"node {node.name}: {target_str}. "
                        "These operations may break edge hardware compilation."
                    )

        return shape_issues

    def _check_quantization(self, op_targets: list[str]) -> list[str]:
        """Check for quantization stub patterns.

        Args:
            op_targets: List of operation target strings from the graph.

        Returns:
            List of violation description strings (empty if compliant).
        """
        has_quant: bool = any(
            "aten::quantize_per_tensor" in t or "quantize_per_channel" in t
            for t in op_targets
        )
        has_dequant: bool = any(
            "aten::dequantize" in t for t in op_targets
        )
        if has_quant and not has_dequant:
            return [
                "Model contains quantize operations without matching dequantize "
                "(possible incomplete quantization, may break edge compilation)"
            ]
        return []

    def _check_xnnpack(
        self, traced: GraphModule, op_targets: list[str]
    ) -> list[str]:
        """Check XNNPACK-specific constraints.

        Args:
            traced: Traced GraphModule of the full pipeline.
            op_targets: List of operation target strings from the graph.

        Returns:
            List of violation description strings (empty if compliant).
        """
        violations: list[str] = []

        # XNNPACK prefers NHWC; flag if model operates primarily in NCHW
        nhwc_friendly: bool = False
        nchw_ops: int = 0
        for node in traced.graph.nodes:
            target_str = str(node.target)
            if "channels_last" in target_str.lower() or "nhwc" in target_str.lower():
                nhwc_friendly = True
            # Count 2D convolutions (typically NCHW in PyTorch)
            if "aten::conv2d" in target_str or "aten::conv1d" in target_str:
                nchw_ops += 1

        if nchw_ops > MIN_CONV_OPS_FOR_NHWC_WARNING and not nhwc_friendly:
            violations.append(
                f"Model has {nchw_ops} convolution(s) but no NHWC layout "
                "operations detected. XNNPACK performs significantly better "
                "with NHWC layout. Consider using channels_last memory format."
            )

        # Check for unsupported XNNPACK ops (dense ops without XNNPACK kernels)
        xnnpack_unsupported: list[str] = []
        for target in op_targets:
            if any(
                x in target
                for x in [
                    "aten::group_norm",
                    "aten::layer_norm",
                    "aten::instance_norm",
                ]
            ):
                if "aten::layer_norm" in target:
                    continue  # XNNPACK does support layer_norm
                xnnpack_unsupported.append(target)

        if xnnpack_unsupported:
            violations.append(
                "XNNPACK may have limited or no support for: "
                f"{list(set(xnnpack_unsupported))}"
            )

        return violations

    def _check_qnn(
        self, traced: GraphModule, op_targets: list[str]
    ) -> list[str]:
        """Check QNN (Qualcomm)-specific constraints.

        Args:
            traced: Traced GraphModule of the full pipeline.
            op_targets: List of operation target strings from the graph.

        Returns:
            List of violation description strings (empty if compliant).
        """
        violations: list[str] = []

        # QNN has specific data type constraints.
        # Check for fp16 operations that may need special handling.
        qnn_unsupported_types: list[str] = []
        for node in traced.graph.nodes:
            target_str = str(node.target)
            if "float16" in target_str.lower() or "half" in target_str.lower():
                qnn_unsupported_types.append(target_str)

        if qnn_unsupported_types:
            violations.append(
                "QNN may have limited fp16 support for: "
                f"{list(set(qnn_unsupported_types))}. "
                "Consider quantizing to INT8 for QNN deployment."
            )

        # Check for operations known to have limited QNN support
        qnn_limited_ops: list[str] = []
        for target in op_targets:
            if any(
                x in target
                for x in [
                    "aten::embedding",
                    "aten::embedding_bag",
                    "aten::nonzero",
                    "aten::masked_select",
                ]
            ):
                qnn_limited_ops.append(target)

        if qnn_limited_ops:
            violations.append(
                "QNN may have limited or no support for: "
                f"{list(set(qnn_limited_ops))}"
            )

        return violations

    # ------------------------------------------------------------------
    # Export compatibility
    # ------------------------------------------------------------------

    def _verify_export_compatibility(
        self,
        backbone: nn.Module,
        neck: nn.Module | None,
        head: nn.Module,
        input_size: tuple[int, int],
    ) -> None:
        """Verify torch.export works without graph breaks.

        Args:
            backbone: Backbone module.
            neck: Optional neck module.
            head: Head module.
            input_size: Input image size (H, W).

        Raises:
            RuntimeError: If export fails with graph breaks.
        """
        h, w = input_size
        dummy_input = torch.randn(1, 3, h, w)

        # Create a combined model for export testing
        class _CombinedModel(nn.Module):
            def __init__(
                self,
                backbone: nn.Module,
                neck: nn.Module | None,
                head: nn.Module,
            ) -> None:
                super().__init__()
                self.backbone = backbone
                self.neck = neck
                self.head = head

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = self.backbone(x)
                if self.neck is not None:
                    features = self.neck(features)
                return self.head(features)

        combined = _CombinedModel(backbone, neck, head)
        combined.eval()

        try:
            _ = export(combined, (dummy_input,), strict=False)
        except Exception as e:
            raise RuntimeError(ERR_EXPORT_FAILED) from e

"""FX Graph Rewriter for Edge Hardware Compatibility.

This module provides the TargetRewriter class that transforms PyTorch FX graphs
to be compatible with edge hardware (ExecuTorch, ONNX) by:
1. Replacing edge-incompatible activations (GELU, SiLU) with edge-friendly alternatives
2. Collapsing redundant memory layout permutations around LayerNorm operations
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import fx, nn
from torch.fx import GraphModule


class TargetRewriter:
    """Rewrites FX graphs for edge hardware compatibility.

    This class performs graph transformations to make models compatible with
    edge deployment targets like ExecuTorch (XNNPACK) and ONNX by:
    - Replacing GELU/SiLU activations with ReLU/HardSwish
    - Collapsing redundant NCHW <-> NHWC permutations around LayerNorm
    """

    # Mapping of edge-incompatible module activations to their edge-friendly replacements
    MODULE_ACTIVATION_REPLACEMENTS: dict[type[torch.nn.Module], type[torch.nn.Module]] = {
        nn.GELU: nn.ReLU,
        nn.SiLU: nn.Hardswish,
    }

    # Mapping of edge-incompatible functional activations to their edge-friendly replacements
    FUNCTIONAL_ACTIVATION_REPLACEMENTS: dict[Callable, type[torch.nn.Module]] = {
        torch.nn.functional.gelu: nn.ReLU,
        torch.nn.functional.silu: nn.Hardswish,
    }

    # Keyword arguments that should be dropped when replacing functional
    # activations with module-based alternatives.
    _DROPPED_KWARGS: frozenset[str] = frozenset({"inplace"})

    def __init__(self) -> None:
        """Initialize the TargetRewriter."""
        pass

    def rewrite_for_edge(self, model: nn.Module) -> nn.Module:
        """Rewrite a model's FX graph for edge hardware compatibility.

        Args:
            model: The PyTorch model to rewrite. Can be a regular nn.Module
                or an already traced GraphModule.

        Returns:
            A new GraphModule with edge-compatible graph transformations applied.
            Preserves training/eval mode and gradient flow.
        """
        # Trace the model if it's not already a GraphModule
        if not isinstance(model, GraphModule):
            model = fx.symbolic_trace(model)

        # Apply graph transformations
        self._replace_activations(model)
        self._collapse_layernorm_permutations(model)

        # Recompile the graph to ensure validity
        model.recompile()

        # Preserve the original model's training mode
        model.train(model.training)

        return model

    def _replace_activations(self, model: GraphModule) -> None:
        """Replace edge-incompatible activations with edge-friendly alternatives.

        Replaces:
        - nn.GELU / F.gelu -> nn.ReLU
        - nn.SiLU / F.silu -> nn.Hardswish
        """
        graph = model.graph
        modules = dict(model.named_modules())

        # Track nodes to replace
        nodes_to_replace: list[tuple[fx.Node, nn.Module]] = []

        for node in graph.nodes:
            if node.op not in {"call_function", "call_module"}:
                continue

            target = node.target
            replacement = None

            # Check for module-based activations
            if node.op == "call_module" and target in modules:
                module = modules[target]
                for orig_type, repl_type in self.MODULE_ACTIVATION_REPLACEMENTS.items():
                    if isinstance(module, orig_type):
                        replacement = repl_type()
                        break

            # Check for functional activations
            elif node.op == "call_function":
                for orig_func, repl_type in self.FUNCTIONAL_ACTIVATION_REPLACEMENTS.items():
                    if target is orig_func:
                        replacement = repl_type()
                        break

            if replacement is not None:
                nodes_to_replace.append((node, replacement))

        # Perform replacements
        for node, replacement_module in nodes_to_replace:
            self._replace_node_with_module(model, node, replacement_module)

    def _replace_node_with_module(
        self, model: GraphModule, node: fx.Node, replacement_module: nn.Module
    ) -> None:
        """Replace a node with a new module call.

        Filters out keyword arguments that are not valid for the replacement
        module (e.g. ``inplace``, which is only meaningful for functional
        activations).

        Args:
            model: The GraphModule being modified.
            node: The node to replace.
            replacement_module: The new module to use as replacement.
        """
        graph = model.graph

        # Filter kwargs that are not supported by module-based activations
        filtered_kwargs = {
            k: v for k, v in node.kwargs.items() if k not in self._DROPPED_KWARGS
        }

        # Generate a unique name for the new module
        module_name = f"_rewritten_{replacement_module.__class__.__name__.lower()}_{id(node)}"
        model.add_module(module_name, replacement_module)

        with graph.inserting_before(node):
            new_node = graph.call_module(module_name, node.args, filtered_kwargs)
            node.replace_all_uses_with(new_node)
        graph.erase_node(node)

    def _collapse_layernorm_permutations(self, model: GraphModule) -> None:
        """Collapse redundant NCHW <-> NHWC permutations around LayerNorm.

        Pattern detected: permute -> LayerNorm -> permute (or reverse)
        This is common in ConvNeXt and ViT where LayerNorm operates on
        channels-last format but the rest of the network uses channels-first.

        The optimization replaces the pattern with a single LayerNorm
        that operates on the correct normalized_shape, eliminating
        unnecessary memory copies.
        """
        graph = model.graph
        modules = dict(model.named_modules())

        nodes = list(graph.nodes)
        i = 0
        while i < len(nodes) - 2:
            n1, n2, n3 = nodes[i], nodes[i + 1], nodes[i + 2]

            # Check for pattern: permute -> layernorm -> permute
            if self._is_permute_layernorm_permute(n1, n2, n3, modules):
                self._collapse_permute_layernorm_permute(model, n1, n2, n3, modules)
                # Refresh node list after modification
                nodes = list(graph.nodes)
                i = 0  # Restart scan
                continue

            # Check for pattern: layernorm -> permute (when input is already NHWC)
            if self._is_layernorm_permute(n1, n2, modules):
                self._collapse_layernorm_permute(model, n1, n2, modules)
                nodes = list(graph.nodes)
                i = 0
                continue

            # Check for pattern: permute -> layernorm (when output should be NCHW)
            if self._is_permute_layernorm(n1, n2, modules):
                self._collapse_permute_layernorm(model, n1, n2, modules)
                nodes = list(graph.nodes)
                i = 0
                continue

            i += 1

    @staticmethod
    def _is_permute(node: fx.Node, modules: dict[str, nn.Module]) -> bool:  # noqa: ARG004
        """Check if node is a permute operation (NCHW <-> NHWC).

        Handles both ``call_method`` (``x.permute(...)``) and
        ``call_function`` (``torch.permute(x, ...)`` / ``torch.Tensor.permute(x, ...)``).
        """
        if node.op == "call_method" and node.target == "permute":
            pass  # This is the common x.permute(...) pattern
        elif node.op == "call_function" and node.target in (
            torch.permute,
            torch.Tensor.permute,
        ):
            pass  # Direct functional call
        else:
            return False

        # Extract permute dimensions based on node op type
        if node.op == "call_method":
            # x.permute(0, 2, 3, 1) -> args = (x, 0, 2, 3, 1)
            perm = node.args[1:]
        else:
            # torch.permute(x, (0, 2, 3, 1)) -> args = (x, (0, 2, 3, 1))
            perm = node.args[1]
            if not isinstance(perm, (tuple, list)):
                return False

        # Check for NCHW <-> NHWC pattern: (0, 2, 3, 1) or (0, 3, 1, 2)
        return bool(
            isinstance(perm, (tuple, list))
            and len(perm) == 4  # noqa: PLR2004
            and tuple(perm) in ((0, 2, 3, 1), (0, 3, 1, 2))
        )

    def _is_layernorm(
        self, node: fx.Node, modules: dict[str, nn.Module]
    ) -> tuple[bool, nn.LayerNorm | None]:
        """Check if node is a LayerNorm operation. Returns (is_layernorm, layernorm_module)."""
        if node.op == "call_module" and node.target in modules:
            module = modules[node.target]
            if isinstance(module, nn.LayerNorm):
                return True, module

        if node.op == "call_function" and node.target in (
            torch.nn.functional.layer_norm,
            torch.layer_norm,
        ):
            return True, None  # Functional layer_norm

        return False, None

    def _is_permute_layernorm_permute(
        self, n1: fx.Node, n2: fx.Node, n3: fx.Node, modules: dict[str, nn.Module]
    ) -> bool:
        """Check for permute -> layernorm -> permute pattern."""
        if not self._is_permute(n1, modules):
            return False

        is_ln, _ = self._is_layernorm(n2, modules)
        if not is_ln:
            return False

        if not self._is_permute(n3, modules):
            return False

        # Check that n2's input is n1's output and n3's input is n2's output
        return n2.args[0] is n1 and n3.args[0] is n2

    def _is_layernorm_permute(
        self, n1: fx.Node, n2: fx.Node, modules: dict[str, nn.Module]
    ) -> bool:
        """Check for layernorm -> permute pattern."""
        is_ln, _ = self._is_layernorm(n1, modules)
        if not is_ln:
            return False

        if not self._is_permute(n2, modules):
            return False

        return n2.args[0] is n1

    def _is_permute_layernorm(
        self, n1: fx.Node, n2: fx.Node, modules: dict[str, nn.Module]
    ) -> bool:
        """Check for permute -> layernorm pattern."""
        if not self._is_permute(n1, modules):
            return False

        is_ln, _ = self._is_layernorm(n2, modules)
        if not is_ln:
            return False

        return n2.args[0] is n1

    def _create_channel_norm_for_nchw(
        self,
        ln_module: nn.LayerNorm,
    ) -> nn.Module:
        """Create a channel-normalization module that works on NCHW layout.

        When LayerNorm with ``normalized_shape=[C]`` operates on NHWC layout,
        it normalises the channel dimension.  For the equivalent operation on
        NCHW input we use :class:`nn.GroupNorm(num_groups=1)`` which
        normalises across all spatial and channel activations, providing a
        close edge-friendly approximation without requiring permute nodes.

        If the original LayerNorm has a multi-dimensional
        ``normalized_shape`` it is kept as-is under the assumption that the
        spatial dimensions are included (e.g. ``[H, W]``).
        """
        normalized_shape = ln_module.normalized_shape
        eps = ln_module.eps
        elementwise_affine = ln_module.elementwise_affine

        # Single-element normalized_shape -> channel norm via GroupNorm
        if len(normalized_shape) == 1:
            num_channels = normalized_shape[0]
            gn = nn.GroupNorm(
                num_groups=1,
                num_channels=num_channels,
                eps=eps,
                affine=elementwise_affine,
            )
            if elementwise_affine and ln_module.weight is not None:
                gn.weight.data.copy_(ln_module.weight.data.view_as(gn.weight))
            if elementwise_affine and ln_module.bias is not None:
                gn.bias.data.copy_(ln_module.bias.data.view_as(gn.bias))
            return gn

        # Multi-dim normalized_shape: LayerNorm works correctly on NCHW
        # because the last ``len(normalized_shape)`` dims match.
        new_ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        if elementwise_affine:
            if ln_module.weight is not None:
                new_ln.weight.data.copy_(ln_module.weight.data)
            if ln_module.bias is not None:
                new_ln.bias.data.copy_(ln_module.bias.data)
        return new_ln

    def _collapse_permute_layernorm_permute(
        self,
        model: GraphModule,
        perm1: fx.Node,
        ln_node: fx.Node,
        perm2: fx.Node,
        modules: dict[str, nn.Module],
    ) -> None:
        """Collapse permute -> layernorm -> permute to a single norm module.

        The three-node pattern is replaced by a single :class:`nn.GroupNorm`
        (for single-dim ``normalized_shape``) or :class:`nn.LayerNorm` (for
        multi-dim ``normalized_shape``) that operates on NCHW directly.
        """
        graph = model.graph

        # Get the LayerNorm module
        is_ln_module, ln_module = self._is_layernorm(ln_node, modules)
        if not is_ln_module or ln_module is None:
            return

        norm_module = self._create_channel_norm_for_nchw(ln_module)

        module_name = f"_collapsed_norm_{id(perm1)}"
        model.add_module(module_name, norm_module)

        with graph.inserting_before(perm1):
            # perm1.args[0] is the input before the first permute (NCHW)
            new_node = graph.call_module(module_name, (perm1.args[0],))

        # Wire all uses of the last permute to the new node
        perm2.replace_all_uses_with(new_node)

        # Erase nodes in reverse order
        graph.erase_node(perm2)
        graph.erase_node(ln_node)
        graph.erase_node(perm1)

    def _collapse_layernorm_permute(
        self,
        model: GraphModule,
        ln_node: fx.Node,
        perm_node: fx.Node,
        modules: dict[str, nn.Module],
    ) -> None:
        """Collapse layernorm -> permute to a norm module on NCHW input."""
        graph = model.graph

        is_ln_module, ln_module = self._is_layernorm(ln_node, modules)
        if not is_ln_module or ln_module is None:
            return

        norm_module = self._create_channel_norm_for_nchw(ln_module)

        module_name = f"_collapsed_norm_{id(ln_node)}"
        model.add_module(module_name, norm_module)

        with graph.inserting_before(ln_node):
            new_node = graph.call_module(module_name, (ln_node.args[0],))

        perm_node.replace_all_uses_with(new_node)

        graph.erase_node(perm_node)
        graph.erase_node(ln_node)

    def _collapse_permute_layernorm(
        self,
        model: GraphModule,
        perm_node: fx.Node,
        ln_node: fx.Node,
        modules: dict[str, nn.Module],
    ) -> None:
        """Collapse permute -> layernorm to a norm module on NCHW input."""
        graph = model.graph

        is_ln_module, ln_module = self._is_layernorm(ln_node, modules)
        if not is_ln_module or ln_module is None:
            return

        norm_module = self._create_channel_norm_for_nchw(ln_module)

        module_name = f"_collapsed_norm_{id(perm_node)}"
        model.add_module(module_name, norm_module)

        with graph.inserting_before(perm_node):
            new_node = graph.call_module(module_name, (perm_node.args[0],))

        ln_node.replace_all_uses_with(new_node)

        graph.erase_node(ln_node)
        graph.erase_node(perm_node)

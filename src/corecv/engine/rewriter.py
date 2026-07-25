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
  """FX Graph Rewriter optimized for Edge and NPU Hardware Compatibility.

  Transforms PyTorch FX graphs for ExecuTorch, ONNX, and NPU targets by:
  1. Replacing GELU/SiLU activations with ReLU/Hardswish in linear time O(N).
  2. Fusing ConvNeXt/Vision MLP blocks into pure NCHW operations (converting
     LayerNorm to GroupNorm and Linear to Conv2d 1x1, completely eliminating
     permute overhead and memory transpositions for NPUs).
  3. Collapsing direct permute -> LayerNorm -> permute patterns.
  4. Safely deleting unused submodules without breaking parameter references in
     functional calls or composite modules.
  """

  # Mapping of edge-incompatible module activations to their edge-friendly replacements
  MODULE_ACTIVATION_REPLACEMENTS: dict[
      type[torch.nn.Module], type[torch.nn.Module]
  ] = {
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
    """Rewrite a model's FX graph for NPU and edge hardware compatibility.

    Args:
        model: The PyTorch model to rewrite. Can be a regular nn.Module or an
          already traced GraphModule.

    Returns:
        A new GraphModule with edge-compatible graph transformations applied
        and orphaned submodules safely purged.
    """
    # Trace the model if it's not already a GraphModule
    if not isinstance(model, GraphModule):
      model = fx.symbolic_trace(model)

    # Step 1: Replace activations (GELU -> ReLU, SiLU -> Hardswish)
    self._replace_activations(model)

    # Step 2: Fuse & collapse permute/LayerNorm patterns into NCHW native blocks
    self._collapse_convnext_and_permute_patterns(model)

    # Step 3: Purge orphaned submodules safely
    self._delete_unused_submodules(model)

    # Recompile the graph to ensure validity
    model.recompile()

    # Preserve original training/eval state
    model.train(model.training)

    return model

  def _replace_activations(self, model: GraphModule) -> None:
    """Replace edge-incompatible activations in a single O(N) linear pass."""
    graph = model.graph
    modules = dict(model.named_modules())
    nodes_to_replace: list[tuple[fx.Node, nn.Module]] = []

    for node in graph.nodes:
      if node.op not in {"call_function", "call_module"}:
        continue

      target = node.target
      replacement = None

      # Check module replacements
      if node.op == "call_module" and target in modules:
        module = modules[target]
        for orig_type, repl_type in self.MODULE_ACTIVATION_REPLACEMENTS.items():
          if isinstance(module, orig_type):
            replacement = repl_type()
            break

      # Check functional replacements
      elif node.op == "call_function":
        for (
            orig_func,
            repl_type,
        ) in self.FUNCTIONAL_ACTIVATION_REPLACEMENTS.items():
          if target is orig_func:
            replacement = repl_type()
            break

      if replacement is not None:
        nodes_to_replace.append((node, replacement))

    for node, replacement_module in nodes_to_replace:
      self._replace_node_with_module(model, node, replacement_module)

  def _collapse_convnext_and_permute_patterns(self, model: GraphModule) -> None:
    """Detects and fuses ConvNeXt MLP blocks and permute-LayerNorm sequences.

    Eliminates all memory transposition (permute) nodes by converting
    LayerNorm -> GroupNorm(1, C) and Linear -> Conv2d(1x1) on NCHW layout.
    """
    graph = model.graph
    modules = dict(model.named_modules())

    # Find all permute(0, 2, 3, 1) nodes (NCHW -> NHWC)
    perm_nodes = [
        node for node in graph.nodes if self._is_permute_to_nhwc(node)
    ]

    for p1 in perm_nodes:
      if p1.graph is None:  # Node was already erased in a previous fusion step
        continue

      input_node = p1.args[0]

      if len(p1.users) != 1:
        continue

      ln_node = list(p1.users.keys())[0]
      is_ln, ln_mod = self._is_layernorm(ln_node, modules)
      if not is_ln or ln_mod is None:
        continue

      # Pattern A: Direct permute -> LayerNorm -> permute
      if len(ln_node.users) == 1:
        next_node = list(ln_node.users.keys())[0]
        if self._is_permute_to_nchw(next_node):
          p2 = next_node
          gn_module = self._layernorm_to_groupnorm(ln_mod)
          gn_name = f"_collapsed_gn_{id(p1)}"
          model.add_module(gn_name, gn_module)

          with graph.inserting_before(p1):
            gn_node = graph.call_module(gn_name, (input_node,))

          p2.replace_all_uses_with(gn_node)
          graph.erase_node(p2)
          graph.erase_node(ln_node)
          graph.erase_node(p1)
          continue

      # Pattern B: ConvNeXt MLP Block: permute -> LayerNorm -> Linear -> Act -> Linear -> permute
      chain = self._match_convnext_mlp_chain(ln_node, modules)
      if chain is not None:
        l1_node, act_node, l2_node, p2_node = chain

        # 1. Convert LayerNorm -> GroupNorm(1, C)
        gn_module = self._layernorm_to_groupnorm(ln_mod)
        gn_name = f"_convnext_gn_{id(p1)}"
        model.add_module(gn_name, gn_module)

        # 2. Convert Linear1 -> Conv2d(1x1)
        l1_mod = modules[l1_node.target]
        conv1_module = self._linear_to_conv2d(l1_mod)
        conv1_name = f"_convnext_conv1_{id(l1_node)}"
        model.add_module(conv1_name, conv1_module)

        # 3. Convert Linear2 -> Conv2d(1x1)
        l2_mod = modules[l2_node.target]
        conv2_module = self._linear_to_conv2d(l2_mod)
        conv2_name = f"_convnext_conv2_{id(l2_node)}"
        model.add_module(conv2_name, conv2_module)

        # Rewire graph in pure NCHW format ensuring correct topological order
        with graph.inserting_before(p1):
          gn_node = graph.call_module(gn_name, (input_node,))
          conv1_node = graph.call_module(conv1_name, (gn_node,))

        # Rewire activation node input to use conv1_node
        act_node.args = tuple(
            conv1_node if arg is l1_node else arg for arg in act_node.args
        )

        # Insert conv2_node AFTER activation node to respect topological order
        with graph.inserting_after(act_node):
          conv2_node = graph.call_module(conv2_name, (act_node,))

        # Replace all uses of the final permute node with conv2_node
        p2_node.replace_all_uses_with(conv2_node)

        # Erase old nodes in reverse topological order
        graph.erase_node(p2_node)
        graph.erase_node(l2_node)
        graph.erase_node(l1_node)
        graph.erase_node(ln_node)
        graph.erase_node(p1)

  def _delete_unused_submodules(self, model: GraphModule) -> None:
    """Safely removes orphaned submodules without deleting required layers or subcomponents."""
    call_module_targets = set()
    get_attr_prefixes = set()

    for node in model.graph.nodes:
      if node.op == "call_module":
        call_module_targets.add(str(node.target))
      elif node.op == "get_attr":
        parts = str(node.target).split(".")
        for i in range(1, len(parts)):
          get_attr_prefixes.add(".".join(parts[:i]))

    needed_modules = set()

    # Add call_module targets and their parents
    for target in call_module_targets:
      needed_modules.add(target)
      parts = target.split(".")
      for i in range(1, len(parts)):
        needed_modules.add(".".join(parts[:i]))

    # Add get_attr prefixes and their parents
    for prefix in get_attr_prefixes:
      needed_modules.add(prefix)
      parts = prefix.split(".")
      for i in range(1, len(parts)):
        needed_modules.add(".".join(parts[:i]))

    all_module_names = [name for name, _ in model.named_modules() if name]
    final_needed = set(needed_modules)

    # Protect subcomponents/children of called modules (e.g. out_proj inside MultiheadAttention)
    for name in all_module_names:
      for target in call_module_targets:
        if name.startswith(f"{target}."):
          final_needed.add(name)
          break

    # Delete only genuinely orphaned submodules
    for name in all_module_names:
      if name not in final_needed:
        try:
          model.delete_submodule(name)
        except (AttributeError, KeyError):
          pass

  def _match_convnext_mlp_chain(
      self, ln_node: fx.Node, modules: dict[str, nn.Module]
  ) -> tuple[fx.Node, fx.Node, fx.Node, fx.Node] | None:
    """Matches the pattern: LayerNorm -> Linear -> Activation -> Linear -> permute(0, 3, 1, 2)."""
    if len(ln_node.users) != 1:
      return None
    l1_node = list(ln_node.users.keys())[0]
    if not self._is_linear(l1_node, modules):
      return None

    if len(l1_node.users) != 1:
      return None
    act_node = list(l1_node.users.keys())[0]

    if len(act_node.users) != 1:
      return None
    l2_node = list(act_node.users.keys())[0]
    if not self._is_linear(l2_node, modules):
      return None

    if len(l2_node.users) != 1:
      return None
    p2_node = list(l2_node.users.keys())[0]
    if not self._is_permute_to_nchw(p2_node):
      return None

    return l1_node, act_node, l2_node, p2_node

  def _replace_node_with_module(
      self, model: GraphModule, node: fx.Node, replacement_module: nn.Module
  ) -> None:
    """Replace a single FX node with a new module call."""
    graph = model.graph
    filtered_kwargs = {
        k: v for k, v in node.kwargs.items() if k not in self._DROPPED_KWARGS
    }
    module_name = (
        f"_rewritten_{replacement_module.__class__.__name__.lower()}_{id(node)}"
    )
    model.add_module(module_name, replacement_module)

    with graph.inserting_before(node):
      new_node = graph.call_module(module_name, node.args, filtered_kwargs)
      node.replace_all_uses_with(new_node)
    graph.erase_node(node)

  # Helper functions for pattern matching and module conversion
  def _is_permute_to_nhwc(self, node: fx.Node) -> bool:
    return self._check_permute_tuple(node, (0, 2, 3, 1))

  def _is_permute_to_nchw(self, node: fx.Node) -> bool:
    return self._check_permute_tuple(node, (0, 3, 1, 2))

  @staticmethod
  def _check_permute_tuple(
      node: fx.Node, target_perm: tuple[int, ...]
  ) -> bool:
    if node.op == "call_method" and node.target == "permute":
      perm = node.args[1:]
    elif node.op == "call_function" and node.target in (
        torch.permute,
        torch.Tensor.permute,
    ):
      perm = node.args[1]
      if not isinstance(perm, (tuple, list)):
        return False
    else:
      return False
    return bool(isinstance(perm, (tuple, list)) and tuple(perm) == target_perm)

  def _is_layernorm(
      self, node: fx.Node, modules: dict[str, nn.Module]
  ) -> tuple[bool, nn.LayerNorm | None]:
    if node.op == "call_module" and node.target in modules:
      module = modules[node.target]
      if isinstance(module, nn.LayerNorm):
        return True, module
    return False, None

  def _is_linear(self, node: fx.Node, modules: dict[str, nn.Module]) -> bool:
    if node.op == "call_module" and node.target in modules:
      return isinstance(modules[node.target], nn.Linear)
    return False

  @staticmethod
  def _layernorm_to_groupnorm(ln: nn.LayerNorm) -> nn.GroupNorm:
    num_channels = (
        ln.normalized_shape[0]
        if isinstance(ln.normalized_shape, (list, tuple))
        else ln.normalized_shape
    )
    gn = nn.GroupNorm(
        num_groups=1,
        num_channels=num_channels,
        eps=ln.eps,
        affine=ln.elementwise_affine,
    )
    if ln.elementwise_affine:
      if ln.weight is not None:
        gn.weight.data.copy_(ln.weight.data.view_as(gn.weight))
      if ln.bias is not None:
        gn.bias.data.copy_(ln.bias.data.view_as(gn.bias))
    return gn

  @staticmethod
  def _linear_to_conv2d(linear: nn.Linear) -> nn.Conv2d:
    conv = nn.Conv2d(
        in_channels=linear.in_features,
        out_channels=linear.out_features,
        kernel_size=1,
        bias=linear.bias is not None,
    )
    conv.weight.data.copy_(
        linear.weight.data.view(linear.out_features, linear.in_features, 1, 1)
    )
    if linear.bias is not None and conv.bias is not None:
      conv.bias.data.copy_(linear.bias.data)
    return conv
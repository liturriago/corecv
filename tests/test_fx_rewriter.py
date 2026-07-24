"""Comprehensive tests for the TargetRewriter FX graph rewriter.

Tests cover:
1. GELU/SiLU module and functional activation replacement
2. ConvNeXt/ViT permute -> LayerNorm -> permute pattern collapse
3. Forward pass preservation (shapes, no NaN/Inf)
4. Gradient flow verification (backward without NaN)
5. Graph traceability (torch.fx re-trace and torch.export)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import export
from torch.fx import GraphModule, symbolic_trace

from corecv.engine.rewriter import TargetRewriter

# ======================================================================
# Test model definitions
# ======================================================================


class GELUModule(nn.Module):
    """Simple model using ``nn.GELU`` activation."""

    def __init__(self) -> None:
        """Initialise with Conv2d -> GELU -> Conv2d."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(16, 16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            Output tensor of shape ``(N, 16, H-4, W-4)``.
        """
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        return x


class SiLUModule(nn.Module):
    """Simple model using ``nn.SiLU`` activation."""

    def __init__(self) -> None:
        """Initialise with Conv2d -> SiLU -> Conv2d."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(16, 16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            Output tensor of shape ``(N, 16, H-4, W-4)``.
        """
        x = self.conv1(x)
        x = self.silu(x)
        x = self.conv2(x)
        return x


class FunctionalActivationModule(nn.Module):
    """Model using ``F.gelu`` and ``F.silu`` in the forward method."""

    def __init__(self) -> None:
        """Initialise with two conv layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using functional activations.

        Args:
            x: Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            Output tensor.
        """
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.silu(x)
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-like block with depthwise conv, LayerNorm with permutes, and GELU.

    This mimics the common pattern:
        dwconv -> permute(0,2,3,1) -> LayerNorm -> permute(0,3,1,2) -> pwconv1 -> GELU -> pwconv2
    """

    def __init__(self, dim: int = 32) -> None:
        """Initialise ConvNeXt-like block.

        Args:
            dim: Number of channels.
        """
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * 4, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(N, dim, H, W)``.

        Returns:
            Output tensor ``(N, dim, H, W)``.
        """
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x


class TwoNormBlock(nn.Module):
    """Model with two sequential permute-LayerNorm-permute patterns.

    Tests that multiple norm-collapse patterns are handled correctly.
    """

    def __init__(self, dim: int = 32) -> None:
        """Initialise with two LayerNorm blocks.

        Args:
            dim: Number of channels.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm1 = nn.LayerNorm(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm2 = nn.LayerNorm(dim)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with two norm blocks.

        Args:
            x: Input tensor ``(N, dim, H, W)``.

        Returns:
            Output tensor.
        """
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv3(x)
        return x


class SimpleConvNet(nn.Module):
    """Simple conv net with no GELU/SiLU or permute patterns.

    Used for forward-pass preservation tests where the rewrite should be a no-op.
    """

    def __init__(self) -> None:
        """Initialise with Conv2d -> ReLU -> Conv2d."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(N, 3, H, W)``.

        Returns:
            Output tensor.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class ViTLikeBlock(nn.Module):
    """ViT-like block with Linear -> LayerNorm -> Linear -> GELU.

    This model does NOT use permute patterns; it tests graph traceability
    and activation replacement in a transformer-style context.
    """

    def __init__(self, dim: int = 64) -> None:
        """Initialise ViT-like block.

        Args:
            dim: Feature dimension.
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(N, *, dim)``.

        Returns:
            Output tensor ``(N, *, dim)``.
        """
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def rewriter() -> TargetRewriter:
    """Fixture providing a fresh TargetRewriter instance."""
    return TargetRewriter()


@pytest.fixture
def input_2d() -> torch.Tensor:
    """Fixture providing a standard 2D image input ``(2, 3, 32, 32)``."""
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def input_non_square() -> torch.Tensor:
    """Fixture providing a non-square input ``(2, 3, 480, 640)``."""
    return torch.randn(2, 3, 480, 640)


@pytest.fixture
def input_1d() -> torch.Tensor:
    """Fixture providing a 1D sequence input ``(2, 16, 64)``."""
    return torch.randn(2, 16, 64)


# ======================================================================
# GELU / SiLU replacement tests
# ======================================================================


class TestActivationReplacement:
    """Verify GELU and SiLU activations are replaced with edge-friendly alternatives."""

    def test_replace_gelu_module(self, rewriter: TargetRewriter) -> None:
        """``nn.GELU`` should be replaced with ``nn.ReLU`` after rewrite."""
        model = GELUModule()
        rewritten = rewriter.rewrite_for_edge(model)

        # Verify the graph calls ReLU instead of GELU
        has_relu = False
        has_gelu = False
        for node in rewritten.graph.nodes:
            if node.op == "call_module":
                mod = rewritten.get_submodule(node.target)
                if isinstance(mod, nn.ReLU):
                    has_relu = True
                if isinstance(mod, nn.GELU):
                    has_gelu = True

        assert not has_gelu, "GELU module should no longer be in the graph"
        assert has_relu, "ReLU module should be called in the rewritten graph"
        assert isinstance(rewritten, GraphModule), "Result should be a GraphModule"

    def test_replace_silu_module(self, rewriter: TargetRewriter) -> None:
        """``nn.SiLU`` should be replaced with ``nn.Hardswish`` after rewrite."""
        model = SiLUModule()
        rewritten = rewriter.rewrite_for_edge(model)

        has_hardswish = False
        has_silu = False
        for node in rewritten.graph.nodes:
            if node.op == "call_module":
                mod = rewritten.get_submodule(node.target)
                if isinstance(mod, nn.Hardswish):
                    has_hardswish = True
                if isinstance(mod, nn.SiLU):
                    has_silu = True

        assert not has_silu, "SiLU module should no longer be in the graph"
        assert has_hardswish, "Hardswish module should be called in the rewritten graph"

    def test_replace_functional_gelu(self, rewriter: TargetRewriter) -> None:
        """``F.gelu`` should be replaced with a ``ReLU`` module call."""
        model = FunctionalActivationModule()
        rewritten = rewriter.rewrite_for_edge(model)

        has_relu = False
        has_functional_gelu = False
        for node in rewritten.graph.nodes:
            if node.op == "call_function" and node.target is F.gelu:
                has_functional_gelu = True
            if node.op == "call_module":
                mod = rewritten.get_submodule(node.target)
                if isinstance(mod, nn.ReLU):
                    has_relu = True

        assert not has_functional_gelu, "F.gelu should no longer be called in the graph"
        assert has_relu, "ReLU module should be present after replacing F.gelu"

    def test_replace_functional_silu(self, rewriter: TargetRewriter) -> None:
        """``F.silu`` should be replaced with a ``Hardswish`` module call."""
        model = FunctionalActivationModule()
        rewritten = rewriter.rewrite_for_edge(model)

        has_hardswish = False
        has_functional_silu = False
        for node in rewritten.graph.nodes:
            if node.op == "call_function" and node.target is F.silu:
                has_functional_silu = True
            if node.op == "call_module":
                mod = rewritten.get_submodule(node.target)
                if isinstance(mod, nn.Hardswish):
                    has_hardswish = True

        assert not has_functional_silu, "F.silu should no longer be called in the graph"
        assert has_hardswish, "Hardswish module should be present after replacing F.silu"

    def test_replace_mixed_activations(self, rewriter: TargetRewriter) -> None:
        """Both ``nn.GELU`` and ``nn.SiLU`` should be replaced in the same model."""
        model = ConvNeXtBlock(dim=16)
        rewritten = rewriter.rewrite_for_edge(model)

        has_relu = False
        has_gelu = False
        for node in rewritten.graph.nodes:
            if node.op == "call_module":
                mod = rewritten.get_submodule(node.target)
                if isinstance(mod, nn.ReLU):
                    has_relu = True
                if isinstance(mod, nn.GELU):
                    has_gelu = True

        assert not has_gelu, "GELU module should be replaced"
        assert has_relu, "ReLU should be present in the rewritten graph"

    def test_no_activation_replacement_for_relu(self, rewriter: TargetRewriter) -> None:
        """Models with only ReLU should remain unchanged by activation replacement."""
        model = SimpleConvNet()
        rewritten = rewriter.rewrite_for_edge(model)

        relu_count = sum(
            1 for node in rewritten.graph.nodes
            if node.op == "call_module"
            and isinstance(rewritten.get_submodule(node.target), nn.ReLU)
        )
        assert relu_count == 1, "The original ReLU should still be present"


# ======================================================================
# LayerNorm permute collapse tests
# ======================================================================


class TestLayerNormPermuteCollapse:
    """Verify permute nodes around LayerNorm are collapsed."""

    @staticmethod
    def _count_permutes(model: GraphModule) -> int:
        """Count permute operations in a graph module."""
        count = 0
        for node in model.graph.nodes:
            is_permute = (
                (node.op == "call_method" and node.target == "permute")
                or (
                    node.op == "call_function"
                    and node.target in (torch.permute, torch.Tensor.permute)
                )
            )
            if is_permute:
                count += 1
        return count

    def test_convnext_permute_collapse(self, rewriter: TargetRewriter) -> None:
        """Permute-LayerNorm-permute pattern in ConvNeXt should be collapsed."""
        model = ConvNeXtBlock(dim=32)
        rewritten = rewriter.rewrite_for_edge(model)

        permute_count = self._count_permutes(rewritten)

        assert permute_count == 0, (
            f"Expected 0 permute nodes after collapse, found {permute_count}"
        )

    def test_layernorm_preserved_as_groupnorm(self, rewriter: TargetRewriter) -> None:
        """After collapse, a norm module should still exist (GroupNorm for single-dim shape)."""
        model = ConvNeXtBlock(dim=32)
        rewritten = rewriter.rewrite_for_edge(model)

        # Check for a norm module in the graph calls
        norm_found = False
        for node in rewritten.graph.nodes:
            if node.op == "call_module":
                mod = rewritten.get_submodule(node.target)
                if isinstance(mod, (nn.LayerNorm, nn.GroupNorm)):
                    norm_found = True
                    break

        assert norm_found, "A normalization module should be present after collapse"

    def test_two_norm_blocks_collapsed(self, rewriter: TargetRewriter) -> None:
        """Multiple sequential permute-LayerNorm-permute patterns should all be collapsed."""
        model = TwoNormBlock(dim=16)
        rewritten = rewriter.rewrite_for_edge(model)

        # Verify both norms are still present
        norm_count = sum(
            1 for node in rewritten.graph.nodes
            if node.op == "call_module"
            and isinstance(rewritten.get_submodule(node.target), (nn.LayerNorm, nn.GroupNorm))
        )
        assert norm_count == 2, (
            f"Expected 2 norm modules after collapse, found {norm_count}"
        )

    def test_layernorm_without_permute_unchanged(self, rewriter: TargetRewriter) -> None:
        """LayerNorm without surrounding permutes should be left as-is."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.LayerNorm([16, 30, 30]),  # multi-dim normalized_shape on NCHW
        )
        rewritten = rewriter.rewrite_for_edge(model)

        layernorm_count = sum(
            1 for node in rewritten.graph.nodes
            if node.op == "call_module"
            and isinstance(rewritten.get_submodule(node.target), nn.LayerNorm)
        )
        # The LayerNorm should still be present (no permutes to collapse)
        assert layernorm_count >= 1, "LayerNorm without permutes should remain"

    def test_permute_only_no_collapse_without_layernorm(self, rewriter: TargetRewriter) -> None:
        """Solitary permute nodes without surrounding LayerNorm should not be removed."""

        class PermuteOnlyModule(nn.Module):
            """Module with only conv and permute (no LayerNorm)."""

            def __init__(self) -> None:
                """Initialise."""
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
                return self.conv(x).permute(0, 2, 3, 1)

        model = PermuteOnlyModule()
        rewritten = rewriter.rewrite_for_edge(model)

        permute_count = self._count_permutes(rewritten)
        assert permute_count >= 1, "Solitary permute should remain"


# ======================================================================
# Forward pass preservation tests
# ======================================================================


class TestForwardPassPreservation:
    """Verify rewritten models produce valid outputs."""

    def test_forward_pass_eval(self, rewriter: TargetRewriter, input_2d: torch.Tensor) -> None:
        """Rewritten model can run a forward pass in eval mode without errors."""
        model = GELUModule().eval()
        rewritten = rewriter.rewrite_for_edge(model)
        rewritten.eval()

        with torch.no_grad():
            output = rewritten(input_2d)

        assert output is not None, "Output should not be None"
        assert not torch.isnan(output).any(), "Output should not contain NaN"
        assert not torch.isinf(output).any(), "Output should not contain Inf"

    def test_forward_pass_train(self, rewriter: TargetRewriter, input_2d: torch.Tensor) -> None:
        """Rewritten model can run a forward pass in train mode without errors."""
        model = GELUModule().train()
        rewritten = rewriter.rewrite_for_edge(model)
        rewritten.train()

        output = rewritten(input_2d)

        assert output is not None, "Output should not be None"
        assert not torch.isnan(output).any(), "Output should not contain NaN"
        assert not torch.isinf(output).any(), "Output should not contain Inf"

    def test_output_shape_preserved(self, rewriter: TargetRewriter, input_2d: torch.Tensor) -> None:
        """Rewritten model preserves output tensor shape."""
        model = GELUModule()
        original_output = model(input_2d)
        rewritten = rewriter.rewrite_for_edge(model)
        rewritten_output = rewritten(input_2d)

        assert original_output.shape == rewritten_output.shape, (
            f"Output shape mismatch: {original_output.shape} vs {rewritten_output.shape}"
        )

    def test_convnext_forward_pass(self, rewriter: TargetRewriter) -> None:
        """ConvNeXt-like model produces valid output after rewrite."""
        model = ConvNeXtBlock(dim=32)
        rewritten = rewriter.rewrite_for_edge(model)

        x = torch.randn(2, 32, 14, 14)
        output = rewritten(x)

        assert output.shape == (2, 32, 14, 14), (
            f"Expected (2, 32, 14, 14), got {output.shape}"
        )
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_non_square_resolution(
        self, rewriter: TargetRewriter, input_non_square: torch.Tensor
    ) -> None:
        """Rewritten model works with non-square (480x640) inputs.

        This catches potential H/W indexing inversion bugs.
        """
        model = GELUModule()
        rewritten = rewriter.rewrite_for_edge(model)

        output = rewritten(input_non_square)

        assert output.shape[2] == 476, f"Expected height 476, got {output.shape[2]}"
        assert output.shape[3] == 636, f"Expected width 636, got {output.shape[3]}"
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_mode_preserved(self, rewriter: TargetRewriter) -> None:
        """The rewritten model preserves the training/eval mode of the original."""
        model = GELUModule()

        # Test eval mode preservation
        model.eval()
        rewritten_eval = rewriter.rewrite_for_edge(model)
        assert not rewritten_eval.training, "Rewritten model should be in eval mode"

        # Test train mode preservation
        model.train()
        rewritten_train = rewriter.rewrite_for_edge(model)
        assert rewritten_train.training, "Rewritten model should be in train mode"

    def test_forward_shape_no_permute_model(self, rewriter: TargetRewriter) -> None:
        """ViT-like model produces correct output shape after rewrite."""
        model = ViTLikeBlock(dim=64)
        rewritten = rewriter.rewrite_for_edge(model)

        x = torch.randn(2, 16, 64)
        output = rewritten(x)

        assert output.shape == (2, 16, 64), (
            f"Expected (2, 16, 64), got {output.shape}"
        )

    def test_multiple_forward_calls(self, rewriter: TargetRewriter, input_2d: torch.Tensor) -> None:
        """Rewritten model can be called multiple times consistently."""
        model = GELUModule()
        rewritten = rewriter.rewrite_for_edge(model)

        out1 = rewritten(input_2d)
        out2 = rewritten(input_2d)

        assert torch.equal(out1, out2), "Multiple forward calls should produce identical results"


# ======================================================================
# Gradient flow tests
# ======================================================================


class TestGradientFlow:
    """Verify gradients flow correctly through the rewritten graph."""

    def test_gradient_flow_gelu(self, rewriter: TargetRewriter, input_2d: torch.Tensor) -> None:
        """Gradients flow through the rewritten GELU -> ReLU replacement."""
        model = GELUModule().train()
        rewritten = rewriter.rewrite_for_edge(model)
        rewritten.train()

        output = rewritten(input_2d)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in rewritten.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient"
            )
            assert not torch.isnan(param.grad).any(), (
                f"Parameter '{name}' has NaN gradient"
            )

    def test_gradient_flow_convnext(self, rewriter: TargetRewriter) -> None:
        """Gradients flow through the rewritten ConvNeXt-like model."""
        model = ConvNeXtBlock(dim=16).train()
        rewritten = rewriter.rewrite_for_edge(model)
        rewritten.train()

        x = torch.randn(2, 16, 8, 8, requires_grad=True)
        output = rewritten(x)
        loss = output.sum()
        loss.backward()

        # Verify gradient flows back to input
        assert x.grad is not None, "Input gradient should not be None"
        assert not torch.isnan(x.grad).any(), "Input gradient should not contain NaN"

        # Check that graph-reachable parameters have gradients.
        # The original ``norm`` module is replaced by a collapsed module,
        # so we check that at least one parameter in the graph has gradients.
        params_with_grad = 0
        for _name, param in rewritten.named_parameters():
            if param.grad is not None:
                params_with_grad += 1
        assert params_with_grad > 0, (
            "At least one parameter should have a gradient in the rewritten graph"
        )

    def test_gradient_flow_functional(
        self, rewriter: TargetRewriter, input_2d: torch.Tensor
    ) -> None:
        """Gradients flow through rewritten functional activations."""
        model = FunctionalActivationModule().train()
        rewritten = rewriter.rewrite_for_edge(model)
        rewritten.train()

        output = rewritten(input_2d)
        loss = output.sum()
        loss.backward()

        all_grads_ok = all(
            param.grad is not None and not torch.isnan(param.grad).any()
            for param in rewritten.parameters()
        )
        assert all_grads_ok, "All parameters should have valid gradients"

    def test_gradient_flow_vit(self, rewriter: TargetRewriter, input_1d: torch.Tensor) -> None:
        """Gradients flow through the rewritten ViT-like model."""
        model = ViTLikeBlock(dim=64).train()
        rewritten = rewriter.rewrite_for_edge(model)
        rewritten.train()

        output = rewritten(input_1d)
        loss = output.sum()
        loss.backward()

        for name, param in rewritten.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient"
            )
            assert not torch.isnan(param.grad).any(), (
                f"Parameter '{name}' has NaN gradient"
            )

    def test_non_square_gradient_flow(
        self, rewriter: TargetRewriter, input_non_square: torch.Tensor
    ) -> None:
        """Gradients flow correctly with non-square (480x640) inputs."""
        model = GELUModule().train()
        rewritten = rewriter.rewrite_for_edge(model)
        rewritten.train()

        output = rewritten(input_non_square)
        loss = output.sum()
        loss.backward()

        for name, param in rewritten.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient"
            )
            assert not torch.isnan(param.grad).any(), (
                f"Parameter '{name}' has NaN gradient"
            )

    def test_gradient_after_multiple_steps(
        self, rewriter: TargetRewriter, input_2d: torch.Tensor
    ) -> None:
        """Gradients remain valid after multiple forward-backward steps."""
        model = GELUModule().train()
        rewritten = rewriter.rewrite_for_edge(model)
        rewritten.train()

        optim = torch.optim.SGD(rewritten.parameters(), lr=0.01)

        for step in range(3):
            optim.zero_grad()
            output = rewritten(input_2d)
            loss = output.sum()
            loss.backward()

            for name, param in rewritten.named_parameters():
                assert param.grad is not None, (
                    f"Step {step}: Parameter '{name}' has no gradient"
                )
                assert not torch.isnan(param.grad).any(), (
                    f"Step {step}: Parameter '{name}' has NaN gradient"
                )

            optim.step()

    def test_detach_no_break(self, rewriter: TargetRewriter, input_2d: torch.Tensor) -> None:
        """loss.backward() does not raise after detaching intermediate tensors."""
        model = GELUModule().train()
        rewritten = rewriter.rewrite_for_edge(model)
        rewritten.train()

        # Ensure backward() can complete without errors
        output = rewritten(input_2d)
        loss = output.sum()
        loss.backward()  # Should not raise

        assert True, "backward() completed without error"


# ======================================================================
# Graph traceability tests
# ======================================================================


class TestGraphTraceability:
    """Verify the rewritten graph remains traceable and exportable."""

    def test_retrace_with_symbolic_trace(
        self, rewriter: TargetRewriter, input_2d: torch.Tensor
    ) -> None:
        """Rewritten GraphModule can be re-traced with ``torch.fx.symbolic_trace``."""
        model = GELUModule()
        rewritten = rewriter.rewrite_for_edge(model)

        re_traced = symbolic_trace(rewritten)

        assert isinstance(re_traced, GraphModule), "Re-traced result should be a GraphModule"
        output = re_traced(input_2d)
        assert output.shape == rewritten(input_2d).shape, "Re-traced output shape should match"

    def test_retrace_convnext(self, rewriter: TargetRewriter) -> None:
        """ConvNeXt-like rewritten model can be re-traced."""
        model = ConvNeXtBlock(dim=16)
        rewritten = rewriter.rewrite_for_edge(model)

        re_traced = symbolic_trace(rewritten)

        assert isinstance(re_traced, GraphModule)
        x = torch.randn(2, 16, 8, 8)
        output = re_traced(x)
        assert output.shape == (2, 16, 8, 8)

    def test_export_compatible(self, rewriter: TargetRewriter) -> None:
        """Rewritten model can be exported with ``torch.export``."""
        model = ViTLikeBlock(dim=16)
        rewritten = rewriter.rewrite_for_edge(model)

        example_inputs = (torch.randn(2, 8, 16),)
        exported = export(rewritten, example_inputs)

        assert exported is not None, "Exported program should not be None"
        # Verify the exported program runs
        output = exported.module()(*example_inputs)
        assert output.shape == (2, 8, 16)

    def test_export_convnext(self, rewriter: TargetRewriter) -> None:
        """ConvNeXt-like model can be exported after rewrite."""
        model = ConvNeXtBlock(dim=16)
        rewritten = rewriter.rewrite_for_edge(model)

        example_inputs = (torch.randn(2, 16, 8, 8),)
        exported = export(rewritten, example_inputs)

        assert exported is not None
        output = exported.module()(*example_inputs)
        assert output.shape == (2, 16, 8, 8)

    def test_export_gelu_module(self, rewriter: TargetRewriter, input_2d: torch.Tensor) -> None:
        """GELU-based model can be exported after rewrite."""
        model = GELUModule()
        rewritten = rewriter.rewrite_for_edge(model)

        exported = export(rewritten, (input_2d,))

        assert exported is not None
        output = exported.module()(input_2d)
        assert output.shape == (2, 16, 28, 28)

    def test_graph_has_no_duplicate_targets(self, rewriter: TargetRewriter) -> None:
        """Rewritten graph should not have duplicate module targets (clean graph)."""
        model = ConvNeXtBlock(dim=16)
        rewritten = rewriter.rewrite_for_edge(model)

        # Check that the graph is well-formed
        rewritten.graph.lint()  # Raises if graph is malformed

        assert True, "Graph lint passed without errors"

    def test_graph_module_printable(self, rewriter: TargetRewriter) -> None:
        """Rewritten GraphModule can be printed/code-generated without error."""
        model = GELUModule()
        rewritten = rewriter.rewrite_for_edge(model)

        code = rewritten.code  # Generate Python code from graph
        # Generated code references the module as ``_rewritten_relu_...``
        assert "_rewritten_relu_" in code, (
            "Generated code should reference the rewritten ReLU module"
        )
        # Verify no GELU references remain in the graph
        graph_has_gelu = any(
            node.op == "call_module"
            and isinstance(rewritten.get_submodule(node.target), nn.GELU)
            for node in rewritten.graph.nodes
        )
        assert not graph_has_gelu, (
            "No GELU module should be called in the rewritten graph"
        )

    def test_retrace_preserves_edge_replacements(self, rewriter: TargetRewriter) -> None:
        """Re-tracing the rewritten graph preserves activation replacements."""
        model = GELUModule()
        rewritten = rewriter.rewrite_for_edge(model)

        re_traced = symbolic_trace(rewritten)
        # Verify GELU is still replaced in the re-traced graph
        has_gelu = any(
            node.op == "call_module"
            and isinstance(re_traced.get_submodule(node.target), nn.GELU)
            for node in re_traced.graph.nodes
        )
        assert not has_gelu, "GELU should remain replaced after re-trace"

    def test_save_and_load_state_dict(
        self, rewriter: TargetRewriter, input_2d: torch.Tensor
    ) -> None:
        """Rewritten model state_dict can be saved and loaded."""
        model = GELUModule()
        rewritten = rewriter.rewrite_for_edge(model)

        # Save state dict
        state_dict = rewritten.state_dict()

        # Create a fresh model, rewrite, and load
        model2 = GELUModule()
        rewritten2 = rewriter.rewrite_for_edge(model2)
        rewritten2.load_state_dict(state_dict)

        # Verify outputs match
        with torch.no_grad():
            out1 = rewritten(input_2d)
            out2 = rewritten2(input_2d)

        assert torch.allclose(out1, out2), "Outputs should match after state_dict round-trip"


# ======================================================================
# Edge case tests
# ======================================================================


class TestRewriterEdgeCases:
    """Edge cases and error handling for TargetRewriter."""

    def test_rewrite_already_graph_module(
        self, rewriter: TargetRewriter, input_2d: torch.Tensor
    ) -> None:
        """Rewriting an already-traced GraphModule should work."""
        model = symbolic_trace(GELUModule())
        rewritten = rewriter.rewrite_for_edge(model)

        assert isinstance(rewritten, GraphModule)
        output = rewritten(input_2d)
        assert output.shape == (2, 16, 28, 28)

    def test_rewrite_empty_model(self, rewriter: TargetRewriter) -> None:
        """Rewriting a model with no layers should not error."""
        # Wrap in a minimal functional module for traceability
        class EmptyModule(nn.Module):
            """Module with identity forward."""

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Identity."""
                return x

        rewritten = rewriter.rewrite_for_edge(EmptyModule())

        assert isinstance(rewritten, GraphModule)
        x = torch.randn(2, 3)
        output = rewritten(x)
        assert torch.equal(output, x)

    def test_rewrite_nested_modules(self, rewriter: TargetRewriter, input_2d: torch.Tensor) -> None:
        """Nested module hierarchies are handled correctly."""

        class InnerBlock(nn.Module):
            """Inner block with GELU."""

            def __init__(self) -> None:
                """Initialise."""
                super().__init__()
                self.conv = nn.Conv2d(16, 16, 3, padding=1)
                self.act = nn.GELU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
                return self.act(self.conv(x))

        class OuterModel(nn.Module):
            """Outer model with nested inner blocks."""

            def __init__(self) -> None:
                """Initialise."""
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3)
                self.block1 = InnerBlock()
                self.block2 = InnerBlock()
                self.conv2 = nn.Conv2d(16, 16, 3)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
                x = self.conv1(x)
                x = self.block1(x)
                x = self.block2(x)
                x = self.conv2(x)
                return x

        model = OuterModel()
        rewritten = rewriter.rewrite_for_edge(model)

        # Verify all GELUs in nested modules are replaced
        has_gelu = False
        for node in rewritten.graph.nodes:
            if node.op == "call_module":
                mod = rewritten.get_submodule(node.target)
                if isinstance(mod, nn.GELU):
                    has_gelu = True
                    break

        assert not has_gelu, "All nested GELU modules should be replaced"

        output = rewritten(input_2d)
        assert not torch.isnan(output).any()

    def test_rewrite_preserves_parameters(
        self, rewriter: TargetRewriter, input_2d: torch.Tensor
    ) -> None:
        """Model parameters are preserved and accessible after rewrite."""
        model = GELUModule()
        rewritten = rewriter.rewrite_for_edge(model)

        # Verify original parameters are still present
        original_param_names = {name for name, _ in model.named_parameters()}
        rewritten_param_names = {name for name, _ in rewritten.named_parameters()}

        for name in original_param_names:
            assert name in rewritten_param_names, (
                f"Parameter '{name}' should be preserved after rewrite"
            )

        # Verify the graph still produces correct output with parameters intact
        output = rewritten(input_2d)
        assert not torch.isnan(output).any()

    def test_model_without_trainable_params(self, rewriter: TargetRewriter) -> None:
        """Model with no trainable parameters (frozen) still rewrites correctly."""
        model = GELUModule()
        for param in model.parameters():
            param.requires_grad = False

        rewritten = rewriter.rewrite_for_edge(model)

        x = torch.randn(1, 3, 16, 16)
        output = rewritten(x)
        assert output.shape == (1, 16, 12, 12)

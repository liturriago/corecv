"""Comprehensive tests for the registry module (CoreRegistry and decorators).

Tests cover basic registry operations, signature validation, decorator
registration, convenience getters/listers, and error handling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from corecv.core import (
    BACKBONE_REGISTRY,
    HEAD_REGISTRY,
    LOSS_REGISTRY,
    NECK_REGISTRY,
    CoreRegistry,
    get_backbone,
    get_head,
    get_loss,
    get_neck,
    list_backbones,
    list_heads,
    list_losses,
    list_necks,
    register_backbone,
    register_head,
    register_loss,
    register_neck,
)

# ======================================================================
# Helper classes / functions used across tests
# ======================================================================


class DummyBackbone(torch.nn.Module):
    """Dummy backbone used for registration tests."""

    def __init__(self, in_channels: int = 3, width_mult: float = 1.0) -> None:
        """Init dummy backbone.

        Args:
            in_channels: Number of input channels.
            width_mult: Width multiplier.
        """
        super().__init__()
        self.in_channels = in_channels
        self.width_mult = width_mult


class DummyNeck(torch.nn.Module):
    """Dummy neck used for registration tests."""

    def __init__(self, in_channels: int = 64) -> None:
        """Init dummy neck.

        Args:
            in_channels: Number of input channels.
        """
        super().__init__()
        self.in_channels = in_channels


class DummyHead(torch.nn.Module):
    """Dummy head used for registration tests."""

    def __init__(self, num_classes: int = 10) -> None:
        """Init dummy head.

        Args:
            num_classes: Number of output classes.
        """
        super().__init__()
        self.num_classes = num_classes


class ModelWithRequiredArg(torch.nn.Module):
    """Dummy model with one required positional argument for signature validation tests."""

    def __init__(self, required_arg: int) -> None:
        """Init.

        Args:
            required_arg: A required argument with no default.
        """
        super().__init__()
        self.required_arg = required_arg


class ModelWithMultipleRequired(torch.nn.Module):
    """Dummy model with multiple required arguments."""

    def __init__(self, a: int, b: str) -> None:
        """Init.

        Args:
            a: First required argument.
            b: Second required argument.
        """
        super().__init__()
        self.a = a
        self.b = b


def dummy_loss_fn(
    _pred: torch.Tensor,
    _target: torch.Tensor,
    _reduction: str = "mean",
) -> torch.Tensor:
    """Dummy loss function returning zero tensor.

    Args:
        _pred: Predicted tensor.
        _target: Target tensor.
        _reduction: Reduction method.

    Returns:
        Zero tensor.
    """
    return torch.tensor(0.0, requires_grad=True)


# ======================================================================
# Tests
# ======================================================================


class TestCoreRegistry:
    """CoreRegistry basic operations: register, get, list, contains."""

    def test_register_class_and_get(self) -> None:
        """Register a class and retrieve it by name."""
        registry: CoreRegistry[torch.nn.Module] = CoreRegistry()
        registry.register("DummyBackbone", DummyBackbone)
        retrieved: type[torch.nn.Module] | Callable[..., torch.nn.Module] = (
            registry.get("DummyBackbone")
        )
        assert retrieved is DummyBackbone

    def test_register_function_and_get(self) -> None:
        """Register a function and retrieve it by name."""
        registry: CoreRegistry[Callable[..., torch.Tensor]] = CoreRegistry()

        def my_fn() -> torch.Tensor:
            return torch.tensor(0.0)

        registry.register("my_fn", my_fn)
        retrieved: type[Callable[..., torch.Tensor]] | Callable[
            ...,
            torch.Tensor,
        ] = registry.get("my_fn")
        assert retrieved is my_fn

    def test_list_returns_sorted_names(self) -> None:
        """``list()`` returns names in alphabetical order."""
        registry: CoreRegistry[Any] = CoreRegistry()
        registry.register("zebra", lambda: None)
        registry.register("alpha", lambda: None)
        registry.register("beta", lambda: None)

        names: list[str] = registry.list()
        assert names == ["alpha", "beta", "zebra"]

    def test_contains_returns_true_for_registered(self) -> None:
        """``contains(name)`` returns ``True`` for a registered name."""
        registry: CoreRegistry[Any] = CoreRegistry()
        registry.register("DummyBackbone", DummyBackbone)
        assert registry.contains("DummyBackbone") is True

    def test_contains_returns_false_for_unregistered(self) -> None:
        """``contains(name)`` returns ``False`` for an unregistered name."""
        registry: CoreRegistry[Any] = CoreRegistry()
        assert registry.contains("NonExistent") is False

    def test_list_empty_registry(self) -> None:
        """``list()`` on an empty registry returns an empty list."""
        registry: CoreRegistry[Any] = CoreRegistry()
        assert registry.list() == []

    def test_get_returns_same_object_after_multiple_registrations(
        self,
    ) -> None:
        """Multiple registrations of different names each return correctly."""
        registry: CoreRegistry[Any] = CoreRegistry()
        registry.register("a", DummyBackbone)
        registry.register("b", DummyNeck)
        assert registry.get("a") is DummyBackbone
        assert registry.get("b") is DummyNeck


class TestSignatureValidation:
    """``validate_signature`` correctness for valid/invalid arguments."""

    def test_valid_args_kwargs(self) -> None:
        """Valid positional and keyword args return ``True``."""
        registry: CoreRegistry[torch.nn.Module] = CoreRegistry()
        registry.register("DummyBackbone", DummyBackbone)

        result: bool = registry.validate_signature(
            "DummyBackbone", 3, width_mult=1.5
        )
        assert result is True

    def test_valid_only_defaults(self) -> None:
        """Omitting optional arguments (which have defaults) is valid."""
        registry: CoreRegistry[torch.nn.Module] = CoreRegistry()
        registry.register("DummyBackbone", DummyBackbone)

        result: bool = registry.validate_signature("DummyBackbone")
        assert result is True

    def test_missing_required_arg_raises_type_error(self) -> None:
        """Missing a required positional argument raises ``TypeError``."""
        registry: CoreRegistry[torch.nn.Module] = CoreRegistry()
        registry.register("ModelWithRequiredArg", ModelWithRequiredArg)

        with pytest.raises(TypeError, match="required_arg"):
            registry.validate_signature("ModelWithRequiredArg")

    def test_missing_multiple_required_args(self) -> None:
        """Missing multiple required arguments raises ``TypeError``."""
        registry: CoreRegistry[torch.nn.Module] = CoreRegistry()
        registry.register("ModelWithMultipleRequired", ModelWithMultipleRequired)

        with pytest.raises(TypeError):
            registry.validate_signature("ModelWithMultipleRequired")

    def test_extra_unexpected_kwarg_raises_type_error(self) -> None:
        """An unexpected keyword argument raises ``TypeError``."""
        registry: CoreRegistry[torch.nn.Module] = CoreRegistry()
        registry.register("DummyBackbone", DummyBackbone)

        with pytest.raises(TypeError, match="unexpected"):
            registry.validate_signature(
                "DummyBackbone", in_channels=3, nonexistent_arg=42
            )

    def test_validate_function_signature(self) -> None:
        """``validate_signature`` works for functions, not just classes."""
        registry: CoreRegistry[Callable[..., torch.Tensor]] = CoreRegistry()
        registry.register("dummy_loss_fn", dummy_loss_fn)

        result: bool = registry.validate_signature(
            "dummy_loss_fn",
            torch.randn(4, 10),
            torch.randint(0, 10, (4,)),
            _reduction="sum",
        )
        assert result is True

    def test_unregistered_name_raises_key_error(self) -> None:
        """Validating against an unregistered name raises ``KeyError``."""
        registry: CoreRegistry[Any] = CoreRegistry()
        with pytest.raises(KeyError):
            registry.validate_signature("NonExistent")

    def test_error_message_includes_signature(self) -> None:
        """Error message includes the expected signature and provided args."""
        registry: CoreRegistry[torch.nn.Module] = CoreRegistry()
        registry.register("ModelWithRequiredArg", ModelWithRequiredArg)

        with pytest.raises(TypeError) as exc_info:
            registry.validate_signature("ModelWithRequiredArg")
        msg: str = str(exc_info.value)
        assert "Signature mismatch" in msg
        assert "ModelWithRequiredArg" in msg


class TestDecorators:
    """Decorator registration via ``@register_*``."""

    def test_register_backbone_decorator_class(self) -> None:
        r"""``@register_backbone()`` on a class registers it in ``BACKBONE_REGISTRY``."""

        @register_backbone()
        class MyBackbone(torch.nn.Module):
            """Custom backbone."""

            def __init__(self) -> None:
                super().__init__()

        assert BACKBONE_REGISTRY.contains("MyBackbone")
        assert BACKBONE_REGISTRY.get("MyBackbone") is MyBackbone

    def test_register_neck_with_custom_name(self) -> None:
        r"""``@register_neck("custom_name")`` registers under that name."""

        @register_neck("custom_neck")
        class MyNeck(torch.nn.Module):
            """Custom neck."""

            def __init__(self) -> None:
                super().__init__()

        assert NECK_REGISTRY.contains("custom_neck")
        assert NECK_REGISTRY.get("custom_neck") is MyNeck
        # Should NOT be accessible by class __name__
        assert not NECK_REGISTRY.contains("MyNeck")

    def test_register_head_decorator_function(self) -> None:
        r"""``@register_head()`` on a function registers it in ``HEAD_REGISTRY``."""

        @register_head()
        def my_head_fn(x: torch.Tensor) -> torch.Tensor:
            """Dummy head function."""
            return x

        assert HEAD_REGISTRY.contains("my_head_fn")
        assert HEAD_REGISTRY.get("my_head_fn") is my_head_fn

    def test_register_loss_decorator(self) -> None:
        """``@register_loss()`` registers in ``LOSS_REGISTRY``."""

        @register_loss()
        def my_loss_fn(
            _pred: torch.Tensor,
            _target: torch.Tensor,
        ) -> torch.Tensor:
            """Dummy loss function."""
            return torch.tensor(0.0)

        assert LOSS_REGISTRY.contains("my_loss_fn")
        assert LOSS_REGISTRY.get("my_loss_fn") is my_loss_fn

    def test_register_backbone_explicit_name(self) -> None:
        r"""``@register_backbone("explicit_name")`` uses the given name."""

        @register_backbone("explicit_backbone")
        class AnotherBackbone(torch.nn.Module):
            """Another dummy backbone."""

            def __init__(self) -> None:
                super().__init__()

        assert BACKBONE_REGISTRY.contains("explicit_backbone")
        assert BACKBONE_REGISTRY.get("explicit_backbone") is AnotherBackbone
        assert not BACKBONE_REGISTRY.contains("AnotherBackbone")

    def test_decorator_returns_original_object(self) -> None:
        """The decorator returns the original object unchanged."""

        @register_backbone()
        class SomeBackbone(torch.nn.Module):
            """Yet another backbone."""

            def __init__(self) -> None:
                super().__init__()

        assert SomeBackbone is BACKBONE_REGISTRY.get("SomeBackbone")

    def test_head_decorator_keeps_callable(self) -> None:
        """The decorated function remains callable."""

        @register_head()
        def simple_head(x: torch.Tensor) -> torch.Tensor:
            """Simple head.

            Args:
                x: Input tensor.

            Returns:
                Same tensor.
            """
            return x

        t: torch.Tensor = torch.randn(4, 64)
        result: torch.Tensor = simple_head(t)
        assert torch.equal(result, t)


class TestConvenienceFunctions:
    """Global convenience getters and listers."""

    def setup_method(self) -> None:
        """Register known objects before each test (idempotent)."""
        # Global registries persist across tests, so skip if already registered.
        registrations: list[tuple[CoreRegistry[Any], str, Any]] = [
            (BACKBONE_REGISTRY, "conv_test_backbone", DummyBackbone),
            (NECK_REGISTRY, "conv_test_neck", DummyNeck),
            (HEAD_REGISTRY, "conv_test_head", DummyHead),
            (LOSS_REGISTRY, "conv_test_loss", dummy_loss_fn),
        ]
        for reg, name, obj in registrations:
            if not reg.contains(name):
                reg.register(name, obj)

    def test_get_backbone(self) -> None:
        """``get_backbone(name)`` retrieves the registered backbone."""
        assert get_backbone("conv_test_backbone") is DummyBackbone

    def test_get_neck(self) -> None:
        """``get_neck(name)`` retrieves the registered neck."""
        assert get_neck("conv_test_neck") is DummyNeck

    def test_get_head(self) -> None:
        """``get_head(name)`` retrieves the registered head."""
        assert get_head("conv_test_head") is DummyHead

    def test_get_loss(self) -> None:
        """``get_loss(name)`` retrieves the registered loss."""
        assert get_loss("conv_test_loss") is dummy_loss_fn

    def test_list_backbones(self) -> None:
        """``list_backbones()`` returns a list containing registered names."""
        names: list[str] = list_backbones()
        assert "conv_test_backbone" in names
        assert isinstance(names, list)

    def test_list_necks(self) -> None:
        """``list_necks()`` returns a list containing registered names."""
        names: list[str] = list_necks()
        assert "conv_test_neck" in names
        assert isinstance(names, list)

    def test_list_heads(self) -> None:
        """``list_heads()`` returns a list containing registered names."""
        names: list[str] = list_heads()
        assert "conv_test_head" in names
        assert isinstance(names, list)

    def test_list_losses(self) -> None:
        """``list_losses()`` returns a list containing registered names."""
        names: list[str] = list_losses()
        assert "conv_test_loss" in names
        assert isinstance(names, list)

    def test_list_backbones_sorted(self) -> None:
        """Listed backbone names are alphabetically sorted."""
        names: list[str] = list_backbones()
        assert names == sorted(names)

    def test_list_necks_sorted(self) -> None:
        """Listed neck names are alphabetically sorted."""
        names: list[str] = list_necks()
        assert names == sorted(names)


class TestRegistryErrorHandling:
    """Error handling for registry operations."""

    def test_get_nonexistent_raises_key_error(self) -> None:
        r"""``get("nonexistent")`` raises ``KeyError``."""
        registry: CoreRegistry[Any] = CoreRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            registry.get("nonexistent")

    def test_get_nonexistent_backbone_raises_key_error(self) -> None:
        r"""``get_backbone("nonexistent")`` raises ``KeyError``."""
        with pytest.raises(KeyError, match="nonexistent"):
            get_backbone("nonexistent")

    def test_get_nonexistent_neck_raises_key_error(self) -> None:
        r"""``get_neck("nonexistent")`` raises ``KeyError``."""
        with pytest.raises(KeyError, match="nonexistent"):
            get_neck("nonexistent")

    def test_get_nonexistent_head_raises_key_error(self) -> None:
        r"""``get_head("nonexistent")`` raises ``KeyError``."""
        with pytest.raises(KeyError, match="nonexistent"):
            get_head("nonexistent")

    def test_get_nonexistent_loss_raises_key_error(self) -> None:
        r"""``get_loss("nonexistent")`` raises ``KeyError``."""
        with pytest.raises(KeyError, match="nonexistent"):
            get_loss("nonexistent")

    def test_duplicate_registration_raises_value_error(self) -> None:
        """Registering the same name twice raises ``ValueError``."""
        registry: CoreRegistry[Any] = CoreRegistry()

        def fn_a() -> None:
            pass

        def fn_b() -> None:
            pass

        registry.register("dup_name", fn_a)
        with pytest.raises(ValueError, match="dup_name"):
            registry.register("dup_name", fn_b)

    def test_duplicate_registration_global_registry(
        self,
    ) -> None:
        r"""Duplicate registration on a global registry also raises ``ValueError``."""
        # Use a unique name to avoid affecting other tests
        unique_name: str = "dup_global_test_unique"

        @register_backbone(unique_name)
        class _FirstDup(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

        with pytest.raises(ValueError, match=unique_name):

            @register_backbone(unique_name)
            class _SecondDup(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()

    def test_error_message_includes_registered_names(self) -> None:
        """KeyError message includes available names."""
        registry: CoreRegistry[Any] = CoreRegistry()
        registry.register("alpha", lambda: None)
        with pytest.raises(KeyError) as exc_info:
            registry.get("beta")
        msg: str = str(exc_info.value)
        assert "alpha" in msg
        assert "beta" in msg

    def test_error_message_on_duplicate_includes_registry_type(
        self,
    ) -> None:
        """ValueError on duplicate includes the registry class name."""
        registry: CoreRegistry[Any] = CoreRegistry()
        registry.register("item", lambda: None)
        with pytest.raises(ValueError) as exc_info:
            registry.register("item", lambda: None)
        msg: str = str(exc_info.value)
        assert "CoreRegistry" in msg


class TestRegistryEdgeCases:
    """Edge cases for registry operations."""

    def test_register_class_with_constructor_args(self) -> None:
        r"""Register a class that expects constructor args and verify ``validate_signature``."""
        registry: CoreRegistry[torch.nn.Module] = CoreRegistry()
        registry.register("DummyHead", DummyHead)
        assert registry.validate_signature("DummyHead", num_classes=21) is True

    def test_register_lambda(self) -> None:
        """A lambda can be registered and retrieved."""

        def _noop() -> None:
            pass

        registry: CoreRegistry[Callable[[], None]] = CoreRegistry()
        registry.register("_noop", _noop)
        assert registry.get("_noop") is _noop

    def test_contains_on_empty_registry(self) -> None:
        r"""``contains()`` returns ``False`` for any name on an empty registry."""
        registry: CoreRegistry[Any] = CoreRegistry()
        assert registry.contains("anything") is False

    def test_list_does_not_modify_registry(self) -> None:
        """Calling ``list()`` does not alter the registry contents."""
        registry: CoreRegistry[Any] = CoreRegistry()
        registry.register("a", lambda: None)
        registry.list()
        registry.list()
        assert registry.contains("a") is True
        assert len(registry.list()) == 1

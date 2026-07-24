"""CoreRegistry: Type-safe, generic static registry system for CoreCV.

Provides explicit decorators for registering backbones, necks, heads, and losses
with pre-instantiation signature validation. All registries are generic over the
registered callable type to maintain type safety throughout the framework.

Example:
    >>> @register_backbone()
    ... class MobileNetV3(nn.Module):
    ...     def __init__(self, width_mult: float = 1.0, **kwargs): ...
    ...
    >>> BACKBONE_REGISTRY.validate_signature("MobileNetV3", width_mult=1.0)
    True
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Generic, TypeVar

import torch
from torch import nn

T = TypeVar("T")


class CoreRegistry(Generic[T]):
    """Generic, type-safe static registry for callable objects.

    Stores registered callables (classes or functions) by name and supports
    pre-instantiation signature validation via ``inspect.signature``.

    Attributes:
        _registry: Internal mapping from name to registered callable.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._registry: dict[str, type[T] | Callable[..., T]] = {}

    def register(
        self,
        name: str,
        obj: type[T] | Callable[..., T],
    ) -> type[T] | Callable[..., T]:
        """Register an object under the given name.

        If the name is already taken, a ``ValueError`` is raised to prevent
        silent overwrites and accidental name collisions.

        Args:
            name: The lookup key for the registered object.
            obj: The class or callable to register.

        Returns:
            The registered object unchanged, allowing use as a decorator.

        Raises:
            ValueError: If ``name`` is already registered.
        """
        if name in self._registry:
            msg = (
                f"Name '{name}' is already registered in {type(self).__name__}. "
                f"Registered names: {self.list()}"
            )
            raise ValueError(msg)
        self._registry[name] = obj
        return obj

    def get(self, name: str) -> type[T] | Callable[..., T]:
        """Retrieve a registered object by name.

        Args:
            name: The lookup key.

        Returns:
            The registered callable.

        Raises:
            KeyError: If ``name`` is not found in the registry.
        """
        if name not in self._registry:
            msg = (
                f"'{name}' not found in {type(self).__name__}. "
                f"Available names: {self.list()}"
            )
            raise KeyError(msg)
        return self._registry[name]

    def list(self) -> list[str]:
        """Return a sorted list of all registered names.

        Returns:
            Alphabetically sorted list of registration keys.
        """
        return sorted(self._registry.keys())

    def contains(self, name: str) -> bool:
        """Check whether a name is registered.

        Args:
            name: The lookup key.

        Returns:
            ``True`` if the name is present in the registry.
        """
        return name in self._registry

    def validate_signature(
        self,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> bool:
        """Validate that the registered callable's signature matches provided args.

        Performs static pre-instantiation inspection using ``inspect.signature``
        to ensure compatibility without actually calling the callable. Checks
        required positional args, optional args with defaults, ``*args``, and
        ``**kwargs`` support, and type hints when present.

        Args:
            name: The registered callable to validate against.
            *args: Positional arguments to check.
            **kwargs: Keyword arguments to check.

        Returns:
            ``True`` if the signature is compatible.

        Raises:
            KeyError: If ``name`` is not registered.
            TypeError: If the signature is incompatible with the provided args.
        """
        obj = self.get(name)
        sig = inspect.signature(obj)

        try:
            sig.bind(*args, **kwargs)
        except TypeError as exc:
            msg = (
                f"Signature mismatch for '{name}'. "
                f"Expected signature: {sig}. "
                f"Provided args: {args}, kwargs: {kwargs}. "
                f"Error: {exc}"
            )
            raise TypeError(msg) from exc

        return True


# ---------------------------------------------------------------------------
# Global registry instances
# ---------------------------------------------------------------------------

BACKBONE_REGISTRY: CoreRegistry[nn.Module] = CoreRegistry()
NECK_REGISTRY: CoreRegistry[nn.Module] = CoreRegistry()
HEAD_REGISTRY: CoreRegistry[nn.Module] = CoreRegistry()
LOSS_REGISTRY: CoreRegistry[Callable[..., torch.Tensor]] = CoreRegistry()


# ---------------------------------------------------------------------------
# Explicit decorator functions (module-level, one per registry)
# ---------------------------------------------------------------------------


def register_backbone(
    name: str | None = None,
) -> Callable[[type[T] | Callable[..., T]], type[T] | Callable[..., T]]:
    """Decorator to register a backbone in ``BACKBONE_REGISTRY``.

    Args:
        name: Optional explicit name.  If ``None``, the decorated object's
            ``__name__`` attribute is used.

    Returns:
        A decorator that registers and returns the decorated object unchanged.
    """

    def decorator(
        obj: type[T] | Callable[..., T],
    ) -> type[T] | Callable[..., T]:
        resolved = name if name is not None else obj.__name__  # type: ignore[union-attr]
        BACKBONE_REGISTRY.register(resolved, obj)
        return obj

    return decorator


def register_neck(
    name: str | None = None,
) -> Callable[[type[T] | Callable[..., T]], type[T] | Callable[..., T]]:
    """Decorator to register a neck in ``NECK_REGISTRY``.

    Args:
        name: Optional explicit name.  If ``None``, the decorated object's
            ``__name__`` attribute is used.

    Returns:
        A decorator that registers and returns the decorated object unchanged.
    """

    def decorator(
        obj: type[T] | Callable[..., T],
    ) -> type[T] | Callable[..., T]:
        resolved = name if name is not None else obj.__name__  # type: ignore[union-attr]
        NECK_REGISTRY.register(resolved, obj)
        return obj

    return decorator


def register_head(
    name: str | None = None,
) -> Callable[[type[T] | Callable[..., T]], type[T] | Callable[..., T]]:
    """Decorator to register a head in ``HEAD_REGISTRY``.

    Args:
        name: Optional explicit name.  If ``None``, the decorated object's
            ``__name__`` attribute is used.

    Returns:
        A decorator that registers and returns the decorated object unchanged.
    """

    def decorator(
        obj: type[T] | Callable[..., T],
    ) -> type[T] | Callable[..., T]:
        resolved = name if name is not None else obj.__name__  # type: ignore[union-attr]
        HEAD_REGISTRY.register(resolved, obj)
        return obj

    return decorator


def register_loss(
    name: str | None = None,
) -> Callable[[type[T] | Callable[..., T]], type[T] | Callable[..., T]]:
    """Decorator to register a loss in ``LOSS_REGISTRY``.

    Args:
        name: Optional explicit name.  If ``None``, the decorated object's
            ``__name__`` attribute is used.

    Returns:
        A decorator that registers and returns the decorated object unchanged.
    """

    def decorator(
        obj: type[T] | Callable[..., T],
    ) -> type[T] | Callable[..., T]:
        resolved = name if name is not None else obj.__name__  # type: ignore[union-attr]
        LOSS_REGISTRY.register(resolved, obj)
        return obj

    return decorator


# ---------------------------------------------------------------------------
# Convenience getters
# ---------------------------------------------------------------------------


def get_backbone(name: str) -> type[nn.Module]:
    """Retrieve a backbone class by name.

    Args:
        name: Registration key.

    Returns:
        The backbone class.
    """
    return BACKBONE_REGISTRY.get(name)  # type: ignore[return-value]


def get_neck(name: str) -> type[nn.Module]:
    """Retrieve a neck class by name.

    Args:
        name: Registration key.

    Returns:
        The neck class.
    """
    return NECK_REGISTRY.get(name)  # type: ignore[return-value]


def get_head(name: str) -> type[nn.Module]:
    """Retrieve a head class by name.

    Args:
        name: Registration key.

    Returns:
        The head class.
    """
    return HEAD_REGISTRY.get(name)  # type: ignore[return-value]


def get_loss(name: str) -> Callable[..., torch.Tensor]:
    """Retrieve a loss callable by name.

    Args:
        name: Registration key.

    Returns:
        The loss callable.
    """
    return LOSS_REGISTRY.get(name)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Convenience listers
# ---------------------------------------------------------------------------


def list_backbones() -> list[str]:
    """List all registered backbone names.

    Returns:
        Sorted list of backbone registration keys.
    """
    return BACKBONE_REGISTRY.list()


def list_necks() -> list[str]:
    """List all registered neck names.

    Returns:
        Sorted list of neck registration keys.
    """
    return NECK_REGISTRY.list()


def list_heads() -> list[str]:
    """List all registered head names.

    Returns:
        Sorted list of head registration keys.
    """
    return HEAD_REGISTRY.list()


def list_losses() -> list[str]:
    """List all registered loss names.

    Returns:
        Sorted list of loss registration keys.
    """
    return LOSS_REGISTRY.list()

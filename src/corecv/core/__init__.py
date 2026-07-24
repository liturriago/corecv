"""Core module for CoreCV.

Provides the foundational registry system, base backbone interface with
FeatureInfo metadata, and common utilities used across the framework.

Example:
    >>> from corecv.core import CoreRegistry, BACKBONE_REGISTRY
    >>> from corecv.core import register_backbone, get_backbone
"""

from corecv.core.contract import BaseBackbone, FeatureInfo
from corecv.core.registry import (
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

__all__ = [
    # Contract: backbone interface and feature metadata
    "FeatureInfo",
    "BaseBackbone",
    # Registry class
    "CoreRegistry",
    # Global registry instances
    "BACKBONE_REGISTRY",
    "NECK_REGISTRY",
    "HEAD_REGISTRY",
    "LOSS_REGISTRY",
    # Explicit decorators
    "register_backbone",
    "register_neck",
    "register_head",
    "register_loss",
    # Convenience getters
    "get_backbone",
    "get_neck",
    "get_head",
    "get_loss",
    # Convenience listers
    "list_backbones",
    "list_necks",
    "list_heads",
    "list_losses",
]

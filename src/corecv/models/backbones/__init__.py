"""Backbones subpackage for CoreCV.

This subpackage contains TorchVision-based backbone encoders that extend
``BaseBackbone`` and expose ``FeatureInfo`` metadata for downstream heads
and necks. Supported architectures:

- **ResNet**: ResNet-18/34/50/101/152 variants.
- **MobileNetV3**: Lightweight mobile-friendly backbones.
- **ConvNeXt**: Modernized ConvNet architectures.
- **Swin Transformer**: Hierarchical vision transformer.
- **CSP Pyramid**: From-scratch hierarchical CNN with CSP blocks, spatial
  pyramid pooling, and positional self-attention.
"""

from __future__ import annotations

from typing import Literal

from corecv.models.backbones.base import BaseBackbone, FeatureInfo
from corecv.models.backbones.convnext import ConvNeXtBackbone
from corecv.models.backbones.csp_pyramid import CSPPyramidBackbone
from corecv.models.backbones.mobilenetv3 import MobileNetV3Backbone
from corecv.models.backbones.resnet import ResNetBackbone
from corecv.models.backbones.swin import SwinTransformerBackbone

# Union of all supported backbone variant names.
BackboneName = Literal[
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "mobilenetv3_large",
    "mobilenetv3_small",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "swin_t",
    "swin_s",
    "swin_b",
    "csp_nano",
    "csp_small",
    "csp_medium",
    "csp_large",
    "csp_xlarge",
]

# Registry mapping variant name -> backbone class.
_BACKBONE_REGISTRY: dict[str, type[BaseBackbone]] = {
    "resnet18": ResNetBackbone,
    "resnet34": ResNetBackbone,
    "resnet50": ResNetBackbone,
    "resnet101": ResNetBackbone,
    "resnet152": ResNetBackbone,
    "mobilenetv3_large": MobileNetV3Backbone,
    "mobilenetv3_small": MobileNetV3Backbone,
    "convnext_tiny": ConvNeXtBackbone,
    "convnext_small": ConvNeXtBackbone,
    "convnext_base": ConvNeXtBackbone,
    "swin_t": SwinTransformerBackbone,
    "swin_s": SwinTransformerBackbone,
    "swin_b": SwinTransformerBackbone,
    "csp_nano": CSPPyramidBackbone,
    "csp_small": CSPPyramidBackbone,
    "csp_medium": CSPPyramidBackbone,
    "csp_large": CSPPyramidBackbone,
    "csp_xlarge": CSPPyramidBackbone,
}


def create_backbone(
    name: BackboneName,
    *,
    pretrained: bool = False,
) -> BaseBackbone:
    """Create a backbone by name.

    Args:
        name: Backbone variant name (e.g., ``resnet50``, ``swin_t``).
        pretrained: If ``True``, load ImageNet-1K pretrained weights.

    Returns:
        An instantiated :class:`BaseBackbone` subclass.

    Raises:
        ValueError: If *name* is not a recognized backbone variant.

    Example:
        >>> from corecv.models.backbones import create_backbone
        >>> backbone = create_backbone("resnet50", pretrained=True)
        >>> backbone.feature_info.channels
        [256, 512, 1024, 2048]
    """
    if name not in _BACKBONE_REGISTRY:
        msg = f"Unknown backbone: {name!r}. Choose from {list(_BACKBONE_REGISTRY)}"
        raise ValueError(msg)

    backbone_cls = _BACKBONE_REGISTRY[name]

    # All backbone families accept (variant, pretrained).
    return backbone_cls(name, pretrained=pretrained)  # type: ignore[call-arg]


__all__ = [
    "BackboneName",
    "BaseBackbone",
    "CSPPyramidBackbone",
    "ConvNeXtBackbone",
    "FeatureInfo",
    "MobileNetV3Backbone",
    "ResNetBackbone",
    "SwinTransformerBackbone",
    "create_backbone",
]

"""Contract module defining the backbone interface and feature metadata.

This module establishes the foundational contract that all backbones
(e.g. MobileNetV3, ResNet, ConvNeXt, ViT) must adhere to. It provides:

- :class:`FeatureInfo`: An immutable, frozen dataclass describing the channel
  counts and strides of a backbone's multi-scale feature maps.
- :class:`BaseBackbone`: An abstract base class extending
  :class:`torch.nn.Module` that enforces implementation of the
  ``feature_info`` property across all concrete backbone subclasses.

Example:
    >>> class MyBackbone(BaseBackbone):
    ...     @property
    ...     def feature_info(self) -> FeatureInfo:
    ...         return FeatureInfo(
    ...             channels={"res2": 256, "res3": 512, "res4": 1024},
    ...             strides={"res2": 4, "res3": 8, "res4": 16},
    ...         )
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict

import torch.nn as nn


@dataclass(frozen=True)
class FeatureInfo:
    """Immutable metadata describing a backbone's multi-scale feature maps.

    Each feature level (e.g. ``"res2"``, ``"res3"``, ``"res4"``) maps to a
    channel count and a stride relative to the input image. This information
    is consumed by necks, heads, and graph-level tooling to wire up
    downstream components correctly.

    Attributes:
        channels: Mapping from feature level name to channel count.
        strides: Mapping from feature level name to stride.
    """

    channels: Dict[str, int]
    strides: Dict[str, int]


class BaseBackbone(nn.Module, abc.ABC):
    """Abstract base class for all vision backbones in CoreCV.

    Every concrete backbone (MobileNetV3, ResNet, ConvNeXt, ViT, etc.) must
    subclass :class:`BaseBackbone` and implement the :attr:`feature_info`
    property, which exposes the channel counts and strides of the backbone's
    intermediate feature maps.

    Subclasses are also expected to implement the standard
    :meth:`torch.nn.Module.forward` method to define the forward pass.

    Example:
        >>> class MyBackbone(BaseBackbone):
        ...     @property
        ...     def feature_info(self) -> FeatureInfo:
        ...         return FeatureInfo(
        ...             channels={"res2": 256, "res3": 512},
        ...             strides={"res2": 4, "res3": 8},
        ...         )
        ...
        >>> backbone = MyBackbone()
        >>> backbone.feature_info.channels["res2"]
        256
    """

    @property
    @abc.abstractmethod
    def feature_info(self) -> FeatureInfo:
        """Return the feature metadata for this backbone.

        Concrete backbones must implement this property to expose the
        channel counts and strides of their intermediate feature maps.

        Returns:
            A :class:`FeatureInfo` instance describing the channels and
            strides at each feature level.
        """
        ...

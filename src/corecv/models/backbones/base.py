"""Base backbone and FeatureInfo metadata for CoreCV.

Provides the abstract ``BaseBackbone`` class and ``FeatureInfo`` dataclass
that all CoreCV backbone encoders must implement.  ``FeatureInfo`` carries
channel counts, strides, and level names so downstream necks and heads can
dynamically configure themselves without hardcoded assumptions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class FeatureInfo:
    """Metadata describing multi-scale feature maps produced by a backbone.

    Attributes:
        channels: Number of channels at each feature level, ordered from
            finest (highest resolution) to coarsest.
        strides: Downsampling factor relative to the input image at each
            level (e.g., ``[4, 8, 16, 32]``).
        names: Human-readable names for each level (e.g.,
            ``["C2", "C3", "C4", "C5"]``).
    """

    channels: list[int]
    strides: list[int] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    @property
    def num_levels(self) -> int:
        """Return the number of feature levels."""
        return len(self.channels)


class BaseBackbone(nn.Module, ABC):
    """Abstract base class for all CoreCV backbone encoders.

    Every concrete backbone must:

    1. Call ``super().__init__(feature_info=...)`` with a populated
       :class:`FeatureInfo` instance.
    2. Implement ``forward(x) -> (features, feature_info)``.

    The ``feature_info`` attribute is frozen after ``__init__`` and is
    consumed by necks and heads to auto-configure their input dimensions.
    """

    feature_info: FeatureInfo

    def __init__(self, *, feature_info: FeatureInfo) -> None:
        """Initialize the base backbone.

        Args:
            feature_info: Metadata about the multi-scale features this
                backbone produces.
        """
        super().__init__()
        self.feature_info = feature_info

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[list[Tensor], FeatureInfo]:
        """Extract multi-scale features from the input tensor.

        Args:
            x: Input image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Tuple of ``(features, feature_info)`` where *features* is a
            list of tensors ordered from finest to coarsest resolution,
            and *feature_info* carries channel/stride metadata.
        """

# noqa: INP001
"""Semantic segmentation losses: Cross-Entropy, Dice, and Focal.

Provides three loss modules for semantic segmentation tasks where model
outputs are logits of shape ``[B, num_classes, H, W]`` and targets are
class-index masks of shape ``[B, H, W]`` with ``dtype=torch.long``.

Typical usage::

    ce_loss = SegmentationCrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss(num_classes=21)
    focal_loss = SegmentationFocalLoss(num_classes=21)

    loss = ce_loss(logits, masks) + dice_loss(logits, masks)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class SegmentationCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for semantic segmentation.

    Wraps ``torch.nn.functional.cross_entropy`` for pixel-wise classification
    with optional class weighting and void-pixel ignoring.
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = 255,
    ) -> None:
        """Initialize the segmentation cross-entropy loss.

        Args:
            weight: Optional class weights tensor of shape ``(num_classes,)``
                for balancing class frequencies.
            ignore_index: Class index to ignore (void pixels).
        """
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: Raw model output of shape ``(B, num_classes, H, W)``.
            target: Ground-truth masks of shape ``(B, H, W)`` with class
                indices in ``[0, num_classes)``.

        Returns:
            Scalar loss tensor.
        """
        return F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
        )


class DiceLoss(nn.Module):
    """Multi-class Dice loss for semantic segmentation.

    Computes the Dice coefficient per class after softmax, then returns
    ``1 - mean(dice_per_class)`` as the loss value.
    """

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1e-5,
        ignore_index: int = 255,
    ) -> None:
        """Initialize the Dice loss.

        Args:
            num_classes: Number of segmentation classes.
            smooth: Smoothing constant for numerical stability.
            ignore_index: Class index to ignore (void pixels).
        """
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute multi-class Dice loss.

        Args:
            logits: Raw model output of shape ``(B, num_classes, H, W)``.
            target: Ground-truth masks of shape ``(B, H, W)`` with class
                indices in ``[0, num_classes)``.

        Returns:
            Scalar loss tensor equal to ``1 - mean(dice_per_class)``.
        """
        probabilities = F.softmax(logits, dim=1)

        valid_mask = (target != self.ignore_index).unsqueeze(1)  # (B, 1, H, W)
        safe_target = target.clone()
        safe_target[~valid_mask.squeeze(1)] = 0

        one_hot_target = torch.zeros_like(probabilities)
        one_hot_target.scatter_(1, safe_target.unsqueeze(1), 1.0)

        probabilities = probabilities * valid_mask
        one_hot_target = one_hot_target * valid_mask

        intersection = (probabilities * one_hot_target).sum(dim=(0, 2, 3))
        cardinality = probabilities.sum(dim=(0, 2, 3)) + one_hot_target.sum(dim=(0, 2, 3))

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class SegmentationFocalLoss(nn.Module):
    """Numerically stable and memory-efficient Focal loss for semantic segmentation.

    Leverages ``F.cross_entropy`` under the hood to avoid allocating dense
    one-hot tensors of shape ``(B, num_classes, H, W)``, dramatically reducing
    VRAM footprint while natively supporting ``ignore_index``.
    """

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        alpha: float = 0.25,
        ignore_index: int = 255,
    ) -> None:
        """Initialize the segmentation focal loss.

        Args:
            num_classes: Number of segmentation classes.
            gamma: Focusing parameter that down-weights easy examples.
            alpha: Balancing factor for positive class weighting.
            ignore_index: Class index to ignore (void pixels).
        """
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model output of shape ``(B, num_classes, H, W)``.
            target: Ground-truth masks of shape ``(B, H, W)`` with class
                indices in ``[0, num_classes)``.

        Returns:
            Scalar loss tensor normalized by valid (non-ignored) pixels.
        """
        # Pixel-wise Cross Entropy loss: shape (B, H, W)
        # Returns 0.0 for ignored pixels natively
        ce_loss = F.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # Derive probability of the true class: p_t = exp(-ce_loss)
        p_t = torch.exp(-ce_loss)

        # Compute focal weight factor: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Apply alpha weighting and focal factor
        focal_loss = self.alpha * focal_weight * ce_loss

        # Mask and normalize strictly over valid (non-ignored) pixels
        valid_mask = target != self.ignore_index
        num_valid = valid_mask.sum()

        if num_valid == 0:
            return logits.sum() * 0.0

        return focal_loss[valid_mask].sum() / num_valid

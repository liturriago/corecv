"""Segmentation losses: Cross-Entropy, Dice, and Focal.

Provides three loss modules for semantic segmentation tasks where model
outputs are logits of shape ``(B, C, H, W)`` and targets are class-index
masks of shape ``(B, H, W)`` with ``dtype=torch.long``.

Typical usage::

    ce_loss = SegmentationCrossEntropyLoss(ignore_index=255)
    dice_loss = SegmentationDiceLoss(smooth=1.0)
    focal_loss = SegmentationFocalLoss(gamma=2.0, alpha=0.25)

    loss = ce_loss(logits, masks)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

_LOGITS_NDIM = 4
_MASK_NDIM = 3
_REDUCTIONS: tuple[str, ...] = ("mean", "sum", "none")


def _validate_reduction(reduction: str) -> None:
    """Validate the reduction mode.

    Args:
        reduction: One of ``mean``, ``sum``, or ``none``.

    Raises:
        ValueError: If *reduction* is not a supported mode.

    """
    if reduction not in _REDUCTIONS:
        msg = f"reduction must be one of 'mean', 'sum', 'none', got {reduction!r}"
        raise ValueError(msg)


def _validate_segmentation_inputs(logits: Tensor, targets: Tensor) -> None:
    """Validate segmentation loss inputs.

    Args:
        logits: Raw model output of shape ``(B, C, H, W)``.
        targets: Ground-truth class indices of shape ``(B, H, W)``.

    Raises:
        ValueError: If *logits* is not 4D, *targets* is not 3D, or the batch
            or spatial shapes do not match.

    """
    if logits.ndim != _LOGITS_NDIM:
        msg = f"logits must be 4D with shape (B, C, H, W), got shape {tuple(logits.shape)}"
        raise ValueError(msg)
    if targets.ndim != _MASK_NDIM:
        msg = f"targets must be 3D with shape (B, H, W), got shape {tuple(targets.shape)}"
        raise ValueError(msg)
    if logits.shape[0] != targets.shape[0]:
        msg = (
            f"logits batch size {logits.shape[0]} does not match "
            f"targets batch size {targets.shape[0]}"
        )
        raise ValueError(msg)
    if logits.shape[-2:] != targets.shape[-2:]:
        msg = (
            f"logits spatial shape {tuple(logits.shape[-2:])} does not match "
            f"targets spatial shape {tuple(targets.shape[-2:])}"
        )
        raise ValueError(msg)


class SegmentationCrossEntropyLoss(nn.Module):
    """Per-pixel cross-entropy loss for semantic segmentation.

    Wraps ``torch.nn.functional.cross_entropy`` for pixel-wise classification
    with optional class weighting, label smoothing, an ``ignore_index``, and
    reduction.
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        label_smoothing: float = 0.0,
        ignore_index: int = 255,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        """Initialize the segmentation cross-entropy loss.

        Args:
            weight: Optional class weights tensor of shape ``(num_classes,)``
                for balancing class frequencies.
            label_smoothing: Label smoothing factor in ``[0, 1)``.
            ignore_index: Class index to ignore when computing the loss
                (e.g. 255 for void/unknown pixels).
            reduction: Reduction to apply to the pixelwise loss. One of
                ``mean``, ``sum``, or ``none``.

        """
        super().__init__()
        _validate_reduction(reduction)
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute per-pixel cross-entropy loss.

        Args:
            logits: Raw model output of shape ``(B, C, H, W)``.
            targets: Ground-truth class indices of shape ``(B, H, W)`` with
                ``dtype=torch.long``.

        Returns:
            Loss tensor reduced per ``self.reduction``.

        """
        _validate_segmentation_inputs(logits, targets)
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class SegmentationDiceLoss(nn.Module):
    """Soft Dice loss for semantic segmentation.

    Computes ``1 - mean(class Dice)`` where each class Dice coefficient is
    pooled over all valid pixels of the batch:
    ``2 * |pred ∩ target| / (|pred| + |target|)`` with Laplace smoothing.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = 255,
    ) -> None:
        """Initialize the segmentation Dice loss.

        Args:
            smooth: Laplace smoothing term added to the Dice numerator and
                denominator to avoid division by zero.
            ignore_index: Class index to ignore when computing the loss
                (e.g. 255 for void/unknown pixels).

        Raises:
            ValueError: If *smooth* is negative.

        """
        super().__init__()
        if smooth < 0:
            msg = f"smooth must be non-negative, got {smooth}"
            raise ValueError(msg)
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute soft Dice loss.

        Args:
            logits: Raw model output of shape ``(B, C, H, W)``.
            targets: Ground-truth class indices of shape ``(B, H, W)`` with
                ``dtype=torch.long``.

        Returns:
            Scalar ``1 - mean(class Dice)`` loss tensor.

        """
        _validate_segmentation_inputs(logits, targets)

        num_classes = logits.shape[1]
        valid = targets != self.ignore_index
        safe_targets = targets.where(valid, torch.zeros_like(targets))
        valid_float = valid.unsqueeze(1).float()

        probs = F.softmax(logits, dim=1)
        # Zero out probabilities at ignored pixels so they contribute nothing.
        probs = probs * valid_float

        targets_oh = F.one_hot(safe_targets, num_classes=num_classes)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float() * valid_float

        intersection = (probs * targets_oh).sum(dim=(0, 2, 3))  # (C,)
        pred_area = probs.sum(dim=(0, 2, 3))  # (C,)
        target_area = targets_oh.sum(dim=(0, 2, 3))  # (C,)

        dice_per_class = (2 * intersection + self.smooth) / (
            pred_area + target_area + self.smooth
        )

        return 1.0 - dice_per_class.mean()


class SegmentationFocalLoss(nn.Module):
    """Focal loss for semantic segmentation.

    Applies a focusing factor ``(1 - p_t)^gamma`` to the per-pixel
    cross-entropy so that well-classified pixels contribute less to the
    total loss. Pixels equal to ``ignore_index`` are excluded.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | Tensor | None = None,
        ignore_index: int = 255,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        """Initialize the segmentation focal loss.

        Args:
            gamma: Focusing parameter that down-weights easy pixels.
            alpha: Optional class-balancing weights. Either a scalar ``float``
                applied uniformly to all pixels, or a per-class tensor of
                shape ``(num_classes,)`` whose entry is selected by the
                ground-truth class of each pixel. When ``None``, no balancing
                is applied.
            ignore_index: Class index to ignore when computing the loss
                (e.g. 255 for void/unknown pixels).
            reduction: Reduction to apply to the pixelwise loss. One of
                ``mean``, ``sum``, or ``none``.

        Raises:
            ValueError: If *gamma* is negative or *reduction* is not one of
                ``mean``, ``sum``, ``none``.

        """
        super().__init__()
        if gamma < 0:
            msg = f"gamma must be non-negative, got {gamma}"
            raise ValueError(msg)
        _validate_reduction(reduction)
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model output of shape ``(B, C, H, W)``.
            targets: Ground-truth class indices of shape ``(B, H, W)`` with
                ``dtype=torch.long``.

        Returns:
            Loss tensor reduced per ``self.reduction``. With the ``none``
            reduction, ignored pixels are set to zero and the spatial shape
            ``(B, H, W)`` is preserved.

        Raises:
            ValueError: If *alpha* is a tensor whose shape does not match the
                number of classes.

        """
        _validate_segmentation_inputs(logits, targets)

        num_classes = logits.shape[1]
        valid = targets != self.ignore_index
        safe_targets = targets.where(valid, torch.zeros_like(targets))

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        log_p_t = log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        p_t = probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            if isinstance(self.alpha, Tensor):
                if self.alpha.ndim != 1 or self.alpha.shape[0] != num_classes:
                    msg = (
                        f"alpha must be a 1D tensor with shape ({num_classes},), "
                        f"got shape {tuple(self.alpha.shape)}"
                    )
                    raise ValueError(msg)
                alpha_t = self.alpha.to(device=logits.device)[safe_targets]
            else:
                alpha_t = self.alpha
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_p_t
        loss = loss.masked_fill(~valid, 0.0)

        if self.reduction == "mean":
            return loss.sum() / valid.sum().clamp(min=1)
        if self.reduction == "sum":
            return loss.sum()
        return loss

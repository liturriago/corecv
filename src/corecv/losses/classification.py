"""Image classification losses: Cross-Entropy and Focal.

Provides two loss modules for image classification tasks where model
outputs are logits of shape ``(B, num_classes)`` and targets are
class-index labels of shape ``(B,)`` with ``dtype=torch.long``.

Typical usage::

    ce_loss = ClassificationCrossEntropyLoss(label_smoothing=0.1)
    focal_loss = ClassificationFocalLoss(gamma=2.0, alpha=0.25)

    loss = ce_loss(logits, labels)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

_LOGITS_NDIM = 2
_LABELS_NDIM = 1
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


def _validate_classification_inputs(logits: Tensor, labels: Tensor) -> None:
    """Validate classification loss inputs.

    Args:
        logits: Raw model output of shape ``(B, num_classes)``.
        labels: Ground-truth class indices of shape ``(B,)``.

    Raises:
        ValueError: If *logits* is not 2D, *labels* is not 1D, the batch
            sizes differ, or *labels* contains indices outside
            ``[0, num_classes)``.

    """
    if logits.ndim != _LOGITS_NDIM:
        msg = f"logits must be 2D with shape (B, num_classes), got shape {tuple(logits.shape)}"
        raise ValueError(msg)
    if labels.ndim != _LABELS_NDIM:
        msg = f"labels must be 1D with shape (B,), got shape {tuple(labels.shape)}"
        raise ValueError(msg)
    if labels.shape[0] != logits.shape[0]:
        msg = (
            f"labels batch size {labels.shape[0]} does not match "
            f"logits batch size {logits.shape[0]}"
        )
        raise ValueError(msg)
    num_classes = logits.shape[1]
    if labels.numel() > 0 and (labels.min() < 0 or labels.max() >= num_classes):
        msg = (
            f"labels contain indices outside [0, {num_classes}): "
            f"min={labels.min().item()}, max={labels.max().item()}"
        )
        raise ValueError(msg)


class ClassificationCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for image classification.

    Wraps ``torch.nn.functional.cross_entropy`` for sample-wise classification
    with optional class weighting, label smoothing, and reduction.
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        """Initialize the classification cross-entropy loss.

        Args:
            weight: Optional class weights tensor of shape ``(num_classes,)``
                for balancing class frequencies.
            label_smoothing: Label smoothing factor in ``[0, 1)``.
            reduction: Reduction to apply to the elementwise loss. One of
                ``mean``, ``sum``, or ``none``.

        """
        super().__init__()
        _validate_reduction(reduction)
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: Raw model output of shape ``(B, num_classes)``.
            labels: Ground-truth class indices of shape ``(B,)`` with
                values in ``[0, num_classes)``.

        Returns:
            Loss tensor reduced per ``self.reduction``.

        """
        _validate_classification_inputs(logits, labels)
        return F.cross_entropy(
            logits,
            labels,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )


class ClassificationFocalLoss(nn.Module):
    """Focal loss for image classification.

    Applies a focusing factor ``(1 - p_t)^gamma`` to cross-entropy so that
    well-classified samples contribute less to the total loss.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | Tensor | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        """Initialize the classification focal loss.

        Args:
            gamma: Focusing parameter that down-weights easy examples.
            alpha: Optional class-balancing weights. Either a scalar ``float``
                applied uniformly to all samples, or a per-class tensor of
                shape ``(num_classes,)`` whose entry is selected by the
                ground-truth label of each sample. When ``None``, no
                balancing is applied.
            reduction: Reduction to apply to the elementwise loss. One of
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
        self.reduction = reduction

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model output of shape ``(B, num_classes)``.
            labels: Ground-truth class indices of shape ``(B,)`` with
                values in ``[0, num_classes)``.

        Returns:
            Loss tensor reduced per ``self.reduction``.

        Raises:
            ValueError: If *alpha* is a tensor whose shape does not match the
                number of classes.

        """
        _validate_classification_inputs(logits, labels)

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        log_p_t = log_probs[batch_idx, labels]
        p_t = probs[batch_idx, labels]

        focal_weight = (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            if isinstance(self.alpha, Tensor):
                if self.alpha.ndim != 1 or self.alpha.shape[0] != logits.shape[1]:
                    msg = (
                        f"alpha must be a 1D tensor with shape ({logits.shape[1]},), "
                        f"got shape {tuple(self.alpha.shape)}"
                    )
                    raise ValueError(msg)
                alpha_t = self.alpha.to(device=logits.device)[labels]
            else:
                alpha_t = self.alpha
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_p_t

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

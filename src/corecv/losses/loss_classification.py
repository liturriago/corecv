# noqa: INP001
"""Image classification losses: Cross-Entropy and Focal.

Provides two loss modules for image classification tasks where model
outputs are logits of shape ``[B, num_classes]`` and targets are
class-index labels of shape ``[B]`` with ``dtype=torch.long``.

Typical usage::

    ce_loss = ClassificationCrossEntropyLoss(label_smoothing=0.1)
    focal_loss = ClassificationFocalLoss(gamma=2.0, alpha=0.25)

    loss = ce_loss(logits, labels)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class ClassificationCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for image classification.

    Wraps ``torch.nn.functional.cross_entropy`` for sample-wise classification
    with optional class weighting and label smoothing.
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        """Initialize the classification cross-entropy loss.

        Args:
            weight: Optional class weights tensor of shape ``(num_classes,)``
                for balancing class frequencies.
            label_smoothing: Label smoothing factor in ``[0, 1)``.
        """
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: Raw model output of shape ``(B, num_classes)``.
            labels: Ground-truth class indices of shape ``(B,)`` with
                values in ``[0, num_classes)``.

        Returns:
            Scalar loss tensor.
        """
        return F.cross_entropy(
            logits,
            labels,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )


class ClassificationFocalLoss(nn.Module):
    """Focal loss for image classification.

    Applies a focusing factor ``(1 - p_t)^gamma`` to cross-entropy so that
    well-classified samples contribute less to the total loss.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
    ) -> None:
        """Initialize the classification focal loss.

        Args:
            gamma: Focusing parameter that down-weights easy examples.
            alpha: Balancing factor per class. When ``None``, uniform
                weighting is applied.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model output of shape ``(B, num_classes)``.
            labels: Ground-truth class indices of shape ``(B,)`` with
                values in ``[0, num_classes)``.

        Returns:
            Scalar loss tensor.
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        log_p_t = log_probs[batch_idx, labels]
        p_t = probs[batch_idx, labels]

        focal_weight = (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight

        loss = -focal_weight * log_p_t
        return loss.mean()

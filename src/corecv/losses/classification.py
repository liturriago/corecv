"""GPU-native classification loss functions for CoreCV.

Provides focal loss and label-smoothing cross-entropy, both implemented as
pure vectorised PyTorch operations with **zero** CPU-GPU synchronisations
during forward or backward passes.  No Python for-loops, no ``.item()``
calls, no ``.cpu()`` transfers — everything stays on VRAM.

Example:
    >>> import torch
    >>> from corecv.losses.classification import FocalLoss
    >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    >>> logits = torch.randn(8, 80, 32, 32, device="cuda")
    >>> targets = torch.randint(0, 80, (8, 32, 32), device="cuda")
    >>> loss = loss_fn(logits, targets)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FocalLoss(nn.Module):
    """Focal Loss for dense object detection (Lin et al. ICCV 2017).

    Reduces the relative loss for well-classified examples, focusing training
    on hard negatives.  The formulation is::

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where ``p_t = exp(-CE)`` and ``alpha_t`` is a per-class weighting factor.

    This implementation computes the standard cross-entropy with
    ``reduction='none'`` and applies the focal modulator entirely via
    vectorised tensor ops — no Python loops.

    Args:
        alpha: Weighting factor.  Can be a scalar ``float`` applied to all
            classes, or a 1-D ``Tensor`` of shape ``(C,)`` with per-class
            weights.
        gamma: Focal focusing parameter.  Higher values down-weight easy
            examples more aggressively.  Default ``2.0``.
        reduction: Aggregation mode.  ``'mean'`` (default) returns the
            mean loss over all elements; ``'sum'`` returns the sum;
            ``'none'`` returns the raw per-element loss tensor.

    Example:
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        >>> logits = torch.randn(4, 80, 16, 16, device="cuda")
        >>> targets = torch.randint(0, 80, (4, 16, 16), device="cuda")
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        alpha: float | Tensor = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """Initialise FocalLoss with alpha, gamma, and reduction mode."""
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            msg = f"Invalid reduction: {reduction!r}. Must be 'mean', 'sum', or 'none'."
            raise ValueError(msg)

        self.gamma = float(gamma)
        self.reduction = reduction

        # Store alpha as a buffer so it moves with .to(device) automatically.
        # If a Tensor is passed it is registered directly; a scalar float is
        # broadcast later during forward.
        if isinstance(alpha, Tensor):
            self.register_buffer("alpha", alpha.float())
        else:
            # Will be broadcast to match logits shape in forward
            self.alpha = float(alpha)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            inputs: Unnormalised logits of shape ``(B, C, ...)`` where ``C``
                is the number of classes.  Any trailing spatial dimensions
                are supported.
            targets: Ground-truth class indices of shape ``(B, ...)`` with
                the same trailing spatial dimensions as *inputs* (excluding
                the class axis).  Values must be in ``[0, C)``.

        Returns:
            Scalar or tensor loss depending on ``self.reduction``.
        """
        # Standard cross-entropy without reduction — gives per-element loss
        ce_loss: Tensor = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
        )

        # p_t = exp(-CE) is the model's estimated probability for the
        # ground-truth class.
        pt: Tensor = torch.exp(-ce_loss)

        # Focal modulator: (1 - p_t)^gamma
        focal_weight: Tensor = (1.0 - pt).pow(self.gamma)

        # Alpha weighting: broadcast scalar or match channel dim
        alpha = self.alpha
        if isinstance(alpha, float):
            # Scalar — simply multiply
            loss = alpha * focal_weight * ce_loss
        else:
            # Per-class alpha tensor of shape (C,).
            # Gather the alpha value for each spatial position's target
            # class.  We expand alpha to full (B, C, *spatial) shape,
            # then use torch.gather along dim=1 with the target indices.
            alpha_shape = [1] * inputs.dim()  # (1, C, 1, 1, ...)
            alpha_shape[1] = -1  # sentinel for C
            alpha_expanded: Tensor = alpha.view(alpha_shape)
            # Expand to full (B, C, *spatial) — expand does not allocate
            alpha_expanded = alpha_expanded.expand_as(inputs)
            # Gather along class dim using target indices
            alpha_t: Tensor = torch.gather(
                alpha_expanded,
                dim=1,
                index=targets.unsqueeze(1),
            ).squeeze(1)
            loss = alpha_t * focal_weight * ce_loss

        return self._reduce(loss)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reduce(self, loss: Tensor) -> Tensor:
        """Apply the configured reduction mode.

        Args:
            loss: Raw per-element loss tensor.

        Returns:
            Reduced scalar or tensor.
        """
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'


class LabelSmoothingCrossEntropy(nn.Module):
    """Label-smoothing cross-entropy loss.

    Instead of a hard one-hot target ``[0, 0, 1, 0, ...]``, this loss
    assigns ``(1 - smoothing)`` probability mass to the correct class and
    distributes ``smoothing / (K - 1)`` mass uniformly across all other
    classes, where ``K`` is the number of classes.  This discourages the
    model from becoming over-confident and acts as a regulariser.

    Formulation::

        L = (1 - smoothing) * CE(y, p) + smoothing * mean_k(log_softmax(z))

    where ``CE`` is the standard cross-entropy and ``mean_k`` averages the
    log-softmax over all ``K`` classes.

    The implementation is fully vectorised using ``F.log_softmax`` +
    ``F.nll_loss`` with no Python loops and no CPU-GPU sync points.

    Args:
        smoothing: Label-smoothing factor in ``[0, 1)``.  ``0.0`` is
            equivalent to standard cross-entropy.  Default ``0.1``.
        reduction: Aggregation mode (``'mean'`` | ``'sum'`` | ``'none'``).
            Default ``'mean'``.

    Example:
        >>> loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        >>> logits = torch.randn(16, 100, device="cuda")
        >>> targets = torch.randint(0, 100, (16,), device="cuda")
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        """Initialise LabelSmoothingCrossEntropy with smoothing factor and reduction mode."""
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            msg = f"smoothing must be in [0, 1), got {smoothing}"
            raise ValueError(msg)
        if reduction not in ("mean", "sum", "none"):
            msg = f"Invalid reduction: {reduction!r}. Must be 'mean', 'sum', or 'none'."
            raise ValueError(msg)

        self.smoothing = float(smoothing)
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute label-smoothing cross-entropy.

        Args:
            inputs: Unnormalised logits of shape ``(B, C, ...)`` where ``C``
                is the number of classes.
            targets: Ground-truth class indices of shape ``(B, ...)`` with
                values in ``[0, C)``.

        Returns:
            Scalar or tensor loss depending on ``self.reduction``.
        """
        log_probs: Tensor = F.log_softmax(inputs, dim=1)

        # NLL part: -log_probs gathered at target positions, averaged over
        # all spatial / batch dimensions.
        nll_loss: Tensor = F.nll_loss(
            log_probs,
            targets,
            reduction="none",
        )

        # Smooth part: negative mean of log-softmax over all classes.
        # log_probs has shape (B, C, ...); mean over dim=1 gives (B, ...).
        smooth_loss: Tensor = -log_probs.mean(dim=1)

        # Combine
        loss: Tensor = (
            (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        )

        return self._reduce(loss)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reduce(self, loss: Tensor) -> Tensor:
        """Apply the configured reduction mode.

        Args:
            loss: Raw per-element loss tensor.

        Returns:
            Reduced scalar or tensor.
        """
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'

"""GPU-native segmentation loss functions for CoreCV.

Provides Dice loss and a combined CrossEntropy + Dice loss, both implemented
as pure vectorised PyTorch operations with **zero** CPU-GPU synchronisations
during forward or backward passes.  No Python for-loops over channels or
batch elements — every computation uses broadcasting, ``F.one_hot``, and
standard tensor ops that remain entirely on VRAM.

Example:
    >>> import torch
    >>> from corecv.losses.segmentation import DiceLoss
    >>> loss_fn = DiceLoss(smooth=1.0)
    >>> logits = torch.randn(4, 21, 256, 256, device="cuda")
    >>> targets = torch.randint(0, 21, (4, 256, 256), device="cuda")
    >>> loss = loss_fn(logits, targets)
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class DiceLoss(nn.Module):
    """Dice coefficient loss for semantic segmentation.

    The Dice coefficient measures overlap between prediction and target::

        Dice = 2 * sum(p * t) / (sum(p) + sum(t) + smooth)

    The loss is ``1 - Dice``.  A value of ``smooth`` (default ``1.0``)
    prevents division by zero.

    This implementation is **fully vectorised**: ``F.one_hot`` expands targets
    to ``(B, H, W, C)`` in a single kernel call, then the Dice computation
    is a sequence of batch reductions with no Python loops over channels or
    spatial locations.

    Args:
        smooth: Laplace smoothing factor to avoid division by zero.
            Default ``1.0``.
        reduction: Aggregation mode (``'mean'`` | ``'sum'`` | ``'none'``).
            Default ``'mean'``.

    Note:
        Inputs should be **logits** (not probabilities).  The module applies
        channel-wise softmax internally.

    Example:
        >>> loss_fn = DiceLoss(smooth=1.0)
        >>> logits = torch.randn(2, 10, 128, 128, device="cuda")
        >>> targets = torch.randint(0, 10, (2, 128, 128), device="cuda")
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        """Initialise DiceLoss with smoothing factor and reduction mode."""
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            msg = f"Invalid reduction: {reduction!r}. Must be 'mean', 'sum', or 'none'."
            raise ValueError(msg)

        self.smooth = float(smooth)
        self.reduction = reduction

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute Dice loss.

        Args:
            inputs: Logits of shape ``(B, C, H, W)`` where ``C`` is the
                number of segmentation classes.
            targets: Ground-truth class indices of shape ``(B, H, W)`` with
                values in ``[0, C)``.

        Returns:
            Scalar or tensor loss depending on ``self.reduction``.
        """
        num_classes: int = inputs.shape[1]

        # --- Softmax predictions: (B, C, H, W) -> probabilities --------
        probs: Tensor = F.softmax(inputs, dim=1)

        # --- One-hot encode targets: (B, H, W) -> (B, H, W, C) --------
        # F.one_hot expects Long dtype
        targets_long = targets.long()
        one_hot: Tensor = F.one_hot(targets_long, num_classes=num_classes)
        # (B, H, W, C) -> (B, C, H, W) to match probs layout
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        # --- Vectorised Dice computation --------------------------------
        # Flatten spatial dims: (B, C, H, W) -> (B, C, H*W)
        probs_flat: Tensor = probs.flatten(2)      # (B, C, N)
        target_flat: Tensor = one_hot.flatten(2)    # (B, C, N)

        # Intersection: sum of element-wise product across spatial dim
        # (B, C, N) -> (B, C)  [sum over N]
        intersection: Tensor = (probs_flat * target_flat).sum(dim=2)

        probs_sum: Tensor = probs_flat.sum(dim=2)
        target_sum: Tensor = target_flat.sum(dim=2)

        # Dice coefficient per (batch, class): (B, C)
        dice: Tensor = (2.0 * intersection + self.smooth) / (
            probs_sum + target_sum + self.smooth
        )

        # Dice loss per (batch, class): (B, C)
        dice_loss: Tensor = 1.0 - dice

        # --- Per-sample reduction across classes ------------------------
        # Mean over classes to get per-sample loss: (B,)
        per_sample_loss: Tensor = dice_loss.mean(dim=1)

        return self._reduce(per_sample_loss)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reduce(self, loss: Tensor) -> Tensor:
        """Apply the configured reduction mode.

        Args:
            loss: Per-sample loss tensor of shape ``(B,)``.

        Returns:
            Reduced scalar or tensor.
        """
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'


class CombinedSegmentationLoss(nn.Module):
    """Combined Cross-Entropy + Dice loss for segmentation.

    Summation of a weighted ``F.cross_entropy`` term and a
    :class:`DiceLoss` term, designed for common semantic segmentation
    training where both pixel-wise classification accuracy and region
    overlap matter.

    Handles ``ignore_index`` transparently: pixels marked with
    ``ignore_index`` in the target are excluded from **both** the CE and
    the Dice computation, ensuring no gradient leakage from unlabeled
    regions.

    The entire pipeline is 100% GPU-native with zero CPU-GPU sync points.

    Args:
        ce_weight: Weight of the cross-entropy term.  Default ``1.0``.
        dice_weight: Weight of the Dice term.  Default ``1.0``.
        ignore_index: Class index to ignore during loss computation.
            Default ``-100`` (same as ``F.cross_entropy`` default).

    Example:
        >>> loss_fn = CombinedSegmentationLoss(
        ...     ce_weight=1.0, dice_weight=1.0, ignore_index=255,
        ... )
        >>> logits = torch.randn(8, 21, 256, 256, device="cuda")
        >>> targets = torch.full((8, 256, 256), 255, device="cuda")
        >>> targets[:, :128, :128] = torch.randint(0, 21, (8, 128, 128), device="cuda")
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = -100,
    ) -> None:
        """Initialise CombinedSegmentationLoss with term weights and ignore index."""
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ignore_index = int(ignore_index)
        self._dice_fn = DiceLoss(smooth=1.0, reduction="mean")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute the combined CE + Dice loss.

        Args:
            inputs: Logits of shape ``(B, C, H, W)``.
            targets: Ground-truth class indices of shape ``(B, H, W)``.
                Pixels equal to ``self.ignore_index`` are excluded from
                both loss terms.

        Returns:
            Weighted scalar loss.
        """
        # ---- Cross-entropy component ------------------------------------
        ce_loss: Tensor = F.cross_entropy(
            inputs,
            targets,
            ignore_index=self.ignore_index,
            reduction="mean",
        )

        # ---- Dice component with ignore-mask ----------------------------
        # Build a boolean mask of valid (non-ignored) pixels.
        valid_mask: Tensor = targets != self.ignore_index  # (B, H, W)

        # Replace ignored targets with a safe dummy class (0) so that
        # F.one_hot does not produce out-of-range indices.  The mask
        # ensures these positions contribute zero to the Dice sum.
        safe_targets: Tensor = targets.clone()
        safe_targets[~valid_mask] = 0

        num_classes: int = inputs.shape[1]

        # Softmax predictions
        probs: Tensor = F.softmax(inputs, dim=1)

        # One-hot targets: (B, H, W) -> (B, H, W, C) -> (B, C, H, W)
        one_hot: Tensor = F.one_hot(safe_targets, num_classes=num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        # Apply the valid mask to both predictions and targets so that
        # ignored positions contribute exactly zero to all sums.
        # valid_mask: (B, H, W) -> (B, 1, H, W) for broadcasting
        mask_4d: Tensor = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)
        probs_masked: Tensor = probs * mask_4d
        target_masked: Tensor = one_hot * mask_4d

        # Vectorised Dice over valid pixels only
        probs_flat: Tensor = probs_masked.flatten(2)   # (B, C, N)
        target_flat: Tensor = target_masked.flatten(2)  # (B, C, N)

        intersection: Tensor = (probs_flat * target_flat).sum(dim=2)
        probs_sum: Tensor = probs_flat.sum(dim=2)
        target_sum: Tensor = target_flat.sum(dim=2)

        smooth = 1.0
        dice: Tensor = (2.0 * intersection + smooth) / (
            probs_sum + target_sum + smooth
        )
        dice_loss: Tensor = (1.0 - dice).mean(dim=1)  # per-sample, mean over classes

        # If all pixels in a sample are ignored, the dice loss is 0
        # (division by zero is avoided because smooth > 0).
        dice_loss_mean: Tensor = dice_loss.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss_mean

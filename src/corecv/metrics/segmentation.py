"""GPU-native semantic segmentation metrics engine for CoreCV.

Provides an accumulator-based :class:`SegmentationMetrics` that computes
mean IoU, pixel accuracy, and mean Dice coefficient — all on VRAM with
**zero** CPU-GPU synchronisations during ``update()``.  Per-class
intersection and union counts are stored in GPU-resident buffers via
``nn.register_buffer``.

The ``compute()`` method performs only the final reductions and returns a
plain Python ``dict`` — that is the only point where implicit CPU transfer
occurs.

No pycocotools, no Python for-loops, no ``.cpu()`` calls in hot paths.

Example:
    >>> import torch
    >>> from corecv.metrics.segmentation import SegmentationMetrics
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> metrics = SegmentationMetrics(num_classes=21, device=device)
    >>> logits = torch.randn(4, 21, 128, 128, device=device)
    >>> targets = torch.randint(0, 21, (4, 128, 128), device=device)
    >>> metrics.update(logits, targets)
    >>> results = metrics.compute()
    >>> results["miou"]
    0.05
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_EXPECTED_PREDS_DIM = 4
_EXPECTED_TARGETS_DIM = 3


class SegmentationMetrics(nn.Module):
    """Accumulator-based semantic segmentation metrics.

    Maintains per-class intersection and union counts in VRAM-resident
    buffers so that ``update()`` never triggers a CPU-GPU synchronisation.
    Only ``compute()`` performs the final divisions.

    Supported metrics:

    * **miou** — Mean Intersection-over-Union averaged over valid classes.
    * **pixel_accuracy** — Fraction of correctly classified pixels (among
      valid, non-ignored pixels).
    * **mean_dice** — Mean Dice coefficient averaged over valid classes.
    * **per_class_iou** — IoU for each class as a ``Tensor`` of shape
      ``(num_classes,)``.
    * **per_class_dice** — Dice for each class as a ``Tensor`` of shape
      ``(num_classes,)``.

    Args:
        num_classes: Number of segmentation classes.
        ignore_index: Class index to exclude from metric computation.
            Default ``-100`` (matches ``F.cross_entropy`` convention).
        device: Device on which to allocate accumulator buffers.

    Example:
        >>> metrics = SegmentationMetrics(num_classes=21, ignore_index=255, device="cuda")
        >>> for logits, targets in loader:
        ...     metrics.update(logits, targets)
        >>> print(metrics.compute()["miou"])
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -100,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialise SegmentationMetrics with num_classes, ignore_index, and device."""
        super().__init__()
        if num_classes < 1:
            msg = f"num_classes must be >= 1, got {num_classes}"
            raise ValueError(msg)

        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.device = torch.device(device)

        # ------------------------------------------------------------------
        # Accumulator buffers — all on VRAM.
        #
        # For each class c we track:
        #   intersection[c] = # pixels where pred == c AND target == c
        #   union[c]        = # pixels where pred == c OR  target == c
        #   pred_sum[c]     = # pixels where pred == c  (for Dice numerator)
        #   target_sum[c]   = # pixels where target == c (for Dice numerator)
        # ------------------------------------------------------------------
        self.register_buffer(
            "intersection",
            torch.zeros(num_classes, dtype=torch.int64, device=self.device),
        )
        self.register_buffer(
            "union",
            torch.zeros(num_classes, dtype=torch.int64, device=self.device),
        )
        self.register_buffer(
            "pred_sum",
            torch.zeros(num_classes, dtype=torch.int64, device=self.device),
        )
        self.register_buffer(
            "target_sum",
            torch.zeros(num_classes, dtype=torch.int64, device=self.device),
        )
        self.register_buffer(
            "total_correct",
            torch.zeros(1, dtype=torch.int64, device=self.device),
        )
        self.register_buffer(
            "total_valid",
            torch.zeros(1, dtype=torch.int64, device=self.device),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Accumulate a batch of predictions and ground-truth masks.

        Args:
            preds: Logits of shape ``(B, C, H, W)`` where ``C`` is the
                number of classes.
            targets: Ground-truth class indices of shape ``(B, H, W)``
                with values in ``[0, C)`` and optionally
                ``self.ignore_index`` for ignored pixels.

        Raises:
            ValueError: If tensor shapes or value ranges are invalid.
        """
        self._validate_inputs(preds, targets)

        # Move to accumulator device (no-op if already there).
        preds = preds.to(device=self.device)
        targets = targets.to(device=self.device)

        # Derive predicted class indices: argmax over channel dim.
        pred_classes: Tensor = preds.argmax(dim=1)  # (B, H, W)

        # Build a validity mask — exclude ignore_index pixels.
        valid_mask: Tensor = targets != self.ignore_index  # (B, H, W)

        # Replace ignored targets with a safe dummy (0) so that
        # F.one_hot does not produce out-of-range indices.
        safe_targets: Tensor = targets.clone()
        safe_targets[~valid_mask] = 0

        # ---- One-hot encode preds and targets --------------------------
        # (B, H, W) -> (B, H, W, C)
        oh_pred: Tensor = F.one_hot(pred_classes, self.num_classes)
        oh_target: Tensor = F.one_hot(safe_targets, self.num_classes)

        # Apply validity mask: (B, H, W) -> (B, H, W, 1) for broadcasting.
        mask_4d: Tensor = valid_mask.unsqueeze(-1)  # (B, H, W, 1)
        oh_pred = oh_pred.masked_fill(~mask_4d, 0)
        oh_target = oh_target.masked_fill(~mask_4d, 0)

        # Flatten spatial and batch dims: (B, H, W, C) -> (N, C)
        N: int = oh_pred.shape[0] * oh_pred.shape[1] * oh_pred.shape[2]
        oh_pred_flat: Tensor = oh_pred.reshape(N, self.num_classes)
        oh_target_flat: Tensor = oh_target.reshape(N, self.num_classes)

        # ---- Accumulate per-class counts (pure tensor ops) -------------
        # intersection[c] = sum of (oh_pred[:, c] * oh_target[:, c])
        # union[c]        = sum of clamp(oh_pred[:, c] + oh_target[:, c])
        inter_batch: Tensor = (oh_pred_flat * oh_target_flat).sum(dim=0)  # (C,)
        union_batch: Tensor = oh_pred_flat.sum(dim=0) + oh_target_flat.sum(
            dim=0
        ) - inter_batch  # (C,)

        self.intersection += inter_batch.to(dtype=torch.int64)
        self.union += union_batch.to(dtype=torch.int64)
        self.pred_sum += oh_pred_flat.sum(dim=0).to(dtype=torch.int64)
        self.target_sum += oh_target_flat.sum(dim=0).to(dtype=torch.int64)

        # ---- Pixel accuracy (correct / valid) -------------------------
        correct_batch: Tensor = (
            (pred_classes == targets) & valid_mask
        ).sum()
        valid_batch: Tensor = valid_mask.sum()

        self.total_correct += correct_batch.to(dtype=torch.int64)
        self.total_valid += valid_batch.to(dtype=torch.int64)

    def compute(self) -> dict[str, float | Tensor]:
        """Compute all metrics from the accumulated state.

        Returns:
            Dictionary with the following keys:

            * ``"miou"`` — Mean IoU across valid classes (float).
            * ``"pixel_accuracy"`` — Overall pixel accuracy (float).
            * ``"mean_dice"`` — Mean Dice across valid classes (float).
            * ``"per_class_iou"`` — Per-class IoU tensor ``(C,)``.
            * ``"per_class_dice"`` — Per-class Dice tensor ``(C,)``.
        """
        eps: float = 1e-8

        # ---- IoU per class: intersection / union -----------------------
        inter: Tensor = self.intersection.float()  # (C,)
        union: Tensor = self.union.float()  # (C,)

        # A class is "valid" if it appeared in either predictions or targets.
        valid_classes: Tensor = union > 0  # (C,)

        iou_per_class: Tensor = torch.zeros(
            self.num_classes, dtype=torch.float32, device=self.device
        )
        # Only divide where union > 0 to avoid NaN for absent classes.
        iou_per_class[valid_classes] = inter[valid_classes] / (
            union[valid_classes] + eps
        )

        # Mean IoU — average only over classes that actually appeared.
        num_valid: int = valid_classes.sum().item()
        miou: float = (iou_per_class.sum() / max(num_valid, 1)).item()

        # ---- Pixel accuracy -------------------------------------------
        pixel_accuracy: float = (
            self.total_correct.float() / (self.total_valid.float() + eps)
        ).item()

        # ---- Dice per class: 2 * intersection / (pred_sum + target_sum)
        p_sum: Tensor = self.pred_sum.float()
        t_sum: Tensor = self.target_sum.float()

        dice_per_class: Tensor = torch.zeros(
            self.num_classes, dtype=torch.float32, device=self.device
        )
        dice_per_class[valid_classes] = (
            2.0 * inter[valid_classes]
            / (p_sum[valid_classes] + t_sum[valid_classes] + eps)
        )

        mean_dice: float = (dice_per_class.sum() / max(num_valid, 1)).item()

        return {
            "miou": miou,
            "pixel_accuracy": pixel_accuracy,
            "mean_dice": mean_dice,
            "per_class_iou": iou_per_class,
            "per_class_dice": dice_per_class,
        }

    def reset(self) -> None:
        """Reset all accumulator buffers to zero."""
        self.intersection.zero_()
        self.union.zero_()
        self.pred_sum.zero_()
        self.target_sum.zero_()
        self.total_correct.zero_()
        self.total_valid.zero_()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_inputs(self, preds: Tensor, targets: Tensor) -> None:
        """Validate shapes and value ranges.

        Args:
            preds: Logits tensor ``(B, C, H, W)``.
            targets: Target tensor ``(B, H, W)``.

        Raises:
            ValueError: On shape mismatch or out-of-range values.
        """
        if preds.dim() != _EXPECTED_PREDS_DIM:
            msg = f"preds must be 4-D (B, C, H, W), got shape {tuple(preds.shape)}"
            raise ValueError(msg)
        if targets.dim() != _EXPECTED_TARGETS_DIM:
            msg = f"targets must be 3-D (B, H, W), got shape {tuple(targets.shape)}"
            raise ValueError(msg)
        if preds.shape[0] != targets.shape[0]:
            msg = (
                f"Batch size mismatch: preds={preds.shape[0]}, "
                f"targets={targets.shape[0]}"
            )
            raise ValueError(msg)
        if preds.shape[2] != targets.shape[1] or preds.shape[3] != targets.shape[2]:
            msg = (
                f"Spatial size mismatch: preds H×W = {preds.shape[2]}×{preds.shape[3]}, "
                f"targets H×W = {targets.shape[1]}×{targets.shape[2]}"
            )
            raise ValueError(msg)
        if preds.shape[1] != self.num_classes:
            msg = (
                f"preds has {preds.shape[1]} channels but expected "
                f"{self.num_classes} classes"
            )
            raise ValueError(msg)

        # Check for out-of-range target values (excluding ignore_index).
        non_ignore: Tensor = targets != self.ignore_index
        if non_ignore.any():
            target_vals: Tensor = targets[non_ignore]
            if (target_vals < 0).any() or (target_vals >= self.num_classes).any():
                msg = (
                    f"targets contain values outside [0, {self.num_classes}) "
                    f"(excluding ignore_index={self.ignore_index}). "
                    f"Range: [{target_vals.min().item()}, {target_vals.max().item()}]"
                )
                raise ValueError(msg)

"""Segmentation evaluation metrics: Precision, Recall, Dice, and IoU.

Computes macro-averaged precision, recall, dice, and IoU across all classes
by accumulating a pixel-wise confusion matrix across validation batches.

Typical usage::

    metrics = SegmentationMetrics(num_classes=21)

    for batch in val_loader:
        images = batch["images"]
        masks = batch["masks"]
        logits = model(images)

        metrics.update(logits, masks)

    results = metrics.compute()
    print(results)
    # {"precision": 0.85, "recall": 0.83, "dice": 0.84, "iou": 0.73}
    metrics.reset()
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_LOGITS_NDIM = 4
_MASK_NDIM = 3


def _build_confusion_matrix(
    preds: Tensor,
    targets: Tensor,
    num_classes: int,
) -> Tensor:
    """Build a pixel-wise confusion matrix from predictions and targets.

    Args:
        preds: Predicted class indices of shape ``(B, H, W)`` with dtype ``torch.long``.
        targets: Ground-truth class indices of shape ``(B, H, W)`` with dtype ``torch.long``.
        num_classes: Total number of classes.

    Returns:
        Confusion matrix of shape ``(num_classes, num_classes)`` where entry
        ``[i, j]`` is the count of pixels with target ``i`` and prediction ``j``.

    """
    indices = targets * num_classes + preds
    confusion_flat = torch.bincount(indices.flatten(), minlength=num_classes * num_classes)
    return confusion_flat.reshape(num_classes, num_classes)


class SegmentationMetrics:
    """Evaluator for Semantic Segmentation metrics (Precision, Recall, Dice, IoU).

    Accumulates a pixel-wise confusion matrix across validation batches using
    O(1) memory, then evaluates performance using macro-averaged precision,
    recall, dice, and IoU.
    """

    def __init__(
        self,
        num_classes: int,
        eps: float = 1e-16,
    ) -> None:
        """Initialize SegmentationMetrics evaluator.

        Args:
            num_classes: Total number of segmentation classes.
            eps: Small constant for numerical stability.

        """
        self.num_classes = num_classes
        self.eps = eps

        # O(1) state: pixel-wise confusion matrix.
        self._confusion_matrix = torch.zeros(
            num_classes,
            num_classes,
            dtype=torch.long,
        )
        self.results: dict[str, float] = {}

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update metric state with a new batch of predictions and targets.

        Pixels whose target or prediction falls outside ``[0, num_classes)``
        (e.g. the ``ignore_index``/void value 255) are excluded from the
        accumulated confusion matrix.

        Args:
            preds: Either logits of shape ``(B, C, H, W)`` or class-index
                predictions of shape ``(B, H, W)`` with dtype ``torch.long``.
            targets: Ground-truth class indices of shape ``(B, H, W)`` with
                dtype ``torch.long``.

        Raises:
            ValueError: If *preds* is neither 4D logits nor 3D class indices,
                if *targets* is not 3D, or if batch/spatial shapes do not match.

        """
        if preds.ndim not in (_LOGITS_NDIM, _MASK_NDIM):
            msg = (
                "preds must be 4D logits (B, C, H, W) or 3D class indices "
                f"(B, H, W), got shape {tuple(preds.shape)}"
            )
            raise ValueError(msg)
        if targets.ndim != _MASK_NDIM:
            msg = f"targets must be 3D with shape (B, H, W), got shape {tuple(targets.shape)}"
            raise ValueError(msg)
        if preds.shape[0] != targets.shape[0]:
            msg = (
                f"preds batch size {preds.shape[0]} does not match "
                f"targets batch size {targets.shape[0]}"
            )
            raise ValueError(msg)
        if preds.shape[-2:] != targets.shape[-2:]:
            msg = (
                f"preds spatial shape {tuple(preds.shape[-2:])} does not match "
                f"targets spatial shape {tuple(targets.shape[-2:])}"
            )
            raise ValueError(msg)
        if preds.ndim == _LOGITS_NDIM and preds.shape[1] != self.num_classes:
            msg = (
                f"preds logits have {preds.shape[1]} classes but SegmentationMetrics "
                f"was initialized with num_classes={self.num_classes}"
            )
            raise ValueError(msg)

        preds_long = (
            preds.detach().argmax(dim=1).long()
            if preds.ndim == _LOGITS_NDIM
            else preds.detach().long()
        )
        targets_long = targets.detach().long()

        # Exclude ignore/void pixels (target or prediction outside [0, num_classes)).
        valid = (
            (preds_long >= 0)
            & (preds_long < self.num_classes)
            & (targets_long >= 0)
            & (targets_long < self.num_classes)
        )

        # Build confusion matrix on GPU, then transfer only the fixed-size matrix to CPU.
        batch_confusion = _build_confusion_matrix(
            preds_long[valid],
            targets_long[valid],
            self.num_classes,
        )
        self._confusion_matrix += batch_confusion.cpu()

    def reset(self) -> None:
        """Reset all accumulated counters."""
        self._confusion_matrix.zero_()
        self.results = {}

    def compute(self) -> dict[str, float]:
        """Compute evaluation metrics over all accumulated data.

        Returns:
            Dictionary containing macro-averaged metric values:
            - ``precision``: Macro-averaged precision across present classes.
            - ``recall``: Macro-averaged recall across present classes.
            - ``dice``: Macro-averaged Dice coefficient across present classes.
            - ``iou``: Macro-averaged Intersection-over-Union across present classes.

        """
        if self._confusion_matrix.sum().item() == 0:
            self.results = {
                "precision": 0.0,
                "recall": 0.0,
                "dice": 0.0,
                "iou": 0.0,
            }
            return self.results

        # Compute per-class true positives, false positives, and false negatives.
        confusion = self._confusion_matrix.float()

        true_positives = confusion.diag()
        false_positives = confusion.sum(dim=0) - true_positives
        false_negatives = confusion.sum(dim=1) - true_positives

        present_classes = (true_positives + false_negatives) > 0

        precision_per_class = true_positives / (true_positives + false_positives + self.eps)
        recall_per_class = true_positives / (true_positives + false_negatives + self.eps)
        dice_per_class = (2 * true_positives) / (
            2 * true_positives + false_positives + false_negatives + self.eps
        )
        iou_per_class = true_positives / (
            true_positives + false_positives + false_negatives + self.eps
        )

        if present_classes.any():
            self.results = {
                "precision": round(precision_per_class[present_classes].mean().item(), 4),
                "recall": round(recall_per_class[present_classes].mean().item(), 4),
                "dice": round(dice_per_class[present_classes].mean().item(), 4),
                "iou": round(iou_per_class[present_classes].mean().item(), 4),
            }
        else:
            self.results = {
                "precision": 0.0,
                "recall": 0.0,
                "dice": 0.0,
                "iou": 0.0,
            }

        return self.results

    def print_results(self, stage: str) -> None:
        """Print the evaluation results for a given stage.

        Args:
            stage: Name of the stage (train, val, test).

        Raises:
            RuntimeError: If :meth:`compute` has not been called yet.

        """
        if not self.results:
            msg = "No results available. Call compute() before print_results()."
            raise RuntimeError(msg)

        logger.info(
            "%s | precision=%.4f recall=%.4f dice=%.4f iou=%.4f",
            stage,
            self.results["precision"],
            self.results["recall"],
            self.results["dice"],
            self.results["iou"],
        )

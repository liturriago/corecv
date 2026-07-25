"""Semantic segmentation evaluation metrics: Dice, IoU, Precision, and Recall.

Computes macro-averaged Dice score, Intersection over Union (IoU), precision,
and recall across all classes by accumulating a confusion matrix over
validation batches.

Typical usage::

    metrics = SegmentationMetrics(num_classes=21)

    for batch in val_loader:
        images = batch["images"]
        masks = batch["masks"]
        logits = model(images)

        metrics.update(logits, masks)

    results = metrics.compute()
    print(results)
    # {"dice": 0.78, "iou": 0.65, "precision": 0.80, "recall": 0.76}
    metrics.reset()
"""

from __future__ import annotations

import torch
from torch import Tensor


def _build_confusion_matrix(
    predictions: Tensor,
    targets: Tensor,
    num_classes: int,
    ignore_index: int,
) -> Tensor:
    """Build a confusion matrix from predictions and targets.

    Args:
        predictions: Predicted class indices of shape ``(N,)`` with dtype ``torch.long``.
        targets: Ground-truth class indices of shape ``(N,)`` with dtype ``torch.long``.
        num_classes: Total number of classes.
        ignore_index: Class index to ignore in metric computation.

    Returns:
        Confusion matrix of shape ``(num_classes, num_classes)`` where entry
        ``[i, j]`` is the count of pixels with prediction ``i`` and target ``j``.
    """
    valid_mask = targets != ignore_index
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]

    indices = targets * num_classes + predictions
    confusion_flat = torch.bincount(indices, minlength=num_classes * num_classes)
    return confusion_flat.reshape(num_classes, num_classes)


class SegmentationMetrics:
    """Evaluator for Semantic Segmentation metrics (Dice, IoU, Precision, Recall).

    Accumulates a confusion matrix across validation batches and evaluates
    performance using macro-averaged Dice, IoU, precision, and recall.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        eps: float = 1e-16,
    ) -> None:
        """Initialize SegmentationMetrics evaluator.

        Args:
            num_classes: Total number of segmentation classes.
            ignore_index: Class index to ignore (e.g. void/unlabeled pixels).
            eps: Small constant for numerical stability.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

        self._confusion_matrix = torch.zeros(
            num_classes,
            num_classes,
            dtype=torch.long,
        )

    def update(self, logits: Tensor, masks: Tensor) -> None:
        """Update metric state with a new batch of predictions and ground truths.

        Args:
            logits: Model output logits of shape ``(B, C, H, W)``.
            masks: Ground-truth segmentation masks of shape ``(B, H, W)``
                with dtype ``torch.long``.
        """
        # Build confusion matrix on the original device (GPU) to avoid PCIe bottleneck
        predictions = logits.detach().argmax(dim=1).long()
        masks_long = masks.detach().long()

        batch_confusion = _build_confusion_matrix(
            predictions.flatten(),
            masks_long.flatten(),
            self.num_classes,
            self.ignore_index,
        )
        # Transfer only the fixed-size confusion matrix to CPU for accumulation
        self._confusion_matrix += batch_confusion.cpu()

    def reset(self) -> None:
        """Reset the accumulated confusion matrix."""
        self._confusion_matrix.zero_()

    def compute(self) -> dict[str, float]:
        """Compute evaluation metrics over all accumulated data.

        Returns:
            Dictionary containing computed metric values:
            - ``dice``: Macro-averaged Dice score across all classes.
            - ``iou``: Macro-averaged Intersection over Union across all classes.
            - ``precision``: Macro-averaged precision across all classes.
            - ``recall``: Macro-averaged recall across all classes.
        """
        confusion = self._confusion_matrix.float()

        true_positives = confusion.diag()
        false_positives = confusion.sum(dim=0) - true_positives
        false_negatives = confusion.sum(dim=1) - true_positives

        present_classes = (true_positives + false_negatives) > 0

        if not present_classes.any():
            return {
                "dice": 0.0,
                "iou": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        dice_per_class = (2.0 * true_positives) / (
            2.0 * true_positives + false_positives + false_negatives + self.eps
        )
        iou_per_class = true_positives / (
            true_positives + false_positives + false_negatives + self.eps
        )
        precision_per_class = true_positives / (true_positives + false_positives + self.eps)
        recall_per_class = true_positives / (true_positives + false_negatives + self.eps)

        macro_dice = dice_per_class[present_classes].mean().item()
        macro_iou = iou_per_class[present_classes].mean().item()
        macro_precision = precision_per_class[present_classes].mean().item()
        macro_recall = recall_per_class[present_classes].mean().item()

        return {
            "dice": round(macro_dice, 4),
            "iou": round(macro_iou, 4),
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
        }


if __name__ == "__main__":
    torch.manual_seed(42)
    num_classes = 5
    batch_size = 4
    height = 32
    width = 32

    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=255)

    for _ in range(3):
        logits = torch.randn(batch_size, num_classes, height, width)
        masks = torch.randint(0, num_classes, (batch_size, height, width))
        masks[0, :4, :4] = 255  # simulate ignore_index pixels
        metrics.update(logits, masks)

    results = metrics.compute()

    print("--- Segmentation Metrics Sanity Test ---")  # noqa: T201
    for key, value in results.items():
        print(f"  {key:12s}: {value:.4f}")  # noqa: T201

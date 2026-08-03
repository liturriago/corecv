"""Image classification evaluation metrics: Precision, Recall, and Top-K Accuracy.

Computes macro-averaged precision, recall, and top-k accuracy across all classes
by accumulating predictions and labels across validation batches.

Typical usage::

    metrics = ClassificationMetrics(num_classes=10)

    for batch in val_loader:
        images = batch["images"]
        labels = batch["labels"]
        logits = model(images)

        metrics.update(logits, labels)

    results = metrics.compute()
    print(results)
    # {"precision": 0.85, "recall": 0.83, "top1_acc": 0.84, "top5_acc": 0.96}
    metrics.reset()
"""

from __future__ import annotations

import torch
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


def _topk_accuracy(
    logits: Tensor,
    labels: Tensor,
    top_k: int,
) -> Tensor:
    """Compute top-k accuracy for a batch of predictions.

    Args:
        logits: Predicted logits of shape ``(B, C)``.
        labels: Ground-truth class indices of shape ``(B,)``.
        top_k: Number of top predictions to consider.

    Returns:
        Boolean tensor of shape ``(B,)`` indicating correct predictions.
    """
    _, topk_indices = logits.topk(top_k, dim=1, largest=True, sorted=True)
    correct = topk_indices.eq(labels.unsqueeze(1).expand_as(topk_indices))
    return correct.any(dim=1)


def _build_classification_confusion_matrix(
    predictions: Tensor,
    targets: Tensor,
    num_classes: int,
) -> Tensor:
    """Build a confusion matrix from predictions and targets.

    Args:
        predictions: Predicted class indices of shape ``(B,)`` with dtype ``torch.long``.
        targets: Ground-truth class indices of shape ``(B,)`` with dtype ``torch.long``.
        num_classes: Total number of classes.

    Returns:
        Confusion matrix of shape ``(num_classes, num_classes)`` where entry
        ``[i, j]`` is the count of samples with prediction ``i`` and target ``j``.
    """
    indices = targets * num_classes + predictions
    confusion_flat = torch.bincount(indices, minlength=num_classes * num_classes)
    return confusion_flat.reshape(num_classes, num_classes)


class ClassificationMetrics:
    """Evaluator for Image Classification metrics (Precision, Recall, Top-K Accuracy).

    Accumulates a confusion matrix and top-k counters across validation batches
    using O(1) memory, then evaluates performance using macro-averaged precision,
    recall, and top-k accuracy.
    """

    def __init__(
        self,
        num_classes: int,
        top_k: tuple[int, ...] = (1, 5),
        eps: float = 1e-16,
    ) -> None:
        """Initialize ClassificationMetrics evaluator.

        Args:
            num_classes: Total number of classification classes.
            top_k: Tuple of k values for top-k accuracy computation.
            eps: Small constant for numerical stability.
        """
        self.num_classes = num_classes
        self.top_k = top_k
        self.eps = eps

        # O(1) state: confusion matrix and top-k counters
        self._confusion_matrix = torch.zeros(
            num_classes,
            num_classes,
            dtype=torch.long,
        )
        self._topk_correct_counts: dict[int, int] = dict.fromkeys(top_k, 0)
        self._total_samples: int = 0

    def update(self, logits: Tensor, labels: Tensor) -> None:
        """Update metric state with a new batch of predictions and ground truths.

        Args:
            logits: Model output logits of shape ``(B, C)``.
            labels: Ground-truth class indices of shape ``(B,)`` with dtype ``torch.long``.
        """
        # Compute predictions on the original device (GPU) to avoid PCIe bottleneck
        preds = logits.detach().argmax(dim=1).long()
        labels_long = labels.detach().long()

        # Build confusion matrix on GPU, then transfer only the fixed-size matrix to CPU
        batch_confusion = _build_classification_confusion_matrix(
            preds,
            labels_long,
            self.num_classes,
        )
        self._confusion_matrix += batch_confusion.cpu()

        # Accumulate top-k correct counts incrementally
        batch_size = labels.shape[0]
        self._total_samples += batch_size

        for k_value in self.top_k:
            if k_value <= self.num_classes:
                correct = _topk_accuracy(logits.detach(), labels_long, k_value)
                self._topk_correct_counts[k_value] += correct.sum().item()

    def reset(self) -> None:
        """Reset all accumulated counters."""
        self._confusion_matrix.zero_()
        self._topk_correct_counts = dict.fromkeys(self.top_k, 0)
        self._total_samples = 0

    def compute(self) -> dict[str, float]:
        """Compute evaluation metrics over all accumulated data.

        Returns:
            Dictionary containing computed metric values:
            - ``precision``: Macro-averaged precision across all classes.
            - ``recall``: Macro-averaged recall across all classes.
            - ``top1_acc``: Top-1 accuracy.
            - ``top5_acc``: Top-5 accuracy (only if ``num_classes >= 5``).
        """
        if self._total_samples == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "top1_acc": 0.0,
                "top5_acc": 0.0,
            }

        # Compute precision and recall from confusion matrix
        confusion = self._confusion_matrix.float()

        true_positives = confusion.diag()
        false_positives = confusion.sum(dim=0) - true_positives
        false_negatives = confusion.sum(dim=1) - true_positives

        present_classes = (true_positives + false_negatives) > 0

        precision_per_class = true_positives / (true_positives + false_positives + self.eps)
        recall_per_class = true_positives / (true_positives + false_negatives + self.eps)

        num_present = present_classes.sum().item()
        if num_present > 0:
            macro_precision = precision_per_class[present_classes].mean().item()
            macro_recall = recall_per_class[present_classes].mean().item()
        else:
            macro_precision = 0.0
            macro_recall = 0.0

        self.results: dict[str, float] = {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
        }

        # Compute top-k accuracy from accumulated counters
        for k_value in self.top_k:
            if k_value <= self.num_classes:
                accuracy = self._topk_correct_counts[k_value] / self._total_samples
                self.results[f"top{k_value}_acc"] = round(accuracy, 4)

        if f"top{1}_acc" not in self.results:
            self.results["top1_acc"] = 0.0
        if f"top{5}_acc" not in self.results:
            self.results["top5_acc"] = 0.0

        return self.results
    
    def print_results(self, stage: str) -> None:
        """Print the evaluation results for a given stage.

        Args:
            stage: Name of the stage (train, val, test).
        
        Returns:
            None
        """

        logger.info(
                "%s | precision=%.4f recall=%.4f top1_acc=%.4f top5_acc=%.4f",
                stage,
                self.results["precision"],
                self.results["recall"],
                self.results["top1_acc"],
                self.results["top5_acc"],)
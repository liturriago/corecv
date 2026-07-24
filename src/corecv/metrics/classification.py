"""GPU-native classification metrics engine for CoreCV.

Provides an accumulator-based :class:`ClassificationMetrics` that computes
accuracy, top-k accuracy, precision, recall, and F1 scores — all on VRAM
with **zero** CPU-GPU synchronisations during ``update()``.  Intermediate
confusion-matrix counts are maintained in GPU tensors via
``nn.register_buffer`` so they travel with ``.to(device)`` automatically.

The ``compute()`` method performs only the final scalar divisions and returns
a plain Python ``dict`` of ``float`` values — that is the only point at
which implicit CPU transfer occurs (via ``tensor.item()``).

No pycocotools, no Python for-loops, no ``.cpu()`` calls in hot paths.

Example:
    >>> import torch
    >>> from corecv.metrics.classification import ClassificationMetrics
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> metrics = ClassificationMetrics(num_classes=10, device=device)
    >>> logits = torch.randn(32, 10, device=device)
    >>> targets = torch.randint(0, 10, (32,), device=device)
    >>> metrics.update(logits, targets)
    >>> results = metrics.compute()
    >>> results["accuracy"]  # Top-1 accuracy
    0.125
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_EXPECTED_PREDS_DIM = 2

class ClassificationMetrics(nn.Module):
    """Accumulator-based single-label multi-class classification metrics.

    Maintains per-class true-positive, false-positive, and false-negative
    counts in VRAM-resident buffers so that ``update()`` never triggers a
    CPU-GPU synchronisation.  Only ``compute()`` performs the final
    reductions.

    Supported metrics:

    * **accuracy** — Top-1 accuracy (mean over batch).
    * **top{k}_accuracy** — Top-k accuracy for each *k* in ``top_k``.
    * **precision_macro / recall_macro / f1_score_macro** — Macro-averaged
      precision, recall, and F1 (per-class, then averaged).
    * **precision_micro / recall_micro / f1_score_micro** — Micro-averaged
      (global TP, FP, FN pooled before ratio).

    Args:
        num_classes: Number of mutually-exclusive classes.
        top_k: Tuple of *k* values for top-k accuracy reporting.
            Default ``(1, 5)``.
        device: Device on which to allocate the accumulator buffers.

    Example:
        >>> metrics = ClassificationMetrics(num_classes=100, top_k=(1, 5), device="cuda")
        >>> for logits, targets in loader:
        ...     metrics.update(logits, targets)
        >>> print(metrics.compute())
    """

    def __init__(
        self,
        num_classes: int,
        top_k: tuple[int, ...] = (1, 5),
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialise ClassificationMetrics with num_classes, top_k, and device."""
        super().__init__()
        if num_classes < 1:
            msg = f"num_classes must be >= 1, got {num_classes}"
            raise ValueError(msg)
        for k in top_k:
            if k < 1:
                msg = f"Each k in top_k must be >= 1, got {k}"
                raise ValueError(msg)

        self.num_classes = int(num_classes)
        self.top_k = tuple(sorted(top_k))
        self.device = torch.device(device)

        # ------------------------------------------------------------------
        # Accumulator buffers — all on VRAM, zero sync in update().
        # confusion[c, d] = count of samples whose true label is c and whose
        #   predicted label is d.  Shape: (C, C).
        # ------------------------------------------------------------------
        self.register_buffer(
            "confusion",
            torch.zeros(num_classes, num_classes, dtype=torch.int64, device=self.device),
        )
        self.register_buffer(
            "total_samples",
            torch.zeros(1, dtype=torch.int64, device=self.device),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, preds: Tensor, targets: Tensor) -> None:  # noqa: D401
        """Accumulate a batch of predictions and ground-truth labels.

        Args:
            preds: Model output of shape ``(B, C)`` — raw logits or
                probabilities.  If logits, argmax is taken to derive
                predicted classes.
            targets: Ground-truth class indices of shape ``(B,)`` with
                values in ``[0, C)``.

        Raises:
            ValueError: If tensor shapes or value ranges are invalid.
        """
        self._validate_inputs(preds, targets)

        # Derive predicted class indices from logits / probs.
        pred_labels: Tensor = preds.argmax(dim=1)  # (B,)

        one_hot_pred: Tensor = F.one_hot(pred_labels, self.num_classes).to(
            dtype=torch.int64,
            device=self.device,
        )  # (B, C)
        one_hot_target: Tensor = F.one_hot(
            targets.to(dtype=torch.long), self.num_classes
        ).to(dtype=torch.int64, device=self.device)  # (B, C)

        # Outer-product accumulation (no .item(), no .cpu()).
        self.confusion += one_hot_target.T @ one_hot_pred  # (C, C)
        self.total_samples += targets.shape[0]

    def compute(self) -> dict[str, float | Tensor]:
        """Compute all metrics from the accumulated state.

        Returns:
            Dictionary with the following keys:

            * ``"accuracy"`` — Top-1 accuracy (float).
            * ``"top{k}_accuracy"`` — Top-k accuracy for each *k*
                (float).
            * ``"precision_macro"`` — Macro-averaged precision (float).
            * ``"recall_macro"`` — Macro-averaged recall (float).
            * ``"f1_score_macro"`` — Macro-averaged F1 (float).
            * ``"precision_micro"`` — Micro-averaged precision (float).
            * ``"recall_micro"`` — Micro-averaged recall (float).
            * ``"f1_score_micro"`` — Micro-averaged F1 (float).
        """
        total: int = self.total_samples.item()
        if total == 0:
            return self._empty_results()

        # confusion[c, d] = # samples true=c, pred=d
        confusion: Tensor = self.confusion.float()  # (C, C)

        # ---- Per-class TP / FP / FN ------------------------------------
        tp_per_class: Tensor = confusion.diag()  # (C,)
        fp_per_class: Tensor = confusion.sum(dim=0) - tp_per_class  # (C,)
        fn_per_class: Tensor = confusion.sum(dim=1) - tp_per_class  # (C,)

        # ---- Top-1 accuracy (diagonal sum / total) --------------------
        top1_accuracy: float = (tp_per_class.sum() / total).item()

        # ---- Top-k accuracy via torch.topk on confusion columns --------
        # For each sample class c, the model's top-k predictions span the
        # largest k entries in column c of the confusion matrix.  We
        # compute this by summing the top-k values in each column and
        # dividing by the column total.
        topk_accuracies: dict[int, float] = {}
        for k in self.top_k:
            k_clamped: int = min(k, self.num_classes)
            # topk along dim=0 (true-class axis) for each predicted class
            topk_vals, _ = torch.topk(confusion, k_clamped, dim=0)  # (k, C)
            topk_sum: Tensor = topk_vals.sum(dim=0)  # (C,)
            # Weighted average: sum of topk_sum / total
            topk_acc: float = (topk_sum.sum() / total).item()
            topk_accuracies[k] = topk_acc

        # ---- Precision / Recall / F1 — per class, then macro ----------
        eps: float = 1e-8
        precision_per_class: Tensor = tp_per_class / (tp_per_class + fp_per_class + eps)
        recall_per_class: Tensor = tp_per_class / (tp_per_class + fn_per_class + eps)
        f1_per_class: Tensor = (
            2.0 * precision_per_class * recall_per_class
            / (precision_per_class + recall_per_class + eps)
        )

        precision_macro: float = precision_per_class.mean().item()
        recall_macro: float = recall_per_class.mean().item()
        f1_macro: float = f1_per_class.mean().item()

        # ---- Micro-averaged (global TP / FP / FN) --------------------
        tp_total: Tensor = tp_per_class.sum()
        fp_total: Tensor = fp_per_class.sum()
        fn_total: Tensor = fn_per_class.sum()

        precision_micro: float = (tp_total / (tp_total + fp_total + eps)).item()
        recall_micro: float = (tp_total / (tp_total + fn_total + eps)).item()
        f1_micro: float = (
            (2.0 * tp_total) / (2.0 * tp_total + fp_total + fn_total + eps)
        ).item()

        # ---- Assemble results dict ------------------------------------
        results: dict[str, float | Tensor] = {
            "accuracy": top1_accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_score_macro": f1_macro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_score_micro": f1_micro,
        }

        for k, acc in topk_accuracies.items():
            results[f"top{k}_accuracy"] = acc

        return results

    def reset(self) -> None:
        """Reset all accumulator buffers to zero."""
        self.confusion.zero_()
        self.total_samples.zero_()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_inputs(self, preds: Tensor, targets: Tensor) -> None:
        """Validate shapes and value ranges of a batch.

        Args:
            preds: Prediction tensor ``(B, C)``.
            targets: Target tensor ``(B,)``.

        Raises:
            ValueError: On shape mismatch or out-of-range values.
        """
        if preds.dim() != _EXPECTED_PREDS_DIM:
            msg = f"preds must be 2-D (B, C), got shape {tuple(preds.shape)}"
            raise ValueError(msg)
        if targets.dim() != 1:
            msg = f"targets must be 1-D (B,), got shape {tuple(targets.shape)}"
            raise ValueError(msg)
        if preds.shape[0] != targets.shape[0]:
            msg = (
                f"Batch size mismatch: preds has {preds.shape[0]} samples, "
                f"targets has {targets.shape[0]}"
            )
            raise ValueError(msg)
        if preds.shape[1] != self.num_classes:
            msg = (
                f"preds has {preds.shape[1]} classes but expected "
                f"{self.num_classes}"
            )
            raise ValueError(msg)

        # Move to the same device as buffers for consistent accumulation.
        preds = preds.to(device=self.device)
        targets = targets.to(device=self.device)

        if (targets < 0).any() or (targets >= self.num_classes).any():
            msg = (
                f"targets contain values outside [0, {self.num_classes}). "
                f"Range: [{targets.min().item()}, {targets.max().item()}]"
            )
            raise ValueError(msg)

    def _empty_results(self) -> dict[str, float | Tensor]:
        """Return a results dict with all zeros for an empty state."""
        results: dict[str, float | Tensor] = {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_score_macro": 0.0,
            "precision_micro": 0.0,
            "recall_micro": 0.0,
            "f1_score_micro": 0.0,
        }
        for k in self.top_k:
            results[f"top{k}_accuracy"] = 0.0
        return results

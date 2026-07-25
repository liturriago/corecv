"""Evaluation engines for classification, segmentation, and detection tasks.

Provides :class:`ClassificationEvaluator`, :class:`SegmentationEvaluator`, and
:class:`DetectionEvaluator` for running inference-only evaluation loops with
GPU-native metric computation and loss tracking.

Example::

    from corecv.engine.evaluation import ClassificationEvaluator

    evaluator = ClassificationEvaluator(
        model=model,
        test_loader=test_loader,
        loss_fn=loss_fn,
        num_classes=10,
    )
    results = evaluator.evaluate()
    print(results)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from corecv.engine.train import _batch_to_device
from corecv.losses.loss_detection import DualHeadDetectionLoss
from corecv.metrics.metric_classification import ClassificationMetrics
from corecv.metrics.metric_detection import DetectionMetrics
from corecv.metrics.metric_segmentation import SegmentationMetrics

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification Evaluator
# ---------------------------------------------------------------------------


class ClassificationEvaluator:
    """Evaluator for image classification models.

    Runs a full inference pass over a test DataLoader, computes loss and
    classification metrics (precision, recall, top-k accuracy), and returns
    all results as a flat dictionary.

    Attributes:
        model: The classification model moved to the target device.
        test_loader: DataLoader yielding dicts with ``"images"`` and
            ``"labels"`` keys.
        loss_fn: Loss function accepting ``(logits, labels)``.
        num_classes: Number of classification classes.
        device: Target device for computation.

    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        loss_fn: nn.Module,
        num_classes: int,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize the classification evaluator.

        Args:
            model: Classification model mapping ``(B, 3, H, W)`` to
                ``(B, num_classes)``.
            test_loader: Test data loader yielding dicts with ``"images"``
                and ``"labels"`` keys.
            loss_fn: Loss function accepting ``(logits, labels)`` and
                returning a scalar loss tensor.
            num_classes: Number of target classification classes.
            device: Target device string or :class:`torch.device`.

        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.device = torch.device(device)

    def evaluate(self) -> dict[str, float | Tensor]:
        """Run evaluation on the test set and return computed metrics.

        Sets the model to evaluation mode, iterates over all batches with
        gradients disabled, accumulates loss and metrics, then returns a
        summary dictionary.

        Returns:
            Dictionary containing:

            - ``loss``: Mean loss across all batches.
            - ``precision``: Macro-averaged precision.
            - ``recall``: Macro-averaged recall.
            - ``top1_acc``: Top-1 accuracy.
            - ``top5_acc``: Top-5 accuracy.
            - ``confusion_matrix``: Confusion matrix of shape
              ``(num_classes, num_classes)``.
            - ``f1_curve``: Dictionary with F1-score curve data.

        """
        self.model.eval()
        metrics = ClassificationMetrics(num_classes=self.num_classes)
        metrics.reset()

        total_loss = 0.0
        num_batches = 0
        all_preds: list[Tensor] = []
        all_labels: list[Tensor] = []
        all_probs: list[Tensor] = []

        with torch.no_grad():
            for raw_batch in self.test_loader:
                batch = _batch_to_device(raw_batch, self.device)
                images = batch["images"]
                labels = batch["labels"]

                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

                metrics.update(logits, labels)
                total_loss += loss.item()
                num_batches += 1

                # Accumulate predictions for confusion matrix and F1 curve
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())

        results: dict[str, float | Tensor] = metrics.compute()
        results["loss"] = total_loss / max(num_batches, 1)

        # Compute confusion matrix
        preds_tensor = torch.cat(all_preds, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)
        probs_tensor = torch.cat(all_probs, dim=0)
        results["confusion_matrix"] = self._compute_confusion_matrix(
            preds_tensor,
            labels_tensor,
        )
        results["f1_curve"] = self._compute_f1_curve(probs_tensor, labels_tensor)

        logger.info(
            "Classification evaluation | loss=%.4f precision=%.4f "
            "recall=%.4f top1_acc=%.4f top5_acc=%.4f",
            results["loss"],
            results["precision"],
            results["recall"],
            results["top1_acc"],
            results["top5_acc"],
        )
        return results

    def _compute_confusion_matrix(self, preds: Tensor, labels: Tensor) -> Tensor:
        """Compute confusion matrix from predictions and labels.

        Args:
            preds: Predicted class indices of shape ``(N,)``.
            labels: Ground truth class indices of shape ``(N,)``.

        Returns:
            Confusion matrix of shape ``(num_classes, num_classes)`` where
            entry ``[i, j]`` is the count of samples with true label ``i``
            predicted as ``j``.

        """
        confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        for true_label in range(self.num_classes):
            mask = labels == true_label
            if mask.any():
                pred_counts = torch.bincount(preds[mask], minlength=self.num_classes)
                confusion[true_label] = pred_counts[: self.num_classes]
        return confusion

    def _compute_f1_curve(self, probs: Tensor, labels: Tensor) -> dict[str, Tensor]:
        """Compute F1-score curve at different confidence thresholds.

        Args:
            probs: Predicted probabilities of shape ``(N, num_classes)``.
            labels: Ground truth class indices of shape ``(N,)``.

        Returns:
            Dictionary with keys:
            - ``thresholds``: Tensor of threshold values.
            - ``f1_per_class``: F1 scores per class at each threshold.
            - ``f1_macro``: Macro-averaged F1 at each threshold.

        """
        thresholds = torch.linspace(0.0, 1.0, 11)
        max_probs, preds = probs.max(dim=1)

        f1_per_class = torch.zeros(11, self.num_classes)
        f1_macro = torch.zeros(11)

        for idx, threshold in enumerate(thresholds):
            # Filter predictions by confidence threshold
            mask = max_probs >= threshold
            if not mask.any():
                continue

            filtered_preds = preds[mask]
            filtered_labels = labels[mask]

            # Compute F1 per class
            for cls in range(self.num_classes):
                tp = ((filtered_preds == cls) & (filtered_labels == cls)).sum().float()
                fp = ((filtered_preds == cls) & (filtered_labels != cls)).sum().float()
                fn = ((filtered_preds != cls) & (filtered_labels == cls)).sum().float()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                f1_per_class[idx, cls] = f1

            f1_macro[idx] = f1_per_class[idx].mean()

        return {
            "thresholds": thresholds,
            "f1_per_class": f1_per_class,
            "f1_macro": f1_macro,
        }

    def plot_confusion_matrix(
        self,
        confusion_matrix: Tensor,
        class_names: list[str] | None = None,
        save_path: str | Path | None = None,
        *,
        show: bool = True,
    ) -> None:
        """Plot confusion matrix as a heatmap.

        Args:
            confusion_matrix: Confusion matrix tensor of shape ``(num_classes, num_classes)``.
            class_names: Optional list of class names for axis labels.
            save_path: Optional path to save the figure. If None, figure is not saved.
            show: Whether to display the plot. Defaults to True.

        """
        cm_np = confusion_matrix.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_np, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        if class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]

        ax.set(
            xticks=np.arange(cm_np.shape[1]),
            yticks=np.arange(cm_np.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fmt = "d" if cm_np.max() > 0 else ""
        thresh = cm_np.max() / 2.0
        for i in range(cm_np.shape[0]):
            for j in range(cm_np.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm_np[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm_np[i, j] > thresh else "black",
                )

        fig.tight_layout()
        ax.set_title("Confusion Matrix")

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Confusion matrix saved to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_f1_curve(
        self,
        f1_curve: dict[str, Tensor],
        save_path: str | Path | None = None,
        *,
        show: bool = True,
    ) -> None:
        """Plot F1-score curve across confidence thresholds.

        Args:
            f1_curve: Dictionary with keys ``"thresholds"``, ``"f1_per_class"``,
                and ``"f1_macro"``.
            save_path: Optional path to save the figure. If None, figure is not saved.
            show: Whether to display the plot. Defaults to True.

        """
        thresholds = f1_curve["thresholds"].cpu().numpy()
        f1_macro = f1_curve["f1_macro"].cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, f1_macro, "b-", linewidth=2, label="Macro F1")
        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_title("F1-Score Curve")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(visible=True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower left")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("F1 curve saved to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Segmentation Evaluator
# ---------------------------------------------------------------------------


class SegmentationEvaluator:
    """Evaluator for semantic segmentation models.

    Runs a full inference pass over a test DataLoader, computes loss and
    segmentation metrics (Dice, IoU, precision, recall), and returns all
    results as a flat dictionary.

    Attributes:
        model: The segmentation model moved to the target device.
        test_loader: DataLoader yielding dicts with ``"images"`` and
            ``"masks"`` keys.
        loss_fn: Loss function accepting ``(logits, masks)``.
        num_classes: Number of segmentation classes.
        device: Target device for computation.

    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        loss_fn: nn.Module,
        num_classes: int,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize the segmentation evaluator.

        Args:
            model: Segmentation model mapping ``(B, 3, H, W)`` to
                ``(B, num_classes, H, W)``.
            test_loader: Test data loader yielding dicts with ``"images"``
                and ``"masks"`` keys.
            loss_fn: Loss function accepting ``(logits, masks)`` and
                returning a scalar loss tensor.
            num_classes: Number of target segmentation classes.
            device: Target device string or :class:`torch.device`.

        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.device = torch.device(device)

    def evaluate(self) -> dict[str, float | Tensor]:
        """Run evaluation on the test set and return computed metrics.

        Sets the model to evaluation mode, iterates over all batches with
        gradients disabled, accumulates loss and metrics, then returns a
        summary dictionary.

        Returns:
            Dictionary containing:

            - ``loss``: Mean loss across all batches.
            - ``dice``: Macro-averaged Dice score.
            - ``iou``: Macro-averaged Intersection over Union.
            - ``precision``: Macro-averaged precision.
            - ``recall``: Macro-averaged recall.
            - ``confusion_matrix``: Confusion matrix of shape
              ``(num_classes, num_classes)``.
            - ``f1_curve``: Dictionary with F1-score curve data.

        """
        self.model.eval()
        metrics = SegmentationMetrics(num_classes=self.num_classes)
        metrics.reset()

        total_loss = 0.0
        num_batches = 0
        all_preds: list[Tensor] = []
        all_masks: list[Tensor] = []
        all_probs: list[Tensor] = []

        with torch.no_grad():
            for raw_batch in self.test_loader:
                batch = _batch_to_device(raw_batch, self.device)
                images = batch["images"]
                masks = batch["masks"]

                logits = self.model(images)
                loss = self.loss_fn(logits, masks)

                metrics.update(logits, masks)
                total_loss += loss.item()
                num_batches += 1

                # Accumulate predictions for confusion matrix and F1 curve
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_masks.append(masks.cpu())
                all_probs.append(probs.cpu())

        results: dict[str, float | Tensor] = metrics.compute()
        results["loss"] = total_loss / max(num_batches, 1)

        # Compute confusion matrix
        preds_tensor = torch.cat(all_preds, dim=0)
        masks_tensor = torch.cat(all_masks, dim=0)
        probs_tensor = torch.cat(all_probs, dim=0)
        results["confusion_matrix"] = self._compute_confusion_matrix(
            preds_tensor,
            masks_tensor,
        )
        results["f1_curve"] = self._compute_f1_curve(probs_tensor, masks_tensor)

        logger.info(
            "Segmentation evaluation | loss=%.4f dice=%.4f iou=%.4f precision=%.4f recall=%.4f",
            results["loss"],
            results["dice"],
            results["iou"],
            results["precision"],
            results["recall"],
        )
        return results

    def _compute_confusion_matrix(self, preds: Tensor, masks: Tensor) -> Tensor:
        """Compute confusion matrix from predictions and masks.

        Args:
            preds: Predicted class indices of shape ``(B, H, W)``.
            masks: Ground truth class indices of shape ``(B, H, W)``.

        Returns:
            Confusion matrix of shape ``(num_classes, num_classes)`` where
            entry ``[i, j]`` is the count of pixels with true label ``i``
            predicted as ``j``.

        """
        confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        preds_flat = preds.flatten()
        masks_flat = masks.flatten()

        for true_label in range(self.num_classes):
            mask = masks_flat == true_label
            if mask.any():
                pred_counts = torch.bincount(
                    preds_flat[mask],
                    minlength=self.num_classes,
                )
                confusion[true_label] = pred_counts[: self.num_classes]
        return confusion

    def _compute_f1_curve(self, probs: Tensor, masks: Tensor) -> dict[str, Tensor]:
        """Compute F1-score curve at different confidence thresholds.

        Args:
            probs: Predicted probabilities of shape ``(B, num_classes, H, W)``.
            masks: Ground truth class indices of shape ``(B, H, W)``.

        Returns:
            Dictionary with keys:
            - ``thresholds``: Tensor of threshold values.
            - ``f1_per_class``: F1 scores per class at each threshold.
            - ``f1_macro``: Macro-averaged F1 at each threshold.

        """
        thresholds = torch.linspace(0.0, 1.0, 11)
        max_probs, preds = probs.max(dim=1)

        # Flatten spatial dimensions
        max_probs_flat = max_probs.flatten()
        preds_flat = preds.flatten()
        masks_flat = masks.flatten()

        f1_per_class = torch.zeros(11, self.num_classes)
        f1_macro = torch.zeros(11)

        for idx, threshold in enumerate(thresholds):
            # Filter predictions by confidence threshold
            mask = max_probs_flat >= threshold
            if not mask.any():
                continue

            filtered_preds = preds_flat[mask]
            filtered_masks = masks_flat[mask]

            # Compute F1 per class
            for cls in range(self.num_classes):
                tp = ((filtered_preds == cls) & (filtered_masks == cls)).sum().float()
                fp = ((filtered_preds == cls) & (filtered_masks != cls)).sum().float()
                fn = ((filtered_preds != cls) & (filtered_masks == cls)).sum().float()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                f1_per_class[idx, cls] = f1

            f1_macro[idx] = f1_per_class[idx].mean()

        return {
            "thresholds": thresholds,
            "f1_per_class": f1_per_class,
            "f1_macro": f1_macro,
        }

    def plot_confusion_matrix(
        self,
        confusion_matrix: Tensor,
        class_names: list[str] | None = None,
        save_path: str | Path | None = None,
        *,
        show: bool = True,
    ) -> None:
        """Plot confusion matrix as a heatmap.

        Args:
            confusion_matrix: Confusion matrix tensor of shape ``(num_classes, num_classes)``.
            class_names: Optional list of class names for axis labels.
            save_path: Optional path to save the figure. If None, figure is not saved.
            show: Whether to display the plot. Defaults to True.

        """
        cm_np = confusion_matrix.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_np, interpolation="nearest", cmap=plt.cm.Greens)
        ax.figure.colorbar(im, ax=ax)

        if class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]

        ax.set(
            xticks=np.arange(cm_np.shape[1]),
            yticks=np.arange(cm_np.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fmt = "d" if cm_np.max() > 0 else ""
        thresh = cm_np.max() / 2.0
        for i in range(cm_np.shape[0]):
            for j in range(cm_np.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm_np[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm_np[i, j] > thresh else "black",
                )

        fig.tight_layout()
        ax.set_title("Segmentation Confusion Matrix")

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Confusion matrix saved to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_f1_curve(
        self,
        f1_curve: dict[str, Tensor],
        save_path: str | Path | None = None,
        *,
        show: bool = True,
    ) -> None:
        """Plot F1-score curve across confidence thresholds.

        Args:
            f1_curve: Dictionary with keys ``"thresholds"``, ``"f1_per_class"``,
                and ``"f1_macro"``.
            save_path: Optional path to save the figure. If None, figure is not saved.
            show: Whether to display the plot. Defaults to True.

        """
        thresholds = f1_curve["thresholds"].cpu().numpy()
        f1_macro = f1_curve["f1_macro"].cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, f1_macro, "g-", linewidth=2, label="Macro F1")
        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_title("Segmentation F1-Score Curve")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(visible=True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower left")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("F1 curve saved to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Detection Evaluator
# ---------------------------------------------------------------------------


class DetectionEvaluator:
    """Evaluator for anchor-free dual-head object detection models.

    Runs a full inference pass over a test DataLoader using the One-to-One
    (O2O) head for metric computation.  All loss and metric calculations
    run entirely on GPU via ``torchvision.ops`` with no CPU round-trips.

    Attributes:
        model: The detection model moved to the target device.
        test_loader: DataLoader yielding dicts with ``"images"`` and
            ``"targets"`` keys.
        loss_fn: :class:`DualHeadDetectionLoss` instance.
        num_classes: Number of foreground object classes.
        device: Target device for computation.

    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        loss_fn: DualHeadDetectionLoss,
        num_classes: int,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize the detection evaluator.

        Args:
            model: Detection model mapping ``(B, 3, H, W)`` to a tuple
                of ``(preds_o2m, preds_o2o)`` predictions.
            test_loader: Test data loader yielding dicts with ``"images"``
                and ``"targets"`` keys.
            loss_fn: :class:`DualHeadDetectionLoss` instance accepting
                ``(preds_o2m, preds_o2o, targets)``.
            num_classes: Number of foreground object classes.
            device: Target device string or :class:`torch.device`.

        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.device = torch.device(device)

    @staticmethod
    def _format_o2o_predictions(
        logits_o2o: Tensor,
        boxes_o2o: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Convert raw O2O head outputs to metric-compatible format.

        Applies sigmoid to logits and extracts per-anchor class scores
        and labels via argmax over classes.

        Args:
            logits_o2o: Raw O2O logits of shape ``(B, N, C)``.
            boxes_o2o: Predicted boxes of shape ``(B, N, 4)``.

        Returns:
            Tuple of ``(boxes, scores, labels)`` with shapes
            ``(B, N, 4)``, ``(B, N)``, and ``(B, N)``.

        """
        scores_all = logits_o2o.sigmoid()
        max_scores, pred_labels = scores_all.max(dim=-1)
        return boxes_o2o, max_scores, pred_labels.long()

    def evaluate(self) -> dict[str, float | Tensor]:
        """Run evaluation on the test set and return computed metrics.

        Sets the model to evaluation mode, iterates over all batches with
        gradients disabled, accumulates loss and detection metrics, then
        returns a summary dictionary.

        Returns:
            Dictionary containing:

            - ``loss``: Mean total loss across all batches.
            - ``mAP50``: Mean Average Precision at IoU = 0.50.
            - ``mAP50-95``: Mean Average Precision across IoU thresholds.
            - ``precision``: Precision at the confidence threshold.
            - ``recall``: Recall at the confidence threshold.
            - ``f1``: F1 score at the confidence threshold.
            - ``confusion_matrix``: Confusion matrix of shape
              ``(num_classes, num_classes)``.
            - ``f1_curve``: Dictionary with F1-score curve data.

        """
        self.model.eval()
        metrics = DetectionMetrics(num_classes=self.num_classes)
        metrics.reset()

        total_loss = 0.0
        num_batches = 0
        all_pred_boxes: list[Tensor] = []
        all_pred_scores: list[Tensor] = []
        all_pred_labels: list[Tensor] = []
        all_gt_boxes: list[Tensor] = []
        all_gt_labels: list[Tensor] = []

        with torch.no_grad():
            for raw_batch in self.test_loader:
                batch = _batch_to_device(raw_batch, self.device)
                images = batch["images"]
                targets = batch["targets"]

                preds_o2m, preds_o2o = self.model(images)
                loss_dict: dict[str, Tensor] = self.loss_fn(
                    preds_o2m,
                    preds_o2o,
                    targets,
                )
                loss = loss_dict["loss_total"]

                logits_o2o, boxes_o2o = preds_o2o
                formatted = self._format_o2o_predictions(logits_o2o, boxes_o2o)
                metrics.update(formatted, targets)

                total_loss += loss.item()
                num_batches += 1

                # Accumulate predictions for confusion matrix and F1 curve
                boxes, scores, labels = formatted
                for batch_idx in range(boxes.shape[0]):
                    all_pred_boxes.append(boxes[batch_idx].cpu())
                    all_pred_scores.append(scores[batch_idx].cpu())
                    all_pred_labels.append(labels[batch_idx].cpu())

                    # Extract ground truth for this image
                    mask = targets[:, 0] == batch_idx
                    if mask.any():
                        gt_targets = targets[mask]
                        all_gt_boxes.append(gt_targets[:, 2:6].cpu())
                        all_gt_labels.append(gt_targets[:, 1].long().cpu())
                    else:
                        all_gt_boxes.append(torch.zeros(0, 4))
                        all_gt_labels.append(torch.zeros(0, dtype=torch.long))

        results: dict[str, float | Tensor] = metrics.compute()
        results["loss"] = total_loss / max(num_batches, 1)

        # Compute confusion matrix and F1 curve
        results["confusion_matrix"] = self._compute_confusion_matrix(
            all_pred_boxes,
            all_pred_scores,
            all_pred_labels,
            all_gt_boxes,
            all_gt_labels,
        )
        results["f1_curve"] = self._compute_f1_curve(
            all_pred_boxes,
            all_pred_scores,
            all_pred_labels,
            all_gt_boxes,
            all_gt_labels,
        )

        logger.info(
            "Detection evaluation | loss=%.4f mAP50=%.4f "
            "mAP50-95=%.4f precision=%.4f recall=%.4f f1=%.4f",
            results["loss"],
            results["mAP50"],
            results["mAP50-95"],
            results["precision"],
            results["recall"],
            results["f1"],
        )
        return results

    def _compute_confusion_matrix(
        self,
        pred_boxes: list[Tensor],
        pred_scores: list[Tensor],
        pred_labels: list[Tensor],
        gt_boxes: list[Tensor],
        gt_labels: list[Tensor],
    ) -> Tensor:
        """Compute confusion matrix from predictions and ground truth.

        Args:
            pred_boxes: List of predicted boxes per image.
            pred_scores: List of predicted scores per image.
            pred_labels: List of predicted labels per image.
            gt_boxes: List of ground truth boxes per image.
            gt_labels: List of ground truth labels per image.

        Returns:
            Confusion matrix of shape ``(num_classes, num_classes)`` where
            entry ``[i, j]`` is the count of detections with true class ``i``
            predicted as ``j``.

        """
        confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        iou_threshold = 0.5

        for pred_b, pred_s, pred_l, gt_b, gt_l in zip(
            pred_boxes,
            pred_scores,
            pred_labels,
            gt_boxes,
            gt_labels,
            strict=False,
        ):
            if len(gt_b) == 0:
                # No ground truth: all predictions are false positives
                # FP goes to a special "background" column (not counted in matrix)
                continue

            if len(pred_b) == 0:
                # No predictions: all ground truths are false negatives
                continue

            # Compute IoU between predictions and ground truth
            ious = box_iou(pred_b, gt_b)

            # Match predictions to ground truth
            gt_matched = torch.zeros(len(gt_b), dtype=torch.bool)

            # Sort predictions by score (descending)
            sorted_indices = pred_s.argsort(descending=True)

            for pred_idx in sorted_indices:
                pred_cls = pred_l[pred_idx]
                pred_iou = ious[pred_idx]

                # Find best matching ground truth
                pred_iou[gt_matched] = 0.0
                max_iou, best_gt_idx = pred_iou.max(dim=0)

                if max_iou >= iou_threshold:
                    true_cls = gt_l[best_gt_idx]
                    confusion[true_cls, pred_cls] += 1
                    gt_matched[best_gt_idx] = True

        return confusion

    def _compute_f1_curve(
        self,
        pred_boxes: list[Tensor],
        pred_scores: list[Tensor],
        pred_labels: list[Tensor],
        gt_boxes: list[Tensor],
        gt_labels: list[Tensor],
    ) -> dict[str, Tensor]:
        """Compute F1-score curve at different confidence thresholds.

        Args:
            pred_boxes: List of predicted boxes per image.
            pred_scores: List of predicted scores per image.
            pred_labels: List of predicted labels per image.
            gt_boxes: List of ground truth boxes per image.
            gt_labels: List of ground truth labels per image.

        Returns:
            Dictionary with keys:
            - ``thresholds``: Tensor of threshold values.
            - ``f1``: F1 score at each threshold.
            - ``precision``: Precision at each threshold.
            - ``recall``: Recall at each threshold.

        """
        thresholds = torch.linspace(0.0, 1.0, 11)
        iou_threshold = 0.5

        f1_scores = torch.zeros(11)
        precisions = torch.zeros(11)
        recalls = torch.zeros(11)

        # Count total ground truth objects
        total_gt = sum(len(gt_l) for gt_l in gt_labels)
        if total_gt == 0:
            return {
                "thresholds": thresholds,
                "f1": f1_scores,
                "precision": precisions,
                "recall": recalls,
            }

        for idx, threshold in enumerate(thresholds):
            total_tp = 0
            total_fp = 0

            for pred_b, pred_s, _pred_l, gt_b, _gt_l in zip(
                pred_boxes,
                pred_scores,
                pred_labels,
                gt_boxes,
                gt_labels,
                strict=False,
            ):
                # Filter predictions by confidence threshold
                mask = pred_s >= threshold
                if not mask.any():
                    continue

                filtered_boxes = pred_b[mask]

                if len(gt_b) == 0:
                    # No ground truth: all predictions are false positives
                    total_fp += len(filtered_boxes)
                    continue

                # Compute IoU between filtered predictions and ground truth
                ious = box_iou(filtered_boxes, gt_b)

                # Match predictions to ground truth
                gt_matched = torch.zeros(len(gt_b), dtype=torch.bool)

                # Sort by score (descending)
                sorted_indices = filtered_boxes[:, 0].argsort(descending=True)

                for pred_idx in sorted_indices:
                    pred_iou = ious[pred_idx]
                    pred_iou[gt_matched] = 0.0
                    max_iou, best_gt_idx = pred_iou.max(dim=0)

                    if max_iou >= iou_threshold:
                        total_tp += 1
                        gt_matched[best_gt_idx] = True
                    else:
                        total_fp += 1

            precision = total_tp / (total_tp + total_fp + 1e-8)
            recall = total_tp / (total_gt + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            f1_scores[idx] = f1
            precisions[idx] = precision
            recalls[idx] = recall

        return {
            "thresholds": thresholds,
            "f1": f1_scores,
            "precision": precisions,
            "recall": recalls,
        }

    def plot_confusion_matrix(
        self,
        confusion_matrix: Tensor,
        class_names: list[str] | None = None,
        save_path: str | Path | None = None,
        *,
        show: bool = True,
    ) -> None:
        """Plot confusion matrix as a heatmap.

        Args:
            confusion_matrix: Confusion matrix tensor of shape ``(num_classes, num_classes)``.
            class_names: Optional list of class names for axis labels.
            save_path: Optional path to save the figure. If None, figure is not saved.
            show: Whether to display the plot. Defaults to True.

        """
        cm_np = confusion_matrix.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_np, interpolation="nearest", cmap=plt.cm.Reds)
        ax.figure.colorbar(im, ax=ax)

        if class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]

        ax.set(
            xticks=np.arange(cm_np.shape[1]),
            yticks=np.arange(cm_np.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fmt = "d" if cm_np.max() > 0 else ""
        thresh = cm_np.max() / 2.0
        for i in range(cm_np.shape[0]):
            for j in range(cm_np.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm_np[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm_np[i, j] > thresh else "black",
                )

        fig.tight_layout()
        ax.set_title("Detection Confusion Matrix")

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Confusion matrix saved to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_f1_curve(
        self,
        f1_curve: dict[str, Tensor],
        save_path: str | Path | None = None,
        *,
        show: bool = True,
    ) -> None:
        """Plot F1-score, precision, and recall curves across confidence thresholds.

        Args:
            f1_curve: Dictionary with keys ``"thresholds"``, ``"f1"``,
                ``"precision"``, and ``"recall"``.
            save_path: Optional path to save the figure. If None, figure is not saved.
            show: Whether to display the plot. Defaults to True.

        """
        thresholds = f1_curve["thresholds"].cpu().numpy()
        f1_scores = f1_curve["f1"].cpu().numpy()
        precisions = f1_curve["precision"].cpu().numpy()
        recalls = f1_curve["recall"].cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, f1_scores, "r-", linewidth=2, label="F1")
        ax.plot(thresholds, precisions, "b--", linewidth=1.5, label="Precision")
        ax.plot(thresholds, recalls, "g--", linewidth=1.5, label="Recall")
        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Detection F1-Score Curve")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(visible=True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower left")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("F1 curve saved to %s", save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Constants for the sanity check
    _NUM_CLASSES_CLS = 4
    _NUM_CLASSES_SEG = 4
    _NUM_CLASSES_DET = 4
    _IMG_SIZE = 32
    _BATCH_SIZE = 2
    _NUM_SAMPLES = 8
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Dummy classification model and data
    # ------------------------------------------------------------------

    class _DummyClsModel(nn.Module):
        """Minimal classification model for sanity checking."""

        def __init__(self, num_classes: int) -> None:
            """Initialize with adaptive average pooling and a linear head."""
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(3, num_classes)

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass: pool then project to class logits."""
            return self.fc(self.pool(x).flatten(1))

    class _FakeClsDataset(torch.utils.data.Dataset):
        """Fake classification dataset producing random images and labels."""

        def __init__(self, n: int, num_classes: int, img_size: int) -> None:
            """Pre-generate random images and labels."""
            self.images = torch.randn(n, 3, img_size, img_size)
            self.labels = torch.randint(0, num_classes, (n,))

        def __len__(self) -> int:
            """Return the number of samples."""
            return self.labels.shape[0]

        def __getitem__(self, idx: int) -> dict[str, Tensor]:
            """Return a single sample as a dictionary."""
            return {
                "images": self.images[idx],
                "labels": self.labels[idx],
                "image_ids": torch.tensor(idx),
            }

    # ------------------------------------------------------------------
    # Dummy segmentation model and data
    # ------------------------------------------------------------------

    class _DummySegModel(nn.Module):
        """Minimal segmentation model for sanity checking."""

        def __init__(self, num_classes: int) -> None:
            """Initialize with a single 1x1 convolution head."""
            super().__init__()
            self.head = nn.Conv2d(3, num_classes, kernel_size=1)

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass: pixel-wise classification via convolution."""
            return self.head(x)

    class _FakeSegDataset(torch.utils.data.Dataset):
        """Fake segmentation dataset producing random images and masks."""

        def __init__(self, n: int, num_classes: int, img_size: int) -> None:
            """Pre-generate random images and masks."""
            self.images = torch.randn(n, 3, img_size, img_size)
            self.masks = torch.randint(0, num_classes, (n, img_size, img_size))

        def __len__(self) -> int:
            """Return the number of samples."""
            return self.images.shape[0]

        def __getitem__(self, idx: int) -> dict[str, Tensor]:
            """Return a single sample as a dictionary."""
            return {
                "images": self.images[idx],
                "masks": self.masks[idx],
                "image_ids": torch.tensor(idx),
            }

    # ------------------------------------------------------------------
    # Dummy detection model and data
    # ------------------------------------------------------------------

    class _DummyDetModel(nn.Module):
        """Minimal dual-head detection model for sanity checking."""

        def __init__(self, num_classes: int, num_anchors: int = 50) -> None:
            """Initialize with shared pooling and separate cls/box heads."""
            super().__init__()
            self.num_classes = num_classes
            self.num_anchors = num_anchors
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.cls_head = nn.Linear(3, num_anchors * num_classes)
            self.box_head = nn.Linear(3, num_anchors * 4)

        def forward(
            self,
            x: Tensor,
        ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
            """Forward pass returning (preds_o2m, preds_o2o) tuples."""
            batch_size = x.shape[0]
            pooled = self.pool(x).flatten(1)
            logits = self.cls_head(pooled).reshape(
                batch_size,
                self.num_anchors,
                self.num_classes,
            )
            boxes = (
                self.box_head(pooled)
                .reshape(
                    batch_size,
                    self.num_anchors,
                    4,
                )
                .sigmoid()
            )
            return (logits, boxes), (logits.clone(), boxes.clone())

    class _FakeDetDataset(torch.utils.data.Dataset):
        """Fake detection dataset producing random images and targets."""

        def __init__(self, n: int, num_classes: int, img_size: int) -> None:
            """Store parameters for on-the-fly sample generation."""
            self.n = n
            self.num_classes = num_classes
            self.img_size = img_size

        def __len__(self) -> int:
            """Return the number of samples."""
            return self.n

        def __getitem__(self, idx: int) -> dict[str, Tensor]:
            """Return a single sample with variable-size targets."""
            image = torch.randn(3, self.img_size, self.img_size)
            num_gt = torch.randint(1, 4, (1,)).item()
            boxes = torch.rand(num_gt, 4) * 0.5
            boxes[:, 2:] = boxes[:, :2] + 0.1
            labels = torch.randint(0, self.num_classes, (num_gt, 1)).float()
            targets = torch.cat([labels, boxes], dim=1)
            return {
                "images": image,
                "targets": targets,
                "image_ids": torch.tensor(idx),
            }

    def _det_collate(
        batch: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """Collate detection samples by adding batch indices to targets.

        Args:
            batch: List of sample dictionaries from the dataset.

        Returns:
            Collated dictionary with flat targets tensor ``(N, 6)``.

        """
        images = torch.stack([sample["images"] for sample in batch])
        all_targets: list[Tensor] = []
        for batch_idx, sample in enumerate(batch):
            per_image = sample["targets"]
            batch_col = torch.full(
                (per_image.shape[0], 1),
                float(batch_idx),
            )
            all_targets.append(torch.cat([batch_col, per_image], dim=1))
        return {
            "images": images,
            "targets": torch.cat(all_targets, dim=0),
            "image_ids": torch.stack(
                [sample["image_ids"] for sample in batch],
            ),
        }

    # ------------------------------------------------------------------
    # Build data loaders
    # ------------------------------------------------------------------

    cls_loader = DataLoader(
        _FakeClsDataset(_NUM_SAMPLES, _NUM_CLASSES_CLS, _IMG_SIZE),
        batch_size=_BATCH_SIZE,
    )

    seg_loader = DataLoader(
        _FakeSegDataset(_NUM_SAMPLES, _NUM_CLASSES_SEG, _IMG_SIZE),
        batch_size=_BATCH_SIZE,
    )

    det_loader = DataLoader(
        _FakeDetDataset(_NUM_SAMPLES, _NUM_CLASSES_DET, _IMG_SIZE),
        batch_size=_BATCH_SIZE,
        collate_fn=_det_collate,
    )

    # ------------------------------------------------------------------
    # Build models and losses
    # ------------------------------------------------------------------

    from corecv.losses.loss_classification import ClassificationCrossEntropyLoss
    from corecv.losses.loss_segmentation import SegmentationCrossEntropyLoss

    cls_model = _DummyClsModel(_NUM_CLASSES_CLS)
    seg_model = _DummySegModel(_NUM_CLASSES_SEG)
    det_model = _DummyDetModel(_NUM_CLASSES_DET)

    cls_loss_fn = ClassificationCrossEntropyLoss()
    seg_loss_fn = SegmentationCrossEntropyLoss()
    det_loss_fn = DualHeadDetectionLoss(num_classes=_NUM_CLASSES_DET)

    # ------------------------------------------------------------------
    # Instantiate evaluators and run
    # ------------------------------------------------------------------

    print("=== Classification Evaluator ===")  # noqa: T201
    cls_evaluator = ClassificationEvaluator(
        model=cls_model,
        test_loader=cls_loader,
        loss_fn=cls_loss_fn,
        num_classes=_NUM_CLASSES_CLS,
        device=_DEVICE,
    )
    cls_results = cls_evaluator.evaluate()
    print(f"Classification results: {cls_results}")  # noqa: T201
    print(f"Confusion matrix shape: {cls_results['confusion_matrix'].shape}")  # noqa: T201
    print(f"F1 curve thresholds: {cls_results['f1_curve']['thresholds']}")  # noqa: T201
    print(f"F1 macro scores: {cls_results['f1_curve']['f1_macro']}")  # noqa: T201

    # Generate plots (save to files, don't show)
    cls_evaluator.plot_confusion_matrix(
        cls_results["confusion_matrix"],
        save_path="classification_confusion_matrix.png",
        show=False,
    )
    cls_evaluator.plot_f1_curve(
        cls_results["f1_curve"],
        save_path="classification_f1_curve.png",
        show=False,
    )
    print("Plots saved: classification_confusion_matrix.png, classification_f1_curve.png\n")  # noqa: T201

    print("=== Segmentation Evaluator ===")  # noqa: T201
    seg_evaluator = SegmentationEvaluator(
        model=seg_model,
        test_loader=seg_loader,
        loss_fn=seg_loss_fn,
        num_classes=_NUM_CLASSES_SEG,
        device=_DEVICE,
    )
    seg_results = seg_evaluator.evaluate()
    print(f"Segmentation results: {seg_results}")  # noqa: T201
    print(f"Confusion matrix shape: {seg_results['confusion_matrix'].shape}")  # noqa: T201
    print(f"F1 curve thresholds: {seg_results['f1_curve']['thresholds']}")  # noqa: T201
    print(f"F1 macro scores: {seg_results['f1_curve']['f1_macro']}")  # noqa: T201

    # Generate plots (save to files, don't show)
    seg_evaluator.plot_confusion_matrix(
        seg_results["confusion_matrix"],
        save_path="segmentation_confusion_matrix.png",
        show=False,
    )
    seg_evaluator.plot_f1_curve(
        seg_results["f1_curve"],
        save_path="segmentation_f1_curve.png",
        show=False,
    )
    print("Plots saved: segmentation_confusion_matrix.png, segmentation_f1_curve.png\n")  # noqa: T201

    print("=== Detection Evaluator ===")  # noqa: T201
    det_evaluator = DetectionEvaluator(
        model=det_model,
        test_loader=det_loader,
        loss_fn=det_loss_fn,
        num_classes=_NUM_CLASSES_DET,
        device=_DEVICE,
    )
    det_results = det_evaluator.evaluate()
    print(f"Detection results: {det_results}")  # noqa: T201
    print(f"Confusion matrix shape: {det_results['confusion_matrix'].shape}")  # noqa: T201
    print(f"F1 curve thresholds: {det_results['f1_curve']['thresholds']}")  # noqa: T201
    print(f"F1 scores: {det_results['f1_curve']['f1']}")  # noqa: T201

    # Generate plots (save to files, don't show)
    det_evaluator.plot_confusion_matrix(
        det_results["confusion_matrix"],
        save_path="detection_confusion_matrix.png",
        show=False,
    )
    det_evaluator.plot_f1_curve(
        det_results["f1_curve"],
        save_path="detection_f1_curve.png",
        show=False,
    )
    print("Plots saved: detection_confusion_matrix.png, detection_f1_curve.png\n")  # noqa: T201

    print("All evaluators completed successfully with plots generated.")  # noqa: T201

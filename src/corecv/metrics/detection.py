"""Object detection evaluation metrics for CoreCV.

Accumulates per-image predictions and ground-truth boxes across validation
batches and computes mean Average Precision (mAP) together with precision,
recall, and F1 at a confidence threshold.

Reference:
    Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torchvision.ops import box_iou

logger = logging.getLogger(__name__)

# IoU thresholds used for the mAP50-95 computation: 0.5, 0.55, ..., 0.95.
_DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = tuple(t / 100 for t in range(50, 100, 5))

# IoU threshold used for the precision/recall/F1 computation.
_PR_IOU_THRESHOLD = 0.5


class DetectionMetrics:
    """Evaluator for object detection metrics (mAP, precision, recall, F1).

    Accumulates per-image predictions and ground-truth boxes in O(1) memory
    relative to the number of anchors, then evaluates performance with mean
    Average Precision over COCO IoU thresholds and precision/recall/F1 at the
    configured confidence threshold.

    Predictions are expected from the one-to-one head of a dual-head
    detection model: ``(logits, boxes)`` with shapes ``(B, A, num_classes)``
    and ``(B, A, 4)``. Targets are flat ``(N, 6)`` tensors with columns
    ``[batch_index, class, x1, y1, x2, y2]``.

    Example:
        >>> import torch
        >>> from corecv.metrics.detection import DetectionMetrics
        >>> metrics = DetectionMetrics(num_classes=1, conf_threshold=0.0)
        >>> logits = torch.tensor([[[5.0]]])
        >>> boxes = torch.tensor([[[10.0, 10.0, 40.0, 40.0]]])
        >>> targets = torch.tensor([[0.0, 0, 10.0, 10.0, 40.0, 40.0]])
        >>> metrics.update((logits, boxes), targets)
        >>> metrics.compute()["mAP50"]
        1.0

    """

    def __init__(
        self,
        num_classes: int,
        conf_threshold: float = 0.25,
        iou_thresholds: tuple[float, ...] = _DEFAULT_IOU_THRESHOLDS,
        eps: float = 1e-9,
    ) -> None:
        """Initialize DetectionMetrics evaluator.

        Args:
            num_classes: Number of output classes.
            conf_threshold: Minimum confidence score to keep a prediction.
            iou_thresholds: IoU thresholds over which average precision is
                computed.
            eps: Small constant for numerical stability.

        """
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_thresholds = iou_thresholds
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated per-image predictions and ground truths."""
        self._pred_boxes: list[Tensor] = []
        self._pred_scores: list[Tensor] = []
        self._pred_labels: list[Tensor] = []
        self._gt_boxes: list[Tensor] = []
        self._gt_labels: list[Tensor] = []
        self.results: dict[str, float] = {}

    def update(self, preds_o2o: tuple[Tensor, Tensor], targets: Tensor) -> None:
        """Update metric state with a batch of one-to-one predictions.

        Args:
            preds_o2o: ``(logits, boxes)`` of the one-to-one head with shapes
                ``(B, A, num_classes)`` and ``(B, A, 4)``.
            targets: Flat targets of shape ``(N, 6)`` with columns
                ``[batch_index, class, x1, y1, x2, y2]``.

        """
        logits, boxes = preds_o2o
        scores_all = logits.sigmoid()
        max_scores, pred_labels = scores_all.max(dim=-1)

        for batch_idx in range(boxes.shape[0]):
            keep = max_scores[batch_idx] >= self.conf_threshold
            self._pred_boxes.append(boxes[batch_idx][keep].cpu())
            self._pred_scores.append(max_scores[batch_idx][keep].cpu())
            self._pred_labels.append(pred_labels[batch_idx][keep].cpu())

            mask = targets[:, 0] == batch_idx
            if mask.any():
                per_image = targets[mask]
                self._gt_boxes.append(per_image[:, 2:6].cpu())
                self._gt_labels.append(per_image[:, 1].long().cpu())
            else:
                self._gt_boxes.append(torch.zeros(0, 4))
                self._gt_labels.append(torch.zeros(0, dtype=torch.long))

    def compute(self) -> dict[str, float]:
        """Compute detection metrics over all accumulated data.

        Returns:
            Dictionary containing:
            - ``mAP50``: Mean average precision at IoU = 0.5.
            - ``mAP50-95``: Mean average precision over the configured IoU
              thresholds.
            - ``precision``: Precision at the confidence threshold.
            - ``recall``: Recall at the confidence threshold.
            - ``f1``: F1 score at the confidence threshold.

        """
        total_gt = sum(len(gt) for gt in self._gt_labels)
        if not self._pred_boxes or total_gt == 0:
            self.results = {
                "mAP50": 0.0,
                "mAP50-95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
            return self.results

        ap50 = self._mean_ap((0.5,))
        ap50_95 = self._mean_ap(self.iou_thresholds)
        precision, recall, f1 = self._pr_at_threshold()

        self.results = {
            "mAP50": round(ap50, 4),
            "mAP50-95": round(ap50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        return self.results

    def _mean_ap(self, iou_thresholds: tuple[float, ...]) -> float:
        """Compute the mean average precision over the given IoU thresholds.

        Averages over the classes present in the ground truth, following the
        macro-averaging convention of the other CoreCV metrics.

        """
        present_classes = [
            cls
            for cls in range(self.num_classes)
            if any((gt_labels == cls).any() for gt_labels in self._gt_labels)
        ]
        if not present_classes:
            return 0.0

        values: list[float] = []
        for iou_threshold in iou_thresholds:
            class_aps = [
                self._ap_class(cls, iou_threshold) for cls in present_classes
            ]
            values.append(sum(class_aps) / len(class_aps))
        return sum(values) / len(values)

    def _ap_class(self, cls: int, iou_threshold: float) -> float:
        """Compute the average precision of a single class at an IoU threshold.

        Args:
            cls: Class index.
            iou_threshold: IoU threshold for matching predictions to ground
                truth.

        Returns:
            Average precision in ``[0, 1]``.

        """
        detections: list[tuple[float, bool]] = []
        num_gt = 0

        for pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels in zip(
            self._pred_boxes,
            self._pred_scores,
            self._pred_labels,
            self._gt_boxes,
            self._gt_labels,
            strict=False,
        ):
            gt_cls = gt_boxes[gt_labels == cls]
            num_gt += gt_cls.shape[0]

            pred_mask = pred_labels == cls
            pred_cls_boxes = pred_boxes[pred_mask]
            pred_cls_scores = pred_scores[pred_mask]
            if pred_cls_boxes.shape[0] == 0:
                continue
            if gt_cls.shape[0] == 0:
                detections.extend((score.item(), False) for score in pred_cls_scores)
                continue

            order = pred_cls_scores.argsort(descending=True)
            pred_cls_boxes = pred_cls_boxes[order]
            pred_cls_scores = pred_cls_scores[order]
            overlaps = box_iou(pred_cls_boxes, gt_cls)
            matched = torch.zeros(gt_cls.shape[0], dtype=torch.bool)

            for idx in range(pred_cls_boxes.shape[0]):
                row = overlaps[idx].clone()
                row[matched] = 0.0
                best_iou, best_gt_idx = row.max(dim=0)
                if best_iou >= iou_threshold:
                    matched[best_gt_idx] = True
                    detections.append((pred_cls_scores[idx].item(), True))
                else:
                    detections.append((pred_cls_scores[idx].item(), False))

        if num_gt == 0 or not detections:
            return 0.0

        detections.sort(key=lambda item: item[0], reverse=True)
        true_positives = torch.tensor(
            [1.0 if is_tp else 0.0 for _, is_tp in detections],
        )
        cum_tp = true_positives.cumsum(dim=0)
        cum_fp = (1.0 - true_positives).cumsum(dim=0)
        recall = cum_tp / num_gt
        precision = cum_tp / (cum_tp + cum_fp + self.eps)
        return self._ap_101(precision, recall)

    @staticmethod
    def _ap_101(precision: Tensor, recall: Tensor) -> float:
        """Compute the 101-point interpolated average precision."""
        ap = 0.0
        for t in range(101):
            threshold = t / 100
            above = recall >= threshold
            if above.any():
                ap += precision[above].max().item() / 101
        return ap

    def _pr_at_threshold(self) -> tuple[float, float, float]:
        """Compute precision, recall, and F1 at the confidence threshold."""
        total_tp = 0
        total_fp = 0
        total_gt = 0

        for pred_boxes, pred_scores, _pred_labels, gt_boxes, _gt_labels in zip(
            self._pred_boxes,
            self._pred_scores,
            self._pred_labels,
            self._gt_boxes,
            self._gt_labels,
            strict=False,
        ):
            total_gt += gt_boxes.shape[0]
            if pred_boxes.shape[0] == 0:
                continue
            if gt_boxes.shape[0] == 0:
                total_fp += pred_boxes.shape[0]
                continue

            order = pred_scores.argsort(descending=True)
            sorted_boxes = pred_boxes[order]
            overlaps = box_iou(sorted_boxes, gt_boxes)
            matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)

            for idx in range(sorted_boxes.shape[0]):
                row = overlaps[idx].clone()
                row[matched] = 0.0
                best_iou, best_gt_idx = row.max(dim=0)
                if best_iou >= _PR_IOU_THRESHOLD:
                    matched[best_gt_idx] = True
                    total_tp += 1
                else:
                    total_fp += 1

        precision = total_tp / (total_tp + total_fp + self.eps)
        recall = total_tp / (total_gt + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        return precision, recall, f1

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
            "%s | mAP50=%.4f mAP50-95=%.4f precision=%.4f recall=%.4f f1=%.4f",
            stage,
            self.results["mAP50"],
            self.results["mAP50-95"],
            self.results["precision"],
            self.results["recall"],
            self.results["f1"],
        )

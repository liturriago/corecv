"""Object detection evaluation metrics: mAP50, mAP50-95, Precision, Recall, and F1.

Computes standard COCO-style Mean Average Precision (mAP) and precision/recall
metrics across multiple IoU thresholds [0.50:0.05:0.95] using ``torchvision.ops.box_iou``.

Typical usage::

    metrics = DetectionMetrics(num_classes=80)

    for batch in val_loader:
        images = batch["images"]
        targets = batch["targets"]  # (N_total, 6) from detection_collate_fn
        preds = model(images)       # List of dicts with 'boxes', 'scores', 'labels'

        metrics.update(preds, targets)

    results = metrics.compute()
    print(results)
    # {"mAP50": 0.65, "mAP50-95": 0.42, "precision": 0.72, "recall": 0.68, "f1": 0.70}
    metrics.reset()
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.ops import box_iou

_NUM_PREDICTION_FORMAT_FIELDS = 3


def _compute_ap(recalls: Tensor, precisions: Tensor) -> Tensor:
    """Compute Average Precision using 101-point COCO interpolation.

    Args:
        recalls: 1D tensor of cumulative recall values.
        precisions: 1D tensor of cumulative precision values.

    Returns:
        Scalar tensor containing the computed AP value.
    """
    if recalls.numel() == 0:
        return torch.tensor(0.0, device=recalls.device)

    device = recalls.device

    # Append sentinel values at start and end
    mrec = torch.cat(
        [
            torch.tensor([0.0], device=device),
            recalls,
            torch.tensor([1.0], device=device),
        ]
    )
    mpre = torch.cat(
        [
            torch.tensor([0.0], device=device),
            precisions,
            torch.tensor([0.0], device=device),
        ]
    )

    # Compute precision envelope (make precision monotonically non-increasing)
    for i in range(mpre.shape[0] - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    # 101-point recall sampling (COCO standard: 0.00, 0.01, ..., 1.00)
    recall_points = torch.linspace(0, 1, 101, device=device)
    indices = torch.searchsorted(mrec, recall_points, right=False)
    indices = indices.clamp(max=mpre.shape[0] - 1)

    return mpre[indices].mean()


def _parse_targets(
    targets: Tensor | list[dict[str, Tensor]],
    batch_size: int,
) -> list[tuple[Tensor, Tensor]]:
    """Parse targets into a per-image list of (boxes, labels).

    Args:
        targets: Flat targets tensor of shape ``(N, 6)`` where each row is
            ``[batch_idx, class_id, x1, y1, x2, y2]`` or a list of dictionaries
            with keys ``'boxes'`` and ``'labels'``.
        batch_size: Number of images in the batch.

    Returns:
        List of tuples ``(boxes, labels)`` per image in the batch.
    """
    if isinstance(targets, Tensor):
        device = targets.device
        per_image: list[tuple[Tensor, Tensor]] = []

        for batch_idx in range(batch_size):
            if targets.numel() == 0:
                per_image.append(
                    (
                        torch.zeros((0, 4), device=device, dtype=targets.dtype),
                        torch.zeros((0,), device=device, dtype=torch.long),
                    )
                )
                continue

            mask = targets[:, 0] == batch_idx
            img_targets = targets[mask]

            if img_targets.numel() == 0:
                per_image.append(
                    (
                        torch.zeros((0, 4), device=device, dtype=targets.dtype),
                        torch.zeros((0,), device=device, dtype=torch.long),
                    )
                )
            else:
                boxes = img_targets[:, 2:6]
                labels = img_targets[:, 1].long()
                per_image.append((boxes, labels))

        return per_image

    return [(t["boxes"], t["labels"].long()) for t in targets]


def _parse_predictions(
    preds: list[dict[str, Tensor]] | tuple[Tensor, Tensor, Tensor],
) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Parse predictions into a per-image list of (boxes, scores, labels).

    Args:
        preds: Predictions formatted as a list of dictionaries containing
            ``'boxes'``, ``'scores'``, and ``'labels'``, or a tuple of batched
            tensors ``(boxes, scores, labels)``.

    Returns:
        List of tuples ``(boxes, scores, labels)`` per image.
    """
    if isinstance(preds, list):
        return [(p["boxes"], p["scores"], p["labels"].long()) for p in preds]
    if isinstance(preds, tuple) and len(preds) == _NUM_PREDICTION_FORMAT_FIELDS:
        boxes_b, scores_b, labels_b = preds
        batch_size = boxes_b.shape[0]
        return [(boxes_b[i], scores_b[i], labels_b[i].long()) for i in range(batch_size)]

    msg = f"Unsupported predictions format: {type(preds)}"
    raise ValueError(msg)


class DetectionMetrics:
    """Evaluator for Object Detection metrics (mAP50, mAP50-95, Precision, Recall, F1).

    Accumulates predictions and targets across validation batches and evaluates
    performance using ``torchvision.ops.box_iou``.
    """

    def __init__(
        self,
        num_classes: int,
        conf_threshold: float = 0.25,
        eps: float = 1e-16,
    ) -> None:
        """Initialize DetectionMetrics evaluator.

        Args:
            num_classes: Total number of foreground object classes.
            conf_threshold: Confidence score threshold for precision/recall calculation.
            eps: Small constant for numerical stability.
        """
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.eps = eps

        # 10 IoU thresholds: 0.50, 0.55, ..., 0.95
        self.iou_thresholds = torch.linspace(0.50, 0.95, 10)
        self._image_records: list[dict[str, Tensor]] = []

    def update(
        self,
        preds: list[dict[str, Tensor]] | tuple[Tensor, Tensor, Tensor],
        targets: Tensor | list[dict[str, Tensor]],
    ) -> None:
        """Update metric state with a new batch of predictions and ground truths.

        Args:
            preds: Model predictions for the batch.
            targets: Ground-truth annotations for the batch.
        """
        parsed_preds = _parse_predictions(preds)
        batch_size = len(parsed_preds)
        parsed_targets = _parse_targets(targets, batch_size)

        for (p_boxes, p_scores, p_labels), (g_boxes, g_labels) in zip(
            parsed_preds, parsed_targets, strict=True
        ):
            # Move to CPU to prevent VRAM accumulation during evaluation loop
            self._image_records.append(
                {
                    "pred_boxes": p_boxes.detach().cpu(),
                    "pred_scores": p_scores.detach().cpu(),
                    "pred_labels": p_labels.detach().cpu(),
                    "gt_boxes": g_boxes.detach().cpu(),
                    "gt_labels": g_labels.detach().cpu(),
                }
            )

    def reset(self) -> None:
        """Reset all accumulated predictions and targets."""
        self._image_records.clear()

    def _compute_ap_for_class(
        self,
        all_scores: Tensor,
        all_tp: Tensor,
        total_gt_class: int,
    ) -> Tensor:
        """Compute AP across IoU thresholds for a single class.

        Args:
            all_scores: Concatenated confidence scores for all predictions.
            all_tp: Concatenated TP tensor of shape ``(N, num_iou_thresh)``.
            total_gt_class: Total number of ground truth instances for this class.

        Returns:
            Tensor of AP values for each IoU threshold.
        """
        num_iou_thresh = self.iou_thresholds.shape[0]
        sort_order = torch.argsort(all_scores, descending=True)
        all_scores = all_scores[sort_order]
        all_tp = all_tp[sort_order]

        ap_class = torch.zeros(num_iou_thresh)
        for k in range(num_iou_thresh):
            tp_k = all_tp[:, k].float()
            fp_k = 1.0 - tp_k

            cum_tp = torch.cumsum(tp_k, dim=0)
            cum_fp = torch.cumsum(fp_k, dim=0)

            recalls = cum_tp / (total_gt_class + self.eps)
            precisions = cum_tp / (cum_tp + cum_fp + self.eps)

            ap_class[k] = _compute_ap(recalls, precisions)

        return ap_class

    def _compute_precision_recall_at_threshold(
        self,
        all_scores: Tensor,
        all_tp: Tensor,
        total_gt_class: int,
    ) -> tuple[float, float]:
        """Compute precision and recall at the confidence threshold.

        Args:
            all_scores: Concatenated confidence scores for all predictions.
            all_tp: Concatenated TP tensor of shape ``(N, num_iou_thresh)``.
            total_gt_class: Total number of ground truth instances for this class.

        Returns:
            Tuple of ``(precision, recall)`` at the confidence threshold.
        """
        conf_mask = all_scores >= self.conf_threshold
        if conf_mask.any():
            tp_conf = all_tp[conf_mask, 0].float().sum().item()
            fp_conf = (1.0 - all_tp[conf_mask, 0].float()).sum().item()
            prec_c = tp_conf / (tp_conf + fp_conf + self.eps)
            rec_c = tp_conf / (total_gt_class + self.eps)
        else:
            prec_c = 0.0
            rec_c = 0.0

        return prec_c, rec_c

    def _process_class(
        self,
        class_id: int,
    ) -> tuple[Tensor | None, float, float, int]:
        """Process predictions and ground truths for a single class.

        Args:
            class_id: The class ID to process.

        Returns:
            Tuple of ``(ap_class, precision, recall, total_gt_class)``.
            Returns ``(None, 0.0, 0.0, 0)`` if no ground truth exists.
        """
        num_iou_thresh = self.iou_thresholds.shape[0]
        class_scores_list: list[Tensor] = []
        class_tp_list: list[Tensor] = []
        total_gt_class = 0

        for record in self._image_records:
            p_mask = record["pred_labels"] == class_id
            g_mask = record["gt_labels"] == class_id

            p_boxes = record["pred_boxes"][p_mask]
            p_scores = record["pred_scores"][p_mask]
            g_boxes = record["gt_boxes"][g_mask]

            num_p = p_boxes.shape[0]
            num_g = g_boxes.shape[0]
            total_gt_class += num_g

            if num_p == 0:
                continue

            tp_img = torch.zeros((num_p, num_iou_thresh), dtype=torch.bool)

            if num_g > 0:
                ious = box_iou(p_boxes, g_boxes)

                for k, tau in enumerate(self.iou_thresholds):
                    gt_matched = torch.zeros(num_g, dtype=torch.bool)

                    sorted_p_indices = torch.argsort(p_scores, descending=True)
                    for p_idx in sorted_p_indices:
                        p_ious = ious[p_idx].clone()
                        p_ious[gt_matched] = -1.0
                        max_iou, best_gt_idx = p_ious.max(dim=0)

                        if max_iou >= tau:
                            tp_img[p_idx, k] = True
                            gt_matched[best_gt_idx] = True

            class_scores_list.append(p_scores)
            class_tp_list.append(tp_img)

        if total_gt_class == 0:
            return None, 0.0, 0.0, 0

        if not class_scores_list:
            return torch.zeros(num_iou_thresh), 0.0, 0.0, total_gt_class

        all_scores = torch.cat(class_scores_list, dim=0)
        all_tp = torch.cat(class_tp_list, dim=0)

        ap_class = self._compute_ap_for_class(all_scores, all_tp, total_gt_class)
        prec_c, rec_c = self._compute_precision_recall_at_threshold(
            all_scores, all_tp, total_gt_class
        )

        return ap_class, prec_c, rec_c, total_gt_class

    def compute(self) -> dict[str, float]:
        """Compute evaluation metrics over all accumulated data.

        Returns:
            Dictionary containing computed metric values:
            - ``mAP50``: Mean Average Precision at IoU = 0.50.
            - ``mAP50-95``: Mean Average Precision across IoU thresholds [0.50:0.05:0.95].
            - ``precision``: Precision score at the confidence threshold.
            - ``recall``: Recall score at the confidence threshold.
            - ``f1``: F1 score at the confidence threshold.
        """
        if not self._image_records:
            return {
                "mAP50": 0.0,
                "mAP50-95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        ap_per_class: list[Tensor] = []
        precisions_per_class: list[float] = []
        recalls_per_class: list[float] = []

        for class_id in range(self.num_classes):
            ap_class, prec_c, rec_c, total_gt_class = self._process_class(class_id)

            if total_gt_class == 0:
                continue

            ap_per_class.append(ap_class)
            precisions_per_class.append(prec_c)
            recalls_per_class.append(rec_c)

        if not ap_per_class:
            return {
                "mAP50": 0.0,
                "mAP50-95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        ap_tensor = torch.stack(ap_per_class, dim=0)
        map50 = ap_tensor[:, 0].mean().item()
        map50_95 = ap_tensor.mean().item()

        avg_precision = sum(precisions_per_class) / len(precisions_per_class)
        avg_recall = sum(recalls_per_class) / len(recalls_per_class)
        f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall + self.eps)

        return {
            "mAP50": round(map50, 4),
            "mAP50-95": round(map50_95, 4),
            "precision": round(avg_precision, 4),
            "recall": round(avg_recall, 4),
            "f1": round(f1, 4),
        }


if __name__ == "__main__":
    torch.manual_seed(42)
    num_classes = 80
    batch_size = 4

    metrics = DetectionMetrics(num_classes=num_classes, conf_threshold=0.25)

    # 1. Generar Ground Truths ficticios
    targets_list: list[Tensor] = []
    gt_per_image: list[tuple[Tensor, Tensor]] = []

    for b in range(batch_size):
        num_gt = 5
        gt_classes = torch.randint(0, num_classes, (num_gt, 1)).float()
        gt_boxes = torch.rand(num_gt, 4) * 0.5
        gt_boxes[:, 2:] = gt_boxes[:, :2] + gt_boxes[:, 2:].clamp(min=0.2)

        batch_col = torch.full((num_gt, 1), float(b))
        targets_list.append(torch.cat([batch_col, gt_classes, gt_boxes], dim=1))
        gt_per_image.append((gt_boxes, gt_classes.squeeze(1).long()))

    targets = torch.cat(targets_list, dim=0)

    # 2. Generar Predicciones basadas en los GTs (con ligero ruido para simular un buen modelo)
    preds: list[dict[str, Tensor]] = []
    for b in range(batch_size):
        gt_b, labels_b = gt_per_image[b]

        # Copiamos las cajas reales y les sumamos un pequeño ruido
        noise = (torch.rand_like(gt_b) - 0.5) * 0.02
        pred_boxes = (gt_b + noise).clamp(0.0, 1.0)

        # Puntuaciones altas de confianza
        pred_scores = 0.70 + torch.rand(len(gt_b)) * (0.99 - 0.70)
        pred_labels = labels_b.clone()

        preds.append(
            {
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels,
            }
        )

    metrics.update(preds, targets)
    results = metrics.compute()

    print("--- Detection Metrics Sanity Test (Near perfect predictions) ---")  # noqa: T201
    for k, v in results.items():
        print(f"  {k:10s}: {v:.4f}")  # noqa: T201

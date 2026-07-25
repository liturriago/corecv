"""Dual-head detection loss with Task-Aligned Assigner for NMS-free training.

Implements a modular loss for single-stage anchor-free object detection with
dual-head architecture (One-to-Many and One-to-One). Uses ``torchvision.ops``
for all IoU-based computations to keep everything GPU-native. No external
dependencies (pycocotools, scipy, etc.) are used.

Typical usage::

    loss_fn = DualHeadDetectionLoss(num_classes=80)
    loss_dict = loss_fn(preds_o2m, preds_o2o, targets)
    loss_dict["loss_total"].backward()
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.ops import (
    box_iou,
    complete_box_iou_loss,
    sigmoid_focal_loss,
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _split_targets_by_batch(
    targets: Tensor,
    batch_size: int,
) -> list[tuple[Tensor, Tensor]]:
    """Split a flat targets tensor into per-image ``(boxes, labels)`` tuples.

    Args:
        targets: Targets tensor of shape ``(N, 6)`` where each row is
            ``[batch_idx, class_id, x1, y1, x2, y2]``.
        batch_size: Number of images in the batch.

    Returns:
        A list of length *batch_size*. Each element is a tuple of
        ``(boxes, labels)`` for that image. Empty images receive zero-length
        tensors on the same device as *targets*.
    """
    device = targets.device
    per_image: list[tuple[Tensor, Tensor]] = []

    for batch_idx in range(batch_size):
        if targets.numel() == 0:
            per_image.append(
                (
                    torch.zeros(0, 4, device=device, dtype=targets.dtype),
                    torch.zeros(0, device=device, dtype=torch.long),
                ),
            )
            continue

        mask = targets[:, 0] == batch_idx
        image_targets = targets[mask]

        if image_targets.numel() == 0:
            per_image.append(
                (
                    torch.zeros(0, 4, device=device, dtype=targets.dtype),
                    torch.zeros(0, device=device, dtype=torch.long),
                ),
            )
        else:
            boxes = image_targets[:, 2:6]
            labels = image_targets[:, 1].long()
            per_image.append((boxes, labels))

    return per_image


# ---------------------------------------------------------------------------
# Task-Aligned Assigner
# ---------------------------------------------------------------------------


class TaskAlignedAssigner(nn.Module):
    """Task-Aligned Assigner (TAL) for anchor-free object detection.

    Assigns ground-truth targets to prediction anchors using an alignment
    metric that combines classification score and bounding-box IoU::

        metric = score**alpha * iou**beta

    Includes collision resolution so that each anchor is uniquely assigned to
    at most one ground-truth object (the one with highest metric score).
    """

    def __init__(
        self,
        alpha: float = 0.9,
        beta: float = 6.0,
        topk: int = 13,
        eps: float = 1e-9,
    ) -> None:
        """Initialize the Task-Aligned Assigner.

        Args:
            alpha: Exponent for the classification-score component.
            beta: Exponent for the IoU component.
            topk: Number of top anchors per ground truth for the O2M head.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.topk = topk
        self.eps = eps

    def forward(
        self,
        pred_boxes: Tensor,
        pred_logits: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        *,
        top_one: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Assign ground-truth targets to prediction anchors for one image.

        Args:
            pred_boxes: Predicted boxes, shape ``(N_anchors, 4)`` in
                ``[x1, y1, x2, y2]`` format.
            pred_logits: Raw classification logits,
                shape ``(N_anchors, num_classes)``.
            gt_boxes: Ground-truth boxes, shape ``(N_gt, 4)``.
            gt_labels: Ground-truth class labels, shape ``(N_gt,)`` with
                integer class indices in ``[0, num_classes)``.
            top_one: When ``True`` select only the single best anchor per
                ground truth (O2O head). When ``False`` select *topk* anchors
                per ground truth (O2M head).

        Returns:
            Tuple of four tensors:

            - **anchor_indices** ``(K,)`` -- unique indices into *pred_boxes*.
            - **gt_indices** ``(K,)`` -- indices into *gt_boxes*.
            - **target_labels** ``(K,)`` -- assigned class labels.
            - **alignment_scores** ``(K,)`` -- per-assignment metric values
              normalized to ``[0, 1]`` for soft-target focal loss.
        """
        num_anchors = pred_boxes.shape[0]
        num_gt = gt_boxes.shape[0]
        device = pred_boxes.device

        # No ground truth → empty assignment
        if num_gt == 0:
            empty_long = torch.zeros(0, device=device, dtype=torch.long)
            empty_float = torch.zeros(0, device=device, dtype=pred_boxes.dtype)
            return empty_long, empty_long, empty_long, empty_float

        # Pairwise IoU between all GT boxes and all anchors: (N_gt, N_anchors)
        pairwise_ious: Tensor = box_iou(gt_boxes, pred_boxes)

        # Classification scores for the GT classes: (N_gt, N_anchors)
        pred_cls_scores = pred_logits.detach().sigmoid()  # (N_anchors, C)
        gt_cls_scores = pred_cls_scores[:, gt_labels].T  # (N_gt, N_anchors)

        # Alignment metric: (N_gt, N_anchors)
        alignment_metric: Tensor = (gt_cls_scores**self.alpha) * (pairwise_ious**self.beta)

        # Number of candidates per ground truth
        num_candidates = 1 if top_one else min(self.topk, num_anchors)

        # Top-K (or top-1) indices per ground truth: (N_gt, num_candidates)
        _, topk_indices = alignment_metric.topk(num_candidates, dim=1)

        # Create a boolean mask of top-K candidate anchors
        topk_mask = torch.zeros_like(alignment_metric, dtype=torch.bool)
        topk_mask.scatter_(1, topk_indices, True)

        # Zero out metrics outside top-K candidates
        candidate_metrics = torch.where(
            topk_mask,
            alignment_metric,
            torch.zeros_like(alignment_metric),
        )

        # Collision resolution: pick GT with maximum alignment metric per anchor
        max_scores, best_gt_indices = candidate_metrics.max(dim=0)  # (N_anchors,)

        # Filter valid positive anchors
        pos_mask = max_scores > self.eps
        anchor_indices = torch.where(pos_mask)[0]  # (K,)
        gt_indices = best_gt_indices[anchor_indices]  # (K,)

        # Gather per-assignment labels
        target_labels = gt_labels[gt_indices]  # (K,)

        # Normalize alignment scores per GT to [0, 1] for soft classification targets
        max_align_per_gt = alignment_metric.max(dim=1, keepdim=True).values.clamp(
            min=self.eps,
        )
        normalized_metrics = alignment_metric / max_align_per_gt
        alignment_scores = normalized_metrics[gt_indices, anchor_indices]  # (K,)

        return anchor_indices, gt_indices, target_labels, alignment_scores


# ---------------------------------------------------------------------------
# Dual-Head Detection Loss
# ---------------------------------------------------------------------------


class DualHeadDetectionLoss(nn.Module):
    """Dual-head detection loss for NMS-free anchor-free object detection.

    Combines losses from a **One-to-Many** (O2M) training head and a
    **One-to-One** (O2O) inference head::

        Loss = Loss_O2M + alpha_o2o * Loss_O2O

    All loss components (focal loss, CIoU loss, IoU computation) run entirely
    on GPU via ``torchvision.ops`` with no CPU round-trips.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_classes: int,
        alpha: float = 0.9,
        beta: float = 6.0,
        topk: int = 13,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        alpha_o2o: float = 1.0,
    ) -> None:
        """Initialize the dual-head detection loss.

        Args:
            num_classes: Number of foreground object classes.
            alpha: TAL exponent for the classification-score component.
            beta: TAL exponent for the IoU component.
            topk: Number of top anchors per GT for the O2M head.
            focal_gamma: Gamma parameter for focal loss.
            focal_alpha: Alpha balancing factor for focal loss.
            alpha_o2o: Scalar multiplier for the O2O loss in the total.
        """
        super().__init__()
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.alpha_o2o = alpha_o2o

        self.assigner = TaskAlignedAssigner(
            alpha=alpha,
            beta=beta,
            topk=topk,
        )

    # ------------------------------------------------------------------
    # Internal loss helpers
    # ------------------------------------------------------------------

    def _compute_cls_loss(
        self,
        pred_logits: Tensor,
        target_labels: Tensor,
        anchor_indices: Tensor,
        alignment_scores: Tensor,
        num_positives: int,
    ) -> Tensor:
        """Compute task-aligned focal classification loss using soft targets.

        Target vector contains soft alignment scores in ``[0, 1]`` for assigned
        positive classes and 0 elsewhere.

        Args:
            pred_logits: Raw logits ``(N_anchors, num_classes)``.
            target_labels: Assigned class labels ``(K,)``.
            anchor_indices: Positive anchor indices ``(K,)``.
            alignment_scores: Per-assignment normalized metric values ``(K,)``.
            num_positives: Total positive count for normalisation.

        Returns:
            Scalar loss tensor connected to the computation graph.
        """
        if num_positives == 0:
            return pred_logits.sum() * 0.0

        # Construct soft classification target matrix: (N_anchors, C)
        target_scores = torch.zeros_like(pred_logits)
        target_scores[anchor_indices, target_labels] = alignment_scores.clamp(0.0, 1.0)

        # Single-pass focal loss with soft targets
        focal_loss = sigmoid_focal_loss(
            pred_logits,
            target_scores,
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
            reduction="sum",
        )

        return focal_loss / max(num_positives, 1)

    def _compute_head_losses(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        *,
        top_one: bool,
    ) -> tuple[Tensor, Tensor, int]:
        """Compute classification and box-regression losses for one head.

        Args:
            pred_logits: Per-image logits ``(N_anchors, num_classes)``.
            pred_boxes: Per-image predicted boxes ``(N_anchors, 4)``.
            gt_boxes: Ground-truth boxes ``(N_gt, 4)``.
            gt_labels: Ground-truth class labels ``(N_gt,)``.
            top_one: Use top-1 (O2O) assignment instead of top-K (O2M).

        Returns:
            Tuple of ``(cls_loss, box_loss, num_positives)``.
        """
        anchor_indices, gt_indices, target_labels, alignment_scores = self.assigner(
            pred_boxes,
            pred_logits,
            gt_boxes,
            gt_labels,
            top_one=top_one,
        )

        num_positives: int = anchor_indices.shape[0]

        if num_positives == 0:
            zero_loss = pred_logits.sum() * 0.0
            return zero_loss, zero_loss, 0

        # Box regression: CIoU loss on positive assignments only
        assigned_pred_boxes = pred_boxes[anchor_indices]  # (K, 4)
        assigned_gt_boxes = gt_boxes[gt_indices]  # (K, 4)
        box_loss = complete_box_iou_loss(assigned_pred_boxes, assigned_gt_boxes, reduction="mean")

        # Classification: focal loss weighted by soft alignment targets
        cls_loss = self._compute_cls_loss(
            pred_logits,
            target_labels,
            anchor_indices,
            alignment_scores,
            num_positives,
        )

        return cls_loss, box_loss, num_positives

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        preds_o2m: tuple[Tensor, Tensor],
        preds_o2o: tuple[Tensor, Tensor],
        targets: Tensor,
    ) -> dict[str, Tensor]:
        """Compute dual-head detection losses for a batch.

        Args:
            preds_o2m: O2M head predictions as
                ``(pred_logits, pred_boxes)`` with shapes
                ``(B, N_anchors, num_classes)`` and ``(B, N_anchors, 4)``.
            preds_o2o: O2O head predictions, same shapes as *preds_o2m*.
            targets: Collated targets tensor of shape ``(N, 6)`` where each
                row is ``[batch_idx, class_id, x1, y1, x2, y2]``.

        Returns:
            Dictionary with keys:

            - ``loss_total``: Combined ``L_o2m + alpha_o2o * L_o2o``.
            - ``loss_cls_o2m``: O2M focal classification loss.
            - ``loss_box_o2m``: O2M CIoU box loss.
            - ``loss_cls_o2o``: O2O focal classification loss.
            - ``loss_box_o2o``: O2O CIoU box loss.
        """
        pred_logits_o2m, pred_boxes_o2m = preds_o2m
        pred_logits_o2o, pred_boxes_o2o = preds_o2o
        batch_size = pred_logits_o2m.shape[0]
        device = pred_logits_o2m.device

        # Accumulator tensors connected to autograd graph
        loss_cls_o2m_acc = pred_logits_o2m.sum() * 0.0
        loss_box_o2m_acc = pred_boxes_o2m.sum() * 0.0
        loss_cls_o2o_acc = pred_logits_o2o.sum() * 0.0
        loss_box_o2o_acc = pred_boxes_o2o.sum() * 0.0

        per_image_targets = _split_targets_by_batch(targets, batch_size)

        for image_idx in range(batch_size):
            gt_boxes, gt_labels = per_image_targets[image_idx]

            # O2M head (top-K assignment)
            cls_o2m, box_o2m, n_pos_o2m = self._compute_head_losses(
                pred_logits_o2m[image_idx],
                pred_boxes_o2m[image_idx],
                gt_boxes,
                gt_labels,
                top_one=False,
            )

            # O2O head (top-1 assignment)
            cls_o2o, box_o2o, n_pos_o2o = self._compute_head_losses(
                pred_logits_o2o[image_idx],
                pred_boxes_o2o[image_idx],
                gt_boxes,
                gt_labels,
                top_one=True,
            )

            if n_pos_o2m > 0:
                loss_cls_o2m_acc = loss_cls_o2m_acc + cls_o2m
                loss_box_o2m_acc = loss_box_o2m_acc + box_o2m
            if n_pos_o2o > 0:
                loss_cls_o2o_acc = loss_cls_o2o_acc + cls_o2o
                loss_box_o2o_acc = loss_box_o2o_acc + box_o2o

        # Normalize accumulators by batch size
        loss_cls_o2m_acc = loss_cls_o2m_acc / batch_size
        loss_box_o2m_acc = loss_box_o2m_acc / batch_size
        loss_cls_o2o_acc = loss_cls_o2o_acc / batch_size
        loss_box_o2o_acc = loss_box_o2o_acc / batch_size

        loss_total = (
            loss_cls_o2m_acc
            + loss_box_o2m_acc
            + self.alpha_o2o * (loss_cls_o2o_acc + loss_box_o2o_acc)
        )

        return {
            "loss_total": loss_total.squeeze(0),
            "loss_cls_o2m": loss_cls_o2m_acc.squeeze(0),
            "loss_box_o2m": loss_box_o2m_acc.squeeze(0),
            "loss_cls_o2o": loss_cls_o2o_acc.squeeze(0),
            "loss_box_o2o": loss_box_o2o_acc.squeeze(0),
        }

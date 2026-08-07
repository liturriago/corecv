"""Detection losses for CoreCV.

Provides the :class:`DualHeadDetectionLoss`, the training criterion for
anchor-free dual-head detectors. It assigns ground-truth boxes to anchors
with task-aligned label assignment (TAL), trains a one-to-many and a
one-to-one head in parallel, and progressively shifts supervision weight
from the one-to-many head to the one-to-one head used at inference.

Reference:
    Feng et al., "Task-aligned One-stage Object Detection", ICCV 2021.
    Zheng et al., "Distance-IoU Loss: Faster and Better Learning for Bounding
    Box Regression", AAAI 2020.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

# Grid-cell offset used to center anchor points within each cell.
_CELL_OFFSET = 0.5

# Default pixel stride of each feature level, ordered finest to coarsest.
_DEFAULT_STRIDES: tuple[int, ...] = (8, 16, 32)

# Target size of a GT box dimension that is smaller than the finest stride.
_SMALL_TARGET_STRIDE = 16.0

# Bounds of the finest anchor-grid side used to derive anchor points.
_MIN_SIDE = 1
_MAX_SIDE = 2048

# Task-aligned metric exponents and stability constant.
_ALPHA = 0.5
_BETA = 6.0
_EPS = 1e-9

# Default per-term loss weights (box, class, regression).
_BOX_WEIGHT = 9.83
_CLS_WEIGHT = 0.65
_REG_WEIGHT = 0.96

# Default TAL top-k values.
_TOPK_O2M = 10
_TOPK_O2O = 7
_TOPK2_O2O = 1

# Default progressive one-to-many head weights: (start, end).
_PROGRESSIVE_WEIGHTS: tuple[float, float] = (0.8, 0.1)


@dataclass(frozen=True, slots=True)
class TaskAlignedConfig:
    """Hyperparameters of the task-aligned label assignment.

    Attributes:
        alpha: Exponent applied to the classification score.
        beta: Exponent applied to the box overlap.
        eps: Small constant for numerical stability.
        topk_one2many: Top-k candidates kept per object for the one-to-many
            head.
        topk_one2one: Top-k candidates kept per object for the one-to-one
            head before refinement.
        topk2_one2one: Number of anchors kept per object after refinement for
            the one-to-one head. Using ``1`` yields a single anchor per
            object, which enables NMS-free inference.

    """

    alpha: float = _ALPHA
    beta: float = _BETA
    eps: float = _EPS
    topk_one2many: int = _TOPK_O2M
    topk_one2one: int = _TOPK_O2O
    topk2_one2one: int = _TOPK2_O2O


@dataclass(frozen=True, slots=True)
class DetectionLossWeights:
    """Per-term weights of the detection loss.

    Attributes:
        box: Weight of the CIoU box loss.
        cls: Weight of the class BCE loss.
        reg: Weight of the distance regression loss.

    """

    box: float = _BOX_WEIGHT
    cls: float = _CLS_WEIGHT
    reg: float = _REG_WEIGHT


def _ciou(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Compute the Complete-IoU between boxes.

    Args:
        pred: Predicted boxes of shape ``(..., 4)`` in ``(x1, y1, x2, y2)``
            format.
        target: Target boxes of shape ``(..., 4)`` in the same format.
        eps: Small constant for numerical stability.

    Returns:
        CIoU tensor broadcast against the leading dimensions of *pred* and
        *target*.

    """
    pred_x1, pred_y1, pred_x2, pred_y2 = pred.unbind(dim=-1)
    target_x1, target_y1, target_x2, target_y2 = target.unbind(dim=-1)

    pred_w = (pred_x2 - pred_x1).clamp(min=eps)
    pred_h = (pred_y2 - pred_y1).clamp(min=eps)
    target_w = (target_x2 - target_x1).clamp(min=eps)
    target_h = (target_y2 - target_y1).clamp(min=eps)

    inter_x1 = torch.maximum(pred_x1, target_x1)
    inter_y1 = torch.maximum(pred_y1, target_y1)
    inter_x2 = torch.minimum(pred_x2, target_x2)
    inter_y2 = torch.minimum(pred_y2, target_y2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = pred_w * pred_h + target_w * target_h - inter
    iou = inter / union.clamp(min=eps)

    # Distance penalty: center distance normalized by the enclosing diagonal.
    enclosure_x1 = torch.minimum(pred_x1, target_x1)
    enclosure_y1 = torch.minimum(pred_y1, target_y1)
    enclosure_x2 = torch.maximum(pred_x2, target_x2)
    enclosure_y2 = torch.maximum(pred_y2, target_y2)
    diagonal = (enclosure_x2 - enclosure_x1) ** 2 + (enclosure_y2 - enclosure_y1) ** 2

    pred_center_x = (pred_x1 + pred_x2) / 2
    pred_center_y = (pred_y1 + pred_y2) / 2
    target_center_x = (target_x1 + target_x2) / 2
    target_center_y = (target_y1 + target_y2) / 2
    center_distance = (pred_center_x - target_center_x) ** 2 + (
        pred_center_y - target_center_y
    ) ** 2

    distance_iou = iou - center_distance / diagonal.clamp(min=eps)

    # Aspect-ratio penalty term.
    aspect = (
        (4 / math.pi**2)
        * (torch.atan(pred_w / pred_h) - torch.atan(target_w / target_h)) ** 2
    )
    alpha = aspect / (1 - iou + aspect).clamp(min=eps)
    return distance_iou - alpha * aspect


def _make_anchors(
    spatial_shapes: list[tuple[int, int]],
    strides: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Generate anchor points and strides for a feature pyramid.

    Args:
        spatial_shapes: Spatial ``(H, W)`` size of each feature level,
            ordered from finest to coarsest.
        strides: Pixel stride of each level relative to the input image.
        device: Device for the generated tensors.
        dtype: Dtype for the generated tensors.

    Returns:
        Tuple of ``(anchor_points, stride_tensor)`` with shapes ``(A, 2)``
        and ``(A, 1)``, where ``A`` is the total number of anchors.

    """
    points: list[Tensor] = []
    stride_tensors: list[Tensor] = []
    for (height, width), stride in zip(spatial_shapes, strides, strict=True):
        rows, cols = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="xy",
        )
        cell_points = torch.stack([cols, rows], dim=-1).reshape(-1, 2) + _CELL_OFFSET
        points.append(cell_points * stride)
        stride_tensors.append(
            torch.full((height * width, 1), float(stride), device=device, dtype=dtype),
        )
    return torch.cat(points, dim=0), torch.cat(stride_tensors, dim=0)


def _derive_anchors(
    num_boxes: int,
    strides: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Derive anchor points from the anchor count and the pyramid strides.

    Assumes a square input whose feature pyramid is produced by stride-2
    convolutions, i.e. each level halves its resolution with ``ceil``. The
    finest side length is found by search so the derived anchor count always
    matches *num_boxes*. For irregular input resolutions, pass explicit
    anchor points to the loss instead.

    Args:
        num_boxes: Total number of anchors across all levels.
        strides: Pixel stride of each level, ordered finest to coarsest.
        device: Device for the generated tensors.
        dtype: Dtype for the generated tensors.

    Returns:
        Tuple of ``(anchor_points, stride_tensor)`` with shapes ``(A, 2)``
        and ``(A, 1)``.

    """
    num_levels = len(strides)
    for side0 in range(_MIN_SIDE, _MAX_SIDE):
        sides = [side0]
        for _ in range(num_levels - 1):
            sides.append((sides[-1] + 1) // 2)
        if sum(side * side for side in sides) == num_boxes:
            spatial_shapes = [(side, side) for side in sides]
            return _make_anchors(spatial_shapes, strides, device=device, dtype=dtype)
    msg = f"could not derive an anchor grid for {num_boxes} anchors and strides {strides}"
    raise ValueError(msg)


def _box_distances(anchor_points: Tensor, boxes: Tensor, stride_tensor: Tensor) -> Tensor:
    """Compute stride-normalized distances from anchor points to box edges.

    Args:
        anchor_points: Anchor points of shape ``(N, 2)`` in image coordinates.
        boxes: Boxes of shape ``(N, 4)`` in ``(x1, y1, x2, y2)`` format.
        stride_tensor: Per-anchor strides of shape ``(N, 1)``.

    Returns:
        Distances ``[left, top, right, bottom]`` of shape ``(N, 4)`` in cell
        units.

    """
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    left = (anchor_points[:, 0] - x1) / stride_tensor[:, 0]
    top = (anchor_points[:, 1] - y1) / stride_tensor[:, 0]
    right = (x2 - anchor_points[:, 0]) / stride_tensor[:, 0]
    bottom = (y2 - anchor_points[:, 1]) / stride_tensor[:, 0]
    return torch.stack([left, top, right, bottom], dim=-1)


def _task_aligned_assign(  # noqa: PLR0913
    pred_scores: Tensor,
    pred_bboxes: Tensor,
    anchor_points: Tensor,
    stride_tensor: Tensor,
    gt_labels: Tensor,
    gt_bboxes: Tensor,
    gt_mask: Tensor,
    *,
    topk: int,
    topk2: int | None,
    alpha: float,
    beta: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Assign ground-truth boxes to anchors with task-aligned matching.

    Args:
        pred_scores: Detached class probabilities of shape ``(B, A, nc)``.
        pred_bboxes: Detached decoded boxes of shape ``(B, A, 4)`` in image
            coordinates.
        anchor_points: Anchor points of shape ``(A, 2)`` in image coordinates.
        stride_tensor: Per-anchor strides of shape ``(A, 1)``.
        gt_labels: Ground-truth class indices of shape ``(B, G)``.
        gt_bboxes: Ground-truth boxes of shape ``(B, G, 4)`` in image
            coordinates.
        gt_mask: Validity mask of shape ``(B, G)``.
        topk: Candidate anchors kept per object before refinement.
        topk2: Anchors kept per object after refinement. ``None`` keeps all
            *topk* candidates.
        alpha: Classification exponent of the alignment metric.
        beta: Overlap exponent of the alignment metric.
        eps: Small constant for numerical stability.

    Returns:
        Tuple of ``(target_scores, target_bboxes, fg_mask, target_gt_idx)``
        with shapes ``(B, A, nc)``, ``(B, A, 4)``, ``(B, A)``, and
        ``(B, A)``.

    """
    batch, num_anchors, num_classes = pred_scores.shape
    num_gt = gt_labels.shape[1]

    # STAL: expand per-dimension GT boxes smaller than the finest stride.
    x1, y1, x2, y2 = gt_bboxes.unbind(dim=-1)
    widths = x2 - x1
    heights = y2 - y1
    min_stride = stride_tensor.min()
    small_width = (widths < min_stride) & gt_mask
    small_height = (heights < min_stride) & gt_mask
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    half_extent = _SMALL_TARGET_STRIDE / 2
    x1 = torch.where(small_width, center_x - half_extent, x1)
    x2 = torch.where(small_width, center_x + half_extent, x2)
    y1 = torch.where(small_height, center_y - half_extent, y1)
    y2 = torch.where(small_height, center_y + half_extent, y2)

    # Step 1: candidate anchors whose center lies inside the GT box.
    center_x = anchor_points[:, 0].unsqueeze(0).unsqueeze(1)
    center_y = anchor_points[:, 1].unsqueeze(0).unsqueeze(1)
    mask_in_gts = (
        (center_x > x1.unsqueeze(-1) + eps)
        & (center_x < x2.unsqueeze(-1) - eps)
        & (center_y > y1.unsqueeze(-1) + eps)
        & (center_y < y2.unsqueeze(-1) - eps)
    )

    # Step 2: task-aligned metric combining class and localization quality.
    class_scores = pred_scores.permute(0, 2, 1)
    bbox_scores = class_scores.gather(
        1,
        gt_labels.unsqueeze(-1).expand(batch, num_gt, num_anchors),
    )
    overlaps = _ciou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(2)).clamp(min=0.0)
    alignment = bbox_scores.pow(alpha) * overlaps.pow(beta)
    alignment = alignment * gt_mask.unsqueeze(-1)

    # Step 3: top-k per object with cross-object deduplication.
    _, topk_indices = alignment.topk(topk, dim=-1)
    mask_topk = torch.zeros_like(alignment)
    mask_topk.scatter_(-1, topk_indices, 1.0)
    in_multiple = mask_topk.sum(dim=1) > 1
    mask_topk = mask_topk * (~in_multiple).unsqueeze(1)
    mask_pos = mask_topk * mask_in_gts * gt_mask.unsqueeze(-1)

    # Step 4: conflict resolution and optional topk2 refinement.
    overlap_masked = overlaps * mask_pos
    fg_mask = overlap_masked.max(dim=1).values > 0
    if topk2 is not None and topk2 != topk:
        masked_alignment = alignment * mask_pos
        _, topk2_indices = masked_alignment.topk(topk2, dim=-1)
        mask_topk2 = torch.zeros_like(alignment)
        mask_topk2.scatter_(-1, topk2_indices, 1.0)
        mask_pos = mask_pos * mask_topk2
        overlap_masked = overlaps * mask_pos
        fg_mask = overlap_masked.max(dim=1).values > 0
    target_gt_idx = overlap_masked.argmax(dim=1)

    # Step 5: soft IoU-aware class targets.
    best_alignment = (alignment * mask_pos).max(dim=-1).values
    best_overlap = overlap_masked.max(dim=-1).values
    normalized = (
        alignment
        * mask_pos
        * best_overlap.unsqueeze(-1)
        / (best_alignment.unsqueeze(-1) + eps)
    )
    anchor_scores = normalized.max(dim=1).values
    target_labels = gt_labels.gather(1, target_gt_idx)
    target_bboxes = gt_bboxes.gather(
        1,
        target_gt_idx.unsqueeze(-1).expand(-1, -1, 4),
    )
    target_scores = F.one_hot(target_labels, num_classes).float() * anchor_scores.unsqueeze(-1)

    return target_scores, target_bboxes, fg_mask, target_gt_idx


class DualHeadDetectionLoss(nn.Module):
    """Dual-head detection loss with task-aligned label assignment.

    Trains the one-to-many and one-to-one heads of an anchor-free dual-head
    detector in parallel. The one-to-many head supervises the backbone with a
    dense assignment, while the one-to-one head imitates it with a single
    anchor per object. A progressive weight shifts supervision from the
    one-to-many head toward the one-to-one head used at inference.

    Targets are provided as a flat ``(N, 6)`` tensor with columns
    ``[batch_index, class, x1, y1, x2, y2]``.

    Example:
        >>> import torch
        >>> from corecv.losses.detection import DualHeadDetectionLoss
        >>> loss_fn = DualHeadDetectionLoss(num_classes=4)
        >>> preds = (torch.zeros(2, 336, 4), torch.ones(2, 336, 4))
        >>> targets = torch.tensor([[0, 1, 10.0, 10.0, 40.0, 40.0]])
        >>> loss = loss_fn(preds, preds, targets)
        >>> loss["loss_total"].shape
        torch.Size([])

    """

    def __init__(
        self,
        num_classes: int,
        *,
        strides: tuple[int, ...] = _DEFAULT_STRIDES,
        weights: DetectionLossWeights | None = None,
        assignment: TaskAlignedConfig | None = None,
        progressive_weights: tuple[float, float] = _PROGRESSIVE_WEIGHTS,
    ) -> None:
        """Initialize the dual-head detection loss.

        Args:
            num_classes: Number of output classes.
            strides: Pixel stride of each feature level, ordered finest to
                coarsest. Must match the detection model's pyramid.
            weights: Per-term loss weights. Defaults to the standard recipe.
            assignment: Task-aligned assignment hyperparameters.
            progressive_weights: ``(start, end)`` weights of the one-to-many
                head across training; the one-to-one head weight is the
                complement.

        """
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.weights = weights if weights is not None else DetectionLossWeights()
        self.assignment = assignment if assignment is not None else TaskAlignedConfig()
        self.progressive_weights = progressive_weights
        self.progress = 0.0

    def set_progress(self, progress: float) -> None:
        """Set the training progress used by the progressive head weights.

        Args:
            progress: Fraction of training completed, in ``[0, 1]``.

        """
        self.progress = min(max(progress, 0.0), 1.0)

    def _build_targets(self, targets: Tensor, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        """Split flat targets into per-image padded ground-truth tensors.

        Args:
            targets: Flat targets of shape ``(N, 6)`` with columns
                ``[batch_index, class, x1, y1, x2, y2]``.
            batch_size: Number of images in the batch.

        Returns:
            Tuple of ``(gt_labels, gt_bboxes, gt_mask)`` with shapes
            ``(B, G)``, ``(B, G, 4)``, and ``(B, G)``.

        """
        counts = torch.bincount(targets[:, 0].long(), minlength=batch_size)
        num_gt = int(counts.max().item())
        device = targets.device

        gt_labels = torch.zeros(batch_size, num_gt, dtype=torch.long, device=device)
        gt_bboxes = torch.zeros(batch_size, num_gt, 4, dtype=targets.dtype, device=device)
        gt_mask = torch.zeros(batch_size, num_gt, dtype=torch.bool, device=device)

        for batch_idx in range(batch_size):
            per_image = targets[targets[:, 0] == batch_idx]
            num = per_image.shape[0]
            if num == 0:
                continue
            gt_labels[batch_idx, :num] = per_image[:, 1].long()
            gt_bboxes[batch_idx, :num] = per_image[:, 2:6]
            gt_mask[batch_idx, :num] = True
        return gt_labels, gt_bboxes, gt_mask

    def _head_loss(  # noqa: PLR0913
        self,
        logits: Tensor,
        boxes: Tensor,
        anchor_points: Tensor,
        stride_tensor: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        gt_mask: Tensor,
        *,
        topk: int,
        topk2: int | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the loss of a single detection head.

        Args:
            logits: Class logits of shape ``(B, A, nc)``.
            boxes: Decoded boxes of shape ``(B, A, 4)`` in image coordinates.
            anchor_points: Anchor points of shape ``(A, 2)``.
            stride_tensor: Per-anchor strides of shape ``(A, 1)``.
            gt_labels: Ground-truth class indices of shape ``(B, G)``.
            gt_bboxes: Ground-truth boxes of shape ``(B, G, 4)``.
            gt_mask: Validity mask of shape ``(B, G)``.
            topk: Candidate anchors kept per object.
            topk2: Anchors kept per object after refinement.

        Returns:
            Tuple of ``(iou_loss, cls_loss, reg_loss)`` scalar tensors.

        """
        if gt_labels.shape[1] == 0:
            zero = torch.zeros((), device=logits.device, dtype=torch.float32)
            return zero, zero, zero

        target_scores, target_bboxes, fg_mask, _ = _task_aligned_assign(
            logits.sigmoid().detach(),
            boxes.detach(),
            anchor_points,
            stride_tensor,
            gt_labels,
            gt_bboxes,
            gt_mask,
            topk=topk,
            topk2=topk2,
            alpha=self.assignment.alpha,
            beta=self.assignment.beta,
            eps=self.assignment.eps,
        )

        score_sum = target_scores.sum().clamp(min=1.0)
        weights = target_scores[fg_mask].sum(dim=-1, keepdim=True)

        # Expand anchor metadata to the batch dimension for boolean indexing.
        batch = logits.shape[0]
        anchor_full = anchor_points.unsqueeze(0).expand(batch, -1, -1)
        stride_full = stride_tensor.unsqueeze(0).expand(batch, -1, -1)

        # CIoU box loss on stride-normalized boxes.
        pred_norm = boxes[fg_mask] / stride_full[fg_mask]
        target_norm = target_bboxes[fg_mask] / stride_full[fg_mask]
        iou_loss = ((1 - _ciou(pred_norm, target_norm)) * weights.squeeze(-1)).sum() / score_sum

        # Class BCE over all anchors with soft IoU-aware targets.
        cls_loss = (
            F.binary_cross_entropy_with_logits(logits, target_scores, reduction="sum")
            / score_sum
        )

        # Distance regression L1 on stride-normalized distances.
        pred_distances = _box_distances(
            anchor_full[fg_mask],
            boxes[fg_mask],
            stride_full[fg_mask],
        )
        target_distances = _box_distances(
            anchor_full[fg_mask],
            target_bboxes[fg_mask],
            stride_full[fg_mask],
        )
        reg_loss = (
            (pred_distances - target_distances).abs().sum(dim=-1, keepdim=True) * weights
        ).sum() / score_sum

        return iou_loss, cls_loss, reg_loss

    def forward(
        self,
        preds_o2m: tuple[Tensor, Tensor],
        preds_o2o: tuple[Tensor, Tensor],
        targets: Tensor,
        *,
        anchor_points: Tensor | None = None,
        stride_tensor: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute the combined dual-head detection loss.

        Args:
            preds_o2m: ``(logits, boxes)`` of the one-to-many head with shapes
                ``(B, A, nc)`` and ``(B, A, 4)``.
            preds_o2o: ``(logits, boxes)`` of the one-to-one head.
            targets: Flat targets of shape ``(N, 6)`` with columns
                ``[batch_index, class, x1, y1, x2, y2]``.
            anchor_points: Optional anchor points of shape ``(A, 2)``. When
                ``None``, they are derived from the anchor count assuming a
                square power-of-two pyramid.
            stride_tensor: Optional per-anchor strides of shape ``(A, 1)``.

        Returns:
            Dictionary with the weighted ``loss_iou``, ``loss_cls``,
            ``loss_reg`` terms and the combined ``loss_total``.

        """
        o2m_logits, o2m_boxes = preds_o2m
        o2o_logits, o2o_boxes = preds_o2o
        batch_size = o2m_logits.shape[0]
        gt_labels, gt_bboxes, gt_mask = self._build_targets(targets, batch_size)

        if anchor_points is None or stride_tensor is None:
            anchor_points, stride_tensor = _derive_anchors(
                o2m_boxes.shape[1],
                self.strides,
                device=o2m_boxes.device,
                dtype=o2m_boxes.dtype,
            )

        o2m_losses = self._head_loss(
            o2m_logits,
            o2m_boxes,
            anchor_points,
            stride_tensor,
            gt_labels,
            gt_bboxes,
            gt_mask,
            topk=self.assignment.topk_one2many,
            topk2=None,
        )
        o2o_losses = self._head_loss(
            o2o_logits,
            o2o_boxes,
            anchor_points,
            stride_tensor,
            gt_labels,
            gt_bboxes,
            gt_mask,
            topk=self.assignment.topk_one2one,
            topk2=self.assignment.topk2_one2one,
        )

        start, end = self.progressive_weights
        weight_o2m = start + (end - start) * self.progress
        weight_o2o = 1.0 - weight_o2m

        iou_term = weight_o2m * o2m_losses[0] + weight_o2o * o2o_losses[0]
        cls_term = weight_o2m * o2m_losses[1] + weight_o2o * o2o_losses[1]
        reg_term = weight_o2m * o2m_losses[2] + weight_o2o * o2o_losses[2]

        return {
            "loss_iou": self.weights.box * iou_term,
            "loss_cls": self.weights.cls * cls_term,
            "loss_reg": self.weights.reg * reg_term,
            "loss_total": self.weights.box * iou_term
            + self.weights.cls * cls_term
            + self.weights.reg * reg_term,
        }

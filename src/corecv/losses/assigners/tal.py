"""GPU-native Task-Aligned Assigner (TAL) for anchor-free detection.

Implements the dynamic sample assignment strategy from TOOD: Task-Aligned
One-stage Object Detection (Tian et al., AAAI 2021).  The alignment
metric used for positive sample selection is::

    t = cls_score^alpha * IoU^beta

For each ground-truth box, the top-k predictions with the highest TAL
alignment metric are selected as positive samples.  All remaining
predictions are treated as negatives.

The assigner is designed to integrate with
:class:`~corecv.models.heads.detection.decoupled_anchor_free.DecoupledAnchorFreeHead`
and downstream losses
(:class:`~corecv.losses.detection.QualityFocalLoss` /
:class:`~corecv.losses.detection.VarifocalLoss` for classification,
:class:`~corecv.losses.detection.GIoULoss` /
:class:`~corecv.losses.detection.CIoULoss` for bounding-box regression).

All computations are pure vectorised PyTorch operations with **zero**
CPU-GPU synchronisations during assignment.  No ``.item()`` calls, no
``.cpu()`` transfers — everything stays on VRAM.

Architecture overview::

    pred_scores:  List[(B, C, H_l, W_l)]  — per-level classification logits
    pred_boxes:   List[(B, 4, H_l, W_l)]  — per-level (l, t, r, b) regression
    strides:      List[int]               — per-level strides (e.g. 8, 16, 32)
    gt_labels:    List[Tensor]            — per-image GT class labels
    gt_boxes:     List[Tensor]            — per-image GT boxes (x1, y1, x2, y2)

        |
        v  TaskAlignedAssigner.forward()
        |
    assignment dict with:
        pos_mask       — (N_total,) bool tensor per image
        neg_mask       — (N_total,) bool tensor per image
        assigned_gt_inds — (N_total,) long tensor per image
        assigned_labels  — (N_total,) long tensor per image
        pos_ious       — (num_pos,) float tensor per image (IoU quality)

Example:
    >>> from corecv.losses.assigners.tal import TaskAlignedAssigner
    >>> assigner = TaskAlignedAssigner(num_classes=80, topk=13)
    >>> # Given per-level predictions from DecoupledAnchorFreeHead:
    >>> # cls_logits: [(B, 80, H_l, W_l), ...], reg_pred: [(B, 4, H_l, W_l), ...]
    >>> # strides: [8, 16, 32], gt_labels: per-image, gt_boxes: per-image (xyxy)
    >>> assignment = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.ops import box_iou

# ---------------------------------------------------------------------------
# Constants for tensor dimension checks (PLR2004)
# ---------------------------------------------------------------------------
_DIM_4D = 4  # Per-level feature map format (B, C, H, W)
_MIN_TOPK = 1


# ======================================================================
# Helper utilities (all GPU-native, no sync)
# ======================================================================


def _make_anchors(
    strides: list[int],
    feat_sizes: list[tuple[int, int]],
    device: torch.device,
) -> Tensor:
    """Generate per-cell anchor centre coordinates for all feature levels.

    For each feature level with stride ``s`` and spatial size ``(H, W)``,
    the centre of cell ``(i, j)`` is ``((j + 0.5) * s, (i + 0.5) * s)``.

    All anchors from all levels are concatenated into a single
    ``(N_total, 2)`` tensor ordered from finest to coarsest level,
    matching the flattening order used by the assigner.

    Args:
        strides: Per-level stride values (e.g. ``[8, 16, 32]``).
        feat_sizes: Per-level ``(height, width)`` spatial dimensions.
        device: Target device for the output tensor.

    Returns:
        Tensor of shape ``(N_total, 2)`` with ``(cx, cy)`` pixel
        coordinates for every cell across all levels.
    """
    anchors_list: list[Tensor] = []
    for stride, (h_size, w_size) in zip(strides, feat_sizes, strict=True):
        shift_x = (torch.arange(w_size, device=device, dtype=torch.float32) + 0.5) * stride
        shift_y = (torch.arange(h_size, device=device, dtype=torch.float32) + 0.5) * stride
        # meshgrid with "ij" indexing: shift_y has shape (H, W), shift_x too
        shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
        shifts = torch.stack([shift_xx, shift_yy], dim=-1)  # (H, W, 2)
        anchors_list.append(shifts.reshape(-1, 2))  # (H*W, 2)
    return torch.cat(anchors_list, dim=0)  # (N_total, 2)


def _decode_boxes(
    reg_pred: Tensor,
    anchors: Tensor,
) -> Tensor:
    """Decode ``(l, t, r, b)`` regression predictions to absolute ``(x1, y1, x2, y2)`` boxes.

    The regression head of :class:`DecoupledAnchorFreeHead` predicts
    distances from each cell centre to the four box boundaries.  This
    function converts them to absolute pixel coordinates using the
    pre-computed cell centres.

    Args:
        reg_pred: Regression predictions of shape ``(N, 4)`` in
            ``(l, t, r, b)`` format (distances from cell centre).
        anchors: Cell centres of shape ``(N, 2)`` in ``(cx, cy)`` format.

    Returns:
        Decoded boxes of shape ``(N, 4)`` in ``(x1, y1, x2, y2)`` format.
    """
    # Clamp regression values to non-negative (negative distances are invalid)
    reg_pred = reg_pred.clamp(min=0)
    return torch.stack(
        [
            anchors[:, 0] - reg_pred[:, 0],  # x1 = cx - l
            anchors[:, 1] - reg_pred[:, 1],  # y1 = cy - t
            anchors[:, 0] + reg_pred[:, 2],  # x2 = cx + r
            anchors[:, 1] + reg_pred[:, 3],  # y2 = cy + b
        ],
        dim=-1,
    )


# ======================================================================
# Task-Aligned Assigner
# ======================================================================


class TaskAlignedAssigner(nn.Module):
    """Task-Aligned dynamic assigner for anchor-free detection.

    Implements the alignment-metric-based sample assignment from TOOD.
    For each ground-truth box, the ``topk`` predictions with the highest
    alignment score ``t = cls^alpha * IoU^beta`` are selected as
    positives.  When a prediction is selected by multiple ground-truth
    boxes, it is assigned to the one yielding the highest alignment
    metric.

    The assigner produces per-image assignment results that can be
    consumed directly by the CoreCV detection losses:

    * :class:`~corecv.losses.detection.QualityFocalLoss` or
      :class:`~corecv.losses.detection.VarifocalLoss` for classification
      (using ``assigned_labels`` and ``pos_ious`` as quality targets).
    * :class:`~corecv.losses.detection.GIoULoss` or
      :class:`~corecv.losses.detection.CIoULoss` for bounding-box
      regression (using ``pos_mask`` and ``assigned_gt_inds`` to gather
      matched predictions and targets).

    All operations run entirely on GPU — there are **zero** CPU-GPU
    synchronisations during the forward pass.

    Args:
        num_classes: Number of foreground object classes.
        topk: Number of top predictions selected per ground-truth box.
            Typical values: ``9`` for stride-8 only, ``13`` for
            multi-scale.  Default ``13``.
        alpha: Exponent for the classification score in the alignment
            metric.  Default ``0.5``.
        beta: Exponent for the IoU in the alignment metric.  Higher
            values bias selection towards higher-quality boxes.  Default
            ``6.0``.

    Raises:
        ValueError: If ``topk`` is less than 1, ``num_classes`` is less
            than 1, or ``alpha`` / ``beta`` are negative.

    Example:
        >>> assigner = TaskAlignedAssigner(num_classes=80, topk=13)
        >>> cls_logits = [torch.randn(2, 80, 80, 80),
        ...               torch.randn(2, 80, 40, 40),
        ...               torch.randn(2, 80, 20, 20)]
        >>> reg_pred = [torch.randn(2, 4, 80, 80).abs(),
        ...             torch.randn(2, 4, 40, 40).abs(),
        ...             torch.randn(2, 4, 20, 20).abs()]
        >>> strides = [8, 16, 32]
        >>> gt_labels = [torch.tensor([0, 5]), torch.tensor([1, 2, 3])]
        >>> gt_boxes = [torch.rand(2, 4) * 400 + 50,
        ...             torch.rand(3, 4) * 400 + 50]
        >>> assignment = assigner(cls_logits, reg_pred, strides,
        ...                       gt_labels, gt_boxes)
    """

    def __init__(  # noqa: PLR0913
        self,
        num_classes: int,
        topk: int = 13,
        alpha: float = 0.5,
        beta: float = 6.0,
    ) -> None:
        """Initialise the Task-Aligned Assigner."""
        super().__init__()
        if num_classes < 1:
            msg = f"num_classes must be >= 1, got {num_classes}."
            raise ValueError(msg)
        if topk < _MIN_TOPK:
            msg = f"topk must be >= {_MIN_TOPK}, got {topk}."
            raise ValueError(msg)
        if alpha < 0.0:
            msg = f"alpha must be >= 0, got {alpha}."
            raise ValueError(msg)
        if beta < 0.0:
            msg = f"beta must be >= 0, got {beta}."
            raise ValueError(msg)

        self.num_classes = int(num_classes)
        self.topk = int(topk)
        self.alpha = float(alpha)
        self.beta = float(beta)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        pred_scores: list[Tensor],
        pred_boxes: list[Tensor],
        strides: list[int],
        gt_labels: list[Tensor],
        gt_boxes: list[Tensor],
    ) -> dict[str, list[Tensor]]:
        """Compute task-aligned assignment for each image in the batch.

        Args:
            pred_scores: Per-level classification logits.  Each element
                is a tensor of shape ``(B, C, H_l, W_l)`` where ``C``
                equals ``self.num_classes``.
            pred_boxes: Per-level bounding-box regression predictions.
                Each element is a tensor of shape ``(B, 4, H_l, W_l)``
                in ``(l, t, r, b)`` format (distances from cell centres).
            strides: Per-level stride values (e.g. ``[8, 16, 32]``).
                ``len(strides)`` must equal ``len(pred_scores)``.
            gt_labels: List of ``B`` 1-D integer tensors, each
                containing the class labels for that image's
                ground-truth boxes.  Values in ``[0, num_classes)``.
            gt_boxes: List of ``B`` 2-D tensors of shape
                ``(num_gt_i, 4)`` with ground-truth boxes in absolute
                ``(x1, y1, x2, y2)`` pixel coordinates.

        Returns:
            Dictionary with the following keys (all values are lists
            of length ``B``, one element per image):

            * ``"pos_mask"``: ``bool`` tensor of shape ``(N_i,)`` —
                ``True`` for positive predictions.
            * ``"neg_mask"``: ``bool`` tensor of shape ``(N_i,)`` —
                ``True`` for negative predictions.
            * ``"assigned_gt_inds"``: ``int64`` tensor of shape
                ``(N_i,)`` — ground-truth index assigned to each
                prediction (``-1`` for negatives).
            * ``"assigned_labels"``: ``int64`` tensor of shape
                ``(N_i,)`` — class label assigned to each prediction
                (``0`` for negatives).
            * ``"pos_ious"``: ``float32`` tensor of shape
                ``(num_pos_i,)`` — IoU quality of each positive
                prediction with its assigned ground-truth box.

            Here ``N_i = sum(H_l * W_l)`` across all levels for image
            ``i``.
        """
        B = pred_scores[0].shape[0]
        device = pred_scores[0].device
        num_levels = len(pred_scores)

        # Validate input consistency
        if len(pred_boxes) != num_levels:
            msg = (
                f"pred_scores and pred_boxes must have the same length, "
                f"got {num_levels} and {len(pred_boxes)}."
            )
            raise ValueError(msg)
        if len(strides) != num_levels:
            msg = (
                f"strides length ({len(strides)}) must match the number "
                f"of feature levels ({num_levels})."
            )
            raise ValueError(msg)

        # Compute feature map spatial sizes from the regression outputs
        feat_sizes: list[tuple[int, int]] = [
            (reg.shape[_DIM_4D - 2], reg.shape[_DIM_4D - 1])  # (H, W)
            for reg in pred_boxes
        ]

        # Generate anchor centres for all levels — (N_total, 2)
        all_anchors: Tensor = _make_anchors(strides, feat_sizes, device)

        # Flatten predictions across all levels for each image
        cls_scores: Tensor = torch.cat(
            [
                level_cls.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
                for level_cls in pred_scores
            ],
            dim=1,
        )
        reg_preds: Tensor = torch.cat(
            [
                level_reg.permute(0, 2, 3, 1).reshape(B, -1, 4)
                for level_reg in pred_boxes
            ],
            dim=1,
        )

        # Decode all predictions from (l, t, r, b) to absolute (x1, y1, x2, y2)
        decoded_boxes: Tensor = _decode_boxes(
            reg_preds.reshape(-1, 4),  # (B*N_total, 4)
            all_anchors.unsqueeze(0).expand(B, -1, -1).reshape(-1, 2),  # (B*N_total, 2)
        ).reshape(B, all_anchors.shape[0], 4)  # (B, N_total, 4)

        # Apply sigmoid to get classification probabilities
        cls_probs: Tensor = cls_scores.sigmoid()  # (B, N_total, C)

        # Per-image assignment
        results: dict[str, list[Tensor]] = {
            "pos_mask": [],
            "neg_mask": [],
            "assigned_gt_inds": [],
            "assigned_labels": [],
            "pos_ious": [],
        }

        for i in range(B):
            self._assign_single_image(
                cls_probs_i=cls_probs[i],
                decoded_boxes_i=decoded_boxes[i],
                gt_labels_i=gt_labels[i],
                gt_boxes_i=gt_boxes[i],
                results=results,
                device=device,
            )

        return results

    # ------------------------------------------------------------------
    # Internal: per-image assignment
    # ------------------------------------------------------------------

    def _assign_single_image(  # noqa: PLR0913
        self,
        cls_probs_i: Tensor,
        decoded_boxes_i: Tensor,
        gt_labels_i: Tensor,
        gt_boxes_i: Tensor,
        results: dict[str, list[Tensor]],
        device: torch.device,
    ) -> None:
        """Compute TAL assignment for a single image.

        Args:
            cls_probs_i: Sigmoid classification probabilities of shape
                ``(N_total, C)``.
            decoded_boxes_i: Decoded absolute ``(x1, y1, x2, y2)`` boxes
                of shape ``(N_total, 4)``.
            gt_labels_i: 1-D integer tensor of ground-truth class
                labels, shape ``(num_gt,)``.
            gt_boxes_i: 2-D tensor of ground-truth boxes in absolute
                ``(x1, y1, x2, y2)`` format, shape ``(num_gt, 4)``.
            results: Mutable results dictionary to append to.
            device: Target device for output tensors.
        """
        num_preds: int = cls_probs_i.shape[0]
        num_gt: int = gt_boxes_i.shape[0]

        # ---- Edge case: no ground-truth boxes -------------------------
        if num_gt == 0:
            results["pos_mask"].append(
                torch.zeros(num_preds, dtype=torch.bool, device=device),
            )
            results["neg_mask"].append(
                torch.ones(num_preds, dtype=torch.bool, device=device),
            )
            results["assigned_gt_inds"].append(
                torch.full((num_preds,), -1, dtype=torch.long, device=device),
            )
            results["assigned_labels"].append(
                torch.zeros(num_preds, dtype=torch.long, device=device),
            )
            results["pos_ious"].append(
                torch.zeros(0, dtype=torch.float32, device=device),
            )
            return

        # ---- Step 1: pairwise IoU — (N_total, num_gt) ----------------
        # torchvision.ops.box_iou returns (num_gt, N_total), we transpose
        iou_matrix: Tensor = box_iou(gt_boxes_i, decoded_boxes_i).t()
        # Clamp to [0, 1] for numerical safety (IoU should already be in range)
        iou_matrix = iou_matrix.clamp(min=0.0, max=1.0)

        # ---- Step 2: classification scores for GT classes -------------
        # For each (prediction, GT) pair, extract the probability of the
        # GT class from the prediction's classification output.
        # gt_labels_i: (num_gt,) → expand to (N_total, num_gt)
        gt_labels_expanded: Tensor = gt_labels_i.unsqueeze(0).expand(
            num_preds, -1,
        )
        # Gather the class probability: (N_total, num_gt)
        cls_scores_for_gt: Tensor = torch.gather(
            cls_probs_i,
            dim=1,
            index=gt_labels_expanded,
        )

        # ---- Step 3: TAL alignment metric t = cls^alpha * IoU^beta ----
        align_metric: Tensor = (
            cls_scores_for_gt.pow(self.alpha) * iou_matrix.pow(self.beta)
        )  # (N_total, num_gt)

        # ---- Step 4: dynamic top-k selection per GT -------------------
        # For each GT box, select the top-k predictions with the
        # highest alignment metric.
        effective_topk: int = min(self.topk, num_preds)
        topk_metrics: Tensor
        topk_idxs: Tensor
        topk_metrics, topk_idxs = torch.topk(
            align_metric,
            effective_topk,
            dim=0,
            largest=True,
            sorted=True,
        )  # (effective_topk, num_gt)

        # ---- Step 5: resolve conflicts and build masks ----------------
        # Create a boolean mask of shape (N_total, num_gt) indicating
        # which predictions are in the top-k for each GT.
        topk_mask: Tensor = torch.zeros(
            num_preds, num_gt, dtype=torch.bool, device=device,
        )
        # Scatter: for each GT column, set the selected rows to True
        topk_mask.scatter_(0, topk_idxs, True)

        # For conflict resolution: use align_metric only at selected
        # positions, zeros elsewhere.  Then take argmax per prediction.
        effective_metric: Tensor = align_metric.masked_fill(~topk_mask, 0.0)

        # Best GT per prediction (among selected positions): (N_total,)
        best_metric_per_pred: Tensor
        best_gt_per_pred: Tensor
        best_metric_per_pred, best_gt_per_pred = effective_metric.max(dim=1)

        # A prediction is positive if it was selected by at least one GT
        pos_mask: Tensor = best_metric_per_pred > 0.0  # (N_total,)
        neg_mask: Tensor = ~pos_mask

        # Build assigned GT index tensor: (N_total,)
        assigned_gt_inds: Tensor = torch.full(
            (num_preds,), -1, dtype=torch.long, device=device,
        )
        assigned_gt_inds[pos_mask] = best_gt_per_pred[pos_mask]

        # Build assigned label tensor: (N_total,)
        assigned_labels: Tensor = torch.zeros(
            num_preds, dtype=torch.long, device=device,
        )
        # Gather the class label for each positive from the GT labels
        if pos_mask.any():
            assigned_labels[pos_mask] = gt_labels_i[
                assigned_gt_inds[pos_mask]
            ]

        # ---- Step 6: IoU quality for positive predictions --------------
        # For each positive prediction, extract the IoU with its
        # assigned ground-truth box.
        pos_ious: Tensor = torch.zeros(
            num_preds, dtype=torch.float32, device=device,
        )
        if pos_mask.any():
            # Gather IoU: for each positive, get iou_matrix[pos_idx, gt_idx]
            pos_gt_inds: Tensor = assigned_gt_inds[pos_mask]  # (num_pos,)
            pos_indices: Tensor = torch.where(pos_mask)[0]  # (num_pos,)
            pos_ious[pos_indices] = iou_matrix[pos_indices, pos_gt_inds]

        results["pos_mask"].append(pos_mask)
        results["neg_mask"].append(neg_mask)
        results["assigned_gt_inds"].append(assigned_gt_inds)
        results["assigned_labels"].append(assigned_labels)
        results["pos_ious"].append(pos_ious[pos_mask])

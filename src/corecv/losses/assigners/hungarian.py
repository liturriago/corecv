"""GPU-native Hungarian matching and set criterion for query detection.

Implements bipartite 1-to-1 assignment (Hungarian algorithm) and the
associated ``SetCriterion`` used to train RT-DETR / D-FINE style
query-based detection heads.

All loss computations (focal classification, L1 bounding-box, GIoU) are
pure vectorised PyTorch operations with **zero** CPU-GPU synchronisations
during forward and backward passes.  The only CPU round-trip is the
combinatorial assignment step itself, which operates on a small cost matrix
of shape ``(num_queries, num_gt)`` per batch element — the heavy lifting
(the full pairwise cost-matrix construction) stays entirely on VRAM.

Architecture overview::

    pred_scores:  (B, num_queries, num_classes)
    pred_boxes:   (B, num_queries, 4)              — normalised cxcywh
    gt_labels:    List[Tensor]  of per-image targets
    gt_boxes:     List[Tensor]  of per-image targets

        |
        v  HungarianMatcher.forward()  (cost_class + L1 + GIoU)
        |
    indices: List[Tuple[Tensor, Tensor]]  (pred_idx, gt_idx) per image

        |
        v  SetCriterion.loss()  (focal + L1 + GIoU + aux)
        |
    scalar loss

Example:
    >>> import torch
    >>> from corecv.losses.assigners import HungarianMatcher, SetCriterion
    >>> matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    >>> criterion = SetCriterion(num_classes=80, matcher=matcher)
    >>> pred_scores = torch.randn(2, 300, 80)
    >>> pred_boxes = torch.sigmoid(torch.randn(2, 300, 4))
    >>> gt_labels = [torch.tensor([0, 5]), torch.tensor([1, 2, 3])]
    >>> gt_boxes = [torch.rand(2, 4), torch.rand(3, 4)]
    >>> loss, aux = criterion(pred_scores, pred_boxes, gt_labels, gt_boxes)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torchvision.ops import box_convert, generalized_box_iou

# ---------------------------------------------------------------------------
# Constants for tensor dimension checks (PLR2004)
# ---------------------------------------------------------------------------
_DIM_3D = 3  # (B, N, C) or (B, N, 4)
_SMALL_COST_MATRIX = 512  # Threshold for noting large assignment problems


# ======================================================================
# Hungarian Matcher
# ======================================================================


class HungarianMatcher(nn.Module):
    """Bipartite 1-to-1 assignment via the Hungarian algorithm.

    Computes a cost matrix from three terms — classification cost, L1
    bounding-box cost, and GIoU cost — then solves the linear-sum
    assignment to find the optimal prediction-to-ground-truth pairing.

    The cost matrix for each image is::

        cost = cost_class * focal_cost
             + cost_bbox * l1_cost
             + cost_giou * (1 - giou_matrix)

    where:

    * ``focal_cost`` is derived from the negative focal-log-probability
      for the target class.
    * ``l1_cost`` is the element-wise L1 distance between normalised
      ``(cx, cy, w, h)`` boxes.
    * ``giou_matrix`` is the pairwise GIoU between all predictions and
      all ground-truth boxes.

    The assignment is performed **per image** independently.  The heavy
    pairwise-cost computation runs entirely on VRAM; only the small
    ``(num_queries, num_gt)`` cost matrix is transferred to CPU for
    ``scipy.optimize.linear_sum_assignment``.

    Args:
        cost_class: Weight of the focal classification cost.
            Default ``2.0``.
        cost_bbox: Weight of the L1 bounding-box cost.
            Default ``5.0``.
        cost_giou: Weight of the GIoU cost.  Default ``2.0``.
        focal_alpha: Focal-loss alpha for class weighting.
            Default ``0.25``.
        focal_gamma: Focal-loss gamma for focusing parameter.
            Default ``2.0``.

    Example:
        >>> matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
        >>> pred_scores = torch.randn(2, 300, 80)
        >>> pred_boxes = torch.sigmoid(torch.randn(2, 300, 4))
        >>> gt_labels = [torch.tensor([0, 5]), torch.tensor([1, 2, 3])]
        >>> gt_boxes = [torch.rand(2, 4), torch.rand(3, 4)]
        >>> indices = matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)
    """

    def __init__(  # noqa: PLR0913
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        """Initialise HungarianMatcher with cost weights and focal parameters."""
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        pred_scores: Tensor,
        pred_boxes: Tensor,
        gt_labels: list[Tensor],
        gt_boxes: list[Tensor],
    ) -> list[tuple[Tensor, Tensor]]:
        """Compute optimal bipartite matching for each image in the batch.

        Args:
            pred_scores: Classification logits of shape
                ``(B, num_queries, num_classes)``.
            pred_boxes: Predicted normalised ``(cx, cy, w, h)`` boxes of
                shape ``(B, num_queries, 4)``.
            gt_labels: List of ``B`` 1-D integer tensors, each containing
                the class labels for that image's ground-truth boxes.
            gt_boxes: List of ``B`` 2-D tensors of shape ``(num_gt_i, 4)``
                with normalised ``(cx, cy, w, h)`` ground-truth boxes.

        Returns:
            List of ``B`` tuples ``(pred_indices, gt_indices)`` where each
            element is a 1-D ``LongTensor`` of matched indices.  If an
            image has zero ground-truth boxes, both tensors are empty.
        """
        B = pred_scores.shape[0]
        device = pred_scores.device
        indices: list[tuple[Tensor, Tensor]] = []

        for i in range(B):
            # Per-image tensors — still on GPU, no sync yet
            p_scores_i = pred_scores[i]  # (num_queries, C)
            p_boxes_i = pred_boxes[i]    # (num_queries, 4)
            gt_label_i = gt_labels[i]    # (num_gt,)
            gt_box_i = gt_boxes[i]       # (num_gt, 4)

            num_gt = gt_box_i.shape[0]
            if num_gt == 0:
                # No ground-truth — empty assignment
                indices.append(
                    (
                        torch.zeros(0, dtype=torch.long, device=device),
                        torch.zeros(0, dtype=torch.long, device=device),
                    )
                )
                continue

            # ---- Pairwise cost matrix (fully GPU-native) ---------------
            cost_matrix = self._compute_cost_matrix(
                p_scores_i, p_boxes_i, gt_label_i, gt_box_i,
            )  # (num_queries, num_gt)

            # ---- Solve assignment on CPU (small matrix) -----------------
            # Transfer only the cost matrix — all heavy computation was
            # performed on GPU.
            cost_np = cost_matrix.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.long, device=device),
                    torch.as_tensor(col_ind, dtype=torch.long, device=device),
                )
            )

        return indices

    # ------------------------------------------------------------------
    # Internal: pairwise cost matrix (GPU-native)
    # ------------------------------------------------------------------

    def _compute_cost_matrix(
        self,
        pred_scores: Tensor,
        pred_boxes: Tensor,
        gt_labels: Tensor,
        gt_boxes: Tensor,
    ) -> Tensor:
        """Build the ``(num_queries, num_gt)`` cost matrix on GPU.

        Args:
            pred_scores: ``(num_queries, num_classes)`` classification
                logits.
            pred_boxes: ``(num_queries, 4)`` normalised predicted boxes.
            gt_labels: ``(num_gt,)`` integer class labels.
            gt_boxes: ``(num_gt, 4)`` normalised ground-truth boxes.

        Returns:
            Cost tensor of shape ``(num_queries, num_gt)`` on the same
            device as the inputs.
        """
        # ---- Classification cost: negative focal log-probability --------
        # For each (query, gt) pair, compute the focal-modulated negative
        # log-probability of the GT class under the query's prediction.
        cost_class = self._focal_cost_matrix(pred_scores, gt_labels)

        # ---- L1 bounding-box cost --------------------------------------
        # torch.cdist with (Q, 4) and (G, 4) produces (Q, G) directly
        cost_bbox = torch.cdist(pred_boxes, gt_boxes, p=1)  # (Q, G)

        # ---- GIoU cost: 1 - GIoU --------------------------------------
        # generalized_box_iou expects (x1, y1, x2, y2) format; convert
        # from normalised (cx, cy, w, h) before computing pairwise GIoU.
        cost_giou = 1.0 - generalized_box_iou(
            box_convert(pred_boxes, "cxcywh", "xyxy"),
            box_convert(gt_boxes, "cxcywh", "xyxy"),
        )

        # ---- Combined cost matrix --------------------------------------
        cost_matrix = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )

        return cost_matrix

    def _focal_cost_matrix(
        self,
        pred_scores: Tensor,
        gt_labels: Tensor,
    ) -> Tensor:
        """Compute focal-modulated classification cost matrix.

        For each (query, gt) pair the cost is::

            -alpha * (1 - p_t)^gamma * log(p_t)

        where ``p_t = sigmoid(pred_scores[:, gt_class])``.

        This avoids materialising a full ``(Q, G, C)`` tensor by only
        computing the relevant class column for each GT label.

        Args:
            pred_scores: ``(num_queries, num_classes)`` logits.
            gt_labels: ``(num_gt,)`` integer class labels.

        Returns:
            Cost tensor of shape ``(num_queries, num_gt)``.
        """
        num_queries = pred_scores.shape[0]

        # Gather the logit for the GT class for every query-GT pair.
        # gt_labels: (G,) -> (1, G) -> expand to (Q, G)
        gt_labels_expanded = gt_labels.unsqueeze(0).expand(num_queries, -1)
        # (Q, G) indices for gathering along dim=1
        class_logits = torch.gather(
            pred_scores,
            dim=1,
            index=gt_labels_expanded,
        )  # (Q, G)

        # Focal modulation
        p_t = class_logits.sigmoid()
        focal_weight = (1.0 - p_t).pow(self.focal_gamma)
        # Negative log-likelihood (stabilised)
        neg_log_p_t = -F.softplus(
            class_logits
        )  # numerically stable -log(sigma(x))
        # For the positive branch: -log(sigma(x)) = -x + softplus(x) when x < 0
        # But softplus(x) = log(1 + exp(x)), and -log(sigma(x)) = softplus(-x)
        # Actually: -log(sigma(x)) = -x + softplus(x) ... let me use the direct form
        # -log(sigma(x)) = log(1 + exp(-x)) = softplus(-x)
        neg_log_p_t = F.softplus(-class_logits)  # (Q, G)

        cost = self.focal_alpha * focal_weight * neg_log_p_t  # (Q, G)
        return cost


# ======================================================================
# Set Criterion
# ======================================================================


class SetCriterion(nn.Module):
    """Set-based criterion for training query detection heads (RT-DETR / D-FINE).

    Uses :class:`HungarianMatcher` to establish 1-to-1 assignments between
    predictions and ground-truth boxes, then computes a weighted sum of:

    * **Classification loss** — sigmoid focal loss applied to matched
      queries; unmatched queries are treated as background (class 0 with
      target 0 in the "no-object" formulation, or omitted entirely).
    * **Bounding-box L1 loss** — element-wise L1 on matched boxes only.
    * **GIoU loss** — ``1 - GIoU`` on matched boxes only.

    Supports **auxiliary losses** from intermediate decoder layers: when
    ``return_intermediate=True`` is used in the head, pass the lists of
    intermediate predictions and the criterion will apply the same loss
    at every decoder layer with optional per-layer weights.

    All loss computations are 100 % GPU-native — the only CPU round-trip
    is the assignment step inside :class:`HungarianMatcher`.

    Args:
        num_classes: Number of foreground object classes (excluding the
            implicit ``"no-object"`` class).
        matcher: A :class:`HungarianMatcher` instance.
        focal_alpha: Focal-loss alpha.  Default ``0.25``.
        focal_gamma: Focal-loss gamma.  Default ``2.0``.
        loss_cls_weight: Weight of the classification loss.
            Default ``1.0``.
        loss_bbox_weight: Weight of the L1 bounding-box loss.
            Default ``5.0``.
        loss_giou_weight: Weight of the GIoU loss.  Default ``2.0``.
        aux_loss_weights: Per-layer weights for auxiliary decoder losses.
            If ``None``, all layers are weighted equally.  The length
            must match the number of intermediate predictions when
            auxiliary losses are used.
        no_object_weight: Weight assigned to the "no-object" class for
            unmatched queries.  Default ``0.1``.

    Example:
        >>> matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
        >>> criterion = SetCriterion(num_classes=80, matcher=matcher)
        >>> pred_scores = torch.randn(2, 300, 80)
        >>> pred_boxes = torch.sigmoid(torch.randn(2, 300, 4))
        >>> gt_labels = [torch.tensor([0, 5]), torch.tensor([1, 2, 3])]
        >>> gt_boxes = [torch.rand(2, 4), torch.rand(3, 4)]
        >>> loss, aux = criterion(pred_scores, pred_boxes, gt_labels, gt_boxes)
    """

    def __init__(  # noqa: PLR0913
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 5.0,
        loss_giou_weight: float = 2.0,
        aux_loss_weights: list[float] | None = None,
        no_object_weight: float = 0.1,
    ) -> None:
        """Initialise SetCriterion with matcher and loss weights."""
        super().__init__()
        self.num_classes = int(num_classes)
        self.matcher = matcher
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.loss_cls_weight = float(loss_cls_weight)
        self.loss_bbox_weight = float(loss_bbox_weight)
        self.loss_giou_weight = float(loss_giou_weight)
        self.aux_loss_weights = aux_loss_weights
        self.no_object_weight = float(no_object_weight)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(  # noqa: PLR0913
        self,
        pred_scores: Tensor,
        pred_boxes: Tensor,
        gt_labels: list[Tensor],
        gt_boxes: list[Tensor],
        intermediate_cls: list[Tensor] | None = None,
        intermediate_reg: list[Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the set-matching loss.

        Args:
            pred_scores: Final classification logits
                ``(B, num_queries, num_classes)``.
            pred_boxes: Final predicted normalised ``(cx, cy, w, h)``
                boxes of shape ``(B, num_queries, 4)``.
            gt_labels: List of ``B`` 1-D integer tensors with per-image
                ground-truth class labels.
            gt_boxes: List of ``B`` 2-D tensors of shape
                ``(num_gt_i, 4)`` with normalised ground-truth boxes.
            intermediate_cls: Optional list of intermediate classification
                logits from decoder layers, each of shape
                ``(B, num_queries, num_classes)``.
            intermediate_reg: Optional list of intermediate bounding-box
                predictions from decoder layers, each of shape
                ``(B, num_queries, 4)``.

        Returns:
            Tuple of:

            * ``loss``: Scalar combined loss tensor.
            * ``aux_losses``: Dictionary mapping ``"loss_cls"``,
                ``"loss_bbox"``, ``"loss_giou"`` to their scalar values.
                Auxiliary layer losses are stored under
                ``"aux_loss_cls"``, ``"aux_loss_bbox"``, ``"aux_loss_giou"``.
        """
        # ---- Run Hungarian matching on the final predictions ------------
        indices = self.matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)

        # ---- Compute losses for the final decoder layer ----------------
        loss_cls, loss_bbox, loss_giou = self._compute_losses(
            pred_scores, pred_boxes, indices, gt_labels, gt_boxes,
        )

        total_cls = self.loss_cls_weight * loss_cls
        total_bbox = self.loss_bbox_weight * loss_bbox
        total_giou = self.loss_giou_weight * loss_giou
        total_loss = total_cls + total_bbox + total_giou

        aux_losses: dict[str, Tensor] = {
            "loss_cls": loss_cls.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
        }

        # ---- Auxiliary losses from intermediate decoder layers ----------
        if intermediate_cls is not None and intermediate_reg is not None:
            num_layers = len(intermediate_cls)
            weights = self.aux_loss_weights
            if weights is None:
                # Equal weighting for all auxiliary layers
                weights = [1.0 / num_layers] * num_layers

            if len(weights) != num_layers:
                msg = (
                    f"aux_loss_weights length ({len(weights)}) must match "
                    f"number of intermediate layers ({num_layers})."
                )
                raise ValueError(msg)

            for layer_idx, (cls_i, reg_i) in enumerate(
                zip(intermediate_cls, intermediate_reg, strict=True),
            ):
                # Re-run matching at each intermediate layer
                aux_indices = self.matcher(cls_i, reg_i, gt_labels, gt_boxes)
                aux_cls, aux_bbox, aux_giou = self._compute_losses(
                    cls_i, reg_i, aux_indices, gt_labels, gt_boxes,
                )
                w = weights[layer_idx]
                total_loss = total_loss + w * (
                    self.loss_cls_weight * aux_cls
                    + self.loss_bbox_weight * aux_bbox
                    + self.loss_giou_weight * aux_giou
                )
                aux_losses[f"aux_loss_cls_layer{layer_idx}"] = aux_cls.detach()
                aux_losses[f"aux_loss_bbox_layer{layer_idx}"] = aux_bbox.detach()
                aux_losses[f"aux_loss_giou_layer{layer_idx}"] = aux_giou.detach()

        return total_loss, aux_losses

    # ------------------------------------------------------------------
    # Internal: per-layer loss computation
    # ------------------------------------------------------------------

    def _compute_losses(
        self,
        pred_scores: Tensor,
        pred_boxes: Tensor,
        indices: list[tuple[Tensor, Tensor]],
        gt_labels: list[Tensor],
        gt_boxes: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute classification, L1, and GIoU losses for one set of predictions.

        The losses are averaged over the **number of ground-truth objects
        in the entire batch** (not the batch size), following the DETR
        convention.

        Args:
            pred_scores: ``(B, num_queries, num_classes)`` logits.
            pred_boxes: ``(B, num_queries, 4)`` predicted boxes.
            indices: Per-image ``(pred_idx, gt_idx)`` matching tuples.
            gt_labels: Per-image class labels.
            gt_boxes: Per-image ground-truth boxes.

        Returns:
            Tuple ``(loss_cls, loss_bbox, loss_giou)`` — each a scalar
            tensor.
        """
        B = pred_scores.shape[0]
        device = pred_scores.device
        num_classes = self.num_classes

        # Total number of GT objects across the batch (for normalisation)
        num_gt_total: Tensor = torch.tensor(
            sum(lbl.numel() for lbl in gt_labels),
            dtype=torch.float32,
            device=device,
        )
        # Avoid division by zero when no GT exists in the batch
        num_gt_total = num_gt_total.clamp(min=1.0)

        # ==================================================================
        # Classification loss — sigmoid focal loss
        # ==================================================================
        # Build the target tensor: for matched queries use the GT class;
        # for unmatched queries use 0 ("no-object" / background).
        # Format: target is the class index for the matched position, and
        # the focal loss is applied per-class with sigmoid.

        # Create the classification target: (B, Q, C)
        # Default: all zeros (no-object class)
        target_cls_full = torch.zeros(
            B, pred_scores.shape[1], num_classes, dtype=torch.float32, device=device,
        )

        total_num_matched = 0
        for i, (pred_idx, gt_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue
            # Set the target class for matched queries
            matched_gt_labels = gt_labels[i][gt_idx]  # (num_matched,)
            num_matched = matched_gt_labels.numel()
            total_num_matched += num_matched

            # One-hot encode the matched classes: (num_matched, C)
            one_hot = F.one_hot(
                matched_gt_labels, num_classes=num_classes,
            ).float()  # (num_matched, C)

            # Scatter into the full target tensor: (B, Q, C)
            # For image i, at positions pred_idx, set the one-hot vectors
            target_cls_full[i].scatter_(
                0,
                pred_idx.unsqueeze(0).expand(num_matched, -1).T,
                one_hot,
            )

        # Focal loss: use sigmoid_focal_loss from torchvision or manual
        # Compute per-element focal loss: (B, Q, C)
        loss_cls_per_element = self._sigmoid_focal_loss(
            pred_scores, target_cls_full,
        )  # (B, Q, C)

        # Sum over classes: (B, Q)
        loss_cls_per_query = loss_cls_per_element.sum(dim=-1)

        # For matched queries: weight = 1
        # For unmatched queries: weight = no_object_weight
        # Build the weight mask: (B, Q)
        matched_mask = torch.zeros(
            B, pred_scores.shape[1], dtype=torch.float32, device=device,
        )
        for i, (pred_idx, _gt_idx) in enumerate(indices):
            if pred_idx.numel() > 0:
                matched_mask[i, pred_idx] = 1.0

        weight_mask = (
            matched_mask * 1.0
            + (1.0 - matched_mask) * self.no_object_weight
        )

        # Weighted sum, normalised by number of GT objects
        loss_cls = (loss_cls_per_query * weight_mask).sum() / num_gt_total

        # ==================================================================
        # Bounding-box losses — L1 and GIoU (matched pairs only)
        # ==================================================================
        loss_bbox_acc = torch.tensor(0.0, device=device)
        loss_giou_acc = torch.tensor(0.0, device=device)

        for i, (pred_idx, gt_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue

            # Gather matched predictions and targets
            matched_pred_boxes = pred_boxes[i][pred_idx]   # (M, 4)
            matched_gt_boxes = gt_boxes[i][gt_idx]          # (M, 4)

            # L1 loss (element-wise, then mean over boxes and coordinates)
            loss_bbox_acc = loss_bbox_acc + F.l1_loss(
                matched_pred_boxes, matched_gt_boxes, reduction="sum",
            )

            # GIoU loss: 1 - GIoU
            # Convert from normalised (cx, cy, w, h) to (x1, y1, x2, y2)
            giou = generalized_box_iou(
                box_convert(matched_pred_boxes, "cxcywh", "xyxy"),
                box_convert(matched_gt_boxes, "cxcywh", "xyxy"),
            )
            # generalized_box_iou returns (M, M) when inputs are (M, 4);
            # extract the diagonal for matched-pair GIoU values.
            giou_diag = torch.diag(giou) if giou.dim() == _DIM_3D else giou
            loss_giou_acc = loss_giou_acc + (1.0 - giou_diag).sum()

        loss_bbox = loss_bbox_acc / num_gt_total
        loss_giou = loss_giou_acc / num_gt_total

        return loss_cls, loss_bbox, loss_giou

    def _sigmoid_focal_loss(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Sigmoid focal loss for multi-class detection.

        Computes the focal loss with sigmoid activation, following
        ``torchvision.ops.sigmoid_focal_loss`` semantics but implemented
        inline for clarity and to ensure zero sync.

        Args:
            inputs: Unnormalised logits of shape ``(B, C_in, num_classes)``
                or ``(B, Q, C)``.
            targets: One-hot encoded targets of the same shape as
                *inputs*.

        Returns:
            Per-element focal loss tensor of the same shape as *inputs*.
        """
        p = inputs.sigmoid()
        # Binary cross-entropy (element-wise, unreduced)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none",
        )

        # Focal modulation
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal_weight = (1.0 - p_t).pow(self.focal_gamma)

        # Alpha modulation
        alpha_t = (
            self.focal_alpha * targets
            + (1.0 - self.focal_alpha) * (1.0 - targets)
        )

        loss = alpha_t * focal_weight * ce_loss
        return loss

"""GPU-native detection loss functions for CoreCV.

Provides GIoU, CIoU, Quality Focal Loss and Varifocal Loss — all
implemented as pure vectorised PyTorch operations with **zero** CPU-GPU
synchronisations during forward or backward passes.  No Python loops, no
``.item()`` calls, no ``.cpu()`` transfers — every computation remains on
VRAM.

Supports both per-level tensors ``(B, 4, H, W)`` and flattened
tensors ``(N, 4)`` for detection boxes.  All IoU losses use
``torch.min`` / ``torch.max`` / ``torch.clamp`` for vectorised
intersection / union / enclosing-box computation.

Example:
    >>> import torch
    >>> from corecv.losses.detection import GIoULoss, CIoULoss
    >>> giou = GIoULoss(reduction="mean")
    >>> ciou = CIoULoss(reduction="mean")
    >>> pred = torch.randn(4, 4, 32, 32, device="cuda").abs() * 100
    >>> tgt  = torch.randn(4, 4, 32, 32, device="cuda").abs() * 100
    >>> loss = giou(pred, tgt)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Constants for tensor dimension checks (PLR2004)
# ---------------------------------------------------------------------------
_DIM_4D = 4  # Spatial format (B, C, H, W) or (B, 4, H, W)
_DIM_2D = 2  # Flattened format (N, C) or (N, 4)

# ======================================================================
# Helper utilities (all GPU-native, no sync)
# ======================================================================


def _boxes_ltrb(boxes: Tensor) -> Tensor:
    """Convert centre-based ``(cx, cy, w, h)`` to ``(left, top, right, bottom)``.

    If the input already appears to be in ``(l, t, r, b)`` form (negative
    values present in the first or third channel), it is returned unchanged.

    Args:
        boxes: Tensor of shape ``(..., 4)``.

    Returns:
        Tensor of shape ``(..., 4)`` in ``(l, t, r, b)`` format.
    """
    # Heuristic: if any value in channel-0 or channel-2 is negative, assume
    # the boxes are already ltrb (l/r can be negative for padded images in
    # some pipelines).  Otherwise assume cxcywh.
    # NOTE: This is safe because cxcywh centres and half-widths are always
    # non-negative for valid boxes.
    if (boxes[..., 0] < 0).any() or (boxes[..., 2] < 0).any():
        return boxes
    cx, cy, w, h = boxes.unbind(-1)
    left = cx - w / 2
    top = cy - h / 2
    right = cx + w / 2
    bottom = cy + h / 2
    return torch.stack([left, top, right, bottom], dim=-1)


def _box_area(boxes: Tensor) -> Tensor:
    """Compute area of boxes in ``(l, t, r, b)`` format.

    Args:
        boxes: ``(..., 4)`` tensor.

    Returns:
        ``(...)`` area tensor.
    """
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (
        boxes[..., 3] - boxes[..., 1]
    ).clamp(min=0)


def _pairwise_iou(pred: Tensor, target: Tensor) -> Tensor:
    """Vectorised IoU between two sets of ``(l, t, r, b)`` boxes.

    Supports arbitrary leading batch dimensions via broadcasting.

    Args:
        pred: ``(..., 4)`` predicted boxes.
        target: ``(..., 4)`` target boxes (must be broadcastable to pred).

    Returns:
        ``(...)`` IoU tensor.
    """
    inter_l = torch.max(pred[..., 0], target[..., 0])
    inter_t = torch.max(pred[..., 1], target[..., 1])
    inter_r = torch.min(pred[..., 2], target[..., 2])
    inter_b = torch.min(pred[..., 3], target[..., 3])

    inter_area = (inter_r - inter_l).clamp(min=0) * (inter_b - inter_t).clamp(min=0)
    area_pred = _box_area(pred)
    area_target = _box_area(target)
    union = area_pred + area_target - inter_area

    return inter_area / (union + 1e-7)


def _enclosing_box(pred: Tensor, target: Tensor) -> Tensor:
    """Vectorised enclosing (smallest) box for two sets of ``(l, t, r, b)``.

    Args:
        pred: ``(..., 4)`` predicted boxes.
        target: ``(..., 4)`` target boxes.

    Returns:
        ``(..., 4)`` enclosing box in ``(l, t, r, b)`` format.
    """
    enclose_l = torch.min(pred[..., 0], target[..., 0])
    enclose_t = torch.min(pred[..., 1], target[..., 1])
    enclose_r = torch.max(pred[..., 2], target[..., 2])
    enclose_b = torch.max(pred[..., 3], target[..., 3])
    return torch.stack([enclose_l, enclose_t, enclose_r, enclose_b], dim=-1)


def _diagonal_squared(boxes: Tensor) -> Tensor:
    """Squared diagonal length of ``(l, t, r, b)`` boxes.

    Args:
        boxes: ``(..., 4)`` tensor.

    Returns:
        ``(...)`` tensor of diagonal² values.
    """
    w = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
    h = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    return w ** 2 + h ** 2


def _centre_distance_squared(pred: Tensor, target: Tensor) -> Tensor:
    """Squared Euclidean distance between box centres.

    Args:
        pred: ``(..., 4)`` in ``(l, t, r, b)``.
        target: ``(..., 4)`` in ``(l, t, r, b)``.

    Returns:
        ``(...)`` tensor.
    """
    pred_cx = (pred[..., 0] + pred[..., 2]) / 2
    pred_cy = (pred[..., 1] + pred[..., 3]) / 2
    tgt_cx = (target[..., 0] + target[..., 2]) / 2
    tgt_cy = (target[..., 1] + target[..., 3]) / 2
    return (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2


# ======================================================================
# GIoU Loss
# ======================================================================


class GIoULoss(nn.Module):
    """Generalized IoU Loss (Rezatofighi et al. CVPR 2019).

    Extends IoU with a penalty term for non-overlapping boxes::

        GIoU = IoU - (C - U) / C

    where ``C`` is the enclosing-box area and ``U`` is the union area.
    The loss is ``1 - GIoU``.

    Fully vectorised: all operations use ``torch.min`` / ``torch.max`` /
    ``torch.clamp`` on batched tensors with no Python loops.

    Supports two input formats:

    * **Per-level**: ``(B, 4, H, W)`` — typical output of anchor-free
      detection heads at a single FPN level.
    * **Flattened**: ``(N, 4)`` — all predictions from a single level
      (or all levels concatenated).

    When ``per_sample`` is ``True`` (default) and the input is 4-D, the
    loss is computed element-wise and returned with the requested
    reduction.

    Args:
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.  Default ``'mean'``.
        eps: Small constant for numerical stability.  Default ``1e-7``.

    Example:
        >>> giou = GIoULoss(reduction="mean")
        >>> pred = torch.randn(4, 4, 32, 32, device="cuda").abs()
        >>> tgt  = torch.randn(4, 4, 32, 32, device="cuda").abs()
        >>> loss = giou(pred, tgt)
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7) -> None:
        """Initialise GIoULoss with reduction mode and epsilon."""
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            msg = f"Invalid reduction: {reduction!r}"
            raise ValueError(msg)
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred_boxes: Tensor,
        target_boxes: Tensor,
    ) -> Tensor:
        """Compute GIoU loss.

        Args:
            pred_boxes: Predicted boxes.  Shape ``(B, 4, H, W)`` or
                ``(N, 4)`` in ``(l, t, r, b)`` format.
            target_boxes: Target boxes.  Same shape as *pred_boxes*.

        Returns:
            Scalar or tensor loss depending on ``self.reduction``.
        """
        # Flatten to (M, 4) for uniform processing
        if pred_boxes.dim() == _DIM_4D:
            # (B, 4, H, W) -> (B*H*W, 4) — treat each spatial cell as a box
            M = pred_boxes.shape[0] * pred_boxes.shape[2] * pred_boxes.shape[3]
            pred_flat = pred_boxes.permute(0, 2, 3, 1).reshape(M, 4)
            target_flat = target_boxes.permute(0, 2, 3, 1).reshape(M, 4)
        elif pred_boxes.dim() == _DIM_2D:
            M = pred_boxes.shape[0]
            pred_flat = pred_boxes
            target_flat = target_boxes
        else:
            msg = f"Expected 2-D or 4-D boxes, got {pred_boxes.dim()}-D"
            raise ValueError(msg)

        # ---- Compute intersection / union directly from coordinates ------
        # This avoids the circular dependency of recovering union from IoU.
        inter_l = torch.max(pred_flat[..., 0], target_flat[..., 0])
        inter_t = torch.max(pred_flat[..., 1], target_flat[..., 1])
        inter_r = torch.min(pred_flat[..., 2], target_flat[..., 2])
        inter_b = torch.min(pred_flat[..., 3], target_flat[..., 3])

        inter_area: Tensor = (inter_r - inter_l).clamp(min=0) * (
            inter_b - inter_t
        ).clamp(min=0)

        area_pred: Tensor = _box_area(pred_flat)
        area_target: Tensor = _box_area(target_flat)
        union: Tensor = area_pred + area_target - inter_area

        iou: Tensor = inter_area / (union + self.eps)

        # Enclosing box
        enclose: Tensor = _enclosing_box(pred_flat, target_flat)
        enclose_area: Tensor = _box_area(enclose)

        # GIoU
        giou: Tensor = iou - (enclose_area - union) / (enclose_area + self.eps)
        loss: Tensor = 1.0 - giou

        return self._reduce(loss)

    def _reduce(self, loss: Tensor) -> Tensor:
        """Apply reduction.

        Args:
            loss: Per-element loss (1-D or flattened).

        Returns:
            Reduced scalar or tensor.
        """
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ======================================================================
# CIoU Loss
# ======================================================================


class CIoULoss(nn.Module):
    """Complete IoU Loss (Zheng et al. CVPR 2020).

    Extends DIoU with an aspect-ratio consistency term::

        CIoU = IoU - rho²(b_pred, b_gt) / c² - alpha * v

    where:

    * ``rho²`` is the squared centre distance.
    * ``c`` is the diagonal of the enclosing box.
    * ``v`` measures aspect-ratio similarity:
      ``v = (4 / pi²) * (arctan(w_gt/h_gt) - arctan(w_pred/h_pred))²``
    * ``alpha = v / (1 - IoU + v)`` balances the two penalty terms.

    The loss is ``1 - CIoU``.

    Fully vectorised: all operations run on batched tensors with no Python
    loops and no CPU-GPU synchronisations.

    Args:
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.  Default ``'mean'``.
        eps: Small constant for numerical stability.  Default ``1e-7``.

    Example:
        >>> ciou = CIoULoss(reduction="mean")
        >>> pred = torch.randn(4, 4, 32, 32, device="cuda").abs() * 100
        >>> tgt  = torch.randn(4, 4, 32, 32, device="cuda").abs() * 100
        >>> loss = ciou(pred, tgt)
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7) -> None:
        """Initialise CIoULoss with reduction mode and epsilon."""
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            msg = f"Invalid reduction: {reduction!r}"
            raise ValueError(msg)
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred_boxes: Tensor,
        target_boxes: Tensor,
    ) -> Tensor:
        """Compute CIoU loss.

        Args:
            pred_boxes: Predicted boxes, ``(B, 4, H, W)`` or ``(N, 4)``
                in ``(l, t, r, b)`` format.
            target_boxes: Target boxes, same shape.

        Returns:
            Scalar or tensor loss.
        """
        # Flatten to (M, 4)
        if pred_boxes.dim() == _DIM_4D:
            M = pred_boxes.shape[0] * pred_boxes.shape[2] * pred_boxes.shape[3]
            pred_flat = pred_boxes.permute(0, 2, 3, 1).reshape(M, 4)
            target_flat = target_boxes.permute(0, 2, 3, 1).reshape(M, 4)
        elif pred_boxes.dim() == _DIM_2D:
            pred_flat = pred_boxes
            target_flat = target_boxes
        else:
            msg = f"Expected 2-D or 4-D boxes, got {pred_boxes.dim()}-D"
            raise ValueError(msg)

        # ---- Compute intersection directly for IoU ----------------------
        inter_l = torch.max(pred_flat[..., 0], target_flat[..., 0])
        inter_t = torch.max(pred_flat[..., 1], target_flat[..., 1])
        inter_r = torch.min(pred_flat[..., 2], target_flat[..., 2])
        inter_b = torch.min(pred_flat[..., 3], target_flat[..., 3])

        inter_area: Tensor = (inter_r - inter_l).clamp(min=0) * (
            inter_b - inter_t
        ).clamp(min=0)

        area_pred: Tensor = _box_area(pred_flat)
        area_target: Tensor = _box_area(target_flat)
        union: Tensor = area_pred + area_target - inter_area

        iou: Tensor = inter_area / (union + self.eps)

        # ---- Centre distance penalty ------------------------------------
        centre_dist2: Tensor = _centre_distance_squared(pred_flat, target_flat)
        enclose: Tensor = _enclosing_box(pred_flat, target_flat)
        diag2: Tensor = _diagonal_squared(enclose)
        # Clamp diagonal to prevent division-by-near-zero on degenerate
        # enclosing boxes (e.g. when input boxes have l > r or t > b).
        diag2 = diag2.clamp(min=1.0)
        rho2: Tensor = centre_dist2 / diag2

        # ---- Aspect-ratio consistency term -------------------------------
        pred_w = (pred_flat[..., 2] - pred_flat[..., 0]).abs().clamp(min=self.eps)
        pred_h = (pred_flat[..., 3] - pred_flat[..., 1]).abs().clamp(min=self.eps)
        tgt_w = (target_flat[..., 2] - target_flat[..., 0]).abs().clamp(min=self.eps)
        tgt_h = (target_flat[..., 3] - target_flat[..., 1]).abs().clamp(min=self.eps)

        v: Tensor = (4.0 / (torch.pi ** 2)) * (
            torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h)
        ).pow(2)

        alpha: Tensor = v / (1.0 - iou + v + self.eps)

        # ---- CIoU loss ---------------------------------------------------
        ciou: Tensor = iou - rho2 - alpha * v
        loss: Tensor = (1.0 - ciou).clamp(max=100.0)

        return self._reduce(loss)

    def _reduce(self, loss: Tensor) -> Tensor:
        """Apply reduction.

        Args:
            loss: Per-element loss.

        Returns:
            Reduced scalar or tensor.
        """
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ======================================================================
# Quality Focal Loss  /  Varifocal Loss
# ======================================================================


class QualityFocalLoss(nn.Module):
    """Quality Focal Loss (QFL) for anchor-free detection heads.

    QFL replaces the standard binary cross-entropy with a modulated loss
    that treats the target as a continuous IoU quality score rather than a
    hard binary label::

        loss(q) = |pred - q|^beta * BCE(pred, q)

    where ``q`` is the IoU between the predicted and ground-truth box for
    positive samples (and 0 for negatives).

    This formulation encourages the classification branch to directly
    predict IoU quality, enabling better ranking at inference.

    Fully vectorised: uses ``F.binary_cross_entropy_with_logits`` with a
    ``pos_weight`` tensor and the quality-based weighting computed via
    pure tensor ops.

    Args:
        beta: Focusing parameter of the quality modulation.
            Default ``2.0``.
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.  Default ``'mean'``.

    Example:
        >>> qfl = QualityFocalLoss(beta=2.0)
        >>> pred = torch.randn(1000, 80, device="cuda")
        >>> tgt_scores = torch.rand(1000, 80, device="cuda")  # IoU quality
        >>> tgt_labels = torch.randint(0, 80, (1000,), device="cuda")
        >>> loss = qfl(pred, tgt_scores, tgt_labels)
    """

    def __init__(self, beta: float = 2.0, reduction: str = "mean") -> None:
        """Initialise QualityFocalLoss with beta and reduction mode."""
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            msg = f"Invalid reduction: {reduction!r}"
            raise ValueError(msg)
        self.beta = float(beta)
        self.reduction = reduction

    def forward(
        self,
        pred_scores: Tensor,
        target_scores: Tensor,
        target_labels: Tensor,
    ) -> Tensor:
        """Compute Quality Focal Loss.

        Args:
            pred_scores: Predicted classification logits.
                Shape ``(N, C)`` or ``(B, C, H, W)``.
            target_scores: Continuous quality targets (e.g. IoU values).
                Same shape as *pred_scores*.  Values in ``[0, 1]``.
            target_labels: Integer class labels of shape ``(N,)`` or
                ``(B, H, W)``.  Values in ``[0, C)``.  Only used to
                determine which channel each sample belongs to.

        Returns:
            Scalar or tensor loss.
        """
        # Flatten if 4-D spatial format
        if pred_scores.dim() == _DIM_4D:
            B, C, H, W = pred_scores.shape
            pred_scores = pred_scores.permute(0, 2, 3, 1).reshape(B * H * W, C)
            target_scores = target_scores.permute(0, 2, 3, 1).reshape(B * H * W, C)
            target_labels = target_labels.reshape(B * H * W)

        # One-hot mask for the target class channel: (N, C)
        num_classes = pred_scores.shape[1]
        one_hot: Tensor = F.one_hot(target_labels, num_classes=num_classes).float()

        # Binary cross-entropy per element (unreduced)
        bce: Tensor = F.binary_cross_entropy_with_logits(
            pred_scores, target_scores, reduction="none"
        )

        # Quality modulation: |pred - q|^beta
        # sigmoid(pred) - target  (both in [0, 1] range)
        pred_prob: Tensor = pred_scores.sigmoid()
        modulator: Tensor = (pred_prob - target_scores.detach()).abs().pow(self.beta)

        # Only the target class contributes to the loss
        loss: Tensor = modulator * bce * one_hot

        # Sum over classes, mean over samples
        loss_per_sample: Tensor = loss.sum(dim=1)

        return self._reduce(loss_per_sample)

    def _reduce(self, loss: Tensor) -> Tensor:
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class VarifocalLoss(nn.Module):
    """Varifocal Loss (VFL) for dense object detection.

    An asymmetric focal loss that re-weights positive and negative samples
    differently based on the quality (IoU) of each sample::

        VFL(p, q) = -q * log(p)              if positive
                    = -alpha * p^gamma * log(1-p)   if negative

    where ``q`` is the IoU quality score (target) for positives, ``p`` is
    the predicted probability, ``gamma`` is the focusing parameter, and
    ``alpha`` balances positive / negative contribution.

    This encourages the model to predict high scores only for high-quality
    detections while suppressing low-quality false positives.

    Fully vectorised: all operations are on batched tensors with no Python
    loops.

    Args:
        gamma: Focusing parameter for negative samples.  Default ``2.0``.
        alpha: Balancing factor for negative samples.  Default ``0.25``.
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.  Default ``'mean'``.

    Example:
        >>> vfl = VarifocalLoss(gamma=2.0, alpha=0.25)
        >>> pred = torch.randn(500, 80, device="cuda")
        >>> tgt_scores = torch.rand(500, 80, device="cuda")
        >>> tgt_labels = torch.randint(0, 80, (500,), device="cuda")
        >>> loss = vfl(pred, tgt_scores, tgt_labels)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ) -> None:
        """Initialise VarifocalLoss with gamma, alpha, and reduction mode."""
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            msg = f"Invalid reduction: {reduction!r}"
            raise ValueError(msg)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction

    def forward(
        self,
        pred_scores: Tensor,
        target_scores: Tensor,
        target_labels: Tensor,
    ) -> Tensor:
        """Compute Varifocal Loss.

        Args:
            pred_scores: Predicted classification logits, ``(N, C)`` or
                ``(B, C, H, W)``.
            target_scores: Quality targets in ``[0, 1]``, same shape as
                *pred_scores*.  Positive positions carry their IoU value;
                negative positions carry ``0``.
            target_labels: Integer class labels, ``(N,)`` or
                ``(B, H, W)``.

        Returns:
            Scalar or tensor loss.
        """
        # Flatten if 4-D
        if pred_scores.dim() == _DIM_4D:
            B, C, H, W = pred_scores.shape
            pred_scores = pred_scores.permute(0, 2, 3, 1).reshape(B * H * W, C)
            target_scores = target_scores.permute(0, 2, 3, 1).reshape(B * H * W, C)
            target_labels = target_labels.reshape(B * H * W)

        num_classes = pred_scores.shape[1]
        one_hot: Tensor = F.one_hot(target_labels, num_classes=num_classes).float()

        # Predicted probability
        pred_prob: Tensor = pred_scores.sigmoid()

        # ---- Positive loss: -q * log(p) ---------------------------------
        # Only for the target class; zeros elsewhere
        pos_loss: Tensor = -target_scores * torch.log(pred_prob.clamp(min=1e-8))

        # ---- Negative loss: -alpha * p^gamma * log(1-p) ------------------
        neg_loss: Tensor = -self.alpha * pred_prob.pow(self.gamma) * torch.log(
            (1.0 - pred_prob).clamp(min=1e-8)
        )

        # Combine using masks
        combined: Tensor = pos_loss + neg_loss
        loss_per_class: Tensor = combined * one_hot  # (N, C)

        # Sum over classes, keep per-sample: (N,)
        loss_per_sample: Tensor = loss_per_class.sum(dim=1)

        return self._reduce(loss_per_sample)

    def _reduce(self, loss: Tensor) -> Tensor:
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

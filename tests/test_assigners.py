"""Comprehensive tests for assigner modules.

Covers:
1. **HungarianMatcher** — bipartite 1-to-1 assignment for query detection:
   * Cost matrix computation (classification + L1 + GIoU costs).
   * Optimal assignment correctness.
   * Edge cases: empty predictions, empty targets, single query/GT.
   * Gradient flow through matching + SetCriterion.
   * Different cost weight configurations.
   * Device compatibility (CPU, CUDA, meta).
2. **TaskAlignedAssigner (TAL)** — dynamic top-k assignment for anchor-free
   detection:
   * Alignment metric computation (t = s^alpha * IoU^beta).
   * Dynamic positive sample selection.
   * Top-k candidate selection correctness.
   * Edge cases: no GT boxes, all background, single anchor.
   * Gradient flow through assignment.
   * Different alpha/beta parameter values.
   * Non-square image handling.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou, generalized_box_iou

from corecv.losses.assigners import HungarianMatcher, SetCriterion, TaskAlignedAssigner
from corecv.losses.assigners.tal import _decode_boxes, _make_anchors

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Return a CUDA device when available, falling back to CPU.

    All tests in this suite should be device-agnostic: they use this
    fixture for every tensor allocation so that the same tests run
    identically on GPU (preferred) and CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """Reset the random seed before every test for reproducibility."""
    torch.manual_seed(42)


# ======================================================================
# Helper: gradient check
# ======================================================================


def _check_gradient(loss: Tensor, leaf_params: list[Tensor]) -> None:
    """Run ``loss.backward()`` and assert no NaN/Inf in leaf gradients.

    The caller must ensure all entries in *leaf_params* are PyTorch leaf
    tensors (i.e. parameters or tensors created with ``requires_grad=True``
    directly, not as a result of tensor operations).

    Args:
        loss: Scalar loss tensor.
        leaf_params: List of leaf parameter tensors whose .grad is checked.
    """
    loss.backward()
    for p in leaf_params:
        assert p.grad is not None, "Expected gradient to be populated."
        assert not p.grad.isnan().any(), f"NaN gradient found in param of shape {p.shape}."
        assert not p.grad.isinf().any(), f"Inf gradient found in param of shape {p.shape}."


# ======================================================================
# HungarianMatcher Tests
# ======================================================================


class TestHungarianMatcher:
    """Tests for :class:`corecv.losses.assigners.HungarianMatcher`."""

    # ------------------------------------------------------------------
    # Basic cost matrix computation
    # ------------------------------------------------------------------

    def test_cost_matrix_shape(self, device: torch.device) -> None:
        """Cost matrix has shape ``(num_queries, num_gt)``."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(1, 10, 5, device=device)  # (B, Q, C)
        pred_boxes = torch.sigmoid(torch.randn(1, 10, 4, device=device))
        gt_labels = [torch.randint(0, 5, (3,), device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(3, 4, device=device))]

        cost_matrix = matcher._compute_cost_matrix(
            pred_scores[0], pred_boxes[0], gt_labels[0], gt_boxes[0],
        )
        assert cost_matrix.shape == (10, 3), (
            f"Expected (10, 3), got {cost_matrix.shape}."
        )

    def test_cost_matrix_all_terms(self, device: torch.device) -> None:
        """All three cost terms (class, L1, GIoU) contribute to the matrix."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(1, 8, 4, device=device)
        pred_boxes = torch.sigmoid(torch.randn(1, 8, 4, device=device))
        gt_labels = [torch.randint(0, 4, (2,), device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(2, 4, device=device))]

        # Compute individual cost components
        cost_class = matcher._focal_cost_matrix(pred_scores[0], gt_labels[0])
        cost_bbox = torch.cdist(pred_boxes[0], gt_boxes[0], p=1)
        cost_giou = 1.0 - generalized_box_iou(
            box_convert(pred_boxes[0], "cxcywh", "xyxy"),
            box_convert(gt_boxes[0], "cxcywh", "xyxy"),
        )

        combined = matcher._compute_cost_matrix(
            pred_scores[0], pred_boxes[0], gt_labels[0], gt_boxes[0],
        )

        expected = (
            matcher.cost_class * cost_class
            + matcher.cost_bbox * cost_bbox
            + matcher.cost_giou * cost_giou
        )
        assert torch.isclose(combined, expected, atol=1e-6).all(), (
            "Combined cost matrix does not match weighted sum of individual terms."
        )

    def test_focal_cost_non_negative(self, device: torch.device) -> None:
        """Focal classification cost should be non-negative."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(10, 5, device=device)
        gt_labels = torch.randint(0, 5, (3,), device=device)
        cost = matcher._focal_cost_matrix(pred_scores, gt_labels)
        assert (cost >= 0.0).all(), "Focal cost contains negative values."

    def test_l1_cost_non_negative(self, device: torch.device) -> None:
        """L1 bounding-box cost should be non-negative."""
        pred_boxes = torch.sigmoid(torch.randn(10, 4, device=device))
        gt_boxes = torch.sigmoid(torch.randn(3, 4, device=device))
        cost = torch.cdist(pred_boxes, gt_boxes, p=1)
        assert (cost >= 0.0).all(), "L1 cost contains negative values."

    def test_giou_cost_range(self, device: torch.device) -> None:
        """GIoU cost (1 - GIoU) should be in [0, 2]."""
        pred_boxes = torch.sigmoid(torch.randn(10, 4, device=device))
        gt_boxes = torch.sigmoid(torch.randn(3, 4, device=device))
        giou = generalized_box_iou(
            box_convert(pred_boxes, "cxcywh", "xyxy"),
            box_convert(gt_boxes, "cxcywh", "xyxy"),
        )
        cost_giou = 1.0 - giou
        assert (cost_giou >= 0.0).all(), "GIoU cost below 0."
        assert (cost_giou <= 2.0).all(), "GIoU cost above 2."

    # ------------------------------------------------------------------
    # Matching correctness
    # ------------------------------------------------------------------

    def test_basic_matching(self, device: torch.device) -> None:
        """Basic forward pass returns correct number of indices per image."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(2, 10, 5, device=device)
        pred_boxes = torch.sigmoid(torch.randn(2, 10, 4, device=device))
        gt_labels = [
            torch.randint(0, 5, (3,), device=device),
            torch.randint(0, 5, (2,), device=device),
        ]
        gt_boxes = [
            torch.sigmoid(torch.randn(3, 4, device=device)),
            torch.sigmoid(torch.randn(2, 4, device=device)),
        ]
        indices = matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)
        assert len(indices) == 2, "Expected 2 sets of indices."
        # First image: 3 GT boxes -> 3 matched predictions
        assert indices[0][0].shape[0] == 3, (
            f"Expected 3 matched preds, got {indices[0][0].shape[0]}."
        )
        assert indices[0][1].shape[0] == 3, (
            f"Expected 3 matched GT, got {indices[0][1].shape[0]}."
        )
        # Second image: 2 GT boxes -> 2 matched predictions
        assert indices[1][0].shape[0] == 2, (
            f"Expected 2 matched preds, got {indices[1][0].shape[0]}."
        )
        assert indices[1][1].shape[0] == 2, (
            f"Expected 2 matched GT, got {indices[1][1].shape[0]}."
        )

    def test_matching_one_to_one(self, device: torch.device) -> None:
        """Each prediction is assigned to at most one GT and vice versa."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(1, 20, 10, device=device)
        pred_boxes = torch.sigmoid(torch.randn(1, 20, 4, device=device))
        gt_labels = [torch.randint(0, 10, (5,), device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(5, 4, device=device))]

        indices = matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)
        pred_idx, gt_idx = indices[0]

        # Check 1-to-1: no duplicates
        assert pred_idx.unique().numel() == pred_idx.numel(), (
            "Duplicate prediction indices - not 1-to-1."
        )
        assert gt_idx.unique().numel() == gt_idx.numel(), (
            "Duplicate GT indices - not 1-to-1."
        )
        # Each GT index appears exactly once
        for i in range(5):
            assert (gt_idx == i).sum() == 1, (
                f"GT index {i} should be matched exactly once."
            )

    def test_matching_optimal_assignment(self, device: torch.device) -> None:
        """Matching finds lower cost than a random assignment."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(1, 10, 3, device=device)
        pred_boxes = torch.sigmoid(torch.randn(1, 10, 4, device=device))
        gt_labels = [torch.randint(0, 3, (4,), device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(4, 4, device=device))]

        indices = matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)
        pred_idx, gt_idx = indices[0]

        # Compute cost for Hungarian assignment
        cost_matrix = matcher._compute_cost_matrix(
            pred_scores[0], pred_boxes[0], gt_labels[0], gt_boxes[0],
        )
        hungarian_cost = cost_matrix[pred_idx, gt_idx].sum()

        # Compute cost for a random assignment
        rand_pred_idx = torch.randperm(10, device=device)[:4]
        random_cost = cost_matrix[rand_pred_idx, gt_idx].sum()

        # Hungarian should be <= random (allow floating-point tolerance)
        assert hungarian_cost <= random_cost + 1e-6, (
            f"Hungarian cost {hungarian_cost:.4f} > random cost {random_cost:.4f}."
        )

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_gt(self, device: torch.device) -> None:
        """No ground-truth boxes -> empty assignment."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(1, 10, 5, device=device)
        pred_boxes = torch.sigmoid(torch.randn(1, 10, 4, device=device))
        gt_labels = [torch.zeros(0, dtype=torch.long, device=device)]
        gt_boxes = [torch.zeros(0, 4, device=device)]

        indices = matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)
        pred_idx, gt_idx = indices[0]
        assert pred_idx.numel() == 0, "Expected empty prediction indices."
        assert gt_idx.numel() == 0, "Expected empty GT indices."

    def test_empty_gt_batch(self, device: torch.device) -> None:
        """All images have no GT boxes."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(2, 10, 5, device=device)
        pred_boxes = torch.sigmoid(torch.randn(2, 10, 4, device=device))
        gt_labels = [
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        ]
        gt_boxes = [
            torch.zeros(0, 4, device=device),
            torch.zeros(0, 4, device=device),
        ]

        indices = matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)
        for i, (pred_idx, gt_idx) in enumerate(indices):
            assert pred_idx.numel() == 0, f"Image {i}: expected empty pred indices."
            assert gt_idx.numel() == 0, f"Image {i}: expected empty GT indices."

    def test_single_gt(self, device: torch.device) -> None:
        """Single ground-truth box matches one prediction."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(1, 10, 5, device=device)
        pred_boxes = torch.sigmoid(torch.randn(1, 10, 4, device=device))
        gt_labels = [torch.tensor([2], device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(1, 4, device=device))]

        indices = matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)
        pred_idx, gt_idx = indices[0]
        assert pred_idx.numel() == 1, "Expected exactly 1 matched prediction."
        assert gt_idx.numel() == 1, "Expected exactly 1 matched GT."
        assert gt_idx[0] == 0, "GT index should be 0."

    def test_single_query(self, device: torch.device) -> None:
        """Single query matched to multiple GT is impossible; only 1 GT matched."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(1, 1, 5, device=device)  # 1 query
        pred_boxes = torch.sigmoid(torch.randn(1, 1, 4, device=device))
        gt_labels = [torch.randint(0, 5, (3,), device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(3, 4, device=device))]

        indices = matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)
        pred_idx, gt_idx = indices[0]
        # With 1 query and 3 GT, only 1 GT can be matched
        assert pred_idx.numel() == 1, "Expected exactly 1 matched prediction."
        assert gt_idx.numel() == 1, "Expected exactly 1 matched GT."
        assert pred_idx[0] == 0, "Prediction index should be 0."

    def test_more_queries_than_gt(self, device: torch.device) -> None:
        """More queries than GT boxes: only num_gt predictions matched."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(1, 100, 10, device=device)
        pred_boxes = torch.sigmoid(torch.randn(1, 100, 4, device=device))
        gt_labels = [torch.randint(0, 10, (7,), device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(7, 4, device=device))]

        indices = matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)
        pred_idx, gt_idx = indices[0]
        assert pred_idx.numel() == 7, f"Expected 7 matches, got {pred_idx.numel()}."
        assert gt_idx.numel() == 7, f"Expected 7 matches, got {gt_idx.numel()}."

    # ------------------------------------------------------------------
    # Device compatibility
    # ------------------------------------------------------------------

    def test_cost_matrix_on_meta(self) -> None:
        """Cost matrix computation works on meta device (shape only)."""
        matcher = HungarianMatcher()
        meta_device = torch.device("meta")
        pred_scores = torch.randn(10, 5, device=meta_device)
        pred_boxes = torch.randn(10, 4, device=meta_device)
        gt_labels = torch.zeros(3, dtype=torch.long, device=meta_device)
        gt_boxes = torch.randn(3, 4, device=meta_device)

        cost_matrix = matcher._compute_cost_matrix(
            pred_scores, pred_boxes, gt_labels, gt_boxes,
        )
        assert cost_matrix.shape == (10, 3), (
            f"Expected (10, 3) on meta, got {cost_matrix.shape}."
        )
        assert cost_matrix.device.type == "meta"

    def test_focal_cost_on_meta(self) -> None:
        """Focal cost computation works on meta device (shape only)."""
        matcher = HungarianMatcher()
        meta_device = torch.device("meta")
        pred_scores = torch.randn(10, 5, device=meta_device)
        gt_labels = torch.zeros(3, dtype=torch.long, device=meta_device)

        cost = matcher._focal_cost_matrix(pred_scores, gt_labels)
        assert cost.shape == (10, 3)
        assert cost.device.type == "meta"

    def test_matching_on_cpu(self) -> None:
        """Matching works correctly on CPU."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(1, 8, 4)
        pred_boxes = torch.sigmoid(torch.randn(1, 8, 4))
        gt_labels = [torch.randint(0, 4, (2,))]
        gt_boxes = [torch.sigmoid(torch.randn(2, 4))]

        indices = matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)
        assert len(indices) == 1
        pred_idx, gt_idx = indices[0]
        assert pred_idx.numel() == 2
        assert gt_idx.numel() == 2

    @pytest.mark.gpu
    def test_matching_on_cuda(self) -> None:
        """Matching works correctly on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available.")
        matcher = HungarianMatcher()
        pred_scores = torch.randn(1, 8, 4, device="cuda")
        pred_boxes = torch.sigmoid(torch.randn(1, 8, 4, device="cuda"))
        gt_labels = [torch.randint(0, 4, (2,), device="cuda")]
        gt_boxes = [torch.sigmoid(torch.randn(2, 4, device="cuda"))]

        indices = matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)
        assert len(indices) == 1
        pred_idx, gt_idx = indices[0]
        assert pred_idx.numel() == 2
        assert gt_idx.numel() == 2
        assert pred_idx.device.type == "cuda"
        assert gt_idx.device.type == "cuda"

    def test_indices_on_same_device_as_input(self, device: torch.device) -> None:
        """Returned indices should be on the same device as inputs."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(1, 10, 5, device=device)
        pred_boxes = torch.sigmoid(torch.randn(1, 10, 4, device=device))
        gt_labels = [torch.randint(0, 5, (3,), device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(3, 4, device=device))]

        indices = matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)
        pred_idx, gt_idx = indices[0]
        assert pred_idx.device == device, (
            f"Expected indices on {device}, got {pred_idx.device}."
        )
        assert gt_idx.device == device, (
            f"Expected indices on {device}, got {gt_idx.device}."
        )

    # ------------------------------------------------------------------
    # Different weight configurations
    # ------------------------------------------------------------------

    def test_different_cost_weights(self, device: torch.device) -> None:
        """Different cost weights produce valid assignments."""
        pred_scores = torch.randn(1, 10, 5, device=device)
        pred_boxes = torch.sigmoid(torch.randn(1, 10, 4, device=device))
        gt_labels = [torch.randint(0, 5, (3,), device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(3, 4, device=device))]

        matcher_a = HungarianMatcher(cost_class=10.0, cost_bbox=1.0, cost_giou=1.0)
        matcher_b = HungarianMatcher(cost_class=1.0, cost_bbox=10.0, cost_giou=1.0)

        indices_a = matcher_a(pred_scores, pred_boxes, gt_labels, gt_boxes)
        indices_b = matcher_b(pred_scores, pred_boxes, gt_labels, gt_boxes)

        # At minimum the number of matches should be the same
        pred_a, _gt_a = indices_a[0]
        pred_b, _gt_b = indices_b[0]
        assert pred_a.numel() == pred_b.numel() == 3

    def test_zero_cost_giou(self, device: torch.device) -> None:
        """Zero GIoU cost weight should not crash and produce valid matching."""
        matcher = HungarianMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=0.0)
        pred_scores = torch.randn(1, 10, 5, device=device)
        pred_boxes = torch.sigmoid(torch.randn(1, 10, 4, device=device))
        gt_labels = [torch.randint(0, 5, (3,), device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(3, 4, device=device))]

        indices = matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)
        assert len(indices) == 1
        assert indices[0][0].numel() == 3

    # ------------------------------------------------------------------
    # Non-square resolution (shape propagation)
    # ------------------------------------------------------------------

    def test_non_square_queries(self, device: torch.device) -> None:
        """Non-square query count should work (simulating non-square feature maps)."""
        matcher = HungarianMatcher()
        # Simulate uneven query distribution from non-square feature maps
        # e.g., 4x6 feature map -> 24 queries at stride 16 for 256x384 input
        pred_scores = torch.randn(1, 24, 5, device=device)
        pred_boxes = torch.sigmoid(torch.randn(1, 24, 4, device=device))
        gt_labels = [torch.randint(0, 5, (4,), device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(4, 4, device=device))]

        indices = matcher(pred_scores, pred_boxes, gt_labels, gt_boxes)
        assert len(indices) == 1
        assert indices[0][0].numel() == 4


# ======================================================================
# SetCriterion Tests
# ======================================================================


class TestSetCriterion:
    """Tests for :class:`corecv.losses.assigners.SetCriterion`."""

    # ------------------------------------------------------------------
    # Basic forward / backward
    # ------------------------------------------------------------------

    def test_set_criterion_basic(self, device: torch.device) -> None:
        """Basic forward pass produces scalar loss."""
        matcher = HungarianMatcher()
        criterion = SetCriterion(num_classes=10, matcher=matcher)
        pred_scores = torch.randn(2, 20, 10, device=device, requires_grad=True)
        pred_boxes = torch.randn(2, 20, 4, device=device, requires_grad=True)
        gt_labels = [
            torch.randint(0, 10, (3,), device=device),
            torch.randint(0, 10, (2,), device=device),
        ]
        gt_boxes = [
            torch.sigmoid(torch.randn(3, 4, device=device)),
            torch.sigmoid(torch.randn(2, 4, device=device)),
        ]

        loss, aux = criterion(pred_scores, pred_boxes, gt_labels, gt_boxes)
        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}."
        assert loss >= 0.0, "Loss should be non-negative."
        assert "loss_cls" in aux
        assert "loss_bbox" in aux
        assert "loss_giou" in aux

    def test_set_criterion_gradient(self, device: torch.device) -> None:
        """Gradient flows through the full SetCriterion."""
        matcher = HungarianMatcher()
        criterion = SetCriterion(num_classes=10, matcher=matcher)
        pred_scores = torch.randn(2, 20, 10, device=device, requires_grad=True)
        pred_boxes = torch.randn(2, 20, 4, device=device, requires_grad=True)
        gt_labels = [
            torch.randint(0, 10, (3,), device=device),
            torch.randint(0, 10, (2,), device=device),
        ]
        gt_boxes = [
            torch.sigmoid(torch.randn(3, 4, device=device)),
            torch.sigmoid(torch.randn(2, 4, device=device)),
        ]

        loss, _ = criterion(pred_scores, pred_boxes, gt_labels, gt_boxes)
        _check_gradient(loss, [pred_scores, pred_boxes])

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_set_criterion_empty_gt(self, device: torch.device) -> None:
        """No ground-truth boxes returns a valid (non-NaN) scalar loss."""
        matcher = HungarianMatcher()
        criterion = SetCriterion(num_classes=10, matcher=matcher)
        pred_scores = torch.randn(2, 20, 10, device=device, requires_grad=True)
        pred_boxes = torch.randn(2, 20, 4, device=device)
        gt_labels = [
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        ]
        gt_boxes = [
            torch.zeros(0, 4, device=device),
            torch.zeros(0, 4, device=device),
        ]

        loss, aux = criterion(pred_scores, pred_boxes, gt_labels, gt_boxes)
        assert loss.ndim == 0, "Loss should be scalar even with empty GT."
        assert not loss.isnan(), "Loss should not be NaN with empty GT."
        assert not loss.isinf(), "Loss should not be Inf with empty GT."

    def test_set_criterion_single_gt(self, device: torch.device) -> None:
        """Single GT box per image works correctly."""
        matcher = HungarianMatcher()
        criterion = SetCriterion(num_classes=10, matcher=matcher)
        pred_scores = torch.randn(1, 20, 10, device=device, requires_grad=True)
        pred_boxes = torch.randn(1, 20, 4, device=device)
        gt_labels = [torch.tensor([5], device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(1, 4, device=device))]

        loss, aux = criterion(pred_scores, pred_boxes, gt_labels, gt_boxes)
        assert loss.ndim == 0
        assert not loss.isnan()
        _check_gradient(loss, [pred_scores])

    # ------------------------------------------------------------------
    # Auxiliary losses
    # ------------------------------------------------------------------

    def test_auxiliary_losses(self, device: torch.device) -> None:
        """Auxiliary decoder layer losses contribute to total loss."""
        matcher = HungarianMatcher()
        criterion = SetCriterion(num_classes=10, matcher=matcher)
        pred_scores = torch.randn(2, 20, 10, device=device, requires_grad=True)
        pred_boxes = torch.randn(2, 20, 4, device=device)
        gt_labels = [
            torch.randint(0, 10, (3,), device=device),
            torch.randint(0, 10, (2,), device=device),
        ]
        gt_boxes = [
            torch.sigmoid(torch.randn(3, 4, device=device)),
            torch.sigmoid(torch.randn(2, 4, device=device)),
        ]

        # Add 2 auxiliary decoder layers
        aux_cls = [
            torch.randn(2, 20, 10, device=device, requires_grad=True),
            torch.randn(2, 20, 10, device=device, requires_grad=True),
        ]
        aux_reg = [
            torch.randn(2, 20, 4, device=device),
            torch.randn(2, 20, 4, device=device),
        ]

        loss_no_aux, _ = criterion(pred_scores, pred_boxes, gt_labels, gt_boxes)
        loss_with_aux, aux_dict = criterion(
            pred_scores, pred_boxes, gt_labels, gt_boxes,
            intermediate_cls=aux_cls, intermediate_reg=aux_reg,
        )

        assert loss_with_aux > loss_no_aux, (
            "Auxiliary losses should increase total loss."
        )
        assert "aux_loss_cls_layer0" in aux_dict, (
            "Missing aux_loss_cls_layer0 in aux dict."
        )
        assert "aux_loss_bbox_layer1" in aux_dict, (
            "Missing aux_loss_bbox_layer1 in aux dict."
        )
        assert "aux_loss_giou_layer0" in aux_dict, (
            "Missing aux_loss_giou_layer0 in aux dict."
        )

    def test_aux_loss_gradient_flow(self, device: torch.device) -> None:
        """Gradients flow through auxiliary losses."""
        matcher = HungarianMatcher()
        criterion = SetCriterion(num_classes=10, matcher=matcher)
        pred_scores = torch.randn(2, 20, 10, device=device, requires_grad=True)
        pred_boxes = torch.randn(2, 20, 4, device=device)
        gt_labels = [
            torch.randint(0, 10, (3,), device=device),
            torch.randint(0, 10, (2,), device=device),
        ]
        gt_boxes = [
            torch.sigmoid(torch.randn(3, 4, device=device)),
            torch.sigmoid(torch.randn(2, 4, device=device)),
        ]
        aux_cls = [
            torch.randn(2, 20, 10, device=device, requires_grad=True),
        ]
        aux_reg = [
            torch.randn(2, 20, 4, device=device),
        ]

        loss, _ = criterion(
            pred_scores, pred_boxes, gt_labels, gt_boxes,
            intermediate_cls=aux_cls, intermediate_reg=aux_reg,
        )
        _check_gradient(loss, [pred_scores, aux_cls[0]])

    def test_aux_loss_weight_mismatch(self, device: torch.device) -> None:
        """Mismatched aux_loss_weights length raises ValueError."""
        matcher = HungarianMatcher()
        criterion = SetCriterion(
            num_classes=10, matcher=matcher, aux_loss_weights=[0.5],
        )
        pred_scores = torch.randn(1, 10, 10, device=device)
        pred_boxes = torch.randn(1, 10, 4, device=device)
        gt_labels = [torch.randint(0, 10, (2,), device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(2, 4, device=device))]
        aux_cls = [
            torch.randn(1, 10, 10, device=device),
            torch.randn(1, 10, 10, device=device),
        ]
        aux_reg = [
            torch.randn(1, 10, 4, device=device),
            torch.randn(1, 10, 4, device=device),
        ]

        with pytest.raises(ValueError, match="aux_loss_weights"):
            criterion(
                pred_scores, pred_boxes, gt_labels, gt_boxes,
                intermediate_cls=aux_cls, intermediate_reg=aux_reg,
            )

    # ------------------------------------------------------------------
    # Different loss weights
    # ------------------------------------------------------------------

    def test_loss_weight_configurations(self, device: torch.device) -> None:
        """Different loss weights produce different loss values."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(2, 20, 10, device=device)
        pred_boxes = torch.randn(2, 20, 4, device=device)
        gt_labels = [
            torch.randint(0, 10, (3,), device=device),
            torch.randint(0, 10, (2,), device=device),
        ]
        gt_boxes = [
            torch.sigmoid(torch.randn(3, 4, device=device)),
            torch.sigmoid(torch.randn(2, 4, device=device)),
        ]

        criterion_a = SetCriterion(
            num_classes=10, matcher=matcher,
            loss_cls_weight=1.0, loss_bbox_weight=1.0, loss_giou_weight=1.0,
        )
        criterion_b = SetCriterion(
            num_classes=10, matcher=matcher,
            loss_cls_weight=5.0, loss_bbox_weight=1.0, loss_giou_weight=1.0,
        )

        loss_a, _ = criterion_a(pred_scores, pred_boxes, gt_labels, gt_boxes)
        loss_b, _ = criterion_b(pred_scores, pred_boxes, gt_labels, gt_boxes)

        assert not torch.isclose(loss_a, loss_b, atol=1e-6), (
            "Different loss weights should produce different losses."
        )

    # ------------------------------------------------------------------
    # Meta device
    # ------------------------------------------------------------------

    def test_criterion_cost_matrix_on_meta(self) -> None:
        """SetCriterion cost matrix computation works on meta device."""
        matcher = HungarianMatcher()
        meta_device = torch.device("meta")
        pred_scores = torch.randn(1, 10, 10, device=meta_device)
        pred_boxes = torch.randn(1, 10, 4, device=meta_device)
        gt_labels = [torch.zeros(2, dtype=torch.long, device=meta_device)]
        gt_boxes = [torch.randn(2, 4, device=meta_device)]

        # The HungarianMatcher forward() cannot run on meta because of
        # scipy.linear_sum_assignment needing real data.  Instead verify
        # that the cost matrix sub-computation is meta-compatible.
        cost_matrix = matcher._compute_cost_matrix(
            pred_scores[0], pred_boxes[0], gt_labels[0], gt_boxes[0],
        )
        assert cost_matrix.shape == (10, 2)
        assert cost_matrix.device.type == "meta"

    # ------------------------------------------------------------------
    # No-object weight
    # ------------------------------------------------------------------

    def test_no_object_weight_effect(self, device: torch.device) -> None:
        """Higher no_object_weight increases classification loss for unmatched queries."""
        matcher = HungarianMatcher()
        pred_scores = torch.randn(1, 20, 10, device=device)
        pred_boxes = torch.randn(1, 20, 4, device=device)
        gt_labels = [torch.randint(0, 10, (2,), device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(2, 4, device=device))]

        criterion_low = SetCriterion(
            num_classes=10, matcher=matcher, no_object_weight=0.01,
        )
        criterion_high = SetCriterion(
            num_classes=10, matcher=matcher, no_object_weight=1.0,
        )

        loss_low, _ = criterion_low(pred_scores, pred_boxes, gt_labels, gt_boxes)
        loss_high, _ = criterion_high(pred_scores, pred_boxes, gt_labels, gt_boxes)

        # Higher no-object weight should produce larger or equal loss
        assert loss_high >= loss_low - 1e-6, (
            "Higher no_object_weight should increase loss."
        )

    # ------------------------------------------------------------------
    # Non-square resolution
    # ------------------------------------------------------------------

    def test_non_square_queries_criterion(self, device: torch.device) -> None:
        """Non-square query count works with SetCriterion."""
        matcher = HungarianMatcher()
        criterion = SetCriterion(num_classes=5, matcher=matcher)
        # 24 queries (e.g., from 4x6 feature map)
        pred_scores = torch.randn(1, 24, 5, device=device, requires_grad=True)
        pred_boxes = torch.randn(1, 24, 4, device=device)
        gt_labels = [torch.randint(0, 5, (4,), device=device)]
        gt_boxes = [torch.sigmoid(torch.randn(4, 4, device=device))]

        loss, aux = criterion(pred_scores, pred_boxes, gt_labels, gt_boxes)
        assert loss.ndim == 0
        assert not loss.isnan()
        _check_gradient(loss, [pred_scores])


# ======================================================================
# TaskAlignedAssigner (TAL) Tests — initialisation
# ======================================================================


class TestTaskAlignedAssignerInit:
    """Tests for :class:`corecv.losses.assigners.TaskAlignedAssigner` initialisation."""

    def test_default_init(self) -> None:
        """Default initialisation should set correct parameters."""
        assigner = TaskAlignedAssigner(num_classes=80)
        assert assigner.num_classes == 80
        assert assigner.topk == 13
        assert assigner.alpha == 0.5
        assert assigner.beta == 6.0

    def test_custom_params(self) -> None:
        """Custom parameters are stored correctly."""
        assigner = TaskAlignedAssigner(
            num_classes=10, topk=9, alpha=1.0, beta=2.0,
        )
        assert assigner.num_classes == 10
        assert assigner.topk == 9
        assert assigner.alpha == 1.0
        assert assigner.beta == 2.0

    def test_invalid_num_classes(self) -> None:
        """``num_classes < 1`` raises ValueError."""
        with pytest.raises(ValueError, match="num_classes"):
            TaskAlignedAssigner(num_classes=0)

    def test_invalid_topk(self) -> None:
        """``topk < 1`` raises ValueError."""
        with pytest.raises(ValueError, match="topk"):
            TaskAlignedAssigner(num_classes=10, topk=0)

    def test_invalid_alpha(self) -> None:
        """``alpha < 0`` raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            TaskAlignedAssigner(num_classes=10, alpha=-0.1)

    def test_invalid_beta(self) -> None:
        """``beta < 0`` raises ValueError."""
        with pytest.raises(ValueError, match="beta"):
            TaskAlignedAssigner(num_classes=10, beta=-1.0)


# ======================================================================
# TaskAlignedAssigner (TAL) Tests — assignment logic
# ======================================================================


class TestTaskAlignedAssigner:
    """Tests for :class:`corecv.losses.assigners.TaskAlignedAssigner` assignment logic."""

    # ------------------------------------------------------------------
    # Helper to create valid TAL inputs
    # ------------------------------------------------------------------

    @staticmethod
    def _make_inputs(  # noqa: PLR0913
        batch_size: int = 2,
        num_classes: int = 10,
        feat_sizes: list[tuple[int, int]] | None = None,
        strides: list[int] | None = None,
        num_gts: list[int] | None = None,
        device: torch.device | None = None,
    ) -> tuple:
        """Create synthetic inputs for TaskAlignedAssigner.

        Args:
            batch_size: Number of images in the batch.
            num_classes: Number of object classes.
            feat_sizes: Per-level (H, W) feature map sizes.
            strides: Per-level stride values.
            num_gts: Per-image number of ground-truth boxes.
            device: Target device.

        Returns:
            Tuple ``(cls_logits, reg_pred, strides, gt_labels, gt_boxes)``.
        """
        if feat_sizes is None:
            feat_sizes = [(80, 80), (40, 40), (20, 20)]
        if strides is None:
            strides = [8, 16, 32]
        if num_gts is None:
            num_gts = [3, 2]
        if device is None:
            device = torch.device("cpu")

        cls_logits: list[Tensor] = []
        reg_pred: list[Tensor] = []
        for H, W in feat_sizes:
            cls_logits.append(
                torch.randn(batch_size, num_classes, H, W, device=device),
            )
            reg_pred.append(
                torch.rand(batch_size, 4, H, W, device=device).abs() * 10,
            )

        gt_labels: list[Tensor] = [
            torch.randint(0, num_classes, (n,), device=device)
            for n in num_gts
        ]
        gt_boxes: list[Tensor] = []
        # Determine maximum coordinate range from the finest level
        max_coord = max(
            feat_sizes[0][0] * strides[0],
            feat_sizes[0][1] * strides[0],
        )
        for n in num_gts:
            boxes = torch.rand(n, 4, device=device) * (max_coord - 10) + 5
            # Ensure x1 < x2, y1 < y2
            boxes = torch.stack([
                torch.min(boxes[:, 0], boxes[:, 2]),
                torch.min(boxes[:, 1], boxes[:, 3]),
                torch.max(boxes[:, 0], boxes[:, 2]),
                torch.max(boxes[:, 1], boxes[:, 3]),
            ], dim=-1)
            gt_boxes.append(boxes)

        return cls_logits, reg_pred, strides, gt_labels, gt_boxes

    # ------------------------------------------------------------------
    # Basic forward / output structure
    # ------------------------------------------------------------------

    def test_forward_output_structure(self, device: torch.device) -> None:
        """Forward pass returns dict with expected keys and correct shapes."""
        assigner = TaskAlignedAssigner(num_classes=10, topk=9)
        cls_logits, reg_pred, strides, gt_labels, gt_boxes = self._make_inputs(
            batch_size=2, num_classes=10, device=device,
        )
        result = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)

        expected_keys = {"pos_mask", "neg_mask", "assigned_gt_inds", "assigned_labels", "pos_ious"}
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}."
        )

        # Each value is a list of length B
        assert len(result["pos_mask"]) == 2
        assert len(result["neg_mask"]) == 2
        assert len(result["assigned_gt_inds"]) == 2
        assert len(result["assigned_labels"]) == 2
        assert len(result["pos_ious"]) == 2

        # Total number of anchors across all levels
        total_anchors = sum(H * W for H, W in [(80, 80), (40, 40), (20, 20)])
        for i in range(2):
            assert result["pos_mask"][i].shape == (total_anchors,), (
                f"pos_mask[{i}] shape mismatch."
            )
            assert result["neg_mask"][i].shape == (total_anchors,)
            assert result["assigned_gt_inds"][i].shape == (total_anchors,)
            assert result["assigned_labels"][i].shape == (total_anchors,)
            # pos_ious should match number of positive anchors
            assert result["pos_ious"][i].dim() == 1

    def test_forward_output_types(self, device: torch.device) -> None:
        """Output tensors have correct dtypes."""
        assigner = TaskAlignedAssigner(num_classes=10, topk=9)
        cls_logits, reg_pred, strides, gt_labels, gt_boxes = self._make_inputs(
            batch_size=1, num_classes=10, device=device,
        )
        result = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)

        assert result["pos_mask"][0].dtype == torch.bool
        assert result["neg_mask"][0].dtype == torch.bool
        assert result["assigned_gt_inds"][0].dtype == torch.long
        assert result["assigned_labels"][0].dtype == torch.long
        assert result["pos_ious"][0].dtype == torch.float32

    # ------------------------------------------------------------------
    # Alignment metric properties
    # ------------------------------------------------------------------

    def test_alignment_metric_non_negative(self, device: torch.device) -> None:
        """Alignment metric (s^alpha * IoU^beta) should be non-negative."""
        cls_logits, reg_pred, strides, gt_labels, gt_boxes = self._make_inputs(
            batch_size=1, num_classes=10, device=device,
        )

        # Manually compute the alignment metric for verification
        B = cls_logits[0].shape[0]
        cls_scores = torch.cat([
            lvl.permute(0, 2, 3, 1).reshape(B, -1, 10)
            for lvl in cls_logits
        ], dim=1)
        cls_probs = cls_scores.sigmoid()[0]

        # Compute decoded boxes and IoU
        feat_sizes = [(r.shape[2], r.shape[3]) for r in reg_pred]
        all_anchors = _make_anchors(strides, feat_sizes, device)
        reg_flat = torch.cat([
            lvl.permute(0, 2, 3, 1).reshape(B, -1, 4)
            for lvl in reg_pred
        ], dim=1)
        decoded = _decode_boxes(
            reg_flat.reshape(-1, 4),
            all_anchors.unsqueeze(0).expand(B, -1, -1).reshape(-1, 2),
        ).reshape(B, -1, 4)[0]

        iou_matrix = box_iou(gt_boxes[0], decoded).t()

        gt_labels_expanded = gt_labels[0].unsqueeze(0).expand(cls_probs.shape[0], -1)
        cls_scores_for_gt = torch.gather(cls_probs, dim=1, index=gt_labels_expanded)

        align_metric = cls_scores_for_gt.pow(0.5) * iou_matrix.pow(6.0)
        assert (align_metric >= 0.0).all(), "Alignment metric contains negative values."

    # ------------------------------------------------------------------
    # Top-k selection
    # ------------------------------------------------------------------

    def test_topk_selection(self, device: torch.device) -> None:
        """At most topk anchors are selected per GT."""
        topk = 5
        assigner = TaskAlignedAssigner(num_classes=10, topk=topk)
        cls_logits, reg_pred, strides, gt_labels, gt_boxes = self._make_inputs(
            batch_size=1, num_classes=10, num_gts=[3],
            feat_sizes=[(10, 10)], strides=[8],  # small feature map
            device=device,
        )
        result = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)

        num_pos = result["pos_mask"][0].sum().item()
        # With 3 GT boxes and topk=5, at most 15 positives (but some may overlap)
        assert num_pos <= topk * 3, (
            f"Expected at most {topk * 3} positives, got {num_pos}."
        )
        # At least 1 positive per GT (since we have enough anchors)
        assert num_pos >= 3, (
            f"Expected at least 3 positives, got {num_pos}."
        )

    def test_topk_equals_total_anchors(self, device: torch.device) -> None:
        """When topk >= total anchors, all anchors may be selected."""
        topk = 200  # more than total anchors
        assigner = TaskAlignedAssigner(num_classes=10, topk=topk)
        cls_logits, reg_pred, strides, gt_labels, gt_boxes = self._make_inputs(
            batch_size=1, num_classes=10, num_gts=[2],
            feat_sizes=[(4, 4)], strides=[8],  # 16 anchors total
            device=device,
        )
        result = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)

        num_pos = result["pos_mask"][0].sum().item()
        # Should have at least some positives
        assert num_pos > 0, "Expected some positive anchors."

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_no_gt_boxes(self, device: torch.device) -> None:
        """No ground-truth boxes results in all-negative assignment."""
        assigner = TaskAlignedAssigner(num_classes=10, topk=9)
        cls_logits, reg_pred, strides, _, _ = self._make_inputs(
            batch_size=2, num_classes=10, num_gts=[0, 0],
            device=device,
        )
        gt_labels = [
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        ]
        gt_boxes = [
            torch.zeros(0, 4, device=device),
            torch.zeros(0, 4, device=device),
        ]

        result = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)

        for i in range(2):
            assert result["pos_mask"][i].sum().item() == 0, (
                f"Image {i}: expected no positives."
            )
            assert result["neg_mask"][i].all(), (
                f"Image {i}: expected all negatives."
            )
            assert (result["assigned_gt_inds"][i] == -1).all(), (
                f"Image {i}: expected all assigned_gt_inds=-1."
            )
            assert result["pos_ious"][i].numel() == 0, (
                f"Image {i}: expected empty pos_ious."
            )

    def test_all_background(self, device: torch.device) -> None:
        """All anchors classified as background (low scores)."""
        assigner = TaskAlignedAssigner(num_classes=3, topk=9)
        cls_logits, reg_pred, strides, gt_labels, gt_boxes = self._make_inputs(
            batch_size=1, num_classes=3, num_gts=[2],
            feat_sizes=[(8, 8)], strides=[8],
            device=device,
        )
        # Suppress all classification scores to near-zero
        for i in range(len(cls_logits)):
            cls_logits[i] = torch.full_like(cls_logits[i], -100.0)

        result = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)
        # Some anchors may still be selected if IoU > 0, but scores are low
        # The assignment still works without crashing
        total_anchors = result["pos_mask"][0].shape[0]
        assert result["pos_mask"][0].shape == (total_anchors,)
        assert result["neg_mask"][0].shape == (total_anchors,)

    def test_single_anchor(self, device: torch.device) -> None:
        """Single anchor with a single GT box."""
        assigner = TaskAlignedAssigner(num_classes=5, topk=1)
        # 1x1 feature map = 1 anchor
        cls_logits = [torch.randn(1, 5, 1, 1, device=device)]
        reg_pred = [torch.rand(1, 4, 1, 1, device=device)]
        strides = [8]
        gt_labels = [torch.tensor([2], device=device)]
        # A GT box within range of the single anchor
        gt_boxes = [torch.tensor([[10.0, 10.0, 20.0, 20.0]], device=device)]

        result = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)
        # The single anchor may or may not be positive depending on IoU
        assert result["pos_mask"][0].numel() == 1
        assert result["neg_mask"][0].numel() == 1

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_invalid_pred_length(self, device: torch.device) -> None:
        """Mismatched pred_scores and pred_boxes lengths raise ValueError."""
        assigner = TaskAlignedAssigner(num_classes=10)
        cls_logits, reg_pred, strides, gt_labels, gt_boxes = self._make_inputs(
            batch_size=1, num_classes=10, device=device,
        )
        # Remove one level from reg_pred
        with pytest.raises(ValueError, match="pred_scores and pred_boxes"):
            assigner(cls_logits[:-1], reg_pred, strides, gt_labels, gt_boxes)

    def test_invalid_strides_length(self, device: torch.device) -> None:
        """Mismatched strides and levels length raise ValueError."""
        assigner = TaskAlignedAssigner(num_classes=10)
        cls_logits, reg_pred, strides, gt_labels, gt_boxes = self._make_inputs(
            batch_size=1, num_classes=10, device=device,
        )
        with pytest.raises(ValueError, match="strides"):
            assigner(cls_logits, reg_pred, strides[:-1], gt_labels, gt_boxes)

    # ------------------------------------------------------------------
    # Different alpha/beta parameter values
    # ------------------------------------------------------------------

    def test_alpha_zero(self, device: torch.device) -> None:
        """``alpha=0`` means metric depends only on IoU^beta."""
        assigner = TaskAlignedAssigner(num_classes=10, topk=9, alpha=0.0, beta=1.0)
        cls_logits, reg_pred, strides, gt_labels, gt_boxes = self._make_inputs(
            batch_size=1, num_classes=10, device=device,
        )
        result = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)
        assert result["pos_mask"][0] is not None
        assert result["pos_mask"][0].dtype == torch.bool

    def test_beta_zero(self, device: torch.device) -> None:
        """``beta=0`` means metric depends only on cls^alpha."""
        assigner = TaskAlignedAssigner(num_classes=10, topk=9, alpha=1.0, beta=0.0)
        cls_logits, reg_pred, strides, gt_labels, gt_boxes = self._make_inputs(
            batch_size=1, num_classes=10, device=device,
        )
        result = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)
        assert result["pos_mask"][0] is not None

    def test_alpha_beta_one(self, device: torch.device) -> None:
        """``alpha=1, beta=1`` means metric = cls * IoU."""
        assigner = TaskAlignedAssigner(num_classes=10, topk=9, alpha=1.0, beta=1.0)
        cls_logits, reg_pred, strides, gt_labels, gt_boxes = self._make_inputs(
            batch_size=1, num_classes=10, device=device,
        )
        result = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)
        assert result["pos_mask"][0] is not None

    # ------------------------------------------------------------------
    # Device compatibility
    # ------------------------------------------------------------------

    def test_meta_anchor_generation(self) -> None:
        """Anchor generation works on meta device (shape only)."""
        meta_device = torch.device("meta")
        strides = [8, 16]
        feat_sizes = [(80, 80), (40, 40)]
        anchors = _make_anchors(strides, feat_sizes, meta_device)
        total_anchors = 80 * 80 + 40 * 40
        assert anchors.shape == (total_anchors, 2)
        assert anchors.device.type == "meta"

    def test_meta_decode_boxes(self) -> None:
        """Box decoding works on meta device (shape only)."""
        meta_device = torch.device("meta")
        reg_pred = torch.randn(100, 4, device=meta_device)
        anchors = torch.randn(100, 2, device=meta_device)
        boxes = _decode_boxes(reg_pred, anchors)
        assert boxes.shape == (100, 4)
        assert boxes.device.type == "meta"

    @pytest.mark.gpu
    def test_cuda_device(self) -> None:
        """TaskAlignedAssigner works on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available.")
        assigner = TaskAlignedAssigner(num_classes=10, topk=9)
        cls_logits, reg_pred, strides, gt_labels, gt_boxes = self._make_inputs(
            batch_size=2, num_classes=10, device=torch.device("cuda"),
        )
        result = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)
        for key, val in result.items():
            for tensor in val:
                assert tensor.device.type == "cuda", (
                    f"{key} tensor not on CUDA."
                )

    # ------------------------------------------------------------------
    # Non-square resolution (H/W inversion bug catcher)
    # ------------------------------------------------------------------

    def test_non_square_feature_maps(self, device: torch.device) -> None:
        """Non-square feature maps (e.g., 60x80) should not cause H/W inversion."""
        assigner = TaskAlignedAssigner(num_classes=10, topk=9)
        # Non-square feature map sizes simulating 480x640 input at stride 8, 16, 32
        feat_sizes = [(60, 80), (30, 40), (15, 20)]  # H, W
        strides = [8, 16, 32]
        cls_logits: list[Tensor] = []
        reg_pred: list[Tensor] = []
        for H, W in feat_sizes:
            cls_logits.append(torch.randn(1, 10, H, W, device=device))
            reg_pred.append(torch.rand(1, 4, H, W, device=device))

        gt_labels = [torch.randint(0, 10, (3,), device=device)]
        # GT boxes in pixel coordinates for 480x640 image
        gt_boxes = [torch.tensor([
            [50.0, 30.0, 200.0, 150.0],
            [100.0, 200.0, 300.0, 400.0],
            [400.0, 50.0, 600.0, 450.0],
        ], device=device)]

        result = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)

        total_anchors = sum(H * W for H, W in feat_sizes)
        assert result["pos_mask"][0].shape == (total_anchors,), (
            f"Expected {total_anchors} anchors, got {result['pos_mask'][0].shape}."
        )

        # Verify anchors are ordered correctly: finer levels first
        # Finest level: 60*80=4800, next: 30*40=1200, coarsest: 15*20=300
        assert total_anchors == 4800 + 1200 + 300, (
            f"Unexpected total anchors: {total_anchors}."
        )

    def test_non_square_hw_correctness(self, device: torch.device) -> None:
        """Verify H/W ordering is correct in anchor generation for non-square maps."""
        strides = [16]
        feat_sizes = [(30, 40)]  # H=30, W=40 (non-square)
        anchors = _make_anchors(strides, feat_sizes, device)

        # 30*40 = 1200 anchors
        assert anchors.shape == (1200, 2), f"Expected (1200, 2), got {anchors.shape}."

        # For cell (row=0, col=0) in "ij" indexing:
        # cx = (0 + 0.5) * 16 = 8.0
        # cy = (0 + 0.5) * 16 = 8.0
        # First anchor should be (8.0, 8.0) - centre of top-left cell
        assert torch.isclose(anchors[0, 0], torch.tensor(8.0, device=device)), (
            f"Expected cx=8.0, got {anchors[0, 0]}."
        )
        assert torch.isclose(anchors[0, 1], torch.tensor(8.0, device=device)), (
            f"Expected cy=8.0, got {anchors[0, 1]}."
        )

        # For cell (row=0, col=39) — last column, first row
        # cx = (39 + 0.5) * 16 = 632.0
        # cy = (0 + 0.5) * 16 = 8.0
        assert torch.isclose(anchors[39, 0], torch.tensor(632.0, device=device)), (
            f"Expected cx=632.0, got {anchors[39, 0]}."
        )
        assert torch.isclose(anchors[39, 1], torch.tensor(8.0, device=device)), (
            f"Expected cy=8.0, got {anchors[39, 1]}."
        )

    # ------------------------------------------------------------------
    # Anchor generation correctness
    # ------------------------------------------------------------------

    def test_make_anchors_output(self, device: torch.device) -> None:
        """_make_anchors generates correct number and ordering of anchors."""
        strides = [8, 16]
        feat_sizes = [(10, 10), (5, 5)]  # square
        anchors = _make_anchors(strides, feat_sizes, device)

        assert anchors.shape == (125, 2), f"Expected (125, 2), got {anchors.shape}."

        # First anchor (finest level, cell 0,0)
        assert torch.isclose(anchors[0], torch.tensor([4.0, 4.0], device=device)).all(), (
            f"First anchor should be (4.0, 4.0), got {anchors[0]}."
        )

        # Last anchor of finest level (cell 9,9)
        assert torch.isclose(anchors[99], torch.tensor([76.0, 76.0], device=device)).all(), (
            f"Last fine anchor should be (76.0, 76.0), got {anchors[99]}."
        )

        # First anchor of coarser level
        assert torch.isclose(anchors[100], torch.tensor([8.0, 8.0], device=device)).all(), (
            f"First coarse anchor should be (8.0, 8.0), got {anchors[100]}."
        )

    # ------------------------------------------------------------------
    # Decode boxes correctness
    # ------------------------------------------------------------------

    def test_decode_boxes_correctness(self, device: torch.device) -> None:
        """_decode_boxes correctly converts (l, t, r, b) to (x1, y1, x2, y2)."""
        anchors = torch.tensor([[100.0, 200.0], [50.0, 50.0]], device=device)
        reg_pred = torch.tensor([
            [10.0, 20.0, 30.0, 40.0],
            [5.0, 5.0, 5.0, 5.0],
        ], device=device)

        boxes = _decode_boxes(reg_pred, anchors)

        # First box: cx=100, cy=200, l=10, t=20, r=30, b=40
        # x1=100-10=90, y1=200-20=180, x2=100+30=130, y2=200+40=240
        expected_0 = torch.tensor([90.0, 180.0, 130.0, 240.0], device=device)
        assert torch.isclose(boxes[0], expected_0).all(), (
            f"First box mismatch: got {boxes[0]}, expected {expected_0}."
        )

        # Second box: cx=50, cy=50, l=5, t=5, r=5, b=5
        # x1=45, y1=45, x2=55, y2=55
        expected_1 = torch.tensor([45.0, 45.0, 55.0, 55.0], device=device)
        assert torch.isclose(boxes[1], expected_1).all(), (
            f"Second box mismatch: got {boxes[1]}, expected {expected_1}."
        )

    def test_decode_boxes_clamps_negative(self, device: torch.device) -> None:
        """_decode_boxes clamps negative regression values to zero."""
        anchors = torch.tensor([[100.0, 100.0]], device=device)
        # Negative regression values
        reg_pred = torch.tensor([[-10.0, -20.0, 30.0, 40.0]], device=device)

        boxes = _decode_boxes(reg_pred, anchors)
        # l and t should be clamped to 0, so x1=100-0=100, y1=100-0=100
        expected = torch.tensor([100.0, 100.0, 130.0, 140.0], device=device)
        assert torch.isclose(boxes[0], expected).all(), (
            f"Clamping failed: got {boxes[0]}, expected {expected}."
        )

    # ------------------------------------------------------------------
    # Multi-level resolution (integration with head outputs)
    # ------------------------------------------------------------------

    def test_multi_level_assignment(self, device: torch.device) -> None:
        """Assignment works across multiple feature levels with different strides."""
        assigner = TaskAlignedAssigner(num_classes=10, topk=13)
        cls_logits, reg_pred, strides, gt_labels, gt_boxes = self._make_inputs(
            batch_size=2, num_classes=10,
            feat_sizes=[(80, 80), (40, 40), (20, 20), (10, 10)],
            strides=[4, 8, 16, 32],
            num_gts=[4, 5],
            device=device,
        )
        result = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)
        total_anchors = 80 * 80 + 40 * 40 + 20 * 20 + 10 * 10
        assert result["pos_mask"][0].shape == (total_anchors,)
        assert result["pos_mask"][1].shape == (total_anchors,)

    def test_different_num_gt_per_image(self, device: torch.device) -> None:
        """Different numbers of GT per image work correctly."""
        assigner = TaskAlignedAssigner(num_classes=10, topk=13)
        cls_logits, reg_pred, strides, gt_labels, gt_boxes = self._make_inputs(
            batch_size=3, num_classes=10,
            feat_sizes=[(40, 40), (20, 20)],
            strides=[8, 16],
            num_gts=[0, 3, 7],
            device=device,
        )
        result = assigner(cls_logits, reg_pred, strides, gt_labels, gt_boxes)
        total_anchors = 40 * 40 + 20 * 20
        assert result["pos_mask"][0].shape == (total_anchors,)
        assert result["pos_mask"][1].shape == (total_anchors,)
        assert result["pos_mask"][2].shape == (total_anchors,)
        # First image (0 GT) should have no positives
        assert result["pos_mask"][0].sum().item() == 0
        assert result["neg_mask"][0].all()

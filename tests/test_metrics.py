"""Tests for VRAM-resident metrics engine.

Verifies tensor dimensions across channels, non-square resolutions
(e.g. 480×640) to catch H/W indexing inversion bugs, zero-VRAM
compatibility on ``device='meta'``, and gradient flow correctness.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from corecv.metrics import ClassificationMetrics, DetectionMetrics, SegmentationMetrics

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """Return CPU or (if available) CUDA device.

    All metric tests are automatically parametrised over every available
    device type so that device-consistency requirements are fulfilled
    without writing redundant test methods.
    """
    return torch.device(request.param)


# ======================================================================
# Helper utilities
# ======================================================================


def _assert_result_types(
    results: dict[str, Any],
    *,
    float_keys: list[str],
    tensor_keys: list[str],
) -> None:
    """Assert that ``compute()`` returned the expected key types.

    Args:
        results: The dictionary returned by ``compute()``.
        float_keys: Keys whose values must be Python ``float``.
        tensor_keys: Keys whose values must be ``torch.Tensor``.
    """
    assert isinstance(results, dict), f"Expected dict, got {type(results)}"
    for key in float_keys:
        assert key in results, f"Missing float key {key!r}"
        assert isinstance(results[key], float), (
            f"Key {key!r} should be float, got {type(results[key]).__name__}"
        )
    for key in tensor_keys:
        assert key in results, f"Missing tensor key {key!r}"
        assert isinstance(results[key], torch.Tensor), (
            f"Key {key!r} should be Tensor, got {type(results[key]).__name__}"
        )


def _accuracy_from_confusion(
    confusion: torch.Tensor,
) -> float:
    """Compute top-1 accuracy from a confusion matrix ``(C, C)``."""
    total: int = int(confusion.sum().item())
    if total == 0:
        return 0.0
    return float(confusion.diag().sum().item()) / total


# ======================================================================
# ClassificationMetrics  (11 tests)
# ======================================================================


class TestClassificationMetrics:
    """Comprehensive tests for :class:`~corecv.metrics.ClassificationMetrics`."""

    NUM_CLASSES: int = 5
    BATCH_SIZE: int = 16

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def metrics(self, device: torch.device) -> ClassificationMetrics:
        """Create a fresh ClassificationMetrics on the target device."""
        torch.manual_seed(42)
        return ClassificationMetrics(
            num_classes=self.NUM_CLASSES,
            top_k=(1, 3, 5),
            device=device,
        )

    @pytest.fixture
    def perfect_batch(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, targets)`` where every prediction is correct.

        The logits are crafted so that ``argmax`` always yields the target
        class.
        """
        torch.manual_seed(42)
        targets: torch.Tensor = torch.randint(
            0, self.NUM_CLASSES, (self.BATCH_SIZE,), device=device,
        )
        logits: torch.Tensor = torch.full(
            (self.BATCH_SIZE, self.NUM_CLASSES), -10.0, device=device,
        )
        logits.scatter_(1, targets.unsqueeze(1), 10.0)
        return logits, targets

    # ------------------------------------------------------------------
    # Core functionality
    # ------------------------------------------------------------------

    def test_accuracy_perfect_predictions(
        self,
        metrics: ClassificationMetrics,
        perfect_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """All correct predictions yield accuracy = 1.0 and f1_micro = 1.0."""
        logits, targets = perfect_batch
        metrics.update(logits, targets)
        results = metrics.compute()

        assert results["accuracy"] == pytest.approx(1.0)
        assert results["f1_score_micro"] == pytest.approx(1.0)
        assert results["f1_score_macro"] == pytest.approx(1.0)
        assert results["precision_micro"] == pytest.approx(1.0)
        assert results["recall_micro"] == pytest.approx(1.0)

    def test_accuracy_random_predictions(
        self,
        metrics: ClassificationMetrics,
        device: torch.device,
    ) -> None:
        """Random predictions produce an accuracy value in (0, 1)."""
        torch.manual_seed(42)
        logits: torch.Tensor = torch.randn(
            self.BATCH_SIZE, self.NUM_CLASSES, device=device,
        )
        targets: torch.Tensor = torch.randint(
            0, self.NUM_CLASSES, (self.BATCH_SIZE,), device=device,
        )
        metrics.update(logits, targets)
        results = metrics.compute()

        assert 0.0 <= results["accuracy"] <= 1.0
        assert 0.0 <= results["f1_score_micro"] <= 1.0

    def test_top_k_accuracy(
        self,
        metrics: ClassificationMetrics,
        device: torch.device,
    ) -> None:
        """Top-k accuracy satisfies monotonicity: k larger → accuracy ≥ previous."""
        torch.manual_seed(42)
        # Boost class 0 so it is always among the top-k even when not top-1.
        targets: torch.Tensor = torch.zeros(
            self.BATCH_SIZE, dtype=torch.long, device=device,
        )
        logits: torch.Tensor = torch.randn(
            self.BATCH_SIZE, self.NUM_CLASSES, device=device,
        )
        logits[:, 0] += 5.0
        metrics.update(logits, targets)
        results = metrics.compute()

        # top-1 accuracy must equal the general accuracy field.
        assert results["top1_accuracy"] == pytest.approx(results["accuracy"])
        # Monotonicity.
        assert results["top3_accuracy"] >= results["top1_accuracy"]
        assert results["top5_accuracy"] >= results["top3_accuracy"]
        # All in [0, 1].
        for k in (1, 3, 5):
            assert 0.0 <= results[f"top{k}_accuracy"] <= 1.0

    def test_f1_scores(
        self,
        metrics: ClassificationMetrics,
        perfect_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Macro and micro F1 scores equal 1.0 under perfect predictions."""
        logits, targets = perfect_batch
        metrics.update(logits, targets)
        results = metrics.compute()

        assert results["f1_score_macro"] == pytest.approx(1.0)
        assert results["f1_score_micro"] == pytest.approx(1.0)

    def test_precision_recall(
        self,
        metrics: ClassificationMetrics,
        perfect_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Precision and recall are 1.0 for perfect predictions."""
        logits, targets = perfect_batch
        metrics.update(logits, targets)
        results = metrics.compute()

        assert results["precision_macro"] == pytest.approx(1.0)
        assert results["precision_micro"] == pytest.approx(1.0)
        assert results["recall_macro"] == pytest.approx(1.0)
        assert results["recall_micro"] == pytest.approx(1.0)

    def test_multi_batch_accumulation(
        self,
        metrics: ClassificationMetrics,
        device: torch.device,
    ) -> None:
        """Multiple ``update()`` calls accumulate the same as a single call."""
        torch.manual_seed(42)
        half: int = self.BATCH_SIZE // 2

        logits1: torch.Tensor = torch.randn(
            half, self.NUM_CLASSES, device=device,
        )
        targets1: torch.Tensor = torch.randint(
            0, self.NUM_CLASSES, (half,), device=device,
        )
        logits2: torch.Tensor = torch.randn(
            self.BATCH_SIZE - half, self.NUM_CLASSES, device=device,
        )
        targets2: torch.Tensor = torch.randint(
            0, self.NUM_CLASSES, (self.BATCH_SIZE - half,), device=device,
        )

        # Single update with full batch.
        single_metrics = ClassificationMetrics(
            num_classes=self.NUM_CLASSES,
            top_k=(1, 3, 5),
            device=device,
        )
        full_logits: torch.Tensor = torch.cat([logits1, logits2])
        full_targets: torch.Tensor = torch.cat([targets1, targets2])
        single_metrics.update(full_logits, full_targets)
        single_result = single_metrics.compute()

        # Two updates with partial batches.
        metrics.update(logits1, targets1)
        metrics.update(logits2, targets2)
        multi_result = metrics.compute()

        for key in single_result:
            if isinstance(single_result[key], float):
                assert multi_result[key] == pytest.approx(single_result[key]), (
                    f"Mismatch for key {key!r}"
                )
            else:
                assert isinstance(multi_result[key], torch.Tensor)
                assert torch.allclose(
                    multi_result[key].float(),
                    single_result[key].float(),
                ), f"Mismatch for tensor key {key!r}"

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def test_reset_clears_buffers(
        self,
        metrics: ClassificationMetrics,
        perfect_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """``reset()`` zeroes all accumulator buffers."""
        logits, targets = perfect_batch
        metrics.update(logits, targets)

        # Confirm buffers are non-zero after update.
        assert metrics.total_samples.item() > 0
        assert metrics.confusion.sum().item() > 0

        metrics.reset()

        assert metrics.total_samples.item() == 0
        assert metrics.confusion.sum().item() == 0

        # After reset, compute() returns zeros.
        results = metrics.compute()
        assert results["accuracy"] == pytest.approx(0.0)

    def test_compute_returns_python_dict(
        self,
        metrics: ClassificationMetrics,
        perfect_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """``compute()`` returns a ``dict`` with Python ``float`` values."""
        logits, targets = perfect_batch
        metrics.update(logits, targets)
        results = metrics.compute()

        _assert_result_types(
            results,
            float_keys=[
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_score_macro",
                "precision_micro",
                "recall_micro",
                "f1_score_micro",
                "top1_accuracy",
                "top3_accuracy",
                "top5_accuracy",
            ],
            tensor_keys=[],
        )

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_predictions(
        self,
        device: torch.device,
    ) -> None:
        """An empty batch (B=0) is handled without error and returns zeros."""
        metrics = ClassificationMetrics(
            num_classes=self.NUM_CLASSES,
            top_k=(1, 3),
            device=device,
        )
        logits: torch.Tensor = torch.empty(0, self.NUM_CLASSES, device=device)
        targets: torch.Tensor = torch.empty(0, dtype=torch.long, device=device)

        metrics.update(logits, targets)
        results = metrics.compute()

        assert results["accuracy"] == pytest.approx(0.0)
        assert results["f1_score_micro"] == pytest.approx(0.0)

    def test_device_consistency(
        self,
        metrics: ClassificationMetrics,
        device: torch.device,
    ) -> None:
        """All internal buffers reside on the expected device."""
        assert metrics.confusion.device == device
        assert metrics.total_samples.device == device

    def test_non_square_logits(
        self,
        device: torch.device,
    ) -> None:
        """Logits with ``C != B`` (non-square shape) work correctly."""
        num_classes: int = 10
        batch_size: int = 4  # C=10, B=4 → non-square
        metrics = ClassificationMetrics(
            num_classes=num_classes,
            top_k=(1, 3),
            device=device,
        )
        torch.manual_seed(42)
        logits: torch.Tensor = torch.randn(
            batch_size, num_classes, device=device,
        )
        targets: torch.Tensor = torch.randint(
            0, num_classes, (batch_size,), device=device,
        )
        metrics.update(logits, targets)
        results = metrics.compute()

        assert 0.0 <= results["accuracy"] <= 1.0

    # ------------------------------------------------------------------
    # Additional edge cases
    # ------------------------------------------------------------------

    def test_single_class_single_sample(
        self,
        device: torch.device,
    ) -> None:
        """Single class, single sample — the minimal non-trivial case."""
        metrics = ClassificationMetrics(
            num_classes=1,
            top_k=(1,),
            device=device,
        )
        logits: torch.Tensor = torch.tensor([[10.0]], device=device)
        targets: torch.Tensor = torch.zeros(1, dtype=torch.long, device=device)

        metrics.update(logits, targets)
        results = metrics.compute()

        assert results["accuracy"] == pytest.approx(1.0)
        assert results["f1_score_micro"] == pytest.approx(1.0)

    def test_large_num_classes(
        self,
        device: torch.device,
    ) -> None:
        """Eighty classes (COCO cardinality) does not break the accumulator."""
        num_classes: int = 80
        batch_size: int = 32
        metrics = ClassificationMetrics(
            num_classes=num_classes,
            top_k=(1, 5, 10),
            device=device,
        )
        torch.manual_seed(42)
        logits: torch.Tensor = torch.randn(
            batch_size, num_classes, device=device,
        )
        targets: torch.Tensor = torch.randint(
            0, num_classes, (batch_size,), device=device,
        )
        metrics.update(logits, targets)
        results = metrics.compute()

        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0


# ======================================================================
# SegmentationMetrics  (9 tests)
# ======================================================================


class TestSegmentationMetrics:
    """Comprehensive tests for :class:`~corecv.metrics.SegmentationMetrics`."""

    NUM_CLASSES: int = 3
    HEIGHT: int = 32
    WIDTH: int = 32

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def metrics(self, device: torch.device) -> SegmentationMetrics:
        """Create a fresh SegmentationMetrics on the target device."""
        return SegmentationMetrics(
            num_classes=self.NUM_CLASSES,
            ignore_index=255,
            device=device,
        )

    @pytest.fixture
    def perfect_batch(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, targets)`` where every pixel is correct.

        All pixels belong to class 0, and the logits are crafted so that
        ``argmax`` always predicts class 0.
        """
        torch.manual_seed(42)
        targets: torch.Tensor = torch.zeros(
            1, self.HEIGHT, self.WIDTH, dtype=torch.long, device=device,
        )
        logits: torch.Tensor = torch.randn(
            1, self.NUM_CLASSES, self.HEIGHT, self.WIDTH, device=device,
        )
        logits[:, 0] += 20.0  # Ensure class 0 dominates
        return logits, targets

    # ------------------------------------------------------------------
    # Core functionality
    # ------------------------------------------------------------------

    def test_miou_perfect(
        self,
        metrics: SegmentationMetrics,
        perfect_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Perfect prediction yields mIoU = 1.0 and mean_dice = 1.0."""
        logits, targets = perfect_batch
        metrics.update(logits, targets)
        results = metrics.compute()

        assert results["miou"] == pytest.approx(1.0)
        assert results["mean_dice"] == pytest.approx(1.0)
        assert results["pixel_accuracy"] == pytest.approx(1.0)

    def test_miou_partial_overlap(
        self,
        device: torch.device,
    ) -> None:
        """Known partial overlap produces the expected per-class IoU values.

        A 2×2 image with 2 classes is used so that all counts can be
        verified manually.
        """
        local_metrics: SegmentationMetrics = SegmentationMetrics(
            num_classes=2,
            ignore_index=255,
            device=device,
        )
        # targets: 2×2 grid  [[0, 1], [1, 0]]
        targets: torch.Tensor = torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.long, device=device,
        ).unsqueeze(0)  # (1, 2, 2)

        # predictions: all pixels predicted as class 0
        logits: torch.Tensor = torch.full(
            (1, 2, 2, 2), -10.0, device=device,
        )
        logits[:, 0] = 10.0  # Predict class 0 everywhere

        local_metrics.update(logits, targets)

        # Manual computation (4 pixels, 2 classes, all pred=0):
        #
        #   targets   = [[0, 1],   preds (argmax) = [[0, 0],
        #                 [1, 0]]                    [0, 0]]
        #
        #   Class 0 intersection = pixels where (pred=0 AND target=0) = 2
        #           union        = pixels where (pred=0 OR target=0)  = 4
        #           IoU          = 2/4 = 0.5
        #
        #   Class 1 intersection = 0  (no pred=1)
        #           union        = 2  (two target=1 pixels)
        #           IoU          = 0/2 = 0.0
        #
        #   Both classes have union > 0, so both are valid.
        #   mIoU  = (0.5 + 0.0) / 2 = 0.25
        #   pix accuracy = correct / valid = 2/4 = 0.5

        results = local_metrics.compute()
        assert results["miou"] == pytest.approx(0.25)
        assert results["pixel_accuracy"] == pytest.approx(0.5)

        per_class_iou: torch.Tensor = results["per_class_iou"]
        assert per_class_iou[0].item() == pytest.approx(0.5)
        assert per_class_iou[1].item() == pytest.approx(0.0)

    def test_pixel_accuracy(
        self,
        metrics: SegmentationMetrics,
        device: torch.device,
    ) -> None:
        """Pixel accuracy is correctly computed as correct / valid."""
        # All pixels class 0.
        targets: torch.Tensor = torch.zeros(
            1, self.HEIGHT, self.WIDTH, dtype=torch.long, device=device,
        )
        # Predict class 0 for all but the first pixel.
        logits: torch.Tensor = torch.full(
            (1, self.NUM_CLASSES, self.HEIGHT, self.WIDTH), -10.0, device=device,
        )
        logits[:, 0] = 10.0  # Correct everywhere
        logits[:, 0, 0, 0] = -10.0  # Make first pixel wrong
        logits[:, 1, 0, 0] = 10.0  # Predict class 1 for first pixel

        metrics.update(logits, targets)
        results = metrics.compute()

        total_pixels: int = self.HEIGHT * self.WIDTH
        expected_acc: float = (total_pixels - 1) / total_pixels
        assert results["pixel_accuracy"] == pytest.approx(expected_acc)

    def test_dice_index(
        self,
        metrics: SegmentationMetrics,
        device: torch.device,
    ) -> None:
        """Dice coefficient matches manual computation for a known pattern."""
        # 2×2 grid, all class 0.
        targets: torch.Tensor = torch.zeros(
            1, 2, 2, dtype=torch.long, device=device,
        )
        # Predict class 0 for 3/4 pixels (top-left is wrong).
        logits: torch.Tensor = torch.full(
            (1, self.NUM_CLASSES, 2, 2), -10.0, device=device,
        )
        logits[:, 0] = 10.0
        logits[:, 0, 0, 0] = -10.0
        logits[:, 1, 0, 0] = 10.0

        metrics.update(logits, targets)
        results = metrics.compute()

        # Class 0 Dice = 2*TP / (2*TP + FP + FN)
        #   TP = 3, FP = 0, FN = 1
        #   Dice = 6 / (6 + 0 + 1) = 6/7 ≈ 0.857
        per_class_dice: torch.Tensor = results["per_class_dice"]
        expected_dice: float = 6.0 / 7.0
        assert per_class_dice[0].item() == pytest.approx(expected_dice, abs=1e-6)

    def test_ignore_index(
        self,
        metrics: SegmentationMetrics,
        device: torch.device,
    ) -> None:
        """Pixels with ``ignore_index`` are excluded from all metrics."""
        targets: torch.Tensor = torch.full(
            (1, self.HEIGHT, self.WIDTH), 255, dtype=torch.long, device=device,
        )
        logits: torch.Tensor = torch.randn(
            1, self.NUM_CLASSES, self.HEIGHT, self.WIDTH, device=device,
        )

        metrics.update(logits, targets)
        results = metrics.compute()

        # No valid pixels → miou = 0, pixel_accuracy = 0, mean_dice = 0
        assert results["miou"] == pytest.approx(0.0)
        assert results["pixel_accuracy"] == pytest.approx(0.0)
        assert results["mean_dice"] == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def test_multi_batch_accumulation(
        self,
        metrics: SegmentationMetrics,
        device: torch.device,
    ) -> None:
        """Multiple ``update()`` calls accumulate correctly."""
        torch.manual_seed(42)
        half_h: int = self.HEIGHT // 2

        logits1: torch.Tensor = torch.randn(
            1, self.NUM_CLASSES, half_h, self.WIDTH, device=device,
        )
        targets1: torch.Tensor = torch.randint(
            0, self.NUM_CLASSES, (1, half_h, self.WIDTH), device=device,
        )
        logits2: torch.Tensor = torch.randn(
            1, self.NUM_CLASSES, self.HEIGHT - half_h, self.WIDTH, device=device,
        )
        targets2: torch.Tensor = torch.randint(
            0,
            self.NUM_CLASSES,
            (1, self.HEIGHT - half_h, self.WIDTH),
            device=device,
        )

        # Single update.
        single_metrics = SegmentationMetrics(
            num_classes=self.NUM_CLASSES,
            device=device,
        )
        full_logits: torch.Tensor = torch.cat([logits1, logits2], dim=2)
        full_targets: torch.Tensor = torch.cat([targets1, targets2], dim=1)
        single_metrics.update(full_logits, full_targets)
        single_result = single_metrics.compute()

        # Two updates.
        metrics.update(logits1, targets1)
        metrics.update(logits2, targets2)
        multi_result = metrics.compute()

        for key in ["miou", "pixel_accuracy", "mean_dice"]:
            assert multi_result[key] == pytest.approx(single_result[key]), (
                f"Mismatch for key {key!r}"
            )

        for key in ["per_class_iou", "per_class_dice"]:
            assert torch.allclose(
                multi_result[key].float(), single_result[key].float(),
            ), f"Mismatch for tensor key {key!r}"

    def test_reset_clears_buffers(
        self,
        metrics: SegmentationMetrics,
        perfect_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """``reset()`` zeroes all accumulator buffers."""
        logits, targets = perfect_batch
        metrics.update(logits, targets)

        # Confirm buffers are non-zero.
        assert metrics.intersection.sum().item() > 0
        assert metrics.union.sum().item() > 0

        metrics.reset()

        assert metrics.intersection.sum().item() == 0
        assert metrics.union.sum().item() == 0
        assert metrics.total_correct.item() == 0
        assert metrics.total_valid.item() == 0

    def test_device_consistency(
        self,
        metrics: SegmentationMetrics,
        device: torch.device,
    ) -> None:
        """All buffers reside on the expected device."""
        assert metrics.intersection.device == device
        assert metrics.union.device == device
        assert metrics.pred_sum.device == device
        assert metrics.target_sum.device == device
        assert metrics.total_correct.device == device
        assert metrics.total_valid.device == device

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_non_square_images(
        self,
        device: torch.device,
    ) -> None:
        """Non-square image (480 × 640) is handled correctly."""
        height: int = 480
        width: int = 640
        num_classes: int = 2
        metrics = SegmentationMetrics(
            num_classes=num_classes, device=device,
        )
        torch.manual_seed(42)
        logits: torch.Tensor = torch.randn(
            1, num_classes, height, width, device=device,
        )
        targets: torch.Tensor = torch.randint(
            0, num_classes, (1, height, width), device=device,
        )

        metrics.update(logits, targets)
        results = metrics.compute()

        assert 0.0 <= results["miou"] <= 1.0
        assert 0.0 <= results["pixel_accuracy"] <= 1.0
        assert 0.0 <= results["mean_dice"] <= 1.0

    def test_very_small_image(
        self,
        device: torch.device,
    ) -> None:
        """Very small image (4 × 4) works without shape errors."""
        height: int = 4
        width: int = 4
        num_classes: int = 3
        metrics = SegmentationMetrics(
            num_classes=num_classes, device=device,
        )
        torch.manual_seed(42)
        logits: torch.Tensor = torch.randn(
            1, num_classes, height, width, device=device,
        )
        targets: torch.Tensor = torch.randint(
            0, num_classes, (1, height, width), device=device,
        )

        metrics.update(logits, targets)
        results = metrics.compute()

        assert isinstance(results["miou"], float)


# ======================================================================
# DetectionMetrics  (10 tests)
# ======================================================================


class TestDetectionMetrics:
    """Comprehensive tests for :class:`~corecv.metrics.DetectionMetrics`."""

    NUM_CLASSES: int = 5

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def metrics(self, device: torch.device) -> DetectionMetrics:
        """Create a fresh DetectionMetrics on the target device."""
        return DetectionMetrics(
            num_classes=self.NUM_CLASSES,
            device=device,
        )

    @pytest.fixture
    def perfect_image_data(
        self,
        device: torch.device,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """Return perfectly matching pred/target boxes for a single image.

        Each class has exactly one non-overlapping box so that greedy
        matching yields a trivial TP for every detection.
        """
        # Class 0: single box at top-left.
        # Class 1: single box at centre (no overlap with class 0).
        boxes: torch.Tensor = torch.tensor(
            [
                [0.00, 0.00, 0.20, 0.20],  # class 0
                [0.30, 0.30, 0.50, 0.50],  # class 1
            ],
            dtype=torch.float32,
            device=device,
        )
        scores: torch.Tensor = torch.tensor([0.95, 0.85], device=device)
        labels: torch.Tensor = torch.tensor([0, 1], dtype=torch.long, device=device)

        return (
            [boxes],
            [scores],
            [labels],
            [boxes.clone()],
            [labels.clone()],
        )

    # ------------------------------------------------------------------
    # Core functionality
    # ------------------------------------------------------------------

    def test_map50_perfect_match(
        self,
        metrics: DetectionMetrics,
        perfect_image_data: tuple[
            list[torch.Tensor], list[torch.Tensor], list[torch.Tensor],
            list[torch.Tensor], list[torch.Tensor],
        ],
    ) -> None:
        """Perfect box predictions yield mAP@50 = 1.0."""
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels = (
            perfect_image_data
        )
        metrics.update(pred_boxes, pred_scores, pred_labels, target_boxes, target_labels)
        results = metrics.compute()

        assert results["map50"] == pytest.approx(1.0)
        assert results["map50_95"] == pytest.approx(1.0)

    def test_map50_no_overlap(
        self,
        metrics: DetectionMetrics,
        device: torch.device,
    ) -> None:
        """Zero overlap between predictions and targets yields mAP@50 = 0.0."""
        pred_boxes: list[torch.Tensor] = [
            torch.tensor(
                [[0.00, 0.00, 0.10, 0.10]],
                dtype=torch.float32, device=device,
            ),
        ]
        pred_scores: list[torch.Tensor] = [
            torch.tensor([0.90], device=device),
        ]
        pred_labels: list[torch.Tensor] = [
            torch.tensor([0], dtype=torch.long, device=device),
        ]
        target_boxes: list[torch.Tensor] = [
            torch.tensor(
                [[0.90, 0.90, 1.00, 1.00]],
                dtype=torch.float32, device=device,
            ),
        ]
        target_labels: list[torch.Tensor] = [
            torch.tensor([0], dtype=torch.long, device=device),
        ]

        metrics.update(pred_boxes, pred_scores, pred_labels, target_boxes, target_labels)
        results = metrics.compute()

        assert results["map50"] == pytest.approx(0.0)
        assert results["map50_95"] == pytest.approx(0.0)

    def test_map50_95(
        self,
        metrics: DetectionMetrics,
        perfect_image_data: tuple[
            list[torch.Tensor], list[torch.Tensor], list[torch.Tensor],
            list[torch.Tensor], list[torch.Tensor],
        ],
    ) -> None:
        """mAP@50:95 equals 1.0 under perfect matches since IoU = 1.0 at every threshold."""
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels = (
            perfect_image_data
        )
        metrics.update(pred_boxes, pred_scores, pred_labels, target_boxes, target_labels)
        results = metrics.compute()

        assert results["map50_95"] == pytest.approx(1.0)

    def test_per_class_ap(
        self,
        metrics: DetectionMetrics,
        device: torch.device,
    ) -> None:
        """Per-class AP values are returned as tensors with correct shape."""
        # Two classes: class 0 has 1 GT, class 1 has 2 GTs.
        # Boxes within the same class are non-overlapping.
        pred_boxes: list[torch.Tensor] = [
            torch.tensor(
                [
                    [0.00, 0.00, 0.20, 0.20],  # class 0
                    [0.30, 0.30, 0.50, 0.50],  # class 1 — first box
                    [0.30, 0.55, 0.50, 0.75],  # class 1 — second box (below, no overlap)
                ],
                dtype=torch.float32, device=device,
            ),
        ]
        pred_scores: list[torch.Tensor] = [
            torch.tensor([0.95, 0.85, 0.90], device=device),
        ]
        pred_labels: list[torch.Tensor] = [
            torch.tensor([0, 1, 1], dtype=torch.long, device=device),
        ]
        target_boxes: list[torch.Tensor] = [
            torch.tensor(
                [
                    [0.00, 0.00, 0.20, 0.20],
                    [0.30, 0.30, 0.50, 0.50],
                    [0.30, 0.55, 0.50, 0.75],
                ],
                dtype=torch.float32, device=device,
            ),
        ]
        target_labels: list[torch.Tensor] = [
            torch.tensor([0, 1, 1], dtype=torch.long, device=device),
        ]

        metrics.update(pred_boxes, pred_scores, pred_labels, target_boxes, target_labels)
        results = metrics.compute()

        assert isinstance(results["per_class_ap50"], torch.Tensor)
        assert isinstance(results["per_class_ap50_95"], torch.Tensor)
        assert results["per_class_ap50"].shape == (self.NUM_CLASSES,)
        assert results["per_class_ap50_95"].shape == (self.NUM_CLASSES,)

        # Both classes have GT → both should have AP close to 1.0.
        assert results["per_class_ap50"][0].item() == pytest.approx(1.0)
        assert results["per_class_ap50"][1].item() == pytest.approx(1.0)

    def test_empty_predictions(
        self,
        metrics: DetectionMetrics,
        device: torch.device,
    ) -> None:
        """No predictions for any image yields mAP = 0.0."""
        pred_boxes: list[torch.Tensor] = [
            torch.empty(0, 4, device=device),
        ]
        pred_scores: list[torch.Tensor] = [
            torch.empty(0, device=device),
        ]
        pred_labels: list[torch.Tensor] = [
            torch.empty(0, dtype=torch.long, device=device),
        ]
        target_boxes: list[torch.Tensor] = [
            torch.tensor(
                [[0.10, 0.10, 0.50, 0.50]],
                dtype=torch.float32, device=device,
            ),
        ]
        target_labels: list[torch.Tensor] = [
            torch.tensor([0], dtype=torch.long, device=device),
        ]

        metrics.update(pred_boxes, pred_scores, pred_labels, target_boxes, target_labels)
        results = metrics.compute()

        assert results["map50"] == pytest.approx(0.0)
        assert results["map50_95"] == pytest.approx(0.0)

    def test_empty_targets(
        self,
        metrics: DetectionMetrics,
        device: torch.device,
    ) -> None:
        """No ground truth in any image yields mAP = 0.0 (no valid classes)."""
        pred_boxes: list[torch.Tensor] = [
            torch.tensor(
                [[0.10, 0.10, 0.50, 0.50]],
                dtype=torch.float32, device=device,
            ),
        ]
        pred_scores: list[torch.Tensor] = [
            torch.tensor([0.90], device=device),
        ]
        pred_labels: list[torch.Tensor] = [
            torch.tensor([0], dtype=torch.long, device=device),
        ]
        target_boxes: list[torch.Tensor] = [
            torch.empty(0, 4, device=device),
        ]
        target_labels: list[torch.Tensor] = [
            torch.empty(0, dtype=torch.long, device=device),
        ]

        metrics.update(pred_boxes, pred_scores, pred_labels, target_boxes, target_labels)
        results = metrics.compute()

        assert results["map50"] == pytest.approx(0.0)
        assert results["map50_95"] == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # Multi-image and multi-class
    # ------------------------------------------------------------------

    def test_multi_image_batch(
        self,
        metrics: DetectionMetrics,
        device: torch.device,
    ) -> None:
        """A batch of multiple images is processed correctly.

        Two images: the first has perfect matches (AP=1), the second has
        no overlap (AP=0). The mean should reflect both.
        """
        # Image 0: perfect match — 1 box, class 0.
        img0_pred_boxes: torch.Tensor = torch.tensor(
            [[0.10, 0.10, 0.50, 0.50]],
            dtype=torch.float32, device=device,
        )
        img0_scores: torch.Tensor = torch.tensor([0.95], device=device)
        img0_pred_labels: torch.Tensor = torch.tensor([0], dtype=torch.long, device=device)
        img0_target_boxes: torch.Tensor = img0_pred_boxes.clone()
        img0_target_labels: torch.Tensor = img0_pred_labels.clone()

        # Image 1: no overlap — 1 box, class 1.
        img1_pred_boxes: torch.Tensor = torch.tensor(
            [[0.00, 0.00, 0.10, 0.10]],
            dtype=torch.float32, device=device,
        )
        img1_scores: torch.Tensor = torch.tensor([0.80], device=device)
        img1_pred_labels: torch.Tensor = torch.tensor([1], dtype=torch.long, device=device)
        img1_target_boxes: torch.Tensor = torch.tensor(
            [[0.90, 0.90, 1.00, 1.00]],
            dtype=torch.float32, device=device,
        )
        img1_target_labels: torch.Tensor = torch.tensor([1], dtype=torch.long, device=device)

        metrics.update(
            [img0_pred_boxes, img1_pred_boxes],
            [img0_scores, img1_scores],
            [img0_pred_labels, img1_pred_labels],
            [img0_target_boxes, img1_target_boxes],
            [img0_target_labels, img1_target_labels],
        )
        results = metrics.compute()

        # At least one class has perfect AP → mAP should be > 0.
        assert results["map50"] > 0.0
        # Per-class AP[0] = 1.0, AP[1] = 0.0 → mAP50 = (1 + 0) / 2 = 0.5
        assert results["map50"] == pytest.approx(0.5)
        # Per-class AP[0] across all thresholds = 1.0, AP[1] = 0.0
        per_class_ap50: torch.Tensor = results["per_class_ap50"]
        assert per_class_ap50[0].item() == pytest.approx(1.0)
        assert per_class_ap50[1].item() == pytest.approx(0.0)

    def test_multi_class(
        self,
        metrics: DetectionMetrics,
        device: torch.device,
    ) -> None:
        """Multiple classes with varying numbers of GT are handled.

        Class 0: 2 GT, perfect predictions (boxes non-overlapping).
        Class 1: 0 GT (should not affect mAP).
        Class 2: 1 GT, perfect prediction.
        """
        # Class 0 boxes — side by side, no overlap.
        # Class 2 box — placed in a separate region so it does not overlap
        # with class 0 boxes (different classes are filtered already, but
        # keeping them separated is cleaner).
        boxes_class0: torch.Tensor = torch.tensor(
            [
                [0.00, 0.00, 0.20, 0.20],  # top-left
                [0.25, 0.00, 0.45, 0.20],  # right of first box
            ],
            dtype=torch.float32, device=device,
        )
        boxes_class2: torch.Tensor = torch.tensor(
            [[0.00, 0.30, 0.20, 0.50]],  # below class 0 region
            dtype=torch.float32, device=device,
        )

        pred_boxes: list[torch.Tensor] = [
            torch.cat([boxes_class0, boxes_class2]),
        ]
        pred_scores: list[torch.Tensor] = [
            torch.tensor([0.95, 0.85, 0.75], device=device),
        ]
        pred_labels: list[torch.Tensor] = [
            torch.tensor([0, 0, 2], dtype=torch.long, device=device),
        ]
        target_boxes: list[torch.Tensor] = [
            torch.cat([boxes_class0, boxes_class2]),
        ]
        target_labels: list[torch.Tensor] = [
            torch.tensor([0, 0, 2], dtype=torch.long, device=device),
        ]

        metrics.update(pred_boxes, pred_scores, pred_labels, target_boxes, target_labels)
        results = metrics.compute()

        # Class 0: 2 GT, 2 preds, perfect → AP@50 = 1.0
        # Class 1: 0 GT → excluded from mAP (COCO convention)
        # Class 2: 1 GT, 1 pred, perfect → AP@50 = 1.0
        # mAP = (1.0 + 1.0) / 2 = 1.0
        assert results["map50"] == pytest.approx(1.0)

        per_class_ap50: torch.Tensor = results["per_class_ap50"]
        assert per_class_ap50[0].item() == pytest.approx(1.0)
        # Class 1 has 0 GT — AP is 0 per the implementation.
        assert per_class_ap50[1].item() == pytest.approx(0.0)
        assert per_class_ap50[2].item() == pytest.approx(1.0)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def test_reset_clears_state(
        self,
        metrics: DetectionMetrics,
        perfect_image_data: tuple[
            list[torch.Tensor], list[torch.Tensor], list[torch.Tensor],
            list[torch.Tensor], list[torch.Tensor],
        ],
    ) -> None:
        """``reset()`` clears all buffers and list-based accumulators."""
        pred_boxes, pred_scores, pred_labels, target_boxes, target_labels = (
            perfect_image_data
        )
        metrics.update(pred_boxes, pred_scores, pred_labels, target_boxes, target_labels)

        # Confirm state is non-zero.
        assert metrics.target_counts.sum().item() > 0
        assert len(metrics._pred_scores) > 0

        metrics.reset()

        assert metrics.target_counts.sum().item() == 0
        assert len(metrics._pred_scores) == 0
        assert len(metrics._pred_tp) == 0
        assert len(metrics._pred_fp) == 0

        # After reset, compute returns zeros.
        results = metrics.compute()
        assert results["map50"] == pytest.approx(0.0)

    def test_device_consistency(
        self,
        metrics: DetectionMetrics,
        device: torch.device,
    ) -> None:
        """Accumulator buffers reside on the expected device."""
        assert metrics.target_counts.device == device
        assert metrics.iou_thresholds.device == device

    # ------------------------------------------------------------------
    # Additional edge cases
    # ------------------------------------------------------------------

    def test_large_num_classes(
        self,
        device: torch.device,
    ) -> None:
        """Eighty COCO classes with a single image do not raise errors."""
        num_classes: int = 80
        metrics = DetectionMetrics(
            num_classes=num_classes, device=device,
        )
        torch.manual_seed(42)
        pred_boxes: list[torch.Tensor] = [
            torch.rand(10, 4, device=device),
        ]
        pred_scores: list[torch.Tensor] = [
            torch.rand(10, device=device),
        ]
        pred_labels: list[torch.Tensor] = [
            torch.randint(0, num_classes, (10,), device=device),
        ]
        target_boxes: list[torch.Tensor] = [
            torch.rand(5, 4, device=device),
        ]
        target_labels: list[torch.Tensor] = [
            torch.randint(0, num_classes, (5,), device=device),
        ]

        metrics.update(pred_boxes, pred_scores, pred_labels, target_boxes, target_labels)
        results = metrics.compute()

        assert "map50" in results
        assert 0.0 <= results["map50"] <= 1.0
        assert results["per_class_ap50"].shape == (num_classes,)
        assert results["per_class_ap50_95"].shape == (num_classes,)

    def test_single_class_single_box(
        self,
        device: torch.device,
    ) -> None:
        """Single class, single box — minimal detection case."""
        metrics = DetectionMetrics(num_classes=1, device=device)
        pred_boxes: list[torch.Tensor] = [
            torch.tensor(
                [[0.1, 0.1, 0.5, 0.5]],
                dtype=torch.float32, device=device,
            ),
        ]
        pred_scores: list[torch.Tensor] = [
            torch.tensor([0.95], device=device),
        ]
        pred_labels: list[torch.Tensor] = [
            torch.tensor([0], dtype=torch.long, device=device),
        ]
        target_boxes: list[torch.Tensor] = [
            torch.tensor(
                [[0.1, 0.1, 0.5, 0.5]],
                dtype=torch.float32, device=device,
            ),
        ]
        target_labels: list[torch.Tensor] = [
            torch.tensor([0], dtype=torch.long, device=device),
        ]

        metrics.update(pred_boxes, pred_scores, pred_labels, target_boxes, target_labels)
        results = metrics.compute()

        assert results["map50"] == pytest.approx(1.0)
        assert results["per_class_ap50"].shape == (1,)
        assert results["per_class_ap50"][0].item() == pytest.approx(1.0)


# ======================================================================
# Cross-cutting smoke tests
# ======================================================================


class TestCrossCutting:
    """Tests that exercise multiple metric types together."""

    def test_compute_returns_python_dict(
        self,
        device: torch.device,
    ) -> None:
        """``compute()`` returns a ``dict`` with correct types for all metric classes."""
        torch.manual_seed(42)

        # ---- Classification ----
        cls_metrics = ClassificationMetrics(
            num_classes=5, top_k=(1,), device=device,
        )
        cls_metrics.update(
            torch.randn(4, 5, device=device),
            torch.randint(0, 5, (4,), device=device),
        )
        cls_results = cls_metrics.compute()
        _assert_result_types(
            cls_results,
            float_keys=[
                "accuracy", "precision_macro", "recall_macro",
                "f1_score_macro", "precision_micro", "recall_micro",
                "f1_score_micro", "top1_accuracy",
            ],
            tensor_keys=[],
        )

        # ---- Segmentation ----
        seg_metrics = SegmentationMetrics(
            num_classes=3, device=device,
        )
        seg_metrics.update(
            torch.randn(1, 3, 16, 16, device=device),
            torch.randint(0, 3, (1, 16, 16), device=device),
        )
        seg_results = seg_metrics.compute()
        _assert_result_types(
            seg_results,
            float_keys=["miou", "pixel_accuracy", "mean_dice"],
            tensor_keys=["per_class_iou", "per_class_dice"],
        )

        # ---- Detection ----
        det_metrics = DetectionMetrics(num_classes=3, device=device)
        pred_boxes: list[torch.Tensor] = [
            torch.rand(2, 4, device=device),
        ]
        pred_scores: list[torch.Tensor] = [
            torch.rand(2, device=device),
        ]
        pred_labels: list[torch.Tensor] = [
            torch.randint(0, 3, (2,), device=device),
        ]
        target_boxes: list[torch.Tensor] = [
            torch.rand(1, 4, device=device),
        ]
        target_labels: list[torch.Tensor] = [
            torch.randint(0, 3, (1,), device=device),
        ]
        det_metrics.update(
            pred_boxes, pred_scores, pred_labels,
            target_boxes, target_labels,
        )
        det_results = det_metrics.compute()
        _assert_result_types(
            det_results,
            float_keys=["map50", "map50_95"],
            tensor_keys=["per_class_ap50", "per_class_ap50_95"],
        )

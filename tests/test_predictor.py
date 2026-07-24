"""Tests for CorePredictor accelerated inference engine.

Verifies that the predictor correctly handles preprocessing, inference, and
post-processing for classification, segmentation, and detection tasks using
synthetic data and mock models.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from corecv.engine.predictor import CorePredictor, _letterbox

# ---------------------------------------------------------------------------
# Mock models for each task
# ---------------------------------------------------------------------------


class _MockClassificationModel(nn.Module):
    """Minimal mock for a classification model."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        feat = torch.randn(b, 64, 1, 1, device=x.device)
        return self.fc(feat.flatten(1))


class _MockSegmentationModel(nn.Module):
    """Minimal mock for a segmentation model."""

    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        return torch.randn(b, self.num_classes, h, w, device=x.device)


class _MockDetectionModel(nn.Module):
    """Minimal mock for a detection model with FCOS-style output."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.backbone = nn.Identity()
        self.neck = None
        self.num_classes = num_classes
        # Mock head attributes
        self.head = nn.Module()
        self.head.num_classes = num_classes
        self.head.level_strides = [8, 16, 32]

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        b, _c, h, w = x.shape
        levels = []
        for s in self.head.level_strides:
            fh = max(h // s, 1)
            fw = max(w // s, 1)
            levels.append((fh, fw))

        cls_logits = [
            torch.randn(b, self.num_classes, fh, fw, device=x.device)
            for fh, fw in levels
        ]
        reg_pred = [
            torch.randn(b, 4, fh, fw, device=x.device).abs() * 10
            for fh, fw in levels
        ]
        centerness = [
            torch.randn(b, 1, fh, fw, device=x.device)
            for fh, fw in levels
        ]
        return {
            "cls_logits": cls_logits,
            "reg_pred": reg_pred,
            "centerness": centerness,
        }


class _MockQueryDetectionModel(nn.Module):
    """Minimal mock for a detection model with RT-DETR-style output."""

    def __init__(self, num_classes: int = 5, num_queries: int = 50) -> None:  # noqa: ARG002
        super().__init__()
        self.backbone = nn.Identity()
        self.neck = None
        self.head = nn.Module()
        self.head.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        return {
            "cls_logits": torch.randn(b, 50, self.head.num_classes, device=x.device),
            "pred_boxes": torch.rand(b, 50, 4, device=x.device),
        }


# ---------------------------------------------------------------------------
# Letterbox tests
# ---------------------------------------------------------------------------


class TestLetterbox:
    """Tests for the _letterbox helper."""

    def test_letterbox_preserves_aspect_ratio(self) -> None:
        """Letterbox should produce target dimensions with padding."""
        img = torch.randn(3, 100, 200)
        padded, scale, pad_tl, orig_hw = _letterbox(img, 640, 640)

        assert padded.shape == (3, 640, 640)
        assert orig_hw == (100, 200)
        assert scale == pytest.approx(640 / 200)
        assert pad_tl[0] > 0  # top padding
        assert pad_tl[1] == 0  # no horizontal padding (aspect ratio 2:1 fits)

    def test_letterbox_square_input(self) -> None:
        """Square input should not need padding."""
        img = torch.randn(3, 640, 640)
        padded, scale, pad_tl, orig_hw = _letterbox(img, 640, 640)

        assert padded.shape == (3, 640, 640)
        assert scale == pytest.approx(1.0)
        assert pad_tl == (0, 0)

    def test_letterbox_tall_input(self) -> None:
        """Tall image should get horizontal padding."""
        img = torch.randn(3, 640, 320)
        padded, scale, pad_tl, orig_hw = _letterbox(img, 640, 640)

        assert padded.shape == (3, 640, 640)
        assert pad_tl[1] > 0  # horizontal padding


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


class TestPreprocessing:
    """Tests for CorePredictor._preprocess."""

    def test_preprocess_single_image(self) -> None:
        """Single numpy image should produce a batch of size 1."""
        model = _MockClassificationModel()
        predictor = CorePredictor(model=model, task="classification")
        images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)]

        batch, original_sizes, letterbox_meta = predictor._preprocess(images)

        assert batch.shape == (1, 3, 640, 640)
        assert batch.dtype == torch.float32
        assert len(original_sizes) == 1
        assert original_sizes[0] == (480, 640)
        assert len(letterbox_meta) == 1

    def test_preprocess_batch(self) -> None:
        """Multiple images should produce a stacked batch."""
        model = _MockClassificationModel()
        predictor = CorePredictor(model=model, task="classification")
        images = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8),
        ]

        batch, original_sizes, letterbox_meta = predictor._preprocess(images)

        assert batch.shape == (2, 3, 640, 640)
        assert len(original_sizes) == 2
        assert original_sizes[0] == (480, 640)
        assert original_sizes[1] == (300, 300)

    def test_preprocess_normalization(self) -> None:
        """Preprocessed tensor should be ImageNet-normalised."""
        model = _MockClassificationModel()
        predictor = CorePredictor(
            model=model,
            task="classification",
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )
        img = np.full((640, 640, 3), 128, dtype=np.uint8)

        batch, _, _ = predictor._preprocess([img])

        # With mean=0, std=1, values should be 128/255 ~= 0.502
        assert batch.mean() == pytest.approx(0.502, abs=0.01)


# ---------------------------------------------------------------------------
# Classification predict tests
# ---------------------------------------------------------------------------


class TestClassificationPredict:
    """Tests for classification prediction end-to-end."""

    def test_predict_numpy_array(self) -> None:
        """Predict on a single numpy array."""
        model = _MockClassificationModel(num_classes=10)
        predictor = CorePredictor(
            model=model, task="classification", topk=3,
        )
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        results = predictor.predict(img)

        assert len(results) == 1
        assert results[0].task == "classification"
        assert results[0].classification is not None
        assert results[0].classification.class_ids.shape == (3,)
        assert results[0].classification.scores.shape == (3,)

    def test_predict_tensor(self) -> None:
        """Predict on a single tensor."""
        model = _MockClassificationModel(num_classes=5)
        predictor = CorePredictor(model=model, task="classification", topk=2)
        img = torch.rand(3, 224, 224)
        results = predictor.predict(img)

        assert len(results) == 1
        assert results[0].classification is not None
        assert results[0].classification.class_ids.shape == (2,)

    def test_predict_batch(self) -> None:
        """Predict on a list of images."""
        model = _MockClassificationModel(num_classes=10)
        predictor = CorePredictor(model=model, task="classification")
        imgs = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(4)]
        results = predictor.predict(imgs)

        assert len(results) == 4
        for r in results:
            assert r.classification is not None

    def test_predict_batch_tensor(self) -> None:
        """predict_batch should work with pre-batched tensors."""
        model = _MockClassificationModel(num_classes=10)
        predictor = CorePredictor(model=model, task="classification", topk=3)
        tensors = [torch.rand(3, 224, 224) for _ in range(3)]
        results = predictor.predict_batch(tensors)

        assert len(results) == 3
        for r in results:
            assert r.classification is not None
            assert r.classification.class_ids.shape == (3,)

    def test_topk_clamped_to_num_classes(self) -> None:
        """Topk should be clamped when larger than num_classes."""
        model = _MockClassificationModel(num_classes=3)
        predictor = CorePredictor(
            model=model, task="classification", topk=10,
        )
        img = torch.rand(3, 224, 224)
        results = predictor.predict(img)

        assert results[0].classification.class_ids.shape == (3,)


# ---------------------------------------------------------------------------
# Segmentation predict tests
# ---------------------------------------------------------------------------


class TestSegmentationPredict:
    """Tests for segmentation prediction end-to-end."""

    def test_predict_segmentation(self) -> None:
        """Predict on a numpy image for segmentation."""
        model = _MockSegmentationModel(num_classes=21)
        predictor = CorePredictor(
            model=model, task="segmentation", input_size=(256, 256),
        )
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        results = predictor.predict(img)

        assert len(results) == 1
        assert results[0].task == "segmentation"
        assert results[0].segmentation is not None
        # Mask should be resized to original image size
        assert results[0].segmentation.mask.shape == (480, 640)
        assert results[0].segmentation.original_size == (480, 640)

    def test_predict_segmentation_batch(self) -> None:
        """Batch segmentation prediction."""
        model = _MockSegmentationModel(num_classes=10)
        predictor = CorePredictor(
            model=model, task="segmentation", input_size=(256, 256),
        )
        imgs = [
            np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8),
            np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8),
        ]
        results = predictor.predict(imgs)

        assert len(results) == 2
        assert results[0].segmentation.mask.shape == (300, 400)
        assert results[1].segmentation.mask.shape == (200, 200)


# ---------------------------------------------------------------------------
# Detection predict tests (FCOS / DecoupledAnchorFree)
# ---------------------------------------------------------------------------


class TestDetectionPredict:
    """Tests for detection prediction end-to-end."""

    def test_predict_fcos_detection(self) -> None:
        """Predict on a numpy image for FCOS-style detection."""
        model = _MockDetectionModel(num_classes=5)
        predictor = CorePredictor(
            model=model, task="detection", input_size=(640, 640),
        )
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        results = predictor.predict(img)

        assert len(results) == 1
        assert results[0].task == "detection"
        assert results[0].detection is not None
        # Boxes should have 4 columns
        assert results[0].detection.boxes.shape[1] == 4
        assert results[0].detection.scores.shape == results[0].detection.class_ids.shape

    def test_predict_query_detection(self) -> None:
        """Predict on a numpy image for query-based (RT-DETR) detection."""
        model = _MockQueryDetectionModel(num_classes=5)
        predictor = CorePredictor(
            model=model, task="detection", input_size=(640, 640),
        )
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        results = predictor.predict(img)

        assert len(results) == 1
        assert results[0].detection is not None
        assert results[0].detection.boxes.shape[1] == 4


# ---------------------------------------------------------------------------
# Rescale boxes test
# ---------------------------------------------------------------------------


class TestRescaleBoxes:
    """Tests for CorePredictor._rescale_boxes."""

    def test_rescale_identity(self) -> None:
        """Rescaling to the same size should be a no-op."""
        boxes = torch.tensor([[10, 20, 30, 40]], dtype=torch.float32)
        rescaled = CorePredictor._rescale_boxes(boxes, (100, 100), (100, 100))
        assert torch.allclose(boxes, rescaled)

    def test_rescale_2x(self) -> None:
        """Rescaling by 2x should double box coordinates."""
        boxes = torch.tensor([[10, 20, 30, 40]], dtype=torch.float32)
        rescaled = CorePredictor._rescale_boxes(boxes, (100, 100), (200, 200))
        expected = torch.tensor([[20, 40, 60, 80]], dtype=torch.float32)
        assert torch.allclose(rescaled, expected)

    def test_rescale_empty(self) -> None:
        """Rescaling empty boxes should return empty tensor."""
        boxes = torch.zeros(0, 4)
        rescaled = CorePredictor._rescale_boxes(boxes, (100, 100), (200, 200))
        assert rescaled.shape == (0, 4)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_task_raises(self) -> None:
        """Invalid task string should raise ValueError."""
        model = _MockClassificationModel()
        with pytest.raises(ValueError, match="Unsupported task"):
            CorePredictor(model=model, task="invalid_task")

    def test_file_not_found(self) -> None:
        """Non-existent file path should raise FileNotFoundError."""
        model = _MockClassificationModel()
        predictor = CorePredictor(model=model, task="classification")
        with pytest.raises(FileNotFoundError):
            predictor.predict("nonexistent_image.jpg")

    def test_invalid_numpy_shape(self) -> None:
        """Unexpected numpy shape should raise TypeError."""
        model = _MockClassificationModel()
        predictor = CorePredictor(model=model, task="classification")
        with pytest.raises(TypeError, match="Expected HWC or CHW"):
            predictor._numpy_to_tensor(np.random.rand(100, 100, 5))

    def test_empty_batch_detection(self) -> None:
        """Detection with all scores below threshold returns empty results."""
        model = _MockDetectionModel(num_classes=5)
        predictor = CorePredictor(
            model=model,
            task="detection",
            conf_threshold=0.99,  # Very high threshold
            input_size=(640, 640),
        )
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        results = predictor.predict(img)

        assert len(results) == 1
        assert results[0].detection is not None
        # All boxes filtered out
        assert results[0].detection.boxes.shape[0] == 0

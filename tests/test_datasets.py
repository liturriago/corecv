"""Comprehensive tests for all three dataset types.

Tests cover:

1.  :class:`ClassificationDataset` — basic loading, transforms (raw
    Albumentations and ``CoordinatedTransform``), non-square resolutions,
    missing-root error.
2.  :class:`SegmentationDataset` — O(N) pre-flight check, ``.npz`` cache
    generation and reading, dynamic mask loading from disk (not from cache),
    synchronised ``CoordinatedTransform`` (image + mask), non-square
    resolutions, cache invalidation via mtime, ``ignore_index``, directory
    and empty-dataset errors.
3.  :class:`DetectionDataset` — COCO and YOLO annotation formats, O(N)
    pre-flight check, cache generation/reading, synchronised transforms
    (image + bboxes + labels), ``bbox_format`` output control (``xyxy``
    vs ``norm_xyxy``), non-square resolutions, cache invalidation via
    mtime, user-supplied ``class_names``, unsupported-format errors.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import albumentations as A
import numpy as np
import pytest
import torch
from PIL import Image

from corecv.data.datasets import DetectionDataset, SegmentationDataset
from corecv.data.datasets.classification import ClassificationDataset
from corecv.data.transforms import (
    ClassificationTransformConfig,
    DetectionTransformConfig,
    SegmentationTransformConfig,
    build_transforms,
)

# ======================================================================
# Helper functions
# ======================================================================


def _create_image(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    """Create a synthetic RGB image saved to *path*.

    Args:
        path: Destination file path (parent must exist).
        size: ``(height, width)`` in pixels.

    Returns:
        The *path* argument.
    """
    arr: np.ndarray = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(path))
    return path


def _create_mask(
    path: Path,
    size: tuple[int, int] = (64, 64),
    num_classes: int = 10,
) -> Path:
    """Create a synthetic uint8 segmentation mask saved to *path*.

    Args:
        path: Destination file path.
        size: ``(height, width)`` in pixels.
        num_classes: Values are drawn from ``[0, num_classes)``.

    Returns:
        The *path* argument.
    """
    arr: np.ndarray = np.random.randint(0, num_classes, size, dtype=np.uint8)
    Image.fromarray(arr).save(str(path))
    return path


def _make_coco_json(
    annotation_path: Path,
    images: list[dict],
    annotations: list[dict],
    categories: list[dict],
) -> Path:
    """Write a minimal COCO JSON annotation file.

    Args:
        annotation_path: Destination path.
        images: List of COCO ``images`` entries.
        annotations: List of COCO ``annotations`` entries.
        categories: List of COCO ``categories`` entries.

    Returns:
        The *annotation_path*.
    """
    coco: dict[str, object] = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with annotation_path.open("w") as f:
        json.dump(coco, f)
    return annotation_path


def _make_yolo_label(
    label_path: Path,
    boxes: list[tuple[int, float, float, float, float]],
) -> Path:
    """Write a YOLO-format label file.

    Each box tuple is ``(class_id, cx, cy, w, h)`` with normalised values
    in ``[0, 1]``.

    Args:
        label_path: Destination ``.txt`` file.
        boxes: List of YOLO annotation tuples.

    Returns:
        The *label_path*.
    """
    with label_path.open("w") as f:
        for class_id, cx, cy, w, h in boxes:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return label_path


# ======================================================================
# Fixtures — classification
# ======================================================================


@pytest.fixture
def classification_root(tmp_path: Path) -> Path:
    """Create a temporary ImageFolder-style classification dataset.

    Structure::

        <root>/
            class_0/
                img_000.png
                img_001.png
            class_1/
                img_002.png
                img_003.png
    """
    root: Path = tmp_path / "classification_data"
    for cls_idx in range(2):
        cls_dir: Path = root / f"class_{cls_idx}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        for sample_idx in range(2):
            _create_image(cls_dir / f"img_{sample_idx:03d}.png", size=(64, 64))
    return root


@pytest.fixture
def classification_non_square_root(tmp_path: Path) -> Path:
    """Classification root with a non-square (48, 64) image."""
    root: Path = tmp_path / "cls_ns"
    cls_dir: Path = root / "class_0"
    cls_dir.mkdir(parents=True, exist_ok=True)
    _create_image(cls_dir / "img.png", size=(48, 64))
    return root


# ======================================================================
# Fixtures — segmentation
# ======================================================================


@pytest.fixture
def segmentation_root(tmp_path: Path) -> Path:
    """Create a temporary segmentation dataset.

    Structure::

        <root>/
            images/
                img_000.png
                img_001.png
            masks/
                img_000.png
                img_001.png
    """
    root: Path = tmp_path / "segmentation_data"
    images_dir: Path = root / "images"
    masks_dir: Path = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(2):
        _create_image(images_dir / f"img_{idx:03d}.png", size=(64, 64))
        _create_mask(masks_dir / f"img_{idx:03d}.png", size=(64, 64), num_classes=10)
    return root


@pytest.fixture
def segmentation_non_square_root(tmp_path: Path) -> Path:
    """Segmentation root with non-square (48, 64) images/masks."""
    root: Path = tmp_path / "seg_ns"
    images_dir: Path = root / "images"
    masks_dir: Path = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    _create_image(images_dir / "img_000.png", size=(48, 64))
    _create_mask(masks_dir / "img_000.png", size=(48, 64), num_classes=10)
    return root


# ======================================================================
# Fixtures — detection (COCO)
# ======================================================================


@pytest.fixture
def detection_coco_root(tmp_path: Path) -> Path:
    """Create a temporary COCO detection dataset.

    Structure::

        <root>/
            image_000.png
            image_001.png
            annotations.json

    Two categories (cat, dog), two images.
    """
    root: Path = tmp_path / "detection_coco"
    root.mkdir(parents=True, exist_ok=True)

    img_w: int = 100
    img_h: int = 100
    for idx in range(2):
        _create_image(root / f"image_{idx:03d}.png", size=(img_h, img_w))

    images: list[dict[str, object]] = [
        {"id": 1, "file_name": "image_000.png", "width": img_w, "height": img_h},
        {"id": 2, "file_name": "image_001.png", "width": img_w, "height": img_h},
    ]
    annotations: list[dict[str, object]] = [
        {"id": 1, "image_id": 1, "category_id": 1,
         "bbox": [10, 10, 20, 30], "area": 600, "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 2,
         "bbox": [30, 40, 50, 20], "area": 1000, "iscrowd": 0},
        {"id": 3, "image_id": 2, "category_id": 1,
         "bbox": [5, 5, 80, 90], "area": 7200, "iscrowd": 0},
    ]
    categories: list[dict[str, object]] = [
        {"id": 1, "name": "cat", "supercategory": "animal"},
        {"id": 2, "name": "dog", "supercategory": "animal"},
    ]
    _make_coco_json(root / "annotations.json", images, annotations, categories)
    return root


@pytest.fixture
def non_square_coco_root(tmp_path: Path) -> Path:
    """COCO detection root with a non-square (48, 64) image."""
    root: Path = tmp_path / "det_coco_ns"
    root.mkdir(parents=True, exist_ok=True)

    img_h: int = 48
    img_w: int = 64
    _create_image(root / "image_000.png", size=(img_h, img_w))

    images: list[dict[str, object]] = [
        {"id": 1, "file_name": "image_000.png", "width": img_w, "height": img_h},
    ]
    annotations: list[dict[str, object]] = [
        {"id": 1, "image_id": 1, "category_id": 1,
         "bbox": [5, 5, 30, 20], "area": 600, "iscrowd": 0},
    ]
    categories: list[dict[str, object]] = [
        {"id": 1, "name": "cat", "supercategory": "animal"},
    ]
    _make_coco_json(root / "annotations.json", images, annotations, categories)
    return root


# ======================================================================
# Fixtures — detection (YOLO)
# ======================================================================


@pytest.fixture
def detection_yolo_root(tmp_path: Path) -> Path:
    """Create a temporary YOLO detection dataset.

    Structure::

        <root>/
            image_000.png
            image_001.png
            labels/
                image_000.txt
                image_001.txt
    """
    root: Path = tmp_path / "detection_yolo"
    root.mkdir(parents=True, exist_ok=True)

    img_w: int = 100
    img_h: int = 100
    for idx in range(2):
        _create_image(root / f"image_{idx:03d}.png", size=(img_h, img_w))

    labels_dir: Path = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    _make_yolo_label(
        labels_dir / "image_000.txt",
        [
            (0, 0.3, 0.4, 0.2, 0.3),
            (1, 0.6, 0.5, 0.4, 0.2),
        ],
    )
    _make_yolo_label(
        labels_dir / "image_001.txt",
        [
            (0, 0.5, 0.5, 0.8, 0.9),
        ],
    )
    return root


@pytest.fixture
def non_square_yolo_root(tmp_path: Path) -> Path:
    """YOLO detection root with a non-square (48, 64) image."""
    root: Path = tmp_path / "det_yolo_ns"
    root.mkdir(parents=True, exist_ok=True)

    _create_image(root / "image_000.png", size=(48, 64))

    labels_dir: Path = root / "labels"
    labels_dir.mkdir()
    _make_yolo_label(labels_dir / "image_000.txt", [(0, 0.5, 0.5, 0.8, 0.8)])
    return root


# ======================================================================
# ClassificationDataset tests
# ======================================================================


class TestClassificationDataset:
    """Verify :class:`ClassificationDataset` loading, transforms, edge cases."""

    # ------------------------------------------------------------------
    # Basic loading
    # ------------------------------------------------------------------

    def test_basic_loading(self, classification_root: Path) -> None:
        """Images are discovered from class-named subdirectories."""
        dataset = ClassificationDataset(str(classification_root))
        assert len(dataset) == 4
        assert dataset.classes == ["class_0", "class_1"]
        assert dataset.class_to_idx["class_0"] == 0
        assert dataset.class_to_idx["class_1"] == 1

    def test_getitem_shape(self, classification_root: Path) -> None:
        """``__getitem__`` returns ``(image_tensor, label)`` with correct shapes."""
        dataset = ClassificationDataset(str(classification_root))
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 64, 64), f"Image shape: {image.shape}"
        assert isinstance(label, int)
        assert label in (0, 1)

    def test_getitem_all_indices(self, classification_root: Path) -> None:
        """Every index yields a valid ``(image, label)`` pair."""
        dataset = ClassificationDataset(str(classification_root))
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            assert image.shape == (3, 64, 64)
            assert 0 <= label <= 1

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def test_with_albumentations_transform(self, classification_root: Path) -> None:
        """Raw Albumentations pipeline is applied to images."""
        transform = A.Compose([
            A.Resize(height=32, width=32),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        dataset = ClassificationDataset(str(classification_root), transform=transform)
        image, _ = dataset[0]
        assert image.shape == (3, 32, 32), f"Expected (3, 32, 32), got {image.shape}"

    def test_with_coordinated_transform(self, classification_root: Path) -> None:
        """``CoordinatedTransform`` (classification config) produces resized images."""
        config = ClassificationTransformConfig(image_size=(32, 32))
        transform = build_transforms(config)
        dataset = ClassificationDataset(str(classification_root), transform=transform)
        image, _ = dataset[0]
        assert image.shape == (3, 32, 32), f"Expected (3, 32, 32), got {image.shape}"

    def test_no_transform(self, classification_root: Path) -> None:
        """Without a transform, images are returned as raw uint8 tensors."""
        dataset = ClassificationDataset(str(classification_root))
        image, _ = dataset[0]
        assert image.dtype == torch.uint8
        # Raw pixel values — not normalised, so range should be [0, 255]
        assert image.max() <= 255.0

    # ------------------------------------------------------------------
    # Non-square resolution
    # ------------------------------------------------------------------

    def test_non_square_resolution(self, classification_non_square_root: Path) -> None:
        """Non-square (48, 64) image yields correct ``[C, H, W]`` tensor."""
        dataset = ClassificationDataset(str(classification_non_square_root))
        image, _ = dataset[0]
        assert image.shape[1] == 48, f"Expected height 48, got {image.shape[1]}"
        assert image.shape[2] == 64, f"Expected width 64, got {image.shape[2]}"

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_missing_root(self) -> None:
        """Non-existent root raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            ClassificationDataset("/nonexistent/path")


# ======================================================================
# SegmentationDataset tests
# ======================================================================


class TestSegmentationDataset:
    """Verify :class:`SegmentationDataset` loading, cache, transforms, errors."""

    # ------------------------------------------------------------------
    # Basic loading
    # ------------------------------------------------------------------

    def test_basic_loading(self, segmentation_root: Path) -> None:
        """Image-mask pairs are discovered from ``images/`` and ``masks/`` dirs."""
        dataset = SegmentationDataset(str(segmentation_root), num_classes=10)
        assert len(dataset) == 2
        assert len(dataset.image_paths) == 2
        assert len(dataset.mask_paths) == 2

    def test_getitem_shapes(self, segmentation_root: Path) -> None:
        """``__getitem__`` returns ``(image, mask)`` with expected shapes and dtypes."""
        dataset = SegmentationDataset(str(segmentation_root), num_classes=10)
        image, mask = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

        # Image has shape (C, H, W) with float32 dtype.
        assert image.shape == (3, 64, 64), f"Image shape: {image.shape}"
        assert image.dtype == torch.float32

        # Mask has shape (H, W) with long dtype.
        assert mask.shape == (64, 64), f"Mask shape: {mask.shape}"
        assert mask.dtype == torch.long

    # ------------------------------------------------------------------
    # Pre-flight check and caching
    # ------------------------------------------------------------------

    def test_preflight_validates_masks(self, tmp_path: Path) -> None:
        """Invalid mask values raise ``ValueError`` during pre-flight."""
        root: Path = tmp_path / "seg_invalid"
        (root / "images").mkdir(parents=True)
        (root / "masks").mkdir(parents=True)

        _create_image(root / "images" / "img.png", size=(16, 16))
        # Mask with value >= num_classes
        bad_mask: np.ndarray = np.full((16, 16), 99, dtype=np.uint8)
        Image.fromarray(bad_mask).save(str(root / "masks" / "img.png"))

        with pytest.raises(ValueError, match="outside the valid range"):
            SegmentationDataset(str(root), num_classes=10)

    def test_cache_file_generation(self, segmentation_root: Path) -> None:
        """``.npz`` cache file is created after dataset initialisation."""
        dataset = SegmentationDataset(str(segmentation_root), num_classes=10)
        assert dataset.cache_path.exists(), "Cache file should exist after init"

    def test_cache_loading(self, segmentation_root: Path) -> None:
        """Second initialisation loads from cache without re-validation."""
        ds1 = SegmentationDataset(str(segmentation_root), num_classes=10)
        ds2 = SegmentationDataset(str(segmentation_root), num_classes=10)
        assert len(ds1) == len(ds2)
        for i in range(len(ds1)):
            img1, msk1 = ds1[i]
            img2, msk2 = ds2[i]
            assert img1.shape == img2.shape
            assert msk1.shape == msk2.shape

    def test_cache_contains_only_metadata(self, segmentation_root: Path) -> None:
        """Cache stores only path manifest, not mask pixel arrays."""
        dataset = SegmentationDataset(str(segmentation_root), num_classes=10)
        with np.load(dataset.cache_path) as loader:
            # Should contain metadata and cache_version but no pixel data
            assert "mask" not in loader
            assert "mask_pixels" not in loader
            assert "metadata" in loader
            assert "cache_version" in loader

    # ------------------------------------------------------------------
    # Mask loading
    # ------------------------------------------------------------------

    def test_mask_read_from_disk_dynamically(self, segmentation_root: Path) -> None:
        """Mask pixel data is read from disk at ``__getitem__`` time."""
        dataset = SegmentationDataset(str(segmentation_root), num_classes=10)
        _, mask = dataset[0]
        assert mask.numel() > 0
        assert mask.min() >= 0
        assert mask.max() < 10

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def test_with_coordinated_transform(self, segmentation_root: Path) -> None:
        """``CoordinatedTransform`` applies synchronised image + mask augmentation."""
        config = SegmentationTransformConfig(image_size=(32, 32))
        transform = build_transforms(config)
        dataset = SegmentationDataset(
            str(segmentation_root), num_classes=10, transforms=transform,
        )
        image, mask = dataset[0]
        assert image.shape == (3, 32, 32), f"Image shape mismatch: {image.shape}"
        assert mask.shape == (32, 32), f"Mask shape mismatch: {mask.shape}"

    # ------------------------------------------------------------------
    # ignore_index
    # ------------------------------------------------------------------

    def test_ignore_index(self, tmp_path: Path) -> None:
        """Pixels matching ``ignore_index`` are excluded from range validation."""
        root: Path = tmp_path / "seg_ignore"
        (root / "images").mkdir(parents=True)
        (root / "masks").mkdir(parents=True)

        _create_image(root / "images" / "img.png", size=(16, 16))

        # Mask with a 255 (ignore_index) pixel
        mask_arr: np.ndarray = np.random.randint(0, 5, (16, 16), dtype=np.uint8)
        mask_arr[0, 0] = 255
        Image.fromarray(mask_arr).save(str(root / "masks" / "img.png"))

        dataset = SegmentationDataset(str(root), num_classes=5, ignore_index=255)
        assert len(dataset) == 1

    # ------------------------------------------------------------------
    # Non-square resolution
    # ------------------------------------------------------------------

    def test_non_square_resolution(self, segmentation_non_square_root: Path) -> None:
        """Non-square (48, 64) images produce correct ``[C, H, W]`` and ``[H, W]`` shapes."""
        dataset = SegmentationDataset(str(segmentation_non_square_root), num_classes=10)
        image, mask = dataset[0]
        # Image has shape (C, H, W).
        assert image.shape[1] == 48, f"Expected height 48, got {image.shape[1]}"
        assert image.shape[2] == 64, f"Expected width 64, got {image.shape[2]}"
        # Mask has shape (H, W).
        assert mask.shape[0] == 48
        assert mask.shape[1] == 64

    # ------------------------------------------------------------------
    # Cache invalidation (mtime-based)
    # ------------------------------------------------------------------

    def test_cache_invalidation_mtime(self, segmentation_root: Path) -> None:
        """Cache is rebuilt when a mask file's modification time changes."""
        dataset = SegmentationDataset(str(segmentation_root), num_classes=10)
        cache_mtime: float = dataset.cache_path.stat().st_mtime

        # "Touch" a mask file to update its mtime
        mask_path: Path = dataset.mask_paths[0]
        time.sleep(0.02)  # ensure distinct mtime
        mask_path.touch()

        # Re-initialise — should detect change and rebuild
        dataset2 = SegmentationDataset(str(segmentation_root), num_classes=10)
        assert dataset2.cache_path.stat().st_mtime_ns != cache_mtime, (
            "Cache should have been rebuilt after mask mtime change"
        )
        assert len(dataset2) == 2

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_missing_images_dir(self, tmp_path: Path) -> None:
        """Missing ``images/`` directory raises ``NotADirectoryError``."""
        root: Path = tmp_path / "seg_no_images"
        root.mkdir(parents=True)
        (root / "masks").mkdir()
        with pytest.raises(NotADirectoryError):
            SegmentationDataset(str(root), num_classes=10)

    def test_missing_masks_dir(self, tmp_path: Path) -> None:
        """Missing ``masks/`` directory raises ``NotADirectoryError``."""
        root: Path = tmp_path / "seg_no_masks"
        root.mkdir(parents=True)
        (root / "images").mkdir()
        with pytest.raises(NotADirectoryError):
            SegmentationDataset(str(root), num_classes=10)

    def test_empty_dataset(self, tmp_path: Path) -> None:
        """No valid image-mask pairs raises ``RuntimeError``."""
        root: Path = tmp_path / "seg_empty"
        root.mkdir(parents=True)
        (root / "images").mkdir()
        (root / "masks").mkdir()
        with pytest.raises(RuntimeError, match="No valid image-mask pairs"):
            SegmentationDataset(str(root), num_classes=10)


# ======================================================================
# DetectionDataset tests — COCO format
# ======================================================================


class TestDetectionDatasetCOCO:
    """Verify :class:`DetectionDataset` with COCO JSON annotations."""

    # ------------------------------------------------------------------
    # Basic loading
    # ------------------------------------------------------------------

    def test_basic_loading(self, detection_coco_root: Path) -> None:
        """COCO annotations are loaded and parsed correctly."""
        dataset = DetectionDataset(
            str(detection_coco_root),
            format="coco",
            use_cache=False,
            bbox_format="xyxy",
        )
        assert len(dataset) == 2
        assert dataset.num_classes == 2
        assert dataset.class_names == ["cat", "dog"]

    def test_getitem_shapes(self, detection_coco_root: Path) -> None:
        """``__getitem__`` returns ``(image, bboxes, labels)`` with correct shapes."""
        dataset = DetectionDataset(
            str(detection_coco_root),
            format="coco",
            use_cache=False,
            bbox_format="xyxy",
        )
        image, bboxes, labels = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(bboxes, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

        # Image has shape (C, H, W) with float32 dtype.
        assert image.dim() == 3
        assert image.shape[0] == 3
        assert image.dtype == torch.float32

        # Bboxes have shape (N, 4).
        assert bboxes.shape[1] == 4
        assert bboxes.dtype == torch.float32

        # Labels have shape (N,).
        assert labels.dim() == 1
        assert labels.dtype == torch.int64

    # ------------------------------------------------------------------
    # Bbox format output
    # ------------------------------------------------------------------

    def test_bbox_format_xyxy(self, detection_coco_root: Path) -> None:
        """``bbox_format='xyxy'`` returns absolute pixel coordinates."""
        dataset = DetectionDataset(
            str(detection_coco_root),
            format="coco",
            use_cache=False,
            bbox_format="xyxy",
        )
        _, bboxes, _ = dataset[0]
        assert bboxes.shape[1] == 4
        # Absolute coordinates should be within image dimensions (100, 100)
        assert bboxes.min() >= 0.0
        assert bboxes.max() <= 100.0

    def test_bbox_format_norm_xyxy(self, detection_coco_root: Path) -> None:
        """``bbox_format='norm_xyxy'`` returns normalised ``[0, 1]`` coordinates."""
        dataset = DetectionDataset(
            str(detection_coco_root),
            format="coco",
            use_cache=False,
            bbox_format="norm_xyxy",
        )
        _, bboxes, _ = dataset[0]
        assert bboxes.min() >= 0.0
        assert bboxes.max() <= 1.0

    # ------------------------------------------------------------------
    # Pre-flight check and caching
    # ------------------------------------------------------------------

    def test_cache_generation(self, detection_coco_root: Path) -> None:
        """Cache ``.npz`` file is created when ``use_cache=True``."""
        dataset = DetectionDataset(
            str(detection_coco_root),
            format="coco",
            use_cache=True,
            bbox_format="xyxy",
        )
        assert dataset._cache_path.exists(), "Cache file should exist"

    def test_cache_loading(self, detection_coco_root: Path) -> None:
        """Second initialisation loads from cache without re-parsing annotations."""
        ds1 = DetectionDataset(
            str(detection_coco_root),
            format="coco",
            use_cache=True,
            bbox_format="xyxy",
        )
        ds2 = DetectionDataset(
            str(detection_coco_root),
            format="coco",
            use_cache=True,
            bbox_format="xyxy",
        )
        assert len(ds1) == len(ds2)

    def test_cache_invalidation_mtime(self, detection_coco_root: Path) -> None:
        """Cache is rebuilt when the COCO annotation file's mtime changes."""
        dataset = DetectionDataset(
            str(detection_coco_root),
            format="coco",
            use_cache=True,
            bbox_format="xyxy",
        )
        cache_path: Path = dataset._cache_path
        orig_mtime_ns: float = cache_path.stat().st_mtime_ns

        # Modify the annotation file to invalidate cache
        ann_path: Path = detection_coco_root / "annotations.json"
        time.sleep(0.05)
        with ann_path.open("a") as f:
            f.write(" ")

        # Re-initialise — should detect change and rebuild
        DetectionDataset(
            str(detection_coco_root),
            format="coco",
            use_cache=True,
            bbox_format="xyxy",
        )
        new_mtime_ns: float = cache_path.stat().st_mtime_ns
        assert new_mtime_ns != orig_mtime_ns, "Cache should have been rebuilt"

    def test_preflight_validates_bbox_geometry(self, tmp_path: Path) -> None:
        """Non-positive bbox dimensions raise ``ValueError`` during pre-flight."""
        root: Path = tmp_path / "det_bad_bbox"
        root.mkdir(parents=True)
        _create_image(root / "img.png", size=(100, 100))

        images: list[dict[str, object]] = [
            {"id": 1, "file_name": "img.png", "width": 100, "height": 100},
        ]
        annotations: list[dict[str, object]] = [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [10, 10, 0, 30], "area": 0, "iscrowd": 0},
        ]
        categories: list[dict[str, object]] = [
            {"id": 1, "name": "cat", "supercategory": "animal"},
        ]
        _make_coco_json(root / "annotations.json", images, annotations, categories)

        with pytest.raises(ValueError, match="Non-positive bbox dimensions"):
            DetectionDataset(str(root), format="coco", use_cache=False)

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def test_with_coordinated_transform(self, detection_coco_root: Path) -> None:
        """``CoordinatedTransform`` applies synchronised image + bbox + label augmentation."""
        config = DetectionTransformConfig(
            image_size=(32, 32), bbox_format="pascal_voc", min_area=0,
        )
        transform = build_transforms(config)
        dataset = DetectionDataset(
            str(detection_coco_root),
            format="coco",
            transform=transform,
            use_cache=False,
            bbox_format="xyxy",
        )
        image, bboxes, labels = dataset[0]
        assert image.shape == (3, 32, 32), f"Image shape: {image.shape}"
        assert bboxes.shape[1] == 4
        assert labels.shape[0] == bboxes.shape[0]

    # ------------------------------------------------------------------
    # Non-square resolution
    # ------------------------------------------------------------------

    def test_non_square_resolution(self, non_square_coco_root: Path) -> None:
        """Non-square (48, 64) images produce correct tensor shapes."""
        dataset = DetectionDataset(
            str(non_square_coco_root),
            format="coco",
            use_cache=False,
            bbox_format="norm_xyxy",
        )
        image, bboxes, labels = dataset[0]
        # Image has shape (C, H, W).
        assert image.shape[1] == 48, f"Expected height 48, got {image.shape[1]}"
        assert image.shape[2] == 64, f"Expected width 64, got {image.shape[2]}"
        assert bboxes.shape[1] == 4
        assert labels.shape[0] == bboxes.shape[0]

    # ------------------------------------------------------------------
    # User-supplied class names
    # ------------------------------------------------------------------

    def test_user_class_names(self, detection_coco_root: Path) -> None:
        """User-supplied ``class_names`` override COCO JSON categories."""
        user_names: list[str] = ["custom_cat", "custom_dog"]
        dataset = DetectionDataset(
            str(detection_coco_root),
            format="coco",
            use_cache=False,
            bbox_format="xyxy",
            class_names=user_names,
        )
        assert dataset.class_names == user_names


# ======================================================================
# DetectionDataset tests — YOLO format
# ======================================================================


class TestDetectionDatasetYOLO:
    """Verify :class:`DetectionDataset` with YOLO txt annotations."""

    # ------------------------------------------------------------------
    # Basic loading
    # ------------------------------------------------------------------

    def test_basic_loading(self, detection_yolo_root: Path) -> None:
        """YOLO annotations are loaded and parsed correctly."""
        dataset = DetectionDataset(
            str(detection_yolo_root),
            format="yolo",
            use_cache=False,
            bbox_format="xyxy",
        )
        assert len(dataset) == 2

    def test_getitem_shapes(self, detection_yolo_root: Path) -> None:
        """``__getitem__`` returns correct shapes for YOLO data."""
        dataset = DetectionDataset(
            str(detection_yolo_root),
            format="yolo",
            use_cache=False,
            bbox_format="xyxy",
        )
        image, bboxes, labels = dataset[0]
        # Image has shape (C, H, W).
        assert image.shape[0] == 3
        assert bboxes.shape[1] == 4
        assert labels.shape[0] == bboxes.shape[0]

    # ------------------------------------------------------------------
    # Pre-flight check and caching
    # ------------------------------------------------------------------

    def test_cache_generation(self, detection_yolo_root: Path) -> None:
        """Cache is generated for YOLO-format datasets."""
        dataset = DetectionDataset(
            str(detection_yolo_root),
            format="yolo",
            use_cache=True,
            bbox_format="xyxy",
        )
        assert dataset._cache_path.exists()

    def test_cache_loading(self, detection_yolo_root: Path) -> None:
        """Second initialisation loads YOLO cache without re-parsing."""
        ds1 = DetectionDataset(
            str(detection_yolo_root),
            format="yolo",
            use_cache=True,
            bbox_format="xyxy",
        )
        ds2 = DetectionDataset(
            str(detection_yolo_root),
            format="yolo",
            use_cache=True,
            bbox_format="xyxy",
        )
        assert len(ds1) == len(ds2)

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def test_with_coordinated_transform(self, detection_yolo_root: Path) -> None:
        """``CoordinatedTransform`` works with YOLO annotations."""
        config = DetectionTransformConfig(
            image_size=(32, 32), bbox_format="pascal_voc", min_area=0,
        )
        transform = build_transforms(config)
        dataset = DetectionDataset(
            str(detection_yolo_root),
            format="yolo",
            transform=transform,
            use_cache=False,
            bbox_format="xyxy",
        )
        image, bboxes, labels = dataset[0]
        assert image.shape == (3, 32, 32)
        assert bboxes.shape[1] == 4
        assert labels.shape[0] == bboxes.shape[0]

    # ------------------------------------------------------------------
    # Non-square resolution
    # ------------------------------------------------------------------

    def test_non_square_resolution(self, non_square_yolo_root: Path) -> None:
        """YOLO dataset with non-square (48, 64) images yields correct shapes."""
        dataset = DetectionDataset(
            str(non_square_yolo_root),
            format="yolo",
            use_cache=False,
            bbox_format="xyxy",
        )
        image, bboxes, labels = dataset[0]
        assert image.shape[1] == 48, f"Expected height 48, got {image.shape[1]}"
        assert image.shape[2] == 64, f"Expected width 64, got {image.shape[2]}"
        assert bboxes.shape[1] == 4
        assert labels.shape[0] == bboxes.shape[0]

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_missing_labels_dir(self, tmp_path: Path) -> None:
        """Missing YOLO labels directory raises ``FileNotFoundError``."""
        root: Path = tmp_path / "yolo_no_labels"
        root.mkdir()
        with pytest.raises(FileNotFoundError, match="YOLO labels directory not found"):
            DetectionDataset(str(root), format="yolo", use_cache=False)


# ======================================================================
# DetectionDataset error cases (shared for both formats)
# ======================================================================


class TestDetectionDatasetErrors:
    """Verify error handling in :class:`DetectionDataset`."""

    def test_invalid_format(self, tmp_path: Path) -> None:
        """Unsupported annotation format raises ``ValueError``."""
        root: Path = tmp_path / "det_invalid_format"
        root.mkdir()
        with pytest.raises(ValueError, match="Unsupported annotation format"):
            DetectionDataset(str(root), format="invalid")

    def test_missing_root(self) -> None:
        """Non-existent root raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            DetectionDataset("/nonexistent", format="coco")

    def test_missing_coco_annotation_file(self, tmp_path: Path) -> None:
        """Missing COCO annotation file raises ``FileNotFoundError``."""
        root: Path = tmp_path / "det_missing_ann"
        root.mkdir()
        with pytest.raises(FileNotFoundError, match="COCO annotation file not found"):
            DetectionDataset(str(root), format="coco", use_cache=False)


# ======================================================================
# Boolean Transform Flag Tests (transform=True / transforms=True)
# ======================================================================


class TestBooleanTransformSupport:
    """Verify transform=True / transforms=True boolean flag across datasets."""

    def test_classification_boolean_transform(self, classification_root: Path) -> None:
        """ClassificationDataset with transform=True automatically builds default pipeline."""
        dataset = ClassificationDataset(
            str(classification_root),
            transform=True,
            image_size=(224, 224),
        )
        assert dataset.transform is not None
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)

    def test_detection_boolean_transform(self, detection_coco_root: Path) -> None:
        """DetectionDataset with transforms=True automatically builds default pipeline."""
        dataset = DetectionDataset(
            str(detection_coco_root),
            format="coco",
            transforms=True,
            image_size=(640, 640),
            use_cache=False,
        )
        assert dataset._transform is not None
        image, bboxes, labels = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 640, 640)
        assert bboxes.ndim == 2

    def test_segmentation_boolean_transform(self, segmentation_root: Path) -> None:
        """SegmentationDataset with transform=True automatically builds default pipeline."""
        dataset = SegmentationDataset(
            str(segmentation_root),
            num_classes=10,
            transform=True,
            image_size=(256, 256),
        )
        assert dataset.transforms is not None
        image, mask = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 256, 256)
        assert mask.shape == (256, 256)


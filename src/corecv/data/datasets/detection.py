"""Detection dataset for object detection tasks.

Provides a :class:`DetectionDataset` that supports COCO JSON and YOLO
txt annotation formats with a hybrid cache strategy.  On first
initialisation an O(N) pre-flight check validates all annotations and
writes a persistent cache file (``.npz``) containing compact NumPy
arrays for bounding boxes and labels.  Subsequent runs load directly
from the cache, bypassing annotation-file I/O each epoch.  Cache
validity is determined by comparing file modification timestamps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from corecv.data.transforms import CoordinatedTransform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bbox format conversion helpers
# ---------------------------------------------------------------------------

_ALBUM_FORMATS = frozenset({"pascal_voc", "coco", "yolo"})
"""Supported Albumentations bbox formats."""


def _convert_bbox_format(
    bboxes: np.ndarray,
    src: str,
    dst: str,
    img_w: float,
    img_h: float,
) -> np.ndarray:
    """Convert bounding boxes between coordinate formats.

    Supports the three formats recognised by Albumentations:
    ``"pascal_voc"`` (absolute ``[x_min, y_min, x_max, y_max]``),
    ``"coco"`` (absolute ``[x, y, w, h]``), and ``"yolo"``
    (normalised ``[x_center, y_center, w, h]``).

    Args:
        bboxes: Array of shape ``(N, 4)``.
        src: Source format identifier.
        dst: Destination format identifier.
        img_w: Image width in pixels (used for YOLO<->absolute).
        img_h: Image height in pixels (used for YOLO<->absolute).

    Returns:
        Converted array of shape ``(N, 4)``.
    """
    if src == dst:
        return bboxes.copy()

    # ------------------------------------------------------------------
    # Step 1: normalise everything to pascal_voc (absolute XYXY)
    # ------------------------------------------------------------------
    if src == "pascal_voc":
        xyxy = bboxes.copy()
    elif src == "coco":
        # [x, y, w, h] -> [x1, y1, x2, y2]
        xyxy = np.empty_like(bboxes)
        xyxy[:, 0] = bboxes[:, 0]  # x_min
        xyxy[:, 1] = bboxes[:, 1]  # y_min
        xyxy[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x_max
        xyxy[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y_max
    elif src == "yolo":
        # [cx, cy, w, h] normalised -> [x1, y1, x2, y2] absolute
        half_w = bboxes[:, 2] * img_w * 0.5
        half_h = bboxes[:, 3] * img_h * 0.5
        cx = bboxes[:, 0] * img_w
        cy = bboxes[:, 1] * img_h
        xyxy = np.empty_like(bboxes)
        xyxy[:, 0] = cx - half_w
        xyxy[:, 1] = cy - half_h
        xyxy[:, 2] = cx + half_w
        xyxy[:, 3] = cy + half_h
    else:
        msg = f"Unknown source bbox format: {src!r}. Supported: {sorted(_ALBUM_FORMATS)}"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Step 2: convert from pascal_voc to dst format
    # ------------------------------------------------------------------
    if dst == "pascal_voc":
        return xyxy

    if dst == "coco":
        result = np.empty_like(xyxy)
        result[:, 0] = xyxy[:, 0]  # x
        result[:, 1] = xyxy[:, 1]  # y
        result[:, 2] = xyxy[:, 2] - xyxy[:, 0]  # w
        result[:, 3] = xyxy[:, 3] - xyxy[:, 1]  # h
        return result

    if dst == "yolo":
        result = np.empty_like(xyxy)
        result[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) * 0.5 / img_w  # cx
        result[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) * 0.5 / img_h  # cy
        result[:, 2] = (xyxy[:, 2] - xyxy[:, 0]) / img_w  # w
        result[:, 3] = (xyxy[:, 3] - xyxy[:, 1]) / img_h  # h
        return result

    msg = f"Unknown destination bbox format: {dst!r}. Supported: {sorted(_ALBUM_FORMATS)}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# DetectionDataset
# ---------------------------------------------------------------------------


class DetectionDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Dataset for object detection with COCO and YOLO annotation formats.

    Implements a hybrid cache strategy:

    1.  On first use, an O(N) **pre-flight check** scans every annotation
        file, validates bounding boxes, and writes a persistent cache
        file (``.npz``) containing compact NumPy arrays for bboxes and
        labels.
    2.  On subsequent runs the cache is loaded directly into RAM,
        avoiding per-epoch I/O on annotation files.
    3.  **Cache invalidation** compares stored file modification times
        against the current filesystem mtimes.  If any source file has
        changed the cache is rebuilt.

    Each sample is a ``(image, bboxes, labels)`` tuple where:

    * ``image``  — ``torch.Tensor``, shape ``(C, H, W)``, dtype
      ``torch.float32``, values in ``[0, 1]`` (if untransformed) or
      normalised per the transform pipeline.
    * ``bboxes`` — ``torch.Tensor``, shape ``(N, 4)`` in XYXY format,
      either absolute pixels or normalised ``[0, 1]``.
    * ``labels`` — ``torch.Tensor``, shape ``(N,)`` with class indices.

    Args:
        root: Root directory containing images and (for YOLO) label
            files.
        annotation_path: Path to annotation file (COCO JSON) or label
            directory (YOLO).  If ``None``, inferred from ``root``:
            ``root / "annotations.json"`` for COCO,
            ``root / "labels"`` for YOLO.
        format: Annotation format — ``"coco"`` or ``"yolo"``.
        transform: Optional :class:`CoordinatedTransform` pipeline for
            synchronised image + bbox + label augmentation.  The
            pipeline's ``bbox_format`` should be ``"pascal_voc"``
            (recommended); other formats are converted automatically.
        image_size: Target ``(height, width)`` for image loading.
            Ignored when a ``transform`` is provided (the transform
            config controls resizing).
        cache_dir: Directory for the cache file.  If ``None``, uses
            ``root / ".cache"``.
        use_cache: Whether to enable the on-disk cache.  Pass
            ``False`` to force re-validation on every initialisation.
        bbox_format: Output bounding box format.  ``"xyxy"`` for
            absolute pixel coordinates, ``"norm_xyxy"`` for normalised
            ``[0, 1]`` values.
        class_names: Optional list of class names for COCO.  If
            ``None``, extracted from the COCO JSON categories.

    Raises:
        FileNotFoundError: If the root directory, annotation path, or
            an image referenced in the annotations does not exist.
        ValueError: If an unsupported format is given, or if an
            annotation contains invalid bounding boxes.

    Example:
        >>> from corecv.data.datasets import DetectionDataset
        >>> dataset = DetectionDataset(
        ...     root="data/coco",
        ...     format="coco",
        ...     bbox_format="norm_xyxy",
        ... )
        >>> image, bboxes, labels = dataset[0]
        >>> image.shape
        torch.Size([3, 640, 640])
        >>> bboxes.shape
        torch.Size([3, 4])
        >>> labels.shape
        torch.Size([3])
    """

    CACHE_VERSION: ClassVar[int] = 3
    """Integer version identifier bumped when cache format changes."""

    _MTIME_TOLERANCE: ClassVar[float] = 0.01
    """Maximum absolute difference (seconds) for mtime-based cache validation."""

    SUPPORTED_FORMATS: ClassVar[frozenset[str]] = frozenset({"coco", "yolo"})
    """Annotation formats recognised by this dataset."""

    _IMAGE_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({
        ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
    })
    """Image file extensions that are searched for when matching YOLO labels."""

    def __init__(  # noqa: PLR0913
        self,
        root: str | Path,
        annotation_path: str | Path | None = None,
        format: str = "coco",
        transform: CoordinatedTransform | None = None,
        image_size: tuple[int, int] = (640, 640),
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        bbox_format: Literal["xyxy", "norm_xyxy"] = "norm_xyxy",
        class_names: Sequence[str] | None = None,
    ) -> None:
        """Initialise the detection dataset.

        Args:
            root: Root directory containing images.
            annotation_path: Path to annotation file or label dir.
            format: Annotation format (``"coco"`` or ``"yolo"``).
            transform: Optional transform pipeline.
            image_size: Target ``(height, width)`` for images.
            cache_dir: Directory for cache file.
            use_cache: Whether to use on-disk caching.
            bbox_format: Output bbox format
                (``"xyxy"`` or ``"norm_xyxy"``).
            class_names: Optional list of class names.
        """
        super().__init__()

        self._root: Path = Path(root)
        if not self._root.is_dir():
            msg = f"Root directory does not exist: {self._root}"
            raise FileNotFoundError(msg)

        if format not in self.SUPPORTED_FORMATS:
            msg = (
                f"Unsupported annotation format: {format!r}. "
                f"Supported: {sorted(self.SUPPORTED_FORMATS)}"
            )
            raise ValueError(msg)
        self._format: str = format

        self._transform: CoordinatedTransform | None = transform
        self._image_size: tuple[int, int] = image_size
        self._bbox_format: str = bbox_format
        self._use_cache: bool = use_cache

        # Resolve annotation path
        if annotation_path is not None:
            self._annotation_path: Path = Path(annotation_path)
        elif format == "coco":
            self._annotation_path = self._root / "annotations.json"
        else:
            self._annotation_path = self._root / "labels"

        # Resolve cache directory
        if cache_dir is not None:
            self._cache_dir: Path = Path(cache_dir)
        else:
            self._cache_dir = self._root / ".cache"

        # Derive cache file name from annotation path hash
        self._cache_path: Path = self._cache_dir / self._cache_filename()

        # User-supplied class names (COCO override)
        self._user_class_names: Sequence[str] | None = class_names

        # In-memory state (populated by _initialize)
        self._image_paths: list[Path] = []
        self._bboxes: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._class_names: list[str] = []

        # Original image dimensions (needed for normalisation)
        self._image_dims: list[tuple[int, int]] = []

        self._initialize()

    # ------------------------------------------------------------------
    # Cache file name
    # ------------------------------------------------------------------

    def _cache_filename(self) -> str:
        """Generate a deterministic cache file name.

        Hashes the absolute annotation path so that different datasets
        (different roots or annotation files) produce distinct cache
        files.

        Returns:
            A string like ``"detection_coco_<hash>.npz"``.
        """
        resolved: str = str(self._annotation_path.resolve())
        digest: str = hashlib.md5(resolved.encode()).hexdigest()[:16]
        return f"detection_{self._format}_{digest}.npz"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Load existing cache or build a new one.

        If caching is enabled and a valid cache file exists on disk it
        is loaded directly.  Otherwise a full pre-flight check is run
        and the cache is (re-)built.
        """
        if self._use_cache and self._cache_path.is_file():
            if self._validate_cache():
                logger.info("Loading cache from %s", self._cache_path)
                self._load_cache()
                return
            logger.info("Cache invalid, rebuilding from source.")

        logger.info("Running pre-flight annotation validation ...")
        self._build_cache()

    # ------------------------------------------------------------------
    # Cache validation
    # ------------------------------------------------------------------

    def _validate_cache(self) -> bool:
        """Check whether the on-disk cache is still valid.

        Compares stored file modification timestamps against current
        filesystem mtimes for each annotation source file.  Also
        verifies that the cache version matches.

        Returns:
            ``True`` if the cache is usable, ``False`` otherwise.
        """
        try:
            loaded: dict[str, Any] = dict(
                np.load(self._cache_path, allow_pickle=True)
            )
        except Exception:  # noqa: BLE001
            return False

        required_keys: frozenset[str] = frozenset({
            "image_paths", "num_boxes", "bboxes", "labels", "metadata",
        })
        if not required_keys.issubset(loaded.keys()):
            return False

        metadata: dict[str, Any] = loaded["metadata"].item()
        valid: bool = True

        # Version check
        valid = valid and metadata.get("cache_version") == self.CACHE_VERSION

        # Format check
        valid = valid and metadata.get("format") == self._format

        # Bbox format check
        valid = valid and metadata.get("bbox_format") == self._bbox_format

        if not valid:
            return False

        # Mtime check for every tracked annotation file
        stored_mtimes: dict[str, float] = metadata.get("annotation_mtimes", {})
        for path_str, stored_mtime in stored_mtimes.items():
            p: Path = Path(path_str)
            if not p.is_file():
                return False
            current_mtime: float = p.stat().st_mtime
            if abs(current_mtime - stored_mtime) > self._MTIME_TOLERANCE:
                return False

        return True

    # ------------------------------------------------------------------
    # Cache loading
    # ------------------------------------------------------------------

    def _load_cache(self) -> None:
        """Load annotations from the on-disk cache into RAM.

        Reconstructs per-sample bounding-box and label arrays from the
        flat concatenated storage using the ``num_boxes`` index array.
        """
        loaded: dict[str, Any] = dict(
            np.load(self._cache_path, allow_pickle=True)
        )

        # Image paths
        raw_paths: np.ndarray = loaded["image_paths"]
        self._image_paths = [Path(str(p)) for p in raw_paths]

        # Per-sample arrays
        num_boxes: np.ndarray = loaded["num_boxes"]
        bboxes_flat: np.ndarray = loaded["bboxes"]
        labels_flat: np.ndarray = loaded["labels"]

        end: np.ndarray = np.cumsum(num_boxes)
        start: np.ndarray = np.concatenate([[0], end[:-1]])

        self._bboxes = [
            bboxes_flat[st:en] for st, en in zip(start, end, strict=True)  # noqa: SIM118
        ]
        self._labels = [
            labels_flat[st:en] for st, en in zip(start, end, strict=True)  # noqa: SIM118
        ]

        # Class names
        raw_names: np.ndarray = loaded.get("class_names", np.array([]))
        self._class_names = (
            list(raw_names) if raw_names.size > 0 else []
        )

        # Image dimensions
        if "image_dims" in loaded:
            dims_data: np.ndarray = loaded["image_dims"]
            self._image_dims = [
                (int(dims_data[i, 0]), int(dims_data[i, 1]))
                for i in range(dims_data.shape[0])
            ]
        else:
            self._image_dims = [(0, 0)] * len(self._image_paths)

        logger.info(
            "Loaded %d samples (%d boxes) from cache.",
            len(self._image_paths),
            int(num_boxes.sum()),
        )

    # ------------------------------------------------------------------
    # Cache building (pre-flight check)
    # ------------------------------------------------------------------

    def _build_cache(self) -> None:
        """Run the O(N) pre-flight check and write a new cache file.

        Parses all annotation files, validates bounding boxes, populates
        the in-memory caches, and persists the result to disk as a
        compressed ``.npz`` archive.
        """
        if self._format == "coco":
            self._load_coco_annotations()
        elif self._format == "yolo":
            self._load_yolo_annotations()
        else:
            # Defensive —— the constructor should have already rejected
            # unknown formats.
            msg = f"Unsupported annotation format: {self._format!r}"
            raise ValueError(msg)

        if self._use_cache:
            self._save_cache()

        logger.info(
            "Pre-flight check complete: %d samples, %d classes.",
            len(self._image_paths),
            len(self._class_names),
        )

    # ------------------------------------------------------------------
    # COCO format loader
    # ------------------------------------------------------------------

    def _load_coco_annotations(self) -> None:
        """Parse a COCO JSON annotation file and populate in-memory caches.

        Reads ``images``, ``annotations``, and ``categories`` arrays
        from the JSON file, validates every bounding box, and stores
        the results in the instance lists.

        Raises:
            FileNotFoundError: If the annotation file or any referenced
                image does not exist.
            ValueError: If a bounding box has invalid geometry.
        """
        ann_path: Path = self._annotation_path
        if not ann_path.is_file():
            msg = f"COCO annotation file not found: {ann_path}"
            raise FileNotFoundError(msg)

        with ann_path.open("r") as f:
            coco: dict[str, Any] = json.load(f)

        # Category id → index mapping
        cat_id_to_idx: dict[int, int] = self._build_coco_category_map(coco)

        # Image lookup
        image_id_to_info: dict[int, dict[str, Any]] = {}
        for img in coco.get("images", []):
            image_id_to_info[img["id"]] = img

        # Group annotations by image id (default to empty list)
        image_id_to_anns: dict[int, list[dict[str, Any]]] = {}
        for img in coco.get("images", []):
            image_id_to_anns[img["id"]] = []
        for ann in coco.get("annotations", []):
            image_id_to_anns.setdefault(ann["image_id"], []).append(ann)

        # Process each image (sorted for determinism)
        sorted_ids: list[int] = sorted(image_id_to_anns.keys())
        for img_id in sorted_ids:
            self._process_coco_image(
                img_id, image_id_to_info, image_id_to_anns, cat_id_to_idx,
            )

    def _build_coco_category_map(
        self,
        coco: dict[str, Any],
    ) -> dict[int, int]:
        """Build a mapping from COCO category id to zero-based index.

        Args:
            coco: Parsed COCO JSON dictionary.

        Returns:
            Dict mapping category id → class index.
        """
        cat_id_to_idx: dict[int, int] = {}
        if self._user_class_names is not None:
            self._class_names = list(self._user_class_names)
            for idx in range(len(self._user_class_names)):
                cat_id_to_idx[idx] = idx
        else:
            for idx, cat in enumerate(coco.get("categories", [])):
                cat_id_to_idx[cat["id"]] = idx
                self._class_names.append(cat["name"])
        return cat_id_to_idx

    def _process_coco_image(
        self,
        img_id: int,
        image_id_to_info: dict[int, dict[str, Any]],
        image_id_to_anns: dict[int, list[dict[str, Any]]],
        cat_id_to_idx: dict[int, int],
    ) -> None:
        """Validate and store annotations for a single COCO image.

        Args:
            img_id: COCO image id.
            image_id_to_info: Mapping from image id to image metadata.
            image_id_to_anns: Mapping from image id to annotations.
            cat_id_to_idx: Mapping from category id to class index.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If a bounding box has invalid geometry.
        """
        img_info: dict[str, Any] = image_id_to_info[img_id]
        file_name: str = img_info["file_name"]
        img_path: Path = self._root / file_name
        if not img_path.is_file():
            msg = f"Image referenced in COCO JSON not found: {img_path}"
            raise FileNotFoundError(msg)

        img_w: float = float(img_info["width"])
        img_h: float = float(img_info["height"])

        boxes: list[list[float]] = []
        lbls: list[int] = []
        eps: float = 1.0

        for ann in image_id_to_anns[img_id]:
            x = float(ann["bbox"][0])
            y = float(ann["bbox"][1])
            w = float(ann["bbox"][2])
            h = float(ann["bbox"][3])
            cat_id: int = int(ann["category_id"])

            if cat_id not in cat_id_to_idx:
                continue

            if w <= 0 or h <= 0:
                msg = (
                    f"Non-positive bbox dimensions ({w:.2f}, {h:.2f}) "
                    f"in {file_name} (category {cat_id})"
                )
                raise ValueError(msg)

            x1: float = x
            y1: float = y
            x2: float = x + w
            y2: float = y + h

            if x1 < -eps or y1 < -eps or x2 > img_w + eps or y2 > img_h + eps:
                msg = (
                    f"Bbox [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
                    f"exceeds image dimensions ({img_w:.0f}, {img_h:.0f}) "
                    f"in {file_name}"
                )
                raise ValueError(msg)

            boxes.append([x1, y1, x2, y2])
            lbls.append(cat_id_to_idx[cat_id])

        self._image_paths.append(img_path)
        self._bboxes.append(
            np.array(boxes, dtype=np.float64) if boxes
            else np.empty((0, 4), dtype=np.float64)
        )
        self._labels.append(
            np.array(lbls, dtype=np.int64) if lbls
            else np.empty((0,), dtype=np.int64)
        )
        self._image_dims.append((int(img_w), int(img_h)))

    # ------------------------------------------------------------------
    # YOLO format loader
    # ------------------------------------------------------------------

    _YOLO_EXPECTED_FIELDS: ClassVar[int] = 5
    """Number of space-separated fields per YOLO label line."""

    def _load_yolo_annotations(self) -> None:
        """Parse YOLO-format label files (``.txt``) and populate caches.

        Scans the labels directory for ``.txt`` files, matches each to
        a corresponding image by file stem, reads image dimensions to
        convert YOLO normalised coordinates to absolute pixel values,
        validates every box, and populates the in-memory caches.

        Raises:
            FileNotFoundError: If the labels directory or an image
                matching a label file does not exist.
            ValueError: If a label line is malformed or has out-of-range
                coordinates.
        """
        labels_dir: Path = self._annotation_path
        if not labels_dir.is_dir():
            msg = f"YOLO labels directory not found: {labels_dir}"
            raise FileNotFoundError(msg)

        label_files: list[Path] = sorted(labels_dir.glob("*.txt"))
        if not label_files:
            msg = f"No YOLO label files (*.txt) found in {labels_dir}"
            raise FileNotFoundError(msg)

        # Optional classes.txt
        classes_path: Path = self._root / "classes.txt"
        if classes_path.is_file():
            with classes_path.open("r") as f:
                self._class_names = [
                    raw.strip() for raw in f if raw.strip()
                ]

        for label_path in label_files:
            self._process_yolo_label_file(label_path)

    def _find_yolo_image(self, label_path: Path) -> Path:
        """Locate the image file corresponding to a YOLO label file.

        Args:
            label_path: Path to the ``.txt`` label file.

        Returns:
            Path to the matching image file.

        Raises:
            FileNotFoundError: If no matching image is found.
        """
        stem: str = label_path.stem
        for ext in sorted(self._IMAGE_EXTENSIONS):
            candidate: Path = label_path.with_suffix(ext)
            if candidate.is_file():
                return candidate
            candidate = self._root / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
        msg = f"No matching image found for label file: {label_path.name}"
        raise FileNotFoundError(msg)

    def _parse_yolo_line(
        self,
        line: str,
        label_path: Path,
        line_num: int,
    ) -> tuple[int, float, float, float, float] | None:
        """Parse a single YOLO annotation line.

        Args:
            line: Raw line from a YOLO label file.
            label_path: Path to the label file (for error messages).
            line_num: Line number (for error messages).

        Returns:
            A tuple ``(class_id, cx, cy, w, h)`` or ``None`` if the
            line is blank or a comment.

        Raises:
            ValueError: If the line is malformed or coordinates are
                out of range.
        """
        stripped: str = line.strip()
        if not stripped or stripped.startswith("#"):
            return None

        parts: list[str] = stripped.split()
        if len(parts) != self._YOLO_EXPECTED_FIELDS:
            msg = (
                f"Invalid YOLO annotation in {label_path.name}:{line_num}. "
                f"Expected {self._YOLO_EXPECTED_FIELDS} values, "
                f"got {len(parts)}."
            )
            raise ValueError(msg)

        try:
            class_id: int = int(parts[0])
            cx: float = float(parts[1])
            cy: float = float(parts[2])
            nw: float = float(parts[3])
            nh: float = float(parts[4])
        except ValueError as exc:
            msg = (
                f"Non-numeric value in {label_path.name}:{line_num}: {exc}"
            )
            raise ValueError(msg) from exc

        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            msg = (
                f"YOLO centre out of [0, 1] in {label_path.name}:{line_num}: "
                f"({cx:.4f}, {cy:.4f})"
            )
            raise ValueError(msg)
        if nw <= 0 or nh <= 0 or nw > 1.0 or nh > 1.0:
            msg = (
                f"YOLO dimensions out of range in {label_path.name}:{line_num}: "
                f"({nw:.4f}, {nh:.4f})"
            )
            raise ValueError(msg)

        return class_id, cx, cy, nw, nh

    def _process_yolo_label_file(self, label_path: Path) -> None:
        """Validate and store annotations from a single YOLO label file.

        Args:
            label_path: Path to the ``.txt`` label file.

        Raises:
            FileNotFoundError: If the matching image file is not found.
            ValueError: If annotations are malformed or out of range.
        """
        img_path: Path = self._find_yolo_image(label_path)

        img_cv: np.ndarray | None = cv2.imread(str(img_path))
        if img_cv is None:
            msg = f"Failed to read image (corrupt?): {img_path}"
            raise ValueError(msg)
        img_h: int
        img_w: int
        img_h, img_w = img_cv.shape[:2]

        boxes: list[list[float]] = []
        lbls: list[int] = []

        with label_path.open("r") as f:
            for line_num, raw_line in enumerate(f, 1):
                parsed = self._parse_yolo_line(raw_line, label_path, line_num)
                if parsed is None:
                    continue

                class_id, cx, cy, nw, nh = parsed

                # YOLO (cx, cy, w, h) normalized → absolute XYXY
                half_w: float = nw * img_w * 0.5
                half_h: float = nh * img_h * 0.5
                x1_abs: float = cx * img_w - half_w
                y1_abs: float = cy * img_h - half_h
                x2_abs: float = cx * img_w + half_w
                y2_abs: float = cy * img_h + half_h

                boxes.append([x1_abs, y1_abs, x2_abs, y2_abs])
                lbls.append(class_id)

        self._image_paths.append(img_path)
        self._bboxes.append(
            np.array(boxes, dtype=np.float64) if boxes
            else np.empty((0, 4), dtype=np.float64)
        )
        self._labels.append(
            np.array(lbls, dtype=np.int64) if lbls
            else np.empty((0,), dtype=np.int64)
        )
        self._image_dims.append((img_w, img_h))

    # ------------------------------------------------------------------
    # Persist cache to disk
    # ------------------------------------------------------------------

    def _save_cache(self) -> None:
        """Write the current annotation cache to disk as a compressed .npz.

        Stores flat concatenated bbox/label arrays alongside an index
        (``num_boxes``) for per-sample reconstruction.  Metadata
        includes file modification times for cache invalidation.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Collect mtimes of all source annotation files
        annotation_mtimes: dict[str, float] = {}
        if self._format == "coco":
            ann_resolved: Path = self._annotation_path.resolve()
            annotation_mtimes[str(ann_resolved)] = ann_resolved.stat().st_mtime
        elif self._format == "yolo":
            ann_dir_resolved: Path = self._annotation_path.resolve()
            if ann_dir_resolved.is_dir():
                annotation_mtimes[str(ann_dir_resolved)] = ann_dir_resolved.stat().st_mtime
                for lbl_file in sorted(ann_dir_resolved.glob("*.txt")):
                    annotation_mtimes[str(lbl_file.resolve())] = lbl_file.stat().st_mtime

        # Flatten arrays
        num_boxes_arr: np.ndarray = np.array(
            [len(b) for b in self._bboxes], dtype=np.int64
        )
        bboxes_flat: np.ndarray = (
            np.concatenate(self._bboxes, axis=0).astype(np.float32)
            if self._bboxes
            else np.empty((0, 4), dtype=np.float32)
        )
        labels_flat: np.ndarray = (
            np.concatenate(self._labels, axis=0).astype(np.int64)
            if self._labels
            else np.empty((0,), dtype=np.int64)
        )

        # Image dimensions as (N, 2) array
        dims_arr: np.ndarray = np.array(self._image_dims, dtype=np.int32)

        metadata: dict[str, Any] = {
            "cache_version": self.CACHE_VERSION,
            "format": self._format,
            "annotation_mtimes": annotation_mtimes,
            "bbox_format": self._bbox_format,
            "created_at": time.time(),
        }

        np.savez_compressed(
            self._cache_path,
            image_paths=np.array(
                [str(p) for p in self._image_paths], dtype=object
            ),
            num_boxes=num_boxes_arr,
            bboxes=bboxes_flat,
            labels=labels_flat,
            class_names=np.array(self._class_names, dtype=object),
            image_dims=dims_arr,
            metadata=np.array(metadata, dtype=object),
        )

        logger.info("Cache saved to %s", self._cache_path)

    # ------------------------------------------------------------------
    # Item access
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return the ``(image, bboxes, labels)`` tuple at *index*.

        Loads the image from disk and retrieves cached bounding boxes
        and labels from the in-memory cache.

        Args:
            index: Sample index.

        Returns:
            A tuple ``(image, bboxes, labels)`` where

            - ``image`` has shape ``(C, H, W)``.
            - ``bboxes`` has shape ``(N, 4)`` in XYXY format.
            - ``labels`` has shape ``(N,)`` with integer class indices.
        """
        img_path: Path = self._image_paths[index]
        image: np.ndarray = cv2.imread(str(img_path))
        if image is None:
            msg = f"Failed to read image at index {index}: {img_path}"
            raise ValueError(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Retrieve cached bboxes (internal format: absolute XYXY)
        bboxes: np.ndarray = self._bboxes[index].copy()
        labels: np.ndarray = self._labels[index].copy()

        img_h: int
        img_w: int
        img_h, img_w = image.shape[:2]

        # ------------------------------------------------------------------
        # Apply transforms if available
        # ------------------------------------------------------------------
        if self._transform is not None:
            # Determine expected bbox format from the transform config
            expected_fmt: str = "pascal_voc"  # default / recommended
            if hasattr(self._transform, "config"):
                cfg: Any = self._transform.config
                if hasattr(cfg, "bbox_format"):
                    expected_fmt = cfg.bbox_format  # type: ignore[union-attr]

            # Convert internal (absolute XYXY) to the pipeline's format
            tf_bboxes: np.ndarray = _convert_bbox_format(
                bboxes, "pascal_voc", expected_fmt, float(img_w), float(img_h),
            )

            transformed = self._transform(
                image=image,
                bboxes=tf_bboxes.tolist() if len(tf_bboxes) > 0 else None,
                labels=labels.tolist() if len(labels) > 0 else None,
            )

            image = transformed.image  # (H, W, C), values depend on pipeline
            out_bboxes: np.ndarray = np.array(
                transformed.bboxes, dtype=np.float32
            )
            out_labels: np.ndarray = np.array(
                transformed.labels, dtype=np.int64
            )

            # Convert back to absolute XYXY if needed
            if expected_fmt != "pascal_voc":
                out_bboxes = _convert_bbox_format(
                    out_bboxes,
                    expected_fmt,
                    "pascal_voc",
                    float(image.shape[1]),
                    float(image.shape[0]),
                )

            # Determine output image dimensions for normalisation
            out_h: int
            out_w: int
            out_h, out_w = image.shape[:2]
        else:
            # Manual normalisation (no transform pipeline)
            image = image.astype(np.float32) / 255.0
            out_bboxes = bboxes.astype(np.float32)
            out_labels = labels.astype(np.int64)
            out_h, out_w = img_h, img_w

        # ------------------------------------------------------------------
        # Convert to output bbox format
        # ------------------------------------------------------------------
        if self._bbox_format == "norm_xyxy":
            out_bboxes = (
                out_bboxes / [float(out_w), float(out_h), float(out_w), float(out_h)]
            ).astype(np.float32)

        # Convert image to tensor (C, H, W)
        image_tensor: Tensor = (
            torch.from_numpy(image).permute(2, 0, 1).float()
        )
        bboxes_tensor: Tensor = torch.from_numpy(out_bboxes)
        labels_tensor: Tensor = torch.from_numpy(out_labels)

        return image_tensor, bboxes_tensor, labels_tensor

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Total sample count.
        """
        return len(self._image_paths)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> list[str]:
        """Return the list of class name strings.

        Returns:
            Ordered list of class names (index = class id).
        """
        return list(self._class_names)

    @property
    def num_classes(self) -> int:
        """Return the number of unique classes.

        Returns:
            Class count.
        """
        return len(self._class_names)

    @property
    def image_paths(self) -> list[Path]:
        """Return the list of image file paths.

        Returns:
            Ordered list of paths, one per sample.
        """
        return list(self._image_paths)

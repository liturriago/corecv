"""Detection dataset with multi-format label loading and Albumentations transforms."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Literal

import albumentations as A  # noqa: N812
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

LabelFormat = Literal["folder", "csv", "json"]
BBoxFormat = Literal["xyxy", "yolo"]

# Candidate column names for dynamic CSV column detection.
_IMAGE_COLUMNS: tuple[str, ...] = ("filename", "image", "image_id", "filepath", "file_name")
_CLASS_COLUMNS: tuple[str, ...] = ("label", "class_id", "class", "target", "category")
_XYXY_COLUMNS: tuple[str, ...] = ("x1", "y1", "x2", "y2")
_YOLO_COLUMNS: tuple[str, ...] = ("x_center", "y_center", "width", "height")

# Number of coordinates per box.
_BBOX_FIELDS = 4

# Number of fields per annotation: class id plus four box coordinates.
_ANNOTATION_FIELDS = 5


def _dense_labels(labels_list: list[list[int]]) -> list[int]:
    """Return the distinct labels of a dataset in first-seen order."""
    distinct: list[int] = []
    for labels in labels_list:
        for label in labels:
            if label not in distinct:
                distinct.append(label)
    return distinct


def _find_column(headers: list[str], candidates: tuple[str, ...]) -> str:
    """Find the first candidate column present in the CSV headers."""
    for candidate in candidates:
        if candidate in headers:
            return candidate
    msg = f"CSV must have a column among: {', '.join(candidates)}"
    raise ValueError(msg)


def _find_bbox_columns(headers: list[str], candidates: tuple[str, ...]) -> list[str]:
    """Find all four box columns in the CSV headers."""
    found = [candidate for candidate in candidates if candidate in headers]
    if len(found) != _BBOX_FIELDS:
        msg = f"CSV must have bbox columns among: {', '.join(candidates)}"
        raise ValueError(msg)
    return found


class DetectionDataset(Dataset):
    """Efficient PyTorch Dataset for Object Detection.

    Loads image and bounding-box annotation pairs from multiple annotation
    formats and applies Albumentations transforms jointly to images and
    boxes so that spatial augmentations (crops, flips, affine) stay in sync.

    Supported formats:

    - ``folder``: YOLO-style sidecar ``.txt`` annotations, one line per
      object: ``class_id x_center y_center width height`` (normalized).
    - ``csv``: annotation file with image, class, and box columns.
    - ``json``: annotation file mapping filenames to lists of
      ``[class_id, x1, y1, x2, y2]`` (or normalized boxes when *bbox_format*
      is ``yolo``).

    Each sample yields a ``"targets"`` tensor of shape ``(G, 5)`` with
    columns ``[class, x1, y1, x2, y2]`` in image coordinates.
    """

    def __init__(  # noqa: PLR0913
        self,
        img_dir: str | Path,
        label_format: LabelFormat,
        ann_source: str | Path | None = None,
        img_size: tuple[int, int] = (640, 640),
        *,
        augment: bool = True,
        bbox_format: BBoxFormat = "xyxy",
        num_classes: int | None = None,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize the detection dataset.

        Args:
            img_dir: Directory containing images.
            label_format: One of 'folder', 'csv', or 'json'.
            ann_source: Annotation source (CSV or JSON file). Required for
                the ``csv``/``json`` formats.
            img_size: Target (height, width) for resizing.
            augment: Whether to apply data augmentation.
            bbox_format: Box encoding of the annotation source. One of
                ``xyxy`` (absolute pixel ``[x1, y1, x2, y2]``) or ``yolo``
                (normalized ``[x_center, y_center, width, height]``). The
                ``folder`` format always uses ``yolo``.
            num_classes: Optional number of classes. When provided, samples
                whose labels exceed this range raise an error.
            mean: Per-channel mean for image normalization.
            std: Per-channel std for image normalization.

        """
        super().__init__()
        self.img_dir = Path(img_dir)
        self.ann_source = Path(ann_source) if ann_source is not None else None
        self.label_format = label_format.lower()
        self.img_size = img_size
        self.augment = augment
        self.num_classes = num_classes
        self.bbox_format = "yolo" if label_format == "folder" else bbox_format

        self.classes: list[Any] = []
        self.samples: list[tuple[Path, list[list[float]], list[int]]] = []

        self._load_samples()

        if not self.samples:
            msg = f"No valid samples found in directory: {self.img_dir}"
            raise FileNotFoundError(msg)

        self.transform = self._build_transforms(mean, std)

    def _load_samples(self) -> None:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if self.label_format == "folder":
            self._load_from_folder(valid_exts)
        elif self.label_format == "csv":
            self._load_from_csv(valid_exts)
        elif self.label_format == "json":
            self._load_from_json(valid_exts)
        else:
            msg = f"Unsupported label_format: {self.label_format}"
            raise ValueError(msg)

    def _load_from_folder(self, valid_exts: set[str]) -> None:
        """Load YOLO-style image and sidecar ``.txt`` annotation pairs."""
        images = sorted(
            [
                f
                for f in self.img_dir.iterdir()
                if f.is_file() and f.suffix.lower() in valid_exts
            ],
        )

        if not images:
            msg = f"No images found in: {self.img_dir}"
            raise FileNotFoundError(msg)

        for img_path in images:
            ann_path = self.img_dir / (img_path.stem + ".txt")
            if not ann_path.is_file():
                logger.warning("No annotation file found for: %s", img_path)
                continue
            boxes, labels = self._parse_yolo_txt(ann_path)
            if not boxes:
                logger.warning("No valid objects in annotation: %s", ann_path)
                continue
            self.samples.append((img_path, boxes, labels))

        if self.num_classes is not None and self.samples:
            max_label = max(max(labels) for _, _, labels in self.samples)
            if max_label >= self.num_classes:
                msg = f"annotation label {max_label} exceeds num_classes={self.num_classes}"
                raise ValueError(msg)

    @staticmethod
    def _parse_yolo_txt(ann_path: Path) -> tuple[list[list[float]], list[int]]:
        """Parse a YOLO-style text annotation into boxes and labels."""
        boxes: list[list[float]] = []
        labels: list[int] = []
        for raw_line in ann_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != _ANNOTATION_FIELDS:
                logger.warning("Skipping malformed annotation line in %s: %s", ann_path, line)
                continue
            cls_id, cx, cy, width, height = parts
            width = float(width)
            height = float(height)
            if width <= 0 or height <= 0:
                continue
            boxes.append([float(cx), float(cy), width, height])
            labels.append(int(cls_id))
        return boxes, labels

    def _load_from_csv(self, valid_exts: set[str]) -> None:
        """Parse detection samples from a CSV file with dynamic column detection."""
        if not self.ann_source.is_file():
            msg = f"CSV annotation file not found: {self.ann_source}"
            raise FileNotFoundError(msg)

        with self.ann_source.open() as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if headers is None:
                msg = "CSV file is empty or has no headers"
                raise ValueError(msg)
            rows = list(reader)

        img_col = _find_column(headers, _IMAGE_COLUMNS)
        class_col = _find_column(headers, _CLASS_COLUMNS)
        box_candidates = _YOLO_COLUMNS if self.bbox_format == "yolo" else _XYXY_COLUMNS
        box_cols = _find_bbox_columns(headers, box_candidates)

        class_set: list[Any] = []
        for row in rows:
            if row[class_col] not in class_set:
                class_set.append(row[class_col])
        self.classes = class_set
        class_to_idx = {name: idx for idx, name in enumerate(class_set)}
        self._validate_class_count(len(class_set))

        samples_by_image: dict[str, tuple[list[list[float]], list[int]]] = {}
        for row in rows:
            filename = row[img_col]
            box = [float(row[col]) for col in box_cols]
            label = class_to_idx[row[class_col]]
            boxes, labels = samples_by_image.setdefault(filename, ([], []))
            boxes.append(box)
            labels.append(label)

        for filename, (boxes, labels) in samples_by_image.items():
            self._add_valid_sample(self.img_dir / filename, boxes, labels, valid_exts, "CSV")

    def _load_from_json(self, valid_exts: set[str]) -> None:
        """Parse detection samples from a JSON annotation file."""
        if not self.ann_source.is_file():
            msg = f"JSON annotation file not found: {self.ann_source}"
            raise FileNotFoundError(msg)

        with self.ann_source.open() as f:
            data = json.load(f)

        if not isinstance(data, dict):
            msg = "JSON annotation must be a dict mapping filenames to box lists"
            raise TypeError(msg)

        entries: dict[str, tuple[list[list[float]], list[int]]] = {}
        for filename, anns in data.items():
            entries[filename] = self._parse_json_annotations(anns, filename)

        distinct = _dense_labels([labels for _, labels in entries.values()])
        self.classes = distinct
        class_to_idx = {label: idx for idx, label in enumerate(distinct)}
        self._validate_class_count(len(distinct))

        for filename, (boxes, labels) in entries.items():
            remapped = [class_to_idx[label] for label in labels]
            self._add_valid_sample(self.img_dir / filename, boxes, remapped, valid_exts, "JSON")

    @staticmethod
    def _parse_json_annotations(anns: object, filename: str) -> tuple[list[list[float]], list[int]]:
        """Parse a single JSON annotation list into boxes and labels."""
        if not isinstance(anns, list):
            msg = f"JSON annotation for {filename!r} must be a list"
            raise TypeError(msg)

        boxes: list[list[float]] = []
        labels: list[int] = []
        for ann in anns:
            if len(ann) != _ANNOTATION_FIELDS:
                logger.warning("Skipping malformed annotation for %s: %s", filename, ann)
                continue
            boxes.append([float(value) for value in ann[1:]])
            labels.append(int(ann[0]))
        return boxes, labels

    def _validate_class_count(self, num_labels: int) -> None:
        """Validate the number of classes against *num_classes* when provided."""
        if self.num_classes is not None and num_labels > self.num_classes:
            msg = f"dataset has {num_labels} classes but num_classes={self.num_classes}"
            raise ValueError(msg)

    def _add_valid_sample(
        self,
        img_path: Path,
        boxes: list[list[float]],
        labels: list[int],
        valid_exts: set[str],
        source: str,
    ) -> None:
        """Append a sample if the image exists and has valid boxes."""
        if img_path.suffix.lower() not in valid_exts:
            return
        if not img_path.is_file():
            logger.warning("Skipping missing image referenced in %s: %s", source, img_path)
            return

        valid: list[tuple[list[float], int]] = [
            (box, label)
            for box, label in zip(boxes, labels, strict=False)
            if self._valid_box(box)
        ]
        if not valid:
            logger.warning("No valid boxes for: %s", img_path)
            return

        valid_boxes = [box for box, _ in valid]
        valid_labels = [label for _, label in valid]
        self.samples.append((img_path, valid_boxes, valid_labels))

    def _valid_box(self, box: list[float]) -> bool:
        """Return whether a box has positive extent in its source format."""
        if self.bbox_format == "xyxy":
            x1, y1, x2, y2 = box
            return x2 > x1 and y2 > y1
        _cx, _cy, width, height = box
        return width > 0 and height > 0

    def _to_xyxy(
        self,
        boxes: list[list[float]],
        image_shape: tuple[int, int, int],
    ) -> list[list[float]]:
        """Convert boxes from the source format to absolute pixel ``xyxy``."""
        if self.bbox_format == "xyxy":
            return boxes
        height, width = image_shape[:2]
        converted: list[list[float]] = []
        for cx, cy, bw, bh in boxes:
            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            x2 = (cx + bw / 2) * width
            y2 = (cy + bh / 2) * height
            converted.append([x1, y1, x2, y2])
        return converted

    def _build_transforms(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> A.Compose:
        h, w = self.img_size[0], self.img_size[1]

        if self.augment:
            transforms_list = [
                A.RandomResizedCrop(size=(h, w), scale=(0.8, 1.0), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.0625, 0.0625),
                    rotate=(-15, 15),
                    p=0.5,
                ),
                A.Resize(height=h, width=w),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        else:
            transforms_list = [
                A.Resize(height=h, width=w),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]

        return A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.0,
            ),
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and return the image, targets, and index at the given position.

        Returns:
            A dict with ``"images"`` (transformed image tensor of shape
            ``(C, H, W)``), ``"targets"`` (boxes tensor of shape ``(G, 5)``
            with columns ``[class, x1, y1, x2, y2]``), and ``"image_ids"``
            (scalar dataset index of this sample).

        """
        img_path, raw_boxes, labels = self.samples[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            msg = f"Failed to read image at: {img_path}"
            raise ValueError(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = self._to_xyxy(raw_boxes, image.shape)
        transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
        targets = self._build_targets(transformed["bboxes"], transformed["class_labels"])

        return {
            "images": transformed["image"],
            "targets": targets,
            "image_ids": torch.tensor(idx),
        }

    @staticmethod
    def _build_targets(bboxes: list[list[float]], class_labels: list[int]) -> torch.Tensor:
        """Build a per-image targets tensor from transformed boxes and labels."""
        if not bboxes:
            return torch.zeros(0, 5, dtype=torch.float32)
        labels_t = torch.tensor(class_labels, dtype=torch.long).unsqueeze(1)
        boxes_t = torch.tensor(bboxes, dtype=torch.float32)
        return torch.cat([labels_t, boxes_t], dim=1)


def detection_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    """Collate a list of detection samples into a batched dictionary.

    Prepend a batch-index column to each sample's ``(G, 5)`` targets, yielding
    a flat ``(N, 6)`` targets tensor with columns
    ``[batch_index, class, x1, y1, x2, y2]``.

    """
    images = torch.stack([item["images"] for item in batch], dim=0)

    all_targets: list[torch.Tensor] = []
    for batch_idx, item in enumerate(batch):
        per_image = item["targets"]
        batch_col = torch.full((per_image.shape[0], 1), float(batch_idx))
        all_targets.append(torch.cat([batch_col, per_image], dim=1))
    targets = torch.cat(all_targets, dim=0) if all_targets else torch.zeros(0, 6)

    image_ids = torch.stack([item["image_ids"] for item in batch], dim=0)
    return {"images": images, "targets": targets, "image_ids": image_ids}


def create_detection_dataloader(  # noqa: PLR0913
    img_dir: str | Path,
    label_format: LabelFormat,
    ann_source: str | Path | None = None,
    batch_size: int = 8,
    img_size: tuple[int, int] = (640, 640),
    num_workers: int = 4,
    *,
    augment: bool = True,
    bbox_format: BBoxFormat = "xyxy",
    num_classes: int | None = None,
    shuffle: bool | None = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for object detection.

    Args:
        img_dir: Directory containing images.
        label_format: One of 'folder', 'csv', or 'json'.
        ann_source: Annotation source path (CSV or JSON).
        batch_size: Number of samples per batch.
        img_size: Target (height, width) for resizing.
        num_workers: Number of subprocesses for data loading.
        augment: Whether to apply data augmentation.
        bbox_format: Box encoding of the annotation source. One of ``xyxy``
            or ``yolo``. The ``folder`` format always uses ``yolo``.
        num_classes: Optional number of classes for label validation.
        shuffle: Whether to shuffle samples each epoch. When ``None``
            (default), defaults to *augment* (i.e. shuffled for training,
            ordered for evaluation). Pass ``shuffle=False`` explicitly for an
            ordered evaluation dataloader that still uses augmentation.
        pin_memory: Whether to pin batch memory for faster GPU transfers.

    Returns:
        A PyTorch DataLoader for object detection.

    """
    dataset = DetectionDataset(
        img_dir=img_dir,
        label_format=label_format,
        ann_source=ann_source,
        img_size=img_size,
        augment=augment,
        bbox_format=bbox_format,
        num_classes=num_classes,
    )

    if shuffle is None:
        shuffle = augment

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=pin_memory,
    )

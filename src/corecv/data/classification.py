"""Classification dataset with multi-format label loading and Albumentations transforms."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Literal

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

LabelFormat = Literal["folder", "csv", "json"]


class ClassificationDataset(Dataset):
    """Efficient PyTorch Dataset for Image Classification.

    Supports multiple label source formats and Albumentations transforms.
    """

    def __init__(
        self,
        img_dir: str | Path,
        ann_source: str | Path | None,
        label_format: LabelFormat,
        img_size: tuple[int, int] = (224, 224),
        augment: bool = True,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize the classification dataset.

        Args:
            img_dir: Directory containing images (or class subdirectories for folder format).
            ann_source: Annotation source (folder, CSV file, or JSON file).
            label_format: One of 'folder', 'csv', or 'json'.
            img_size: Target (height, width) for resizing.
            augment: Whether to apply data augmentation.
            mean: Per-channel mean for normalization.
            std: Per-channel std for normalization.
        """
        super().__init__()
        self.img_dir = Path(img_dir)
        self.ann_source = Path(ann_source) if ann_source is not None else None
        self.label_format = label_format.lower()
        self.img_size = img_size
        self.augment = augment

        self.classes: list[str] = []
        self.samples: list[tuple[Path, int]] = []

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

    def _load_from_folder(self, valid_exts: set) -> None:
        class_dirs = sorted(
            [d for d in self.img_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        )

        if not class_dirs:
            msg = f"No class subdirectories found in: {self.img_dir}"
            raise FileNotFoundError(msg)

        self.classes = [d.name for d in class_dirs]
        class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        for class_dir in class_dirs:
            label = class_to_idx[class_dir.name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in valid_exts:
                    self.samples.append((img_path, label))

    def _load_from_csv(self, valid_exts: set) -> None:
        """Parse classification samples from a CSV file with dynamic column detection."""
        if not self.ann_source.is_file():
            msg = f"CSV annotation file not found: {self.ann_source}"
            raise FileNotFoundError(msg)

        label_set: dict[str, int] = {}

        with self.ann_source.open() as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if headers is None:
                msg = "CSV file is empty or has no headers"
                raise ValueError(msg)

            # Dynamically locate image filename column
            img_col = None
            for candidate in ("filename", "image", "image_id", "filepath", "file_name"):
                if candidate in headers:
                    img_col = candidate
                    break

            if img_col is None:
                msg = "CSV must have a 'filename', 'image', 'image_id', 'filepath', or 'file_name' column"
                raise ValueError(msg)

            # Dynamically locate label column
            label_col = None
            for candidate in ("label", "class_id", "class", "target", "category"):
                if candidate in headers:
                    label_col = candidate
                    break

            if label_col is None:
                msg = "CSV must have a 'label', 'class_id', 'class', 'target', or 'category' column"
                raise ValueError(msg)

            rows = list(reader)

        for row in rows:
            label_val = row[label_col]
            if label_val not in label_set:
                label_set[label_val] = len(label_set)

        self.classes = sorted(label_set.keys(), key=lambda x: label_set[x])
        class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        for row in rows:
            filename = row[img_col]
            img_path = self.img_dir / filename
            self._add_valid_sample(img_path, class_to_idx[row[label_col]], valid_exts, "CSV")

    def _load_from_json(self, valid_exts: set) -> None:
        if not self.ann_source.is_file():
            msg = f"JSON annotation file not found: {self.ann_source}"
            raise FileNotFoundError(msg)

        with self.ann_source.open() as f:
            data = json.load(f)

        if not isinstance(data, dict):
            msg = "JSON annotation must be a dict mapping filenames to class IDs"
            raise TypeError(msg)

        all_labels = sorted(set(data.values()))
        self.classes = [str(lbl) for lbl in all_labels]
        label_to_idx = {lbl: idx for idx, lbl in enumerate(all_labels)}

        for filename, label in data.items():
            self._add_valid_sample(self.img_dir / filename, label_to_idx[label], valid_exts, "JSON")

    def _add_valid_sample(
        self,
        img_path: Path,
        label: int,
        valid_exts: set,
        source: str,
    ) -> None:
        """Append a sample if the image exists and has a supported extension."""
        if img_path.suffix.lower() not in valid_exts:
            return
        if not img_path.is_file():
            logger.warning("Skipping missing image referenced in %s: %s", source, img_path)
            return
        self.samples.append((img_path, label))

    def _build_transforms(
        self, mean: tuple[float, float, float], std: tuple[float, float, float]
    ) -> A.Compose:
        h, w = self.img_size[0], self.img_size[1]

        if self.augment:
            transforms_list = [
                A.RandomResizedCrop(size=(h, w), scale=(0.8, 1.0), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
                A.Affine(
                    scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-15, 15), p=0.5
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

        return A.Compose(transforms_list)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and return the image, label, and index at the given position.

        Returns:
            A dict with ``"images"`` (transformed image tensor of shape
            ``(C, H, W)``), ``"labels"`` (scalar class index), and
            ``"image_ids"`` (scalar dataset index of this sample).

        Note:
            ``"image_ids"`` is the dataset index ``idx``, i.e. the position of
            the sample in :attr:`samples`. It is stable across epochs and
            dataloader workers because DataLoader always calls ``__getitem__``
            with dataset indices; use it to correlate predictions back to
            :attr:`samples`.

        """
        img_path, label = self.samples[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            msg = f"Failed to read image at: {img_path}"
            raise ValueError(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image)

        return {
            "images": transformed["image"],
            "labels": torch.tensor(label, dtype=torch.long),
            "image_ids": torch.tensor(idx),
        }


def classification_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    """Collate a list of classification samples into a batched dictionary."""
    images = torch.stack([item["images"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    image_ids = torch.stack([item["image_ids"] for item in batch], dim=0)

    return {"images": images, "labels": labels, "image_ids": image_ids}


def create_classification_dataloader(
    img_dir: str | Path,
    ann_source: str | Path | None,
    label_format: LabelFormat,
    batch_size: int = 16,
    img_size: tuple[int, int] = (224, 224),
    augment: bool = True,
    num_workers: int = 4,
    *,
    shuffle: bool | None = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for image classification.

    Args:
        img_dir: Directory containing images or class subdirectories.
        ann_source: Annotation source path.
        label_format: One of 'folder', 'csv', or 'json'.
        batch_size: Number of samples per batch.
        img_size: Target (height, width) for resizing.
        augment: Whether to apply data augmentation.
        num_workers: Number of subprocesses for data loading.
        shuffle: Whether to shuffle samples each epoch. When ``None``
            (default), defaults to *augment* (i.e. shuffled for training,
            ordered for evaluation). Pass ``shuffle=False`` explicitly for an
            ordered evaluation dataloader that still uses augmentation.
        pin_memory: Whether to pin batch memory for faster GPU transfers.

    Returns:
        A PyTorch DataLoader for classification.
    """
    dataset = ClassificationDataset(
        img_dir=img_dir,
        ann_source=ann_source,
        label_format=label_format,
        img_size=img_size,
        augment=augment,
    )

    if shuffle is None:
        shuffle = augment

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=classification_collate_fn,
        pin_memory=pin_memory,
    )

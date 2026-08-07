"""Segmentation dataset with multi-format label loading and Albumentations transforms."""

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


class SegmentationDataset(Dataset):
    """Efficient PyTorch Dataset for Semantic Segmentation.

    Loads image and segmentation-mask pairs from multiple annotation formats
    and applies Albumentations transforms jointly to images and masks so that
    spatial augmentations (crops, flips, affine) stay in sync.
    """

    def __init__(
        self,
        img_dir: str | Path,
        mask_dir: str | Path | None,
        ann_source: str | Path | None,
        label_format: LabelFormat,
        img_size: tuple[int, int] = (512, 512),
        augment: bool = True,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize the segmentation dataset.

        Args:
            img_dir: Directory containing images.
            mask_dir: Directory containing masks. Required for the ``folder``
                format; optional base directory for relative mask paths in the
                ``csv``/``json`` formats (falls back to *img_dir*).
            ann_source: Annotation source (CSV or JSON file). Required for the
                ``csv``/``json`` formats.
            label_format: One of 'folder', 'csv', or 'json'.
            img_size: Target (height, width) for resizing.
            augment: Whether to apply data augmentation.
            mean: Per-channel mean for image normalization.
            std: Per-channel std for image normalization.
        """
        super().__init__()
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.ann_source = Path(ann_source) if ann_source is not None else None
        self.label_format = label_format.lower()
        self.img_size = img_size
        self.augment = augment

        self.samples: list[tuple[Path, Path]] = []

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
        if self.mask_dir is None:
            msg = "mask_dir is required when label_format='folder'"
            raise ValueError(msg)

        images = sorted(
            [
                f
                for f in self.img_dir.iterdir()
                if f.is_file() and f.suffix.lower() in valid_exts
            ]
        )

        if not images:
            msg = f"No images found in: {self.img_dir}"
            raise FileNotFoundError(msg)

        for img_path in images:
            mask_path = self._find_mask(img_path)
            if mask_path is None:
                logger.warning("No matching mask found for: %s", img_path)
                continue
            self.samples.append((img_path, mask_path))

    def _find_mask(self, img_path: Path) -> Path | None:
        """Locate the mask for an image by matching its stem in mask_dir."""
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            candidate = self.mask_dir / (img_path.stem + ext)
            if candidate.is_file():
                return candidate
        return None

    def _load_from_csv(self, valid_exts: set) -> None:
        """Parse segmentation samples from a CSV file with dynamic column detection."""
        if not self.ann_source.is_file():
            msg = f"CSV annotation file not found: {self.ann_source}"
            raise FileNotFoundError(msg)

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
                msg = (
                    "CSV must have a 'filename', 'image', 'image_id', "
                    "'filepath', or 'file_name' column"
                )
                raise ValueError(msg)

            # Dynamically locate mask filename column
            mask_col = None
            for candidate in (
                "mask",
                "mask_file",
                "mask_path",
                "mask_filename",
                "label",
                "annotation",
                "target",
            ):
                if candidate in headers:
                    mask_col = candidate
                    break

            if mask_col is None:
                msg = (
                    "CSV must have a 'mask', 'mask_file', 'mask_path', "
                    "'mask_filename', 'label', 'annotation', or 'target' column"
                )
                raise ValueError(msg)

            rows = list(reader)

        mask_base = self.mask_dir or self.img_dir

        for row in rows:
            img_path = self.img_dir / row[img_col]
            mask_path = mask_base / row[mask_col]
            self._add_valid_sample(img_path, mask_path, valid_exts, "CSV")

    def _load_from_json(self, valid_exts: set) -> None:
        if not self.ann_source.is_file():
            msg = f"JSON annotation file not found: {self.ann_source}"
            raise FileNotFoundError(msg)

        with self.ann_source.open() as f:
            data = json.load(f)

        if not isinstance(data, dict):
            msg = "JSON annotation must be a dict mapping filenames to mask filenames"
            raise TypeError(msg)

        mask_base = self.mask_dir or self.img_dir

        for filename, mask_filename in data.items():
            img_path = self.img_dir / filename
            mask_path = mask_base / mask_filename
            self._add_valid_sample(img_path, mask_path, valid_exts, "JSON")

    def _add_valid_sample(
        self,
        img_path: Path,
        mask_path: Path,
        valid_exts: set,
        source: str,
    ) -> None:
        """Append a sample if both image and mask exist with supported extensions."""
        if img_path.suffix.lower() not in valid_exts:
            return
        if mask_path.suffix.lower() not in valid_exts:
            return
        if not img_path.is_file():
            logger.warning("Skipping missing image referenced in %s: %s", source, img_path)
            return
        if not mask_path.is_file():
            logger.warning("Skipping missing mask referenced in %s: %s", source, mask_path)
            return
        self.samples.append((img_path, mask_path))

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
        """Load and return the image, mask, and index at the given position.

        Returns:
            A dict with ``"images"`` (transformed image tensor of shape
            ``(C, H, W)``), ``"masks"`` (class-index mask of shape ``(H, W)``
            with ``torch.long`` dtype), and ``"image_ids"`` (scalar dataset
            index of this sample).

        Note:
            ``"image_ids"`` is the dataset index ``idx``, i.e. the position of
            the sample in :attr:`samples`. It is stable across epochs and
            dataloader workers because DataLoader always calls ``__getitem__``
            with dataset indices; use it to correlate predictions back to
            :attr:`samples`.

        """
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            msg = f"Failed to read image at: {img_path}"
            raise ValueError(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            msg = f"Failed to read mask at: {mask_path}"
            raise ValueError(msg)

        transformed = self.transform(image=image, mask=mask)

        return {
            "images": transformed["image"],
            "masks": transformed["mask"].long().squeeze(0),
            "image_ids": torch.tensor(idx),
        }


def segmentation_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    """Collate a list of segmentation samples into a batched dictionary."""
    images = torch.stack([item["images"] for item in batch], dim=0)
    masks = torch.stack([item["masks"] for item in batch], dim=0)
    image_ids = torch.stack([item["image_ids"] for item in batch], dim=0)

    return {"images": images, "masks": masks, "image_ids": image_ids}


def create_segmentation_dataloader(
    img_dir: str | Path,
    mask_dir: str | Path | None,
    ann_source: str | Path | None,
    label_format: LabelFormat,
    batch_size: int = 8,
    img_size: tuple[int, int] = (512, 512),
    augment: bool = True,
    num_workers: int = 4,
    *,
    shuffle: bool | None = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for semantic segmentation.

    Args:
        img_dir: Directory containing images.
        mask_dir: Directory containing masks. Required for the ``folder``
            format; optional base directory for relative mask paths in the
            ``csv``/``json`` formats (falls back to *img_dir*).
        ann_source: Annotation source path (CSV or JSON).
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
        A PyTorch DataLoader for semantic segmentation.
    """
    dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
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
        collate_fn=segmentation_collate_fn,
        pin_memory=pin_memory,
    )

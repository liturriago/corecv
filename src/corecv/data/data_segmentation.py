"""Segmentation dataset with multi-format mask loading and Albumentations transforms."""

import json
from pathlib import Path
from typing import Any, Literal

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

MaskFormat = Literal["png_mask", "coco_rle", "yolo_seg"]


def _decode_rle(rle: dict[str, Any], h: int, w: int) -> np.ndarray:
    counts = rle["counts"]
    if isinstance(counts, str):
        m = np.zeros(len(counts), dtype=np.uint8)
        for i in range(len(counts)):
            m[i] = ord(counts[i])
        rle_decoded = np.unpackbits(m).reshape(-1, 8)[:, ::-1].flatten()
    else:
        rle_decoded = np.zeros(int(np.sum(counts)), dtype=np.uint8)
        idx = 0
        for i, count in enumerate(counts):
            if i % 2 == 1:
                rle_decoded[idx : idx + count] = 1
            idx += count
    return rle_decoded.reshape((w, h), order="F").T


class SegmentationDataset(Dataset):
    """Efficient PyTorch Dataset for Semantic/Instance Segmentation.

    Supports dynamic multi-format mask parsing and Albumentations transforms.
    Masks are loaded on-the-fly from disk to avoid exhausting VRAM.
    """

    def __init__(
        self,
        img_dir: str | Path,
        ann_source: str | Path,
        mask_format: MaskFormat,
        img_size: tuple[int, int] = (640, 640),
        augment: bool = True,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize the segmentation dataset.

        Args:
            img_dir: Directory containing images.
            ann_source: Annotation source (directory of masks, COCO JSON, or YOLO seg txts).
            mask_format: One of 'png_mask', 'coco_rle', or 'yolo_seg'.
            img_size: Target (height, width) for resizing.
            augment: Whether to apply data augmentation.
            mean: Per-channel mean for normalization.
            std: Per-channel std for normalization.
        """
        super().__init__()
        self.img_dir = Path(img_dir)
        self.ann_source = Path(ann_source)
        self.mask_format = mask_format.lower()
        self.img_size = img_size
        self.augment = augment

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.img_paths: list[Path] = sorted(
            [p for p in self.img_dir.iterdir() if p.suffix.lower() in valid_exts]
        )

        if not self.img_paths:
            msg = f"No valid images found in directory: {self.img_dir}"
            raise FileNotFoundError(msg)

        self.coco_index: dict[str, list[dict[str, Any]]] = {}
        if self.mask_format == "coco_rle" and self.ann_source.is_file():
            self._build_coco_index()

        self.transform = self._build_transforms(mean, std)

    def _build_coco_index(self) -> None:
        with self.ann_source.open() as f:
            data = json.load(f)

        id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
        for ann in data["annotations"]:
            img_name = id_to_filename.get(ann["image_id"])
            if img_name:
                if img_name not in self.coco_index:
                    self.coco_index[img_name] = []
                self.coco_index[img_name].append(ann)

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

    def _load_png_mask(self, img_name: str, orig_h: int, orig_w: int) -> np.ndarray:
        stem = Path(img_name).stem
        mask_candidates = [
            self.ann_source / f"{stem}.png",
            self.ann_source / f"{stem}.bmp",
        ]
        mask_path = None
        for candidate in mask_candidates:
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            msg = f"No mask file found for image '{img_name}' in directory: {self.ann_source}"
            raise FileNotFoundError(msg)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            msg = f"Failed to read mask at: {mask_path}"
            raise ValueError(msg)
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask

    def _load_coco_rle_mask(self, img_name: str, orig_h: int, orig_w: int) -> np.ndarray:
        anns = self.coco_index.get(img_name, [])
        if not anns:
            return np.zeros((orig_h, orig_w), dtype=np.int64)

        mask = np.zeros((orig_h, orig_w), dtype=np.int64)
        for ann in anns:
            seg = ann.get("segmentation")
            if seg is None:
                continue

            if isinstance(seg, dict) and "counts" in seg:
                binary = _decode_rle(seg, orig_h, orig_w)
            elif isinstance(seg, list):
                polygons = np.array(seg, dtype=np.int32).reshape(-1, 2)
                binary = np.zeros((orig_h, orig_w), dtype=np.uint8)
                cv2.fillPoly(binary, [polygons], 1)
            else:
                continue

            cls_id = int(ann.get("category_id", 1))
            mask[binary == 1] = cls_id

        return mask

    def _load_yolo_seg_mask(self, img_name: str, orig_h: int, orig_w: int) -> np.ndarray:
        """Parse YOLO segmentation format TXT files into a integer mask tensor."""
        txt_path = self.ann_source / f"{Path(img_name).stem}.txt"
        if not txt_path.exists():
            return np.zeros((orig_h, orig_w), dtype=np.int64)

        mask = np.zeros((orig_h, orig_w), dtype=np.int64)
        with txt_path.open() as f:
            for line in f:
                parts = line.strip().split()
                # A valid polygon needs class_id + at least 3 points (x, y) = 1 + 6 = 7 values
                if len(parts) < 7:
                    continue
                cls_id = int(parts[0])
                coords = np.array(list(map(float, parts[1:]))).reshape(-1, 2)
                coords[:, 0] *= orig_w
                coords[:, 1] *= orig_h
                coords = coords.astype(np.int32)
                
                # Ensure we have at least 3 valid vertices before drawing the polygon
                if len(coords) >= 3:
                    cv2.fillPoly(mask, [coords], cls_id)

        return mask

    def _parse_annotation_from_disk(self, img_name: str, orig_w: int, orig_h: int) -> np.ndarray:
        if self.mask_format == "png_mask":
            return self._load_png_mask(img_name, orig_h, orig_w)
        elif self.mask_format == "coco_rle":
            return self._load_coco_rle_mask(img_name, orig_h, orig_w)
        elif self.mask_format == "yolo_seg":
            return self._load_yolo_seg_mask(img_name, orig_h, orig_w)
        else:
            msg = f"Unsupported mask format: {self.mask_format}"
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and return the image and mask at the given index."""
        img_path = self.img_paths[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            msg = f"Failed to read image at: {img_path}"
            raise ValueError(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        mask = self._parse_annotation_from_disk(img_path.name, orig_w, orig_h)

        transformed = self.transform(image=image, mask=mask)

        transformed_image = transformed["image"]
        transformed_mask = torch.from_numpy(transformed["mask"]).long()

        return {
            "image": transformed_image,
            "mask": transformed_mask,
            "image_id": torch.tensor([idx]),
        }


def segmentation_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    """Collate a list of segmentation samples into a batched dictionary."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    image_ids = torch.cat([item["image_id"] for item in batch], dim=0)

    return {"images": images, "masks": masks, "image_ids": image_ids}


def create_segmentation_dataloader(
    img_dir: str | Path,
    ann_source: str | Path,
    mask_format: MaskFormat,
    batch_size: int = 16,
    img_size: tuple[int, int] = (640, 640),
    augment: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for segmentation.

    Args:
        img_dir: Directory containing images.
        ann_source: Annotation source path.
        mask_format: One of 'png_mask', 'coco_rle', or 'yolo_seg'.
        batch_size: Number of samples per batch.
        img_size: Target (height, width) for resizing.
        augment: Whether to apply data augmentation.
        num_workers: Number of subprocesses for data loading.

    Returns:
        A PyTorch DataLoader for segmentation.
    """
    dataset = SegmentationDataset(
        img_dir=img_dir,
        ann_source=ann_source,
        mask_format=mask_format,
        img_size=img_size,
        augment=augment,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        collate_fn=segmentation_collate_fn,
        pin_memory=True,
    )

"""Detection dataset with multi-format bbox parsing and Albumentations transforms."""

import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

BBoxFormat = Literal["pascal_voc", "coco", "yolo", "albumentations"]


class DetectionDataset(Dataset):
    """Efficient PyTorch Dataset for Object Detection.

    Supports dynamic format parsing, strategic LRU annotation caching,
    and Albumentations transforms.
    """

    def __init__(
        self,
        img_dir: str | Path,
        ann_source: str | Path,
        bbox_format: BBoxFormat,
        img_size: tuple[int, int] = (640, 640),
        augment: bool = True,
        cache_size: int = 10000,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize the detection dataset.

        Args:
            img_dir: Directory containing images.
            ann_source: Annotation source (directory, COCO JSON, etc.).
            bbox_format: One of 'pascal_voc', 'coco', 'yolo', or 'albumentations'.
            img_size: Target (height, width) for resizing.
            augment: Whether to apply data augmentation.
            cache_size: Maximum number of annotations to cache via LRU.
            mean: Per-channel mean for normalization.
            std: Per-channel std for normalization.
        """
        super().__init__()
        self.img_dir = Path(img_dir)
        self.ann_source = Path(ann_source)
        self.bbox_format = bbox_format.lower()
        self.img_size = img_size
        self.augment = augment
        self.cache_size = cache_size

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.img_paths: list[Path] = sorted(
            [p for p in self.img_dir.iterdir() if p.suffix.lower() in valid_exts]
        )

        if not self.img_paths:
            msg = f"No valid images found in directory: {self.img_dir}"
            raise FileNotFoundError(msg)

        self.coco_index: dict[str, list[dict[str, Any]]] = {}
        if self.bbox_format == "coco" and self.ann_source.is_file():
            self._build_coco_index()

        self.transform = self._build_transforms(mean, std)
        self._get_cached_annotation = lru_cache(maxsize=self.cache_size)(
            self._parse_annotation_from_disk
        )

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
        bbox_params = A.BboxParams(
            format="albumentations", label_fields=["category_ids"], min_visibility=0.1
        )

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

        return A.Compose(transforms_list, bbox_params=bbox_params)

    def _parse_annotation_from_disk(
        self, img_name: str, orig_w: int, orig_h: int
    ) -> tuple[np.ndarray, np.ndarray]:
        boxes: list[list[float]] = []
        labels: list[int] = []

        if self.bbox_format == "yolo":
            txt_path = self.ann_source / f"{Path(img_name).stem}.txt"
            if txt_path.exists():
                with txt_path.open() as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            xc, yc, w, h = map(float, parts[1:5])
                            x1 = max(0.0, xc - w / 2.0)
                            y1 = max(0.0, yc - h / 2.0)
                            x2 = min(1.0, xc + w / 2.0)
                            y2 = min(1.0, yc + h / 2.0)
                            if x2 > x1 and y2 > y1:
                                boxes.append([x1, y1, x2, y2])
                                labels.append(cls_id)

        elif self.bbox_format == "pascal_voc":
            xml_path = self.ann_source / f"{Path(img_name).stem}.xml"
            if xml_path.exists():
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    cls_id = (
                        int(obj.find("category_id").text)
                        if obj.find("category_id") is not None
                        else 0
                    )
                    bndbox = obj.find("bndbox")
                    x1 = float(bndbox.find("xmin").text) / orig_w
                    y1 = float(bndbox.find("ymin").text) / orig_h
                    x2 = float(bndbox.find("xmax").text) / orig_w
                    y2 = float(bndbox.find("ymax").text) / orig_h
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls_id)

        elif self.bbox_format == "coco":
            anns = self.coco_index.get(img_name, [])
            for ann in anns:
                x, y, w, h = ann["bbox"]
                x1 = max(0.0, x / orig_w)
                y1 = max(0.0, y / orig_h)
                x2 = min(1.0, (x + w) / orig_w)
                y2 = min(1.0, (y + h) / orig_h)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(ann["category_id"]))

        elif self.bbox_format == "albumentations":
            txt_path = self.ann_source / f"{Path(img_name).stem}.txt"
            if txt_path.exists():
                with txt_path.open() as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x1, y1, x2, y2 = map(float, parts[1:5])
                            if x2 > x1 and y2 > y1:
                                boxes.append([x1, y1, x2, y2])
                                labels.append(cls_id)

        if boxes:
            boxes_arr = np.array(boxes, dtype=np.float32)
        else:
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        return boxes_arr, labels_arr

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and return the image, boxes, and labels at the given index."""
        img_path = self.img_paths[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            msg = f"Failed to read image at: {img_path}"
            raise ValueError(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        boxes, labels = self._get_cached_annotation(img_path.name, orig_w, orig_h)

        transformed = self.transform(image=image, bboxes=boxes, category_ids=labels)

        transformed_image = transformed["image"]
        transformed_boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        transformed_labels = torch.tensor(transformed["category_ids"], dtype=torch.long)

        if transformed_boxes.numel() == 0:
            transformed_boxes = torch.zeros((0, 4), dtype=torch.float32)

        return {
            "image": transformed_image,
            "boxes": transformed_boxes,
            "labels": transformed_labels,
            "image_id": torch.tensor([idx]),
        }


def detection_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate a list of detection samples into a batched dictionary with a unified targets tensor.

    Args:
        batch: List of sample dictionaries returned by DetectionDataset.__getitem__.

    Returns:
        A dictionary containing:
            - 'images': Batched image tensor of shape (B, C, H, W).
            - 'targets': Tensor of shape (N_totales_batch, 6) where each row is
                         [batch_idx, label, x1, y1, x2, y2].
            - 'image_ids': Vector of image IDs in the batch of shape (B,).
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    image_ids = torch.cat([item["image_id"] for item in batch], dim=0)

    targets_list: list[torch.Tensor] = []

    for batch_idx, item in enumerate(batch):
        boxes = item["boxes"]  # Tensor (N, 4)
        labels = item["labels"]  # Tensor (N,)

        num_boxes = boxes.shape[0]
        if num_boxes > 0:
            # Asignar el índice local dentro del batch (0 a B-1)
            batch_idx_tensor = torch.full((num_boxes, 1), fill_value=batch_idx, dtype=torch.float32)
            labels_tensor = labels.unsqueeze(1).float()

            # Concatenar batch_idx, labels y coordenadas
            single_target = torch.cat([batch_idx_tensor, labels_tensor, boxes], dim=1)
            targets_list.append(single_target)

    # Aplanar las cajas de todas las imágenes en una matriz continua
    if targets_list:
        targets = torch.cat(targets_list, dim=0)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)

    return {
        "images": images,
        "targets": targets,
        "image_ids": image_ids,
    }


def create_detection_dataloader(
    img_dir: str | Path,
    ann_source: str | Path,
    bbox_format: BBoxFormat,
    batch_size: int = 16,
    img_size: tuple[int, int] = (640, 640),
    augment: bool = True,
    num_workers: int = 4,
    cache_size: int = 10000,
) -> DataLoader:
    """Create a DataLoader for object detection.

    Args:
        img_dir: Directory containing images.
        ann_source: Annotation source path.
        bbox_format: One of 'pascal_voc', 'coco', 'yolo', or 'albumentations'.
        batch_size: Number of samples per batch.
        img_size: Target (height, width) for resizing.
        augment: Whether to apply data augmentation.
        num_workers: Number of subprocesses for data loading.
        cache_size: Maximum number of annotations to cache via LRU.

    Returns:
        A PyTorch DataLoader for detection.
    """
    dataset = DetectionDataset(
        img_dir=img_dir,
        ann_source=ann_source,
        bbox_format=bbox_format,
        img_size=img_size,
        augment=augment,
        cache_size=cache_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
    )

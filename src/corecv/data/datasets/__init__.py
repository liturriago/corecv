"""Datasets module for CoreCV.

Provides dataset implementations for common computer vision tasks,
including object detection with COCO and YOLO annotation formats,
and semantic segmentation with on-disk mask loading.

Example:
    >>> from corecv.data.datasets import DetectionDataset, SegmentationDataset
    >>> dataset = DetectionDataset(root="/data/coco", format="coco")
    >>> image, bboxes, labels = dataset[0]
    >>> image.shape
    torch.Size([3, 640, 640])
    >>> bboxes.shape
    torch.Size([5, 4])
    >>> labels.shape
    torch.Size([5])
"""

from corecv.data.datasets.detection import DetectionDataset
from corecv.data.datasets.segmentation import SegmentationDataset

__all__ = [
    "DetectionDataset",
    "SegmentationDataset",
]

"""Data transforms module wrapping Albumentations for CoreCV.

Provides coordinated simultaneous transformations for classification,
segmentation, and detection tasks. All augmentations are applied
coherently across images, masks, and bounding boxes in a single
forward pass through the pipeline, ensuring spatial consistency.

The module offers:

- :class:`TransformOutput`: An immutable container for transform results.
- :class:`ClassificationTransformConfig`: Configuration for image-only
  transforms (classification tasks).
- :class:`SegmentationTransformConfig`: Configuration for image + mask
  transforms (semantic segmentation tasks).
- :class:`DetectionTransformConfig`: Configuration for image + bounding
  box transforms (object detection tasks).
- :class:`CoordinatedTransform`: A callable wrapper that applies
  synchronized augmentations and returns structured results.
- :func:`build_transforms`: A factory function that constructs a
  :class:`CoordinatedTransform` from a configuration instance.

Example:
    >>> import numpy as np
    >>> from corecv.data.transforms import (
    ...     DetectionTransformConfig,
    ...     build_transforms,
    ... )
    >>> config = DetectionTransformConfig(image_size=(640, 640))
    >>> transform = build_transforms(config)
    >>> image = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    >>> bboxes = [[10.0, 10.0, 100.0, 100.0]]
    >>> labels = [1]
    >>> output = transform(image=image, bboxes=bboxes, labels=labels)
    >>> output.image.shape
    (640, 640, 3)
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import albumentations as A
import numpy as np

# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransformOutput:
    """Immutable container for the result of a coordinated transform.

    Holds the transformed image, optional mask, and optional bounding
    boxes with their corresponding class labels. This ensures a clean,
    structured return type for all pipeline invocations.

    Attributes:
        image: Transformed image array with shape ``(H, W, C)``.
        mask: Transformed segmentation mask with shape ``(H, W)``,
            or ``None`` if segmentation is not applicable.
        bboxes: Transformed bounding boxes as a tuple of 4-tuples
            ``((x_min, y_min, x_max, y_max), ...)`` in the same
            coordinate system specified by the detection config.
        labels: Class labels corresponding to each bounding box.
    """

    image: np.ndarray
    mask: np.ndarray | None = None
    bboxes: tuple[tuple[float, float, float, float], ...] = ()
    labels: tuple[int, ...] = ()


# ---------------------------------------------------------------------------
# Configuration schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class BaseTransformConfig(ABC):
    """Abstract base configuration for all data transforms.

    Provides the common augmentation parameters shared across all tasks.
    Task-specific configurations (classification, segmentation, detection)
    extend this base with additional fields.

    Attributes:
        task: Discriminant field identifying the task type.
        image_size: Target ``(height, width)`` after resize augmentation.
        horizontal_flip_p: Probability of applying a horizontal flip.
        vertical_flip_p: Probability of applying a vertical flip.
        rotate_limit: Maximum rotation angle in degrees (``0`` disables).
        normalize: Whether to apply ImageNet channel-wise normalization.
        mean: Per-channel mean for normalization.
        std: Per-channel standard deviation for normalization.
        p: Probability for the composed geometric transform block.
    """

    task: str
    image_size: tuple[int, int] = (224, 224)
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.0
    rotate_limit: int = 0
    normalize: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    p: float = 1.0


@dataclass(frozen=True)
class ClassificationTransformConfig(BaseTransformConfig):
    """Configuration for image classification transforms.

    Produces an image-only pipeline. No mask or bounding box handling
    is included.

    Attributes:
        task: Task type discriminant, always ``"classification"``.
    """

    task: Literal["classification"] = "classification"


@dataclass(frozen=True)
class SegmentationTransformConfig(BaseTransformConfig):
    """Configuration for semantic segmentation transforms.

    Produces a pipeline that handles both image and mask inputs, applying
    all spatial augmentations to both in a coordinated manner.

    Attributes:
        task: Task type discriminant, always ``"segmentation"``.
        ignore_index: Target value in the mask to ignore during loss
            computation (stored for downstream consumption).
    """

    task: Literal["segmentation"] = "segmentation"
    ignore_index: int = 255


@dataclass(frozen=True)
class DetectionTransformConfig(BaseTransformConfig):
    """Configuration for object detection transforms.

    Produces a pipeline that handles images, bounding boxes, and class
    labels. Bounding box parameters are configured via the ``bbox_format``,
    ``min_area``, and ``min_visibility`` fields.

    Attributes:
        task: Task type discriminant, always ``"detection"``.
        bbox_format: Bounding box coordinate format accepted by
            Albumentations (``"pascal_voc"``, ``"coco"``, or ``"yolo"``).
        min_area: Minimum bounding box area in pixels to keep a box
            after spatial augmentation (boxes below this are discarded).
        min_visibility: Minimum visibility ratio (0.0--1.0) to keep a
            box after spatial augmentation.
    """

    task: Literal["detection"] = "detection"
    bbox_format: Literal["pascal_voc", "coco", "yolo"] = "pascal_voc"
    min_area: int = 256
    min_visibility: float = 0.3


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def _build_pipeline(config: BaseTransformConfig) -> A.Compose:
    """Build an Albumentations Compose pipeline from a transform config.

    Constructs the augmentation list based on the base config parameters
    and adds detection-specific ``BboxParams`` when the config is a
    :class:`DetectionTransformConfig`.

    Args:
        config: The transform configuration to build from.

    Returns:
        A fully constructed Albumentations Compose pipeline.
    """
    h, w = config.image_size
    transforms: list[A.BasicTransform] = [
        A.Resize(height=h, width=w, p=1.0),
    ]

    # Geometric augmentations
    if config.horizontal_flip_p > 0:
        transforms.append(A.HorizontalFlip(p=config.horizontal_flip_p))
    if config.vertical_flip_p > 0:
        transforms.append(A.VerticalFlip(p=config.vertical_flip_p))
    if config.rotate_limit > 0:
        transforms.append(A.Rotate(limit=config.rotate_limit, p=config.p))

    # Pixel-level augmentations can be appended here as needed.

    # Normalization (always last before the pipeline ends)
    if config.normalize:
        transforms.append(A.Normalize(mean=config.mean, std=config.std))

    # Detection-specific: attach BboxParams for coordinated bbox handling
    if isinstance(config, DetectionTransformConfig):
        bbox_params = A.BboxParams(
            format=config.bbox_format,
            label_fields=["class_labels"],
            min_area=config.min_area,
            min_visibility=config.min_visibility,
        )
        return A.Compose(transforms, bbox_params=bbox_params)

    return A.Compose(transforms)


# ---------------------------------------------------------------------------
# Coordinated transform wrapper
# ---------------------------------------------------------------------------


class CoordinatedTransform:
    """Callable wrapper for coordinated Albumentations transforms.

    Wraps an Albumentations :class:`~albumentations.Compose` pipeline and
    exposes a clean API for applying synchronized augmentations to images,
    masks, and bounding boxes. All spatial transforms (resize, flip,
    rotation, crop) are applied with identical random parameters across
    all inputs, ensuring spatial consistency.

    Instances should be created via :func:`build_transforms` or by
    calling the constructor with a :class:`BaseTransformConfig` subclass.

    Attributes:
        _config: The transform configuration used to build the pipeline.
        _pipeline: The underlying Albumentations Compose pipeline.

    Example:
        >>> import numpy as np
        >>> from corecv.data.transforms import (
        ...     SegmentationTransformConfig,
        ...     CoordinatedTransform,
        ... )
        >>> config = SegmentationTransformConfig(image_size=(512, 512))
        >>> transform = CoordinatedTransform(config)
        >>> image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 21, (512, 512), dtype=np.uint8)
        >>> output = transform(image=image, mask=mask)
        >>> output.image.shape
        (512, 512, 3)
        >>> output.mask.shape
        (512, 512)
    """

    def __init__(self, config: BaseTransformConfig) -> None:
        """Initialize the coordinated transform from a config.

        Args:
            config: A task-specific transform configuration dataclass.
        """
        self._config: BaseTransformConfig = config
        self._pipeline: A.Compose = _build_pipeline(config)

    @property
    def config(self) -> BaseTransformConfig:
        """Return the transform configuration."""
        return self._config

    @property
    def pipeline(self) -> A.Compose:
        """Return the underlying Albumentations Compose pipeline."""
        return self._pipeline

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        bboxes: Sequence[Sequence[float]] | None = None,
        labels: Sequence[int] | None = None,
    ) -> TransformOutput:
        """Apply coordinated transforms to the provided inputs.

        All inputs are augmented in a single synchronized pass through
        the pipeline. Spatial transforms use identical random parameters
        for the image, mask, and bounding boxes.

        Args:
            image: Input image array with shape ``(H, W, C)`` and dtype
                ``uint8`` or ``float32``.
            mask: Optional segmentation mask array with shape ``(H, W)``
                and dtype ``uint8``. Ignored for classification configs.
            bboxes: Optional bounding boxes, each represented as
                ``[x_min, y_min, x_max, y_max]`` (Pascal VOC), ``[x, y,
                w, h]`` (COCO), or ``[x_center, y_center, w, h]``
                normalized (YOLO). Only valid for detection configs.
            labels: Optional class labels corresponding to each bounding
                box. Required when *bboxes* is provided.

        Returns:
            A :class:`TransformOutput` containing the transformed image,
            mask (if provided), and bounding boxes with labels.

        Raises:
            ValueError: If *bboxes* are provided without *labels*.
            ValueError: If *bboxes* or *labels* are provided for a
                non-detection config.
        """
        kwargs: dict[str, Any] = {"image": image}

        if mask is not None:
            kwargs["mask"] = mask

        if bboxes is not None:
            if not isinstance(self._config, DetectionTransformConfig):
                msg = (
                    "Bounding boxes are only supported for detection "
                    f"configs, got {type(self._config).__name__}."
                )
                raise ValueError(msg)
            if labels is None:
                msg = "Labels must be provided when bboxes are given."
                raise ValueError(msg)
            kwargs["bboxes"] = [list(box) for box in bboxes]
            kwargs["class_labels"] = list(labels)

        result: dict[str, Any] = self._pipeline(**kwargs)

        # Normalize output bounding boxes to tuple of 4-tuples
        out_bboxes: tuple[tuple[float, float, float, float], ...] = ()
        out_labels: tuple[int, ...] = ()

        if "bboxes" in result:
            out_bboxes = tuple(
                (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                for box in result["bboxes"]
            )
        if "class_labels" in result:
            out_labels = tuple(int(lbl) for lbl in result["class_labels"])

        return TransformOutput(
            image=result["image"],
            mask=result.get("mask"),
            bboxes=out_bboxes,
            labels=out_labels,
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_transforms(config: BaseTransformConfig) -> CoordinatedTransform:
    """Build a coordinated transform pipeline from a config.

    This is the primary entry point for creating transform pipelines.
    It constructs a :class:`CoordinatedTransform` from the given
    configuration, selecting the appropriate pipeline structure based
    on the task type.

    Args:
        config: A task-specific transform configuration dataclass
            (:class:`ClassificationTransformConfig`,
            :class:`SegmentationTransformConfig`, or
            :class:`DetectionTransformConfig`).

    Returns:
        A :class:`CoordinatedTransform` ready for use in data loading
        or preprocessing pipelines.

    Example:
        >>> from corecv.data.transforms import (
        ...     ClassificationTransformConfig,
        ...     build_transforms,
        ... )
        >>> config = ClassificationTransformConfig(image_size=(224, 224))
        >>> transform = build_transforms(config)
    """
    return CoordinatedTransform(config)

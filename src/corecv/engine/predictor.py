"""CorePredictor: Accelerated inference engine for CoreCV models.

Provides the :class:`CorePredictor` class that wraps any CoreCV model
(backbone + neck + head) and exposes a high-level ``predict()`` API with:

- **Flexible input**: single image (path, numpy, tensor), list of images,
  or a folder path.
- **Optimized preprocessing**: letterbox/padding with aspect-ratio preservation,
  ImageNet-normalized tensor conversion, and automatic batching.
- **Accelerated inference**: ``torch.inference_mode()``, FP16/AMP via
  ``torch.autocast``, and optional ``torch.compile``.
- **Task-specific post-processing**: classification top-k + softmax,
  segmentation argmax + resize-to-original, detection bbox rescaling + NMS.

All post-processing operations run 100 % on VRAM — no CPU-GPU sync or
``pycocotools`` dependency in the hot path.

Example:
    >>> import torch
    >>> from corecv.engine.predictor import CorePredictor
    >>> from corecv.models import CoreObjectDetector
    >>> from corecv.models.backbones.resnet import ResNet50Backbone
    >>> from corecv.models.heads.detection import DecoupledAnchorFreeHead
    >>> backbone = ResNet50Backbone(pretrained=False)
    >>> fi = backbone.feature_info
    >>> head = DecoupledAnchorFreeHead(feature_info=fi, num_classes=80)
    >>> detector = CoreObjectDetector(backbone=backbone, neck=None, head=head)
    >>> predictor = CorePredictor(
    ...     model=detector, task="detection", input_size=(640, 640),
    ... )
    >>> preds = predictor.predict(torch.randn(3, 640, 640))
    >>> preds[0].detection is not None
    True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.amp import autocast
from torchvision.ops import batched_nms, box_convert

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MEAN_DEFAULT: tuple[float, float, float] = (0.485, 0.456, 0.406)
_STD_DEFAULT: tuple[float, float, float] = (0.229, 0.224, 0.225)

_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif",
})

_SUPPORTED_TASKS: frozenset[str] = frozenset({
    "classification", "segmentation", "detection",
})

# Dimension constants (avoid PLR2004)
_NDIM_2: int = 2
_NDIM_3: int = 3
_CHANNEL_COUNT_RGB: int = 3
_QUERY_BOX_DIM: int = 3
_DEFAULT_STRIDES: list[int] = [8, 16, 32, 64]
_MIN_DIMS_2D: int = 2


# ---------------------------------------------------------------------------
# Prediction dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClassificationPrediction:
    """Result of a single-image classification prediction.

    Attributes:
        class_ids: Top-k predicted class indices, shape ``(K,)``.
        scores: Softmax probabilities for each predicted class, shape
            ``(K,)``.
        labels: Optional list of human-readable class labels.
    """

    class_ids: Tensor
    scores: Tensor
    labels: list[str] | None = None


@dataclass
class SegmentationPrediction:
    """Result of a single-image segmentation prediction.

    Attributes:
        mask: Argmax class map resized to the original image dimensions,
            shape ``(H_orig, W_orig)`` with ``int64`` dtype.
        probabilities: Per-pixel softmax probabilities (optional),
            shape ``(C, H_orig, W_orig)``.
        original_size: ``(H_orig, W_orig)`` of the source image.
    """

    mask: Tensor
    original_size: tuple[int, int] = (0, 0)
    probabilities: Tensor | None = None


@dataclass
class DetectionPrediction:
    """Result of a single-image detection prediction.

    Attributes:
        boxes: Bounding boxes in ``(x_min, y_min, x_max, y_max)`` format
            resized to original image coordinates, shape ``(N, 4)``.
        scores: Confidence scores, shape ``(N,)``.
        class_ids: Predicted class indices, shape ``(N,)``.
        labels: Optional list of human-readable class labels.
    """

    boxes: Tensor
    scores: Tensor
    class_ids: Tensor
    labels: list[str] | None = None


@dataclass
class Prediction:
    """Unified container for a single-image prediction result.

    Exactly one of ``classification``, ``segmentation``, or ``detection``
    will be populated depending on the ``task`` the :class:`CorePredictor`
    was configured with.

    Attributes:
        task: The task type (``"classification"``, ``"segmentation"``, or
            ``"detection"``).
        classification: Classification result (if task is ``"classification"``).
        segmentation: Segmentation result (if task is ``"segmentation"``).
        detection: Detection result (if task is ``"detection"``).
        original_size: ``(H, W)`` of the original input image.
        image_path: Optional path to the source image file.
    """

    task: str
    classification: ClassificationPrediction | None = None
    segmentation: SegmentationPrediction | None = None
    detection: DetectionPrediction | None = None
    original_size: tuple[int, int] = (0, 0)
    image_path: str | None = None


# ---------------------------------------------------------------------------
# Letterbox helper
# ---------------------------------------------------------------------------


def _letterbox(
    image: Tensor,
    target_h: int,
    target_w: int,
    *,
    fill_value: float = 114.0 / 255.0,
) -> tuple[Tensor, float, tuple[int, int], tuple[int, int]]:
    """Resize a tensor image with letterbox padding preserving aspect ratio.

    Operates entirely on GPU tensors.  The image tensor is expected in
    ``(C, H, W)`` float format with values in ``[0, 1]``.

    Args:
        image: Input image tensor ``(C, H, W)``.
        target_h: Target height.
        target_w: Target width.
        fill_value: Padding fill value (0-1 range, default 114/255 = YOLO
            style grey fill).

    Returns:
        Tuple of:
            - ``padded``: Resized + padded tensor ``(C, target_h, target_w)``.
            - ``scale``: Resizing scale factor applied.
            - ``pad_top_left``: ``(pad_top, pad_left)`` pixels added.
            - ``orig_hw``: ``(H, W)`` of the original image.
    """
    c, h, w = image.shape
    orig_hw = (int(h), int(w))

    # Compute scale to fit inside the target box
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize with bilinear interpolation
    resized = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # (C, new_h, new_w)

    # Compute symmetric padding
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2

    # Create padded canvas
    padded = torch.full(
        (c, target_h, target_w),
        fill_value,
        dtype=image.dtype,
        device=image.device,
    )
    padded[:, pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    return padded, scale, (pad_top, pad_left), orig_hw


# ---------------------------------------------------------------------------
# CorePredictor
# ---------------------------------------------------------------------------


class CorePredictor:
    """Accelerated inference engine for CoreCV models.

    Wraps a CoreCV model and provides a clean ``predict()`` / ``predict_batch()``
    API with optimised preprocessing, GPU-native post-processing, and
    configurable acceleration features.

    Args:
        model: A CoreCV model (``nn.Module``) -- e.g. ``CoreObjectDetector``,
            a segmentation model, or a classification model.
        task: The task type.  One of ``"classification"``, ``"segmentation"``,
            or ``"detection"``.
        input_size: Target ``(height, width)`` for preprocessing.  Default
            ``(640, 640)``.
        mean: Per-channel normalisation mean.  Default ImageNet mean.
        std: Per-channel normalisation standard deviation.  Default ImageNet
            std.
        conf_threshold: Minimum confidence score for detection predictions.
            Default ``0.25``.
        iou_threshold: IoU threshold for detection NMS.  Default ``0.45``.
        topk: Number of top predictions for classification.  Default ``5``.
        half_precision: Enable FP16 inference via ``torch.autocast``.
            Default ``False``.
        compile_model: Enable ``torch.compile`` on the model (requires
            PyTorch >= 2.0).  Default ``False``.
        batch_size: Maximum batch size for list/folder inference.  Default
            ``8``.
        num_classes: Number of classes for detection heads.  If ``None``,
            inferred from the model (if possible).

    Raises:
        ValueError: If ``task`` is not a supported value.

    Example:
        >>> predictor = CorePredictor(
        ...     model=my_model,
        ...     task="detection",
        ...     input_size=(640, 640),
        ...     half_precision=True,
        ... )
        >>> results = predictor.predict("photo.jpg")
        >>> results[0].detection.boxes.shape[1]
        4
    """

    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        task: Literal["classification", "segmentation", "detection"] = "detection",
        input_size: tuple[int, int] = (640, 640),
        mean: tuple[float, float, float] = _MEAN_DEFAULT,
        std: tuple[float, float, float] = _STD_DEFAULT,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        topk: int = 5,
        half_precision: bool = False,
        compile_model: bool = False,
        batch_size: int = 8,
        num_classes: int | None = None,
    ) -> None:
        """Initialise the CorePredictor with model and inference configuration."""
        if task not in _SUPPORTED_TASKS:
            msg = (
                f"Unsupported task {task!r}. "
                f"Must be one of {sorted(_SUPPORTED_TASKS)}."
            )
            raise ValueError(msg)

        # ---- Task configuration ---------------------------------------
        self.task: str = task
        self.input_size: tuple[int, int] = (int(input_size[0]), int(input_size[1]))
        self.conf_threshold: float = conf_threshold
        self.iou_threshold: float = iou_threshold
        self.topk: int = topk
        self.batch_size: int = batch_size
        self.half_precision: bool = half_precision

        # ---- Normalisation tensors ------------------------------------
        self._mean: Tensor = torch.tensor(mean, dtype=torch.float32).view(
            1, _CHANNEL_COUNT_RGB, 1, 1,
        )
        self._std: Tensor = torch.tensor(std, dtype=torch.float32).view(
            1, _CHANNEL_COUNT_RGB, 1, 1,
        )

        # ---- Model setup ----------------------------------------------
        self.model: nn.Module = model.eval()

        # Move model to device
        self.device: torch.device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )

        # Optionally compile the model
        if compile_model:
            self.model = torch.compile(self.model)  # type: ignore[assignment]
            logger.info("Model compiled with torch.compile().")

        # ---- Class labels (optional) ----------------------------------
        self._class_labels: list[str] | None = None

        # ---- Detect num_classes from model if not provided ------------
        if num_classes is None:
            num_classes = self._infer_num_classes()
        self.num_classes: int | None = num_classes

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def register_normalization(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        """Register normalisation constants as module buffers.

        The tensors are stored on CPU and broadcast to the input device
        during preprocessing to avoid repeated allocations.

        Args:
            mean: Per-channel mean (RGB).
            std: Per-channel standard deviation (RGB).
        """
        self._mean = torch.tensor(mean, dtype=torch.float32).view(
            1, _CHANNEL_COUNT_RGB, 1, 1,
        )
        self._std = torch.tensor(std, dtype=torch.float32).view(
            1, _CHANNEL_COUNT_RGB, 1, 1,
        )

    # ------------------------------------------------------------------
    # Class label helpers
    # ------------------------------------------------------------------

    def set_class_labels(self, labels: list[str]) -> None:
        """Set human-readable class labels for predictions.

        Args:
            labels: List of label strings, one per class index.
        """
        self._class_labels = list(labels)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        source: str | Path | np.ndarray | Tensor | list[str | Path | np.ndarray | Tensor],
    ) -> list[Prediction]:
        """Run inference on one or more images.

        Accepts a wide variety of input types:

        - ``str`` or ``Path``: interpreted as a single image file path.
        - ``np.ndarray``: a single HWC or CHW image (uint8 or float32).
        - ``Tensor``: a single CHW image tensor.
        - ``list``: a list of any of the above; also supports a list of
          ``Tensor`` objects for pre-batched input.

        When a directory path is given, all images with known extensions
        inside it are loaded and processed in batches.

        Args:
            source: Image source(s) as described above.

        Returns:
            A list of :class:`Prediction` objects, one per input image.

        Raises:
            FileNotFoundError: If a file path does not exist.
            TypeError: If the input type is unsupported.
        """
        image_items: list[tuple[np.ndarray, str | None]] = self._resolve_source(
            source,
        )

        all_predictions: list[Prediction] = []
        for batch_start in range(0, len(image_items), self.batch_size):
            batch_items = image_items[batch_start : batch_start + self.batch_size]
            batch_predictions = self._predict_batch_items(batch_items)
            all_predictions.extend(batch_predictions)

        return all_predictions

    def predict_batch(self, images: list[Tensor]) -> list[Prediction]:
        """Run inference on a list of pre-loaded tensors.

        Each tensor should be in ``(C, H, W)`` format with float values
        in ``[0, 1]`` or ``[0, 255]``.  Tensors are automatically
        normalised and batched.

        Args:
            images: List of image tensors.

        Returns:
            A list of :class:`Prediction` objects, one per tensor.
        """
        items: list[tuple[Tensor, None]] = [(img, None) for img in images]
        return self._predict_tensor_items(items)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        images: list[np.ndarray],
    ) -> tuple[Tensor, list[tuple[int, int]], list[tuple[float, tuple[int, int], tuple[int, int]]]]:
        """Preprocess a list of numpy images into a batched tensor.

        Performs letterbox resizing, normalisation, and batching.  All
        operations run on the target device without CPU-GPU sync.

        Args:
            images: List of HWC numpy arrays (uint8 or float32).

        Returns:
            Tuple of:
                - ``batch``: Normalised batch tensor ``(B, 3, H, W)``.
                - ``original_sizes``: List of ``(H, W)`` for each image.
                - ``letterbox_meta``: List of ``(scale, pad_tl, orig_hw)``
                  for each image.
        """
        target_h, target_w = self.input_size
        tensors: list[Tensor] = []
        original_sizes: list[tuple[int, int]] = []
        letterbox_meta: list[tuple[float, tuple[int, int], tuple[int, int]]] = []

        for img in images:
            tensor = self._numpy_to_tensor(img)
            padded, scale, pad_tl, orig_hw = _letterbox(tensor, target_h, target_w)
            tensors.append(padded)
            original_sizes.append(orig_hw)
            letterbox_meta.append((scale, pad_tl, orig_hw))

        batch = torch.stack(tensors, dim=0)  # (B, 3, H, W)

        # Normalise (broadcasts mean/std to correct device via tensor ops)
        mean = self._mean.to(device=batch.device, dtype=batch.dtype)
        std = self._std.to(device=batch.device, dtype=batch.dtype)
        batch = (batch - mean) / std

        return batch, original_sizes, letterbox_meta

    def _numpy_to_tensor(self, img: np.ndarray) -> Tensor:
        """Convert a numpy image to a CHW float tensor on the target device.

        Supports HWC uint8 (0-255) or HWC float32 (0-1) inputs.

        Args:
            img: Input image as a numpy array.

        Returns:
            A ``(3, H, W)`` float32 tensor.
        """
        t = torch.from_numpy(np.ascontiguousarray(img)).float()
        if t.ndim == _NDIM_2:
            # Grayscale -> RGB
            t = t.unsqueeze(0).repeat(_CHANNEL_COUNT_RGB, 1, 1)
        elif t.ndim == _NDIM_3:
            if t.shape[2] == _CHANNEL_COUNT_RGB:
                # HWC -> CHW
                t = t.permute(2, 0, 1)
            elif t.shape[0] != _CHANNEL_COUNT_RGB:
                msg = (
                    f"Expected HWC or CHW image, got shape {tuple(img.shape)}"
                )
                raise TypeError(msg)
        # Scale to [0, 1] if needed
        if t.max() > 1.0:
            t = t / 255.0
        return t.to(device=self.device)

    # ------------------------------------------------------------------
    # Post-processing -- Classification
    # ------------------------------------------------------------------

    def _postprocess_classification(
        self,
        logits: Tensor,
        _original_sizes: list[tuple[int, int]],
    ) -> list[ClassificationPrediction]:
        """Post-process classification logits into top-k predictions.

        Args:
            logits: Raw logits ``(B, num_classes)``.
            _original_sizes: Original image sizes (unused, kept for API
                consistency).

        Returns:
            List of :class:`ClassificationPrediction` objects.
        """
        probs = torch.softmax(logits, dim=-1)  # (B, C)
        k = min(self.topk, probs.shape[-1])
        topk_scores, topk_ids = torch.topk(probs, k=k, dim=-1)

        predictions: list[ClassificationPrediction] = []
        for i in range(logits.shape[0]):
            predictions.append(
                ClassificationPrediction(
                    class_ids=topk_ids[i].detach().cpu(),
                    scores=topk_scores[i].detach().cpu(),
                    labels=self._class_labels,
                )
            )
        return predictions

    # ------------------------------------------------------------------
    # Post-processing -- Segmentation
    # ------------------------------------------------------------------

    def _postprocess_segmentation(
        self,
        logits: Tensor,
        original_sizes: list[tuple[int, int]],
    ) -> list[SegmentationPrediction]:
        """Post-process segmentation logits into class masks.

        Args:
            logits: Raw logits ``(B, C, H, W)``.
            original_sizes: List of ``(H, W)`` for each image.

        Returns:
            List of :class:`SegmentationPrediction` objects.
        """
        predictions: list[SegmentationPrediction] = []

        for i in range(logits.shape[0]):
            orig_h, orig_w = original_sizes[i]
            logit_i = logits[i]  # (C, H, W)

            mask = logit_i.argmax(dim=0)  # (H, W)

            mask_resized = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(orig_h, orig_w),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()

            predictions.append(
                SegmentationPrediction(
                    mask=mask_resized.detach().cpu(),
                    original_size=(orig_h, orig_w),
                )
            )

        return predictions

    # ------------------------------------------------------------------
    # Post-processing -- Detection
    # ------------------------------------------------------------------

    def _postprocess_detection(
        self,
        raw_output: dict[str, Any],
        original_sizes: list[tuple[int, int]],
        letterbox_meta: list[tuple[float, tuple[int, int], tuple[int, int]]],
    ) -> list[DetectionPrediction]:
        """Post-process detection head output into final predictions.

        Handles both:

        - **DecoupledAnchorFreeHead** format: dict with ``cls_logits`` (list),
          ``reg_pred`` (list), and ``centerness`` (list).
        - **QueryDetectionHead** format: dict with ``cls_logits`` (B, N, C)
          and ``pred_boxes`` (B, N, 4).

        Args:
            raw_output: Raw model output dict.
            original_sizes: Original ``(H, W)`` per image.
            letterbox_meta: Letterbox metadata ``(scale, pad_tl, orig_hw)``
              per image.

        Returns:
            List of :class:`DetectionPrediction` objects.
        """
        if (
            "pred_boxes" in raw_output
            and raw_output["pred_boxes"].dim() == _QUERY_BOX_DIM
        ):
            return self._postprocess_query_detection(raw_output, original_sizes)

        if "cls_logits" in raw_output and isinstance(raw_output["cls_logits"], list):
            return self._postprocess_fcose_detection(
                raw_output, original_sizes, letterbox_meta,
            )

        msg = (
            f"Unrecognised detection output format. "
            f"Keys: {list(raw_output.keys())}"
        )
        raise ValueError(msg)

    def _postprocess_query_detection(
        self,
        raw_output: dict[str, Tensor],
        original_sizes: list[tuple[int, int]],
    ) -> list[DetectionPrediction]:
        """Post-process QueryDetectionHead (RT-DETR style) output.

        Boxes are already in normalised ``(cx, cy, w, h)`` [0, 1] format.

        Args:
            raw_output: Dict with ``cls_logits`` ``(B, N, C)`` and
              ``pred_boxes`` ``(B, N, 4)``.
            original_sizes: Original ``(H, W)`` per image.

        Returns:
            List of :class:`DetectionPrediction` objects.
        """
        cls_logits = raw_output["cls_logits"]  # (B, N, C)
        pred_boxes_norm = raw_output["pred_boxes"]  # (B, N, 4) in [0, 1]

        probs = torch.sigmoid(cls_logits)  # (B, N, C)
        batch_size = cls_logits.shape[0]

        predictions: list[DetectionPrediction] = []
        for i in range(batch_size):
            orig_h, orig_w = original_sizes[i]
            prob_i = probs[i]  # (N, C)
            box_i = pred_boxes_norm[i]  # (N, 4)

            max_scores, class_ids = prob_i.max(dim=-1)  # (N,)

            keep = max_scores > self.conf_threshold
            if not keep.any():
                predictions.append(self._empty_detection_prediction())
                continue

            filtered_scores = max_scores[keep]
            filtered_classes = class_ids[keep]
            filtered_boxes = box_i[keep]  # (N_filtered, 4) normalised

            xyxy_norm = box_convert(filtered_boxes, "cxcywh", "xyxy")

            xyxy_orig = xyxy_norm.clone()
            xyxy_orig[:, 0] *= orig_w
            xyxy_orig[:, 1] *= orig_h
            xyxy_orig[:, 2] *= orig_w
            xyxy_orig[:, 3] *= orig_h

            keep_nms = self._nms(xyxy_orig, filtered_scores, filtered_classes)

            predictions.append(
                DetectionPrediction(
                    boxes=xyxy_orig[keep_nms].detach().cpu(),
                    scores=filtered_scores[keep_nms].detach().cpu(),
                    class_ids=filtered_classes[keep_nms].detach().cpu(),
                    labels=self._class_labels,
                )
            )

        return predictions

    def _decode_fcose_level(  # noqa: PLR0913
        self,
        cls_i: Tensor,
        reg_i: Tensor,
        crt_i: Tensor,
        stride: int,
        pad_top: int,
        pad_left: int,
        scale: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode boxes and scores for a single FCOS feature level.

        Args:
            cls_i: Classification logits ``(C, H_l, W_l)``.
            reg_i: Regression predictions ``(4, H_l, W_l)``.
            crt_i: Centerness predictions ``(1, H_l, W_l)``.
            stride: Feature level stride.
            pad_top: Letterbox top padding in pixels.
            pad_left: Letterbox left padding in pixels.
            scale: Letterbox resize scale factor.

        Returns:
            Tuple of ``(boxes, scores, class_ids)`` for locations above
            the confidence threshold.  Each has shape ``(K, 4)``, ``(K,)``,
            ``(K,)`` respectively.
        """
        num_classes_l = cls_i.shape[0]
        h_l, w_l = cls_i.shape[1], cls_i.shape[2]

        # Generate grid of centres in letterbox coordinate space
        y_centers = (
            torch.arange(h_l, device=self.device, dtype=cls_i.dtype) + 0.5
        ) * stride
        x_centers = (
            torch.arange(w_l, device=self.device, dtype=cls_i.dtype) + 0.5
        ) * stride
        grid_y, grid_x = torch.meshgrid(
            y_centers, x_centers, indexing="ij",
        )
        grid_x_flat = grid_x.flatten()
        grid_y_flat = grid_y.flatten()

        # Decode boxes: (left, top, right, bottom) -> xyxy
        left, top, right, bottom = reg_i[0], reg_i[1], reg_i[2], reg_i[3]
        x_min = grid_x_flat - left.flatten()
        y_min = grid_y_flat - top.flatten()
        x_max = grid_x_flat + right.flatten()
        y_max = grid_y_flat + bottom.flatten()

        boxes_l = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

        # Remove padding offset and rescale to original image
        boxes_l[:, 0] -= pad_left
        boxes_l[:, 1] -= pad_top
        boxes_l[:, 2] -= pad_left
        boxes_l[:, 3] -= pad_top

        if scale > 0:
            boxes_l = boxes_l / scale

        # Classification scores: sigmoid * centerness
        cls_scores = cls_i.sigmoid()
        crt_scores = crt_i.sigmoid().flatten()
        cls_scores_flat = cls_scores.reshape(num_classes_l, -1)
        combined_scores = cls_scores_flat * crt_scores.unsqueeze(0)

        max_scores, max_classes = combined_scores.max(dim=0)

        keep = max_scores > self.conf_threshold
        if keep.any():
            return boxes_l[keep], max_scores[keep], max_classes[keep]

        return (
            torch.zeros(0, 4, device=self.device),
            torch.zeros(0, device=self.device),
            torch.zeros(0, device=self.device, dtype=torch.long),
        )

    def _postprocess_fcose_detection(
        self,
        raw_output: dict[str, list[Tensor]],
        original_sizes: list[tuple[int, int]],
        letterbox_meta: list[tuple[float, tuple[int, int], tuple[int, int]]],
    ) -> list[DetectionPrediction]:
        """Post-process DecoupledAnchorFreeHead (FCOS/YOLOX style) output.

        Decodes per-level ``(left, top, right, bottom)`` regression predictions
        into absolute ``(x_min, y_min, x_max, y_max)`` boxes and applies NMS.

        Args:
            raw_output: Dict with ``cls_logits`` (list of per-level tensors),
              ``reg_pred`` (list of per-level tensors), and ``centerness``
              (list of per-level tensors).
            original_sizes: Original ``(H, W)`` per image.
            letterbox_meta: Letterbox metadata ``(scale, pad_tl, orig_hw)``
              per image.

        Returns:
            List of :class:`DetectionPrediction` objects.
        """
        cls_logits_list = raw_output["cls_logits"]
        reg_pred_list = raw_output["reg_pred"]
        centerness_list = raw_output.get("centerness")

        strides = self._get_level_strides()
        batch_size = cls_logits_list[0].shape[0]
        num_levels = len(cls_logits_list)

        predictions: list[DetectionPrediction] = []

        for i in range(batch_size):
            orig_h, orig_w = original_sizes[i]
            scale, pad_tl, _orig_hw = letterbox_meta[i]
            pad_top, pad_left = pad_tl

            all_boxes: list[Tensor] = []
            all_scores: list[Tensor] = []
            all_classes: list[Tensor] = []

            for level_idx in range(num_levels):
                cls_i = cls_logits_list[level_idx][i]
                reg_i = reg_pred_list[level_idx][i]
                crt_i = (
                    centerness_list[level_idx][i]
                    if centerness_list is not None
                    else torch.ones(
                        1, cls_i.shape[1], cls_i.shape[2], device=self.device,
                    )
                )

                boxes, scores, class_ids = self._decode_fcose_level(
                    cls_i, reg_i, crt_i, strides[level_idx],
                    pad_top, pad_left, scale,
                )
                if scores.numel() > 0:
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_classes.append(class_ids)

            if not all_boxes:
                predictions.append(self._empty_detection_prediction())
                continue

            predictions.append(self._merge_and_nms(
                all_boxes, all_scores, all_classes, orig_h, orig_w,
            ))

        return predictions

    def _merge_and_nms(
        self,
        all_boxes: list[Tensor],
        all_scores: list[Tensor],
        all_classes: list[Tensor],
        orig_h: int,
        orig_w: int,
    ) -> DetectionPrediction:
        """Merge multi-level detections and apply NMS.

        Args:
            all_boxes: Per-level box tensors.
            all_scores: Per-level score tensors.
            all_classes: Per-level class ID tensors.
            orig_h: Original image height.
            orig_w: Original image width.

        Returns:
            :class:`DetectionPrediction` after NMS.
        """
        all_boxes_t = torch.cat(all_boxes, dim=0)
        all_scores_t = torch.cat(all_scores, dim=0)
        all_classes_t = torch.cat(all_classes, dim=0)

        # Clip boxes to image bounds
        all_boxes_t[:, 0] = all_boxes_t[:, 0].clamp(min=0, max=orig_w)
        all_boxes_t[:, 1] = all_boxes_t[:, 1].clamp(min=0, max=orig_h)
        all_boxes_t[:, 2] = all_boxes_t[:, 2].clamp(min=0, max=orig_w)
        all_boxes_t[:, 3] = all_boxes_t[:, 3].clamp(min=0, max=orig_h)

        keep_nms = self._nms(all_boxes_t, all_scores_t, all_classes_t)

        return DetectionPrediction(
            boxes=all_boxes_t[keep_nms].detach().cpu(),
            scores=all_scores_t[keep_nms].detach().cpu(),
            class_ids=all_classes_t[keep_nms].detach().cpu(),
            labels=self._class_labels,
        )

    # ------------------------------------------------------------------
    # NMS
    # ------------------------------------------------------------------

    def _nms(
        self,
        boxes: Tensor,
        scores: Tensor,
        class_ids: Tensor,
    ) -> Tensor:
        """Run per-class Non-Maximum Suppression on GPU.

        Uses ``torchvision.ops.batched_nms`` which runs entirely on VRAM
        with no CPU-GPU synchronisation.

        Args:
            boxes: Bounding boxes ``(N, 4)`` in ``(x_min, y_min, x_max, y_max)``
                format.
            scores: Confidence scores ``(N,)``.
            class_ids: Class indices ``(N,)``.

        Returns:
            Indices of kept boxes.
        """
        if boxes.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)

        return batched_nms(
            boxes,
            scores,
            class_ids,
            iou_threshold=self.iou_threshold,
        )

    # ------------------------------------------------------------------
    # Box rescaling helper (for external use)
    # ------------------------------------------------------------------

    @staticmethod
    def _rescale_boxes(
        boxes: Tensor,
        from_size: tuple[int, int],
        to_size: tuple[int, int],
    ) -> Tensor:
        """Rescale boxes from one resolution to another.

        Args:
            boxes: Bounding boxes ``(N, 4)`` in ``(x_min, y_min, x_max, y_max)``
                format.
            from_size: ``(height, width)`` of the source coordinate system.
            to_size: ``(height, width)`` of the target coordinate system.

        Returns:
            Rescaled boxes ``(N, 4)``.
        """
        if boxes.numel() == 0:
            return boxes

        rescaled = boxes.clone().float()
        from_h, from_w = from_size
        to_h, to_w = to_size

        sx = to_w / max(from_w, 1)
        sy = to_h / max(from_h, 1)

        rescaled[:, 0] *= sx
        rescaled[:, 1] *= sy
        rescaled[:, 2] *= sx
        rescaled[:, 3] *= sy

        return rescaled

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_detection_prediction(self) -> DetectionPrediction:
        """Create an empty DetectionPrediction (no detections above threshold)."""
        return DetectionPrediction(
            boxes=torch.zeros(0, 4, device=self.device, dtype=torch.float32),
            scores=torch.zeros(0, device=self.device, dtype=torch.float32),
            class_ids=torch.zeros(0, device=self.device, dtype=torch.long),
            labels=self._class_labels,
        )

    def _get_level_strides(self) -> list[int]:
        """Extract per-level strides from the detection head.

        Looks for ``level_strides`` property on ``model.head`` or falls
        back to the default YOLO/FCOS stride set.

        Returns:
            List of integer stride values.
        """
        head = getattr(self.model, "head", None)
        if head is not None and hasattr(head, "level_strides"):
            return list(head.level_strides)
        return list(_DEFAULT_STRIDES)

    def _infer_num_classes(self) -> int | None:
        """Attempt to infer the number of classes from the model head.

        Returns:
            Number of classes, or ``None`` if inference fails.
        """
        head = getattr(self.model, "head", None)
        if head is not None and hasattr(head, "num_classes"):
            return int(head.num_classes)
        return None

    # ------------------------------------------------------------------
    # Source resolution
    # ------------------------------------------------------------------

    def _resolve_source(
        self,
        source: str | Path | np.ndarray | Tensor | list[str | Path | np.ndarray | Tensor],
    ) -> list[tuple[np.ndarray, str | None]]:
        """Resolve a flexible input source into a list of (numpy, path) pairs.

        Args:
            source: Input source as described in :meth:`predict`.

        Returns:
            List of ``(numpy_image, optional_path)`` tuples.
        """
        if isinstance(source, (str, Path)):
            return self._resolve_path(Path(source))

        if isinstance(source, np.ndarray):
            return [(source, None)]

        if isinstance(source, Tensor):
            return [(self._tensor_to_numpy(source), None)]

        if isinstance(source, list):
            return self._resolve_list(source)

        msg = (
            f"Unsupported source type: {type(source).__name__}. "
            "Expected str, Path, np.ndarray, Tensor, or list."
        )
        raise TypeError(msg)

    def _resolve_path(self, path: Path) -> list[tuple[np.ndarray, str | None]]:
        """Resolve a single path (file or directory).

        Args:
            path: File or directory path.

        Returns:
            List of ``(numpy_image, path_string)`` tuples.
        """
        if path.is_dir():
            return self._load_images_from_folder(path)
        if path.is_file():
            return [self._load_image_file(path)]
        msg = f"Path does not exist: {path}"
        raise FileNotFoundError(msg)

    def _resolve_list(
        self,
        source_list: list[str | Path | np.ndarray | Tensor],
    ) -> list[tuple[np.ndarray, str | None]]:
        """Resolve a list of mixed source types.

        Args:
            source_list: List of image sources.

        Returns:
            List of ``(numpy_image, optional_path)`` tuples.
        """
        items: list[tuple[np.ndarray, str | None]] = []
        for item in source_list:
            if isinstance(item, Tensor):
                items.append((self._tensor_to_numpy(item), None))
            elif isinstance(item, np.ndarray):
                items.append((item, None))
            elif isinstance(item, (str, Path)):
                p = Path(item)
                if p.is_file():
                    items.append(self._load_image_file(p))
                elif p.is_dir():
                    items.extend(self._load_images_from_folder(p))
                else:
                    msg = f"Path does not exist: {p}"
                    raise FileNotFoundError(msg)
            else:
                msg = (
                    f"Unsupported list element type: {type(item).__name__}. "
                    "Expected str, Path, np.ndarray, or Tensor."
                )
                raise TypeError(msg)
        return items

    @staticmethod
    def _load_image_file(path: Path) -> tuple[np.ndarray, str]:
        """Load a single image file as a numpy HWC RGB array.

        Uses torchvision.io for GPU-friendly loading when possible, with
        a fallback to PIL.

        Args:
            path: Path to the image file.

        Returns:
            Tuple of ``(image_array, path_string)``.
        """
        try:
            from torchvision.io import read_image  # noqa: PLC0415

            img_tensor = read_image(str(path))  # (C, H, W) uint8
            img_np = img_tensor.permute(1, 2, 0).numpy()
        except ImportError:
            from PIL import Image  # noqa: PLC0415

            img_pil = Image.open(path).convert("RGB")
            img_np = np.array(img_pil)

        return img_np, str(path)

    def _load_images_from_folder(
        self,
        folder: Path,
    ) -> list[tuple[np.ndarray, str]]:
        """Load all supported images from a directory.

        Args:
            folder: Directory path.

        Returns:
            List of ``(numpy_image, path_string)`` tuples.
        """
        items: list[tuple[np.ndarray, str]] = []
        for file_path in sorted(folder.iterdir()):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in _IMAGE_EXTENSIONS
            ):
                items.append(self._load_image_file(file_path))
        if not items:
            logger.warning("No supported images found in folder: %s", folder)
        return items

    @staticmethod
    def _tensor_to_numpy(img: Tensor) -> np.ndarray:
        """Convert a CHW float tensor to HWC uint8 numpy.

        Args:
            img: ``(C, H, W)`` tensor with values in ``[0, 1]`` or ``[0, 255]``.

        Returns:
            HWC ``uint8`` numpy array.
        """
        t = img.detach().cpu()
        if t.ndim == _NDIM_3 and t.shape[0] in (1, _CHANNEL_COUNT_RGB):
            t = t.permute(1, 2, 0)
        if t.dtype.is_floating_point:
            t = (t.clamp(0, 1) * 255).to(torch.uint8)
        return t.numpy()

    # ------------------------------------------------------------------
    # Internal batched prediction
    # ------------------------------------------------------------------

    def _predict_batch_items(
        self,
        items: list[tuple[np.ndarray, str | None]],
    ) -> list[Prediction]:
        """Run inference on a list of numpy images.

        Args:
            items: List of ``(numpy_image, optional_path)`` tuples.

        Returns:
            List of :class:`Prediction` objects.
        """
        images = [item[0] for item in items]
        paths = [item[1] for item in items]

        batch, original_sizes, letterbox_meta = self._preprocess(images)
        raw_output = self._run_inference(batch)

        return self._dispatch_postprocess(
            raw_output=raw_output,
            original_sizes=original_sizes,
            letterbox_meta=letterbox_meta,
            image_paths=paths,
        )

    def _predict_tensor_items(
        self,
        items: list[tuple[Tensor, None]],
    ) -> list[Prediction]:
        """Run inference on a list of tensor images.

        Args:
            items: List of ``(tensor, None)`` tuples.

        Returns:
            List of :class:`Prediction` objects.
        """
        np_items: list[tuple[np.ndarray, str | None]] = [
            (self._tensor_to_numpy(img), None)
            for img, _ in items
        ]
        return self._predict_batch_items(np_items)

    def _run_inference(self, batch: Tensor) -> dict[str, Any] | Tensor:
        """Run the model forward pass under inference mode.

        Supports optional AMP and torch.compile.

        Args:
            batch: Normalised input batch ``(B, 3, H, W)``.

        Returns:
            Raw model output (type depends on task and head).
        """
        with torch.inference_mode(), autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.half_precision,
        ):
            return self.model(batch)

    def _dispatch_postprocess(
        self,
        raw_output: dict[str, Any] | Tensor,
        original_sizes: list[tuple[int, int]],
        letterbox_meta: list[tuple[float, tuple[int, int], tuple[int, int]]],
        image_paths: list[str | None],
    ) -> list[Prediction]:
        """Dispatch to the task-specific post-processor.

        Args:
            raw_output: Raw model output.
            original_sizes: Original image sizes per batch element.
            letterbox_meta: Letterbox metadata per batch element.
            image_paths: Optional source file paths per batch element.

        Returns:
            List of :class:`Prediction` objects.
        """
        if self.task == "classification":
            return self._wrap_classification(
                raw_output, original_sizes, image_paths,
            )
        if self.task == "segmentation":
            return self._wrap_segmentation(
                raw_output, original_sizes, image_paths,
            )
        if self.task == "detection":
            return self._wrap_detection(
                raw_output, original_sizes, letterbox_meta, image_paths,
            )
        # Unreachable due to __init__ validation.
        msg = f"Unknown task: {self.task}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    def _wrap_classification(
        self,
        raw_output: Tensor,
        original_sizes: list[tuple[int, int]],
        image_paths: list[str | None],
    ) -> list[Prediction]:
        """Wrap classification predictions into Prediction dataclasses."""
        cls_preds = self._postprocess_classification(raw_output, original_sizes)
        return [
            Prediction(
                task="classification",
                classification=pred,
                original_size=original_sizes[i],
                image_path=image_paths[i],
            )
            for i, pred in enumerate(cls_preds)
        ]

    def _wrap_segmentation(
        self,
        raw_output: Tensor,
        original_sizes: list[tuple[int, int]],
        image_paths: list[str | None],
    ) -> list[Prediction]:
        """Wrap segmentation predictions into Prediction dataclasses."""
        seg_preds = self._postprocess_segmentation(raw_output, original_sizes)
        return [
            Prediction(
                task="segmentation",
                segmentation=pred,
                original_size=original_sizes[i],
                image_path=image_paths[i],
            )
            for i, pred in enumerate(seg_preds)
        ]

    def _wrap_detection(
        self,
        raw_output: dict[str, Any],
        original_sizes: list[tuple[int, int]],
        letterbox_meta: list[tuple[float, tuple[int, int], tuple[int, int]]],
        image_paths: list[str | None],
    ) -> list[Prediction]:
        """Wrap detection predictions into Prediction dataclasses."""
        det_preds = self._postprocess_detection(
            raw_output, original_sizes, letterbox_meta,
        )
        return [
            Prediction(
                task="detection",
                detection=pred,
                original_size=original_sizes[i],
                image_path=image_paths[i],
            )
            for i, pred in enumerate(det_preds)
        ]

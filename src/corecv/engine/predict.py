"""Inference engines for classification, segmentation, and detection tasks.

Provides :class:`ImagePredictor` for running inference on a single image or a
folder of images with task-specific post-processing.

Example::

    from corecv.engine.predict import ImagePredictor

    predictor = ImagePredictor(task="detection", model=model, num_classes=80)
    result = predictor.predict_image("image.jpg")
    print(result.boxes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import albumentations as A  # noqa: N812
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor, nn

logger = logging.getLogger(__name__)

# Valid image extensions when predicting over a folder.
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PredictionResult:
    """Prediction for a single image.

    Attributes:
        image_path: Path or identifier of the source image.
        task: Task name (``classification``, ``segmentation``, or
            ``detection``).
        labels: Class indices. For classification these are the top-k class
            indices; for detection, the class of each prediction.
        scores: Confidence scores. For classification these are the top-k
            probabilities; for detection, the confidence of each prediction.
        probabilities: Full softmax distribution for classification.
        mask: Per-pixel class mask of shape ``(H, W)`` for segmentation.
        boxes: Predicted boxes of shape ``(N, 4)`` in ``(x1, y1, x2, y2)``
            image coordinates for detection.

    """

    image_path: str
    task: str
    labels: Tensor | None = None
    scores: Tensor | None = None
    probabilities: Tensor | None = None
    mask: Tensor | None = None
    boxes: Tensor | None = None


class ImagePredictor:
    """Task-aware inference engine for single images or image folders.

    Preprocesses images with the same resize/normalize pipeline used during
    training and runs inference with gradients disabled. Post-processing is
    task-specific:

    - **Classification**: softmax probabilities with top-k labels.
    - **Segmentation**: per-pixel class mask resized to the original image.
    - **Detection**: the one-to-one head output is filtered by confidence and
      reduced to the top-*max_detections* detections (NMS-free); boxes are
      rescaled to the original image coordinates.
    """

    def __init__(  # noqa: PLR0913
        self,
        task: str,
        model: nn.Module,
        num_classes: int,
        *,
        device: str | torch.device = "cuda",
        img_size: tuple[int, int] = (224, 224),
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        top_k: int = 1,
        conf_threshold: float = 0.25,
        max_detections: int = 300,
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize the image predictor.

        Args:
            task: Task to predict. One of ``classification``,
                ``segmentation``, or ``detection``.
            model: Model mapping ``(B, 3, H, W)`` to the task-specific
                output format. Detection models return a
                ``(preds_o2m, preds_o2o)`` tuple.
            num_classes: Number of output classes.
            device: Target device string or :class:`torch.device`.
            img_size: Target ``(height, width)`` for resizing. Detection
                models typically use ``(640, 640)``.
            mean: Per-channel mean for image normalization.
            std: Per-channel std for image normalization.
            top_k: Number of top classes returned for classification.
            conf_threshold: Minimum confidence to keep a detection.
            max_detections: Maximum number of detections returned per image.
            class_names: Optional class names for readable outputs.

        Raises:
            ValueError: If *task* is not a supported task.

        """
        if task not in ("classification", "segmentation", "detection"):
            msg = f"Unknown task: {task}. Choose from classification, segmentation, detection"
            raise ValueError(msg)

        self.task = task
        self.model = model.to(device)
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.img_size = img_size
        self.top_k = top_k
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections
        self.class_names = class_names

        height, width = img_size
        self.transform = A.Compose(
            [
                A.Resize(height=height, width=width),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
        )

    @staticmethod
    def _load_image(image: str | Path | np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Load an image path or array and return it in RGB plus its size."""
        if isinstance(image, np.ndarray):
            array = image
        else:
            array = cv2.imread(str(image))
            if array is None:
                msg = f"Failed to read image at: {image}"
                raise ValueError(msg)
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        height, width = array.shape[:2]
        return array, (height, width)

    def _restore_mask(self, mask: Tensor, orig_height: int, orig_width: int) -> Tensor:
        """Resize a prediction mask back to the original image size."""
        if (mask.shape[0], mask.shape[1]) == (orig_height, orig_width):
            return mask
        resized = nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(orig_height, orig_width),
            mode="nearest",
        )
        return resized.long().squeeze(0).squeeze(0)

    def _postprocess_detection(
        self,
        logits: Tensor,
        boxes: Tensor,
        orig_height: int,
        orig_width: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Filter detections by confidence and rescale boxes to image coords."""
        scores_all = logits.sigmoid()
        max_scores, pred_labels = scores_all.max(dim=-1)
        keep = max_scores >= self.conf_threshold
        boxes = boxes[keep]
        scores = max_scores[keep]
        labels = pred_labels[keep]

        if scores.numel() > self.max_detections:
            top_indices = scores.topk(self.max_detections).indices
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            labels = labels[top_indices]

        scale_x = orig_width / self.img_size[1]
        scale_y = orig_height / self.img_size[0]
        scale = torch.tensor([scale_x, scale_y, scale_x, scale_y])
        return boxes * scale, scores, labels

    @torch.no_grad()
    def predict_image(
        self,
        image: str | Path | np.ndarray,
        image_path: str | Path | None = None,
    ) -> PredictionResult:
        """Run inference on a single image.

        Args:
            image: Path to an image file or a loaded RGB image array.
            image_path: Optional label stored in the result. Defaults to the
                image path, or ``"image"`` when an array is provided.

        Returns:
            A :class:`PredictionResult` with task-specific fields.

        """
        self.model.eval()
        array, (orig_height, orig_width) = self._load_image(image)

        if image_path is not None:
            image_id = str(image_path)
        elif isinstance(image, (str, Path)):
            image_id = str(image)
        else:
            image_id = "image"

        tensor = self.transform(image=array)["image"].unsqueeze(0).to(self.device)

        if self.task == "classification":
            logits = self.model(tensor)
            probs = logits[0].softmax(dim=0).cpu()
            top_probs, top_indices = probs.topk(self.top_k)
            return PredictionResult(
                image_path=image_id,
                task=self.task,
                labels=top_indices,
                scores=top_probs,
                probabilities=probs,
            )

        if self.task == "segmentation":
            logits = self.model(tensor)
            mask = logits[0].argmax(dim=0).long()
            mask = self._restore_mask(mask, orig_height, orig_width)
            return PredictionResult(image_path=image_id, task=self.task, mask=mask)

        # Detection: use the one-to-one head and decode NMS-free predictions.
        preds = self.model(tensor)
        _, preds_o2o = preds
        o2o_logits, o2o_boxes = preds_o2o
        boxes, scores, labels = self._postprocess_detection(
            o2o_logits[0].cpu(),
            o2o_boxes[0].cpu(),
            orig_height,
            orig_width,
        )
        return PredictionResult(
            image_path=image_id,
            task=self.task,
            boxes=boxes,
            scores=scores,
            labels=labels,
        )

    def predict_folder(
        self,
        folder: str | Path,
        *,
        recursive: bool = False,
        extensions: set[str] | None = None,
    ) -> list[PredictionResult]:
        """Run inference on every supported image in a folder.

        Args:
            folder: Directory containing images.
            recursive: Whether to also scan subdirectories.
            extensions: Set of image extensions to consider. Defaults to the
                common image formats.

        Returns:
            List of :class:`PredictionResult`, one per image, ordered by
            filename.

        """
        folder = Path(folder)
        valid_extensions = extensions or _IMAGE_EXTENSIONS
        pattern = "**/*" if recursive else "*"
        paths = sorted(
            path
            for path in folder.glob(pattern)
            if path.is_file() and path.suffix.lower() in valid_extensions
        )

        if not paths:
            logger.warning("No images found in: %s", folder)
        return [self.predict_image(path) for path in paths]

    def predict_images(self, paths: list[str | Path]) -> list[PredictionResult]:
        """Run inference on a list of image paths.

        Args:
            paths: List of image file paths.

        Returns:
            List of :class:`PredictionResult`, one per image, in order.

        """
        return [self.predict_image(path) for path in paths]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from corecv.models.detection import create_detection_model

    model = create_detection_model("csp_nano", num_classes=4, neck="csppanet", neck_out_channels=32)
    predictor = ImagePredictor(
        task="detection",
        model=model,
        num_classes=4,
        device=device,
        img_size=(128, 128),
    )

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    result = predictor.predict_image(image)
    print(f"Detection boxes: {result.boxes.shape}, scores: {result.scores.shape}")  # noqa: T201
    print("Prediction completed successfully.")  # noqa: T201

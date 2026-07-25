"""Training engines for classification, segmentation, and detection tasks.

Provides :class:`ClassificationTrainer`, :class:`SegmentationTrainer`, and
:class:`DetectionTrainer` for managing end-to-end training loops with
GPU-native metric computation, optional LR scheduling, and per-epoch
history tracking.

Example::

    from corecv.engine.train import ClassificationTrainer

    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_classes=10,
    )
    history = trainer.fit(num_epochs=10)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from corecv.losses.loss_classification import ClassificationCrossEntropyLoss
from corecv.losses.loss_detection import DualHeadDetectionLoss
from corecv.losses.loss_segmentation import SegmentationCrossEntropyLoss
from corecv.metrics.metric_classification import ClassificationMetrics
from corecv.metrics.metric_detection import DetectionMetrics
from corecv.metrics.metric_segmentation import SegmentationMetrics

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _batch_to_device(
    batch: dict[str, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    """Move all tensor values in a batch dictionary to the target device.

    Non-tensor values are passed through unchanged.

    Args:
        batch: Dictionary mapping string keys to tensors or other values.
        device: Target device for tensor transfer.

    Returns:
        A new dictionary with all tensor values placed on *device*.
    """
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


# ---------------------------------------------------------------------------
# Classification Trainer
# ---------------------------------------------------------------------------


class ClassificationTrainer:
    """Trainer for image classification models.

    Manages the training and validation loops for classification tasks,
    including forward pass, loss computation, backpropagation, metric
    accumulation, and optional learning rate scheduling.

    Attributes:
        model: The classification model moved to the target device.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        loss_fn: Loss function accepting ``(logits, labels)``.
        optimizer: Optimizer for parameter updates.
        num_classes: Number of classification classes.
        device: Target device for computation.
        scheduler: Optional learning rate scheduler.
    """

    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        num_classes: int,
        device: str | torch.device = "cuda",
        scheduler: LRScheduler | None = None,
    ) -> None:
        """Initialize the classification trainer.

        Args:
            model: Classification model mapping ``(B, 3, H, W)`` to
                ``(B, num_classes)``.
            train_loader: Training data loader yielding dicts with
                ``"images"`` and ``"labels"`` keys.
            val_loader: Validation data loader yielding dicts with
                ``"images"`` and ``"labels"`` keys.
            loss_fn: Loss function accepting ``(logits, labels)`` and
                returning a scalar loss tensor.
            optimizer: Optimizer instance for gradient-based updates.
            num_classes: Number of target classification classes.
            device: Target device string or :class:`torch.device`.
            scheduler: Optional LR scheduler stepped after each epoch.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.scheduler = scheduler
        self._train_metrics = ClassificationMetrics(num_classes=num_classes)
        self._val_metrics = ClassificationMetrics(num_classes=num_classes)

    def train_epoch(self) -> dict[str, float]:
        """Execute a single training epoch.

        Returns:
            Dictionary of training metrics for this epoch.  Always
            contains ``"loss"`` plus all keys returned by
            :meth:`ClassificationMetrics.compute`.
        """
        self.model.train()
        self._train_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        for raw_batch in self.train_loader:
            batch = _batch_to_device(raw_batch, self.device)
            images = batch["images"]
            labels = batch["labels"]

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            self._train_metrics.update(logits.detach(), labels)
            total_loss += loss.item()
            num_batches += 1

        metrics = self._train_metrics.compute()
        metrics["loss"] = total_loss / max(num_batches, 1)
        return metrics

    def validate(self) -> dict[str, float]:
        """Run validation and compute metrics.

        Returns:
            Dictionary of validation metrics.  Always contains ``"loss"``
            plus all keys returned by :meth:`ClassificationMetrics.compute`.
        """
        self.model.eval()
        self._val_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for raw_batch in self.val_loader:
                batch = _batch_to_device(raw_batch, self.device)
                images = batch["images"]
                labels = batch["labels"]

                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

                self._val_metrics.update(logits, labels)
                total_loss += loss.item()
                num_batches += 1

        metrics = self._val_metrics.compute()
        metrics["loss"] = total_loss / max(num_batches, 1)
        return metrics

    def fit(self, num_epochs: int) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            num_epochs: Total number of training epochs to execute.

        Returns:
            Dictionary mapping metric names (prefixed with ``"train_"``
            or ``"val_"``) to lists of per-epoch float values.
        """
        history: dict[str, list[float]] = defaultdict(list)

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            for key, value in train_metrics.items():
                history[f"train_{key}"].append(value)
            for key, value in val_metrics.items():
                history[f"val_{key}"].append(value)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f train_top1_acc=%.4f val_top1_acc=%.4f",
                epoch,
                num_epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                train_metrics["top1_acc"],
                val_metrics["top1_acc"],
            )

        return dict(history)


# ---------------------------------------------------------------------------
# Segmentation Trainer
# ---------------------------------------------------------------------------


class SegmentationTrainer:
    """Trainer for semantic segmentation models.

    Manages training and validation loops for pixel-wise segmentation,
    including forward pass, loss computation, backpropagation, metric
    accumulation, and optional learning rate scheduling.

    Attributes:
        model: The segmentation model moved to the target device.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        loss_fn: Loss function accepting ``(logits, masks)``.
        optimizer: Optimizer for parameter updates.
        num_classes: Number of segmentation classes.
        device: Target device for computation.
        scheduler: Optional learning rate scheduler.
    """

    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        num_classes: int,
        device: str | torch.device = "cuda",
        scheduler: LRScheduler | None = None,
    ) -> None:
        """Initialize the segmentation trainer.

        Args:
            model: Segmentation model mapping ``(B, 3, H, W)`` to
                ``(B, num_classes, H, W)``.
            train_loader: Training data loader yielding dicts with
                ``"images"`` and ``"masks"`` keys.
            val_loader: Validation data loader yielding dicts with
                ``"images"`` and ``"masks"`` keys.
            loss_fn: Loss function accepting ``(logits, masks)`` and
                returning a scalar loss tensor.
            optimizer: Optimizer instance for gradient-based updates.
            num_classes: Number of target segmentation classes.
            device: Target device string or :class:`torch.device`.
            scheduler: Optional LR scheduler stepped after each epoch.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.scheduler = scheduler
        self._train_metrics = SegmentationMetrics(num_classes=num_classes)
        self._val_metrics = SegmentationMetrics(num_classes=num_classes)

    def train_epoch(self) -> dict[str, float]:
        """Execute a single training epoch.

        Returns:
            Dictionary of training metrics for this epoch.  Always
            contains ``"loss"`` plus all keys returned by
            :meth:`SegmentationMetrics.compute`.
        """
        self.model.train()
        self._train_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        for raw_batch in self.train_loader:
            batch = _batch_to_device(raw_batch, self.device)
            images = batch["images"]
            masks = batch["masks"]

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, masks)
            loss.backward()
            self.optimizer.step()

            self._train_metrics.update(logits.detach(), masks)
            total_loss += loss.item()
            num_batches += 1

        metrics = self._train_metrics.compute()
        metrics["loss"] = total_loss / max(num_batches, 1)
        return metrics

    def validate(self) -> dict[str, float]:
        """Run validation and compute metrics.

        Returns:
            Dictionary of validation metrics.  Always contains ``"loss"``
            plus all keys returned by :meth:`SegmentationMetrics.compute`.
        """
        self.model.eval()
        self._val_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for raw_batch in self.val_loader:
                batch = _batch_to_device(raw_batch, self.device)
                images = batch["images"]
                masks = batch["masks"]

                logits = self.model(images)
                loss = self.loss_fn(logits, masks)

                self._val_metrics.update(logits, masks)
                total_loss += loss.item()
                num_batches += 1

        metrics = self._val_metrics.compute()
        metrics["loss"] = total_loss / max(num_batches, 1)
        return metrics

    def fit(self, num_epochs: int) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            num_epochs: Total number of training epochs to execute.

        Returns:
            Dictionary mapping metric names (prefixed with ``"train_"``
            or ``"val_"``) to lists of per-epoch float values.
        """
        history: dict[str, list[float]] = defaultdict(list)

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            for key, value in train_metrics.items():
                history[f"train_{key}"].append(value)
            for key, value in val_metrics.items():
                history[f"val_{key}"].append(value)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f train_dice=%.4f val_dice=%.4f",
                epoch,
                num_epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                train_metrics["dice"],
                val_metrics["dice"],
            )

        return dict(history)


# ---------------------------------------------------------------------------
# Detection Trainer
# ---------------------------------------------------------------------------


class DetectionTrainer:
    """Trainer for anchor-free dual-head object detection models.

    Manages training and validation loops for detection with dual
    One-to-Many (O2M) and One-to-One (O2O) heads.  All loss computation
    and metric evaluation run entirely on GPU via ``torchvision.ops``.

    Attributes:
        model: The detection model moved to the target device.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        loss_fn: Dual-head detection loss instance.
        optimizer: Optimizer for parameter updates.
        num_classes: Number of foreground object classes.
        device: Target device for computation.
        scheduler: Optional learning rate scheduler.
    """

    def __init__(  # noqa: PLR0913
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: DualHeadDetectionLoss,
        optimizer: Optimizer,
        num_classes: int,
        device: str | torch.device = "cuda",
        scheduler: LRScheduler | None = None,
        conf_threshold: float = 0.05,
    ) -> None:
        """Initialize the detection trainer.

        Args:
            model: Detection model mapping ``(B, 3, H, W)`` to a tuple
                of ``(preds_o2m, preds_o2o)`` predictions.
            train_loader: Training data loader yielding dicts with
                ``"images"`` and ``"targets"`` keys.
            val_loader: Validation data loader yielding dicts with
                ``"images"`` and ``"targets"`` keys.
            loss_fn: :class:`DualHeadDetectionLoss` instance accepting
                ``(preds_o2m, preds_o2o, targets)``.
            optimizer: Optimizer instance for gradient-based updates.
            num_classes: Number of foreground object classes.
            device: Target device string or :class:`torch.device`.
            scheduler: Optional LR scheduler stepped after each epoch.
            conf_threshold: Confidence threshold for metric computation.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.scheduler = scheduler
        self.conf_threshold = conf_threshold
        self._train_metrics = DetectionMetrics(num_classes=num_classes, conf_threshold=conf_threshold)
        self._val_metrics = DetectionMetrics(num_classes=num_classes, conf_threshold=conf_threshold)

    @staticmethod
    def _format_o2o_predictions(
        logits_o2o: Tensor,
        boxes_o2o: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Convert raw O2O head outputs to metric-compatible format.

        Applies sigmoid to logits and extracts per-anchor class scores
        and labels via argmax over classes.

        Args:
            logits_o2o: Raw O2O logits of shape ``(B, N, C)``.
            boxes_o2o: Predicted boxes of shape ``(B, N, 4)``.

        Returns:
            Tuple of ``(boxes, scores, labels)`` with shapes
            ``(B, N, 4)``, ``(B, N)``, and ``(B, N)``.
        """
        scores_all = logits_o2o.sigmoid()
        max_scores, pred_labels = scores_all.max(dim=-1)
        return boxes_o2o, max_scores, pred_labels.long()

    def train_epoch(self) -> dict[str, float]:
        """Execute a single training epoch.

        Returns:
            Dictionary of training metrics for this epoch.  Always
            contains ``"loss"`` (total combined loss).
        """
        self.model.train()
        self._train_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        for raw_batch in self.train_loader:
            batch = _batch_to_device(raw_batch, self.device)
            images = batch["images"]
            targets = batch["targets"]

            self.optimizer.zero_grad()
            preds_o2m, preds_o2o = self.model(images)
            loss_dict = self.loss_fn(preds_o2m, preds_o2o, targets)
            loss = loss_dict["loss_total"]

            # Backward only when loss is connected to the computation graph.
            # With randomly initialised models and sparse targets, all
            # assigners may produce zero positives, yielding a detached
            # zero-loss scalar with no grad_fn.
            if loss.requires_grad:
                loss.backward()
                self.optimizer.step()

            logits_o2o, boxes_o2o = preds_o2o
            formatted = self._format_o2o_predictions(
                logits_o2o.detach(),
                boxes_o2o.detach(),
            )
            self._train_metrics.update(formatted, targets)
            total_loss += loss.item()
            num_batches += 1

        metrics: dict[str, float] = {"loss": total_loss / max(num_batches, 1)}
        return metrics

    def validate(self) -> dict[str, float]:
        """Run validation and compute detection metrics.

        Returns:
            Dictionary of validation metrics.  Always contains ``"loss"``
            plus all keys returned by :meth:`DetectionMetrics.compute`
            (``"mAP50"``, ``"mAP50-95"``, ``"precision"``, ``"recall"``,
            ``"f1"``).
        """
        self.model.eval()
        self._val_metrics.reset()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for raw_batch in self.val_loader:
                batch = _batch_to_device(raw_batch, self.device)
                images = batch["images"]
                targets = batch["targets"]

                preds_o2m, preds_o2o = self.model(images)
                loss_dict = self.loss_fn(preds_o2m, preds_o2o, targets)
                loss = loss_dict["loss_total"]

                logits_o2o, boxes_o2o = preds_o2o
                formatted = self._format_o2o_predictions(logits_o2o, boxes_o2o)
                self._val_metrics.update(formatted, targets)
                total_loss += loss.item()
                num_batches += 1

        metrics = self._val_metrics.compute()
        metrics["loss"] = total_loss / max(num_batches, 1)
        return metrics

    def fit(self, num_epochs: int) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            num_epochs: Total number of training epochs to execute.

        Returns:
            Dictionary mapping metric names (prefixed with ``"train_"``
            or ``"val_"``) to lists of per-epoch float values.
        """
        history: dict[str, list[float]] = defaultdict(list)

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            for key, value in train_metrics.items():
                history[f"train_{key}"].append(value)
            for key, value in val_metrics.items():
                history[f"val_{key}"].append(value)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f val_mAP50=%.4f val_mAP50-95=%.4f",
                epoch,
                num_epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["mAP50"],
                val_metrics["mAP50-95"],
            )

        return dict(history)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Constants for the sanity check
    _NUM_CLASSES_CLS = 4
    _NUM_CLASSES_SEG = 4
    _NUM_CLASSES_DET = 4
    _IMG_SIZE = 32
    _BATCH_SIZE = 2
    _NUM_SAMPLES = 8
    _NUM_EPOCHS = 2
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Dummy classification model and data
    # ------------------------------------------------------------------

    class _DummyClsModel(nn.Module):
        """Minimal classification model for sanity checking."""

        def __init__(self, num_classes: int) -> None:
            """Initialize with adaptive average pooling and a linear head."""
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(3, num_classes)

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass: pool then project to class logits."""
            return self.fc(self.pool(x).flatten(1))

    class _FakeClsDataset(torch.utils.data.Dataset):
        """Fake classification dataset producing random images and labels."""

        def __init__(self, n: int, num_classes: int, img_size: int) -> None:
            """Pre-generate random images and labels."""
            self.images = torch.randn(n, 3, img_size, img_size)
            self.labels = torch.randint(0, num_classes, (n,))

        def __len__(self) -> int:
            """Return the number of samples."""
            return self.labels.shape[0]

        def __getitem__(self, idx: int) -> dict[str, Tensor]:
            """Return a single sample as a dictionary."""
            return {
                "images": self.images[idx],
                "labels": self.labels[idx],
                "image_ids": torch.tensor(idx),
            }

    # ------------------------------------------------------------------
    # Dummy segmentation model and data
    # ------------------------------------------------------------------

    class _DummySegModel(nn.Module):
        """Minimal segmentation model for sanity checking."""

        def __init__(self, num_classes: int) -> None:
            """Initialize with a single 1x1 convolution head."""
            super().__init__()
            self.head = nn.Conv2d(3, num_classes, kernel_size=1)

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass: pixel-wise classification via convolution."""
            return self.head(x)

    class _FakeSegDataset(torch.utils.data.Dataset):
        """Fake segmentation dataset producing random images and masks."""

        def __init__(self, n: int, num_classes: int, img_size: int) -> None:
            """Pre-generate random images and masks."""
            self.images = torch.randn(n, 3, img_size, img_size)
            self.masks = torch.randint(0, num_classes, (n, img_size, img_size))

        def __len__(self) -> int:
            """Return the number of samples."""
            return self.images.shape[0]

        def __getitem__(self, idx: int) -> dict[str, Tensor]:
            """Return a single sample as a dictionary."""
            return {
                "images": self.images[idx],
                "masks": self.masks[idx],
                "image_ids": torch.tensor(idx),
            }

    # ------------------------------------------------------------------
    # Dummy detection model and data
    # ------------------------------------------------------------------

    class _DummyDetModel(nn.Module):
        """Minimal dual-head detection model for sanity checking."""

        def __init__(self, num_classes: int, num_anchors: int = 50) -> None:
            """Initialize with shared pooling and separate cls/box heads."""
            super().__init__()
            self.num_classes = num_classes
            self.num_anchors = num_anchors
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.cls_head = nn.Linear(3, num_anchors * num_classes)
            self.box_head = nn.Linear(3, num_anchors * 4)

        def forward(
            self,
            x: Tensor,
        ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
            """Forward pass returning (preds_o2m, preds_o2o) tuples."""
            batch_size = x.shape[0]
            pooled = self.pool(x).flatten(1)
            logits = self.cls_head(pooled).reshape(
                batch_size,
                self.num_anchors,
                self.num_classes,
            )
            boxes = (
                self.box_head(pooled)
                .reshape(
                    batch_size,
                    self.num_anchors,
                    4,
                )
                .sigmoid()
            )
            return (logits, boxes), (logits.clone(), boxes.clone())

    class _FakeDetDataset(torch.utils.data.Dataset):
        """Fake detection dataset producing random images and targets."""

        def __init__(self, n: int, num_classes: int, img_size: int) -> None:
            """Store parameters for on-the-fly sample generation."""
            self.n = n
            self.num_classes = num_classes
            self.img_size = img_size

        def __len__(self) -> int:
            """Return the number of samples."""
            return self.n

        def __getitem__(self, idx: int) -> dict[str, Tensor]:
            """Return a single sample with variable-size targets."""
            image = torch.randn(3, self.img_size, self.img_size)
            num_gt = torch.randint(1, 4, (1,)).item()
            boxes = torch.rand(num_gt, 4) * 0.5
            boxes[:, 2:] = boxes[:, :2] + 0.1
            labels = torch.randint(0, self.num_classes, (num_gt, 1)).float()
            targets = torch.cat([labels, boxes], dim=1)
            return {"images": image, "targets": targets, "image_ids": torch.tensor(idx)}

    def _det_collate(
        batch: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """Collate detection samples by adding batch indices to targets.

        Args:
            batch: List of sample dictionaries from the dataset.

        Returns:
            Collated dictionary with flat targets tensor ``(N, 6)``.
        """
        images = torch.stack([sample["images"] for sample in batch])
        all_targets: list[Tensor] = []
        for batch_idx, sample in enumerate(batch):
            per_image = sample["targets"]
            batch_col = torch.full(
                (per_image.shape[0], 1),
                float(batch_idx),
            )
            all_targets.append(torch.cat([batch_col, per_image], dim=1))
        return {
            "images": images,
            "targets": torch.cat(all_targets, dim=0),
            "image_ids": torch.stack([sample["image_ids"] for sample in batch]),
        }

    # ------------------------------------------------------------------
    # Build data loaders
    # ------------------------------------------------------------------

    cls_train_loader = DataLoader(
        _FakeClsDataset(_NUM_SAMPLES, _NUM_CLASSES_CLS, _IMG_SIZE),
        batch_size=_BATCH_SIZE,
        shuffle=True,
    )
    cls_val_loader = DataLoader(
        _FakeClsDataset(_NUM_SAMPLES, _NUM_CLASSES_CLS, _IMG_SIZE),
        batch_size=_BATCH_SIZE,
    )

    seg_train_loader = DataLoader(
        _FakeSegDataset(_NUM_SAMPLES, _NUM_CLASSES_SEG, _IMG_SIZE),
        batch_size=_BATCH_SIZE,
        shuffle=True,
    )
    seg_val_loader = DataLoader(
        _FakeSegDataset(_NUM_SAMPLES, _NUM_CLASSES_SEG, _IMG_SIZE),
        batch_size=_BATCH_SIZE,
    )

    det_train_loader = DataLoader(
        _FakeDetDataset(_NUM_SAMPLES, _NUM_CLASSES_DET, _IMG_SIZE),
        batch_size=_BATCH_SIZE,
        shuffle=True,
        collate_fn=_det_collate,
    )
    det_val_loader = DataLoader(
        _FakeDetDataset(_NUM_SAMPLES, _NUM_CLASSES_DET, _IMG_SIZE),
        batch_size=_BATCH_SIZE,
        collate_fn=_det_collate,
    )

    # ------------------------------------------------------------------
    # Build models, losses, optimizers
    # ------------------------------------------------------------------

    cls_model = _DummyClsModel(_NUM_CLASSES_CLS).to(_DEVICE)
    seg_model = _DummySegModel(_NUM_CLASSES_SEG).to(_DEVICE)
    det_model = _DummyDetModel(_NUM_CLASSES_DET).to(_DEVICE)

    cls_loss_fn = ClassificationCrossEntropyLoss()
    seg_loss_fn = SegmentationCrossEntropyLoss()
    det_loss_fn = DualHeadDetectionLoss(num_classes=_NUM_CLASSES_DET)

    cls_optimizer = torch.optim.SGD(cls_model.parameters(), lr=0.01)
    seg_optimizer = torch.optim.SGD(seg_model.parameters(), lr=0.01)
    det_optimizer = torch.optim.SGD(det_model.parameters(), lr=0.01)

    cls_scheduler = torch.optim.lr_scheduler.StepLR(cls_optimizer, step_size=1, gamma=0.9)

    # ------------------------------------------------------------------
    # Instantiate trainers and run
    # ------------------------------------------------------------------

    print("=== Classification Trainer ===")  # noqa: T201
    cls_trainer = ClassificationTrainer(
        model=cls_model,
        train_loader=cls_train_loader,
        val_loader=cls_val_loader,
        loss_fn=cls_loss_fn,
        optimizer=cls_optimizer,
        num_classes=_NUM_CLASSES_CLS,
        device=_DEVICE,
        scheduler=cls_scheduler,
    )
    cls_history = cls_trainer.fit(num_epochs=_NUM_EPOCHS)
    print(f"Classification history keys: {list(cls_history.keys())}\n")  # noqa: T201

    print("=== Segmentation Trainer ===")  # noqa: T201
    seg_trainer = SegmentationTrainer(
        model=seg_model,
        train_loader=seg_train_loader,
        val_loader=seg_val_loader,
        loss_fn=seg_loss_fn,
        optimizer=seg_optimizer,
        num_classes=_NUM_CLASSES_SEG,
        device=_DEVICE,
    )
    seg_history = seg_trainer.fit(num_epochs=_NUM_EPOCHS)
    print(f"Segmentation history keys: {list(seg_history.keys())}\n")  # noqa: T201

    print("=== Detection Trainer ===")  # noqa: T201
    det_trainer = DetectionTrainer(
        model=det_model,
        train_loader=det_train_loader,
        val_loader=det_val_loader,
        loss_fn=det_loss_fn,
        optimizer=det_optimizer,
        num_classes=_NUM_CLASSES_DET,
        device=_DEVICE,
    )
    det_history = det_trainer.fit(num_epochs=_NUM_EPOCHS)
    print(f"Detection history keys: {list(det_history.keys())}\n")  # noqa: T201

    print("All trainers completed successfully.")  # noqa: T201

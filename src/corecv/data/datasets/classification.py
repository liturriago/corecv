"""Image classification dataset with Albumentations transform support.

Provides a standard ImageFolder-style dataset that loads images from
class-named subdirectories, applies optional Albumentations transforms
(including those from :mod:`corecv.data.transforms`), and returns
``(image, label)`` tuples where *image* is a ``[C, H, W]`` tensor and
*label* is an integer class index.

The dataset is designed to integrate seamlessly with
:class:`~corecv.data.transforms.CoordinatedTransform` but also accepts
any callable that receives an ``image`` keyword argument and returns a
result with an ``image`` attribute (e.g. :class:`~albumentations.Compose`
which returns a dict with an ``"image"`` key).

Example:
    >>> from corecv.data.datasets import ClassificationDataset
    >>> dataset = ClassificationDataset("path/to/data")
    >>> image, label = dataset[0]
    >>> image.shape
    torch.Size([3, 224, 224])
    >>> label
    0
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from corecv.data.transforms import TransformOutput

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tiff",
        ".tif",
    }
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ClassificationDataset:
    """Image classification dataset following the ImageFolder convention.

    Organises images by class subdirectories under a *root* directory.
    Supports optional Albumentations transforms (e.g.
    :class:`~corecv.data.transforms.CoordinatedTransform`) and returns
    ``(image, label)`` tuples ready for model training.

    Attributes:
        root: Root directory containing class-named subdirectories.
        transform: Optional callable that accepts ``image`` as a keyword
            argument and returns a transformed result with an ``image``
            attribute or ``"image"`` key.
        classes: Sorted list of class names derived from folder names.
        class_to_idx: Mapping from class name to integer index.
        samples: List of ``(filepath, class_index)`` tuples.
    """

    def __init__(
        self,
        root: str,
        transform: Callable[..., object] | None = None,
    ) -> None:
        """Initialise the dataset by scanning *root* for class folders.

        Args:
            root: Path to the dataset root directory containing
                class-named subdirectories.
            transform: Optional callable transform pipeline.  Can be a
                :class:`~corecv.data.transforms.CoordinatedTransform`,
                an :class:`~albumentations.Compose` pipeline, or any
                callable that accepts ``image`` as a keyword argument
                and returns a result containing the transformed image.
                If ``None``, images are returned as raw ``[C, H, W]``
                tensors without any augmentation or normalisation.

        Raises:
            FileNotFoundError: If *root* does not exist.
        """
        self.root: Path = Path(root)
        self.transform: Callable[..., object] | None = transform

        # Discover classes from sorted subdirectory names
        classes: list[str] = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.classes: list[str] = classes
        self.class_to_idx: dict[str, int] = {cls: idx for idx, cls in enumerate(classes)}

        # Build the flat sample list
        self.samples: list[tuple[str, int]] = []
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir: Path = self.root / cls_name
            for fpath in sorted(cls_dir.iterdir()):
                if fpath.suffix.lower() in _IMAGE_EXTENSIONS:
                    self.samples.append((str(fpath), cls_idx))

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            The number of image-label pairs.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return the image-label pair at the given index.

        Loads the image from disk, applies the configured transform
        pipeline (if any), converts the resulting array to a
        ``[C, H, W]`` tensor, and returns it alongside the class index.

        Args:
            index: Sample index (0-based).

        Returns:
            A tuple ``(image, label)`` where *image* is a ``[C, H, W]``
            tensor and *label* is an integer class index.
        """
        path: str
        label: int
        path, label = self.samples[index]

        # Load image as RGB numpy array (H, W, C)
        image: np.ndarray = np.array(Image.open(path).convert("RGB"))

        # Apply transforms if configured
        if self.transform is not None:
            result: object = self.transform(image=image)
            image_arr: np.ndarray = self._extract_image(result)
        else:
            image_arr = image

        # Convert HWC numpy array to CHW tensor
        image_tensor: torch.Tensor = torch.from_numpy(image_arr).permute(2, 0, 1)

        return image_tensor, label

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_image(result: object) -> np.ndarray:
        """Extract the image array from a transform result.

        Supports :class:`TransformOutput` (``.image`` attribute),
        :class:`dict` (``"image"`` key), and raw numpy arrays.

        Args:
            result: The output of the transform pipeline.

        Returns:
            The transformed image as a numpy array with shape
            ``(H, W, C)``.

        Raises:
            TypeError: If *result* is not a recognised type and cannot
                be used as an image directly.
        """
        if isinstance(result, TransformOutput):
            return result.image
        if isinstance(result, dict):
            img: object = result["image"]
            if isinstance(img, np.ndarray):
                return img
            msg: str = f"Expected numpy array under 'image' key, got {type(img).__name__}."
            raise TypeError(msg)
        if isinstance(result, np.ndarray):
            return result
        msg: str = (
            f"Unsupported transform result type: {type(result).__name__}. "
            f"Expected TransformOutput, dict with 'image' key, or numpy array."
        )
        raise TypeError(msg)

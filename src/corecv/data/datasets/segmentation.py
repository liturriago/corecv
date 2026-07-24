"""Semantic segmentation dataset with on-disk mask loading and cache manifest.

Provides :class:`SegmentationDataset` for reading paired images and
uint8-indexed masks from ``images/`` and ``masks/`` subdirectories with
matched 1:1 filenames.  The dataset implements an O(N) pre-flight check
that validates all annotations and generates a compressed ``.npz`` cache
file containing **only** a manifest of validated pairs with file
modification timestamps — no mask pixel arrays are stored in the cache.

Masks are read from disk dynamically at ``__getitem__`` time, avoiding
memory explosion on large datasets.

Example:
    >>> from corecv.data.datasets import SegmentationDataset
    >>> dataset = SegmentationDataset("path/to/data", num_classes=21)
    >>> image, mask = dataset[0]
    >>> image.shape
    torch.Size([3, 480, 640])
    >>> mask.shape
    torch.Size([480, 640])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from corecv.data.transforms import (
    CoordinatedTransform,
    SegmentationTransformConfig,
    build_transforms,
)

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

_CACHE_VERSION: int = 2  # bumped when mask pixel storage was removed from cache
_CACHE_FILENAME: str = "_segmentation_cache.npz"

# Dimensionality and shape-index constants for mask validation
_NDIM_2D: int = 2
_NDIM_3D: int = 3
_CHANNEL_DIM_3D: int = 2  # channel axis in (H, W, C) layout
_SINGLE_CHANNEL: int = 1

_ERROR_MASK_MISSING: str = (
    "Missing mask for image '{image_name}' in {masks_dir}. "
    "Every image in images/ must have a corresponding mask in masks/ "
    "with a matching filename stem."
)
_ERROR_MASK_DTYPE: str = (
    "Mask at {path} has dtype {dtype}, expected uint8. "
    "Segmentation masks must be uint8 indexed arrays "
    "with values in [0, {num_classes})."
)
_ERROR_MASK_SHAPE: str = (
    "Mask at {path} has shape {shape}, expected 2D (H, W) or "
    "3D with 1 channel (H, W, 1)."
)
_ERROR_MASK_VALUE: str = (
    "Mask at {path} contains value {value} at index ({idx0}, {idx1}), "
    "which is outside the valid range [0, {num_classes})"
    "{ignore_suffix}."
)
_ERROR_EMPTY_DATASET: str = (
    "No valid image-mask pairs found in {root}. "
    "Ensure the directory contains both images/ and masks/ "
    "subdirectories with matching filenames."
)
_ERROR_IMAGES_DIR: str = (
    "Images directory not found: {path}. "
    "SegmentationDataset expects root/images/ to exist."
)
_ERROR_MASKS_DIR: str = (
    "Masks directory not found: {path}. "
    "SegmentationDataset expects root/masks/ to exist."
)

# ---------------------------------------------------------------------------
# Metadata structured dtype for cache
# ---------------------------------------------------------------------------

_METADATA_DTYPE: np.dtype = np.dtype(
    [
        ("image_path", "U512"),
        ("mask_path", "U512"),
        ("image_mtime", "i8"),
        ("mask_mtime", "i8"),
    ],
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SegmentationDataset(Dataset[tuple[Tensor, Tensor]]):
    """PyTorch Dataset for semantic segmentation with on-disk mask loading.

    Expects the following directory structure under *root*::

        root/
            images/
                image_001.jpg
                image_002.png
                ...
            masks/
                image_001.png    # uint8, class indices 0..num_classes-1
                image_002.png
                ...

    **Features**:

    - 1:1 matched filename pairing between ``images/`` and ``masks/``.
    - Masks are validated as uint8 with values in ``[0, num_classes)``
      (plus an optional ``ignore_index``).
    - O(N) pre-flight check generates a ``.npz`` cache on first run.
    - The cache stores **only** a manifest of validated pairs with file
      modification timestamps — no mask pixel arrays are persisted,
      avoiding memory explosion on large datasets.
    - Mask pixel data is read from disk dynamically at ``__getitem__``
      time via PIL in uint8 format.
    - Cache invalidation via filesystem mtime comparison — any change
      to a source file triggers a rebuild.
    - Optional :class:`~corecv.data.transforms.CoordinatedTransform`
      integration for synchronised image + mask augmentations.

    Attributes:
        root: Resolved path to the dataset root directory.
        num_classes: Number of semantic classes (used for validation).
        transforms: Optional transform pipeline for synchronised
            image + mask augmentations.
        ignore_index: Optional label value that is excluded from
            range validation (e.g. ``255`` for ignore/crop border).
        image_paths: List of resolved image file paths.
        mask_paths: List of resolved mask file paths.
        cache_path: Path to the on-disk ``.npz`` cache file.
    """

    def __init__(  # noqa: PLR0913
        self,
        root: str | Path,
        num_classes: int,
        transforms: CoordinatedTransform | bool | None = None,
        transform: CoordinatedTransform | bool | None = None,
        image_size: tuple[int, int] | None = None,
        cache_dir: str | Path | None = None,
        ignore_index: int | None = None,
    ) -> None:
        """Initialise the dataset, validating pairs and loading / building cache.

        Args:
            root: Root directory containing ``images/`` and ``masks/``
                subdirectories.
            num_classes: Number of semantic classes.
            transforms: Transform pipeline or boolean flag. Pass ``True``
                to enable standard default segmentation augmentations
                (random flip, rotation, resize, ImageNet normalization).
            transform: Alias for *transforms*.
            image_size: Target ``(height, width)`` tuple.
            cache_dir: Directory for cache file. Defaults to *root*.
            ignore_index: Optional class index to exclude from range validation.
        """
        self._root: Path = Path(root).resolve().absolute()
        self._num_classes: int = num_classes
        self._ignore_index: int | None = ignore_index
        self._image_size: tuple[int, int] | None = image_size

        tf: CoordinatedTransform | bool | None = (
            transforms if transforms is not None else transform
        )
        if tf is True:
            target_size: tuple[int, int] = image_size or (512, 512)
            self._transforms: CoordinatedTransform | None = build_transforms(
                SegmentationTransformConfig(
                    image_size=target_size,
                    ignore_index=ignore_index if ignore_index is not None else 255,
                    horizontal_flip_p=0.5,
                    rotate_limit=15,
                )
            )
        elif callable(tf):
            self._transforms = tf
        else:
            self._transforms = None

        self._images_dir: Path = self._root / "images"
        self._masks_dir: Path = self._root / "masks"

        # Validate directories exist
        if not self._images_dir.is_dir():
            raise NotADirectoryError(
                _ERROR_IMAGES_DIR.format(path=self._images_dir)
            )
        if not self._masks_dir.is_dir():
            raise NotADirectoryError(
                _ERROR_MASKS_DIR.format(path=self._masks_dir)
            )

        # Cache path
        cache_dir_resolved: Path = (
            Path(cache_dir).resolve().absolute()
            if cache_dir is not None
            else self._root
        )
        self._cache_path: Path = cache_dir_resolved / _CACHE_FILENAME

        # Internal state populated by cache load / build
        self._image_paths: list[Path] = []
        self._mask_paths: list[Path] = []

        # Run pre-flight check: validate pairs and load / build cache
        self._load_or_build_cache()

    # ------------------------------------------------------------------
    # Public API: Dataset
    # ------------------------------------------------------------------

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return a single (image, mask) pair at the given index.

        Both the image and mask are read from disk dynamically.  The
        image is returned as a ``[C, H, W]`` float32 tensor and the
        mask as a ``[H, W]`` ``torch.long`` tensor.

        Args:
            index: Sample index (0-based).

        Returns:
            A tuple ``(image_tensor, mask_tensor)``.
        """
        image_path: Path = self._image_paths[index]
        mask_path: Path = self._mask_paths[index]

        # Load image from disk (uint8 HWC)
        image: np.ndarray = self._load_image(image_path)

        # Load mask from disk (uint8 HW)
        mask: np.ndarray = self._load_mask(mask_path)

        # Apply synchronised transforms if configured
        if self._transforms is not None:
            result = self._transforms(image=image, mask=mask)
            image = result.image
            mask = result.mask  # type: ignore[assignment]
            image_tensor: Tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            # Safe default fallback when transforms=False / None:
            if self._image_size is not None:
                h, w = image.shape[:2]
                if (h, w) != self._image_size:
                    img_pil = Image.fromarray(image).resize(
                        (self._image_size[1], self._image_size[0]),
                        Image.Resampling.BILINEAR,
                    )
                    mask_pil = Image.fromarray(mask).resize(
                        (self._image_size[1], self._image_size[0]),
                        Image.Resampling.NEAREST,
                    )
                    image = np.array(img_pil, dtype=np.uint8)
                    mask = np.array(mask_pil, dtype=np.uint8)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        mask_tensor: Tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Total number of valid image-mask pairs.
        """
        return len(self._image_paths)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def image_paths(self) -> list[Path]:
        """Return the list of resolved image file paths."""
        return list(self._image_paths)

    @property
    def mask_paths(self) -> list[Path]:
        """Return the list of resolved mask file paths."""
        return list(self._mask_paths)

    @property
    def root(self) -> Path:
        """Return the dataset root directory."""
        return self._root

    @property
    def num_classes(self) -> int:
        """Return the number of semantic classes."""
        return self._num_classes

    @property
    def transforms(self) -> CoordinatedTransform | None:
        """Return the optional transform pipeline."""
        return self._transforms

    @property
    def ignore_index(self) -> int | None:
        """Return the optional ignore index."""
        return self._ignore_index

    @property
    def cache_path(self) -> Path:
        """Return the path to the on-disk ``.npz`` cache file."""
        return self._cache_path

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _load_or_build_cache(self) -> None:
        """Load the cache if valid, otherwise perform pre-flight and build it.

        Raises:
            RuntimeError: If no valid image-mask pairs are found.
        """
        if self._cache_path.exists() and self._is_cache_valid():
            self._load_cache()
        else:
            self._build_cache()

        if len(self._image_paths) == 0:
            raise RuntimeError(
                _ERROR_EMPTY_DATASET.format(root=self._root)
            )

    def _is_cache_valid(self) -> bool:
        """Check whether the on-disk cache is still valid.

        Compares the stored mtimes of all source files against current
        filesystem mtimes.  Also verifies all files still exist.

        Returns:
            ``True`` if the cache is valid and may be loaded.
        """
        valid: bool = False
        try:
            with np.load(self._cache_path) as loader:
                cached_version: Any = loader.get("cache_version")
                metadata: Any = loader.get("metadata")
                if cached_version == _CACHE_VERSION and metadata is not None and metadata.size > 0:
                    valid = self._check_metadata_mtimes(metadata)
        except Exception:  # noqa: BLE001
            valid = False
        return valid

    @staticmethod
    def _check_metadata_mtimes(metadata: np.ndarray) -> bool:
        """Verify that all files referenced in *metadata* exist and have matching mtimes.

        Args:
            metadata: Structured array with fields ``image_path``,
                ``mask_path``, ``image_mtime``, ``mask_mtime``.

        Returns:
            ``True`` if every file exists and its mtime matches the
            cached value.
        """
        for row in metadata:
            img_path: Path = Path(str(row["image_path"]))
            mask_path: Path = Path(str(row["mask_path"]))
            img_mtime: int = int(row["image_mtime"])
            mask_mtime: int = int(row["mask_mtime"])

            if not img_path.exists():
                return False
            if not mask_path.exists():
                return False
            if img_path.stat().st_mtime_ns != img_mtime:
                return False
            if mask_path.stat().st_mtime_ns != mask_mtime:
                return False
        return True

    def _load_cache(self) -> None:
        """Load image/mask paths from the on-disk cache manifest.

        Only the pair manifest (paths + mtimes) is cached.  Mask pixel
        data is read from disk dynamically at ``__getitem__`` time.
        """
        with np.load(self._cache_path) as loader:
            metadata: np.ndarray = loader["metadata"]

            self._image_paths = [Path(str(row["image_path"])) for row in metadata]
            self._mask_paths = [Path(str(row["mask_path"])) for row in metadata]

    def _build_cache(self) -> None:
        """Scan directories, validate pairs, and build the compressed cache manifest.

        Performs an O(N) pre-flight check:

        1. Discovers all ``images/`` files and matches them 1:1 with
           ``masks/`` files by filename stem.
        2. Loads each mask and validates dtype (uint8), shape (2D or 3D
           single-channel), and value range ``[0, num_classes)``
           (plus optional *ignore_index*).
        3. Stores **only** the pair manifest (paths + mtimes) in a
           compressed ``.npz`` cache file — **no mask pixel arrays**
           are persisted.

        Raises:
            FileNotFoundError: If a mask is missing for an image.
            ValueError: If a mask has invalid dtype, dimensions, or
                out-of-range values.
        """
        image_files: list[Path] = sorted(
            p
            for p in self._images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )

        metadata_list: list[tuple[str, str, int, int]] = []
        image_paths_list: list[Path] = []
        mask_paths_list: list[Path] = []

        for img_path in image_files:
            stem: str = img_path.stem

            # Find matching mask by filename stem
            mask_candidates: list[Path] = sorted(
                p for p in self._masks_dir.iterdir()
                if p.stem == stem and p.is_file()
            )
            if not mask_candidates:
                raise FileNotFoundError(
                    _ERROR_MASK_MISSING.format(
                        image_name=img_path.name,
                        masks_dir=self._masks_dir,
                    )
                )

            # Prefer .png, otherwise use the first match
            mask_path: Path = mask_candidates[0]
            for candidate in mask_candidates:
                if candidate.suffix.lower() == ".png":
                    mask_path = candidate
                    break

            # Validate mask (dtype, shape, value range) before caching
            self._load_mask(mask_path)

            image_paths_list.append(img_path)
            mask_paths_list.append(mask_path)

            img_mtime: int = int(img_path.stat().st_mtime_ns)
            mask_mtime: int = int(mask_path.stat().st_mtime_ns)
            metadata_list.append((str(img_path), str(mask_path), img_mtime, mask_mtime))

        self._image_paths = image_paths_list
        self._mask_paths = mask_paths_list

        if len(self._image_paths) == 0:
            return

        # Build structured metadata array
        metadata_arr: np.ndarray = np.array(
            metadata_list,
            dtype=_METADATA_DTYPE,
        )

        # Build save dictionary — only metadata, no mask pixel arrays
        save_dict: dict[str, Any] = {
            "cache_version": _CACHE_VERSION,
            "metadata": metadata_arr,
        }

        # Ensure cache directory exists
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Write compressed archive
        np.savez_compressed(str(self._cache_path), **save_dict)

    # ------------------------------------------------------------------
    # Image / mask loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        """Load an image from disk as a uint8 RGB numpy array.

        Args:
            path: Path to the image file.

        Returns:
            Image array with shape ``(H, W, 3)`` and dtype ``uint8``.
        """
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            return np.array(img_rgb, dtype=np.uint8)

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load a segmentation mask, validating dtype, shape and values.

        Args:
            path: Path to the mask file.

        Returns:
            Validated mask array with shape ``(H, W)`` and dtype ``uint8``.

        Raises:
            ValueError: If the mask has non-uint8 dtype, unexpected
                dimensionality, or values outside ``[0, num_classes)``
                (excluding *ignore_index* if set).
        """
        with Image.open(path) as msk:
            mask: np.ndarray = np.array(msk)

        # Validate dtype
        if mask.dtype != np.uint8:
            raise ValueError(
                _ERROR_MASK_DTYPE.format(
                    path=path,
                    dtype=mask.dtype,
                    num_classes=self._num_classes,
                )
            )

        # Validate shape: must be 2D (H, W) or 3D with single channel
        if mask.ndim == _NDIM_3D:
            if mask.shape[_CHANNEL_DIM_3D] == _SINGLE_CHANNEL:
                mask = mask[..., 0]
            else:
                raise ValueError(
                    _ERROR_MASK_SHAPE.format(path=path, shape=mask.shape)
                )
        elif mask.ndim != _NDIM_2D:
            raise ValueError(
                _ERROR_MASK_SHAPE.format(path=path, shape=mask.shape)
            )

        # Validate value range
        ignore_suffix: str = (
            f" (excluding ignore_index={self._ignore_index})"
            if self._ignore_index is not None
            else ""
        )
        if self._ignore_index is not None:
            valid_mask_values: np.ndarray = mask[
                (mask < self._num_classes) | (mask == self._ignore_index)
            ]
        else:
            valid_mask_values = mask[mask < self._num_classes]

        if valid_mask_values.size != mask.size:
            # Find the first invalid value for the error message
            invalid: np.ndarray
            if self._ignore_index is not None:
                invalid = mask[
                    (mask >= self._num_classes) & (mask != self._ignore_index)
                ]
            else:
                invalid = mask[mask >= self._num_classes]

            if invalid.size > 0:
                bad_value: int = int(invalid.flat[0])
                bad_idx: tuple[int, ...] = tuple(
                    int(x) for x in np.argwhere(mask == bad_value)[0]
                )
                raise ValueError(
                    _ERROR_MASK_VALUE.format(
                        path=path,
                        value=bad_value,
                        idx0=bad_idx[0],
                        idx1=bad_idx[1],
                        num_classes=self._num_classes,
                        ignore_suffix=ignore_suffix,
                    )
                )

        return mask

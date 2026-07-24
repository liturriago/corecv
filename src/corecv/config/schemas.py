"""Configuration schemas for CoreCV tasks.

Provides polymorphic :func:`dataclasses.dataclass`-based configuration schemas
for classification, segmentation, and detection tasks, along with a factory
function for loading configuration from YAML files or dictionaries.
"""

import dataclasses
import pathlib
import types
from abc import ABC
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True, kw_only=True)
class BaseTaskConfig(ABC):
    """Abstract base configuration for all CV tasks.

    Attributes:
        task: Discriminant field identifying the task type
            (``"classification"``, ``"segmentation"``, or ``"detection"``).
        model_name: Backbone model name
            (e.g. ``"mobilenet_v3_large"``, ``"resnet50"``).
        pretrained: Whether to load pretrained weights.
        num_classes: Number of output classes.
        input_size: Input image dimensions as ``(height, width)``.
        target_hardware: Target hardware platform (``"gpu"`` or ``"edge"``).
        mixed_precision: Whether to enable automatic mixed precision.
    """

    task: str
    model_name: str
    pretrained: bool = True
    num_classes: int
    input_size: tuple[int, int] = (224, 224)
    target_hardware: str = "gpu"
    mixed_precision: bool = True


@dataclass(frozen=True)
class ClassificationConfig(BaseTaskConfig):
    """Configuration for image classification tasks.

    Attributes:
        task: Task type discriminant, always ``"classification"``.
        head_type: Classifier head type (``"linear"`` or ``"mlp"``).
        dropout: Dropout rate applied in the classifier head.
        label_smoothing: Label smoothing epsilon for loss computation.
    """

    task: Literal["classification"] = "classification"
    head_type: str = "linear"
    dropout: float = 0.0
    label_smoothing: float = 0.0


@dataclass(frozen=True)
class SegmentationConfig(BaseTaskConfig):
    """Configuration for semantic segmentation tasks.

    Attributes:
        task: Task type discriminant, always ``"segmentation"``.
        head_type: Segmentation head architecture
            (``"deeplabv3plus"``, ``"unet"``, or ``"fpn"``).
        backbone_out_indices: Indices of backbone feature maps to use.
        decoder_channels: Number of decoder channels in the segmentation head.
        align_corners: Whether to align corners in grid sampling.
        ignore_index: Target value to ignore during loss computation.
    """

    task: Literal["segmentation"] = "segmentation"
    head_type: str = "deeplabv3plus"
    backbone_out_indices: tuple[int, ...] = (0, 1, 2, 3)
    decoder_channels: int = 256
    align_corners: bool = False
    ignore_index: int = 255


@dataclass(frozen=True)
class DetectionConfig(BaseTaskConfig):
    """Configuration for object detection tasks.

    Attributes:
        task: Task type discriminant, always ``"detection"``.
        head_type: Detection head architecture
            (``"retinanet"``, ``"fcos"``, or ``"yolo"``).
        anchor_sizes: Anchor box sizes per feature level.
        aspect_ratios: Aspect ratios for anchor boxes.
        num_anchors: Number of anchor boxes per location.
        box_coder_weights: Weights for bounding box regression.
        score_thresh: Score threshold for detections.
        nms_thresh: NMS IoU threshold.
        detections_per_img: Maximum detections per image.
    """

    task: Literal["detection"] = "detection"
    head_type: str = "retinanet"
    anchor_sizes: tuple[tuple[int, ...], ...] = (
        (32,),
        (64,),
        (128,),
        (256,),
        (512,),
    )
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    num_anchors: int = 9
    box_coder_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    detections_per_img: int = 100


_TASK_MAP: dict[str, type[BaseTaskConfig]] = {
    "classification": ClassificationConfig,
    "segmentation": SegmentationConfig,
    "detection": DetectionConfig,
}


def _check_type(value: object, expected_type: object) -> bool:  # noqa: PLR0911
    """Recursively check whether a value matches an expected type annotation.

    Supports simple types, ``tuple``, ``Literal``, and ``Union`` /
    ``types.UnionType`` (``X | Y``).

    Args:
        value: The value to check.
        expected_type: The expected type annotation resolved via
            :func:`typing.get_type_hints`.

    Returns:
        ``True`` if *value* satisfies *expected_type*, ``False`` otherwise.
    """
    origin: object = get_origin(expected_type)
    args: tuple[Any, ...] = get_args(expected_type)

    # Plain type (e.g. ``str``, ``int``, the class itself)
    if origin is None:
        if expected_type is Any:
            return True
        return isinstance(value, expected_type)  # type: ignore[arg-type]

    # Literal, e.g. ``Literal["classification"]``
    if origin is Literal:
        return value in args

    # Union, e.g. ``str | dict`` or ``Union[str, dict]``
    if origin is Union or origin is types.UnionType:
        return any(_check_type(value, arg) for arg in args)

    # Tuple, e.g. ``tuple[int, int]`` or ``tuple[int, ...]``
    if origin is tuple:
        # YAML serializes tuples as lists; accept both.
        if not isinstance(value, (tuple, list)):
            return False
        if not args:
            return True
        # ``tuple[X, ...]`` (variable-length)
        if args[-1] is Ellipsis:
            item_type: object = args[0]
            return all(_check_type(item, item_type) for item in value)
        # ``tuple[X, Y, Z]`` (fixed-length)
        if len(value) != len(args):
            return False
        return all(
            _check_type(item, t) for item, t in zip(value, args, strict=True)
        )

    return True


def _load_raw_config(
    config_path_or_dict: str | dict[str, Any],
) -> dict[str, Any]:
    """Load and validate raw configuration from path or dictionary.

    Args:
        config_path_or_dict: Path to a YAML file or a configuration dict.

    Returns:
        The raw configuration dictionary.

    Raises:
        FileNotFoundError: If the YAML path does not exist.
        TypeError: If *config_path_or_dict* is neither ``str`` nor ``dict``,
            or if the YAML content is not a mapping.
        ValueError: If the YAML file cannot be parsed.
    """
    if isinstance(config_path_or_dict, dict):
        return config_path_or_dict

    if not isinstance(config_path_or_dict, str):
        raise TypeError(
            f"Expected str or dict, got {type(config_path_or_dict).__name__}"
        )

    path: pathlib.Path = pathlib.Path(config_path_or_dict)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise TypeError(
            f"YAML file must contain a top-level mapping (dict), "
            f"got {type(raw).__name__}"
        )

    return raw


def _validate_config(
    raw: dict[str, Any],
) -> BaseTaskConfig:
    """Validate a raw configuration dict and build the correct dataclass.

    Args:
        raw: The raw configuration dictionary.

    Returns:
        An instance of the appropriate task configuration dataclass.

    Raises:
        ValueError: If the configuration is invalid — unknown ``task``,
            missing required fields, extra unrecognised fields, or type
            mismatches.
    """
    # Determine task and target class
    task: Any = raw.get("task")
    if task not in _TASK_MAP:
        raise ValueError(
            f"Unknown task value: {task!r}. "
            f"Expected one of: {list(_TASK_MAP.keys())}"
        )
    cls: type[BaseTaskConfig] = _TASK_MAP[task]

    # Collect field metadata
    cls_fields: dict[str, dataclasses.Field[Any]] = {
        f.name: f for f in dataclasses.fields(cls)
    }
    cls_type_hints: dict[str, type] = get_type_hints(cls)

    # Check for extra / unrecognised fields
    for key in raw:
        if key not in cls_fields:
            raise ValueError(
                f"Unrecognised field {key!r} for {cls.__name__}. "
                f"Valid fields: {list(cls_fields.keys())}"
            )

    # Check for missing required fields
    for field_name, field_def in cls_fields.items():
        missing_default: bool = (
            field_def.default is dataclasses.MISSING
            and field_def.default_factory is dataclasses.MISSING
        )
        if missing_default and field_name not in raw:
            raise ValueError(
                f"Missing required field {field_name!r} for {cls.__name__}"
            )

    # Check type mismatches
    for key, value in raw.items():
        expected_type: object = cls_type_hints.get(key)
        if expected_type is not None and not _check_type(value, expected_type):
            raise ValueError(
                f"Type mismatch for field {key!r} of {cls.__name__}: "
                f"expected {expected_type}, got {type(value).__name__} "
                f"(value: {value!r})"
            )

    return cls(**raw)


def load_config(config_path_or_dict: str | dict[str, Any]) -> BaseTaskConfig:
    """Load and validate a task configuration.

    Accepts a path to a ``.yaml`` file or a native Python ``dict``.  The
    ``"task"`` key is used as a discriminant to select the correct
    configuration dataclass.

    Args:
        config_path_or_dict: Path to a YAML configuration file or a dictionary
            with configuration key-value pairs.

    Returns:
        An instance of the appropriate task configuration dataclass
        (:class:`ClassificationConfig`, :class:`SegmentationConfig`, or
        :class:`DetectionConfig`).

    Raises:
        FileNotFoundError: If *config_path_or_dict* is a string path that does
            not exist.
        TypeError: If *config_path_or_dict* is neither a ``str`` nor a
            ``dict``, or if the YAML content is not a mapping.
        ValueError: If the configuration is invalid — unknown ``task``,
            missing required fields, extra unrecognised fields, or type
            mismatches.
    """
    raw: dict[str, Any] = _load_raw_config(config_path_or_dict)
    return _validate_config(raw)

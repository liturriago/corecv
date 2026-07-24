# User Guide: Model Training

CoreCV provides a polymorphic training engine managed by `CoreTrainer` and accessible via `CoreModel.train()`. It supports configuration via YAML files, Python dictionaries, `TrainingConfig` dataclasses, or direct keyword arguments (`**kwargs`).

---

## 1. Complete Parameter Reference

### `CoreModel.__init__()` Parameters

| Parameter | Type Hint | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `nn.Module \| str \| Path \| dict` | *Required* | A CoreCV `nn.Module`, plain backbone string name (e.g. `"resnet18"`), raw config `dict`, path to `.pt`/`.pth` checkpoint, or path to `.yaml`/`.yml` configuration file. |
| `task` | `Literal["classification", "segmentation", "detection"]` | *Required* | Task type discriminant. Determines dataset structures, heads, and loss functions. |
| `input_size` | `tuple[int, int]` | `(224, 224)` | Target image dimensions `(height, width)` used for data preprocessing and transform pipelines. |
| `device` | `torch.device \| None` | `None` | Target execution device. If `None`, automatically detects CUDA when available, falling back to CPU. |
| `num_classes` | `int \| None` | `None` | Number of output classes. If `None`, automatically inferred from the model head or config metadata. |
| `pretrained` | `bool` | `True` | Whether to load pretrained backbone weights when initializing from a backbone name string or config `dict`. |
| `neck` | `str \| None` | `None` | Registered neck name (e.g. `"fpn"`, `"panet"`). Overrides task default when initializing from string or `dict`. |
| `head` | `str \| None` | `None` | Registered head name (e.g. `"decoupled_anchor_free"`, `"resunet_decoder"`). Overrides task default when initializing. |
| `**kwargs` | `Any` | *None* | Additional configuration parameters (e.g. `neck_channels=128`, `decoder_channels=128`, `dropout=0.1`) forwarded to component constructors. |

### `CoreModel.train()` & `TrainingConfig` Parameters

| Parameter | Type Hint | Default | Validation & Behavior |
| :--- | :--- | :--- | :--- |
| `config` | `str \| dict \| TrainingConfig \| None` | `None` | Polymorphic config container. Accepts `.yaml` file path, Python `dict`, `TrainingConfig` instance, or `None`. |
| `target_hardware` | `str` | `"server"` | Hardware profile (`"edge"` or `"server"`). `"edge"` applies activation rewrites (GELU→ReLU, SiLU→Hardswish) and LayerNorm collapses **before** building the optimizer. |
| `epochs` | `int` | `100` | Total training epochs. Must be `>= 1`. |
| `lr` | `float` | `0.001` | Base learning rate for optimizer. Must be `> 0.0`. |
| `batch_size` | `int` | `32` | Batch size per device. Must be `>= 1`. |
| `optimizer` | `str` | `"adamw"` | Optimizer choice. Valid options: `"adamw"`, `"adam"`, `"sgd"`. |
| `scheduler` | `str \| None` | `None` | Learning rate scheduler. Valid options: `"cosine"`, `"step"`, `"none"`, or `None`. |
| `amp` | `bool` | `True` | Enables Automatic Mixed Precision via `torch.amp.autocast`. |
| `grad_accum` | `int` | `1` | Gradient accumulation steps. Loss scaled by `1 / grad_accum`. Must be `>= 1`. |
| `clip_grad` | `float \| None` | `1.0` | Max gradient norm for `clip_grad_norm_`. `None` disables clipping. |
| `ema` | `bool` | `True` | Enables Exponential Moving Average (EMA) shadow weights. |
| `ema_decay` | `float` | `0.9999` | EMA decay factor. Must be strictly within `(0.0, 1.0)`. |
| `device` | `str \| None` | `None` | Target device override (e.g. `"cuda:0"`, `"cpu"`). |
| `output_dir` | `str` | `"./checkpoints"` | Directory path for saving checkpoints and training history. |

---

## 2. Practical Examples: Detection vs. Segmentation

=== "Object Detection Workflow"

    ```python
    import torch
    from torch.utils.data import DataLoader
    from corecv.api import CoreModel
    from corecv.losses.detection import DetectionLoss
    from corecv.data.datasets.detection import CocoDetectionDataset

    # 1. Initialize Detection Model
    model = CoreModel(
        model="configs/yolo_detection.yaml",
        task="detection",
        input_size=(640, 640),
        num_classes=80,
    )

    # 2. Setup Dataset and Loaders
    train_dataset = CocoDetectionDataset(
        img_folder="coco/images/train2017",
        ann_file="coco/annotations/instances_train2017.json",
        img_size=(640, 640),
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 3. Configure Loss Function & Loaders
    (model
     .set_train_dataloader(train_loader)
     .set_loss_fn(DetectionLoss(num_classes=80)))

    # 4. Execute Edge-Aware Training
    history = model.train(
        epochs=50,
        lr=1e-3,
        optimizer="adamw",
        scheduler="cosine",
        amp=True,
        target_hardware="edge",  # Applies GELU->ReLU, SiLU->Hardswish rewrites
    )
    ```

=== "Semantic/Instance Segmentation Workflow"

    ```python
    import torch
    from torch.utils.data import DataLoader
    from corecv.api import CoreModel
    from corecv.losses.segmentation import SegmentationLoss
    from corecv.data.datasets.segmentation import SegmentationDataset

    # 1. Initialize Segmentation Model
    model = CoreModel(
        model="configs/deeplabv3_segm.yaml",
        task="segmentation",
        input_size=(512, 512),
        num_classes=19,
    )

    # 2. Setup Segmentation Dataset
    train_dataset = SegmentationDataset(
        image_dir="cityscapes/leftImg8bit/train",
        mask_dir="cityscapes/gtFine/train",
        img_size=(512, 512),
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # 3. Configure Loss Function & Loaders
    (model
     .set_train_dataloader(train_loader)
     .set_loss_fn(SegmentationLoss(num_classes=19)))

    # 4. Execute Server-Grade Training with EMA
    history = model.train(
        epochs=100,
        lr=5e-4,
        batch_size=8,
        optimizer="adamw",
        scheduler="cosine",
        ema=True,
        ema_decay=0.999,
        target_hardware="server",
    )
    ```

---

## 3. Polymorphic Model Initialization & Training Configurations

### Model Initialization Options (`CoreModel`)

```python
# 1. Plain backbone string name
model = CoreModel("resnet18", task="classification", num_classes=10)

# 2. Plain backbone string with direct neck, head, and channel overrides
model = CoreModel(
    "resnet50",
    task="detection",
    neck="panet",                # Registered neck name
    head="query_detection",      # Registered head name
    neck_channels=128,           # Dynamic kwarg forwarded to neck constructor
    num_classes=80,
)

# 3. Raw configuration dictionary
model = CoreModel(
    model={
        "model_name": "convnext_tiny",
        "head_type": "resunet_decoder",
        "decoder_channels": 128,
    },
    task="segmentation",
    num_classes=19,
)

# 4. YAML configuration file
model = CoreModel("configs/yolo_detection.yaml", task="detection")

# 5. Checkpoint file (.pt / .pth)
model = CoreModel("checkpoints/model_epoch_50.pt", task="detection")
```

### Training Configuration Options (`model.train()`)

```python
# Method A: Direct keyword arguments
model.train(epochs=20, lr=0.001, batch_size=16, target_hardware="edge")

# Method B: Python dictionary
model.train({"epochs": 20, "lr": 0.001, "batch_size": 16, "target_hardware": "edge"})

# Method C: YAML Configuration file
model.train("configs/train_params.yaml")

# Method D: Validated TrainingConfig dataclass
from corecv.api.model import TrainingConfig
cfg = TrainingConfig(epochs=20, lr=0.001, batch_size=16, target_hardware="edge")
model.train(cfg)
```

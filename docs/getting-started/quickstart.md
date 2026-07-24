# Getting Started & Quickstart

This guide covers setting up **CoreCV** and running your first training, inference, and model export pipelines.

---

## Installation

CoreCV can be installed using `uv` (recommended for fast dependency resolution) or standard `pip`.

### Using `uv`

```bash
uv pip install corecv
```

Or within a `uv` project workspace:

```bash
uv add corecv
```

### Using `pip`

```bash
pip install corecv
```

### Install from Source

For development or latest features:

```bash
git clone https://github.com/liturriago/corecv.git
cd corecv
uv sync
```

---

## Basic Usage Walkthrough

### 1. Initializing CoreModel

The `CoreModel` class serves as the main entry point to CoreCV:

```python
from corecv.api import CoreModel

# Option A: Initialize directly using a registered backbone string name
model = CoreModel(
    model="resnet18",
    task="classification",
    num_classes=1000,
    input_size=(224, 224),
)

# Option B: Initialize using a raw configuration dictionary
model = CoreModel(
    model={
        "model_name": "resnet50",
        "neck_type": "panet",
        "head_type": "decoupled_anchor_free",
        "neck_channels": 256,
    },
    task="detection",
    num_classes=80,
)

# Option C: Pass a custom PyTorch nn.Module directly
import torch.nn as nn

my_custom_net = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 10),
)

model = CoreModel(my_custom_net, task="classification")
```

---

### 2. Model Training

Run model training using dictionary configurations, YAML files, or direct keyword arguments:

```python
# Train using keyword arguments
history = model.train(
    data="path/to/dataset",
    epochs=25,
    batch_size=64,
    lr=0.001,
    amp=True,
    target_hardware="edge",
)
```

---

### 3. Model Inference

Run inference across image files, directories, or pre-loaded tensors:

```python
# Predict on an image file
results = model.predict("data/sample.jpg", topk=5)

# Output includes confidence scores and labels
for result in results:
    print(f"Predictions: {result.labels}, Scores: {result.scores}")
```

---

### 4. Hardware-Aware Export

Export models to edge targets with automatic graph optimization and compile verification:

```python
# Export to ONNX format optimized for edge devices
export_result = model.export(
    format="onnx",
    target_hardware="edge",
    output_path="models/resnet18_edge.onnx",
)

print(f"Exported model saved to: {export_result}")
```

Next steps:
- Learn more about training options in the [Training User Guide](../user-guide/training.md).
- Explore multi-source inference in the [Inference User Guide](../user-guide/inference.md).
- Check out edge optimization in the [Exporting User Guide](../user-guide/exporting.md).

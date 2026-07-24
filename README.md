# CoreCV - Unified Vision Engine

<p align="center">
  <img src="assets/logo-tagline.svg" alt="CoreCV Logo" width="600">
</p>

<p align="center">
  <strong>Production-Ready Computer Vision Engine with Edge-Aware Optimization & Unified Deployment API</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quickstart-3-lines">Quickstart</a> •
  <a href="#model-zoo-catalogue">Model Zoo</a> •
  <a href="#edge-hardware-export">Edge Export</a> •
  <a href="#documentation">Documentation</a>
</p>

---

## Overview

**CoreCV** is a high-performance, production-grade computer vision library built on PyTorch. It bridges the gap between research agility and edge deployment constraints by consolidating model construction, training loops, multi-source inference, and hardware-targeted export into a unified facade API (`CoreModel`).

With CoreCV, you can instantiate any vision architecture with a single string, train with automatic mixed precision and edge graph rewrites, run GPU-accelerated inference across heterogeneous sources, and export directly to **ONNX** and **ExecuTorch** formats.

---

## Key Features

- 🎯 **Unified Facade API (`CoreModel`)**: Manage complete model lifecycles through three clean methods: `.train()`, `.predict()`, and `.export()`.
- 🦁 **Built-in Model Zoo**: 13 Backbones (`ResNet`, `MobileNetV3`, `ConvNeXt`, `ViT`), 2 Feature Necks (`FPN`, `PANet`), and 5 Task Heads (`Linear`, `ASPP Decoder`, `ResUNet Decoder`, `Decoupled Anchor-Free`, `Query Transformer Head`).
- ⚡ **Polymorphic & Dynamic Component Assembly**: Instantiate models via plain backbone string names (e.g. `"resnet50"`), raw Python dictionaries, or `.yaml` files. Swap necks and heads on the fly (`neck="panet"`, `head="query_detection"`).
- 🚀 **Edge-First Optimization**:
  - `TargetRewriter`: Automatic edge-hardware graph rewrites (GELU → ReLU, SiLU → Hardswish, LayerNorm channel collapses).
  - `MetaProber`: Zero-VRAM shape propagation and static-graph audit using PyTorch `meta` device tensors.
- 📦 **Multi-Format Export**: One-line export pipeline producing optimized ONNX (`.onnx`) and ExecuTorch (`.pte`) artifacts.
- 🛡️ **Type-Safe Static Registry**: `CoreRegistry` with pre-instantiation signature checking and signature kwarg filtering.

---

## Installation

Install CoreCV via `uv` (recommended for fast dependency resolution) or standard `pip`:

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

---

## Quickstart (3 Lines)

```python
from corecv.api import CoreModel

# 1. Instantiate model directly using a backbone string name
model = CoreModel("resnet18", task="classification", num_classes=10)

# 2. Train with edge-aware optimizations
model.train(data="./dataset", epochs=10, lr=1e-3, target_hardware="edge")

# 3. Predict & Export to edge deployment formats
predictions = model.predict("test_image.jpg", topk=5)
paths = model.export(format="onnx", target_hardware="edge")
```

---

## Model Zoo Catalogue

CoreCV features a modular, decoupled architecture where any backbone can be paired with any neck and head:

| Category | Component Family | Registry Keys | Description |
| :--- | :--- | :--- | :--- |
| **Backbones** | **ResNet** | `resnet18`, `resnet34`, `resnet50`, `resnet101` | Residual convolutional feature extractors |
| | **MobileNetV3** | `mobilenet_v3_small`, `mobilenet_v3_large` | Ultra-lightweight edge backbones |
| | **ConvNeXt** | `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large` | Modernized conv-nets with transformer design choices |
| | **ViT** | `vit_tiny`, `vit_small`, `vit_base` | Vision Transformers with `SimplePyramidAdapter` |
| **Necks** | **FPN** | `fpn` | Top-down Feature Pyramid Network |
| | **PANet** | `panet` | Path Aggregation Network (FPN + bottom-up path) |
| **Heads** | **Classification** | `linear_classification` | Global average pooling + linear classifier |
| | **Segmentation** | `aspp_decoder` | DeepLabV3+ style ASPP decoder |
| | | `resunet_decoder` | U-Net style residual skip-connection decoder |
| | **Detection** | `decoupled_anchor_free` | YOLOX/FCOS style decoupled conv head |
| | | `query_detection` | RT-DETR/D-FINE style query transformer decoder |

### Flexible Component Swapping

```python
from corecv.api import CoreModel

# Detection: MobileNetV3-Large + PANet Neck + Query Transformer Head
detector = CoreModel(
    "mobilenet_v3_large",
    task="detection",
    neck="panet",
    head="query_detection",
    neck_channels=128,
    num_classes=80,
)

# Segmentation: ConvNeXt-Tiny + ResUNet Decoder
segmentor = CoreModel(
    "convnext_tiny",
    task="segmentation",
    head="resunet_decoder",
    decoder_channels=128,
    num_classes=19,
)
```

---

## Multi-Source Inference

`CoreModel.predict()` processes single image files, image folders, NumPy arrays, or PyTorch tensors out of the box with GPU-native preprocessing and FP16 support:

```python
# Infer on single image file, folder, or tensor
results = model.predict(
    source="data/test_images/",
    conf_threshold=0.3,
    half_precision=True,  # FP16
    batch_size=16,
    weights="checkpoints/best.pt",  # On-the-fly weights loading
)

for res in results:
    if res.detection:
        print(f"Image {res.image_path}: {len(res.detection.boxes)} boxes detected")
```

---

## Edge Hardware Export

CoreCV compiles and verifies models for edge deployment without allocating GPU VRAM:

```python
# Export model to ONNX & ExecuTorch simultaneously with edge rewrites
paths = model.export(
    format="both",  # Produces .onnx and .pte
    target_hardware="edge",  # Applies GELU->ReLU & SiLU->Hardswish rewrites
    opset=18,
    output_path="exports/detector_edge",
)

print(f"ONNX Model: {paths['onnx']}")
print(f"ExecuTorch Model: {paths['executorch']}")
```

---

## Documentation

The official CoreCV documentation is hosted on GitHub Pages:

👉 **[https://liturriago.github.io/corecv/](https://liturriago.github.io/corecv/)**

### Build and Serve Locally

```bash
# Build the documentation site
uv run mkdocs build --strict

# Start the live preview server
uv run mkdocs serve
```

Once running, open `http://127.0.0.1:8000` in your browser.

---

## License

This project is licensed under the MIT License.

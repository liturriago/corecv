# CoreCV - Unified Vision Engine

<p align="center">
  <img src="assets/logo-tagline.svg" alt="CoreCV Logo" width="600">
</p>

Welcome to **CoreCV**, a production-ready computer vision engine designed for high-performance model development, seamless training, edge-aware hardware acceleration, and unified multi-format export.

---

## Core Philosophy

CoreCV addresses the fragmentation in vision model deployment by unifying model construction, training, inference, and hardware-targeted export into a cohesive Python API. Built on top of PyTorch, CoreCV bridges the gap between research agility and edge deployment constraints.

- **Unified Facade**: Interact with any vision architecture using three primary methods: `.train()`, `.predict()`, and `.export()`.
- **Edge-First Engineering**: Automatic graph rewriting (`TargetRewriter`) and zero-VRAM compilation probing (`MetaProber`) ensure your models run efficiently on constrained edge devices.
- **Type-Safe Modular Architecture**: Declarative architecture registration system (`CoreRegistry`) with signature validation before instantiation.

---

## Feature Matrix

| Capability | Description | CoreCV Component |
| :--- | :--- | :--- |
| **Unified API** | Facade pattern consolidating lifecycle management | `CoreModel` |
| **Polymorphic Training** | Accept configurations via YAML files, dictionaries, or keyword args | `CoreTrainer` |
| **Multi-Source Inference** | Process single images, directories, numpy arrays, or torch tensors seamlessly | `CorePredictor` |
| **Graph Rewriting** | Edge-specific hardware optimizations and operator replacements | `TargetRewriter` |
| **Zero-VRAM Validation** | Validate shape inference and graph exportability on fake meta tensors | `MetaProber` |
| **Multi-Format Export** | Export models to ONNX and ExecuTorch (`.pte`) formats | `CoreExporter` |
| **Registry System** | Generic, type-safe registry system for backbones, necks, heads, and losses | `CoreRegistry` |

---

## Unified 3-Step Quickstart

CoreCV simplifies end-to-end vision workflows into three clean steps:

```python
from corecv.api import CoreModel

# 1. Instantiate model directly using a backbone string name
model = CoreModel("resnet18", task="classification", num_classes=10)

# 2. Train with polymorphic arguments
model.train(
    data="path/to/dataset",
    epochs=10,
    lr=1e-3,
    batch_size=32,
    target_hardware="edge",
)

# 3. Predict & Export to edge deployment formats
predictions = model.predict("test_image.jpg", topk=5)
export_paths = model.export(format="onnx", target_hardware="edge")
```

Next, check out the [Quickstart Guide](getting-started/quickstart.md) for full installation and setup instructions.

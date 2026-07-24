# User Guide: Model Export & Edge Optimization

CoreCV features a dedicated, multi-stage model export engine managed by `CoreExporter`, `TargetRewriter`, and `MetaProber`. It ensures smooth compilation and transition from PyTorch models into edge-optimized **ONNX** and **ExecuTorch** formats.

---

## 1. Complete Parameter Reference

### `CoreModel.export()` & `CoreExporter` Parameters

| Parameter | Type Hint | Default | Description |
| :--- | :--- | :--- | :--- |
| `format` | `str` | `"onnx"` | Target export format. One of `"onnx"`, `"executorch"`, or `"both"`. |
| `target_hardware` | `str` | `"server"` | Hardware profile (`"edge"` or `"server"`). `"edge"` applies activation rewrites (GELU→ReLU, SiLU→Hardswish) and NCHW layout optimizations; `"server"` skips rewrites. |
| `opset` | `int` | `17` | ONNX opset version. Valid options: `17` or `18`. |
| `optimize` | `bool` | `True` | Applies `TargetRewriter` graph optimization passes and XNNPACK delegates (for ExecuTorch). |
| `output_path` | `str \| None` | `None` | Explicit output file path. For `"both"`, used as a prefix (e.g. `"model"` produces `"model.onnx"` and `"model.pte"`). Auto-generated if `None`. |
| `input_shape` | `tuple[int, ...]` | `(1, 3, H, W)` | Dummy input tensor dimensions `(B, C, H, W)` used for graph tracing and shape auditing. |
| `dynamic_axes` | `dict \| None` | `None` | ONNX dynamic axes mapping, e.g. `{"input": {0: "batch", 2: "height", 3: "width"}}`. |
| `weights` | `str \| Path \| None` | `None` | Optional path to a `.pt`/`.pth` checkpoint file loaded into the model before export. |

---

## 2. Multi-Stage Edge Export Pipeline

```
  CoreModel (PyTorch nn.Module)
               │
               ▼
   1. TargetRewriter (Edge Hardware Transformations)
      - Activation Rewrites: GELU -> ReLU, SiLU -> Hardswish
      - LayerNorm Collapse & Constant Folding
               │
               ▼
   2. MetaProber (Zero-VRAM Shape & Op Verification)
      - Dynamic operation audit on 'meta' device tensors
               │
               ▼
   3. Serialized Model Export
      ├── ONNX Export (.onnx) via torch.onnx.export
      └── ExecuTorch Export (.pte) via torch.export
```

---

## 3. Practical Examples: Detection vs. Segmentation Export

=== "Object Detection Export (ONNX & ExecuTorch)"

    ```python
    from corecv.api import CoreModel

    # 1. Initialize Object Detector Facade
    model = CoreModel(
        model="configs/yolo_detection.yaml",
        task="detection",
        input_size=(640, 640),
    )

    # 2. Export to Edge-Optimized ONNX format with Dynamic Axes
    onnx_result = model.export(
        format="onnx",
        target_hardware="edge",  # Applies GELU->ReLU & SiLU->Hardswish rewrites
        opset=18,
        dynamic_axes={"input": {0: "batch_size"}},
        output_path="exports/yolo_detection_edge.onnx",
        weights="runs/train/exp1/best.pt",
    )
    print(f"ONNX Model saved to: {onnx_result['onnx']}")

    # 3. Export to ExecuTorch (.pte) format
    pte_result = model.export(
        format="executorch",
        target_hardware="edge",
        output_path="exports/yolo_detection_edge.pte",
    )
    print(f"ExecuTorch Model saved to: {pte_result['executorch']}")
    ```

=== "Semantic Segmentation Export (Dual Format)"

    ```python
    from corecv.api import CoreModel

    # 1. Initialize Segmentation Model Facade
    model = CoreModel(
        model="configs/deeplabv3_segm.yaml",
        task="segmentation",
        input_size=(512, 512),
    )

    # 2. Export Both Formats Simultaneously
    export_paths = model.export(
        format="both",
        target_hardware="edge",
        output_path="exports/segmentation_model",  # Produces segmentation_model.onnx & segmentation_model.pte
        weights="checkpoints/segm_best.pt",
    )

    print(f"ONNX path: {export_paths['onnx']}")
    print(f"ExecuTorch path: {export_paths['executorch']}")
    ```

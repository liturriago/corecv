# User Guide: Model Inference

CoreCV provides GPU-accelerated, multi-source inference managed by `CorePredictor` under the hood and exposed via `CoreModel.predict()`.

---

## 1. Complete Parameter Reference

### `CoreModel.predict()` & `CorePredictor` Parameters

| Parameter | Type Hint | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `str \| Path \| Tensor \| list[...]` | *Required* | Input image path, folder path, NumPy array, `torch.Tensor` (`CHW` or `NCHW`), or list of paths/tensors. |
| `conf_threshold` | `float \| None` | `None` (uses `0.25`) | Minimum confidence score threshold for filtering object detection bounding boxes. |
| `iou_threshold` | `float \| None` | `None` (uses `0.45`) | Intersection-over-Union (IoU) threshold for Non-Maximum Suppression (NMS). |
| `topk` | `int \| None` | `None` (uses `5`) | Number of top predicted classes to return for classification tasks. |
| `half_precision` | `bool` | `False` | Enables FP16 half-precision inference via `torch.autocast`. |
| `compile_model` | `bool` | `False` | Enables PyTorch 2.0+ `torch.compile()` execution graph optimization. |
| `batch_size` | `int` | `8` | Maximum batch size used for folder or list-based batch inference. |
| `weights` | `str \| Path \| None` | `None` | Optional path to a `.pt`/`.pth` weights file loaded on-the-fly before running inference. |

---

## 2. Prediction Return Objects

`CoreModel.predict()` returns a list of `Prediction` dataclass objects (one per input image):

```python
@dataclass
class Prediction:
    task: str  # "classification", "segmentation", or "detection"
    original_size: tuple[int, int]  # (H, W) of the input image
    image_path: str | None
    
    # Task-specific attributes (populated according to task):
    classification: ClassificationPrediction | None
    segmentation: SegmentationPrediction | None
    detection: DetectionPrediction | None
```

### Task-Specific Output Structures

| Task | Attribute | Dataclass Fields | Description |
| :--- | :--- | :--- | :--- |
| **Classification** | `res.classification` | `class_ids: Tensor`, `scores: Tensor`, `labels: list[str]` | Top-K predicted class indices and softmax probabilities. |
| **Segmentation** | `res.segmentation` | `mask: Tensor`, `original_size: tuple`, `probabilities: Tensor` | Class map `(H_orig, W_orig)` resized to original input dimensions. |
| **Detection** | `res.detection` | `boxes: Tensor`, `scores: Tensor`, `class_ids: Tensor`, `labels: list[str]` | Rescaled bounding boxes `(N, 4)` in `(x_min, y_min, x_max, y_max)` format. |

---

## 3. Practical Examples: Detection vs. Segmentation

=== "Object Detection Inference"

    ```python
    from corecv.api import CoreModel

    # 1. Initialize Detection Model
    model = CoreModel("configs/yolo_detection.yaml", task="detection")

    # 2. Run Inference on Image Folder with On-The-Fly Weights Loading
    results = model.predict(
        source="data/test_images/",
        conf_threshold=0.3,
        iou_threshold=0.5,
        half_precision=True,  # FP16
        batch_size=16,
        weights="runs/train/exp1/checkpoints/best.pt",
    )

    # 3. Process Detection Predictions
    for res in results:
        print(f"Image: {res.image_path} (Original Size: {res.original_size})")
        det = res.detection
        if det is not None:
            print(f"Found {len(det.boxes)} objects:")
            for box, score, cls_id in zip(det.boxes, det.scores, det.class_ids):
                x1, y1, x2, y2 = box.tolist()
                print(f" - Class {cls_id.item()}: Conf {score.item():.2f} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    ```

=== "Semantic Segmentation Inference"

    ```python
    from corecv.api import CoreModel

    # 1. Initialize Segmentation Model
    model = CoreModel("configs/deeplabv3_segm.yaml", task="segmentation")

    # 2. Run Inference on Single Image Tensor or File
    results = model.predict(
        source="data/street_scene.png",
        half_precision=True,
        weights="checkpoints/segmentation_best.pt",
    )

    # 3. Process Segmentation Mask
    res = results[0]
    seg = res.segmentation
    if seg is not None:
        mask = seg.mask  # Tensor (H_orig, W_orig) containing class indices per pixel
        print(f"Segmentation mask shape: {mask.shape}, Unique classes present: {mask.unique().tolist()}")
    ```

---

## 4. Multi-Source Input Capabilities

```python
# Single image path
model.predict("photo.jpg")

# Directory of images
model.predict("dataset/val/")

# NumPy array
import cv2
img = cv2.imread("photo.jpg")
model.predict(img)

# PyTorch Tensor (C, H, W or B, C, H, W)
import torch
tensor_input = torch.randn(3, 640, 640)
model.predict(tensor_input)

# List of mixed sources
model.predict(["img1.jpg", "img2.jpg", tensor_input], batch_size=4)
```

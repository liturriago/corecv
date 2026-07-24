# Model Zoo & Supported Components

CoreCV provides a comprehensive catalogue of production-ready **Backbones**, **Necks**, and **Heads**. Every component is registered in `CoreRegistry` and can be instantiated directly via string keys in `CoreModel` or configuration files.

---

## Catalogue Overview

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                            BACKBONES (13)                               │
 ├─────────────────┬───────────────────┬──────────────────┬────────────────┤
 │     ResNet      │    MobileNetV3    │     ConvNeXt     │      ViT       │
 │   resnet18/34   │ mobilenet_v3_small│  convnext_tiny   │    vit_tiny    │
 │   resnet50/101  │ mobilenet_v3_large│  convnext_small  │    vit_small   │
 │                 │                   │  convnext_base   │    vit_base    │
 │                 │                   │  convnext_large  │                │
 └─────────────────┴───────────────────┴──────────────────┴────────────────┘
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                             NECKS (2)                                   │
 ├───────────────────────────────────┬─────────────────────────────────────┤
 │               FPN                 │                PANet                │
 │              (fpn)                │               (panet)               │
 └───────────────────────────────────┴─────────────────────────────────────┘
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                             HEADS (5)                                   │
 ├───────────────────┬──────────────────────────────┬──────────────────────┤
 │  Classification   │         Segmentation         │      Detection       │
 │linear_classific.  │  aspp_decoder / resunet_dec. │decoupled_a.f. / query│
 └───────────────────┴──────────────────────────────┴──────────────────────┘
```

---

## 1. Backbones Catalogue (`BACKBONE_REGISTRY`)

All backbones inherit from `BaseBackbone` and emit standardized `FeatureInfo` metadata describing output channel sizes and spatial strides across multi-scale feature stages.

### ResNet Family

Standard residual convolutional networks for general-purpose feature extraction.

| Registry Key (`model_name`) | Class Name | Strides | Feature Channels | Pretrained Available |
| :--- | :--- | :--- | :--- | :--- |
| `resnet18` | `ResNet18Backbone` | `[4, 8, 16, 32]` | `[64, 128, 256, 512]` | Yes (ImageNet) |
| `resnet34` | `ResNet34Backbone` | `[4, 8, 16, 32]` | `[64, 128, 256, 512]` | Yes (ImageNet) |
| `resnet50` | `ResNet50Backbone` | `[4, 8, 16, 32]` | `[256, 512, 1024, 2048]` | Yes (ImageNet) |
| `resnet101` | `ResNet101Backbone` | `[4, 8, 16, 32]` | `[256, 512, 1024, 2048]` | Yes (ImageNet) |

### MobileNetV3 Family

Lightweight, hardware-friendly backbones optimized for mobile and edge silicon deployment.

| Registry Key (`model_name`) | Class Name | Strides | Feature Channels | Target Profile |
| :--- | :--- | :--- | :--- | :--- |
| `mobilenet_v3_small` | `MobileNetV3SmallBackbone` | `[4, 8, 16, 32]` | `[16, 24, 48, 96]` | Ultra-Low Power Edge |
| `mobilenet_v3_large` | `MobileNetV3LargeBackbone` | `[4, 8, 16, 32]` | `[24, 40, 112, 160]` | Edge Accelerator |

### ConvNeXt Family

Modernized pure-convolutional architectures incorporating design choices from Vision Transformers.

| Registry Key (`model_name`) | Class Name | Strides | Feature Channels | Target Profile |
| :--- | :--- | :--- | :--- | :--- |
| `convnext_tiny` | `ConvNeXtTinyBackbone` | `[4, 8, 16, 32]` | `[96, 192, 384, 768]` | Compact Server / Edge |
| `convnext_small` | `ConvNeXtSmallBackbone` | `[4, 8, 16, 32]` | `[96, 192, 384, 768]` | High Accuracy Server |
| `convnext_base` | `ConvNeXtBaseBackbone` | `[4, 8, 16, 32]` | `[128, 256, 512, 1024]` | High Performance Server |
| `convnext_large` | `ConvNeXtLargeBackbone` | `[4, 8, 16, 32]` | `[192, 384, 768, 1536]` | State-of-the-Art Server |

### Vision Transformer (ViT) Family

Self-attention based vision backbones wrapped with `SimplePyramidAdapter` to emit multi-scale pyramid features.

| Registry Key (`model_name`) | Class Name | Strides | Feature Channels | Adapter Output |
| :--- | :--- | :--- | :--- | :--- |
| `vit_tiny` | `ViTTinyBackbone` | `[8, 16, 32]` | `[192, 192, 192]` | Multi-scale Pyramid |
| `vit_small` | `ViTSmallBackbone` | `[8, 16, 32]` | `[384, 384, 384]` | Multi-scale Pyramid |
| `vit_base` | `ViTBaseBackbone` | `[8, 16, 32]` | `[768, 768, 768]` | Multi-scale Pyramid |

---

## 2. Necks Catalogue (`NECK_REGISTRY`)

Necks fuse multi-scale features from backbone stages to construct pyramid representations with uniform channel dimensions.

| Registry Key (`neck_type`) | Class Name | Input Requirements | Output Channels | Architectural Description |
| :--- | :--- | :--- | :--- | :--- |
| `fpn` | `FPN` | Multi-scale feature map list `[C2..C5]` | `256` (Configurable via `neck_channels`) | **Feature Pyramid Network**: Top-down pathway with 1x1 lateral convolutions and 3x3 anti-aliasing output convs. |
| `panet` | `PANet` | Multi-scale feature map list `[C2..C5]` | `256` (Configurable via `neck_channels`) | **Path Aggregation Network**: Extends FPN with an additional bottom-up path augmentation to enhance low-level localization cues. |

---

## 3. Heads & Decoders Catalogue (`HEAD_REGISTRY`)

Heads perform task-specific predictions consuming feature maps from backbones or necks.

### Classification Heads

| Registry Key (`head_type`) | Class Name | Output Structure | Parameters |
| :--- | :--- | :--- | :--- |
| `linear_classification` | `LinearClassificationHead` | Class Logits `(B, num_classes)` | `num_classes`, `dropout` |

### Segmentation Decoders

| Registry Key (`head_type`) | Class Name | Output Structure | Description |
| :--- | :--- | :--- | :--- |
| `aspp_decoder` | `ASPPDecoder` | Pixel Mask Logits `(B, num_classes, H, W)` | **DeepLabV3+ Style Decoder**: Combines Atrous Spatial Pyramid Pooling (rates 1, 6, 12, 18) on `C5` with low-level `C2` feature fusion. |
| `resunet_decoder` | `ResUNetDecoder` | Pixel Mask Logits `(B, num_classes, H, W)` | **U-Net Style Decoder**: Progressive upsampling decoder with residual blocks and multi-scale skip connections. |

### Detection Heads

| Registry Key (`head_type`) | Class Name | Output Structure | Description |
| :--- | :--- | :--- | :--- |
| `decoupled_anchor_free` | `DecoupledAnchorFreeHead` | Dict per level: `cls_logits` & `reg_pred` | **YOLOX/FCOS Style Head**: Decoupled 3x3 convolutional branches for class probabilities and bounding box regressions. |
| `query_detection` | `QueryDetectionHead` | Dict: `pred_logits` & `pred_boxes` | **RT-DETR/D-FINE Style Head**: Transformer decoder head using cross-attention over learnable query embeddings. |

---

## 4. Practical Code Examples

### Example 1: Object Detection with Swapped Backbone & Neck

```python
from corecv.api import CoreModel

# MobileNetV3-Large backbone + PANet neck + Query Detection Head
model = CoreModel(
    model="mobilenet_v3_large",
    task="detection",
    neck="panet",
    head="query_detection",
    neck_channels=128,
    num_classes=80,
)
```

### Example 2: Semantic Segmentation with ConvNeXt & U-Net Decoder

```python
from corecv.api import CoreModel

# ConvNeXt-Tiny backbone + ResUNet Decoder
model = CoreModel(
    model="convnext_tiny",
    task="segmentation",
    head="resunet_decoder",
    decoder_channels=128,
    num_classes=19,
)
```

### Example 3: Vision Transformer Classification

```python
from corecv.api import CoreModel

# ViT-Base backbone + Linear Classification Head
model = CoreModel(
    model="vit_base",
    task="classification",
    num_classes=1000,
    dropout=0.2,
)
```

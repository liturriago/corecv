# User Guide: Architecture & Registry System

CoreCV is engineered with a modular, decoupled architecture. Vision models are constructed from three distinct architectural building blocks—**Backbones**, **Necks**, and **Heads**—registered and assembled using a type-safe static registry system (`CoreRegistry`).

---

## 1. Built-in Registry Catalogue

All components in CoreCV are registered in global registries (`BACKBONE_REGISTRY`, `NECK_REGISTRY`, `HEAD_REGISTRY`) and can be selected by passing their string registration keys in configuration dicts or YAML files (`head_type="..."`, `neck_type="..."`).

### Built-in Backbones (`BACKBONE_REGISTRY`)

| Model Family | Registry Keys (`model_name`) | Feature Strides | Default Channels |
| :--- | :--- | :--- | :--- |
| **ResNet** | `resnet18`, `resnet34`, `resnet50`, `resnet101` | `[4, 8, 16, 32]` | `[64, 128, 256, 512]` / `[256, 512, 1024, 2048]` |
| **MobileNetV3** | `mobilenet_v3_small`, `mobilenet_v3_large` | `[4, 8, 16, 32]` | `[16, 24, 48, 96]` / `[24, 40, 112, 160]` |
| **ConvNeXt** | `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large` | `[4, 8, 16, 32]` | `[96, 192, 384, 768]`+ |
| **ViT** | `vit_tiny`, `vit_small`, `vit_base` | `[8, 16, 32]` | SimplePyramidAdapter outputs `[192/384/768]` |

### Built-in Necks (`NECK_REGISTRY`)

| Registry Key (`neck_type`) | Class | Architecture Description |
| :--- | :--- | :--- |
| `fpn` | `FPN` | Feature Pyramid Network (Lin et al., 2017) with top-down pathway and lateral 1x1 convs. |
| `panet` | `PANet` | Path Aggregation Network (Liu et al., 2018) adding bottom-up path augmentation to FPN. |

### Built-in Heads (`HEAD_REGISTRY`)

| Task Category | Registry Key (`head_type`) | Class | Architecture Description |
| :--- | :--- | :--- | :--- |
| **Classification** | `linear_classification` | `LinearClassificationHead` | Global average pooling followed by linear classification layer. |
| **Segmentation** | `aspp_decoder` | `ASPPDecoder` | DeepLabV3+ style decoder with Atrous Spatial Pyramid Pooling and C2 low-level feature fusion. |
| **Segmentation** | `resunet_decoder` | `ResUNetDecoder` | U-Net style decoder with residual blocks and progressive multi-scale upsampling. |
| **Detection** | `decoupled_anchor_free` | `DecoupledAnchorFreeHead` | YOLOX/FCOS style decoupled head with separate classification and box regression branches per level. |
| **Detection** | `query_detection` | `QueryDetectionHead` | RT-DETR/D-FINE style transformer decoder head with learnable query embeddings. |

---

## 2. Modularity & Dynamic Component Resolution

CoreCV does **NOT** hardcode head and neck choices to specific backbone architectures or fixed channel dimensions. Component resolution and parameter passing are fully dynamic and decoupled.

### Internal Resolution Logic (`CoreModel._build_model_from_config`)

Located in [`src/corecv/api/model.py`](file:///c:/Users/Lucas/Documents/GitHub/corecv/src/corecv/api/model.py#L1296-L1409):

```python
# 1. Resolve Backbone via BACKBONE_REGISTRY
backbone_cls = get_backbone(config["model_name"])  # e.g., "resnet50"
backbone = backbone_cls(pretrained=config.get("pretrained", True))

# 2. Resolve Task-Specific Components via HEAD_REGISTRY / NECK_REGISTRY
if task == "classification":
    head_type = config.get("head_type", "linear_classification")
    head_cls = get_head(head_type)
    # Dynamic signature filtering prevents unexpected keyword argument errors
    head_kwargs = CoreModel._filter_kwargs_for_signature(
        head_cls, config, extra_keys={"feature_info": backbone.feature_info, "num_classes": num_classes}
    )
    head = head_cls(**head_kwargs)
    model = nn.Sequential(OrderedDict([("backbone", backbone), ("head", head)]))

elif task == "segmentation":
    head_type = config.get("head_type", "aspp_decoder")
    head_cls = get_head(head_type)
    # Dynamic decoder channels override (default 256)
    decoder_channels = config.get("decoder_channels", 256)
    head_kwargs = CoreModel._filter_kwargs_for_signature(
        head_cls, config, extra_keys={"feature_info": backbone.feature_info, "out_channels": decoder_channels, "num_classes": num_classes}
    )
    head = head_cls(**head_kwargs)
    model = nn.Sequential(OrderedDict([("backbone", backbone), ("head", head)]))

elif task == "detection":
    neck_type = config.get("neck_type", "fpn")
    head_type = config.get("head_type", "decoupled_anchor_free")
    # Dynamic neck channels override (default 256)
    neck_channels = config.get("neck_channels", config.get("out_channels", 256))
    neck_cls = get_neck(neck_type)
    neck_kwargs = CoreModel._filter_kwargs_for_signature(
        neck_cls, config, extra_keys={"feature_info": backbone.feature_info, "out_channels": neck_channels}
    )
    neck = neck_cls(**neck_kwargs)
    
    head_cls = get_head(head_type)
    head_kwargs = CoreModel._filter_kwargs_for_signature(
        head_cls, config, extra_keys={"feature_info": backbone.feature_info, "num_classes": num_classes}
    )
    head = head_cls(**head_kwargs)
    model = CoreObjectDetector(backbone=backbone, neck=neck, head=head)
```

### Flexible Model Initialization Syntaxes

`CoreModel` supports multiple polymorphic initialization styles:

```python
from corecv.api import CoreModel

# Style 1: Plain backbone string name with default components
model = CoreModel("resnet18", task="classification", num_classes=10)

# Style 2: Plain backbone string with direct neck, head, and channel overrides
model_det = CoreModel(
    "mobilenet_v3_large",
    task="detection",
    neck="panet",              # Direct registered neck selection
    head="query_detection",    # Direct registered head selection
    neck_channels=128,         # Dynamically forwarded keyword argument
    num_classes=80,
)

# Style 3: Raw Python configuration dictionary
config_seg = {
    "model_name": "convnext_tiny",
    "head_type": "resunet_decoder",         # Swapped built-in segmentation decoder
    "decoder_channels": 128,                # Custom decoder channel dimension override
    "num_classes": 19,
}
model_seg = CoreModel(config_seg, task="segmentation")
```

---

## 3. Feature Flow: Detection vs. Segmentation

### Object Detection Feature Flow

```
Input (B, 3, H, W)
  │
  ├─> Backbone (e.g., ResNet50) ──> C2 (stride 4), C3 (stride 8), C4 (stride 16), C5 (stride 32)
  │
  ├─> Neck (FPN or PANet) ──────> Top-down & bottom-up fusion ──> P3, P4, P5 (each 256 channels)
  │
  └─> Detection Head ────────────> Per feature level (P3, P4, P5):
                                    ├── Decoupled Head: cls_logits (B, num_classes, H_i, W_i) & reg_pred (B, 4, H_i, W_i)
                                    └── Query Head: Cross-attention over queries ──> (B, N_queries, 4 + num_classes)
```

### Segmentation Feature Flow

```
Input (B, 3, H, W)
  │
  ├─> Backbone ──> High-res low-level features (C2) + Deep semantic features (C5)
  │
  └─> Decoder Head:
       ├── ASPPDecoder: ASPP dilated convs on C5 (dilations 1,6,12,18) + C2 fusion ──> Mask (B, num_classes, H, W)
       └── ResUNetDecoder: Progressive upsampling with residual skip connections  ──> Mask (B, num_classes, H, W)
```

---

## 4. Extending CoreCV via `CoreRegistry`

CoreCV uses `CoreRegistry` to provide generic, type-safe decorators with pre-instantiation signature checking.

### Registering Custom Backbones
```python
import torch
import torch.nn as nn
from corecv.core.contract import BaseBackbone, FeatureInfo
from corecv.core.registry import register_backbone

@register_backbone("CustomResNet")
class CustomResNet(BaseBackbone):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.feature_info = FeatureInfo(
            channels=[64, 128, 256, 512],
            strides=[4, 8, 16, 32],
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [...]
```

### Registering Custom Necks
```python
from corecv.core.registry import register_neck

@register_neck("CustomFPN")
class CustomFPN(nn.Module):
    def __init__(self, feature_info, out_channels: int = 256):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in feature_info.channels
        ])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        return [lat(f) for lat, f in zip(self.laterals, features)]
```

### Registering Custom Heads
```python
from corecv.core.registry import register_head

@register_head("CustomSegHead")
class CustomSegHead(nn.Module):
    def __init__(self, feature_info, out_channels: int = 256, num_classes: int = 19):
        super().__init__()
        self.conv = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.conv(features[0])
```

### Static Signature Validation
Before building models, `CoreRegistry` verifies constructor signatures statically:
```python
from corecv.core.registry import BACKBONE_REGISTRY

# Validates parameters without instantiating the module
BACKBONE_REGISTRY.validate_signature("CustomResNet", pretrained=True)
```

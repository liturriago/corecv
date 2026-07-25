"""ResNet backbone for CoreCV.

Provides ResNet variants (18, 34, 50, 101, 152) as feature extractors
with ``FeatureInfo`` metadata for multi-scale feature consumption.

Each variant exposes multi-scale features from four residual stages
(C2-C5) with strides 4, 8, 16, 32 relative to the input image.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

from corecv.models.backbones.base import BaseBackbone, FeatureInfo

# Type alias for supported ResNet variants.
ResNetVariant = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

# Maps variant name -> (constructor, default weights factory).
_RESNET_REGISTRY: dict[str, tuple[type, type]] = {
    "resnet18": (resnet18, ResNet18_Weights),
    "resnet34": (resnet34, ResNet34_Weights),
    "resnet50": (resnet50, ResNet50_Weights),
    "resnet101": (resnet101, ResNet101_Weights),
    "resnet152": (resnet152, ResNet152_Weights),
}


class ResNetBackbone(BaseBackbone):
    """ResNet multi-scale feature extractor.

    Wraps a TorchVision ``ResNet`` model and exposes four feature levels
    (C2-C5) extracted after each residual stage group.

    Example:
        >>> import torch
        >>> from corecv.models.backbones.resnet import ResNetBackbone
        >>> backbone = ResNetBackbone("resnet50", pretrained=False)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> features, info = backbone(x)
        >>> [f.shape for f in features]
        [torch.Size([2, 256, 56, 56]), torch.Size([2, 512, 28, 28]),
         torch.Size([2, 1024, 14, 14]), torch.Size([2, 2048, 7, 7])]
    """

    def __init__(
        self,
        variant: ResNetVariant = "resnet50",
        *,
        pretrained: bool = False,
    ) -> None:
        """Initialize the ResNet backbone.

        Args:
            variant: ResNet variant name. One of ``resnet18``, ``resnet34``,
                ``resnet50``, ``resnet101``, ``resnet152``.
            pretrained: If ``True``, load ImageNet-1K pretrained weights.
        """
        if variant not in _RESNET_REGISTRY:
            msg = f"Unknown ResNet variant: {variant!r}. Choose from {list(_RESNET_REGISTRY)}"
            raise ValueError(msg)

        constructor, weights_cls = _RESNET_REGISTRY[variant]
        weights = weights_cls.DEFAULT if pretrained else None
        model = constructor(weights=weights)

        # Determine channel dimensions for each stage via dummy forward.
        ch = self._infer_channels(model)

        feature_info = FeatureInfo(
            channels=ch,
            strides=[4, 8, 16, 32],
            names=["C2", "C3", "C4", "C5"],
        )
        super().__init__(feature_info=feature_info)

        # Extract building blocks from the TorchVision ResNet.
        self.conv1: nn.Conv2d = model.conv1
        self.bn1: nn.BatchNorm2d = model.bn1
        self.relu: nn.ReLU = model.relu
        self.maxpool: nn.MaxPool2d = model.maxpool
        self.layer1: nn.Module = model.layer1  # stride 4
        self.layer2: nn.Module = model.layer2  # stride 8
        self.layer3: nn.Module = model.layer3  # stride 16
        self.layer4: nn.Module = model.layer4  # stride 32

    @staticmethod
    def _infer_channels(model: nn.Module) -> list[int]:
        """Infer output channel counts from a ResNet model.

        Args:
            model: A TorchVision ResNet model instance.

        Returns:
            List of four channel counts for layers 1-4.
        """
        dummy = torch.randn(1, 3, 224, 224)
        x = model.conv1(dummy)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        channels: list[int] = []
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            x = layer(x)
            channels.append(x.shape[1])
        return channels

    def forward(self, x: Tensor) -> tuple[list[Tensor], FeatureInfo]:
        """Extract multi-scale features from the input tensor.

        Args:
            x: Input image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Tuple of ``(features, feature_info)`` where *features* is a list
            of four feature tensors with strides 4x, 8x, 16x, 32x relative
            to the input, and *feature_info* contains channel/stride metadata.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c2, c3, c4, c5], self.feature_info


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    for variant in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
        backbone = ResNetBackbone(variant=variant, pretrained=False)
        backbone.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features, info = backbone(dummy_input)

        print(f"--- {variant} ---")  # noqa: T201
        for i, (feat, ch, stride) in enumerate(
            zip(features, info.channels, info.strides, strict=True),
        ):
            print(  # noqa: T201
                f"  Level {i} ({info.names[i]}): "
                f"channels={ch}, stride={stride}, "
                f"shape={feat.shape}",
            )

        assert [f.shape[1] for f in features] == info.channels, "Channel mismatch!"  # noqa: S101
    print("\nAll ResNet backbone tests passed.")  # noqa: T201

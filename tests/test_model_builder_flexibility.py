"""Tests for CoreModel builder flexibility — Milestone 7.

Validates:
- CoreModel("resnet18", task="classification") — plain backbone name string
- CoreModel({"task": "detection", ...}, task="detection") — raw dict
- Custom neck/head parameter propagation via signature inspection
- Dynamic kwarg forwarding prevents TypeError for custom heads/necks
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import nn

from corecv.api import CoreModel
from corecv.core.contract import FeatureInfo
from corecv.core.registry import HEAD_REGISTRY, NECK_REGISTRY, get_backbone
from corecv.models.heads.classification import LinearClassificationHead  # noqa: F401
from corecv.models.heads.detection import DecoupledAnchorFreeHead  # noqa: F401
from corecv.models.heads.segmentation import ASPPDecoder, ResUNetDecoder  # noqa: F401
from corecv.models.necks.fpn import FPN  # noqa: F401
from corecv.models.necks.panet import PANet  # noqa: F401


def _no_download_backbone(backbone_cls: type) -> type:
    """Wrap a backbone class to force ``pretrained=False``."""

    class _Wrapper(backbone_cls):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: object) -> None:
            kwargs.pop("pretrained", None)
            super().__init__(pretrained=False, **kwargs)

    _Wrapper.__name__ = backbone_cls.__name__
    _Wrapper.__qualname__ = backbone_cls.__qualname__
    return _Wrapper


def _patched_get_backbone(name: str) -> type:
    """Return a backbone class that never downloads weights."""
    cls = get_backbone(name)
    return _no_download_backbone(cls)


# ======================================================================
# 1. Plain backbone name string
# ======================================================================


class TestPlainBackboneNameString:
    """CoreModel('resnet18', task=...) — bare string as backbone name."""

    @patch("corecv.api.model.get_backbone", side_effect=_patched_get_backbone)
    def test_init_with_plain_backbone_name_string_classification(
        self,
        _mock: object,
    ) -> None:
        """CoreModel('resnet18', task='classification') succeeds."""
        cm = CoreModel("resnet18", task="classification")
        assert isinstance(cm.model, nn.Module)
        assert cm.task == "classification"
        assert cm.num_classes == 1000

    @patch("corecv.api.model.get_backbone", side_effect=_patched_get_backbone)
    def test_init_with_plain_backbone_name_string_detection(
        self,
        _mock: object,
    ) -> None:
        """CoreModel('resnet50', task='detection') succeeds."""
        cm = CoreModel("resnet18", task="detection")
        assert isinstance(cm.model, nn.Module)
        assert cm.task == "detection"


# ======================================================================
# 2. Raw dict support
# ======================================================================


class TestRawDictInput:
    """CoreModel({...}, task=...) — raw configuration dictionary."""

    @patch("corecv.api.model.get_backbone", side_effect=_patched_get_backbone)
    def test_init_with_raw_dict_detection(self, _mock: object) -> None:
        """Dict with neck_channels=128 produces neck with out_channels=128."""
        cm = CoreModel(
            {
                "model_name": "resnet18",
                "neck_channels": 128,
                "pretrained": False,
            },
            task="detection",
        )
        assert isinstance(cm.model, nn.Module)
        assert cm.task == "detection"
        neck = cm.model.neck
        assert neck.out_channels == 128

    @patch("corecv.api.model.get_backbone", side_effect=_patched_get_backbone)
    def test_init_with_raw_dict_segmentation(self, _mock: object) -> None:
        """Dict with decoder_channels=128 for segmentation."""
        cm = CoreModel(
            {
                "model_name": "resnet18",
                "decoder_channels": 128,
                "pretrained": False,
            },
            task="segmentation",
        )
        assert isinstance(cm.model, nn.Module)
        assert cm.task == "segmentation"

    @patch("corecv.api.model.get_backbone", side_effect=_patched_get_backbone)
    def test_init_with_raw_dict_classification(self, _mock: object) -> None:
        """Dict with num_classes for classification."""
        cm = CoreModel(
            {
                "model_name": "resnet18",
                "num_classes": 42,
                "pretrained": False,
            },
            task="classification",
        )
        assert cm.num_classes == 42


# ======================================================================
# 3. Dynamic kwarg forwarding — custom head
# ======================================================================


class _CustomHead(nn.Module):
    """Custom head with non-standard constructor parameters."""

    def __init__(
        self,
        feature_info: FeatureInfo,
        num_classes: int,
        custom_param: int = 64,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.custom_param = custom_param
        self.pool = nn.AdaptiveAvgPool2d(1)
        total_channels = sum(feature_info.channels.values())
        self.fc = nn.Linear(total_channels, num_classes)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            pooled = [self.pool(f) for f in x]
            flat = torch.cat([p.view(p.size(0), -1) for p in pooled], dim=1)
        else:
            flat = self.pool(x).view(x.size(0), -1)
        return self.fc(flat)


class _CustomNeck(nn.Module):
    """Custom neck with non-standard constructor parameters."""

    def __init__(
        self,
        feature_info: FeatureInfo,
        custom_width: int = 128,
        **kwargs: object,
    ) -> None:
        super().__init__()
        del kwargs
        self.custom_width = custom_width
        self.out_channels = custom_width
        self._feature_info = feature_info
        convs: list[tuple[str, nn.Module]] = []
        for level_name, ch in feature_info.channels.items():
            convs.append((
                f"lateral_{level_name}",
                nn.Conv2d(ch, custom_width, 1),
            ))
        self.convs = nn.ModuleDict(convs)

    def forward(
        self, x: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        levels = sorted(
            self._feature_info.strides,
            key=lambda k: self._feature_info.strides[k],
        )
        return [self.convs[f"lateral_{lvl}"](feat) for lvl, feat in zip(levels, x, strict=True)]


class TestDynamicKwargForwarding:
    """Custom heads/necks with non-standard signatures."""

    @patch("corecv.api.model.get_backbone", side_effect=_patched_get_backbone)
    def test_dynamic_kwarg_forwarding_custom_head(
        self,
        _mock: object,
    ) -> None:
        """Custom head with ``custom_param`` is instantiated without TypeError."""
        head_name = "test_custom_cls_head_m7"
        HEAD_REGISTRY.register(head_name, _CustomHead)
        try:
            cm = CoreModel(
                {
                    "model_name": "resnet18",
                    "head_type": head_name,
                    "num_classes": 10,
                    "custom_param": 96,
                    "pretrained": False,
                },
                task="classification",
            )
            assert isinstance(cm.model, nn.Module)
            head = cm.model.head
            assert isinstance(head, _CustomHead)
            assert head.custom_param == 96
        finally:
            HEAD_REGISTRY._registry.pop(head_name, None)

    @patch("corecv.api.model.get_backbone", side_effect=_patched_get_backbone)
    def test_dynamic_kwarg_forwarding_custom_neck(
        self,
        _mock: object,
    ) -> None:
        """Custom neck with ``custom_width`` is instantiated without TypeError."""
        neck_name = "test_custom_neck_m7"
        NECK_REGISTRY.register(neck_name, _CustomNeck)
        try:
            cm = CoreModel(
                {
                    "model_name": "resnet18",
                    "neck_type": neck_name,
                    "custom_width": 192,
                    "pretrained": False,
                },
                task="detection",
            )
            assert isinstance(cm.model, nn.Module)
            neck = cm.model.neck
            assert isinstance(neck, _CustomNeck)
            assert neck.custom_width == 192
        finally:
            NECK_REGISTRY._registry.pop(neck_name, None)


# ======================================================================
# 4. Neck channels from config
# ======================================================================


class TestNeckChannelsFromConfig:
    """Verify neck_channels in config overrides the default 256."""

    @patch("corecv.api.model.get_backbone", side_effect=_patched_get_backbone)
    def test_neck_channels_from_config(self, _mock: object) -> None:
        """neck_channels=512 is propagated to the FPN neck."""
        cm = CoreModel(
            {
                "model_name": "resnet18",
                "neck_channels": 512,
                "pretrained": False,
            },
            task="detection",
        )
        assert cm.model.neck.out_channels == 512

    @patch("corecv.api.model.get_backbone", side_effect=_patched_get_backbone)
    def test_default_neck_channels_is_256(self, _mock: object) -> None:
        """Default neck out_channels is 256 when not specified."""
        cm = CoreModel(
            {"model_name": "resnet18", "pretrained": False},
            task="detection",
        )
        assert cm.model.neck.out_channels == 256


# ======================================================================
# 5. Backward compatibility
# ======================================================================


class TestBackwardCompatibility:
    """Existing CoreModel(nn.Module) usage still works."""

    def test_backward_compatibility_nn_module(self) -> None:
        """Passing an nn.Module directly still works."""

        class _TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.head = nn.Linear(3, 10)
                self.num_classes = 10

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.head(x.mean(dim=(2, 3)))

        raw = _TinyModel()
        cm = CoreModel(raw, task="classification", input_size=(224, 224))
        assert cm.model is raw
        assert cm.task == "classification"
        assert cm.num_classes == 10


# ======================================================================
# 6. Builder params via __init__ keyword arguments
# ======================================================================


class TestBuilderParamsViaInit:
    """CoreModel builder params (pretrained, neck, head, **kwargs) via __init__."""

    @patch("corecv.api.model.get_backbone", side_effect=_patched_get_backbone)
    def test_init_with_builder_params_neck_head_pretrained(self, _mock: object) -> None:
        """CoreModel with neck='panet', head='decoupled_anchor_free', pretrained=False."""
        cm = CoreModel(
            "resnet18",
            task="detection",
            pretrained=False,
            neck="panet",
            head="decoupled_anchor_free",
        )
        assert isinstance(cm.model, nn.Module)
        assert cm.task == "detection"
        assert isinstance(cm.model.neck, PANet)

    @patch("corecv.api.model.get_backbone", side_effect=_patched_get_backbone)
    def test_init_with_kwargs_forwarding(self, _mock: object) -> None:
        """Kwargs are forwarded to component constructors."""
        cm = CoreModel(
            "resnet18",
            task="detection",
            pretrained=False,
            neck_channels=384,
        )
        assert cm.model.neck.out_channels == 384

    @patch("corecv.api.model.get_backbone", side_effect=_patched_get_backbone)
    def test_init_dict_with_builder_params(self, _mock: object) -> None:
        """Builder params override/merge with dict config."""
        cm = CoreModel(
            {"model_name": "resnet18"},
            task="detection",
            pretrained=False,
            neck="panet",
            neck_channels=192,
        )
        assert isinstance(cm.model, nn.Module)
        assert isinstance(cm.model.neck, PANet)
        assert cm.model.neck.out_channels == 192


# ======================================================================
# Pytest entry point
# ======================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

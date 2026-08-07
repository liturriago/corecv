"""Necks subpackage for CoreCV.

This subpackage contains feature fusion modules (necks) that combine
multi-scale features from backbones before passing them to task-specific
heads. Supported architectures:

- **FPN**: Feature Pyramid Network for top-down feature fusion.
- **PANet**: Path Aggregation Network with bottom-up augmentation.
- **BiFPN**: Bidirectional Feature Pyramid Network with weighted fusion.
- **CSPPANet**: Bidirectional pyramid with concatenation and cross-stage
  partial refinement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from corecv.models.necks.bifpn import BiFPN
from corecv.models.necks.csppanet import CSPPANet
from corecv.models.necks.fpn import FPN
from corecv.models.necks.panet import PANet

if TYPE_CHECKING:
    from torch import nn

# Union of all supported neck names.
NeckName = Literal["fpn", "panet", "bifpn", "csppanet"]

# Registry mapping name -> neck class.
_NECK_REGISTRY: dict[str, type[nn.Module]] = {
    "fpn": FPN,
    "panet": PANet,
    "bifpn": BiFPN,
    "csppanet": CSPPANet,
}


def create_neck(
    name: NeckName,
    in_channels: list[int],
    out_channels: int = 256,
    **kwargs: object,
) -> nn.Module:
    """Create a neck module by name.

    Args:
        name: Neck variant name (e.g., ``fpn``, ``panet``, ``bifpn``).
        in_channels: Channel dimensions of each input feature level,
            ordered from finest (highest resolution) to coarsest.
        out_channels: Number of channels in each output feature map.
        **kwargs: Additional arguments passed to the neck class constructor.

    Returns:
        An instantiated neck :class:`nn.Module`.

    Raises:
        ValueError: If *name* is not a recognized neck name.

    Example:
        >>> from corecv.models.necks import create_neck
        >>> neck = create_neck("fpn", in_channels=[64, 128, 256], out_channels=256)

    """
    if name not in _NECK_REGISTRY:
        msg = f"Unknown neck: {name!r}. Choose from {list(_NECK_REGISTRY)}"
        raise ValueError(msg)

    neck_cls = _NECK_REGISTRY[name]
    return neck_cls(in_channels=in_channels, out_channels=out_channels, **kwargs)


__all__ = [
    "FPN",
    "BiFPN",
    "CSPPANet",
    "NeckName",
    "PANet",
    "create_neck",
]

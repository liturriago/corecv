"""API module for CoreCV.

Provides high-level unified interfaces for CoreCV models, including the
:class:`CoreModel` facade that wraps training, inference, and export
through a single entry point.
"""

from corecv.api.model import CoreModel, ExportConfig, TrainingConfig

__all__ = [
    "CoreModel",
    "ExportConfig",
    "TrainingConfig",
]

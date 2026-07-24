"""CLI module for CoreCV.

Provides the command-line interface for common CoreCV workflows:
training, inference, and model export via the Typer application.
"""

from corecv.cli.main import app

__all__ = ["app"]

"""CoreCV: Production-ready Computer Vision Library."""

__version__ = "0.1.0"


def main() -> None:
    """Entry point for the corecv CLI.

    Delegates to the Typer application defined in :mod:`corecv.cli.main`.
    """
    from corecv.cli.main import app  # noqa: PLC0415
    app()

"""CoreCV CLI — train, predict, and export models from the terminal.

Provides a :class:`typer.Typer` application with three commands:

* ``train`` — Train a CoreCV model.
* ``predict`` — Run inference with a trained model.
* ``export`` — Export a model to ONNX / ExecuTorch.

Usage::

    $ corecv --help
    $ corecv train --help
    $ corecv predict --help
    $ corecv export --help
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from corecv.api import CoreModel
from corecv.api.model import TrainingConfig

# ---------------------------------------------------------------------------
# App & console
# ---------------------------------------------------------------------------

app: typer.Typer = typer.Typer(
    name="corecv",
    help="CoreCV: Production-ready Computer Vision Library",
    add_completion=False,
    no_args_is_help=True,
)

console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    """Configure logging with Rich handler.

    Args:
        verbose: Enable DEBUG-level logging when ``True``.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


# ---------------------------------------------------------------------------
# Shared callbacks
# ---------------------------------------------------------------------------


def _validate_task(task: str) -> str:
    """Normalise and validate the task type.

    Args:
        task: Raw task string from the CLI.

    Returns:
        Lowercased, validated task name.

    Raises:
        SystemExit: If the task is not supported.
    """
    task_norm = task.lower().strip()
    valid = {"classification", "segmentation", "detection"}
    if task_norm not in valid:
        valid_str = ", ".join(sorted(valid))
        console.print(f"[red]Error:[/] Unsupported task {task!r}. Must be one of: {valid_str}")
        raise typer.Exit(code=1)
    return task_norm


def _load_model(path: Path) -> CoreModel:
    """Load a serialized :class:`CoreModel` from a pickle file.

    .. note::
        This is a placeholder implementation. In a production setting,
        consider using ``torch.package`` or a custom serialisation
        format (e.g. via ``torch.save`` model state dicts with a
        companion configuration file).

    Args:
        path: Path to the saved model file.

    Returns:
        A deserialized :class:`CoreModel` instance.

    Raises:
        SystemExit: If the file does not exist or cannot be loaded.
    """
    if not path.exists():
        console.print(f"[red]Error:[/] Model file not found: {path}")
        raise typer.Exit(code=1)

    import pickle  # noqa: PLC0415

    try:
        with path.open("rb") as f:
            model: CoreModel = pickle.load(f)  # noqa: S301
    except Exception as exc:
        console.print(f"[red]Error:[/] Failed to load model from {path}: {exc}")
        raise typer.Exit(code=1) from exc

    return model


def _model_path_argument(**overrides: object) -> typer.Argument:  # noqa: ANN401
    """Return a shared ``typer.Argument`` for model path."""
    kwargs: dict[str, object] = {
        "help": "Path to a serialized CoreModel file (.pkl)",
        "exists": True,
        "dir_okay": False,
        "readable": True,
    }
    kwargs.update(overrides)
    return typer.Argument(..., **kwargs)  # type: ignore[return-value]


# ======================================================================
# Command: train
# ======================================================================


@app.command()
def train(  # noqa: PLR0913
    model_path: Path = _model_path_argument(),
    # --- Training hyperparameters ---
    epochs: int = typer.Option(
        100,
        "--epochs",
        "-e",
        help="Number of training epochs",
        min=1,
    ),
    lr: float = typer.Option(
        0.001,
        "--lr",
        "-l",
        help="Learning rate",
        min=0.0,
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size per device",
        min=1,
    ),
    optimizer: str = typer.Option(
        "adamw",
        "--optimizer",
        "-o",
        help="Optimizer (adamw, adam, sgd)",
    ),
    scheduler: str | None = typer.Option(
        None,
        "--scheduler",
        "-s",
        help="Learning rate scheduler (cosine, step, none)",
    ),
    amp: bool = typer.Option(
        True,
        "--amp/--no-amp",
        help="Enable automatic mixed precision",
    ),
    grad_accum: int = typer.Option(
        1,
        "--grad-accum",
        "-g",
        help="Gradient accumulation steps",
        min=1,
    ),
    ema_decay: float = typer.Option(
        0.9999,
        "--ema-decay",
        help="EMA decay factor (0 < decay < 1)",
    ),
    output_dir: str = typer.Option(
        "./checkpoints",
        "--output-dir",
        "-d",
        help="Directory for checkpoints and logs",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Target device (e.g. 'cuda', 'cpu')",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose (DEBUG) logging",
    ),
) -> None:
    """Train a CoreCV model.

    Loads a serialized :class:`CoreModel` from ``MODEL_PATH`` and runs
    the training loop with the provided hyperparameters.

    Basic usage::

        $ corecv train model.pkl --train-data /path/to/images --epochs 50 --lr 0.001
    """
    _setup_logging(verbose)
    model: CoreModel = _load_model(model_path)

    if scheduler and scheduler.lower() == "none":
        scheduler = None

    config = TrainingConfig(
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
        amp=amp,
        grad_accum=grad_accum,
        ema_decay=ema_decay,
        output_dir=output_dir,
        device=device or "auto",
    )

    dev_str = device or "auto"
    console.print(f"[bold cyan]Starting training[/] for {epochs} epoch(s) on device={dev_str}")
    console.print(f"  Model task: {model.task}")
    console.print(f"  Output dir: {output_dir}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Training...", total=epochs)

        # Capture progress via a callback
        def _progress_callback() -> None:
            progress.update(task_id, advance=1)

        # We intercept: train() does not support progress callbacks natively,
        # so we run synchronously and update at the end.
        # In a real integration, the trainer could report step-level progress.
        history = model.train(config=config)

        progress.update(task_id, completed=epochs)

    final_train_loss = history.get("train", [{}])[-1].get("loss", "N/A")
    console.print(f"[green]Training complete![/] Final train loss: {final_train_loss}")


# ======================================================================
# Command: predict
# ======================================================================


@app.command()
def predict(
    model_path: Path = _model_path_argument(),
    source: str = typer.Argument(
        ...,
        help="Input source — image path, directory, or tensor dump (.pt)",
    ),
    conf_threshold: float | None = typer.Option(
        None,
        "--conf",
        "-c",
        help="Confidence threshold for detection",
        min=0.0,
        max=1.0,
    ),
    iou_threshold: float | None = typer.Option(
        None,
        "--iou",
        help="IoU threshold for NMS",
        min=0.0,
        max=1.0,
    ),
    topk: int | None = typer.Option(
        None,
        "--topk",
        "-k",
        help="Top-K predictions for classification",
        min=1,
    ),
    half: bool = typer.Option(
        False,
        "--half",
        help="Enable FP16 half-precision inference",
    ),
    compile_model: bool = typer.Option(
        False,
        "--compile",
        help="Enable torch.compile (PyTorch >= 2.0)",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        "-b",
        help="Maximum batch size for batched inference",
        min=1,
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Save predictions as JSON to this file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose (DEBUG) logging",
    ),
) -> None:
    """Run inference with a trained CoreCV model.

    Loads a serialized :class:`CoreModel` and runs inference on the
    provided ``SOURCE`` (image file, directory, or tensor file).

    Basic usage::

        $ corecv predict model.pkl image.jpg --topk 5
        $ corecv predict model.pkl /path/to/images/ --batch-size 16
    """
    _setup_logging(verbose)
    model: CoreModel = _load_model(model_path)

    # Resolve source
    source_resolved: str | Path | torch.Tensor = _resolve_source(source)

    console.print(f"[bold cyan]Running inference[/] on {source}")
    console.print(f"  Model task: {model.task}")
    console.print(f"  Half-precision: {half}")
    console.print(f"  Compiled: {compile_model}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("[cyan]Predicting...", total=None)
        predictions = model.predict(
            source=source_resolved,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            topk=topk,
            half_precision=half,
            compile_model=compile_model,
            batch_size=batch_size,
        )

    console.print(f"[green]Inference complete![/] {len(predictions)} prediction(s) returned.")

    # Display a summary table
    table = Table(title="Predictions Summary", title_style="bold cyan")
    table.add_column("#", style="dim")
    table.add_column("Source", style="cyan")
    table.add_column("Task", style="green")

    for i, pred in enumerate(predictions):
        table.add_row(str(i + 1), str(pred.filename or f"input_{i}"), model.task)

    console.print(table)

    # Save to JSON if requested
    if output is not None:
        _save_predictions(predictions, output)


def _resolve_source(source: str) -> str | Path:  # noqa: ANN201
    """Resolve the source argument to a usable format.

    Args:
        source: Raw source string from the CLI.

    Returns:
        A path or tensor suitable for ``CoreModel.predict()``.
    """
    path = Path(source)
    if path.suffix == ".pt":
        import torch  # noqa: PLC0415
        return torch.load(path, map_location="cpu", weights_only=True)  # noqa: S311
    return source


def _save_predictions(
    predictions: list,
    output_path: str,
) -> None:
    """Save predictions to a JSON file.

    Args:
        predictions: List of :class:`Prediction` objects.
        output_path: Destination JSON file path.
    """
    import json  # noqa: PLC0415

    records: list[dict] = []
    for pred in predictions:
        record: dict = {
            "filename": str(pred.filename) if pred.filename else None,
            "task": pred.task if hasattr(pred, "task") else "unknown",
        }
        records.append(record)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    console.print(f"[green]Predictions saved to:[/] {out}")


# ======================================================================
# Command: export
# ======================================================================


@app.command()
def export(
    model_path: Path = _model_path_argument(),
    format: str = typer.Option(
        "onnx",
        "--format",
        "-f",
        help="Export format (onnx, executorch, both)",
    ),
    target_hardware: str = typer.Option(
        "server",
        "--target-hardware",
        "-hw",
        help="Hardware profile (edge, server)",
    ),
    opset: int = typer.Option(
        17,
        "--opset",
        help="ONNX opset version (17 or 18)",
    ),
    optimize: bool = typer.Option(
        True,
        "--optimize/--no-optimize",
        help="Apply graph optimisations and rewrites",
    ),
    output_path: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (or stem for 'both' format)",
    ),
    input_shape: str | None = typer.Option(
        None,
        "--input-shape",
        help="Input shape as comma-separated integers, e.g. '1,3,224,224'",
    ),
    dynamic_axes: bool = typer.Option(
        False,
        "--dynamic-axes",
        help="Enable dynamic batch/height/width axes",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose (DEBUG) logging",
    ),
) -> None:
    """Export a CoreCV model to ONNX and/or ExecuTorch.

    Loads a serialized :class:`CoreModel` and exports it to the specified
    format, optionally applying edge-hardware activation rewrites.

    Basic usage::

        $ corecv export model.pkl --format onnx --target-hardware edge
        $ corecv export model.pkl --format both --opset 18 --output model_export
    """
    _setup_logging(verbose)
    model: CoreModel = _load_model(model_path)

    resolved_shape: tuple[int, ...] | None = None
    if input_shape is not None:
        parts: list[str] = [s.strip() for s in input_shape.split(",")]
        resolved_shape = tuple(int(p) for p in parts)

    resolved_dynamic: dict[str, dict[int, str]] | None = None
    if dynamic_axes:
        resolved_dynamic = {
            "input": {0: "batch", 2: "height", 3: "width"},
        }

    console.print(f"[bold cyan]Exporting model[/] (format={format}, hardware={target_hardware})")
    console.print(f"  Model task: {model.task}")
    console.print(f"  Opset: {opset}")
    console.print(f"  Optimize: {optimize}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("[cyan]Exporting...", total=None)
        results: dict[str, str] = model.export(
            format=format,
            target_hardware=target_hardware,
            opset=opset,
            optimize=optimize,
            output_path=output_path,
            input_shape=resolved_shape,
            dynamic_axes=resolved_dynamic,
        )

    # Display results table
    table = Table(title="Export Results", title_style="bold cyan")
    table.add_column("Format", style="green")
    table.add_column("Path", style="cyan")

    for fmt, path in results.items():
        table.add_row(fmt, path)

    console.print(table)
    console.print("[green]Export complete![/]")


# ======================================================================
# Entry point
# ======================================================================


def main() -> None:
    """Entry point for the ``corecv`` CLI.

    Delegates to :func:`typer.Typer.__call__`.
    """
    app()


if __name__ == "__main__":
    main()

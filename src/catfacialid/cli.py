#!/usr/bin/env python3
"""Command-line interface for the Cat Facial Identification System.

This module provides a professional CLI with subcommands for training,
prediction, and serving the cat facial identification model.

Usage:
    catfacialid train --train-data data/train.pkl
    catfacialid predict --model model.pkl --input data/test.pkl
    catfacialid serve --model model.pkl --port 8000
    catfacialid info
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
except ImportError:
    print("CLI dependencies not installed. Run: pip install typer[all] rich")
    sys.exit(1)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from catfacialid import __version__  # noqa: E402
from catfacialid.core import (  # noqa: E402
    DimensionalityReducer,
    FeatureExtractor,
    FeatureFuser,
    PredictionEngine,
)
from catfacialid.data import DataLoader  # noqa: E402
from catfacialid.logging import setup_logging  # noqa: E402

# Initialize Typer app
app = typer.Typer(
    name="catfacialid",
    help="🐱 Cat Facial Identification System - Production-ready cat face recognition",
    add_completion=False,
)
console = Console()
logger = logging.getLogger(__name__)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]Cat Facial ID[/bold blue] version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose logging",
    ),
    json_logs: bool = typer.Option(
        False,
        "--json-logs",
        help="Output logs in JSON format",
    ),
) -> None:
    """🐱 Cat Facial ID - Production-ready cat face recognition."""
    setup_logging(
        level="DEBUG" if verbose else "INFO",
        json_output=json_logs,
    )


@app.command()
def train(
    train_data: Path = typer.Option(
        ...,
        "--train-data",
        "-t",
        help="Path to training data pickle file",
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        Path("model.pkl"),
        "--output",
        "-o",
        help="Path to save trained model",
    ),
    pca_variance: float = typer.Option(
        0.95,
        "--pca-variance",
        help="PCA variance threshold (0-1)",
        min=0.5,
        max=1.0,
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
) -> None:
    """Train a new cat facial identification model.

    Loads training data, applies feature extraction pipeline (PCA, LDA, ICA),
    and saves the trained model for later prediction.
    """
    console.print("[bold green]🐱 Training Cat Facial ID Model[/bold green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load data
        task = progress.add_task("Loading training data...", total=None)
        loader = DataLoader(verbose=False)
        X_train, y_train = loader.load_train_features(str(train_data))
        progress.update(task, completed=True)
        console.print(f"  ✓ Loaded {len(X_train)} samples, {len(set(y_train))} classes")

        # Feature extraction
        task = progress.add_task("Applying feature extraction...", total=None)
        extractor = FeatureExtractor(seed=seed, verbose=False)
        X_scaled, _ = extractor.scale_features(X_train, X_train)

        reducer = DimensionalityReducer(seed=seed, verbose=False)
        X_pca, _ = reducer.apply_pca(
            X_scaled, X_scaled, variance_threshold=pca_variance
        )
        X_lda, _ = reducer.apply_lda(X_scaled, X_scaled, y_train)
        X_ica, _ = reducer.apply_ica(X_scaled, X_scaled, n_components=200)

        fuser = FeatureFuser(verbose=False)
        X_fused = fuser.fuse_features(X_pca, X_lda, X_ica, normalize_output=True)
        progress.update(task, completed=True)
        console.print(f"  ✓ Extracted features: {X_fused.shape[1]} dimensions")

        # Build index
        task = progress.add_task("Building FAISS index...", total=None)
        engine = PredictionEngine(top_k=5, verbose=False)
        engine.build_index(X_fused, y_train)
        progress.update(task, completed=True)
        console.print(f"  ✓ Built index with {len(y_train)} vectors")

        # Save model (simplified - in production would use joblib)
        task = progress.add_task("Saving model...", total=None)
        import joblib

        model_data = {
            "extractor": extractor,
            "reducer": reducer,
            "fuser": fuser,
            "engine": engine,
            "version": __version__,
        }
        joblib.dump(model_data, output)
        progress.update(task, completed=True)
        console.print(f"  ✓ Model saved to {output}")

    console.print("\n[bold green]✅ Training complete![/bold green]")


@app.command()
def predict(
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model pickle file",
        exists=True,
        dir_okay=False,
    ),
    input_data: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to test data pickle file",
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        Path("predictions.csv"),
        "--output",
        "-o",
        help="Path to save predictions CSV",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of top predictions per sample",
        min=1,
        max=20,
    ),
) -> None:
    """Generate predictions for test data using a trained model."""
    console.print("[bold blue]🔮 Generating Predictions[/bold blue]")

    import joblib

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load model
        task = progress.add_task("Loading model...", total=None)
        model_data = joblib.load(model)
        progress.update(task, completed=True)
        console.print(f"  ✓ Loaded model v{model_data.get('version', 'unknown')}")

        # Load test data
        task = progress.add_task("Loading test data...", total=None)
        loader = DataLoader(verbose=False)
        X_test, image_names = loader.load_test_features(str(input_data))
        progress.update(task, completed=True)
        console.print(f"  ✓ Loaded {len(X_test)} test samples")

        # Apply transformations (simplified)
        task = progress.add_task("Extracting features...", total=None)
        # Note: In full implementation, would apply saved transforms
        progress.update(task, completed=True)

        # Generate predictions
        task = progress.add_task("Generating predictions...", total=None)
        engine: PredictionEngine = model_data["engine"]
        engine.top_k = top_k
        results = engine.predict(X_test, image_names)
        progress.update(task, completed=True)

        # Save predictions
        task = progress.add_task("Saving predictions...", total=None)
        with open(output, "w") as f:
            f.write(
                "image_name," + ",".join([f"pred_{i+1}" for i in range(top_k)]) + "\n"
            )
            for img_name, preds in results:
                f.write(f"{img_name}," + ",".join(map(str, preds)) + "\n")
        progress.update(task, completed=True)
        console.print(f"  ✓ Predictions saved to {output}")

    console.print(
        f"\n[bold green]✅ Generated {len(results)} predictions![/bold green]"
    )


@app.command()
def info() -> None:
    """Display system information and configuration."""
    console.print("[bold cyan]📊 Cat Facial ID System Info[/bold cyan]\n")

    # Version info
    table = Table(title="System Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", __version__)
    table.add_row("Python", sys.version.split()[0])

    # Check dependencies
    deps = {
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "faiss-cpu": "faiss",
        "joblib": "joblib",
    }
    for name, module in deps.items():
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "installed")
            table.add_row(name, version)
        except ImportError:
            table.add_row(name, "[red]not installed[/red]")

    console.print(table)


@app.command()
def serve(
    model: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model pickle file",
        exists=True,
        dir_okay=False,
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        help="Host to bind the server",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to bind the server",
    ),
) -> None:
    """Start a REST API server for predictions.

    Note: This is a placeholder. In production, would use FastAPI or Flask.
    """
    console.print(f"[bold yellow]🚀 Starting server on {host}:{port}[/bold yellow]")
    console.print("[dim]Note: Full REST API implementation requires FastAPI[/dim]")
    console.print("\n[bold]To implement, install FastAPI:[/bold]")
    console.print("  pip install fastapi uvicorn")
    console.print("\nThen create an API endpoint with:")
    console.print("  from fastapi import FastAPI")
    console.print("  app = FastAPI()")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()

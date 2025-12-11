"""Example script demonstrating the complete cat facial identification pipeline.

This script shows how to use all components of the system together,
from data loading through feature extraction to prediction generation.

Usage:
    python examples/complete_pipeline.py --train-path path/to/train.pkl \\
                                         --test-path path/to/test.pkl \\
                                         --output-dir ./results
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from catfacialid.config import SystemConfig  # noqa: E402
from catfacialid.core import (  # noqa: E402
    FeatureExtractor,
    DimensionalityReducer,
    FeatureFuser,
    PredictionEngine,
)
from catfacialid.data import DataLoader  # noqa: E402

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def run_pipeline(
    train_features_path: str,
    test_features_path: str,
    output_dir: str = "./results",
    top_k: int = 3,
    verbose: bool = False,
) -> dict:
    """Execute complete identification pipeline.

    Args:
        train_features_path: Path to training features pickle file.
        test_features_path: Path to test features pickle file.
        output_dir: Directory to save results.
        top_k: Number of top predictions to return.
        verbose: Enable verbose logging.

    Returns:
        Dictionary with results and statistics.

    Raises:
        FileNotFoundError: If input files not found.
        RuntimeError: If pipeline execution fails.
    """
    setup_logging(verbose=verbose)

    results = {
        "status": "success",
        "errors": [],
        "statistics": {},
        "predictions": [],
    }

    try:
        # Configuration
        logger.info("Initializing system configuration...")
        config = SystemConfig.default()
        config.model.top_k_predictions = top_k

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path}")

        # Step 1: Load data
        logger.info("Loading features and labels...")
        loader = DataLoader(verbose=verbose)
        X_train, y_train = loader.load_train_features(train_features_path)
        X_test, image_names = loader.load_test_features(test_features_path)

        # Log statistics
        stats = loader.get_stats()
        results["statistics"]["data"] = stats
        logger.info(f"Loaded training data: {X_train.shape}")
        logger.info(f"Loaded test data: {X_test.shape}")
        logger.info(f"Number of classes: {stats['train_classes']}")

        # Step 2: Feature extraction and scaling
        logger.info("Scaling features...")
        extractor = FeatureExtractor(seed=config.model.random_seed, verbose=verbose)
        X_train_scaled, X_test_scaled = extractor.scale_features(X_train, X_test)

        # Step 3: Dimensionality reduction
        logger.info("Applying dimensionality reduction techniques...")
        reducer = DimensionalityReducer(seed=config.model.random_seed, verbose=verbose)

        # PCA
        X_train_pca, X_test_pca = reducer.apply_pca(
            X_train_scaled,
            X_test_scaled,
            variance_threshold=config.preprocessing.pca_variance_threshold,
        )

        # LDA
        X_train_lda, X_test_lda = reducer.apply_lda(X_train_pca, X_test_pca, y_train)

        # ICA
        X_train_ica, X_test_ica = reducer.apply_ica(
            X_train_scaled,
            X_test_scaled,
            n_components=config.preprocessing.ica_n_components,
            max_iterations=config.preprocessing.ica_max_iterations,
        )

        # Step 4: Feature fusion
        logger.info("Fusing features from multiple sources...")
        fuser = FeatureFuser(verbose=verbose)
        X_train_fused = fuser.fuse_features(
            X_train_pca, X_train_lda, X_train_ica, normalize_output=True
        )
        X_test_fused = fuser.fuse_features(
            X_test_pca, X_test_lda, X_test_ica, normalize_output=True
        )

        results["statistics"]["fused_dimension"] = X_train_fused.shape[1]
        logger.info(f"Fused feature dimension: {X_train_fused.shape[1]}")

        # Step 5: Build index and generate predictions
        logger.info("Building FAISS index...")
        engine = PredictionEngine(top_k=config.model.top_k_predictions, verbose=verbose)
        engine.build_index(X_train_fused, y_train)

        logger.info("Generating predictions...")
        predictions = engine.predict(X_test_fused, image_names)

        results["predictions"] = [
            {"image": img_name, "top_k": top_labels}
            for img_name, top_labels in predictions
        ]

        logger.info(f"Generated predictions for {len(predictions)} samples")

        # Step 6: Save results
        output_file = output_path / "predictions.txt"
        logger.info(f"Saving predictions to {output_file}...")
        with open(output_file, "w") as f:
            f.write("Image,Prediction_1,Prediction_2,Prediction_3\n")
            for prediction in results["predictions"]:
                img = prediction["image"]
                preds = prediction["top_k"]
                line = f"{img},{','.join(map(str, preds))}\n"
                f.write(line)

        logger.info("Pipeline completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        results["status"] = "failed"
        results["errors"].append(str(e))
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        results["status"] = "failed"
        results["errors"].append(str(e))
        raise

    return results


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run complete cat facial identification pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/complete_pipeline.py \\
    --train-path data/train_features.pkl \\
    --test-path data/test_features.pkl

  python examples/complete_pipeline.py \\
    --train-path data/train.pkl \\
    --test-path data/test.pkl \\
    --output-dir ./submissions \\
    --top-k 5 \\
    --verbose
        """,
    )

    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="Path to training features pickle file",
    )

    parser.add_argument(
        "--test-path",
        type=str,
        required=True,
        help="Path to test features pickle file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for predictions (default: ./results)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to return (default: 3)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    try:
        results = run_pipeline(
            train_features_path=args.train_path,
            test_features_path=args.test_path,
            output_dir=args.output_dir,
            top_k=args.top_k,
            verbose=args.verbose,
        )
        print(f"Pipeline completed: {results['status']}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Data loading utilities for the Cat Facial ID system."""

import logging
from pathlib import Path
from typing import Tuple, List, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage training and testing datasets.

    This class handles loading pre-extracted features from pickle files
    and provides utility methods for dataset inspection and validation.
    """

    def __init__(self, verbose: bool = True):
        """Initialize DataLoader.

        Args:
            verbose: Enable verbose logging output.
        """
        self.verbose = verbose
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.test_image_names: Optional[List[str]] = None

    def load_train_features(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load pre-extracted training features and labels.

        Args:
            filepath: Path to the training features pickle file.

        Returns:
            Tuple of (features, labels) as numpy arrays.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the loaded data format is invalid.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Training features file not found: {filepath}")

        try:
            data = joblib.load(filepath)
            if isinstance(data, tuple) and len(data) == 2:
                self.X_train, self.y_train = data
            else:
                msg = f"Expected tuple of (features, labels), got {type(data)}"
                raise ValueError(msg)

            if self.verbose:
                n_classes = len(np.unique(self.y_train))
                logger.info(
                    f"Loaded training features: {self.X_train.shape} "
                    f"with {n_classes} classes"
                )
            return self.X_train, self.y_train
        except Exception as e:
            logger.error(f"Error loading training features: {e}")
            raise

    def load_test_features(
        self, filepath: str
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """Load pre-extracted test features.

        Args:
            filepath: Path to the test features pickle file.

        Returns:
            Tuple of (features, image_names).
            image_names is None if not included.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Test features file not found: {filepath}")

        try:
            data = joblib.load(filepath)
            if isinstance(data, tuple) and len(data) == 2:
                self.X_test, self.test_image_names = data
            elif isinstance(data, np.ndarray):
                self.X_test = data
                self.test_image_names = None
            else:
                raise ValueError(f"Invalid test data format: {type(data)}")

            if self.verbose:
                logger.info(f"Loaded test features: {self.X_test.shape}")
            return self.X_test, self.test_image_names
        except Exception as e:
            logger.error(f"Error loading test features: {e}")
            raise

    def get_class_distribution(self) -> dict:
        """Get distribution of classes in training data.

        Returns:
            Dictionary mapping class labels to sample counts.

        Raises:
            RuntimeError: If training data not yet loaded.
        """
        if self.y_train is None:
            msg = "Training labels not loaded. Call load_train_features first."
            raise RuntimeError(msg)

        unique, counts = np.unique(self.y_train, return_counts=True)
        return dict(zip(unique, counts))

    def get_stats(self) -> dict:
        """Get comprehensive statistics about loaded datasets.

        Returns:
            Dictionary containing dataset statistics.
        """
        stats = {}

        if self.X_train is not None:
            stats["train_samples"] = len(self.X_train)
            stats["train_features_dim"] = self.X_train.shape[1]
            stats["train_classes"] = len(np.unique(self.y_train))
            stats["train_min_value"] = float(self.X_train.min())
            stats["train_max_value"] = float(self.X_train.max())

        if self.X_test is not None:
            stats["test_samples"] = len(self.X_test)
            stats["test_features_dim"] = self.X_test.shape[1]

        return stats

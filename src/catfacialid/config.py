"""Configuration management for Cat Facial ID System.

This module provides centralized configuration management including
hyperparameters, model settings, and path configurations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessingConfig:
    """Configuration for feature preprocessing."""

    pca_variance_threshold: float = 0.95
    lda_max_components: Optional[int] = None
    ica_n_components: int = 200
    ica_max_iterations: int = 200
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Configuration for model hyperparameters."""

    num_classes: int = 500
    feature_dimension: int = 2048
    top_k_predictions: int = 3
    random_seed: int = 42


@dataclass
class DataConfig:
    """Configuration for data paths and loading."""

    train_features_path: Optional[str] = None
    test_features_path: Optional[str] = None
    output_dir: str = "./outputs"
    batch_size: int = 32


@dataclass
class SystemConfig:
    """Overall system configuration."""

    preprocessing: PreprocessingConfig
    model: ModelConfig
    data: DataConfig
    use_cuda: bool = False
    verbose: bool = True

    @classmethod
    def default(cls) -> "SystemConfig":
        """Create default configuration."""
        return cls(
            preprocessing=PreprocessingConfig(),
            model=ModelConfig(),
            data=DataConfig(),
        )

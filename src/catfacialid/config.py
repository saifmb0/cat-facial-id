"""Configuration management for Cat Facial ID System.

This module provides centralized configuration management using Pydantic
for validation, environment variable parsing, and type safety.

Environment Variables:
    CATFACIALID_PCA_VARIANCE: PCA variance threshold (default: 0.95)
    CATFACIALID_LDA_COMPONENTS: LDA max components (optional)
    CATFACIALID_ICA_COMPONENTS: ICA components (default: 200)
    CATFACIALID_ICA_ITERATIONS: ICA max iterations (default: 200)
    CATFACIALID_RANDOM_SEED: Random seed for reproducibility (default: 42)
    CATFACIALID_NUM_CLASSES: Number of classes (default: 500)
    CATFACIALID_FEATURE_DIM: Feature dimension (default: 2048)
    CATFACIALID_TOP_K: Top-k predictions (default: 3)
    CATFACIALID_TRAIN_PATH: Path to training features
    CATFACIALID_TEST_PATH: Path to test features
    CATFACIALID_OUTPUT_DIR: Output directory (default: ./outputs)
    CATFACIALID_BATCH_SIZE: Batch size (default: 32)
    CATFACIALID_USE_CUDA: Enable CUDA (default: false)
    CATFACIALID_VERBOSE: Enable verbose output (default: true)
    CATFACIALID_LOG_LEVEL: Logging level (default: INFO)
    CATFACIALID_LOG_FORMAT: Log format - json or standard (default: standard)
"""

from pathlib import Path
from typing import Any, Dict, Literal, Optional

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback to dataclasses if Pydantic not installed
    from dataclasses import dataclass, field


if PYDANTIC_AVAILABLE:

    class PreprocessingConfig(BaseModel):
        """Configuration for feature preprocessing.

        Attributes:
            pca_variance_threshold: Variance threshold for PCA (0.0 to 1.0)
            lda_max_components: Maximum LDA components (None for automatic)
            ica_n_components: Number of ICA components
            ica_max_iterations: Maximum ICA iterations
            random_seed: Random seed for reproducibility
        """

        pca_variance_threshold: float = Field(
            default=0.95,
            ge=0.0,
            le=1.0,
            description="Variance threshold for PCA dimensionality reduction",
        )
        lda_max_components: Optional[int] = Field(
            default=None,
            ge=1,
            description="Maximum number of LDA components",
        )
        ica_n_components: int = Field(
            default=200,
            ge=1,
            description="Number of ICA components",
        )
        ica_max_iterations: int = Field(
            default=200,
            ge=1,
            description="Maximum ICA iterations",
        )
        random_seed: int = Field(
            default=42,
            ge=0,
            description="Random seed for reproducibility",
        )

        model_config = {"frozen": True, "extra": "forbid"}

    class ModelConfig(BaseModel):
        """Configuration for model hyperparameters.

        Attributes:
            num_classes: Number of unique cat classes
            feature_dimension: Input feature dimension
            top_k_predictions: Number of top predictions to return
            random_seed: Random seed for reproducibility
        """

        num_classes: int = Field(
            default=500,
            ge=1,
            description="Number of unique cat classes",
        )
        feature_dimension: int = Field(
            default=2048,
            ge=1,
            description="Input feature dimension",
        )
        top_k_predictions: int = Field(
            default=3,
            ge=1,
            le=100,
            description="Number of top predictions to return",
        )
        random_seed: int = Field(
            default=42,
            ge=0,
            description="Random seed for reproducibility",
        )

        model_config = {"frozen": True, "extra": "forbid"}

        @field_validator("top_k_predictions")
        @classmethod
        def validate_top_k(cls, v: int, info: Any) -> int:
            """Ensure top_k doesn't exceed num_classes."""
            # Note: Cross-field validation handled at SystemConfig level
            return v

    class DataConfig(BaseModel):
        """Configuration for data paths and loading.

        Attributes:
            train_features_path: Path to training features file
            test_features_path: Path to test features file
            output_dir: Directory for output files
            batch_size: Batch size for processing
        """

        train_features_path: Optional[Path] = Field(
            default=None,
            description="Path to training features file",
        )
        test_features_path: Optional[Path] = Field(
            default=None,
            description="Path to test features file",
        )
        output_dir: Path = Field(
            default=Path("./outputs"),
            description="Directory for output files",
        )
        batch_size: int = Field(
            default=32,
            ge=1,
            le=10000,
            description="Batch size for processing",
        )

        model_config = {"frozen": True, "extra": "forbid"}

        @field_validator("train_features_path", "test_features_path", mode="before")
        @classmethod
        def convert_str_to_path(cls, v: Optional[str]) -> Optional[Path]:
            """Convert string paths to Path objects."""
            if v is None or v == "":
                return None
            return Path(v) if isinstance(v, str) else v

    class LoggingConfig(BaseModel):
        """Configuration for logging.

        Attributes:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format: Log output format (json or standard)
            log_file: Optional log file path
        """

        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
            default="INFO",
            description="Logging level",
        )
        format: Literal["json", "standard"] = Field(
            default="standard",
            description="Log output format",
        )
        log_file: Optional[Path] = Field(
            default=None,
            description="Optional log file path",
        )

        model_config = {"frozen": True, "extra": "forbid"}

    class SystemConfig(BaseSettings):
        """Overall system configuration with environment variable support.

        This class supports loading configuration from environment variables
        with the prefix CATFACIALID_.

        Example:
            ```python
            # Load from environment
            config = SystemConfig()

            # Override specific values
            config = SystemConfig(use_cuda=True, verbose=False)

            # Load from .env file (if python-dotenv installed)
            config = SystemConfig(_env_file=".env")
            ```
        """

        model_config = SettingsConfigDict(
            env_prefix="CATFACIALID_",
            env_nested_delimiter="__",
            case_sensitive=False,
            extra="ignore",
        )

        # Nested configurations (loaded from env with __ delimiter)
        preprocessing: PreprocessingConfig = Field(
            default_factory=PreprocessingConfig
        )
        model: ModelConfig = Field(default_factory=ModelConfig)
        data: DataConfig = Field(default_factory=DataConfig)
        logging: LoggingConfig = Field(default_factory=LoggingConfig)

        # Top-level settings
        use_cuda: bool = Field(
            default=False,
            description="Enable CUDA acceleration",
        )
        verbose: bool = Field(
            default=True,
            description="Enable verbose output",
        )

        @model_validator(mode="after")
        def validate_cross_field(self) -> "SystemConfig":
            """Validate cross-field constraints."""
            if self.model.top_k_predictions > self.model.num_classes:
                raise ValueError(
                    f"top_k_predictions ({self.model.top_k_predictions}) "
                    f"cannot exceed num_classes ({self.model.num_classes})"
                )
            return self

        @classmethod
        def default(cls) -> "SystemConfig":
            """Create default configuration.

            Returns:
                SystemConfig with all default values.
            """
            return cls()

        @classmethod
        def from_yaml(cls, path: Path) -> "SystemConfig":
            """Load configuration from YAML file.

            Args:
                path: Path to YAML configuration file.

            Returns:
                SystemConfig loaded from file.

            Raises:
                ImportError: If PyYAML is not installed.
                FileNotFoundError: If config file doesn't exist.
            """
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "PyYAML required for YAML config. "
                    "Install with: pip install pyyaml"
                )

            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")

            with open(path) as f:
                data = yaml.safe_load(f)

            return cls(**data)

        def to_dict(self) -> Dict[str, Any]:
            """Convert configuration to dictionary.

            Returns:
                Dictionary representation of configuration.
            """
            return self.model_dump()

        def to_yaml(self, path: Path) -> None:
            """Save configuration to YAML file.

            Args:
                path: Path to save YAML configuration.

            Raises:
                ImportError: If PyYAML is not installed.
            """
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "PyYAML required for YAML config. "
                    "Install with: pip install pyyaml"
                )

            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

else:
    # Fallback implementation using dataclasses
    # This ensures the package works without Pydantic installed

    @dataclass
    class PreprocessingConfig:  # type: ignore[no-redef]
        """Configuration for feature preprocessing."""

        pca_variance_threshold: float = 0.95
        lda_max_components: Optional[int] = None
        ica_n_components: int = 200
        ica_max_iterations: int = 200
        random_seed: int = 42

    @dataclass
    class ModelConfig:  # type: ignore[no-redef]
        """Configuration for model hyperparameters."""

        num_classes: int = 500
        feature_dimension: int = 2048
        top_k_predictions: int = 3
        random_seed: int = 42

    @dataclass
    class DataConfig:  # type: ignore[no-redef]
        """Configuration for data paths and loading."""

        train_features_path: Optional[str] = None
        test_features_path: Optional[str] = None
        output_dir: str = "./outputs"
        batch_size: int = 32

    @dataclass
    class LoggingConfig:  # type: ignore[no-redef]
        """Configuration for logging."""

        level: str = "INFO"
        format: str = "standard"
        log_file: Optional[str] = None

    @dataclass
    class SystemConfig:  # type: ignore[no-redef]
        """Overall system configuration."""

        preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
        model: ModelConfig = field(default_factory=ModelConfig)
        data: DataConfig = field(default_factory=DataConfig)
        logging: LoggingConfig = field(default_factory=LoggingConfig)
        use_cuda: bool = False
        verbose: bool = True

        @classmethod
        def default(cls) -> "SystemConfig":
            """Create default configuration."""
            return cls()


# Convenience function for quick access
def get_config(**overrides: Any) -> SystemConfig:
    """Get system configuration with optional overrides.

    Args:
        **overrides: Configuration values to override.

    Returns:
        SystemConfig instance.

    Example:
        ```python
        config = get_config(use_cuda=True, verbose=False)
        ```
    """
    return SystemConfig(**overrides)

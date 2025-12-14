"""Custom exception hierarchy for Cat Facial ID system.

This module defines domain-specific exceptions that provide clear,
actionable error messages and enable precise error handling.

Example:
    try:
        engine.predict(features)
    except ModelNotFittedError:
        engine.fit(train_features, train_labels)
        engine.predict(features)
"""

from typing import Any, Optional


class CatFacialIDError(Exception):
    """Base exception for all Cat Facial ID errors.

    All custom exceptions in this package inherit from this class,
    allowing callers to catch all package-specific errors with a
    single except clause.
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            details: Optional dictionary with additional context.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation with details if present."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ============================================================================
# Data & I/O Exceptions
# ============================================================================


class DataLoadError(CatFacialIDError):
    """Raised when data cannot be loaded from disk.

    This includes file not found, permission errors, and corrupt files.
    """

    pass


class DataValidationError(CatFacialIDError):
    """Raised when input data fails validation checks.

    Examples include incorrect dtypes, missing columns, or out-of-range values.
    """

    pass


class InvalidDataFormatError(CatFacialIDError):
    """Raised when data format doesn't match expected schema.

    For example, when a pickle file contains unexpected structure.
    """

    pass


# ============================================================================
# Model & Inference Exceptions
# ============================================================================


class ModelNotFittedError(CatFacialIDError):
    """Raised when attempting to use an unfitted model.

    Call fit() or build_index() before calling predict().
    """

    def __init__(
        self, model_name: str = "Model", details: Optional[dict[str, Any]] = None
    ) -> None:
        message = (
            f"{model_name} has not been fitted. Call fit() or build_index() first."
        )
        super().__init__(message, details)


class DimensionMismatchError(CatFacialIDError):
    """Raised when feature dimensions don't match expected dimensions.

    This typically occurs when test features have different dimensionality
    than training features.
    """

    def __init__(
        self,
        expected: int,
        actual: int,
        context: str = "features",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        message = f"Dimension mismatch for {context}: expected {expected}, got {actual}"
        super().__init__(message, details or {"expected": expected, "actual": actual})
        self.expected = expected
        self.actual = actual


class InferenceError(CatFacialIDError):
    """Raised when inference fails for any reason.

    This is a general exception for prediction-time failures.
    """

    pass


# ============================================================================
# Configuration Exceptions
# ============================================================================


class ConfigurationError(CatFacialIDError):
    """Raised when configuration is invalid or inconsistent.

    Examples include invalid hyperparameter values or conflicting settings.
    """

    pass


class EnvironmentError(CatFacialIDError):
    """Raised when required environment variables are missing or invalid."""

    def __init__(self, var_name: str, details: Optional[dict[str, Any]] = None) -> None:
        message = f"Required environment variable '{var_name}' is not set or invalid"
        super().__init__(message, details)
        self.var_name = var_name


# ============================================================================
# Feature Processing Exceptions
# ============================================================================


class FeatureExtractionError(CatFacialIDError):
    """Raised when feature extraction fails.

    This may occur due to invalid input images or processing errors.
    """

    pass


class TransformNotFittedError(CatFacialIDError):
    """Raised when a transformer is used before fitting.

    Similar to ModelNotFittedError but for preprocessing transforms.
    """

    def __init__(
        self,
        transform_name: str = "Transform",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        message = f"{transform_name} has not been fitted. Call fit() first."
        super().__init__(message, details)


# ============================================================================
# Index Exceptions
# ============================================================================


class IndexBuildError(CatFacialIDError):
    """Raised when FAISS index construction fails."""

    pass


class IndexSearchError(CatFacialIDError):
    """Raised when FAISS index search fails."""

    pass


__all__ = [
    "CatFacialIDError",
    "DataLoadError",
    "DataValidationError",
    "InvalidDataFormatError",
    "ModelNotFittedError",
    "DimensionMismatchError",
    "InferenceError",
    "ConfigurationError",
    "EnvironmentError",
    "FeatureExtractionError",
    "TransformNotFittedError",
    "IndexBuildError",
    "IndexSearchError",
]

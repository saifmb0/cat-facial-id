"""Data validation schemas using Pandera.

This module provides schema definitions for validating input data
to prevent silent failures in production pipelines.

Example:
    from catfacialid.validation import validate_features, FeatureSchema

    # Validate numpy array as DataFrame
    df = pd.DataFrame(features, columns=[f"feat_{i}" for i in range(features.shape[1])])
    validated_df = FeatureSchema.validate(df)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


try:
    import pandas as pd
    import pandera as pa
    from pandera import Check, Column, DataFrameSchema
    from pandera.typing import DataFrame, Series

    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    logger.warning("Pandera not installed. Data validation will be limited.")


if PANDERA_AVAILABLE:
    # Schema for feature matrices
    FeatureSchema = DataFrameSchema(
        columns={
            # Dynamic columns will be validated by checks below
        },
        checks=[
            # All values should be finite (no NaN or Inf)
            Check(
                lambda df: df.notna().all().all(), error="Features contain NaN values"
            ),
            Check(
                lambda df: np.isfinite(df.values).all(),
                error="Features contain infinite values",
            ),
        ],
        coerce=True,
        strict=False,  # Allow extra columns
    )

    # Schema for prediction output
    PredictionSchema = DataFrameSchema(
        columns={
            "image_name": Column(str, nullable=False),
            "prediction_1": Column(int, nullable=False),
        },
        checks=[
            Check(lambda df: df["image_name"].str.len() > 0, error="Empty image names"),
        ],
        coerce=True,
        strict=False,
    )

    # Schema for training labels
    LabelSchema = pa.SeriesSchema(
        int,
        checks=[
            Check(lambda s: s.min() >= 0, error="Labels must be non-negative"),
        ],
        nullable=False,
        coerce=True,
    )


def validate_feature_array(
    features: np.ndarray,
    expected_dim: Optional[int] = None,
    name: str = "features",
) -> Tuple[bool, List[str]]:
    """Validate a numpy feature array.

    Args:
        features: Feature matrix to validate.
        expected_dim: Expected feature dimension (None to skip check).
        name: Name for error messages.

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors: List[str] = []

    # Check array type
    if not isinstance(features, np.ndarray):
        errors.append(f"{name}: Expected numpy array, got {type(features)}")
        return False, errors

    # Check dimensions
    if features.ndim != 2:
        errors.append(f"{name}: Expected 2D array, got {features.ndim}D")
        return False, errors

    # Check shape
    if features.shape[0] == 0:
        errors.append(f"{name}: Empty array (0 samples)")

    if expected_dim is not None and features.shape[1] != expected_dim:
        errors.append(
            f"{name}: Expected {expected_dim} features, got {features.shape[1]}"
        )

    # Check for NaN
    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        errors.append(f"{name}: Contains {nan_count} NaN values")

    # Check for Inf
    inf_count = np.isinf(features).sum()
    if inf_count > 0:
        errors.append(f"{name}: Contains {inf_count} infinite values")

    # Check dtype
    if not np.issubdtype(features.dtype, np.floating):
        logger.warning(f"{name}: Non-float dtype {features.dtype}, may need conversion")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_labels(
    labels: np.ndarray,
    expected_count: Optional[int] = None,
    name: str = "labels",
) -> Tuple[bool, List[str]]:
    """Validate a numpy label array.

    Args:
        labels: Label array to validate.
        expected_count: Expected number of labels (None to skip check).
        name: Name for error messages.

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors: List[str] = []

    # Check array type
    if not isinstance(labels, np.ndarray):
        errors.append(f"{name}: Expected numpy array, got {type(labels)}")
        return False, errors

    # Check dimensions
    if labels.ndim != 1:
        errors.append(f"{name}: Expected 1D array, got {labels.ndim}D")

    # Check count
    if expected_count is not None and len(labels) != expected_count:
        errors.append(f"{name}: Expected {expected_count} labels, got {len(labels)}")

    # Check for negative labels
    if (labels < 0).any():
        errors.append(f"{name}: Contains negative label values")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[bool, Dict[str, List[str]]]:
    """Validate complete training dataset.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        Tuple of (is_valid, dict of errors by component).
    """
    all_errors: Dict[str, List[str]] = {}

    # Validate features
    features_valid, feature_errors = validate_feature_array(X_train, name="X_train")
    if not features_valid:
        all_errors["features"] = feature_errors

    # Validate labels
    labels_valid, label_errors = validate_labels(
        y_train, expected_count=X_train.shape[0], name="y_train"
    )
    if not labels_valid:
        all_errors["labels"] = label_errors

    # Cross-validation
    if X_train.shape[0] != len(y_train):
        all_errors.setdefault("consistency", []).append(
            f"Feature count ({X_train.shape[0]}) != label count ({len(y_train)})"
        )

    is_valid = len(all_errors) == 0

    if is_valid:
        logger.info(
            f"Training data validated: {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features, {len(np.unique(y_train))} classes"
        )
    else:
        logger.error(f"Training data validation failed: {all_errors}")

    return is_valid, all_errors


__all__ = [
    "validate_feature_array",
    "validate_labels",
    "validate_training_data",
    "PANDERA_AVAILABLE",
]

if PANDERA_AVAILABLE:
    __all__.extend(["FeatureSchema", "PredictionSchema", "LabelSchema"])

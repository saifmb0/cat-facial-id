"""Feature preprocessing and fusion utilities.

This module provides classes for dimensionality reduction (PCA, LDA, ICA)
and feature fusion techniques used in the cat facial identification pipeline.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, normalize

logger = logging.getLogger(__name__)


class DimensionalityReducer:
    """Handles dimensionality reduction using multiple techniques.

    Supports PCA, LDA, and ICA for feature space reduction while
    preserving discriminative information.
    """

    def __init__(self, seed: int = 42, verbose: bool = True):
        """Initialize DimensionalityReducer.

        Args:
            seed: Random seed for reproducibility.
            verbose: Enable verbose logging.
        """
        self.seed = seed
        self.verbose = verbose
        self.pca = None
        self.lda = None
        self.ica = None
        self.pca_scaler = None
        self.ica_scaler = None

    def apply_pca(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        variance_threshold: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply PCA for variance-based dimensionality reduction.

        Args:
            X_train: Training feature matrix.
            X_test: Test feature matrix.
            variance_threshold: Fraction of variance to retain (0-1).

        Returns:
            Tuple of (X_train_pca, X_test_pca).
        """
        self.pca = PCA(
            n_components=variance_threshold,
            svd_solver="full",
            random_state=self.seed,
        )
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)

        if self.verbose:
            logger.info(
                f"PCA: {X_train.shape[1]} -> {X_train_pca.shape[1]} "
                f"(retained {self.pca.explained_variance_ratio_.sum():.2%})"
            )

        return X_train_pca, X_test_pca

    def apply_lda(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        n_components: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply LDA for class-discriminant dimensionality reduction.

        Args:
            X_train: Training feature matrix.
            X_test: Test feature matrix.
            y_train: Training labels.
            n_components: Number of LDA components.
                If None, uses min(features, classes-1).

        Returns:
            Tuple of (X_train_lda, X_test_lda).
        """
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))

        if n_components is None:
            n_components = min(n_features, n_classes - 1)

        self.lda = LDA(n_components=n_components)
        X_train_lda = self.lda.fit_transform(X_train, y_train)
        X_test_lda = self.lda.transform(X_test)

        if self.verbose:
            logger.info(f"LDA: {n_features} -> {n_components} components")

        return X_train_lda, X_test_lda

    def apply_ica(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        n_components: int = 200,
        max_iterations: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply FastICA for independent component analysis.

        Args:
            X_train: Training feature matrix.
            X_test: Test feature matrix.
            n_components: Number of ICA components.
            max_iterations: Maximum iterations for ICA convergence.

        Returns:
            Tuple of (X_train_ica, X_test_ica).
        """
        n_components = min(n_components, X_train.shape[0])

        self.ica = FastICA(
            n_components=n_components,
            max_iter=max_iterations,
            random_state=self.seed,
        )
        X_train_ica = self.ica.fit_transform(X_train)
        X_test_ica = self.ica.transform(X_test)

        if self.verbose:
            logger.info(f"ICA: {X_train.shape[1]} -> {n_components} components")

        return X_train_ica, X_test_ica


class FeatureExtractor:
    """Coordinate preprocessing pipeline with scaling.

    Handles proper scaling of features before applying dimensionality
    reduction techniques to avoid information leakage.
    """

    def __init__(self, seed: int = 42, verbose: bool = True):
        """Initialize FeatureExtractor.

        Args:
            seed: Random seed for reproducibility.
            verbose: Enable verbose logging.
        """
        self.seed = seed
        self.verbose = verbose
        self.scaler = StandardScaler()

    def scale_features(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standardize features using training data statistics.

        Args:
            X_train: Training feature matrix.
            X_test: Test feature matrix.

        Returns:
            Tuple of (X_train_scaled, X_test_scaled).
        """
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if self.verbose:
            logger.info("Features standardized using StandardScaler")

        return X_train_scaled, X_test_scaled


class FeatureFuser:
    """Fuse features from multiple sources into unified representation.

    Combines PCA, LDA, and ICA features through concatenation and
    normalization to create robust fused feature vectors.
    """

    def __init__(self, verbose: bool = True):
        """Initialize FeatureFuser.

        Args:
            verbose: Enable verbose logging.
        """
        self.verbose = verbose
        self.total_fused_dim = None

    def fuse_features(
        self,
        *feature_arrays: np.ndarray,
        normalize_output: bool = True,
    ) -> np.ndarray:
        """Concatenate and optionally normalize multiple feature arrays.

        Args:
            *feature_arrays: Variable number of feature matrices to fuse.
            normalize_output: Whether to L2-normalize the fused features.

        Returns:
            Fused feature matrix.

        Raises:
            ValueError: If feature arrays have inconsistent sample counts.
        """
        if not feature_arrays:
            raise ValueError("At least one feature array must be provided")

        n_samples = feature_arrays[0].shape[0]
        for i, arr in enumerate(feature_arrays[1:], 1):
            if arr.shape[0] != n_samples:
                raise ValueError(
                    f"Feature array {i} has {arr.shape[0]} samples, "
                    f"expected {n_samples}"
                )

        X_fused = np.hstack(feature_arrays)
        self.total_fused_dim = X_fused.shape[1]

        if normalize_output:
            X_fused = normalize(X_fused, axis=1)
            if self.verbose:
                logger.info(
                    f"Features fused and L2-normalized: dim={self.total_fused_dim}"
                )
        else:
            if self.verbose:
                logger.info(f"Features fused: dim={self.total_fused_dim}")

        return X_fused

"""Unit tests for preprocessing and feature extraction."""

import pytest
import numpy as np

from catfacialid.core.preprocessing import (
    DimensionalityReducer,
    FeatureExtractor,
    FeatureFuser,
)


class TestFeatureExtractor:
    """Test suite for FeatureExtractor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with known properties."""
        np.random.seed(42)
        X_train = np.random.randn(100, 50) * 10 + 5
        X_test = np.random.randn(30, 50) * 10 + 5
        return X_train, X_test

    def test_feature_extraction_initialization(self):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor(seed=42, verbose=False)
        assert extractor.seed == 42
        assert extractor.scaler is not None

    def test_scale_features(self, sample_data):
        """Test feature scaling."""
        X_train, X_test = sample_data
        extractor = FeatureExtractor(seed=42, verbose=False)

        X_train_scaled, X_test_scaled = extractor.scale_features(X_train, X_test)

        # Check shapes are preserved
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape

        # Check scaling properties: mean ~0, std ~1
        assert np.abs(X_train_scaled.mean()) < 1e-6
        assert np.abs(X_train_scaled.std() - 1.0) < 1e-6


class TestDimensionalityReducer:
    """Test suite for DimensionalityReducer class."""

    @pytest.fixture
    def sample_data_with_labels(self):
        """Create sample data with labels."""
        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        X_test = np.random.randn(30, 50)
        y_train = np.random.randint(0, 10, 100)
        return X_train, X_test, y_train

    def test_reducer_initialization(self):
        """Test DimensionalityReducer initialization."""
        reducer = DimensionalityReducer(seed=42, verbose=False)
        assert reducer.seed == 42
        assert reducer.pca is None
        assert reducer.lda is None
        assert reducer.ica is None

    def test_apply_pca(self, sample_data_with_labels):
        """Test PCA transformation."""
        X_train, X_test, _ = sample_data_with_labels
        reducer = DimensionalityReducer(seed=42, verbose=False)

        X_train_pca, X_test_pca = reducer.apply_pca(
            X_train, X_test, variance_threshold=0.9
        )

        # Check dimensions are reduced
        assert X_train_pca.shape[1] < X_train.shape[1]
        assert X_test_pca.shape[0] == X_test.shape[0]

        # Check PCA is fitted
        assert reducer.pca is not None

    def test_apply_lda(self, sample_data_with_labels):
        """Test LDA transformation."""
        X_train, X_test, y_train = sample_data_with_labels
        reducer = DimensionalityReducer(seed=42, verbose=False)

        X_train_lda, X_test_lda = reducer.apply_lda(X_train, X_test, y_train)

        # Check dimensions
        n_classes = len(np.unique(y_train))
        assert X_train_lda.shape[1] <= n_classes - 1
        assert X_test_lda.shape[0] == X_test.shape[0]

    def test_apply_ica(self, sample_data_with_labels):
        """Test ICA transformation."""
        X_train, X_test, _ = sample_data_with_labels
        reducer = DimensionalityReducer(seed=42, verbose=False)

        X_train_ica, X_test_ica = reducer.apply_ica(X_train, X_test, n_components=20)

        # Check dimensions
        assert X_train_ica.shape[1] == 20
        assert X_test_ica.shape[1] == 20

    def test_ica_n_components_capped(self, sample_data_with_labels):
        """Test that ICA components are capped at sample count."""
        X_train, X_test, _ = sample_data_with_labels
        reducer = DimensionalityReducer(seed=42, verbose=False)

        # Request more components than samples
        X_train_ica, X_test_ica = reducer.apply_ica(X_train, X_test, n_components=500)

        # Components should be capped at n_samples
        assert X_train_ica.shape[1] == X_train.shape[0]


class TestFeatureFuser:
    """Test suite for FeatureFuser class."""

    @pytest.fixture
    def sample_features(self):
        """Create multiple feature arrays for fusion."""
        np.random.seed(42)
        features = [
            np.random.randn(50, 20),
            np.random.randn(50, 15),
            np.random.randn(50, 25),
        ]
        return features

    def test_fuser_initialization(self):
        """Test FeatureFuser initialization."""
        fuser = FeatureFuser(verbose=False)
        assert fuser.total_fused_dim is None

    def test_fuse_features(self, sample_features):
        """Test feature fusion."""
        fuser = FeatureFuser(verbose=False)
        fused = fuser.fuse_features(*sample_features, normalize_output=False)

        # Check dimension is sum of all feature dims
        expected_dim = sum(f.shape[1] for f in sample_features)
        assert fused.shape[1] == expected_dim
        assert fused.shape[0] == sample_features[0].shape[0]

    def test_fuse_features_with_normalization(self, sample_features):
        """Test feature fusion with L2 normalization."""
        fuser = FeatureFuser(verbose=False)
        fused = fuser.fuse_features(*sample_features, normalize_output=True)

        # Check L2 norms are ~1
        norms = np.linalg.norm(fused, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(norms)))

    def test_fuse_single_feature(self):
        """Test fusion with single feature array."""
        fuser = FeatureFuser(verbose=False)
        X = np.random.randn(50, 20)
        fused = fuser.fuse_features(X)

        assert fused.shape == X.shape

    def test_fuse_mismatched_samples(self):
        """Test error handling for mismatched sample counts."""
        fuser = FeatureFuser(verbose=False)
        X1 = np.random.randn(50, 20)
        X2 = np.random.randn(30, 15)  # Different sample count

        with pytest.raises(ValueError):
            fuser.fuse_features(X1, X2)

    def test_fuse_no_features(self):
        """Test error handling when no features provided."""
        fuser = FeatureFuser(verbose=False)
        with pytest.raises(ValueError):
            fuser.fuse_features()

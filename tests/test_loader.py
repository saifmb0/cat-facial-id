"""Unit tests for data loading utilities."""

import tempfile
from pathlib import Path

import pytest
import numpy as np
import joblib

from catfacialid.data import DataLoader


class TestDataLoader:
    """Test suite for DataLoader class."""

    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data for testing."""
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 10, 100)
        return X, y

    @pytest.fixture
    def sample_test_data(self):
        """Create sample test data for testing."""
        X = np.random.randn(50, 50)
        names = [f"test_{i}.jpg" for i in range(50)]
        return X, names

    @pytest.fixture
    def temp_pkl_files(self, sample_train_data, sample_test_data):
        """Create temporary pickle files with sample data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = Path(tmpdir) / "train.pkl"
            test_path = Path(tmpdir) / "test.pkl"

            joblib.dump(sample_train_data, train_path)
            joblib.dump(sample_test_data, test_path)

            yield train_path, test_path

    def test_loader_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader(verbose=False)
        assert loader.X_train is None
        assert loader.y_train is None
        assert loader.X_test is None
        assert loader.verbose is False

    def test_load_train_features(self, temp_pkl_files, sample_train_data):
        """Test loading training features."""
        train_path, _ = temp_pkl_files
        loader = DataLoader(verbose=False)

        X, y = loader.load_train_features(str(train_path))
        np.testing.assert_array_equal(X, sample_train_data[0])
        np.testing.assert_array_equal(y, sample_train_data[1])

    def test_load_test_features(self, temp_pkl_files, sample_test_data):
        """Test loading test features."""
        _, test_path = temp_pkl_files
        loader = DataLoader(verbose=False)

        X, names = loader.load_test_features(str(test_path))
        np.testing.assert_array_equal(X, sample_test_data[0])
        assert names == sample_test_data[1]

    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        loader = DataLoader(verbose=False)
        with pytest.raises(FileNotFoundError):
            loader.load_train_features("/nonexistent/path.pkl")

    def test_get_class_distribution(self, temp_pkl_files):
        """Test class distribution calculation."""
        train_path, _ = temp_pkl_files
        loader = DataLoader(verbose=False)
        loader.load_train_features(str(train_path))

        dist = loader.get_class_distribution()
        assert isinstance(dist, dict)
        assert len(dist) > 0
        assert sum(dist.values()) == 100

    def test_get_stats(self, temp_pkl_files):
        """Test statistics calculation."""
        train_path, test_path = temp_pkl_files
        loader = DataLoader(verbose=False)
        loader.load_train_features(str(train_path))
        loader.load_test_features(str(test_path))

        stats = loader.get_stats()
        assert "train_samples" in stats
        assert "train_features_dim" in stats
        assert "train_classes" in stats
        assert "test_samples" in stats
        assert stats["train_samples"] == 100
        assert stats["test_samples"] == 50

    def test_stats_without_data(self):
        """Test stats retrieval without loading data."""
        loader = DataLoader(verbose=False)
        stats = loader.get_stats()
        assert stats == {}

"""Unit tests for inference engine."""

import numpy as np
import pytest

from catfacialid.core.inference import FAISSIndex, PredictionEngine


class TestFAISSIndex:
    """Test suite for FAISSIndex class."""

    @pytest.fixture
    def sample_vectors_and_labels(self):
        """Create sample vectors and labels."""
        np.random.seed(42)
        vectors = np.random.randn(100, 50).astype(np.float32)
        labels = np.random.randint(0, 10, 100)
        return vectors, labels

    def test_faiss_index_initialization(self):
        """Test FAISS index initialization."""
        index = FAISSIndex(dimension=50, verbose=False)
        assert index.dimension == 50
        assert index.index is not None
        assert index.labels is None

    def test_add_vectors(self, sample_vectors_and_labels):
        """Test adding vectors to index."""
        vectors, labels = sample_vectors_and_labels
        index = FAISSIndex(dimension=50, verbose=False)
        index.add_vectors(vectors, labels)

        assert index.labels is not None
        assert len(index.labels) == 100
        assert index.get_index_size() == 100

    def test_add_vectors_dimension_mismatch(self, sample_vectors_and_labels):
        """Test error for dimension mismatch."""
        vectors, labels = sample_vectors_and_labels
        index = FAISSIndex(dimension=30, verbose=False)  # Wrong dimension

        with pytest.raises(ValueError):
            index.add_vectors(vectors, labels)

    def test_search_single_vector(self, sample_vectors_and_labels):
        """Test searching with single vector."""
        vectors, labels = sample_vectors_and_labels
        index = FAISSIndex(dimension=50, verbose=False)
        index.add_vectors(vectors, labels)

        query = vectors[0]
        distances, results = index.search(query, k=3)

        assert distances.shape == (1, 3)
        assert len(results) == 1
        assert len(results[0]) == 3

    def test_search_batch_vectors(self, sample_vectors_and_labels):
        """Test searching with batch of vectors."""
        vectors, labels = sample_vectors_and_labels
        index = FAISSIndex(dimension=50, verbose=False)
        index.add_vectors(vectors, labels)

        queries = vectors[:10]
        distances, results = index.search(queries, k=5)

        assert distances.shape == (10, 5)
        assert len(results) == 10

    def test_search_returns_labels(self, sample_vectors_and_labels):
        """Test that search returns original labels."""
        vectors, labels = sample_vectors_and_labels
        index = FAISSIndex(dimension=50, verbose=False)
        index.add_vectors(vectors, labels)

        query = vectors[0:1]
        _, results = index.search(query, k=3)

        # Results should be valid label indices
        for pred_list in results:
            for pred in pred_list:
                assert pred in labels

    def test_get_index_size(self, sample_vectors_and_labels):
        """Test getting index size."""
        vectors, labels = sample_vectors_and_labels
        index = FAISSIndex(dimension=50, verbose=False)

        assert index.get_index_size() == 0

        index.add_vectors(vectors, labels)
        assert index.get_index_size() == 100


class TestPredictionEngine:
    """Test suite for PredictionEngine class."""

    @pytest.fixture
    def engine_and_data(self):
        """Create engine with sample data."""
        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        y_train = np.random.randint(0, 10, 100)
        X_test = np.random.randn(20, 50)

        engine = PredictionEngine(top_k=3, verbose=False)
        return engine, X_train, y_train, X_test

    def test_engine_initialization(self):
        """Test PredictionEngine initialization."""
        engine = PredictionEngine(top_k=5, verbose=False)
        assert engine.top_k == 5
        assert engine.faiss_index is None

    def test_build_index(self, engine_and_data):
        """Test index building."""
        engine, X_train, y_train, _ = engine_and_data
        engine.build_index(X_train, y_train)

        assert engine.faiss_index is not None
        assert engine.faiss_index.get_index_size() == 100

    def test_build_index_mismatch(self, engine_and_data):
        """Test error for mismatched data."""
        engine, X_train, _, _ = engine_and_data
        y_train_wrong = np.random.randint(0, 10, 50)  # Wrong size

        with pytest.raises(ValueError):
            engine.build_index(X_train, y_train_wrong)

    def test_predict(self, engine_and_data):
        """Test prediction generation."""
        engine, X_train, y_train, X_test = engine_and_data
        engine.build_index(X_train, y_train)

        results = engine.predict(X_test)

        assert len(results) == 20
        for img_name, predictions in results:
            assert len(predictions) == 3
            assert img_name.startswith("test_")

    def test_predict_with_image_names(self, engine_and_data):
        """Test prediction with provided image names."""
        engine, X_train, y_train, X_test = engine_and_data
        engine.build_index(X_train, y_train)

        image_names = [f"cat_{i}.jpg" for i in range(20)]
        results = engine.predict(X_test, image_names)

        for (img_name, _), provided_name in zip(results, image_names):
            assert img_name == provided_name

    def test_predict_without_index(self, engine_and_data):
        """Test error when predicting without built index."""
        engine, _, _, X_test = engine_and_data

        with pytest.raises(RuntimeError):
            engine.predict(X_test)

    def test_predict_single(self, engine_and_data):
        """Test single sample prediction."""
        engine, X_train, y_train, X_test = engine_and_data
        engine.build_index(X_train, y_train)

        predictions = engine.predict_single(X_test[0])

        assert len(predictions) == 3
        assert all(p in y_train for p in predictions)

    def test_predict_single_without_index(self, engine_and_data):
        """Test error for single prediction without index."""
        engine, _, _, X_test = engine_and_data

        with pytest.raises(RuntimeError):
            engine.predict_single(X_test[0])

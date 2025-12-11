"""Inference engine using FAISS for efficient similarity search.

This module provides the prediction pipeline using FAISS for fast
k-nearest neighbor search on the training embeddings.
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS-based index for efficient similarity search.

    Wraps FAISS IndexFlatL2 to provide fast k-NN search on embeddings
    with associated label tracking.
    """

    def __init__(self, dimension: int, verbose: bool = True):
        """Initialize FAISS index.

        Args:
            dimension: Dimensionality of feature vectors.
            verbose: Enable verbose logging.
        """
        self.dimension = dimension
        self.verbose = verbose
        self.index = faiss.IndexFlatL2(dimension)
        self.labels = None

    def add_vectors(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        """Add vectors and associated labels to index.

        Args:
            vectors: Feature matrix of shape (n_samples, dimension).
            labels: Class labels of shape (n_samples,).

        Raises:
            ValueError: If vector dimension doesn't match index.
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )

        vectors_float32 = vectors.astype(np.float32)
        self.index.add(vectors_float32)
        self.labels = labels

        if self.verbose:
            logger.info(f"Added {len(vectors)} vectors to FAISS index")

    def search(self, query_vector: np.ndarray, k: int = 3) -> Tuple[np.ndarray, List]:
        """Search for k-nearest neighbors.

        Args:
            query_vector: Single query vector or batch of shape (batch_size, dimension).
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (distances, predicted_labels).
            - distances: L2 distances to neighbors, shape (n_queries, k)
            - predicted_labels: Corresponding labels, shape (n_queries, k)
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_vector.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )

        query_float32 = query_vector.astype(np.float32)
        distances, indices = self.index.search(query_float32, k)

        # Map indices to original labels
        predicted_labels = []
        for idx_list in indices:
            labels_for_query = [self.labels[int(idx)] for idx in idx_list]
            predicted_labels.append(labels_for_query)

        return distances, predicted_labels

    def get_index_size(self) -> int:
        """Get number of vectors in index.

        Returns:
            Number of indexed vectors.
        """
        return self.index.ntotal


class PredictionEngine:
    """End-to-end prediction pipeline with FAISS backend.

    Coordinates feature extraction, indexing, and prediction generation
    for the cat facial identification task.
    """

    def __init__(self, top_k: int = 3, verbose: bool = True):
        """Initialize PredictionEngine.

        Args:
            top_k: Number of top predictions to return.
            verbose: Enable verbose logging.
        """
        self.top_k = top_k
        self.verbose = verbose
        self.faiss_index = None

    def build_index(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Build FAISS index from training data.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
        """
        if X_train.shape[0] != len(y_train):
            raise ValueError(
                f"Training data shape {X_train.shape[0]} does not match "
                f"labels shape {len(y_train)}"
            )

        self.faiss_index = FAISSIndex(X_train.shape[1], verbose=self.verbose)
        self.faiss_index.add_vectors(X_train, y_train)

        if self.verbose:
            logger.info(f"Built FAISS index with {len(y_train)} training samples")

    def predict(
        self, X_test: np.ndarray, image_names: Optional[List[str]] = None
    ) -> List[Tuple]:
        """Generate top-k predictions for test samples.

        Args:
            X_test: Test feature matrix.
            image_names: Optional list of image filenames for results.

        Returns:
            List of tuples (image_name, [top_k_predictions]).

        Raises:
            RuntimeError: If index not yet built.
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built. Call build_index first.")

        distances, predicted_labels = self.faiss_index.search(X_test, self.top_k)

        results = []
        for i, predictions in enumerate(predicted_labels):
            img_name = image_names[i] if image_names else f"test_{i:06d}"
            results.append((img_name, predictions))

        if self.verbose:
            logger.info(f"Generated predictions for {len(results)} test samples")

        return results

    def predict_single(self, features: np.ndarray) -> List:
        """Generate predictions for a single sample.

        Args:
            features: Feature vector of shape (dimension,).

        Returns:
            Top-k predicted class labels.
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built. Call build_index first.")

        _, predictions = self.faiss_index.search(features.reshape(1, -1), self.top_k)
        return predictions[0]

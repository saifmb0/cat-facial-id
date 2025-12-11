# API Documentation

## Overview

This document provides detailed API reference for the Cat Facial Identification System. For usage examples and quick start, see the main README.

## Core Modules

### catfacialid.config

Configuration management for the entire system.

#### SystemConfig

```python
from src.catfacialid.config import SystemConfig

config = SystemConfig.default()
config.model.top_k_predictions = 5
config.preprocessing.pca_variance_threshold = 0.90
```

**Attributes:**
- `preprocessing` (PreprocessingConfig): Feature preprocessing settings
- `model` (ModelConfig): Model hyperparameters
- `data` (DataConfig): Data paths and loading settings
- `use_cuda` (bool): Enable CUDA if available
- `verbose` (bool): Enable verbose logging

**Methods:**
- `default()`: Create default configuration instance

#### PreprocessingConfig

```python
from src.catfacialid.config import PreprocessingConfig

config = PreprocessingConfig(
    pca_variance_threshold=0.95,
    ica_n_components=200,
    random_seed=42
)
```

**Attributes:**
- `pca_variance_threshold` (float): PCA variance retention (0-1), default 0.95
- `lda_max_components` (Optional[int]): Max LDA components, default None
- `ica_n_components` (int): Number of ICA components, default 200
- `ica_max_iterations` (int): ICA convergence iterations, default 200
- `random_seed` (int): Reproducibility seed, default 42

#### ModelConfig

```python
from src.catfacialid.config import ModelConfig

config = ModelConfig(
    num_classes=500,
    top_k_predictions=3,
    random_seed=42
)
```

**Attributes:**
- `num_classes` (int): Total number of cat identity classes, default 500
- `top_k_predictions` (int): Return top-k predictions, default 3
- `random_seed` (int): Reproducibility seed, default 42

#### DataConfig

```python
from src.catfacialid.config import DataConfig

config = DataConfig(
    train_features_path="/path/to/train.pkl",
    output_dir="./predictions"
)
```

**Attributes:**
- `train_features_path` (Optional[str]): Path to training features, default None
- `test_features_path` (Optional[str]): Path to test features, default None
- `output_dir` (str): Output directory for results, default "./outputs"
- `batch_size` (int): Batch size for processing, default 32

---

### catfacialid.data.DataLoader

Load and manage datasets.

```python
from src.catfacialid.data import DataLoader

loader = DataLoader(verbose=True)
X_train, y_train = loader.load_train_features("train.pkl")
X_test, names = loader.load_test_features("test.pkl")
```

#### Methods

**load_train_features(filepath: str) -> Tuple[np.ndarray, np.ndarray]**

Load pre-extracted training features and labels.

- **Args:**
  - `filepath`: Path to pickle file containing (features, labels)
- **Returns:** Tuple of (feature matrix, label vector)
- **Raises:** FileNotFoundError, ValueError

**load_test_features(filepath: str) -> Tuple[np.ndarray, Optional[List[str]]]**

Load pre-extracted test features.

- **Args:**
  - `filepath`: Path to pickle file
- **Returns:** Tuple of (feature matrix, image names)
- **Raises:** FileNotFoundError

**get_class_distribution() -> dict**

Get distribution of classes in training data.

- **Returns:** Dictionary mapping class labels to sample counts
- **Raises:** RuntimeError if training data not loaded

**get_stats() -> dict**

Get comprehensive dataset statistics.

- **Returns:** Dictionary with keys: train_samples, train_features_dim, train_classes, test_samples, etc.

---

### catfacialid.core.preprocessing

Feature extraction and transformation.

#### FeatureExtractor

```python
from src.catfacialid.core import FeatureExtractor

extractor = FeatureExtractor(seed=42, verbose=True)
X_train_scaled, X_test_scaled = extractor.scale_features(X_train, X_test)
```

**Methods:**

**scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]**

Standardize features using StandardScaler.

- **Args:**
  - `X_train`: Training features (n_samples, n_features)
  - `X_test`: Test features (n_samples, n_features)
- **Returns:** Scaled training and test features
- **Effect:** Features will have mean ~0, std ~1

#### DimensionalityReducer

```python
from src.catfacialid.core import DimensionalityReducer

reducer = DimensionalityReducer(seed=42, verbose=True)
X_train_pca, X_test_pca = reducer.apply_pca(X_train, X_test, variance_threshold=0.95)
X_train_lda, X_test_lda = reducer.apply_lda(X_train_pca, X_test_pca, y_train)
X_train_ica, X_test_ica = reducer.apply_ica(X_train, X_test, n_components=200)
```

**Methods:**

**apply_pca(X_train, X_test, variance_threshold=0.95) -> Tuple[np.ndarray, np.ndarray]**

Apply PCA for variance-based dimensionality reduction.

- **Args:**
  - `X_train`: Training features
  - `X_test`: Test features
  - `variance_threshold`: Fraction of variance to retain (0-1)
- **Returns:** Reduced training and test features

**apply_lda(X_train, X_test, y_train, n_components=None) -> Tuple[np.ndarray, np.ndarray]**

Apply LDA for class-discriminant dimensionality reduction.

- **Args:**
  - `X_train`: Training features
  - `X_test`: Test features
  - `y_train`: Training labels
  - `n_components`: Number of components (default: min(features, classes-1))
- **Returns:** LDA-transformed training and test features

**apply_ica(X_train, X_test, n_components=200, max_iterations=200) -> Tuple[np.ndarray, np.ndarray]**

Apply FastICA for independent component analysis.

- **Args:**
  - `X_train`: Training features
  - `X_test`: Test features
  - `n_components`: Number of ICA components
  - `max_iterations`: Convergence iterations
- **Returns:** ICA components for training and test data

#### FeatureFuser

```python
from src.catfacialid.core import FeatureFuser

fuser = FeatureFuser(verbose=True)
X_fused = fuser.fuse_features(X_pca, X_lda, X_ica, normalize_output=True)
```

**Methods:**

**fuse_features(*feature_arrays, normalize_output=True) -> np.ndarray**

Concatenate and normalize multiple feature sources.

- **Args:**
  - `*feature_arrays`: Variable number of feature matrices
  - `normalize_output`: Whether to L2-normalize output
- **Returns:** Fused feature matrix
- **Raises:** ValueError if feature shapes mismatch

---

### catfacialid.core.inference

Prediction engine using FAISS.

#### FAISSIndex

```python
from src.catfacialid.core import FAISSIndex

index = FAISSIndex(dimension=700, verbose=True)
index.add_vectors(X_train_fused, y_train)
distances, predictions = index.search(X_test_fused[0], k=3)
```

**Methods:**

**add_vectors(vectors: np.ndarray, labels: np.ndarray) -> None**

Add vectors and labels to FAISS index.

- **Args:**
  - `vectors`: Feature matrix (n_samples, dimension)
  - `labels`: Class labels (n_samples,)
- **Raises:** ValueError if dimension mismatch

**search(query_vector: np.ndarray, k: int = 3) -> Tuple[np.ndarray, List]**

Search for k-nearest neighbors.

- **Args:**
  - `query_vector`: Query vector or batch (batch_size, dimension)
  - `k`: Number of neighbors to return
- **Returns:** Tuple of (distances, predicted_labels)
- **Note:** Returns L2 distances from FAISS IndexFlatL2

**get_index_size() -> int**

Get number of vectors in index.

- **Returns:** Total indexed vectors

#### PredictionEngine

```python
from src.catfacialid.core import PredictionEngine

engine = PredictionEngine(top_k=3, verbose=True)
engine.build_index(X_train_fused, y_train)
results = engine.predict(X_test_fused, image_names)
single_pred = engine.predict_single(X_test_fused[0])
```

**Methods:**

**build_index(X_train: np.ndarray, y_train: np.ndarray) -> None**

Build FAISS index from training data.

- **Args:**
  - `X_train`: Training feature matrix
  - `y_train`: Training labels
- **Raises:** ValueError if shape mismatch

**predict(X_test: np.ndarray, image_names: Optional[List[str]] = None) -> List[Tuple]**

Generate top-k predictions for test samples.

- **Args:**
  - `X_test`: Test feature matrix
  - `image_names`: Optional image filenames
- **Returns:** List of (image_name, top_k_labels) tuples
- **Raises:** RuntimeError if index not built

**predict_single(features: np.ndarray) -> List**

Get predictions for single sample.

- **Args:**
  - `features`: Feature vector (dimension,)
- **Returns:** Top-k predicted labels
- **Raises:** RuntimeError if index not built

---

## Architecture Overview

```
Input Features (2048-dim ResNet50)
    |
    v
[Feature Scaling]
    |
    +---> [PCA: 2048 -> ~400 dim]
    |
    +---> [LDA: ~400 -> ~500 dim]
    |
    +---> [ICA: 2048 -> 200 dim]
    |
    v
[Feature Fusion]
    |
    v
[L2 Normalization]
    |
    v
[Fused Features: ~1100 dim]
    |
    v
[FAISS IndexFlatL2]
    |
    v
[Top-k NN Search]
    |
    v
[Predictions]
```

---

## Error Handling

All modules use exceptions for error conditions:

- `FileNotFoundError`: Missing data files
- `ValueError`: Invalid data dimensions or configurations
- `RuntimeError`: Operations on uninitialized objects (e.g., predict before build_index)

Always check module docstrings for specific exceptions raised.

---

## Performance Characteristics

- **Memory:** Linear in dataset size O(n)
- **Indexing:** O(n * d) where n=samples, d=dimensions
- **Query:** O(n * d) for exhaustive search with FAISS IndexFlatL2
- **Query latency:** ~5ms per sample on CPU

---

## Example: Complete Pipeline

```python
from src.catfacialid.data import DataLoader
from src.catfacialid.core import (
    FeatureExtractor, DimensionalityReducer, FeatureFuser, PredictionEngine
)
from src.catfacialid.config import SystemConfig

# Configuration
config = SystemConfig.default()

# Load data
loader = DataLoader(verbose=True)
X_train, y_train = loader.load_train_features("train.pkl")
X_test, img_names = loader.load_test_features("test.pkl")

# Preprocess
extractor = FeatureExtractor(seed=config.model.random_seed)
X_train_scaled, X_test_scaled = extractor.scale_features(X_train, X_test)

# Dimensionality reduction
reducer = DimensionalityReducer(seed=config.model.random_seed)
X_pca_train, X_pca_test = reducer.apply_pca(
    X_train_scaled, X_test_scaled,
    variance_threshold=config.preprocessing.pca_variance_threshold
)
X_lda_train, X_lda_test = reducer.apply_lda(X_pca_train, X_pca_test, y_train)
X_ica_train, X_ica_test = reducer.apply_ica(
    X_train_scaled, X_test_scaled,
    n_components=config.preprocessing.ica_n_components
)

# Feature fusion
fuser = FeatureFuser()
X_train_fused = fuser.fuse_features(X_pca_train, X_lda_train, X_ica_train)
X_test_fused = fuser.fuse_features(X_pca_test, X_lda_test, X_ica_test)

# Prediction
engine = PredictionEngine(top_k=config.model.top_k_predictions)
engine.build_index(X_train_fused, y_train)
predictions = engine.predict(X_test_fused, img_names)

# Output
for img_name, top_k_labels in predictions:
    print(f"{img_name}: {top_k_labels}")
```

---

For additional information, consult the main README.md or source code docstrings.

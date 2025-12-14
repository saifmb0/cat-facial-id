# Quick Start Guide

This guide will help you get up and running with Cat Facial ID in minutes.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

## Installation

### Basic Installation

```bash
pip install cat-facial-id
```

### With CLI Support

```bash
pip install cat-facial-id[cli]
```

### Full Installation (all features)

```bash
pip install cat-facial-id[all]
```

## Your First Identification

### Using Python API

```python
import numpy as np
from catfacialid.core.preprocessing import FeaturePreprocessor
from catfacialid.core.inference import FAISSInference

# Prepare your data
# train_features: (N, D) array of feature vectors
# train_labels: (N,) array of cat IDs
# test_features: (M, D) array of query features

# Preprocess features
preprocessor = FeaturePreprocessor()
X_train_processed = preprocessor.fit_transform(train_features, train_labels)
X_test_processed = preprocessor.transform(test_features)

# Build search index and predict
inference = FAISSInference(top_k=3)
inference.fit(X_train_processed, train_labels)
predictions, distances = inference.predict(X_test_processed)

# predictions contains top-3 cat IDs for each query
print(f"Predicted cat IDs: {predictions[0]}")
print(f"Distances: {distances[0]}")
```

### Using CLI

```bash
# Get system info
catfacialid info

# Run prediction
catfacialid predict \
    --features data/test_features.npy \
    --model models/trained_model.pkl \
    --output predictions.csv

# Train a new model
catfacialid train \
    --features data/train_features.npy \
    --labels data/train_labels.csv \
    --output-dir models/
```

## Data Format

### Feature Vectors

Features should be provided as:
- **NumPy arrays**: `.npy` files with shape `(N, D)` where N is number of samples and D is feature dimension
- **CSV files**: Each row is a sample, columns are feature dimensions

### Labels

Labels should be provided as:
- **NumPy arrays**: `.npy` files with shape `(N,)`
- **CSV files**: Single column with cat IDs

## Configuration

### Using Environment Variables

```bash
export CATFACIALID_USE_CUDA=true
export CATFACIALID_VERBOSE=false
export CATFACIALID_LOG_LEVEL=DEBUG
```

### Using Python

```python
from catfacialid.config import SystemConfig, PreprocessingConfig

config = SystemConfig(
    preprocessing=PreprocessingConfig(
        pca_variance_threshold=0.99,
        ica_n_components=100,
    ),
    use_cuda=True,
    verbose=True,
)
```

### Using YAML

```yaml
# config.yaml
preprocessing:
  pca_variance_threshold: 0.99
  ica_n_components: 100
model:
  top_k_predictions: 5
use_cuda: true
verbose: true
```

```python
from pathlib import Path
from catfacialid.config import SystemConfig

config = SystemConfig.from_yaml(Path("config.yaml"))
```

## Next Steps

- Read the [Configuration Guide](configuration.md) for detailed configuration options
- Check the [API Reference](api/core.md) for complete API documentation
- See [Examples](https://github.com/saifmb0/cat-facial-id/tree/main/examples) for more usage patterns

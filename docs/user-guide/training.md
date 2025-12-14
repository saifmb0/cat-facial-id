# Training Guide

This guide covers training the cat identification model.

## Training Pipeline

```python
import numpy as np
from catfacialid.core.preprocessing import FeaturePreprocessor
from catfacialid.core.inference import FAISSInference
from catfacialid.data.loader import load_features, load_labels

# Load your data
X_train = load_features("train_features.npy")
y_train = load_labels("train_labels.csv")

# Step 1: Preprocess features
preprocessor = FeaturePreprocessor(
    pca_variance=0.95,
    ica_components=200,
)
X_processed = preprocessor.fit_transform(X_train, y_train)

# Step 2: Build search index
inference = FAISSInference(top_k=3)
inference.fit(X_processed, y_train)

# Save trained components
import joblib
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(inference, "inference.pkl")
```

## Using the CLI

```bash
catfacialid train \
    --features data/train_features.npy \
    --labels data/train_labels.csv \
    --output-dir models/ \
    --pca-variance 0.95 \
    --ica-components 200 \
    --verbose
```

## Training Configuration

```python
from catfacialid.config import SystemConfig, PreprocessingConfig

config = SystemConfig(
    preprocessing=PreprocessingConfig(
        pca_variance_threshold=0.95,
        lda_max_components=100,
        ica_n_components=200,
        ica_max_iterations=300,
    ),
    verbose=True,
)
```

## Data Requirements

### Features

- **Format**: NumPy array or CSV
- **Shape**: (N, D) where N = samples, D = feature dimension
- **Type**: float32 or float64
- **Values**: Normalized preferred

### Labels

- **Format**: NumPy array or CSV
- **Shape**: (N,) matching feature samples
- **Type**: Integer cat IDs

## Validation

```python
from catfacialid.validation import validate_training_data

# Validate before training
is_valid, errors = validate_training_data(X_train, y_train)

if not is_valid:
    for component, messages in errors.items():
        for msg in messages:
            print(f"{component}: {msg}")
```

## Best Practices

1. **Shuffle data** before training for better generalization
2. **Validate labels** to ensure all classes have sufficient samples
3. **Check for NaN/Inf** in features before training
4. **Use consistent random seeds** for reproducibility

## Memory Optimization

For large datasets:

```python
# Process in batches
batch_size = 10000
for i in range(0, len(X_train), batch_size):
    batch = X_train[i:i+batch_size]
    # Process batch...
```

## Multi-GPU Training

FAISS supports GPU acceleration:

```python
import faiss

# Move index to GPU
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
```

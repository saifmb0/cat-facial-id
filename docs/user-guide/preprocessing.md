# Preprocessing Guide

This guide covers the feature preprocessing pipeline in Cat Facial ID.

## Overview

The preprocessing pipeline transforms raw feature vectors into optimized representations
for cat identification. It uses a fusion of three dimensionality reduction techniques:

1. **PCA**: Captures global variance
2. **LDA**: Maximizes class separability
3. **ICA**: Extracts independent components

## Basic Usage

```python
import numpy as np
from catfacialid.core.preprocessing import FeaturePreprocessor

# Create preprocessor with default settings
preprocessor = FeaturePreprocessor()

# Your training data
X_train = np.random.randn(1000, 2048)  # 1000 samples, 2048 features
y_train = np.random.randint(0, 100, 1000)  # 100 cat classes

# Fit and transform training data
X_train_processed = preprocessor.fit_transform(X_train, y_train)

# Transform test data (no labels needed)
X_test = np.random.randn(100, 2048)
X_test_processed = preprocessor.transform(X_test)
```

## Custom Configuration

```python
preprocessor = FeaturePreprocessor(
    pca_variance=0.99,      # Keep 99% variance
    lda_components=50,      # Max 50 LDA components
    ica_components=100,     # 100 ICA components
    ica_max_iter=300,       # More iterations for convergence
    random_state=42,        # For reproducibility
)
```

## Normalization

Features are L2-normalized after fusion:

```python
from catfacialid.core.preprocessing import normalize

# Normalize features (applied automatically after transform)
X_normalized = normalize(X_processed)
```

## Saving and Loading

```python
import joblib

# Save fitted preprocessor
joblib.dump(preprocessor, "preprocessor.pkl")

# Load preprocessor
preprocessor = joblib.load("preprocessor.pkl")
X_processed = preprocessor.transform(X_new)
```

## Best Practices

1. **PCA Variance**: Use 0.95-0.99 depending on noise level
2. **ICA Components**: Start with 100-200, tune based on performance
3. **Normalization**: Always normalize for FAISS similarity search
4. **Memory**: For large datasets, use incremental PCA

## Technical Details

### Feature Fusion

The final feature vector is created by concatenating:

```
X_fused = [X_pca, X_lda, X_ica]
```

Each component contributes different information:
- **PCA**: Principal directions of variance
- **LDA**: Discriminative directions between classes
- **ICA**: Statistically independent components

### Memory Usage

For N samples with D features:
- Input: N × D floats
- Output: N × (pca_dim + lda_dim + ica_dim) floats

Typical reduction: 2048 → ~300-500 dimensions

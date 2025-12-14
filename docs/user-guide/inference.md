# Inference Guide

This guide covers running predictions with trained models.

## Basic Inference

```python
import joblib
import numpy as np

# Load trained components
preprocessor = joblib.load("preprocessor.pkl")
inference = joblib.load("inference.pkl")

# Load test features
X_test = np.load("test_features.npy")

# Preprocess
X_processed = preprocessor.transform(X_test)

# Predict
predictions, distances = inference.predict(X_processed)

# predictions: (N, top_k) array of cat IDs
# distances: (N, top_k) array of L2 distances
```

## CLI Inference

```bash
catfacialid predict \
    --features data/test_features.npy \
    --model models/ \
    --output predictions.csv \
    --top-k 3
```

## Batch Processing

For large test sets:

```python
batch_size = 1000
all_predictions = []
all_distances = []

for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size]
    batch_processed = preprocessor.transform(batch)
    preds, dists = inference.predict(batch_processed)
    all_predictions.append(preds)
    all_distances.append(dists)

predictions = np.vstack(all_predictions)
distances = np.vstack(all_distances)
```

## Confidence Scores

Convert distances to confidence scores:

```python
def distance_to_confidence(distances, temperature=1.0):
    """Convert L2 distances to confidence scores."""
    # Smaller distance = higher confidence
    scores = np.exp(-distances / temperature)
    # Normalize to sum to 1
    return scores / scores.sum(axis=1, keepdims=True)

confidences = distance_to_confidence(distances)
```

## Threshold-Based Prediction

```python
def predict_with_threshold(predictions, distances, threshold=1.0):
    """Return predictions only if distance below threshold."""
    result = []
    for preds, dists in zip(predictions, distances):
        if dists[0] < threshold:
            result.append(preds[0])
        else:
            result.append(-1)  # Unknown cat
    return np.array(result)
```

## Ensemble Predictions

Combine multiple models:

```python
def ensemble_predict(models, X, top_k=3):
    """Combine predictions from multiple models."""
    all_preds = []
    all_dists = []
    
    for preprocessor, inference in models:
        X_proc = preprocessor.transform(X)
        preds, dists = inference.predict(X_proc)
        all_preds.append(preds)
        all_dists.append(dists)
    
    # Average distances for ranking
    # (implement your fusion strategy)
    return combined_preds, combined_dists
```

## Output Format

### CSV Output

```csv
image_id,prediction_1,prediction_2,prediction_3
img001,cat_42,cat_17,cat_89
img002,cat_103,cat_42,cat_56
```

### JSON Output

```json
{
  "predictions": [
    {
      "image_id": "img001",
      "top_k": [
        {"cat_id": 42, "distance": 0.123, "confidence": 0.89},
        {"cat_id": 17, "distance": 0.456, "confidence": 0.67}
      ]
    }
  ]
}
```

## Performance Tips

1. **GPU FAISS**: Use `faiss-gpu` for large indices
2. **Batch size**: Tune for your hardware
3. **Index type**: IVF for very large indices (>1M vectors)
4. **Preprocessing**: Cache preprocessed features if running multiple queries

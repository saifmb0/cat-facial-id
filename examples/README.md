# Examples

This directory contains example scripts demonstrating how to use the Cat Facial Identification System.

## Complete Pipeline Example

### Usage

Run the complete identification pipeline from data loading to prediction generation:

```bash
python examples/complete_pipeline.py \
    --train-path path/to/train_features.pkl \
    --test-path path/to/test_features.pkl \
    --output-dir ./results
```

### Arguments

- `--train-path` (required): Path to training features pickle file
- `--test-path` (required): Path to test features pickle file
- `--output-dir` (optional): Directory for output files (default: `./results`)
- `--top-k` (optional): Number of top predictions to return (default: 3)
- `--verbose` (optional): Enable verbose logging output

### Output

The script generates a `predictions.txt` file with the format:

```
Image,Prediction_1,Prediction_2,Prediction_3
test/cat_001.jpg,42,127,85
test/cat_002.jpg,156,203,51
...
```

### Example

```bash
# Run with default settings
python examples/complete_pipeline.py \
    --train-path ./data/train_features.pkl \
    --test-path ./data/test_features.pkl

# Run with custom parameters
python examples/complete_pipeline.py \
    --train-path ./data/train_features.pkl \
    --test-path ./data/test_features.pkl \
    --output-dir ./competition_submission \
    --top-k 5 \
    --verbose
```

## Using Individual Components

### Data Loading

```python
from src.catfacialid.data import DataLoader

loader = DataLoader(verbose=True)
X_train, y_train = loader.load_train_features("train.pkl")
X_test, image_names = loader.load_test_features("test.pkl")

# Get dataset statistics
stats = loader.get_stats()
print(f"Training samples: {stats['train_samples']}")
```

### Feature Preprocessing

```python
from src.catfacialid.core import FeatureExtractor, DimensionalityReducer

# Scale features
extractor = FeatureExtractor(seed=42)
X_train_scaled, X_test_scaled = extractor.scale_features(X_train, X_test)

# Apply PCA
reducer = DimensionalityReducer(seed=42)
X_train_pca, X_test_pca = reducer.apply_pca(
    X_train_scaled, X_test_scaled,
    variance_threshold=0.95
)
```

### Feature Fusion

```python
from src.catfacialid.core import FeatureFuser

fuser = FeatureFuser()
X_fused = fuser.fuse_features(X_pca, X_lda, X_ica, normalize_output=True)
```

### Prediction Generation

```python
from src.catfacialid.core import PredictionEngine

engine = PredictionEngine(top_k=3)
engine.build_index(X_train_fused, y_train)
predictions = engine.predict(X_test_fused, image_names)

for img_name, top_3_labels in predictions:
    print(f"{img_name}: {top_3_labels}")
```

## Data Format

### Training Features

Expected format: `(features, labels)`

```python
import joblib

X_train = np.random.randn(1000, 2048)  # 1000 samples, 2048 features (ResNet50)
y_train = np.array([0, 1, 2, ..., 499])  # Class labels

joblib.dump((X_train, y_train), "train_features.pkl")
```

### Test Features

Expected format: `(features, image_names)` or just `features`

```python
X_test = np.random.randn(500, 2048)
image_names = ["test/img_001.jpg", "test/img_002.jpg", ...]

joblib.dump((X_test, image_names), "test_features.pkl")
```

## Performance Considerations

- Input features should be pre-extracted (e.g., ResNet50 embeddings)
- System is optimized for ~500 classes with 1000+ samples per class
- Memory usage scales linearly with dataset size
- Typical inference: ~5ms per sample on CPU

## Common Issues

### FileNotFoundError

Ensure the pickle files exist and paths are correct:
```bash
ls -la path/to/train_features.pkl
ls -la path/to/test_features.pkl
```

### Memory Error

For large datasets, consider:
- Using batch processing on smaller subsets
- Reducing PCA dimensions
- Using GPU FAISS if available (requires faiss-gpu)

### Dimension Mismatch

Verify features have consistent dimensions:
```python
print(X_train.shape)  # Should be (n_samples, n_features)
print(X_test.shape)   # Should be (n_test_samples, n_features)
```

## Next Steps

- Review the API documentation in `docs/API.md`
- Check `README.md` for architecture details
- See `CONTRIBUTING.md` for development guidelines

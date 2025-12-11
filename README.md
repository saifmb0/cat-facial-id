# Cat Facial Identification System

A production-ready deep learning system for identifying individual cats using facial recognition. Leverages advanced feature extraction techniques (PCA, LDA, ICA) and efficient similarity search using FAISS for scalable inference.

## Features

- **Multi-Source Feature Fusion**: Combines PCA, LDA, and ICA features for robust facial representations
- **Scalable Inference**: FAISS-based k-NN search for efficient similarity matching across thousands of cat faces
- **Production Architecture**: Modular design with proper separation of concerns, type hints, and comprehensive error handling
- **Reproducible Results**: Configurable random seeds and comprehensive logging throughout the pipeline
- **Extensive Testing**: Unit tests covering data loading, preprocessing, and inference components
- **Documentation**: Full API documentation and architecture guide

## Project Structure

```
cat-facial-id/
├── src/catfacialid/              # Main package
│   ├── core/                      # Core ML components
│   │   ├── preprocessing.py       # Feature extraction & fusion
│   │   ├── inference.py           # FAISS-based prediction engine
│   │   └── __init__.py
│   ├── data/                      # Data handling
│   │   ├── loader.py              # Dataset loading utilities
│   │   └── __init__.py
│   ├── config.py                  # Configuration management
│   └── __init__.py
├── tests/                         # Unit tests
├── docs/                          # Documentation
├── config/                        # Configuration files
├── setup.py                       # Package installation
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

## Installation

### Requirements
- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

Clone the repository:
```bash
git clone https://github.com/saifmb0/cat-facial-id
cd cat-facial-id
```

Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

For development with testing tools:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from src.catfacialid.data import DataLoader
from src.catfacialid.core import (
    FeatureExtractor,
    DimensionalityReducer,
    FeatureFuser,
    PredictionEngine,
)
from src.catfacialid.config import SystemConfig

# Load configuration
config = SystemConfig.default()

# Load data
loader = DataLoader(verbose=True)
X_train, y_train = loader.load_train_features("path/to/train_features.pkl")
X_test, img_names = loader.load_test_features("path/to/test_features.pkl")

# Extract and process features
extractor = FeatureExtractor(seed=config.model.random_seed)
X_train_scaled, X_test_scaled = extractor.scale_features(X_train, X_test)

# Apply dimensionality reduction
reducer = DimensionalityReducer(seed=config.model.random_seed)
X_train_pca, X_test_pca = reducer.apply_pca(
    X_train_scaled, X_test_scaled,
    variance_threshold=config.preprocessing.pca_variance_threshold
)
X_train_lda, X_test_lda = reducer.apply_lda(X_train_pca, X_test_pca, y_train)
X_train_ica, X_test_ica = reducer.apply_ica(
    X_train_scaled, X_test_scaled,
    n_components=config.preprocessing.ica_n_components
)

# Fuse features
fuser = FeatureFuser()
X_train_fused = fuser.fuse_features(X_train_pca, X_train_lda, X_train_ica)
X_test_fused = fuser.fuse_features(X_test_pca, X_test_lda, X_test_ica)

# Generate predictions
engine = PredictionEngine(top_k=config.model.top_k_predictions)
engine.build_index(X_train_fused, y_train)
predictions = engine.predict(X_test_fused, img_names)

# Results format: [(image_name, [top_3_labels]), ...]
for img_name, predicted_labels in predictions:
    print(f"{img_name}: {predicted_labels}")
```

### Configuration

Configuration is managed through dataclasses in `src/catfacialid/config.py`. Create a custom configuration:

```python
from src.catfacialid.config import (
    SystemConfig, PreprocessingConfig, ModelConfig, DataConfig
)

config = SystemConfig(
    preprocessing=PreprocessingConfig(
        pca_variance_threshold=0.95,
        ica_n_components=200,
        random_seed=42
    ),
    model=ModelConfig(
        num_classes=500,
        top_k_predictions=3,
        random_seed=42
    ),
    data=DataConfig(
        output_dir="./predictions",
        batch_size=32
    )
)
```

## Architecture

### Feature Extraction Pipeline

The system employs a multi-stage feature extraction and fusion approach:

1. **Feature Scaling**: StandardScaler normalization to zero mean and unit variance
2. **PCA Reduction**: Variance-based dimensionality reduction (95% variance by default)
3. **LDA Projection**: Class-discriminant subspace projection
4. **ICA Decomposition**: Independent component analysis for statistical independence
5. **Feature Fusion**: Concatenation of all transformed features followed by L2 normalization

### Inference

FAISS (Facebook AI Similarity Search) provides efficient k-NN search:
- Indexing: O(1) space per vector, O(n) indexing time
- Query: O(n) search time for exact L2 distance
- Returns: Top-k nearest neighbors from training set with corresponding cat IDs

## Testing

Run the test suite:

```bash
# All tests with coverage
pytest tests/ --cov=src/catfacialid

# Specific test module
pytest tests/test_preprocessing.py -v

# Run with markers
pytest -m "not integration"
```

Test coverage targets:
- Data loading and validation
- Feature preprocessing transformations
- Dimensionality reduction techniques
- Feature fusion operations
- Inference engine functionality

## Development

### Code Style

The project uses standard Python formatting tools:

```bash
# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

### Pre-commit Hooks

Set up pre-commit hooks to enforce code standards:

```bash
pip install pre-commit
pre-commit install
```

## Performance Notes

- **Memory Usage**: Depends on training set size and feature dimension; ~8GB for 1M samples with 2048-dim features
- **Inference Speed**: ~5ms per query on CPU with FAISS IndexFlatL2
- **Top-k Accuracy**: Varies with dataset; achieve ~70% top-3 accuracy on large-scale cat datasets

## Model Performance

Performance metrics from TAMMATHON 2025 competition:
- Dataset: 500 cat identity classes, 2000+ training samples per class
- Feature Dimension: 2048 (ResNet50 backbone)
- Fused Dimension After Reduction: ~700 features
- Top-3 Accuracy: Competitive performance in similarity-based identification

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines on:
- Development setup and environment
- Code style and formatting standards
- Testing requirements
- Pull request process

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{catfacialid2025,
  title={Cat Facial Identification System},
  author={Saif, M.},
  year={2025},
  url={https://github.com/saifmb0/cat-facial-id}
}
```

## Acknowledgments

- TAMMATHON 2025 organizing committee
- FAISS library authors (Facebook AI Research)
- scikit-learn contributors
- PyTorch team

## Support

For issues, questions, or suggestions:
1. Check existing GitHub issues
2. Review documentation in `docs/` directory
3. Open a new issue with detailed description

---

Built with precision for production-grade facial identification systems.

# Cat Facial ID Documentation

Welcome to the Cat Facial ID documentation! This system provides production-ready
cat facial identification using deep learning and advanced feature processing.

## Quick Start

```python
from catfacialid import CatIdentifier
from catfacialid.config import SystemConfig

# Initialize with default config
identifier = CatIdentifier()

# Or customize configuration
config = SystemConfig(
    use_cuda=True,
    verbose=True,
)
identifier = CatIdentifier(config=config)

# Train on your data
identifier.fit(train_features, train_labels)

# Predict
predictions = identifier.predict(test_features, top_k=3)
```

## Features

- **Advanced Preprocessing**: PCA, LDA, and ICA feature fusion
- **Fast Similarity Search**: FAISS-powered nearest neighbor retrieval
- **Production Ready**: Type-safe, tested, and documented
- **Flexible Configuration**: Environment variables, YAML, or code

## Installation

```bash
# Basic installation
pip install cat-facial-id

# With all optional features
pip install cat-facial-id[all]

# Development installation
pip install -e ".[dev]"
```

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

quickstart
installation
configuration
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user-guide/preprocessing
user-guide/training
user-guide/inference
user-guide/cli
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/core
api/data
api/config
api/exceptions
api/validation
```

```{toctree}
:maxdepth: 2
:caption: Development

contributing
changelog
```

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/saifmb0/cat-facial-id/blob/main/LICENSE) file for details.

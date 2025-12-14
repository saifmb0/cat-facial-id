# Configuration Guide

This guide covers all configuration options for Cat Facial ID.

## Configuration Methods

Cat Facial ID supports multiple configuration methods:

1. **Python Code**: Direct instantiation
2. **Environment Variables**: Runtime configuration
3. **YAML Files**: File-based configuration
4. **Defaults**: Sensible out-of-the-box settings

## Quick Configuration

### Using Defaults

```python
from catfacialid.config import SystemConfig

config = SystemConfig.default()
```

### Using Environment Variables

```bash
export CATFACIALID_USE_CUDA=true
export CATFACIALID_VERBOSE=false
export CATFACIALID_LOG_LEVEL=DEBUG
export CATFACIALID_PCA_VARIANCE=0.99
```

```python
from catfacialid.config import SystemConfig

# Automatically loads from environment
config = SystemConfig()
```

### Using Python

```python
from catfacialid.config import (
    SystemConfig,
    PreprocessingConfig,
    ModelConfig,
    DataConfig,
)

config = SystemConfig(
    preprocessing=PreprocessingConfig(
        pca_variance_threshold=0.99,
        lda_max_components=50,
        ica_n_components=100,
    ),
    model=ModelConfig(
        num_classes=1000,
        top_k_predictions=5,
    ),
    data=DataConfig(
        output_dir="./results",
        batch_size=64,
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
  lda_max_components: 50
  ica_n_components: 100
  ica_max_iterations: 300
  random_seed: 42

model:
  num_classes: 1000
  feature_dimension: 2048
  top_k_predictions: 5
  random_seed: 42

data:
  train_features_path: ./data/train_features.npy
  test_features_path: ./data/test_features.npy
  output_dir: ./results
  batch_size: 64

logging:
  level: INFO
  format: json
  log_file: ./logs/catfacialid.log

use_cuda: true
verbose: true
```

```python
from pathlib import Path
from catfacialid.config import SystemConfig

config = SystemConfig.from_yaml(Path("config.yaml"))
```

## Configuration Reference

### PreprocessingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pca_variance_threshold` | float | 0.95 | Variance to retain in PCA (0.0-1.0) |
| `lda_max_components` | int | None | Max LDA components (None = automatic) |
| `ica_n_components` | int | 200 | Number of ICA components |
| `ica_max_iterations` | int | 200 | Max ICA iterations |
| `random_seed` | int | 42 | Random seed for reproducibility |

### ModelConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | int | 500 | Number of unique cat classes |
| `feature_dimension` | int | 2048 | Input feature dimension |
| `top_k_predictions` | int | 3 | Number of top predictions |
| `random_seed` | int | 42 | Random seed for reproducibility |

### DataConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_features_path` | Path | None | Path to training features |
| `test_features_path` | Path | None | Path to test features |
| `output_dir` | Path | ./outputs | Output directory |
| `batch_size` | int | 32 | Processing batch size |

### LoggingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | str | INFO | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `format` | str | standard | Log format (json, standard) |
| `log_file` | Path | None | Optional log file path |

### SystemConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preprocessing` | PreprocessingConfig | default | Preprocessing settings |
| `model` | ModelConfig | default | Model settings |
| `data` | DataConfig | default | Data settings |
| `logging` | LoggingConfig | default | Logging settings |
| `use_cuda` | bool | False | Enable CUDA acceleration |
| `verbose` | bool | True | Enable verbose output |

## Environment Variables

All configuration options can be set via environment variables with the `CATFACIALID_` prefix:

| Environment Variable | Config Path |
|---------------------|-------------|
| `CATFACIALID_USE_CUDA` | `use_cuda` |
| `CATFACIALID_VERBOSE` | `verbose` |
| `CATFACIALID_PREPROCESSING__PCA_VARIANCE_THRESHOLD` | `preprocessing.pca_variance_threshold` |
| `CATFACIALID_MODEL__NUM_CLASSES` | `model.num_classes` |
| `CATFACIALID_DATA__OUTPUT_DIR` | `data.output_dir` |
| `CATFACIALID_LOGGING__LEVEL` | `logging.level` |

Note: Nested settings use double underscore (`__`) as delimiter.

## Validation

Configuration values are validated automatically:

```python
from catfacialid.config import SystemConfig

# This will raise ValidationError
config = SystemConfig(
    model=ModelConfig(
        num_classes=10,
        top_k_predictions=20,  # Error: top_k > num_classes
    )
)
```

## Exporting Configuration

```python
config = SystemConfig.default()

# To dictionary
config_dict = config.to_dict()

# To YAML file
config.to_yaml(Path("my_config.yaml"))
```

## Fallback Behavior

If Pydantic is not installed, the configuration falls back to dataclasses with basic functionality:

```python
# Works with or without Pydantic
from catfacialid.config import SystemConfig

config = SystemConfig.default()
```

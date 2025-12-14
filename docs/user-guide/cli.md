# CLI Reference

Complete reference for the Cat Facial ID command-line interface.

## Installation

```bash
pip install cat-facial-id[cli]
```

## Global Options

```bash
catfacialid --help     # Show help
catfacialid --version  # Show version
```

## Commands

### info

Display system and environment information.

```bash
catfacialid info
```

Output includes:
- Package version
- Python version
- PyTorch and CUDA availability
- FAISS information
- System details

### train

Train a new cat identification model.

```bash
catfacialid train [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--features` | PATH | Required | Path to training features |
| `--labels` | PATH | Required | Path to training labels |
| `--output-dir` | PATH | ./models | Output directory |
| `--pca-variance` | FLOAT | 0.95 | PCA variance threshold |
| `--ica-components` | INT | 200 | Number of ICA components |
| `--verbose` | FLAG | False | Enable verbose output |

**Example:**

```bash
catfacialid train \
    --features data/train_features.npy \
    --labels data/train_labels.csv \
    --output-dir models/ \
    --pca-variance 0.99 \
    --verbose
```

### predict

Run predictions on test data.

```bash
catfacialid predict [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--features` | PATH | Required | Path to test features |
| `--model` | PATH | Required | Path to trained model directory |
| `--output` | PATH | predictions.csv | Output file path |
| `--top-k` | INT | 3 | Number of predictions per sample |
| `--verbose` | FLAG | False | Enable verbose output |

**Example:**

```bash
catfacialid predict \
    --features data/test_features.npy \
    --model models/ \
    --output results/predictions.csv \
    --top-k 5
```

### serve

Start the prediction API server.

```bash
catfacialid serve [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | PATH | Required | Path to trained model |
| `--host` | STR | 127.0.0.1 | Server host |
| `--port` | INT | 8000 | Server port |
| `--workers` | INT | 1 | Number of workers |

**Example:**

```bash
catfacialid serve \
    --model models/ \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 4
```

## Environment Variables

The CLI respects configuration from environment variables:

```bash
export CATFACIALID_VERBOSE=true
export CATFACIALID_USE_CUDA=true
export CATFACIALID_LOG_LEVEL=DEBUG

catfacialid train --features data.npy --labels labels.csv
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Model error |

## Shell Completion

Enable shell completion for bash/zsh:

```bash
# Bash
catfacialid --install-completion bash

# Zsh
catfacialid --install-completion zsh

# Fish
catfacialid --install-completion fish
```

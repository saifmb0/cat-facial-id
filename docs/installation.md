# Installation Guide

This guide covers all installation methods for Cat Facial ID.

## Requirements

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, Windows
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large datasets)

## Installation Methods

### From PyPI (Recommended)

```bash
pip install cat-facial-id
```

### From Source

```bash
git clone https://github.com/saifmb0/cat-facial-id.git
cd cat-facial-id
pip install -e .
```

### Using Docker

```bash
docker pull saifmb0/cat-facial-id:latest
docker run -it saifmb0/cat-facial-id:latest
```

## Optional Dependencies

Cat Facial ID has optional dependency groups for different use cases:

### CLI Support

For command-line interface:

```bash
pip install cat-facial-id[cli]
```

### Configuration Validation

For Pydantic-based configuration with environment variable support:

```bash
pip install cat-facial-id[config]
```

### Data Validation

For Pandera-based data validation:

```bash
pip install cat-facial-id[validation]
```

### Development Dependencies

For development and testing:

```bash
pip install cat-facial-id[dev]
```

### All Optional Dependencies

Install everything:

```bash
pip install cat-facial-id[all]
```

## GPU Support

### CUDA (NVIDIA GPUs)

For GPU acceleration with NVIDIA GPUs:

```bash
# Install PyTorch with CUDA support first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install FAISS with GPU support
pip install faiss-gpu

# Then install cat-facial-id
pip install cat-facial-id
```

### Apple Silicon (M1/M2)

PyTorch has native support for Apple Silicon:

```bash
pip install torch torchvision
pip install cat-facial-id
```

## Verifying Installation

```python
import catfacialid

# Check version
print(f"Cat Facial ID version: {catfacialid.__version__}")

# Check if CUDA is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Verify core functionality
from catfacialid.core.preprocessing import FeaturePreprocessor
from catfacialid.core.inference import FAISSInference

print("Installation verified successfully!")
```

## Troubleshooting

### FAISS Installation Issues

If you encounter issues installing FAISS:

```bash
# For CPU-only version
pip install faiss-cpu

# For conda users
conda install -c conda-forge faiss-cpu
```

### PyTorch Installation Issues

If PyTorch installation fails:

```bash
# Use conda for easier installation
conda install pytorch torchvision -c pytorch
```

### Import Errors

If you get import errors after installation:

```bash
# Reinstall in a fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install cat-facial-id
```

## Next Steps

- Follow the [Quick Start Guide](quickstart.md) to get started
- Read about [Configuration](configuration.md) options

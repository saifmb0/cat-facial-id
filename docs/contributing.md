# Contributing to Cat Facial ID

Thank you for your interest in contributing to Cat Facial ID! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- Make (optional, for convenience commands)

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/saifmb0/cat-facial-id.git
cd cat-facial-id

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,all]"

# Verify setup
make test
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_preprocessing.py -v
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make typecheck

# Run all quality checks
make quality
```

### Building Documentation

```bash
# Build documentation
make docs

# Serve documentation locally
make docs-serve
```

## Code Style

- **Formatter**: Black (line length 88)
- **Import Sorting**: isort
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style

### Example

```python
def predict(
    self,
    features: np.ndarray,
    top_k: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict cat identities for query features.
    
    Args:
        features: Query feature array of shape (N, D).
        top_k: Number of top predictions to return.
        
    Returns:
        Tuple containing:
            - predictions: Array of shape (N, top_k) with predicted cat IDs
            - distances: Array of shape (N, top_k) with distances
            
    Raises:
        ModelNotFittedError: If model hasn't been fitted yet.
        DimensionMismatchError: If feature dimensions don't match.
    """
    ...
```

## Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes
4. **Run** quality checks: `make quality`
5. **Commit** with descriptive message
6. **Push** to your fork
7. **Open** a Pull Request

### Commit Messages

Use conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Example: `feat: add ONNX export support`

## Testing Guidelines

- Write tests for all new functionality
- Maintain or improve code coverage
- Use pytest fixtures from `conftest.py`
- Test edge cases and error conditions

## Documentation

- Update docstrings for API changes
- Add examples for new features
- Update README for significant changes
- Keep CHANGELOG updated

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues before creating new ones

Thank you for contributing! 🐱

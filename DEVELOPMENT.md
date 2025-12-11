# Development Guide

This guide provides instructions for setting up a development environment and contributing to the Cat Facial Identification System.

## Environment Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv or conda)

### Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/saifmb0/cat-facial-id.git
cd cat-facial-id
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode with all dependencies:
```bash
pip install -e ".[dev]"
```

4. Set up pre-commit hooks:
```bash
pre-commit install
```

## Development Workflow

### Running Tests

Execute the full test suite:
```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest tests/ --cov=src/catfacialid --cov-report=html
```

Run specific test:
```bash
pytest tests/test_preprocessing.py::TestDimensionalityReducer::test_apply_pca -v
```

Run tests matching a pattern:
```bash
pytest tests/ -k "test_fuse" -v
```

### Code Formatting

Format code with black:
```bash
black src/ tests/
```

Sort imports:
```bash
isort src/ tests/
```

### Code Quality

Lint code:
```bash
flake8 src/ tests/
```

Type check:
```bash
mypy src/
```

Run all quality checks:
```bash
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/
mypy src/
```

### Pre-commit Hooks

Pre-commit hooks automatically format and lint code before commits:
```bash
# Install hooks (one-time)
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Bypass hooks if needed
git commit --no-verify
```

## Project Structure

### Source Code Organization

```
src/catfacialid/
├── __init__.py              # Package initialization
├── config.py                # Configuration dataclasses
├── core/
│   ├── __init__.py
│   ├── preprocessing.py     # Feature extraction and fusion
│   └── inference.py         # Prediction engine
└── data/
    ├── __init__.py
    └── loader.py            # Data loading utilities
```

### Important Files

- `setup.py`: Package configuration and metadata
- `requirements.txt`: Pinned dependency versions
- `pytest.ini`: Test configuration
- `.pre-commit-config.yaml`: Pre-commit hooks configuration
- `.github/workflows/`: CI/CD pipeline definitions

## Adding New Features

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Implement the Feature

- Write clean, modular code
- Add docstrings following Google/NumPy style
- Include type hints
- Handle errors explicitly

### 3. Write Tests

Add tests to `tests/` matching the module being tested:

```python
class TestYourFeature:
    """Test suite for your feature."""
    
    def test_basic_functionality(self):
        """Test basic feature behavior."""
        # Arrange
        data = create_test_data()
        
        # Act
        result = your_function(data)
        
        # Assert
        assert result.shape == expected_shape
```

Run tests and ensure coverage:
```bash
pytest tests/ --cov=src/catfacialid
```

### 4. Format and Lint

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### 5. Update Documentation

- Update docstrings in code
- Update API.md if needed
- Update README if user-facing
- Update CHANGELOG.md

### 6. Commit

Make focused, atomic commits:
```bash
git add .
git commit -m "Add feature: description

Detailed explanation of:
- What the feature does
- Why it was added
- Any breaking changes
"
```

### 7. Create Pull Request

Push your branch and create a PR on GitHub:
```bash
git push origin feature/your-feature-name
```

## Debugging

### Enable Verbose Logging

Most modules accept a `verbose` parameter:

```python
from src.catfacialid.core import DimensionalityReducer

reducer = DimensionalityReducer(verbose=True)
```

### Use Python Debugger

```python
import pdb
pdb.set_trace()  # Execution pauses here

# In debugger:
# n (next line)
# s (step into)
# c (continue)
# p variable_name (print variable)
```

### Profile Performance

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(20)
```

## Common Tasks

### Running All Checks Before Commit

```bash
pytest tests/ --cov=src/catfacialid
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### Updating Dependencies

Edit `requirements.txt` and `setup.py`, then:
```bash
pip install -e ".[dev]" --upgrade
```

### Building Documentation

Documentation is generated from docstrings. Ensure all functions have complete docstrings:

```python
def my_function(param1: int, param2: str) -> dict:
    """Brief description of what the function does.
    
    Longer description explaining the function's purpose,
    parameters, and behavior.
    
    Args:
        param1: Description of parameter 1.
        param2: Description of parameter 2.
    
    Returns:
        Description of the returned value.
    
    Raises:
        ValueError: When validation fails.
        RuntimeError: When operation fails.
    
    Example:
        >>> result = my_function(42, "test")
        >>> result["key"]
        "value"
    """
```

### Releasing a New Version

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Commit changes
4. Tag release:
```bash
git tag -a v1.0.1 -m "Release version 1.0.1"
git push origin v1.0.1
```

The GitHub Actions workflow will automatically publish to PyPI.

## Troubleshooting

### ImportError for modules

Ensure you installed in development mode:
```bash
pip install -e ".[dev]"
```

### Tests not discovered

Ensure test files are named `test_*.py` and are in the `tests/` directory.

### Pre-commit hooks failing

Fix issues manually then try again, or bypass with `git commit --no-verify`

### Type checker errors

Review mypy output and add necessary type hints:
```bash
mypy src/ --show-error-codes
```

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [mypy Type Checker](https://mypy.readthedocs.io/)

## Getting Help

- Review existing issues and discussions on GitHub
- Check docstrings and API.md for usage examples
- Read CONTRIBUTING.md for guidelines
- Open an issue with detailed description

---

Happy coding!

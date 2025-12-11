# Contributing to Cat Facial Identification System

Thank you for your interest in contributing to this project. We welcome contributions from the community and appreciate your effort in making this a better tool for cat facial recognition research.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive community. We are committed to providing a welcoming environment for all contributors regardless of background or experience level.

## Getting Started

### Development Environment Setup

1. Fork the repository and clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/cat-facial-id.git
cd cat-facial-id
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

Execute the test suite before submitting changes:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/catfacialid --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run tests matching pattern
pytest tests/ -k "test_fuse" -v
```

Test coverage should be at least 80% for new code.

### Code Style and Formatting

We enforce consistent code style using industry-standard tools. Before committing, run:

```bash
# Format code with black (line length: 88)
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type check with mypy
mypy src/
```

Pre-commit hooks automatically enforce these before each commit.

### Code Quality Standards

- Write meaningful docstrings following Google/NumPy style for all public functions and classes
- Add type hints to function signatures
- Keep functions focused and under 30 lines when possible
- Use descriptive variable names
- Handle errors explicitly with appropriate exception types
- Add logging for important operations

## Making Changes

### Branching Strategy

1. Create a feature branch from `main`:
```bash
git checkout -b feature/your-feature-name
```

2. Make focused, atomic commits:
```bash
git commit -m "Brief description of change

Longer explanation if needed, describing:
- What problem this solves
- How it solves it
- Any breaking changes or dependencies
"
```

3. Push your branch and create a pull request

### Commit Message Guidelines

- Use imperative mood: "Add feature" not "Added feature"
- Limit first line to 72 characters
- Reference issues when applicable: "Fixes #123"
- Separate concerns into separate commits

### Pull Request Process

1. Update the README if introducing new features
2. Add tests for any new functionality
3. Ensure all tests pass locally: `pytest tests/ --cov`
4. Keep PR focused and manageable in scope
5. Write a clear description of changes and motivation

## Documentation

When adding features:

1. Update relevant docstrings in code
2. Update README.md with usage examples if needed
3. Update docs/API.md with new classes/functions
4. Add inline comments for complex logic

## Reporting Issues

Report bugs through GitHub issues. Include:

- Clear title describing the problem
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant code snippets or error traces
- Relevant dependencies and versions

## Feature Requests

For feature suggestions:

1. Check if already discussed in issues
2. Provide clear use case and motivation
3. Discuss implementation approach
4. Be open to feedback and iteration

## Performance Considerations

When contributing code that affects performance:

1. Benchmark before/after performance
2. Document any performance implications
3. Consider memory usage for large datasets
4. Profile with large-scale data

## Release Process

Maintainers follow semantic versioning:
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

Releases are tagged on main branch.

## Questions?

Feel free to ask questions by:
- Opening a GitHub issue
- Starting a discussion on GitHub
- Reviewing existing documentation

---

Thank you for contributing to making this project better!

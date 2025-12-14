.PHONY: help install install-dev lint format test test-cov clean docker-build docker-run

PYTHON := python3
PACKAGE := catfacialid
SRC_DIR := src
TEST_DIR := tests

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in production mode
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

install-dev:  ## Install package with development dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:  ## Run linters (flake8, mypy)
	flake8 $(SRC_DIR) $(TEST_DIR) --max-line-length=88 --extend-ignore=E203,W503
	isort --check-only $(SRC_DIR) $(TEST_DIR)
	mypy $(SRC_DIR) --ignore-missing-imports

format:  ## Auto-format code with black and isort
	black $(SRC_DIR) $(TEST_DIR) examples
	isort $(SRC_DIR) $(TEST_DIR) examples

test:  ## Run tests with pytest
	pytest $(TEST_DIR) -v

test-cov:  ## Run tests with coverage report
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR)/$(PACKAGE) --cov-report=term-missing --cov-report=html

typecheck:  ## Run strict type checking
	mypy $(SRC_DIR) --strict --ignore-missing-imports

clean:  ## Remove build artifacts and caches
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

docker-build:  ## Build Docker image
	docker build -t $(PACKAGE):latest .

docker-run:  ## Run Docker container interactively
	docker run --rm -it $(PACKAGE):latest

docker-shell:  ## Open shell in Docker container
	docker run --rm -it $(PACKAGE):latest /bin/bash

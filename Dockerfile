# syntax=docker/dockerfile:1

# ---- Builder: create wheel in isolated environment ----
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies required to build any native extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install build tooling first to leverage Docker layer caching
RUN pip install --no-cache-dir --upgrade pip build

# Copy only files needed to build the wheel to keep layers small
COPY requirements.txt setup.py pyproject.toml* README.md LICENSE ./
COPY src src

# Build wheel (output to /tmp/dist)
RUN python -m build --wheel --outdir /tmp/dist

# ---- Runtime: minimal image for execution ----
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user for security
RUN addgroup --system app && adduser --system --ingroup app app
WORKDIR /app

# Copy the prebuilt wheel from builder stage and install
COPY --from=builder /tmp/dist /tmp/dist
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /tmp/dist/*.whl \
    && rm -rf /tmp/dist

# Copy optional assets/scripts that are useful at runtime
COPY examples examples
COPY config config
COPY Submissions Submissions

USER app

# Default command is a helpful no-op; override with your workload.
CMD ["python", "-c", "print('Container ready. Override CMD with your pipeline command, e.g. python examples/complete_pipeline.py --help')"]

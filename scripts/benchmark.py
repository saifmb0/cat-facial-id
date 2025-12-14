#!/usr/bin/env python3
"""Benchmarking script for measuring inference latency and throughput.

This script measures the performance characteristics of the Cat Facial ID
system, providing detailed statistics for production capacity planning.

Usage:
    python scripts/benchmark.py --n-samples 1000 --n-index 10000 --top-k 5

Output:
    - Latency statistics (mean, median, p95, p99)
    - Throughput (queries per second)
    - Memory usage estimates
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from catfacialid.core import FAISSIndex, PredictionEngine  # noqa: E402
from catfacialid.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""

    n_queries: int
    n_index_vectors: int
    dimension: int
    top_k: int
    
    # Latency stats (in milliseconds)
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    
    # Throughput
    queries_per_second: float
    total_time_seconds: float
    
    def __str__(self) -> str:
        """Format results as human-readable string."""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                   BENCHMARK RESULTS                           ║
╠══════════════════════════════════════════════════════════════╣
║  Configuration                                                ║
║  ├─ Queries:          {self.n_queries:>10,}                            ║
║  ├─ Index Size:       {self.n_index_vectors:>10,}                            ║
║  ├─ Dimension:        {self.dimension:>10}                            ║
║  └─ Top-K:            {self.top_k:>10}                            ║
╠══════════════════════════════════════════════════════════════╣
║  Latency (ms)                                                 ║
║  ├─ Mean:             {self.mean_latency_ms:>10.3f}                            ║
║  ├─ Median:           {self.median_latency_ms:>10.3f}                            ║
║  ├─ P95:              {self.p95_latency_ms:>10.3f}                            ║
║  ├─ P99:              {self.p99_latency_ms:>10.3f}                            ║
║  ├─ Min:              {self.min_latency_ms:>10.3f}                            ║
║  ├─ Max:              {self.max_latency_ms:>10.3f}                            ║
║  └─ Std Dev:          {self.std_latency_ms:>10.3f}                            ║
╠══════════════════════════════════════════════════════════════╣
║  Throughput                                                   ║
║  ├─ QPS:              {self.queries_per_second:>10.1f}                            ║
║  └─ Total Time:       {self.total_time_seconds:>10.3f} sec                       ║
╚══════════════════════════════════════════════════════════════╝
"""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": {
                "n_queries": self.n_queries,
                "n_index_vectors": self.n_index_vectors,
                "dimension": self.dimension,
                "top_k": self.top_k,
            },
            "latency_ms": {
                "mean": self.mean_latency_ms,
                "median": self.median_latency_ms,
                "p95": self.p95_latency_ms,
                "p99": self.p99_latency_ms,
                "min": self.min_latency_ms,
                "max": self.max_latency_ms,
                "std": self.std_latency_ms,
            },
            "throughput": {
                "queries_per_second": self.queries_per_second,
                "total_time_seconds": self.total_time_seconds,
            },
        }


def generate_synthetic_data(
    n_vectors: int, dimension: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic feature vectors and labels.

    Args:
        n_vectors: Number of vectors to generate.
        dimension: Dimensionality of each vector.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (features, labels).
    """
    rng = np.random.default_rng(seed)
    features = rng.random((n_vectors, dimension)).astype(np.float32)
    labels = rng.integers(0, 100, size=n_vectors)
    return features, labels


def run_benchmark(
    n_queries: int,
    n_index: int,
    dimension: int,
    top_k: int,
    warmup_queries: int = 100,
    seed: int = 42,
) -> BenchmarkResults:
    """Run benchmark and collect statistics.

    Args:
        n_queries: Number of query vectors to benchmark.
        n_index: Number of vectors in the index.
        dimension: Feature dimension.
        top_k: Number of nearest neighbors to retrieve.
        warmup_queries: Number of warmup queries before timing.
        seed: Random seed for reproducibility.

    Returns:
        BenchmarkResults with timing statistics.
    """
    logger.info(f"Generating synthetic data: {n_index} index vectors, {n_queries} queries")
    
    # Generate data
    index_features, index_labels = generate_synthetic_data(n_index, dimension, seed)
    query_features, _ = generate_synthetic_data(n_queries, dimension, seed + 1)
    warmup_features, _ = generate_synthetic_data(warmup_queries, dimension, seed + 2)

    # Build index
    logger.info("Building FAISS index...")
    engine = PredictionEngine(top_k=top_k, verbose=False)
    engine.build_index(index_features, index_labels)

    # Warmup
    logger.info(f"Running {warmup_queries} warmup queries...")
    for i in range(warmup_queries):
        engine.predict_single(warmup_features[i])

    # Benchmark individual queries
    logger.info(f"Benchmarking {n_queries} queries...")
    latencies: List[float] = []
    
    start_total = time.perf_counter()
    for i in range(n_queries):
        start = time.perf_counter()
        engine.predict_single(query_features[i])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    end_total = time.perf_counter()

    total_time = end_total - start_total
    latencies_array = np.array(latencies)

    return BenchmarkResults(
        n_queries=n_queries,
        n_index_vectors=n_index,
        dimension=dimension,
        top_k=top_k,
        mean_latency_ms=float(np.mean(latencies_array)),
        median_latency_ms=float(np.median(latencies_array)),
        p95_latency_ms=float(np.percentile(latencies_array, 95)),
        p99_latency_ms=float(np.percentile(latencies_array, 99)),
        min_latency_ms=float(np.min(latencies_array)),
        max_latency_ms=float(np.max(latencies_array)),
        std_latency_ms=float(np.std(latencies_array)),
        queries_per_second=n_queries / total_time,
        total_time_seconds=total_time,
    )


def main() -> None:
    """Main entry point for benchmarking script."""
    parser = argparse.ArgumentParser(
        description="Benchmark Cat Facial ID inference performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark with default settings
  python scripts/benchmark.py

  # Full benchmark with 10K queries
  python scripts/benchmark.py --n-samples 10000 --n-index 50000

  # Output results as JSON
  python scripts/benchmark.py --json
        """,
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of query samples to benchmark (default: 1000)",
    )
    parser.add_argument(
        "--n-index",
        type=int,
        default=10000,
        help="Number of vectors in the FAISS index (default: 10000)",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=512,
        help="Feature dimension (default: 512)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbors (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Number of warmup queries (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO", json_output=False)

    # Run benchmark
    results = run_benchmark(
        n_queries=args.n_samples,
        n_index=args.n_index,
        dimension=args.dimension,
        top_k=args.top_k,
        warmup_queries=args.warmup,
        seed=args.seed,
    )

    # Output results
    if args.json:
        import json
        print(json.dumps(results.to_dict(), indent=2))
    else:
        print(results)


if __name__ == "__main__":
    main()

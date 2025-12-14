#!/usr/bin/env python3
"""Generate a sample submission CSV from predictions.

Usage:
    python scripts/generate_sample_submission.py \
        --predictions outputs/predictions.csv \
        --output Submissions/sample_submission.csv \
        --top-k 3

This utility reads a predictions CSV with columns:
    image_name,pred_1,pred_2,...,pred_k
and writes a sample submission with the same format.

If you don't have predictions yet, it will generate a tiny dummy file.
"""

import argparse
import csv
from pathlib import Path
from typing import List


def generate_dummy(output: Path, top_k: int) -> None:
    rows = [
        ("img001.jpg", [42, 17, 89]),
        ("img002.jpg", [103, 42, 56]),
        ("img003.jpg", [5, 12, 7]),
    ]
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["image_name"] + [f"pred_{i+1}" for i in range(top_k)]
        writer.writerow(header)
        for name, preds in rows:
            writer.writerow([name] + preds[:top_k])


def from_predictions(pred_path: Path, output: Path, top_k: int) -> None:
    with pred_path.open() as inp, output.open("w", newline="") as out:
        reader = csv.DictReader(inp)
        fieldnames: List[str] = ["image_name"] + [f"pred_{i+1}" for i in range(top_k)]
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            image_name = row.get("image_name") or row.get("image") or "unknown"
            preds = [row.get(f"pred_{i+1}", "") for i in range(top_k)]
            writer.writerow({"image_name": image_name, **{f"pred_{i+1}": preds[i] for i in range(top_k)}})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample submission CSV")
    parser.add_argument("--predictions", type=Path, help="Path to predictions CSV", default=None)
    parser.add_argument("--output", type=Path, help="Output CSV path", default=Path("Submissions/sample_submission.csv"))
    parser.add_argument("--top-k", type=int, help="Number of predictions per sample", default=3)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.predictions is None or not args.predictions.exists():
        generate_dummy(args.output, args.top_k)
        print(f"Wrote dummy sample submission to {args.output}")
    else:
        from_predictions(args.predictions, args.output, args.top_k)
        print(f"Wrote sample submission from {args.predictions} to {args.output}")


if __name__ == "__main__":
    main()

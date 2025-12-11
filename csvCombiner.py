"""Utility script for combining multiple CSV prediction files into a single submission.

This module provides functionality to aggregate predictions from multiple runs
or multiple machines into a consolidated submission file for the cat facial
identification competition.

Example:
    Combine all CSV files in a directory:
    
    $ python csvCombiner.py --input-dir ./predictions --output-dir ./submissions
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script.
    
    Args:
        verbose: Enable verbose logging output.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def combine_csv_files(
    input_directory: str,
    output_directory: str = "./",
    output_filename: str = "submission.csv",
    exclude_files: list = None,
) -> str:
    """Combine multiple CSV prediction files into single submission.
    
    Iterates through all CSV files in the input directory (excluding specified
    files), concatenates them with proper indexing, and writes to output file.
    
    Args:
        input_directory: Directory containing CSV files to combine.
        output_directory: Directory for output submission file.
        output_filename: Name for combined output file.
        exclude_files: List of filenames to exclude from combination.
    
    Returns:
        Path to the created submission file.
    
    Raises:
        FileNotFoundError: If input directory doesn't exist.
        ValueError: If no CSV files found to combine.
        IOError: If unable to write output file.
    """
    if exclude_files is None:
        exclude_files = []

    input_path = Path(input_directory)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_directory}")

    if not input_path.is_dir():
        raise ValueError(f"Path is not a directory: {input_directory}")

    # Find all CSV files to combine
    csv_files = []
    for filename in sorted(input_path.glob("*.csv")):
        if filename.name not in exclude_files and filename.name != output_filename:
            csv_files.append(filename)

    if not csv_files:
        raise ValueError(
            f"No CSV files found in {input_directory} to combine. "
            f"Excluded files: {exclude_files}"
        )

    logger.info(f"Found {len(csv_files)} CSV files to combine")

    # Load and combine dataframes
    dataframes = []
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            dataframes.append(df)
            logger.debug(f"Loaded {filepath.name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Failed to load {filepath.name}: {e}")
            continue

    if not dataframes:
        raise ValueError("No dataframes were successfully loaded")

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined {len(combined_df)} total rows from {len(dataframes)} files")

    # Create output directory if needed
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # Add numeric index column (starting from 0)
    combined_df.insert(0, "", range(len(combined_df)))

    # Save combined file
    output_file = output_path / output_filename
    try:
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Submission saved to {output_file}")
        return str(output_file)
    except Exception as e:
        raise IOError(f"Failed to write output file: {e}")


def main() -> int:
    """Command-line entry point for CSV combination utility.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Combine multiple CSV prediction files into a single submission.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine CSVs in current directory
  python csvCombiner.py --input-dir .
  
  # Combine with custom output
  python csvCombiner.py --input-dir ./predictions --output-dir ./submissions
  
  # Exclude specific files
  python csvCombiner.py --input-dir . --exclude manual_review.csv
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=".",
        help="Directory containing CSV files to combine (default: current directory)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Directory for output submission file (default: current directory)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="submission.csv",
        help="Name of output submission file (default: submission.csv)",
    )

    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=["submission.csv"],
        help="Filenames to exclude from combination",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    try:
        output_path = combine_csv_files(
            input_directory=args.input_dir,
            output_directory=args.output_dir,
            output_filename=args.output_file,
            exclude_files=args.exclude,
        )
        print(f"Success! Combined submission saved to: {output_path}")
        return 0
    except (FileNotFoundError, ValueError, IOError) as e:
        logger.error(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

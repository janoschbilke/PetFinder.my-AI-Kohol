"""CLI for merging evaluation results from multiple machines.

Usage:

    python -m src.evaluation.merge_results \\
        --results-dir results \\
        --output-dir results/_merged

Recursively scans ``--results-dir`` for ``metrics/*.json`` files
produced by different runs (potentially on different hosts), writes
a merged CSV, and regenerates the comparison plots for the combined
dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.results_manager import merge_results
from src.evaluation.visualize import generate_plots_from_merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge model evaluation results from multiple runs / machines."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root directory containing per-run subdirectories (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write merged CSV + plots (default: <results-dir>/_merged)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only produce merged CSV, skip plot generation.",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    output_dir: Path = args.output_dir or (results_dir / "_merged")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Merging results under: {results_dir}")

    if args.no_plots:
        df = merge_results(results_dir, output_csv=output_dir / "merged_results.csv")
        print(f"Wrote merged CSV with {len(df)} rows -> "
              f"{output_dir / 'merged_results.csv'}")
    else:
        generate_plots_from_merged(results_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()
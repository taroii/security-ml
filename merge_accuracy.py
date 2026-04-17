"""
merge_accuracy.py

Glob all per-cell accuracy CSVs written by main_video.py and combine
into a single long-format DataFrame.

Usage:
    python merge_accuracy.py
    python merge_accuracy.py --results-dir ./accuracy_results --output merged_accuracy.csv
"""

import argparse
import glob
import os
import sys

import pandas as pd


def merge_accuracy(results_dir: str, output_path: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, "acc_k*_sigma*.csv")
    paths = sorted(glob.glob(pattern))

    if not paths:
        print(f"No files matching {pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} per-cell CSVs:")
    for p in paths:
        print(f"  {p}")

    dfs = [pd.read_csv(p) for p in paths]
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values(["k", "sigma"]).reset_index(drop=True)

    merged.to_csv(output_path, index=False, float_format="%.4f")
    print(f"\nMerged ({len(merged)} rows) -> {output_path}")
    print("\nPreview:")
    print(merged.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="./accuracy_results")
    parser.add_argument("--output", type=str, default="./merged_accuracy.csv")
    args = parser.parse_args()

    merge_accuracy(args.results_dir, args.output)
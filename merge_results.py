import argparse
import glob
import os
import sys

import pandas as pd


def merge_results(results_dir: str, output_path: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, "attack_k*_sigma*.csv")
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

    merged.to_csv(output_path, index=False, float_format="%.6f")
    print(f"\nMerged ({len(merged)} rows) -> {output_path}")
    print("\nPreview:")
    print(merged.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--output", type=str, default="./merged_results.csv")
    args = parser.parse_args()

    merge_results(args.results_dir, args.output)
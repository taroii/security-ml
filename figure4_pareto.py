"""
figure4_pareto.py

Figure 4 of "Learnable Obfuscation for Temporally Related Video Data".

Privacy-utility Pareto frontier. For each (k, sigma) cell:
  x-axis = test accuracy of obfuscated classifier (utility)
  y-axis = 1 - normalized adversary score (privacy; larger = more private)

One curve per k, connecting sigma values in increasing order. Each point
labeled by sigma. Baseline accuracy shown as a vertical reference line
(the utility ceiling).

Joins merged_accuracy.csv (from main_video.py + merge_accuracy.py) with
merged_results.csv (from membership_inference.py + merge_results.py)
on (k, sigma).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


K_STYLE = {
    0: ("#d62728", "o", "-"),
    1: ("#ff7f0e", "s", "--"),
    5: ("#1f77b4", "^", "-."),
}


def load_and_join(accuracy_path: str, mia_path: str) -> pd.DataFrame:
    if not os.path.exists(accuracy_path):
        print(f"Accuracy CSV not found: {accuracy_path}", file=sys.stderr)
        print("Run merge_accuracy.py first.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(mia_path):
        print(f"MIA CSV not found: {mia_path}", file=sys.stderr)
        print("Run merge_results.py first.", file=sys.stderr)
        sys.exit(1)

    acc = pd.read_csv(accuracy_path)
    mia = pd.read_csv(mia_path)

    # round sigma to avoid float-equality issues during join
    acc["sigma"] = acc["sigma"].round(6)
    mia["sigma"] = mia["sigma"].round(6)

    joined = pd.merge(acc, mia, on=["k", "sigma"], how="inner")

    if len(joined) == 0:
        print("ERROR: no overlapping (k, sigma) cells between the two CSVs.",
              file=sys.stderr)
        print(f"Accuracy cells: {sorted(set(zip(acc['k'], acc['sigma'])))}",
              file=sys.stderr)
        print(f"MIA cells:      {sorted(set(zip(mia['k'], mia['sigma'])))}",
              file=sys.stderr)
        sys.exit(1)

    return joined


def make_figure(joined: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    # privacy axis: 1 - adversary score (half-integer rule)
    joined = joined.copy()
    joined["privacy"] = 1.0 - joined["score_half"]

    for k in sorted(joined["k"].unique()):
        sub = joined[joined["k"] == k].sort_values("sigma")
        color, marker, ls = K_STYLE.get(k, ("gray", "o", "-"))

        ax.plot(
            sub["accuracy_obfuscated"], sub["privacy"],
            color=color, linestyle=ls, marker=marker,
            markersize=10, linewidth=2.0,
            label=f"$k = {k}$",
        )

        # annotate each point with its sigma value
        for _, row in sub.iterrows():
            ax.annotate(
                f"$\\sigma={row['sigma']:.2f}$",
                xy=(row["accuracy_obfuscated"], row["privacy"]),
                xytext=(8, 5), textcoords="offset points",
                fontsize=9, color=color,
            )

    # baseline accuracy as vertical reference line (utility ceiling)
    if "accuracy_baseline" in joined.columns:
        baseline_acc = joined["accuracy_baseline"].iloc[0]
        ax.axvline(
            x=baseline_acc, color="black", linestyle=":", linewidth=1.5,
            label=f"Baseline accuracy ({baseline_acc:.1f}%)",
        )

    # always-H+ baseline as horizontal privacy reference
    always_h_plus = joined["always_h_plus_half"].mean()
    ax.axhline(
        y=1.0 - always_h_plus,
        color="dimgray", linestyle="--", linewidth=1.0,
        label=f"Always-H$^+$ privacy floor ({1-always_h_plus:.2f})",
    )

    ax.set_xlabel("Test accuracy (%)", fontsize=12)
    ax.set_ylabel(r"Privacy: $1 - \mathrm{normalized\ adversary\ score}$",
                  fontsize=12)
    ax.set_title(
        "Figure 4: Privacy-Utility Pareto Frontier\n"
        "(half-integer weighted MIA score)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy", type=str, default="./merged_accuracy.csv")
    parser.add_argument("--mia", type=str, default="./merged_results.csv")
    parser.add_argument("--output", type=str, default="./fig4_pareto.png")
    args = parser.parse_args()

    joined = load_and_join(args.accuracy, args.mia)
    print(f"Joined {len(joined)} (k, sigma) cells")
    print(joined[["k", "sigma", "accuracy_obfuscated", "score_half"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))

    make_figure(joined, args.output)
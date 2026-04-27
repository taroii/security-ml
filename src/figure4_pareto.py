import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


K_STYLE = {
    0: ("#d62728", "o", "-"),
    1: ("#ff7f0e", "s", "--"),
    5: ("#1f77b4", "^", "-."),
}


# map sigma -> marker size (points). Small sigma -> small marker.
def sigma_to_size(sigma: float) -> float:
    size_map = {0.01: 6, 0.05: 10, 0.10: 14, 0.50: 20}
    return size_map.get(round(sigma, 2), 10)


# Privacy axis: 1 - Attack 3 (frame-level membership inference, half-integer).
# Higher = more private. Utility axis: random-sampling downstream accuracy.
PRIVACY_SCORE_COL = "attack3_score_half"


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
    fig, ax = plt.subplots(figsize=(9, 5.5))

    joined = joined.copy()
    joined["privacy"] = 1.0 - joined[PRIVACY_SCORE_COL]

    for k in sorted(joined["k"].unique()):
        sub = joined[joined["k"] == k].sort_values("sigma")
        color, marker, ls = K_STYLE.get(k, ("gray", "o", "-"))

        ax.plot(
            sub["accuracy_obfuscated"], sub["privacy"],
            color=color, linestyle=ls, linewidth=1.5,
            alpha=0.6, zorder=1,
        )

        sizes = [sigma_to_size(s) ** 2 for s in sub["sigma"]]
        ax.scatter(
            sub["accuracy_obfuscated"], sub["privacy"],
            s=sizes, c=color, marker=marker,
            edgecolors="white", linewidths=1.0,
            zorder=3,
        )

    if "accuracy_baseline" in joined.columns:
        baseline_acc = joined["accuracy_baseline"].iloc[0]
        ax.axvline(
            x=baseline_acc, color="black", linestyle=":", linewidth=1.2,
        )
        ax.text(
            baseline_acc, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] else 0.5,
            f" baseline ({baseline_acc:.1f}%)",
            fontsize=9, color="black", va="top",
        )

    # privacy floor: when adversary perfectly recovers, score = 1, privacy = 0.
    # The closed-form random-guess privacy ceiling is 1 - random_baseline.
    # Re-derive the baseline from CSV metadata (n_universe_clips, clip_len,
    # num_frames) to dodge the off-by-T bug present in older CSVs.
    if {"n_universe_clips", "clip_len", "num_frames"}.issubset(joined.columns):
        n_clips = float(joined["n_universe_clips"].iloc[0])
        clip_len = float(joined["clip_len"].iloc[0])
        T = float(joined["num_frames"].iloc[0])
        rand_baseline = T / (n_clips * clip_len) + 0.5 * (clip_len - T) / (n_clips * clip_len)
        rand_priv = 1.0 - rand_baseline
        ax.axhline(
            y=rand_priv, color="dimgray", linestyle="--", linewidth=1.0,
        )
        ax.text(
            ax.get_xlim()[0] if ax.get_xlim()[0] else 40,
            rand_priv, f" random-guess privacy ({rand_priv:.4f})",
            fontsize=9, color="dimgray", va="bottom",
        )

    ax.set_xlabel("Test accuracy (%)", fontsize=12)
    ax.set_ylabel(
        r"Privacy: $1 - \mathrm{Attack\,3\ half\text{-}integer\ score}$",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    k_handles = [
        Line2D([0], [0],
               color=K_STYLE[k][0], linestyle=K_STYLE[k][2],
               marker=K_STYLE[k][1], markersize=9, linewidth=1.5,
               label=f"$k = {k}$")
        for k in sorted(joined["k"].unique())
    ]
    sigma_vals = sorted(joined["sigma"].unique())
    sigma_handles = [
        Line2D([0], [0],
               color="gray", linestyle="None",
               marker="o", markersize=sigma_to_size(s),
               label=f"$\\sigma = {s:.2f}$")
        for s in sigma_vals
    ]

    leg_k = ax.legend(
        handles=k_handles, title="Mixing",
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=10, title_fontsize=11, frameon=True,
    )
    ax.add_artist(leg_k)

    ax.legend(
        handles=sigma_handles, title="Noise",
        loc="upper left", bbox_to_anchor=(1.02, 0.55),
        fontsize=10, title_fontsize=11, frameon=True,
        labelspacing=1.4, handletextpad=1.0, borderpad=0.8,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy", type=str, default="./merged_accuracy.csv")
    parser.add_argument("--mia", type=str, default="./merged_results.csv")
    parser.add_argument("--output", type=str, default="./images/fig4_pareto.pdf")
    args = parser.parse_args()

    joined = load_and_join(args.accuracy, args.mia)
    print(f"Joined {len(joined)} (k, sigma) cells")
    print(joined[["k", "sigma", "accuracy_obfuscated", PRIVACY_SCORE_COL]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))

    make_figure(joined, args.output)

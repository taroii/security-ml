import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# k -> (color, marker, linestyle)
K_STYLE = {
    0: ("#d62728", "o", "-"),   # red, no mixing
    1: ("#ff7f0e", "s", "--"),  # orange, light mixing
    5: ("#1f77b4", "^", "-."),  # blue, heavy mixing
}


def _style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_h_plus_win_rate(ax, df: pd.DataFrame):
    """Panel (a): H+ win rate vs sigma, one curve per k."""
    for k in sorted(df["k"].unique()):
        sub = df[df["k"] == k].sort_values("sigma")
        color, marker, ls = K_STYLE.get(k, ("gray", "o", "-"))

        ax.errorbar(
            sub["sigma"], sub["h_plus_win_rate"],
            yerr=sub["h_plus_win_se"],
            color=color, linestyle=ls, marker=marker,
            markersize=7, linewidth=2.0, capsize=4,
            label=f"$k = {k}$",
        )

    ax.axhline(
        y=0.5, color="black", linestyle=":", linewidth=1.2,
        label="Chance (0.5)",
    )

    ax.set_xscale("log")
    ax.set_ylim(0.3, 1.05)
    _style_ax(
        ax,
        xlabel=r"Noise level $\sigma$ (log scale)",
        ylabel=r"H$^+$ win rate",
        title=r"(a) Likelihood-Ratio Test Win Rate",
    )
    ax.legend(fontsize=10, loc="best")


def plot_normalized_score(ax, df: pd.DataFrame):
    """
    Panel (b): Normalized weighted score (half) vs sigma, one curve per k.
    Reference lines: Lemma-3 random baseline and always-H+ baseline.
    """
    for k in sorted(df["k"].unique()):
        sub = df[df["k"] == k].sort_values("sigma")
        color, marker, ls = K_STYLE.get(k, ("gray", "o", "-"))

        n_t = sub["n_targets"].iloc[0]
        se = sub["score_half_std"] / np.sqrt(n_t)

        ax.errorbar(
            sub["sigma"], sub["score_half"],
            yerr=se,
            color=color, linestyle=ls, marker=marker,
            markersize=7, linewidth=2.0, capsize=4,
            label=f"$k = {k}$",
        )

    rand_half = df["random_baseline_half"].mean()
    always_h_plus = df["always_h_plus_half"].mean()

    ax.axhline(
        y=always_h_plus, color="dimgray", linestyle="--", linewidth=1.5,
        label=f"Always-H$^+$ baseline ({always_h_plus:.2f})",
    )
    ax.axhline(
        y=rand_half, color="lightgray", linestyle=":", linewidth=1.5,
        label=f"Random baseline (Lemma 3, {rand_half:.4f})",
    )

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.05)
    _style_ax(
        ax,
        xlabel=r"Noise level $\sigma$ (log scale)",
        ylabel=r"Normalized weighted score (half-integer)",
        title=r"(b) Adversary Score vs Noise Level",
    )
    ax.legend(fontsize=10, loc="best")


def make_figure(df: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_h_plus_win_rate(axes[0], df)
    plot_normalized_score(axes[1], df)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./merged_results.csv")
    parser.add_argument("--output", type=str, default="./images/fig2_mia_robustness.png")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        print("Run merge_results.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"k values:     {sorted(df['k'].unique())}")
    print(f"sigma values: {sorted(df['sigma'].unique())}")

    make_figure(df, args.output)
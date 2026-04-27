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
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_attack_panel(ax, df, score_col, std_col, baseline_col,
                       title, ylabel, ylim=None, log_baseline=False):
    """Generic per-attack panel: one curve per k, plus chance baseline."""
    for k in sorted(df["k"].unique()):
        sub = df[df["k"] == k].sort_values("sigma")
        color, marker, ls = K_STYLE.get(k, ("gray", "o", "-"))
        n_t = sub["n_targets"].iloc[0]
        se = sub[std_col] / np.sqrt(n_t)

        ax.errorbar(
            sub["sigma"], sub[score_col],
            yerr=se,
            color=color, linestyle=ls, marker=marker,
            markersize=7, linewidth=2.0, capsize=4,
            label=f"$k = {k}$",
        )

    if baseline_col in df.columns:
        baseline = df[baseline_col].mean()
        ax.axhline(
            y=baseline, color="dimgray", linestyle=":", linewidth=1.5,
            label=f"Random baseline ({baseline:.4f})",
        )

    ax.set_xscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
    _style_ax(
        ax,
        xlabel=r"Noise level $\sigma$ (log scale)",
        ylabel=ylabel,
        title=title,
    )
    ax.legend(fontsize=9, loc="best")


def make_figure(df: pd.DataFrame, save_path: str):
    """
    Three-panel figure: one panel per attack.
      (a) Index inference (Attack 1, half-integer, clip given)
      (b) Clip inference  (Attack 2, top-1)
      (c) Frame-level MIA (Attack 3, half-integer)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    _plot_attack_panel(
        axes[0], df,
        score_col="attack1_score_half",
        std_col="attack1_score_half_std",
        baseline_col="attack1_baseline_half",
        title="(a) Index Inference (clip given)",
        ylabel=r"Half-integer score (clip fixed)",
        ylim=(0.45, 1.05),
    )
    _plot_attack_panel(
        axes[1], df,
        score_col="attack2_top1",
        std_col="attack2_top1_std",
        baseline_col="attack2_baseline",
        title="(b) Clip Inference (top-1)",
        ylabel=r"Top-1 clip accuracy",
        ylim=(-0.02, 1.05),
    )
    _plot_attack_panel(
        axes[2], df,
        score_col="attack3_score_half",
        std_col="attack3_score_half_std",
        baseline_col="attack3_baseline_half",
        title="(c) Frame-level Membership Inference",
        ylabel=r"Half-integer score (whole universe)",
        ylim=(-0.02, 1.05),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./merged_results.csv")
    parser.add_argument("--output", type=str, default="./images/fig2_mia_robustness.pdf")
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

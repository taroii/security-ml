"""
figure3_integer_vs_half.py

Figure 3 of "Learnable Obfuscation for Temporally Related Video Data".

Three side-by-side panels (one per k value) comparing the empirical
adversary score under integer vs half-integer weighting rules at
matched noise level. Empirically validates the prior-success gap
predicted by Lemma 3 and the noise-calibration consequence of
Theorem 7.

For each panel:
  x-axis = noise level sigma (log scale)
  y-axis = normalized adversary score
  curves: score_int (blue, solid) and score_half (red, dashed)
  refs:   integer random baseline (light blue dotted)
          half-integer random baseline (light red dotted)

Shares a y-axis across panels for direct visual comparison.

Reads merged_results.csv produced by merge_results.py.
Writes fig3_int_vs_half.png.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# rule -> (color, marker, linestyle, label)
RULE_STYLE = {
    "int":  ("#1f77b4", "o", "-",  "Integer ($w \\in \\{0, 1\\}$)"),
    "half": ("#d62728", "s", "--", "Half-integer ($w \\in \\{0, 1/2, 1\\}$)"),
}


def _style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_panel(ax, df_k: pd.DataFrame, k: int, show_ylabel: bool):
    """One panel for a single k value: integer vs half-integer score."""
    sub = df_k.sort_values("sigma")
    n_t = sub["n_targets"].iloc[0]

    for rule in ("int", "half"):
        color, marker, ls, label = RULE_STYLE[rule]
        score_col = f"score_{rule}"
        std_col = f"score_{rule}_std"

        se = sub[std_col] / np.sqrt(n_t)

        ax.errorbar(
            sub["sigma"], sub[score_col],
            yerr=se,
            color=color, linestyle=ls, marker=marker,
            markersize=7, linewidth=2.0, capsize=4,
            label=label,
        )

    # analytical random baselines per rule (near-constant across sigma)
    rand_int = sub["random_baseline_int"].mean()
    rand_half = sub["random_baseline_half"].mean()

    ax.axhline(
        y=rand_int, color=RULE_STYLE["int"][0],
        linestyle=":", linewidth=1.0, alpha=0.6,
        label=f"Integer random baseline ({rand_int:.4f})",
    )
    ax.axhline(
        y=rand_half, color=RULE_STYLE["half"][0],
        linestyle=":", linewidth=1.0, alpha=0.6,
        label=f"Half-integer random baseline ({rand_half:.4f})",
    )

    ax.set_xscale("log")
    _style_ax(
        ax,
        xlabel=r"Noise level $\sigma$ (log scale)",
        ylabel=r"Normalized adversary score" if show_ylabel else "",
        title=f"$k = {k}$",
    )
    if not show_ylabel:
        ax.set_yticklabels([])


def make_figure(df: pd.DataFrame, save_path: str):
    ks = sorted(df["k"].unique())
    n_panels = len(ks)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5 * n_panels, 4.5),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    # determine shared y-limits across panels for fair comparison
    ymin = min(df["score_int"].min(), df["score_half"].min(),
               df["random_baseline_int"].min(), df["random_baseline_half"].min())
    ymax = max(df["score_int"].max(), df["score_half"].max())
    ypad = 0.05 * (ymax - ymin)
    yrange = (max(0, ymin - ypad), min(1.05, ymax + ypad))

    for ax, k in zip(axes, ks):
        df_k = df[df["k"] == k]
        plot_panel(ax, df_k, k, show_ylabel=(k == ks[0]))
        ax.set_ylim(yrange)

    # one legend for the whole figure (use first panel's handles)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=2, fontsize=10, frameon=True,
    )

    fig.suptitle(
        "Figure 3: Integer vs Half-Integer Weighted MIA Score "
        "(per mixing intensity $k$)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./merged_results.csv")
    parser.add_argument("--output", type=str, default="./fig3_int_vs_half.png")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        print("Run merge_results.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"k values:     {sorted(df['k'].unique())}")
    print(f"sigma values: {sorted(df['sigma'].unique())}")
    print()
    print("Per-cell score comparison:")
    print(df[["k", "sigma", "score_int", "score_half"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))

    make_figure(df, args.output)
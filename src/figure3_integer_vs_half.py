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
    ax.set_title(title, fontsize=12)
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

    ax.set_xscale("log")
    _style_ax(
        ax,
        xlabel=r"Noise level $\sigma$ (log scale)",
        ylabel=r"Normalized adversary score" if show_ylabel else "",
        title=f"$k = {k}$",
    )


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

    # shared y-limits: pad data range symmetrically, clamped to [0, 1.05]
    ymin = min(df["score_int"].min(), df["score_half"].min())
    ymax = max(df["score_int"].max(), df["score_half"].max())
    pad = 0.5 * (ymax - ymin)
    yrange = (max(0.0, ymin - pad), min(1.05, ymax + pad))

    for ax, k in zip(axes, ks):
        df_k = df[df["k"] == k]
        plot_panel(ax, df_k, k, show_ylabel=(k == ks[0]))
        ax.set_ylim(yrange)

    # one legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=2, fontsize=10, frameon=True,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./merged_results.csv")
    parser.add_argument("--output", type=str, default="./images/fig3_int_vs_half.png")
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
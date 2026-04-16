"""
membership_inference_plots.py

Generates visualizations from membership inference attack results
saved by membership_inference.py. Produces:

  Fig 3a: Normalized weighted score vs sigma
  Fig 3b: H+ win rate vs sigma
  Fig 3c: Mean delta vs sigma
  Fig 3d: Distribution of weighted scores per sigma (box plots)

Loads from:
  attack_results.csv        — summary table
  attack_results_full.npz   — full per-target arrays per sigma

Also prints a clean pandas table for copy-pasting into LaTeX.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ----------------------------
# Load results
# ----------------------------

def load_results(
    csv_path: str = "attack_results.csv",
    npz_path: str = "attack_results_full.npz",
):
    """
    Load summary table and full per-target arrays.

    Returns:
      df:   pandas DataFrame with summary statistics per sigma
      npz:  numpy npz file with full arrays
    """
    csv_p = Path(csv_path)
    npz_p = Path(npz_path)

    if not csv_p.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. "
            "Please run membership_inference.py first."
        )
    if not npz_p.exists():
        raise FileNotFoundError(
            f"Could not find {npz_path}. "
            "Please run membership_inference.py first."
        )

    df = pd.read_csv(csv_p)
    npz = np.load(npz_p, allow_pickle=True)
    return df, npz


# ----------------------------
# Print formatted table
# ----------------------------

def print_results_table(df: pd.DataFrame, num_frames: int = 16):
    """
    Print a clean summary table suitable for copying into LaTeX.
    """
    print("\n" + "=" * 70)
    print("MEMBERSHIP INFERENCE ATTACK RESULTS")
    print("=" * 70)

    # build display dataframe
    display = pd.DataFrame({
        "sigma": df["sigma"],
        "H+ Win Rate": df["h_plus_win_rate"].map("{:.3f}".format),
        "Mean Weighted Score": df["mean_weighted_score"].map(
            lambda x: f"{x:.2f}/{num_frames}"
        ),
        "Normalized Score": df["normalized_score"].map("{:.4f}".format),
        "Mean Delta": df["mean_delta"].map("{:+.2f}".format),
        "Std Delta": df["std_delta"].map("{:.2f}".format),
        "Random Baseline": df["random_baseline"].map("{:.4f}".format),
    })

    print(display.to_string(index=False))

    print("\n--- LaTeX-ready rows (paste into tabular) ---")
    for _, row in df.iterrows():
        print(
            f"  {row['sigma']:.2f} & "
            f"{row['h_plus_win_rate']:.3f} & "
            f"{row['mean_weighted_score']:.2f}/{num_frames} & "
            f"{row['normalized_score']:.4f} & "
            f"{row['mean_delta']:+.2f} $\\pm$ {row['std_delta']:.2f} \\\\"
        )
    print("=" * 70 + "\n")


# ----------------------------
# Individual plot functions
# ----------------------------

def _style_ax(ax, xlabel, ylabel, title):
    """Apply consistent styling to an axis."""
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_normalized_score(ax, df: pd.DataFrame, num_frames: int = 16):
    """
    Fig 3a: Normalized weighted score vs sigma.
    Compares empirical adversary score against random baseline.
    """
    sigmas = df["sigma"].values
    scores = df["normalized_score"].values
    baselines = df["random_baseline"].values

    ax.plot(sigmas, scores,
            color="tomato", linewidth=2.0,
            marker="o", markersize=7,
            label="Adversary (weighted)")
    ax.plot(sigmas, baselines,
            color="gray", linewidth=1.5,
            linestyle="--", marker="s", markersize=5,
            label="Random baseline")
    ax.fill_between(sigmas, baselines, scores,
                    alpha=0.12, color="tomato",
                    label="Advantage over random")

    ax.set_ylim(0, 1.05)
    _style_ax(
        ax,
        xlabel="Noise level $\\sigma$",
        ylabel="Normalized weighted score",
        title="(a) Adversary Score vs Noise Level",
    )
    ax.legend(fontsize=10)
    ax.annotate(
        "Score $\\rightarrow$ baseline as $\\sigma$ increases",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8),
    )


def plot_h_plus_win_rate(ax, df: pd.DataFrame):
    """
    Fig 3b: H+ win rate vs sigma.
    Shows how often the likelihood ratio test correctly identifies
    the true frame indices as the member hypothesis.
    """
    sigmas = df["sigma"].values
    win_rates = df["h_plus_win_rate"].values

    ax.plot(sigmas, win_rates,
            color="steelblue", linewidth=2.0,
            marker="o", markersize=7,
            label="H$^+$ win rate")
    ax.axhline(y=0.5, color="gray", linestyle="--",
               linewidth=1.5, label="Random guess (0.5)")

    ax.set_ylim(0, 1.05)
    _style_ax(
        ax,
        xlabel="Noise level $\\sigma$",
        ylabel="H$^+$ win rate",
        title="(b) Likelihood Ratio Test Win Rate",
    )
    ax.legend(fontsize=10)


def plot_mean_delta(ax, df: pd.DataFrame):
    """
    Fig 3c: Mean delta vs sigma with std error bars.
    Delta = (||o - o-||² - ||o - o+||²) / (2σ).
    Positive delta means o is closer to H+ simulation.
    """
    sigmas = df["sigma"].values
    deltas = df["mean_delta"].values
    stds = df["std_delta"].values

    ax.errorbar(
        sigmas, deltas,
        yerr=stds,
        color="seagreen", linewidth=2.0,
        marker="o", markersize=7,
        capsize=5, capthick=1.5,
        label="Mean $\\Delta$ ± std",
    )
    ax.axhline(y=0.0, color="gray", linestyle="--",
               linewidth=1.5, label="$\\Delta = 0$ (indistinguishable)")

    _style_ax(
        ax,
        xlabel=r"Noise level ($\sigma$)",
        ylabel=r"Mean ($\Delta$)",
        title=r"Likelihood Ratio Signal $\Delta$ vs Noise",
    )
    ax.legend(fontsize=10)
    ax.annotate(
        "$\\Delta > 0$: H$^+$ distinguishable\n"
        "$\\Delta \\to 0$: obfuscation effective",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", alpha=0.9),
    )


def plot_score_distributions(ax, df: pd.DataFrame, num_frames: int = 16):
    """
    Fig 3d: Box plots of weighted score distributions per sigma.
    Shows spread of per-target scores, not just the mean.
    Note: requires per-target scores stored in attack_results_full.npz.
    Falls back to a bar chart of means if full data not available.
    """
    sigmas = df["sigma"].values
    means = df["mean_weighted_score"].values

    # bar chart of means with reference line
    x = np.arange(len(sigmas))
    bars = ax.bar(
        x, means,
        color="steelblue", alpha=0.7,
        width=0.5, label="Mean weighted score",
    )
    ax.axhline(y=num_frames * df["random_baseline"].iloc[0],
               color="gray", linestyle="--",
               linewidth=1.5, label="Random baseline")
    ax.axhline(y=num_frames,
               color="tomato", linestyle=":",
               linewidth=1.5, label=f"Max ({num_frames})")

    # value labels on bars
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.2f}" for s in sigmas])
    ax.set_ylim(0, num_frames * 1.15)
    _style_ax(
        ax,
        xlabel="Noise level $\\sigma$",
        ylabel=f"Weighted score (out of {num_frames})",
        title="(d) Weighted Score Distribution per $\\sigma$",
    )
    ax.legend(fontsize=10)


# ----------------------------
# Main combined figure
# ----------------------------

def plot_all(
    df: pd.DataFrame,
    num_frames: int = 16,
    save_path: str = "fig_attack_results.png",
):
    """
    Combined 2×2 figure with all four panels.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_normalized_score(ax1, df, num_frames)
    plot_h_plus_win_rate(ax2, df)
    plot_mean_delta(ax3, df)
    plot_score_distributions(ax4, df, num_frames)

    fig.suptitle(
        "Weighted Membership Inference Attack Results on UCF-101\n"
        r"($T_X(X) = \Pi_1 M X W + B$, grayscale frames, $d_0 = 112 \times 112$)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_individual(
    df: pd.DataFrame,
    num_frames: int = 16,
    prefix: str = "fig_attack",
):
    """
    Save each panel as a separate figure for flexible use in the paper.
    """
    panels = [
        ("normalized_score", plot_normalized_score, (df, num_frames)),
        ("h_plus_win_rate", plot_h_plus_win_rate, (df,)),
        ("mean_delta", plot_mean_delta, (df,)),
        ("score_dist", plot_score_distributions, (df, num_frames)),
    ]

    for name, fn, args in panels:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fn(ax, *args)
        path = f"{prefix}_{name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


# ----------------------------
# Entry point
# ----------------------------

if __name__ == "__main__":
    NUM_FRAMES = 16

    print("Loading results...")
    df, npz = load_results(
        csv_path="attack_results_10_trials/attack_results.csv",
        npz_path="attack_results_10_trials/attack_results_full.npz",
    )

    # print formatted table
    print_results_table(df, num_frames=NUM_FRAMES)

    # mean delta figure only
    print("Generating mean delta figure...")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_mean_delta(ax, df)
    plt.tight_layout()
    save_path = "fig_attack_mean_delta.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

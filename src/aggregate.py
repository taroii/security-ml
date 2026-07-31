"""Collate per-cell CSVs into the paper's summary tables and the Pareto figure.

Everything here is a pure join/reshape over artifacts already written by
membership_inference.py and main_video.py -- no mechanism or scoring logic
lives in this file.

  attacks   results/attack_k*_sigma*.csv       -> merged_results.csv
            Backs Table 2 (main paper) and Table 8 (Appendix F.1).

  accuracy  accuracy_results/acc_k*_sigma*.csv -> merged_accuracy.csv
            Backs Table 3 (Appendix E.2). When several seeds are present it also
            writes merged_accuracy_by_seed.csv with mean/std/SEM per cell.

  pareto    joins the two above                -> results_revision/
                                                  privacy_utility_pareto.csv
                                                  + images/fig_privacy_utility.pdf
            Backs Table 6 and Figure 5 (Appendix E.5): uninformed
            adversary leakage against downstream accuracy, with the
            no-obfuscation point as the reference.

Usage:
    python src/aggregate.py                 # all three
    python src/aggregate.py --mode attacks
    python src/aggregate.py --mode accuracy
    python src/aggregate.py --mode pareto
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

RESULTS_DIR = "./results"
ACCURACY_DIR = "./accuracy_results"
REVISION_DIR = "./results_revision"
IMG_DIR = "./images"

# The no-obfuscation reference point. Leakage is the k=0 frame-level score at
# sigma=0 (the mechanism is the identity there, so it is the un-obfuscated
# adversary); accuracy is the baseline column every accuracy CSV carries.
NO_OBF_LEAKAGE = 0.835
NO_OBF_ACCURACY = 60.19

# The paper's Attack 2 is the code's attack3 column; see README.md.
LEAKAGE_COL = "attack3_score_half"


def _load_cells(pattern, label):
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"No files matching {pattern}. Run the {label} sweep first.")
    print(f"Found {len(paths)} per-cell CSVs for {label}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def merge_attacks(results_dir=RESULTS_DIR, output="./merged_results.csv"):
    merged = _load_cells(os.path.join(results_dir, "attack_k*_sigma*.csv"),
                         "attacks")
    merged = merged.sort_values(["k", "sigma"]).reset_index(drop=True)
    merged.to_csv(output, index=False, float_format="%.6f")
    print(f"\nMerged ({len(merged)} rows) -> {output}")
    print(merged.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return merged


def merge_accuracy(results_dir=ACCURACY_DIR, output="./merged_accuracy.csv"):
    merged = _load_cells(os.path.join(results_dir, "acc_k*_sigma*.csv"),
                         "accuracy")
    merged = merged.sort_values(["k", "sigma", "seed"]).reset_index(drop=True)
    merged.to_csv(output, index=False, float_format="%.4f")
    print(f"\nMerged ({len(merged)} rows) -> {output}")
    print(merged.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Per-cell summary across seeds. With one seed per cell there is no
    # variance to report; with several this is the mean +- std the utility
    # table would quote, plus the standard error of the mean over seeds.
    if int(merged.groupby(["k", "sigma"])["seed"].nunique().max()) > 1:
        summary = (
            merged.groupby(["k", "sigma"])["accuracy_obfuscated"]
            .agg(n_seeds="count", mean="mean", std=lambda s: s.std(ddof=1))
            .reset_index()
        )
        summary["sem"] = summary["std"] / summary["n_seeds"] ** 0.5
        path = output.replace(".csv", "_by_seed.csv")
        summary.to_csv(path, index=False, float_format="%.4f")
        print(f"\nPer-cell summary over seeds -> {path}")
        print(summary.to_string(index=False,
                                float_format=lambda x: f"{x:.4f}"))
    else:
        print("\n[note] one seed per cell; no across-seed variance to report.")
    return merged


def build_pareto(results_dir=RESULTS_DIR, accuracy_dir=ACCURACY_DIR,
                 out_dir=REVISION_DIR, img_dir=IMG_DIR):
    """Join leakage and accuracy onto a common axis (Table 6, Appendix E.5).

    `leak_red` is the relative reduction against the no-obfuscation leakage and
    `acc_gap` the absolute gain over the no-obfuscation accuracy, so a cell
    Pareto-dominates the deploy-nothing baseline exactly when both are
    positive.
    """
    attacks = _load_cells(os.path.join(results_dir, "attack_k*_sigma*.csv"),
                          "attacks")
    accuracy = _load_cells(os.path.join(accuracy_dir, "acc_k*_sigma*.csv"),
                           "accuracy")

    # One row per (k, sigma); average over seeds on the accuracy side.
    acc = (accuracy.groupby(["k", "sigma"])["accuracy_obfuscated"]
           .mean().reset_index())
    leak = attacks[["k", "sigma", LEAKAGE_COL]].rename(
        columns={LEAKAGE_COL: "leak"})

    df = leak.merge(acc, on=["k", "sigma"], how="inner")
    df = df.rename(columns={"accuracy_obfuscated": "acc"})
    df["leak_red"] = (NO_OBF_LEAKAGE - df["leak"]) / NO_OBF_LEAKAGE * 100.0
    df["acc_gap"] = df["acc"] - NO_OBF_ACCURACY
    df = df.sort_values(["k", "sigma"]).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "privacy_utility_pareto.csv")
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"\nPareto table -> {path}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    dominant = df[(df["leak"] < NO_OBF_LEAKAGE) & (df["acc_gap"] > 0)]
    if len(dominant):
        best = dominant.loc[dominant["leak"].idxmin()]
        print(f"\nPareto-dominant cell: k={int(best['k'])}, "
              f"sigma={best['sigma']:.2f} -- leakage "
              f"{NO_OBF_LEAKAGE:.3f} -> {best['leak']:.3f} "
              f"({best['leak_red']:.1f}% lower) and accuracy "
              f"{NO_OBF_ACCURACY:.2f} -> {best['acc']:.2f} "
              f"({best['acc_gap']:+.2f} pp)")

    _plot_pareto(df, img_dir)
    return df


def _plot_pareto(df, img_dir=IMG_DIR):
    """Privacy--utility frontier (Figure 5, Appendix E.5).

    Accuracy on the x-axis, leakage on the y-axis (lower is more private), so
    the desirable direction is down-and-right and a point Pareto-dominates the
    no-obfuscation star exactly when it lies below and to the right of it. The
    grey line traces the k=1 series, which contains the dominant cell.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(img_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    style = {0: ("#c0392b", "o"), 1: ("#2c3e50", "s"), 5: ("#27ae60", "^")}

    frontier = df[df["k"] == 1].sort_values("sigma", ascending=False)
    ax.plot(frontier["acc"], frontier["leak"], "-", color="gray",
            lw=1.0, alpha=0.8, zorder=1)

    for k, g in df.groupby("k"):
        col, mark = style.get(int(k), ("#888", "o"))
        ax.scatter(g["acc"], g["leak"], color=col, marker=mark, s=42,
                   zorder=3, label=f"$k={int(k)}$")
        for _, r in g.iterrows():
            ax.annotate(f"$\\sigma={r['sigma']:g}$", (r["acc"], r["leak"]),
                        fontsize=6, xytext=(4, 3),
                        textcoords="offset points", color="#444")

    ax.scatter([NO_OBF_ACCURACY], [NO_OBF_LEAKAGE], color="black", marker="*",
               s=220, zorder=4, label="no obfuscation")
    ax.axvline(NO_OBF_ACCURACY, color="gray", ls="--", lw=0.9, alpha=0.8,
               label="utility baseline")

    ax.set_xlabel("test accuracy (%)")
    ax.set_ylabel("Attack 2 leakage (uninformed)")
    ax.set_title("Privacy--utility frontier (UCF-101)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    p = os.path.join(img_dir, "fig_privacy_utility.pdf")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"Figure -> {p}")


MODES = {
    "attacks":  merge_attacks,
    "accuracy": merge_accuracy,
    "pareto":   build_pareto,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=list(MODES) + ["all"], default="all")
    args = parser.parse_args()

    for name in (list(MODES) if args.mode == "all" else [args.mode]):
        print(f"\n########## {name} ##########")
        MODES[name]()

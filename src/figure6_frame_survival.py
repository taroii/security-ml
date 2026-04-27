"""Appendix B: Effect of Mixing Intensity on Attack Difficulty.

  1. Closed-form frame-survival table (stdout + CSV). Columns: k,
     E[target appearances per release], per-appearance weight 1/(2k),
     total expected signal mass. Rows for k in {0, 1, 2, 3, 5, 10}. For
     k >= 1, signal mass = 2ck/n_0 * 1/(2k) = c/n_0 is independent of k.

  2. Empirical k-sweep at fixed sigma=0.05 using the temporal MIA
     three-attack scores (Attack 3 frame-level MIA half-integer; Attack 2
     clip inference top-1). Loaded from
     results/attack_k{0,1,5}_sigma0.0500.csv produced by the rewritten
     membership_inference.py.

The juxtaposition is the point: signal mass is roughly constant in k,
yet attack accuracy still drops with k -- because spreading the signal
across more samples compounds the masking from Pi_1 and +B.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    OUT_PDF = os.path.join(PROJECT_ROOT, "images", "figure6_frame_survival.pdf")
    OUT_CSV = os.path.join(RESULTS_DIR, "figure6_survival_table.csv")

    C = 101
    N_PER_CLASS = 50
    SIGMA = 0.05
    K_TABLE = [0, 1, 2, 3, 5, 10]
    K_AVAILABLE = [0, 1, 5]

    # ---- 1. Closed-form frame-survival table ----
    table_rows = []
    for k in K_TABLE:
        if k == 0:
            # No mixing: target released once at full signal weight.
            app, weight, mass = 1.0, 1.0, 1.0
        else:
            app = 2.0 * C * k / N_PER_CLASS
            weight = 1.0 / (2 * k)
            mass = app * weight
        table_rows.append({
            "k": k,
            "expected_appearances": app,
            "per_appearance_weight": weight,
            "total_signal_mass": mass,
        })
    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(OUT_CSV, index=False, float_format="%.4f")

    print(f"Target-clip frame survival (closed-form; c={C}, n_0={N_PER_CLASS})")
    print(f"{'k':>4}  {'E[apps]':>8}  {'weight':>10}  {'signal mass':>12}")
    print("-" * 40)
    for row in table_rows:
        if row["k"] == 0:
            weight_str = "1.0000"
        else:
            weight_str = f"1/{2 * row['k']}"
        print(
            f"{row['k']:>4}  "
            f"{row['expected_appearances']:>8.2f}  "
            f"{weight_str:>10}  "
            f"{row['total_signal_mass']:>12.4f}"
        )
    print()
    print(
        f"For k >= 1, signal mass = 2ck/n_0 * 1/(2k) = c/n_0 = "
        f"{C / N_PER_CLASS:.2f}, independent of k."
    )
    print(
        "The ablation below shows H^+ win rate drops with k anyway, because "
        "signal spreads across more samples and the masking from Pi_1 + B "
        "compounds."
    )
    print(f"Table CSV -> {OUT_CSV}")
    print()

    # ---- 2. k-sweep ablation at fixed sigma ----
    sigma_str = f"{SIGMA:.4f}"
    abl_rows = []
    for k in K_AVAILABLE:
        csv_path = os.path.join(
            RESULTS_DIR, f"attack_k{k}_sigma{sigma_str}.csv"
        )
        if not os.path.exists(csv_path):
            print(f"  Missing: {csv_path}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        n_tgt = int(df["n_targets"].iloc[0])
        abl_rows.append({
            "k": k,
            "attack2_top1":          float(df["attack2_top1"].iloc[0]),
            "attack2_top1_se":       float(df["attack2_top1_std"].iloc[0]) / np.sqrt(n_tgt),
            "attack3_score_half":    float(df["attack3_score_half"].iloc[0]),
            "attack3_score_half_se": float(df["attack3_score_half_std"].iloc[0]) / np.sqrt(n_tgt),
        })

    if not abl_rows:
        print(
            f"No attack results found for sigma={SIGMA}. "
            f"Run `membership_inference.py --k {{0,1,5}}` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    abl = pd.DataFrame(abl_rows).sort_values("k").reset_index(drop=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.errorbar(
        abl["k"], abl["attack2_top1"], yerr=abl["attack2_top1_se"],
        marker="o", capsize=4, color="steelblue", linewidth=1.5,
    )
    n_clips_baseline = 1.0 / max(1, int(pd.read_csv(
        os.path.join(RESULTS_DIR, f"attack_k{K_AVAILABLE[0]}_sigma{sigma_str}.csv")
    )["n_universe_clips"].iloc[0]))
    ax.axhline(
        n_clips_baseline, linestyle="--", color="gray", linewidth=1,
        label=f"chance ($1/|U|$)",
    )
    ax.set_xlabel("$k$ (mixing parameter)")
    ax.set_ylabel("Clip-inference top-1 (Attack 2)")
    ax.set_title("(a) Attack 2: clip identification")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(abl["k"])
    ax.legend(loc="lower left")

    ax = axes[1]
    ax.errorbar(
        abl["k"], abl["attack3_score_half"], yerr=abl["attack3_score_half_se"],
        marker="s", capsize=4, color="darkorange", linewidth=1.5,
    )
    ax.set_xlabel("$k$ (mixing parameter)")
    ax.set_ylabel("Frame-level MIA score (half-integer)")
    ax.set_title("(b) Attack 3: frame-level recovery")
    ax.set_xticks(abl["k"])
    upper = max(abl["attack3_score_half"].max() * 1.3, 0.2)
    ax.set_ylim(0.0, upper)

    fig.suptitle(
        "Effect of mixing intensity $k$ on attack difficulty "
        f"(fixed $\\sigma={SIGMA}$; error bars $\\pm 1$ SE)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUT_PDF, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_PDF}")

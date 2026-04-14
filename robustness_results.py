"""
Generates two plots illustrating the weighted inference framework
from "Learnable Obfuscation for Temporally Related Video Data":

Plot 1: Prior success probability 1 - delta_0^{>=j} vs threshold j,
        comparing integer weights (exact match only) vs half-integer
        weights (partial credit for temporally proximate frames).
        Based on Lemma 2 (integer) and Lemma 3 (half-integer) of the paper.

Plot 2: Required noise scaling ratio C_half / C_int vs universe size N,
        showing how much additional noise is needed to maintain equivalent
        privacy guarantees under half-integer weighted inference.
        Based on the noise calibration condition (Eq. 16) of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import lgamma
from typing import List


# ----------------------------
# Vectorized log-binomial coefficients
# ----------------------------

def log_comb_vec(n: int, k_arr: np.ndarray) -> np.ndarray:
    """
    Vectorized log C(n, k) for an array of k values.
    Returns -inf where k < 0 or k > n.
    """
    from scipy.special import gammaln
    result = np.full(k_arr.shape, -np.inf)
    valid  = (k_arr >= 0) & (k_arr <= n)
    k_v    = k_arr[valid]
    result[valid] = (
        gammaln(n + 1)
        - gammaln(k_v + 1)
        - gammaln(n - k_v + 1)
    )
    return result


# ----------------------------
# Integer-weight prior (vectorized)
# ----------------------------

def integer_prior_all_j(N: int, n: int) -> np.ndarray:
    """
    Returns array of 1 - delta_0^{>=j} for j = 1, ..., n
    under integer weights (exact membership only).

    P(overlap >= j) where overlap ~ Hypergeometric(N, n, n).
    We compute the full PMF and take the survival function.

    Returns: shape (n,) array indexed by j-1
    """
    from scipy.stats import hypergeom
    # X ~ Hypergeom(M=N, n=n, N=n) — drawing n from N, n are successes
    rv   = hypergeom(M=N, n=n, N=n)
    pmf  = rv.pmf(np.arange(0, n + 1))          # P(overlap = l), l=0..n
    # survival: P(overlap >= j) = sum_{l=j}^{n} pmf[l]
    sf   = np.flip(np.cumsum(np.flip(pmf)))      # sf[j] = P(>=j)
    return sf[1:]                                 # j=1..n


# ----------------------------
# Half-integer prior (vectorized, Lemma 3)
# ----------------------------

def half_integer_prior_all_j(
    N: int,
    n: int,
    m: int,
) -> np.ndarray:
    """
    Returns array of 1 - delta_0^{>=j} for j = 0.5, 1.0, 1.5, ..., n
    under half-integer weights (Lemma 3).

    U partitioned into:
      A: n elements with weight 1   (exact members)
      B: m elements with weight 1/2 (proximate frames)
      C: N-n-m elements with weight 0

    Weighted overlap W = a + b/2 where (a, b) ~ MultivariateHypergeometric.

    Strategy: build full joint PMF P[a, b] over all valid (a, b) pairs
    using vectorized log-space arithmetic, then for each threshold j
    sum over pairs satisfying a + b/2 >= j.

    Returns: shape (2n,) array for j = 0.5, 1.0, ..., n  (step 0.5)
    """
    from scipy.special import gammaln

    N_C      = N - n - m
    log_denom = gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)

    # enumerate all valid (a, b) pairs
    a_vals = np.arange(0, n + 1)
    b_vals = np.arange(0, m + 1)
    A, B   = np.meshgrid(a_vals, b_vals, indexing="ij")  # (n+1, m+1)
    C_mat  = n - A - B                                    # (n+1, m+1)

    # mask invalid entries
    valid = (C_mat >= 0) & (C_mat <= N_C) & (B <= m)

    # log probabilities in log space
    log_p = np.full(A.shape, -np.inf)
    Av, Bv, Cv = A[valid], B[valid], C_mat[valid]

    log_p[valid] = (
        gammaln(n   + 1) - gammaln(Av + 1) - gammaln(n   - Av + 1)
      + gammaln(m   + 1) - gammaln(Bv + 1) - gammaln(m   - Bv + 1)
      + gammaln(N_C + 1) - gammaln(Cv + 1) - gammaln(N_C - Cv + 1)
      - log_denom
    )

    # weighted overlap for each (a, b) cell
    W = A + 0.5 * B                                       # (n+1, m+1)

    # thresholds: j = 0.5, 1.0, 1.5, ..., n  →  2n values
    thresholds = np.arange(1, 2 * n + 1) * 0.5           # (2n,)

    # for each threshold, sum log_p over cells where W >= threshold
    probs = np.zeros(len(thresholds))
    p_mat = np.where(valid, np.exp(np.clip(log_p, -700, 0)), 0.0)

    for idx, thresh in enumerate(thresholds):
        mask         = W >= thresh - 1e-9               # numerical tolerance
        probs[idx]   = p_mat[mask].sum()

    return probs                                         # shape (2n,)


# ----------------------------
# C = sum_j 1/D_j  where D_j = -log(1 - delta_0^{>=j})
# ----------------------------

def compute_C(prior_geq: np.ndarray) -> float:
    """
    C = sum_j 1/D_j  where D_j = -log(1 - delta_0^{>=j}).
    Clips probabilities away from 0 and 1 for numerical stability.
    """
    p      = np.clip(prior_geq, 1e-12, 1.0 - 1e-12)
    D      = -np.log(1.0 - p)
    valid  = D > 1e-12
    return float(np.sum(1.0 / D[valid]))


# ----------------------------
# Plot 1: Prior success probability vs threshold j
# ----------------------------

def plot_prior_success(
    N: int         = 1000,
    n: int         = 50,
    m: int         = 100,
    save_path: str = "fig_prior_success.png",
):
    """
    Plot 1: 1 - delta_0^{>=j} vs j, integer vs half-integer weights.

    Left panel:  linear scale, zoomed to low-j region
    Right panel: log scale showing full tail behavior
    """
    print(f"  Computing integer-weight priors (N={N}, n={n})...")
    prior_int  = integer_prior_all_j(N, n)          # shape (n,)
    j_int      = np.arange(1, n + 1, dtype=float)

    print(f"  Computing half-integer-weight priors (m={m})...")
    prior_half = half_integer_prior_all_j(N, n, m)  # shape (2n,)
    j_half     = np.arange(1, 2 * n + 1) * 0.5

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- left panel: linear scale, zoom to j <= 20 ---
    ax = axes[0]
    zoom = 20
    ax.step(j_int[j_int <= zoom],
            prior_int[j_int <= zoom],
            where="post", color="steelblue", linewidth=2.0,
            label="Integer weights (exact match)")
    ax.step(j_half[j_half <= zoom],
            prior_half[j_half <= zoom],
            where="post", color="tomato", linewidth=2.0, linestyle="--",
            label="Half-integer weights")
    ax.set_xlabel("Threshold $j$", fontsize=13)
    ax.set_ylabel(r"$1 - \delta_0^{\geq j}$  (prior success prob.)",
                  fontsize=13)
    ax.set_title("Prior Success Probability vs Threshold", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, zoom)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"$N={N},\\ n={n},\\ m={m}$\n$q = n/N = {n/N:.3f}$",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    # --- right panel: log scale, full range ---
    ax2 = axes[1]
    mask_int  = prior_int  > 1e-15
    mask_half = prior_half > 1e-15
    ax2.semilogy(j_int[mask_int],   prior_int[mask_int],
                 color="steelblue", linewidth=2.0,
                 marker="o", markersize=3,
                 label="Integer weights")
    ax2.semilogy(j_half[mask_half], prior_half[mask_half],
                 color="tomato", linewidth=2.0,
                 linestyle="--", marker="s", markersize=3,
                 label="Half-integer weights")
    ax2.set_xlabel("Threshold $j$", fontsize=13)
    ax2.set_ylabel(r"$1 - \delta_0^{\geq j}$  (log scale)", fontsize=13)
    ax2.set_title("Prior Success Probability (Log Scale)", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.annotate(
        "Half-integer curve lies strictly\nabove integer curve at all $j$",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9),
    )

    plt.suptitle(
        "Lemma 3: Half-Integer Weights Increase Adversary Prior Success\n"
        r"(larger $1-\delta_0^{\geq j}$ $\Rightarrow$ harder to protect privacy)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ----------------------------
# Plot 2: Noise ratio C_half / C_int vs N (vectorized)
# ----------------------------

def plot_noise_ratio(
    n:            int         = 100,
    m_fractions:  List[float] = [0.05, 0.10, 0.20],
    N_range:      np.ndarray  = None,
    save_path:    str         = "fig_noise_ratio.png",
):
    """
    Plot 2: C_half / C_int vs universe size N (fixed n=100).

    Shows how much additional noise is required to maintain equivalent
    privacy under half-integer weighted inference as N grows (q = n/N
    decreases). Each curve corresponds to a different half-weight
    fraction m/N.
    """
    if N_range is None:
        N_range = np.array([200, 400, 600, 800, 1000, 1500, 2000])

    colors     = ["steelblue", "tomato", "seagreen"]
    linestyles = ["-", "--", "-."]
    markers    = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for color, ls, marker, m_frac in zip(colors, linestyles, markers, m_fractions):
        ratios  = []
        N_valid = []

        for N in N_range:
            if N <= n:
                continue
            m = int(m_frac * N)
            if n + m > N:
                continue

            p_int  = integer_prior_all_j(N, n)
            C_int  = compute_C(p_int)

            p_half = half_integer_prior_all_j(N, n, m)
            C_half = compute_C(p_half)

            if C_int > 1e-12:
                ratios.append(C_half / C_int)
                N_valid.append(N)
                print(f"  N={N:5d}, m/N={m_frac*100:.0f}%: "
                      f"C_int={C_int:.4f}, C_half={C_half:.4f}, "
                      f"ratio={C_half/C_int:.4f}")

        if ratios:
            ax.plot(
                N_valid, ratios,
                color=color, linewidth=2.0,
                linestyle=ls, marker=marker, markersize=7,
                label=f"$m/N = {m_frac*100:.0f}\\%$",
            )

    ax.axhline(y=1.0, color="black", linestyle=":",
               linewidth=1.5, label="Baseline (no extra noise)")
    ax.set_xlabel("Universe size $N$", fontsize=13)
    ax.set_ylabel(
        r"$C_{\mathrm{half}} / C_{\mathrm{int}}$",
        fontsize=13,
    )
    ax.set_title(
        f"Noise Scaling Ratio vs Universe Size  ($n = {n}$)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # secondary x-axis showing q = n/N
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    N_ticks    = N_range[N_range > n]
    q_labels   = [f"{n/N:.3f}" for N in N_ticks]
    ax2.set_xticks(N_ticks)
    ax2.set_xticklabels(q_labels, fontsize=9)
    ax2.set_xlabel("Sampling rate $q = n/N$", fontsize=12)

    ax.annotate(
        "Larger $m/N$ $\\rightarrow$ more partial-credit frames\n"
        "$\\rightarrow$ higher adversary prior $\\rightarrow$ more noise needed",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

# ----------------------------
# Entry point
# ----------------------------

if __name__ == "__main__":
    # --- Plot 1 ---
    print("Generating Plot 1: prior success probability...")
    plot_prior_success(
        N         = 1000,
        n         = 50,
        m         = 100,
        save_path = "fig_prior_success.png",
    )

    # --- Plot 2 ---
    print("\nGenerating Plot 2: noise scaling ratio...")
    plot_noise_ratio(
        n           = 100,
        m_fractions = [0.05, 0.10, 0.20],
        N_range     = np.array([200, 400, 600, 800, 1000, 1500, 2000]),
        save_path   = "fig_noise_ratio.png",
    )

    print("\nDone.")
    print("  fig_prior_success.png")
    print("  fig_noise_ratio.png")
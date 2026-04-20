import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln


# Integer-weight prior (vectorized)
def integer_prior_all_j(N: int, n: int) -> np.ndarray:
    """
    Returns array of 1 - delta_0^{>=j} for j = 1, ..., n
    under integer weights (exact membership only).

    P(overlap >= j) where overlap ~ Hypergeometric(N, n, n).
    We compute the full PMF and take the survival function.

    Returns: shape (n,) array indexed by j-1
    """
    from scipy.stats import hypergeom
    rv = hypergeom(M=N, n=n, N=n)
    pmf = rv.pmf(np.arange(0, n + 1))
    sf = np.flip(np.cumsum(np.flip(pmf)))
    return sf[1:]


# Half-integer prior (vectorized, Lemma 3)
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

    Returns: shape (2n,) array for j = 0.5, 1.0, ..., n  (step 0.5)
    """
    N_C = N - n - m
    log_denom = gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)

    a_vals = np.arange(0, n + 1)
    b_vals = np.arange(0, m + 1)
    A, B = np.meshgrid(a_vals, b_vals, indexing="ij")
    C_mat = n - A - B

    valid = (C_mat >= 0) & (C_mat <= N_C) & (B <= m)

    log_p = np.full(A.shape, -np.inf)
    Av, Bv, Cv = A[valid], B[valid], C_mat[valid]

    log_p[valid] = (
        gammaln(n + 1) - gammaln(Av + 1) - gammaln(n - Av + 1)
        + gammaln(m + 1) - gammaln(Bv + 1) - gammaln(m - Bv + 1)
        + gammaln(N_C + 1) - gammaln(Cv + 1) - gammaln(N_C - Cv + 1)
        - log_denom
    )

    W = A + 0.5 * B

    thresholds = np.arange(1, 2 * n + 1) * 0.5

    probs = np.zeros(len(thresholds))
    p_mat = np.where(valid, np.exp(np.clip(log_p, -700, 0)), 0.0)

    for idx, thresh in enumerate(thresholds):
        mask = W >= thresh - 1e-9
        probs[idx] = p_mat[mask].sum()

    return probs


# Plot: Prior success probability vs threshold j (linear scale)
def plot_prior_success(
    N: int = 1000,
    n: int = 50,
    m: int = 100,
    save_path: str = "fig_prior_success.png",
):
    """
    Plot 1 - delta_0^{>=j} vs j, integer vs half-integer weights,
    linear scale, zoomed to low-j region.
    """
    print(f"  Computing integer-weight priors (N={N}, n={n})...")
    prior_int = integer_prior_all_j(N, n)
    j_int = np.arange(1, n + 1, dtype=float)

    print(f"  Computing half-integer-weight priors (m={m})...")
    prior_half = half_integer_prior_all_j(N, n, m)
    j_half = np.arange(1, 2 * n + 1) * 0.5

    fig, ax = plt.subplots(figsize=(7, 5))

    zoom = 20
    ax.step(j_int[j_int <= zoom],
            prior_int[j_int <= zoom],
            where="post", color="steelblue", linewidth=2.0,
            label="Integer weights")
    ax.step(j_half[j_half <= zoom],
            prior_half[j_half <= zoom],
            where="post", color="tomato", linewidth=2.0, linestyle="--",
            label="Half-integer weights")
    ax.set_xlabel("Threshold ($j$)", fontsize=13)
    ax.set_ylabel(r"Prior success probability ($1 - \delta_0^{\geq j}$)",
                  fontsize=13)
    # ax.set_title("Prior Success Probability vs Threshold", fontsize=13)
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

    # plt.suptitle(
    #     "Lemma 3: Half-Integer Weights Increase Adversary Prior Success\n"
    #     r"(larger $1-\delta_0^{\geq j}$ $\Rightarrow$ harder to protect privacy)",
    #     fontsize=13, fontweight="bold", y=1.02,
    # )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    plot_prior_success(
        N=1000,
        n=50,
        m=100,
        save_path="./images/fig1_prior_success.png",
    )
    print("\nDone.")

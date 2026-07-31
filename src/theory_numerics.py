"""Closed-form numerics for the paper's theory. No dataset required.

One module for every number in the paper that comes from a prior/counting
argument rather than from running the mechanism on video:

  calibration     main-paper Table 1 + images/table_calibration.tex.
                  Prior-dependent constants C under the integer rule, the
                  half-integer rule (Lemma 3), and the general 1/m-resolution
                  rule (Lemma S11, Appendix D), and the sigma inflation they
                  imply. This is the numerical validation of Theorem 4
                  (half-integer noise calibration) and Theorem 5 (general
                  1/m-resolution calibration).

  prior-figure    main-paper Figure 1. Prior success probability
                  1 - delta_0^{>=j} versus threshold j, integer vs.
                  half-integer weights, on the illustrative scenario
                  N=1000, n=50, m=100.

  frame-survival  Table 9 (Appendix F.2). Expected target-clip
                  signal mass under class-k mixing, showing it is
                  c/n_0 = 2.02 independently of k for every k >= 1.

All three share the same prior machinery, defined once below.

Calibration background: under the Xiao isotropic-noise MI bound
MI(sigma) ~ (d/2) tr(Sigma_X) / sigma^2 for large sigma,

    sigma_{1/m} / sigma_int  ~  sqrt(C_{1/m} / C_int)

so the tables report that ratio. The exact MI bound is data-dependent
(Theorem 2 of Xiao et al. 2024); the sqrt approximation is a defensible upper
bound when sigma dominates the data norm.

Usage:
    python src/theory_numerics.py                     # all three
    python src/theory_numerics.py --mode calibration
    python src/theory_numerics.py --mode prior-figure
    python src/theory_numerics.py --mode frame-survival
"""
import argparse
import os

import numpy as np
from scipy.special import gammaln
from scipy.stats import hypergeom

# ----------------------------------------------------------------------
# Fixed configuration. These are the values behind the reported numbers;
# they are constants rather than flags because every table and figure in
# the paper uses exactly this setting.
# ----------------------------------------------------------------------

# Illustrative scenario shared by main Table 1 and main Figure 1.
SCENARIO_N = 1000        # universe size
SCENARIO_n = 50          # truth-set size
SCENARIO_m = 100         # half-credit zone size

# Frame-survival (Table 9, Appendix F.2).
N_CLASSES = 101          # c
N_PER_CLASS = 50         # n_0
K_TABLE = [0, 1, 2, 3, 5, 10]

IMG_DIR = "./images"
OUT_TEX = os.path.join(IMG_DIR, "table_calibration.tex")
OUT_FIG = os.path.join(IMG_DIR, "fig1_prior_success.pdf")


# ----------------------------------------------------------------------
# Prior success probabilities
# ----------------------------------------------------------------------

def integer_prior_all_j(N: int, n: int) -> np.ndarray:
    """1 - delta_0^{>=j} for j = 1, ..., n under integer (exact) weights.

    The overlap of a uniform n-subset with the truth set is
    Hypergeometric(N, n, n); this is its survival function.
    """
    rv = hypergeom(M=N, n=n, N=n)
    pmf = rv.pmf(np.arange(0, n + 1))
    sf = np.flip(np.cumsum(np.flip(pmf)))
    return sf[1:]


def _half_integer_overlap_pmf(N: int, n: int, m: int):
    """Joint pmf of the weighted overlap W = a + b/2 under half-integer
    weights (Lemma 3).

    U is partitioned into the weight-1 set A (|A| = n, the exact members),
    the weight-1/2 set B (|B| = m, the proximate frames), and the weight-0
    remainder C. For a uniform n-subset the counts (a, b, c) drawn from
    (A, B, C) are multivariate hypergeometric.

    Returns (W, p_mat): the weighted-overlap value and its probability on the
    (a, b) grid.
    """
    N_C = N - n - m
    log_denom = gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)

    a_vals = np.arange(0, n + 1)
    b_vals = np.arange(0, m + 1)
    A, B = np.meshgrid(a_vals, b_vals, indexing="ij")
    Cm = n - A - B
    valid = (Cm >= 0) & (Cm <= N_C)

    log_p = np.full(A.shape, -np.inf)
    Av, Bv, Cv = A[valid], B[valid], Cm[valid]
    log_p[valid] = (
        gammaln(n + 1) - gammaln(Av + 1) - gammaln(n - Av + 1)
        + gammaln(m + 1) - gammaln(Bv + 1) - gammaln(m - Bv + 1)
        + gammaln(N_C + 1) - gammaln(Cv + 1) - gammaln(N_C - Cv + 1)
        - log_denom
    )
    p_mat = np.where(valid, np.exp(np.clip(log_p, -700, 0)), 0.0)
    return A + 0.5 * B, p_mat


def half_integer_prior_at_integer_j(N: int, n: int, m: int) -> np.ndarray:
    """1 - delta_0^{>=j} at INTEGER j = 1, ..., n under half-integer weights."""
    W, p_mat = _half_integer_overlap_pmf(N, n, m)
    out = np.zeros(n, dtype=np.float64)
    for idx, j in enumerate(range(1, n + 1)):
        out[idx] = p_mat[W >= j - 1e-9].sum()
    return out


def half_integer_prior_all_j(N: int, n: int, m: int) -> np.ndarray:
    """1 - delta_0^{>=j} on the HALF-INTEGER grid j = 0.5, 1.0, ..., n.

    Same distribution as `half_integer_prior_at_integer_j`, evaluated on the
    finer grid the figure plots (half-integer weights can land the adversary
    on half-integer thresholds, which is the point of the figure).
    """
    W, p_mat = _half_integer_overlap_pmf(N, n, m)
    thresholds = np.arange(1, 2 * n + 1) * 0.5
    probs = np.zeros(len(thresholds))
    for idx, thresh in enumerate(thresholds):
        probs[idx] = p_mat[W >= thresh - 1e-9].sum()
    return probs


def m_resolution_prior_at_integer_j(N: int, n: int, M_counts: list) -> np.ndarray:
    """1 - delta_0^{>=j} for INTEGER j = 1, ..., n under 1/m-resolution
    weights (Lemma S11).

    M_counts[r] = number of universe elements with weight r/m, for
    r = 0, 1, ..., m. M_counts[m] must equal n. The list length determines
    m (= len(M_counts) - 1).

    Brute-force sum over count-vectors (k_0, ..., k_m) with sum k_r = n and
    sum (r/m) k_r >= j. Tractable when n is small (n <= ~30).
    """
    m = len(M_counts) - 1
    assert sum(M_counts) == N, (
        f"M_counts must sum to N={N}, got {sum(M_counts)}"
    )
    assert M_counts[m] == n, (
        f"weight-1 class M_{m} must have size n={n}, got {M_counts[m]}"
    )

    log_denom = gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)
    log_C = np.array([gammaln(M + 1) for M in M_counts])
    out = np.zeros(n, dtype=np.float64)

    def recurse(r, remaining, weighted_overlap_so_far, log_p_so_far):
        if r == m:
            kr = remaining              # last class takes the rest
            if kr < 0 or kr > M_counts[r]:
                return
            log_p = (
                log_p_so_far
                + log_C[r] - gammaln(kr + 1) - gammaln(M_counts[r] - kr + 1)
                - log_denom
            )
            p = float(np.exp(np.clip(log_p, -700, 0)))
            w = weighted_overlap_so_far + (r / m) * kr
            for j_idx, j in enumerate(range(1, n + 1)):
                if w >= j - 1e-9:
                    out[j_idx] += p
            return
        for kr in range(0, min(remaining, M_counts[r]) + 1):
            log_p = (
                log_p_so_far
                + log_C[r] - gammaln(kr + 1) - gammaln(M_counts[r] - kr + 1)
            )
            recurse(r + 1, remaining - kr,
                    weighted_overlap_so_far + (r / m) * kr, log_p)

    recurse(0, n, 0.0, 0.0)
    return out


def C_from_priors(prior_at_int_j: np.ndarray) -> float:
    """C = sum_{j=1}^n 1/D_j with D_j = -log(1 - delta_0^{>=j}) = -log P(W >= j).

    `prior_at_int_j` holds 1 - delta_0^{>=j} = P(W >= j) for j = 1, ..., n, so
    D_j = -log(prior_at_int_j) directly.

    Entries with prior = 0 (thresholds the adversary can never reach by random
    guessing) give D_j = +inf and 1/D_j = 0 -- they contribute nothing and are
    dropped safely.
    """
    safe = np.clip(prior_at_int_j, 1e-300, 1.0)
    D = -np.log(safe)
    finite = (D > 0) & np.isfinite(D)
    return float(np.sum(1.0 / D[finite]))


# ----------------------------------------------------------------------
# calibration -- main-paper Table 1
# ----------------------------------------------------------------------

def _latex_table(rows, out_path: str, c_int: float) -> None:
    """Write a LaTeX tabular block: rule | C | C/C_int | sqrt(C/C_int)."""
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Weight rule & $C$ & $C / C_{\\text{int}}$ & "
        "$\\sigma / \\sigma_{\\text{int}}$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        ratio = r["C"] / c_int
        sigma_ratio = float(np.sqrt(r["C"] / c_int))
        lines.append(
            f"{r['rule']} & {r['C']:.4g} & {ratio:.4f} & {sigma_ratio:.4f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nLaTeX table -> {out_path}")


def run_calibration():
    N, n, m_half = SCENARIO_N, SCENARIO_n, SCENARIO_m

    print("=" * 66)
    print("Numerical validation of Theorems 4 and 5")
    print("=" * 66)
    print(f"Lemma 3 visualization scenario (matches Figure 1): "
          f"N = {N}, n = {n}, m = {m_half}.")
    print()

    rows = []

    print("Integer rule (exact match only):")
    p_int = integer_prior_all_j(N, n)
    C_int = C_from_priors(p_int)
    print(f"  C_int = {C_int:.6f}")
    rows.append({"rule": "Integer", "C": C_int})

    print(f"\nHalf-integer rule (Lemma 3, m = {m_half} half-credit positions):")
    p_half = half_integer_prior_at_integer_j(N, n, m_half)
    C_half = C_from_priors(p_half)
    print(f"  C_half = {C_half:.6f}")
    print(f"  ratio C_int / C_half       = {C_int / C_half:.6f}")
    print(f"  sigma_half / sigma_int (~) = {np.sqrt(C_half / C_int):.6f}")
    rows.append({"rule": "Half-integer", "C": C_half})

    # 1/4-resolution: split the half-credit zone into a closer half at weight
    # 2/4 and a farther half at weight 1/4. An illustrative choice -- Lemma S11
    # holds for any weight-class sizes.
    m_quarter_close = m_half // 2
    m_quarter_far = m_half - m_quarter_close
    print("\n1/4-resolution rule (Lemma S11):")
    print(f"  weight 1/4: {m_quarter_far},  weight 2/4: {m_quarter_close},  "
          f"weight 3/4: 0,  weight 1: {n}")
    M_counts_q = [N - n - m_half, m_quarter_far, m_quarter_close, 0, n]
    p_q = m_resolution_prior_at_integer_j(N, n, M_counts_q)
    C_q = C_from_priors(p_q)
    print(f"  C_{{1/4}} = {C_q:.6f}")
    print(f"  ratio C_int / C_{{1/4}}        = {C_int / C_q:.6f}")
    print(f"  sigma_{{1/4}} / sigma_int (~)  = {np.sqrt(C_q / C_int):.6f}")
    rows.append({"rule": "$1/4$-resolution", "C": C_q})

    print()
    print("=" * 66)
    print("Theorem 5 part (i): C_{1/m} > C_int strictly")
    print(f"  C_int   = {C_int:.4f}")
    print(f"  C_half  = {C_half:.4f}   ({C_half / C_int:.2f}x C_int)  -> "
          f"C_half > C_int: {'YES' if C_half > C_int else 'NO'}")
    print(f"  C_{{1/4}}  = {C_q:.4f}   ({C_q / C_int:.2f}x C_int)  -> "
          f"C_{{1/4}} > C_int: {'YES' if C_q > C_int else 'NO'}")
    print()
    print("Theorem 5 part (ii): sigma_{1/m} > sigma_int.")
    print("  Under Xiao MI bound MI(sigma) ~ const/sigma^2, the ratio is")
    print("  approximately sqrt(C_{1/m}/C_int):")
    print(f"  sigma_half / sigma_int ~ sqrt(C_half/C_int)   "
          f"= {np.sqrt(C_half / C_int):.4f}")
    print(f"  sigma_{{1/4}}  / sigma_int ~ sqrt(C_{{1/4}}/C_int)  "
          f"= {np.sqrt(C_q / C_int):.4f}")
    print()
    print("NOTE: Theorem 5 doesn't claim monotonicity in m -- only that")
    print("C_{1/m} > C_int strictly when intermediate weight classes are")
    print("non-empty. The numeric ordering between half and 1/4 depends on")
    print("the specific weight-class sizes (M_r) chosen.")

    _latex_table(rows, OUT_TEX, c_int=C_int)


# ----------------------------------------------------------------------
# prior-figure -- main-paper Figure 1
# ----------------------------------------------------------------------

def run_prior_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, n, m = SCENARIO_N, SCENARIO_n, SCENARIO_m

    print(f"  Computing integer-weight priors (N={N}, n={n})...")
    prior_int = integer_prior_all_j(N, n)
    j_int = np.arange(1, n + 1, dtype=float)

    print(f"  Computing half-integer-weight priors (m={m})...")
    prior_half = half_integer_prior_all_j(N, n, m)
    j_half = np.arange(1, 2 * n + 1) * 0.5

    fig, ax = plt.subplots(figsize=(7, 5))
    zoom = 20
    ax.step(j_int[j_int <= zoom], prior_int[j_int <= zoom],
            where="post", color="steelblue", linewidth=2.0,
            label="Integer weights")
    ax.step(j_half[j_half <= zoom], prior_half[j_half <= zoom],
            where="post", color="tomato", linewidth=2.0, linestyle="--",
            label="Half-integer weights")
    ax.set_xlabel("Threshold ($j$)", fontsize=13)
    ax.set_ylabel(r"Prior success probability ($1 - \delta_0^{\geq j}$)",
                  fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, zoom)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"$N={N},\\ n={n},\\ m={m}$\n$q = n/N = {n / N:.3f}$",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG) or ".", exist_ok=True)
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_FIG}")


# ----------------------------------------------------------------------
# frame-survival -- Table 9 (Appendix F.2)
# ----------------------------------------------------------------------

def run_frame_survival():
    """Expected target-clip signal mass under class-k mixing.

    For k >= 1 a target clip appears in expectation 2ck/n_0 mixed samples,
    each carrying weight 1/(2k) from the k-fold averaging, so the total
    signal mass is c/n_0 -- independent of k. A first-moment argument
    therefore predicts no variation in attack difficulty across k >= 1, which
    is what Table 9 reports.
    """
    print(f"Target-clip frame survival (closed-form; c={N_CLASSES}, "
          f"n_0={N_PER_CLASS})")
    print(f"{'k':>4}  {'E[apps]':>8}  {'weight':>10}  {'signal mass':>12}")
    print("-" * 40)
    for k in K_TABLE:
        if k == 0:
            # No mixing: the target is released once at full signal weight.
            app, weight, mass, weight_str = 1.0, 1.0, 1.0, "1.0000"
        else:
            app = 2.0 * N_CLASSES * k / N_PER_CLASS
            weight = 1.0 / (2 * k)
            mass = app * weight
            weight_str = f"1/{2 * k}"
        print(f"{k:>4}  {app:>8.2f}  {weight_str:>10}  {mass:>12.4f}")
    print()
    print(f"For k >= 1, signal mass = 2ck/n_0 * 1/(2k) = c/n_0 = "
          f"{N_CLASSES / N_PER_CLASS:.2f}, independent of k.")


MODES = {
    "calibration":    run_calibration,
    "prior-figure":   run_prior_figure,
    "frame-survival": run_frame_survival,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=list(MODES) + ["all"], default="all")
    args = parser.parse_args()

    for name in (list(MODES) if args.mode == "all" else [args.mode]):
        print(f"\n########## {name} ##########")
        MODES[name]()

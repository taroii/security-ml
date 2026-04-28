"""Numerical validation of Theorems 8 and 9.

Given the temporal MIA setup (universe U of (clip, index) pairs, truth set
of size n = T frames per target, half-credit zone m of size clip_len-T
within the target's clip), compute the prior-dependent constants

    D_j := -log(1 - delta_0^{>=j})
    C   := sum_{j=1}^{n} 1 / D_j

under the integer rule, the half-integer rule (Lemma 3), and the general
1/m-resolution rule (Lemma 4) for a few values of m.

Theorem 9 part (i) predicts C_{1/m} > C_int strictly when at least one
intermediate weight class is non-empty. Theorem 9 part (ii) gives the
σ-calibration condition

    MI(σ_{1/m}) <= MI(σ_int) * C_int / C_{1/m} - log(2)/n

Under the Xiao isotropic-noise MI bound MI(σ) ~ (d/2) tr(Σ_X) / σ² for
large σ, this gives

    σ_{1/m} / σ_int  ~  sqrt(C_{1/m} / C_int)

so the table also prints that ratio. The exact MI bound is
data-dependent (Theorem 2 of Xiao et al. 2024); the sqrt approximation
is a defensible upper bound when σ dominates the data norm.

Output: stdout summary + LaTeX tabular block written to a file.
"""
import argparse
import os

import numpy as np
from scipy.special import gammaln


def integer_prior_all_j(N: int, n: int) -> np.ndarray:
    """1 - delta_0^{>=j} for j = 1, ..., n under integer (exact) weights."""
    from scipy.stats import hypergeom
    rv = hypergeom(M=N, n=n, N=n)
    pmf = rv.pmf(np.arange(0, n + 1))
    sf = np.flip(np.cumsum(np.flip(pmf)))
    return sf[1:]


def half_integer_prior_at_integer_j(N: int, n: int, m: int) -> np.ndarray:
    """1 - delta_0^{>=j} for INTEGER j = 1, ..., n under half-integer weights
    (Lemma 3).  Sums the multivariate-hypergeometric pmf of (a, b) over
    {a + b/2 >= j}."""
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
    W = A + 0.5 * B  # weighted overlap

    out = np.zeros(n, dtype=np.float64)
    for idx, j in enumerate(range(1, n + 1)):
        out[idx] = p_mat[W >= j - 1e-9].sum()
    return out


def m_resolution_prior_at_integer_j(
    N: int,
    n: int,
    M_counts: list,
) -> np.ndarray:
    """1 - delta_0^{>=j} for INTEGER j = 1, ..., n under 1/m-resolution
    weights (Lemma 4).

    M_counts[r] = number of universe elements with weight r/m, for
    r = 0, 1, ..., m. M_counts[m] must equal n. The list length
    determines m (= len(M_counts) - 1).

    Brute-force sum over count-vectors (k_0, ..., k_m) with
    sum k_r = n and sum (r/m) k_r >= j. Tractable when n is small
    (n <= ~30).
    """
    m = len(M_counts) - 1
    assert sum(M_counts) == N, (
        f"M_counts must sum to N={N}, got {sum(M_counts)}"
    )
    assert M_counts[m] == n, (
        f"weight-1 class M_{m} must have size n={n}, got {M_counts[m]}"
    )

    log_denom = gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)
    log_C = np.array([
        gammaln(M + 1) for M in M_counts
    ])

    out = np.zeros(n, dtype=np.float64)

    # iterate over compositions (k_0, ..., k_m) with sum = n
    def recurse(r, remaining, ks_so_far, weighted_overlap_so_far,
                log_p_so_far):
        if r == m:
            # last class takes the rest
            kr = remaining
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
            recurse(
                r + 1,
                remaining - kr,
                ks_so_far + [kr],
                weighted_overlap_so_far + (r / m) * kr,
                log_p,
            )

    recurse(0, n, [], 0.0, 0.0)
    return out


def C_from_priors(prior_at_int_j: np.ndarray) -> float:
    """
    C = Σ_{j=1}^n 1/D_j with D_j = -log(1 - δ_0^{>=j}) = -log P(W >= j).

    `prior_at_int_j` is the array of 1 - δ_0^{>=j} = P(W >= j) values
    for j = 1, ..., n. So D_j = -log(prior_at_int_j) directly.

    Entries with prior = 0 (thresholds j the adversary can never reach
    by random guessing) give D_j = +inf and 1/D_j = 0 — they contribute
    nothing and we drop them safely.
    """
    safe = np.clip(prior_at_int_j, 1e-300, 1.0)
    D = -np.log(safe)
    finite = (D > 0) & np.isfinite(D)
    return float(np.sum(1.0 / D[finite]))


def latex_table(rows, out_path: str, c_int: float) -> None:
    """Write a LaTeX tabular block: rule | C | C_int/C | sqrt(C/C_int)."""
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
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nLaTeX table -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Numerical validation of Theorems 8 and 9: compute C under "
            "integer / half-integer / 1/m-resolution weights for the "
            "temporal MIA setup."
        )
    )
    # Per-target Attack 3 setup: universe = (clip, index) pairs;
    # truth set = target's T sampled frames; half-credit zone = same
    # clip's other (clip_len - T) indices.
    parser.add_argument("--scenario", type=str, default="figure1",
                        choices=["figure1", "attack1", "attack3"],
                        help="Which (N, n, m) to use. 'figure1' matches the "
                             "Lemma 3 visualization (N=1000, n=50, m=100). "
                             "'attack1' is the clip-given index inference "
                             "scenario (N=clip_len, n=T, m=window). "
                             "'attack3' is whole-universe (clip, index) "
                             "pair scoring (huge N, m=clip_len-T).")
    parser.add_argument("--n-clips", type=int, default=7954,
                        help="Number of clips in the universe (used by "
                             "scenario=attack3).")
    parser.add_argument("--clip-len", type=int, default=100,
                        help="Frames per clip after CLIP_LEN trim.")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="T = number of sampled frames per target.")
    parser.add_argument("--window", type=int, default=2,
                        help="+-window for half-credit zone in attack1 "
                             "scenario (default 2 = 2%% on clip_len=100).")
    parser.add_argument("--out-tex", type=str,
                        default="./images/table_calibration.tex")
    args = parser.parse_args()

    n = args.num_frames                         # truth-set size = T

    if args.scenario == "figure1":
        # Match Figure 1's illustrative parameters.
        N = 1000
        n = 50
        m_half = 100
        scenario_desc = (
            f"Lemma 3 visualization scenario (matches Figure 1): "
            f"N = {N}, n = {n}, m = {m_half}."
        )
    elif args.scenario == "attack1":
        N = args.clip_len                       # universe = candidate indices
        # Half-credit zone: union of +-window around each true index. For
        # uniformly random T true indices in [0, clip_len), expected size
        # is clip_len * (1 - (1 - (2*window+1)/clip_len)^T) approximately.
        # Use the worst case (no overlap) which is min(T*(2w+1) - T, ...).
        # Cleanest: enumerate one fixed sample to size m precisely is
        # situation-specific; pick T*2*window which assumes no overlaps
        # between adjacent windows AND excludes the T true indices themselves.
        m_half = min(N - n, args.num_frames * 2 * args.window)
        scenario_desc = (
            f"Attack 1 (clip given): N = clip_len = {N}, "
            f"n = T = {n}, m = T*2*window = {m_half} "
            f"(window=+-{args.window})"
        )
    else:
        N = args.n_clips * args.clip_len        # universe = (clip, idx) pairs
        m_half = args.clip_len - args.num_frames  # other indices in target's clip
        scenario_desc = (
            f"Attack 3 (whole universe): N = n_clips * clip_len = "
            f"{N:,}, n = T = {n}, m = clip_len - T = {m_half}. "
            "(Half-credit zone is a tiny fraction of N, so ratios will "
            "be essentially 1.)"
        )

    # Sanity:
    assert m_half >= 0
    assert n + m_half <= N

    print("=" * 66)
    print("Numerical validation of Theorems 8 and 9")
    print("=" * 66)
    print(scenario_desc)
    print()

    rows = []

    # ---- integer (exact match) ----
    print("Integer rule (exact match only):")
    p_int = integer_prior_all_j(N, n)
    C_int = C_from_priors(p_int)
    print(f"  C_int = {C_int:.6f}")
    rows.append({"rule": "Integer", "C": C_int})

    # ---- half-integer (Lemma 3) ----
    print(f"\nHalf-integer rule (Lemma 3, m = {m_half} half-credit positions):")
    p_half = half_integer_prior_at_integer_j(N, n, m_half)
    C_half = C_from_priors(p_half)
    print(f"  C_half = {C_half:.6f}")
    print(f"  ratio C_int / C_half       = {C_int / C_half:.6f}")
    print(f"  sigma_half / sigma_int (~) = {np.sqrt(C_half / C_int):.6f}")
    rows.append({"rule": "Half-integer", "C": C_half})

    # ---- 1/4-resolution (Lemma 4) ----
    # Define weight classes for 1/4: split half-credit zone into two
    # equal halves with weights 2/4 and 1/4 (closer half / farther
    # half). Concretely, pick m_half/2 elements at weight 2/4 (=1/2)
    # and m_half/2 at weight 1/4. Pure illustrative choice.
    m_quarter_close = m_half // 2
    m_quarter_far = m_half - m_quarter_close
    print(f"\n1/4-resolution rule (Lemma 4):")
    print(f"  weight 1/4: {m_quarter_far},  weight 2/4: {m_quarter_close},  "
          f"weight 3/4: 0,  weight 1: {n}")
    M_counts_q = [N - n - m_half, m_quarter_far, m_quarter_close, 0, n]
    p_q = m_resolution_prior_at_integer_j(N, n, M_counts_q)
    C_q = C_from_priors(p_q)
    print(f"  C_{{1/4}} = {C_q:.6f}")
    print(f"  ratio C_int / C_{{1/4}}        = {C_int / C_q:.6f}")
    print(f"  sigma_{{1/4}} / sigma_int (~)  = {np.sqrt(C_q / C_int):.6f}")
    rows.append({"rule": "$1/4$-resolution", "C": C_q})

    # ---- summary ----
    print()
    print("=" * 66)
    print("Theorem 9 part (i): C_{1/m} > C_int strictly")
    half_ok = C_half > C_int
    quarter_ok = C_q > C_int
    print(f"  C_int   = {C_int:.4f}")
    print(f"  C_half  = {C_half:.4f}   ({C_half / C_int:.2f}x C_int)  -> "
          f"C_half > C_int: {'YES' if half_ok else 'NO'}")
    print(f"  C_{{1/4}}  = {C_q:.4f}   ({C_q / C_int:.2f}x C_int)  -> "
          f"C_{{1/4}} > C_int: {'YES' if quarter_ok else 'NO'}")
    print()
    print("Theorem 9 part (ii): sigma_{1/m} > sigma_int.")
    print("  Under Xiao MI bound MI(sigma) ~ const/sigma^2, the ratio is")
    print("  approximately sqrt(C_{1/m}/C_int):")
    print(f"  sigma_half / sigma_int ~ sqrt(C_half/C_int)   "
          f"= {np.sqrt(C_half / C_int):.4f}")
    print(f"  sigma_{{1/4}}  / sigma_int ~ sqrt(C_{{1/4}}/C_int)  "
          f"= {np.sqrt(C_q / C_int):.4f}")
    print()
    print("NOTE: Theorem 9 doesn't claim monotonicity in m -- only that")
    print("C_{1/m} > C_int strictly when intermediate weight classes are")
    print("non-empty. The numeric ordering between half and 1/4 depends on")
    print("the specific weight-class sizes (M_r) chosen.")
    print()

    os.makedirs(os.path.dirname(args.out_tex) or ".", exist_ok=True)
    latex_table(rows, args.out_tex, c_int=C_int)

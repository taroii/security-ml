# Review Notes: *Learnable Obfuscation for Temporally Related Video Data*

This document covers (1) verification of your description, (2) which experiments matter most, (3) what is missing, (4) observations about the current results, (5) detailed answers to your three follow-up questions, and (6) an "ideal figures" list.

---

## 1. Verification of your description

| Your claim | Verdict | Notes |
|---|---|---|
| `main_video.py` compares obfuscated vs unobfuscated performance | ✅ Correct | Trains a baseline `TransformerClassifier` on raw ResNet-18 frame embeddings and an obfuscated version with k-mixing, sample permutation Π₁, label permutation Π₂, and additive Gaussian noise B. |
| `membership_inference.py` is an MIA script | ✅ Correct | Implements the paired H⁺/H⁻ likelihood-ratio attack from §7.2 of Xiao et al. (2024). |
| `membership_inference_plots.py` plots MIA results | ✅ Correct | Currently only saves the mean-Δ panel as `fig_attack_mean_delta.png`. |
| `robustness_results.py` provides extra details | ✅ Correct | Generates the prior-success comparison and noise-scaling ratio plots. |
| The MIA already uses weighted (half-integer) scoring | ✅ Correct | Window = 5% of clip length. |
| The integer-vs-non-integer weight comparison for MIA is missing | ✅ Correct | The MIA pipeline runs once with half-integer weights; no parallel integer run. |

---

## 2. Most important experiments (priority-ordered)

### Tier 1: Already present and load-bearing

1. **Downstream classification utility** (`main_video.py`).
2. **MIA vs σ sweep** (`membership_inference.py`).
3. **Prior-success plot** from `robustness_results.py` — directly visualizes Lemma 3.

### Tier 2: Missing but high-priority

4. **Integer vs half-integer MIA at matched σ** *(missing)*. Run the attack twice — once with `compute_weighted_score` returning {0,1} (exact match only) and once with the half-integer rule — at the same σ values. This is the single most important addition because it directly validates Theorems 6-7.
5. **Mutual information estimation vs σ** *(missing)*. Even a coarse Monte-Carlo MI estimate (or evaluating the Theorem 5 closed-form bounds on your UCF-101 embeddings) would let you plot theoretical vs empirical privacy.

### Tier 3: Strongly recommended

6. **Privacy–utility Pareto curve** combining `main_video.py` accuracy and `membership_inference.py` robustness on a shared σ axis.
7. **Ablation on mixing parameter k**. The interplay between mixing intensity (which reduces MI) and weight resolution m (which increases C) is theoretically central but never demonstrated.

### Tier 4: Nice to have

8. **m > 2 weight resolution sweep** (m ∈ {1, 2, 4, 8}) to make Theorem 7 part (iii) concrete.
9. **Per-class breakdown** of MIA success.

---

## 3. Notes on the current results

### `attack_results.csv`

| σ | H⁺ win rate | Mean weighted score (out of 16) | Normalized score | Mean Δ | Std Δ |
|---|---|---|---|---|---|
| 0.01 | 0.51 | 11.79 | 0.7369 | +1.59e6 | 4.64e7 |
| 0.03 | 0.43 | 11.17 | 0.6978 | −1.93e6 | 1.58e7 |
| 0.05 | 0.50 | 11.73 | 0.7331 | −4.50e5 | 8.90e6 |
| 0.10 | 0.51 | 11.84 | 0.7400 | +2.02e5 | 4.99e6 |

**Key observations:**

(a) The H⁺ win rate hovers at 0.5 across all σ — the attack is at chance throughout the sweep. The mean-Δ figure confirms this with std bars overlapping zero everywhere. **This is a privacy success but it is not a tradeoff curve.** See Q1 below.

(b) The reported `random_baseline` of 1e-4 is the *exact-match* baseline (n_frames / N_total). It is **not** the right comparator for the half-integer scoring rule. See Q2 below.

(c) `fig_noise_ratio.png` has a spike at N = 800 from numerical instability. See Q3 below — recommend cutting this figure.

(d) `fig_prior_success.png` is correct and clean — half-integer curve strictly above integer curve, exactly matching Lemma 3.

### Code-level observations

- `obfuscate_fixed` (line 476) is defined but never called. Either wire it in or remove it.
- `membership_inference_plots.py` loads from `attack_results_10_trials/` (line 328) but the project's CSV is from a different run. Confirm you are plotting what you intend to report.

---

## 4. Detailed answers to your three follow-up questions

### Q1. The H⁺ rate is at chance — should I increase the window for "nearby" frames?

**Short answer: no, the window will not change the H⁺ rate.**

The two scoring components are decoupled in the current code:

- The **H⁺/H⁻ decision** comes from `log_likelihood(o, o_simulated, σ)` on the *obfuscated release* — it does not see frame indices at all.
- The **window** only enters `compute_weighted_score`, which is applied *after* H⁺/H⁻ has been decided to score the adversary's guess.

So widening the window will inflate the *normalized score* (more half-credit) but the H⁺ win rate will stay at ~0.5. You would be papering over the chance behavior, not fixing it.

**Why is the attack at chance?** This is structural, not a noise-floor issue. In your H⁻ construction (lines 393-404 of `membership_inference.py`), you swap the target video's 16 frames for 16 same-class frames from *other* videos, then re-run `class_k_mix` on a frame pool of ~150,000 frames with k=5 and c=101. Each mixed output is the average of 10 frames sampled from the entire pool. Swapping 16 frames out of 150,000 changes essentially nothing in expectation — the H⁺ and H⁻ obfuscated releases are statistically nearly identical. Add Gaussian noise on top and the test has no signal to detect.

**This is actually the correct privacy result.** Your mechanism is so effective that even at σ = 0.01 the attack fails. But a flat curve is not a *tradeoff* — it does not show a privacy *transition*. To make a tradeoff curve, you need to weaken the mechanism enough that the attack can succeed at small σ, then watch it degrade as σ grows. Two ways to do this, in increasing order of how much they change the paper:

1. **Reduce mixing intensity.** Drop k from 5 to 1 (or even 0 — pure permutation + noise). With less averaging, the swap signal becomes detectable. You can sweep k as a parameter and show the H⁺ rate as a function of (σ, k). This *also* satisfies Tier 3 #7 (the k ablation) — kill two birds with one stone.

2. **Make H⁻ stronger.** Currently you swap with same-class frames, which is the "hardest" H⁻ — the swapped frames look statistically similar to the originals. Try a weaker H⁻ that swaps with frames from a *different* class, and you should see meaningful Δ at small σ. This isolates exactly how much the same-class adjacency contributes to privacy.

Of these, **(1) is the better paper move.** It directly produces the k × σ heatmap that demonstrates the central tension your theory describes (mixing reduces MI; noise reduces MI; but they trade against utility differently).

If after both you still see a flat curve, that is the result and you should report it as: *"the linear obfuscation pipeline neutralizes likelihood-ratio MIA at all tested σ, including σ → 0; this is consistent with the bound being loose at the operating point of class-k-mixing with k ≥ 1."* That is a legitimate finding but should be framed deliberately rather than presented as a degraded tradeoff.

### Q2. How should I compute the half-integer random baseline?

**You can compute it analytically from Lemma 3, no Monte-Carlo needed.**

For a uniformly random n-subset $\widehat{X}$ over a universe partitioned into (A: n exact-match, B: m half-weight, C: N−n−m zero-weight), the expected weighted overlap is

$$\mathbb{E}[W] = \mathbb{E}[a] + \tfrac{1}{2}\mathbb{E}[b] = n\cdot\frac{n}{N} + \tfrac{1}{2}\cdot n\cdot\frac{m}{N} = \frac{n(n + m/2)}{N}.$$

Normalized by n: **`baseline = (n + m/2) / N`**.

In your setup, the partition is defined per-target-clip (the universe is the full frame pool of size N_total ≈ 150,000; for a target clip of length L with T = 16 true frames and window 0.05·L, A has 16 elements and B has roughly $\min(2 \cdot 0.05 \cdot L \cdot 16, L - 16)$ elements per clip — typically a few dozen). So m is small relative to N_total and the half-integer baseline is only modestly larger than the exact-match baseline.

**However — this is not why your normalized score is 0.74.** The 0.74 comes from the fact that when H⁺ wins, the adversary returns the *true* linspace frame indices, which trivially score n/n on themselves under any window. Look at `prior_normalized` in `run_attack` (lines 444-455) — it is computing exactly this self-overlap, which is 1.0 by construction except for window-edge cases.

**What you actually want for the plot:**

There are two distinct baselines and you should plot both:

1. **Naïve random baseline** (per Lemma 3): `(n + m/2)/N_total`. Tiny number, ~1e-3 with a generous window. This is the "informed-guesser-with-no-side-channel" baseline.

2. **Always-guess-H⁺ baseline**: this is essentially the prior advantage of the linspace sampling structure, which is approximately the H⁺ win rate × score-when-H⁺-correct. With chance H⁺ rate (0.5) and near-perfect self-score (~1.0), this baseline is ~0.5. **This is what your adversary should be compared against, not the random baseline.** A normalized score of 0.74 vs a 0.5 always-guess-H⁺ baseline is a meaningful 0.24 advantage; vs the 1e-4 random baseline it looks like a 0.74 advantage, which is misleading.

The fix is concrete:

```python
# in run_attack, replace prior_normalized computation with:
analytical_random_baseline = (16 + m_per_clip / 2) / N_total  # Lemma 3
always_h_plus_baseline = compute_weighted_score(  # current prior_normalized
    true_fidxs, true_fidxs, clip_len) / 16
```

Then in the plot, show both baselines as horizontal lines and the adversary score as a curve. The interesting quantity is `adversary - always_h_plus_baseline`, not `adversary - random`.

### Q3. Is `fig_noise_ratio.png` worth including?

**No. Cut it.** Three reasons:

1. **It's a function of N, which is fixed by your dataset.** N (the size of your frame pool) is not an experimental knob you control or sweep. The ratio at one fixed N would be a number, not a plot. Plotting against synthetic N values is illustrative but doesn't connect to your UCF-101 setup.

2. **The numerical instability is hard to fix cleanly.** The non-monotonic spike at N = 800 comes from `compute_C` clipping `prior_geq` to `1 - 1e-12` and then dividing by `−log(1 − p)`, which blows up near the clip boundary. You could either use a more careful tail handling (e.g., compute `−log(1−p)` directly via `−log1p(−p)` and extend precision with mpmath) or restrict the plot to a regime where C is stable. Both add work without strengthening the paper.

3. **`fig_prior_success.png` already makes the central point cleanly.** It directly visualizes Lemma 3 — half-integer prior strictly dominates integer prior at every threshold. The noise ratio is a derived consequence; a reader who understands Lemma 3 and Theorem 7 does not need a separate plot for the ratio. If you want to convey "more noise required," report a single number in the text (e.g., "for our setup, C_half/C_int ≈ X, so σ_half must satisfy MI_half ≤ MI_int / X").

Cutting this figure also tightens the paper around the claims you can defend cleanly.

---

## 5. Ideal figures for the paper

Below is the minimal-but-complete figure set that directly serves the theoretical claims. Each figure has an explicit purpose tied to a result in the paper. Anything not on this list should be dropped.

### Figures that exist and should stay

**Figure 1: Prior success probability vs threshold j (Lemma 3).**
- Status: ✅ exists as `fig_prior_success.png`
- Purpose: directly visualizes that half-integer weights strictly dominate integer weights, the core combinatorial claim of §3.1.
- Recommendation: keep as-is. Maybe drop the right-hand log-scale panel; the linear panel alone is sufficient.

### Figures that exist but need fixing or reframing

**Figure 2: MIA robustness vs σ.**
- Status: ⚠️ exists as `fig_attack_mean_delta.png` but is misleading without a baseline fix and shows no transition.
- Purpose: empirical evidence that the obfuscated mechanism resists likelihood-ratio MIA.
- Recommendation: redo as a 2-panel figure showing (a) H⁺ win rate vs σ with the 0.5 chance line, (b) normalized weighted score vs σ with **both** the analytical Lemma-3 random baseline *and* the always-guess-H⁺ baseline as reference lines. Add k as a second sweep dimension (per Q1) so the curve actually shows a transition.

### Figures that don't exist yet but should

**Figure 3: Integer vs half-integer empirical MIA at matched σ.**
- Status: ❌ missing (Tier 2 #4)
- Purpose: empirical validation of Theorems 6-7 — the central novel theoretical contribution.
- Recommendation: same MIA run, two scoring rules, side by side. Plot normalized score (integer) and normalized score (half-integer) as two curves over σ. Annotate the σ shift required to match scores between rules; compare to theoretical C_int/C_half.

**Figure 4: Privacy–utility Pareto frontier.**
- Status: ❌ missing (Tier 3 #6)
- Purpose: the headline empirical claim of the paper. One plot, two axes, varying σ.
- Recommendation: x-axis = test accuracy from `main_video.py`, y-axis = (1 − normalized MIA score), points labeled by σ. One curve per scoring rule (integer, half-integer). This is the figure most reviewers will look at first.

**Figure 5: Mutual information vs σ (theoretical bound vs empirical MIA).**
- Status: ❌ missing (Tier 2 #5)
- Purpose: validates that MI is *predictive*, not just *bounded* — the claim made in the §6 introduction that you currently do not back up.
- Recommendation: x-axis = σ, two curves: (i) Theorem 5 Type-(I)/(II) upper bound on MI(X;M(X)) evaluated on your data, (ii) empirical MIA success rate. Show that the bound tracks the empirical signal.

### Figures to cut

- **`fig_noise_ratio.png`** — see Q3.
- The right-hand log-scale panel of `fig_prior_success.png` is redundant with the linear panel and adds clutter.
- The 4-panel `plot_all` figure in `membership_inference_plots.py` was never used in the paper LaTeX you sent; the score-distribution panel in particular is currently a bar chart of means (the box plot fallback), which carries no extra information beyond the table. Don't bother resurrecting it.

### Summary

If you only have time for the minimal viable figure set:

| # | Figure | Status | Maps to |
|---|---|---|---|
| 1 | Prior success vs j | ✅ have it | Lemma 3 |
| 2 | MIA robustness vs σ (with proper baselines, with k sweep) | ⚠️ rework | Theorem 6 |
| 3 | Integer vs half-integer MIA at matched σ | ❌ make it | Theorem 7 |
| 4 | Privacy–utility Pareto | ❌ make it | overall claim |

Figure 5 (MI vs σ) is desirable but the most expensive. If time is tight, drop it and make the case via Figures 3 and 4. Drop the 2-panel log scale, the noise ratio plot, and the 4-panel MIA dashboard.
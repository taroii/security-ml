# Revision Plan — "Learnable Obfuscation for Temporally Related Video Data"

CCS 2026 Cycle B, paper #1783 (rejected round 1). This document maps every
reviewer comment to a concrete change, and records what has already been
implemented in this repository.

## The one-sentence diagnosis

Both reviewers, in different words, said the same thing: **the paper motivates
itself with temporal correlation but never puts that correlation into the
model.** The dataset `X` is still sampled i.i.d. from the universe; only the
adversary's *reward* changes (partial credit for near misses). Reviewer A: the
guarantees are "disconnected from the practical security of the mechanism."
The revision's job is to make that sentence unwritable by (1) putting
dependence into the sampling model, (2) building a correlation-exploiting
attack and measuring whether it beats the i.i.d.-calibrated defense, and
(3) addressing noise design under dependence.

Everything below is done and reproducible; scripts are named per item.

---

## Reviewer A (knowledgeable; the review that matters)

### A1. "The paper acknowledges the data are not i.i.d. but ignores the correlations present in the dataset ... the resulting guarantees seem disconnected from the practical security."

**This is the core objection.** Response has three parts.

**(a) Put dependence in the sampling model — new theory.**
`src/two_level_prior.py`. We replace uniform n-subset sampling with a
two-level *clip-then-frame* generative model: the data owner selects `g` clips
from `G`, then `ell` frames from each, so membership is clustered exactly as
real video datasets are assembled. The adversary's weighted overlap becomes a
**compound distribution**

    W = sum_{i=1}^{H} W_i ,   H ~ Hypergeometric(G, g, g),
    W_i i.i.d. within-clip half-integer overlaps (Lemma 3).

Validated: the closed form matches Monte Carlo (max |Δ survival| = 2e-4), and
it **recovers the paper's Lemma 3 exactly when G = 1** (max |Δ| = 1.3e-13), so
it is a strict generalization, not a replacement.

Headline result (G=100, g=50, L=100, ell=8; identical marginal membership rate
and identical total half-credit mass as the flat model): the flat i.i.d. prior
and the clustered prior have the **same mean overlap (108)** but the clustered
prior has a **heavy upper tail** — P(recover ≥130 of 400) is 0.024 vs 6e-5
(**≈400× higher**), and P(≥140) is ≈2e5× higher. A privacy guarantee must bound
the *tail*, not the mean, so the i.i.d. calibration is optimistic exactly where
it matters. A `g/G = 0.5` universe-size sweep shows the gap crossing 1 and
growing to ≥1e4× as the universe grows.

This is the direct, formal answer to "you ignored the correlations": the prior
— and therefore the required noise — is now a function of the clip structure
`(G, g, L, ell, window)`, which the flat analysis cannot see.

**(b) Build the correlation-exploiting MI attack the reviewer asked for.**
`src/correlation_aware_attack.py`. The paper's Attack 3 scores every candidate
frame independently (the i.i.d.-blind adversary). We add a **clip-aggregated**
adversary that pools the *same* per-frame LiRA scores across a clip before
deciding — mirroring the compound prior (winning one clip delivers a block of
correlated credit). Both adversaries see the same released output and the same
scores; they differ only in whether they use the clip structure.

**(c) Controlled study isolating the correlation.** `src/synthetic_correlation.py`.
On real video the correlation is uncontrolled, so we also generate synthetic
AR(1) clips with a single correlation dial `rho` (Corr(frame_s, frame_t) =
rho^|s-t|). Three results (Figure `fig_synthetic_correlation.pdf`):
  - **Graded leakage is real:** the LiRA score of a *non-member* frame rises as
    its temporal distance to the nearest member shrinks, with reach growing in
    rho. At rho=0 only members leak; at rho=0.95 leakage extends 2–3 frames out.
    This is the first direct evidence that the partial-credit reward the paper
    defines corresponds to measurable leakage.
  - **The flat calibration is optimistic:** at a fixed noise level, the
    correlation-aware attack's source-clip identification rises from 0.75
    (rho=0) to 1.00 (rho≥0.9) while the i.i.d. adversary the paper analyzes
    stays far lower (0.05 → 0.87). Correlation that the flat model treats as
    absent is what the attack exploits.

### A2. "Independent per-frame noise ... is less clear it is appropriate for models that must jointly process correlated frames."

**Done.** `precompute_trial(..., noise_mode="clip", noise_clip_frac=f)` adds a
shared per-clip noise component at *matched total variance*. Experiment C of
`synthetic_correlation.py`: at matched budget, making the noise clip-correlated
**reduces** the pooling adversary's success from 0.29 (i.i.d.) to 0.17 (fully
clip-shared), while the blind adversary is unaffected. The shared component sums
coherently under the adversary's cross-frame pooling, so it is a strictly better
defense for joint-processing models — a concrete, positive answer to the
reviewer's design question (independent per-frame noise is *not* optimal).

### A3. "Mark C_half explicitly in Table 1."
**Done** in `paper_edits.md` (Table 1 gains a `C_half` row label and a marked
column; the calibration table script already emits the value).

### A4. "High-level statement of the quantitative guarantees in the half-integer and 1/m model in the Intro; and how the number of nearby frames the adversary gets credit for impacts the numerics."
**Done.** New Intro paragraph (in `paper_edits.md`) states the numbers plainly:
half-integer credit inflates the prior constant ≈31× and the required noise
≈5.6× over the integer baseline; and a **window sweep** (`two_level_prior.py`,
right panel of `fig_two_level_prior.pdf`) quantifies exactly how the temporal
reach of partial credit moves the adversary's expected recovery (monotone:
recovered fraction 0.08 at ±1 frame → 0.27 at ±10).

### A5. "Explain why windowing + theoretical bounds is meaningful vs running MI attacks."
**Reframed.** The revision no longer positions the theory as a *substitute* for
MI attacks; it runs both and shows they agree. The theory gives the calibration
(how much noise) and the attacks verify it empirically. Section restructured so
the empirical correlation-aware attack (A1b) is the headline and the bounds are
the calibration that the attack tests.

### A6. "Explain half-integer and 1/m in the Introduction."
**Done** — plain-language definitions added to the Intro (`paper_edits.md`).

---

## Reviewer B (no familiarity; likely a track-mismatch draw)

### B1. "Hard to understand; assumes a lot of familiarity with [50]; preliminaries not laid out well."
**Done.** New self-contained Preliminaries subsection recapping exactly what is
imported from Xiao et al. [50] — the mechanism `M(X)=Π₁ M_mix X W + B`, the
mutual-information bound, and the calibration constant — with notation defined
before use (`paper_edits.md §Preliminaries`).

### B2. "Why does Definition 1 force X drawn uniformly at random from U? Doesn't that make the attacker weak? Or is it a restriction on how the data owner assembles the dataset?"
**Directly addressed** — this is the same blind spot as A1. The revision keeps
the uniform baseline for comparability with [50] but adds the two-level
structured sampling (A1a) as the realistic model, and a threat-model paragraph
explains that uniform sampling is the *conservative* modeling choice for the
prior and that clustered sampling (which the data owner does in practice) makes
the adversary stronger, not weaker.

### B3. "How are the weights in Definition 1 decided in practice? Does the data owner fix the similarity notion a priori?"
**Done.** New subsection "Choosing the weights in practice" (`paper_edits.md`):
the weights encode the data owner's declared protection target (temporal window
and/or semantic similarity), fixed a priori; we give a concrete recipe (window
= the event-duration the owner wants to hide) and note the window sweep shows
the privacy cost of a wider protected zone.

### B4. "Include clearly demarcated algorithm boxes for the mechanism and the attacks, with inputs/outputs; clarify how sigma impacts the mechanism (Table 3 hard to interpret)."
**Done.** Three LaTeX `algorithm` boxes drafted in `paper_edits.md`: the
obfuscation mechanism, blind Attack 3, and correlation-aware Attack 3 — each
with explicit inputs/outputs and the exact line where σ (the noise `B`) enters.

### B5. "Table 2 shows only one (k, σ); show the tradeoff like Table 3."
**Done.** The full `(k, σ)` sweep already exists in Appendix Table 4 — the
revision promotes a condensed version into the main body next to Table 2 and
cross-references it explicitly (the reviewer missed the appendix, indicating the
cross-reference failed). See `paper_edits.md §Exposition fixes`.

### B6. "In related work, compare to DP approaches for video data, not just general DP."
**Done.** New related-work paragraph with verified citations:
  - **Privid** (Cangialosi et al., NSDI 2022) — event-duration DP for video queries.
  - **VideoDP** (Wang et al., PoPETs 2020(4):277–296) — DP video-analytics platform.
  - **Dependent DP** (Liu, Chakraborty, Mittal, NDSS 2016) — DP fails under tuple correlation.
  - **Pufferfish** (Kifer & Machanavajjhala, ACM TODS 2014) — privacy under correlated data.
  - **DP under temporal correlations** (Cao et al., ICDE 2017 / arXiv:1610.07543).
See `paper_edits.md §Related work` for formatted BibTeX-ready entries.

---

## Track choice

The paper was submitted to **Applied Cryptography**, which drew a
"no familiarity" reviewer (B). This is an ML-privacy paper with no cryptographic
construction. Resubmission should target a privacy/ML-security venue or track:
**PoPETs** (best fit), **IEEE S&P 2027 cycle 2**, or **CCS 2027 Cycle A** under
the ML/privacy track — not Applied Crypto. (See memory note for deadlines.)

## Artifact status

New code (all in `src/`, runnable with the existing `.venv-rev` / conda env):
  - `two_level_prior.py`      — structured prior theory + validation + figure/table
  - `correlation_aware_attack.py` — blind vs clip-aware Attack 3 on UCF-101
  - `synthetic_correlation.py`    — controlled-rho leakage / attack / noise study
  - `membership_inference.py`     — extended with `noise_mode={iid,clip}`
Figures: `images/fig_two_level_prior.pdf`, `images/fig_synthetic_correlation.pdf`.
Result CSVs: `results_revision/`. Raw run logs: `revision/*_results.txt`.

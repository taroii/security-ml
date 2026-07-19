# How to integrate the revised paper

`main.tex` in this directory is the **revised** paper source, incorporating all
reviewer-response changes (see `../revision/REVISION_PLAN.md` for the
comment-by-comment mapping). To drop it into your Overleaf project:

1. **Replace** your existing `main.tex` with this one.
2. **Bibliography:** `references.bib` here is the FULL bib (your existing entries
   + 3 new ones), so you can use it directly. Only three entries are new:
   `liu2016dependence`, `kifer2014pufferfish`, `cao2017temporal` (also in
   `references_additions.bib` if you'd rather merge into your own file). The
   revision reuses your existing `cangialosi2022privid`, `wang2019videodp`
   (VideoDP), and `visor2020usenix` keys — no duplicates introduced.
3. **Upload two new figures** (already in `../images/`), placing them where your
   other figures live:
   - `fig_two_level_prior.pdf`  (Figure: two-level clustered vs i.i.d. prior)
   - `fig_synthetic_correlation.pdf` (Figure: controlled-correlation study)

## What changed vs. the submitted version

- **Abstract / Intro:** now state the quantitative guarantees ($31\times$ prior,
  $5.6\times$ noise), define half-integer / $1/m$ weights, and preview the
  two-level dependence model.
- **New §"What we import from [50]"** (Preliminaries): self-contained recap of
  the mechanism, MI bound, and calibration constant $C$.
- **New §"On the sampling of $X$"**: explains that uniform sampling is the
  conservative baseline, not a restriction on the data owner (Reviewer B).
- **New §"Choosing the weights in practice"**: how the window/threshold is set
  a priori (Reviewer B).
- **New §4.3 "Two-Level Sampling and the Compound Prior"** with
  Lemma (compound prior) + validation + Figure — the core answer to Reviewer A
  ("you ignored the correlations"): dependence is now in the sampling model.
- **New §5.4 "Noise Design under Dependence"**: correlated-noise mechanism.
- **New §6.4 "Correlation-Aware Attack and Correlated Noise"** with Algorithm
  boxes and the synthetic-study Figure (Reviewer A's requested attack).
- **New Related-Work §"Differential Privacy for Video and Correlated Data"**
  (Reviewer B): Privid, VideoDP, Dependent DP, Pufferfish, temporal-corr DP.
- **Three algorithm boxes** (mechanism, blind attack, correlation-aware attack)
  with explicit inputs/outputs and the line where $\sigma$ enters (Reviewer B).
- **Table 1** now inlines the calibration values and labels $C_{\mathrm{half}}$
  explicitly (Reviewer A). It no longer `\input`s an external file.
- **Table 2 caption** cross-references the full appendix sweep (Reviewer B).
- Cleaned the residual "three attacks" wording to two attacks + the separate
  correlation-aware evaluation.

## Compile status: VERIFIED

`main.tex` + `references.bib` (this dir) + the two new figures were compiled
end-to-end with **TeX Live 2026** (`pdflatex` → `bibtex` → `pdflatex` ×2) under
`acmart` sigconf. Result: a clean **18-page** PDF (`main_revised.pdf`, included
here as proof), **zero undefined citations or references**, all figures and the
three algorithm boxes render correctly. The only messages are cosmetic bibtex
metadata warnings (empty `publisher`/`address`, missing page numbers) that are
present in your original bib too and do not affect the build.

- Preamble adds `algorithm` and `algpseudocode` for the algorithm boxes — both
  are in a standard TeX Live and compiled without issue.
- The paper still uses Applied Crypto formatting; per the revision plan, consider
  resubmitting to a privacy/ML track (PoPETs, S&P, or CCS ML/privacy).

## Pre-existing issues flagged by an automated review (NOT changed by me)

These were in your original source; I left them alone to avoid breaking your
working setup, but they are worth reconciling:

1. **`\newtheorem` block (preamble).** ~~Flagged by an automated check as a
   potential "already defined" clash with recent `acmart`.~~ **Not an issue:**
   the project compiles cleanly with this block under TeX Live 2026's `acmart`
   (verified — see Compile status above). Left unchanged.
2. **`log 2` vs `(log 2)/n` inconsistency.** The success-bound numerator is
   written as `MI + log 2` in Lemma 2 and Theorem (half-integer calibration) but
   as `MI + (log 2)/n` in the Section-5 recall and Theorem (general 1/m). These
   are your existing theorem constants (I did not touch the proofs); pick one
   convention and make Lemma 2, its recall, and both calibration theorems agree.
3. **Label with a space/typo:** `\label{memebership inference challenge}` has a
   space and a typo. It compiles (the `\ref`s match the same string) but is
   fragile; consider renaming to e.g. `def:temporal-mia`.
4. **VideoDP metadata:** your existing `wang2019videodp` entry lists authors
   "Tianhao Wang, Blocki, Li, Jha" — the actual VideoDP (PoPETs 2020) is by
   Han Wang, Shangyu Xie, Yuan Hong (arXiv:1909.08729). Worth verifying/fixing
   the author/venue metadata.
5. **Two-level within-clip model:** the compound-prior code samples the `ell`
   member frames uniformly within a clip and treats the `±w` half-credit windows
   as near-disjoint. This matches your evaluation's *random* per-clip frame
   sampling (now stated in the two-level paragraph). If you switch to
   contiguous-block sampling, the window→half-zone mapping in
   `src/two_level_prior.py` should account for overlapping windows. The default
   whole-clip headline numbers (400×, ~2×10^5 tail) are unaffected.

## What the automated review corrected in the revised text

- Fixed a factual contradiction with Table 2 (§6.3 and Conclusion): obfuscation
  *reduces* attack success vs. no-obfuscation (the obfuscated scores are below
  the no-obfuscation column); the text now says "reduces … but does not
  eliminate" and "exceed the chance baseline while falling below the
  no-obfuscation scores," instead of the earlier "increases leakage" /
  "outperform the no-obfuscation baseline."
- Attributed the correlated-noise defense and the strong correlation-aware
  effects explicitly to the *synthetic* controlled study (they were not run on
  real UCF-101), and propagated the real-data "modest gain" caveat into §5.4 and
  the Conclusion.
- Corrected "≈10^5" → "≈2×10^5" for the threshold-140 tail ratio, and softened
  "consistent improvement" (one of four UCF-101 noise cells is slightly negative).

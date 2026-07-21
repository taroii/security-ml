# Merged paper — what changed, section by section

This is the **AAAI-2027 merged paper** (`new.tex`, 23 pp, compiles clean) that
combines: (a) the advisor's new Section 7 correlation theory, (b) the CCS
reviewer fixes, and (c) my experiments that now **validate** Section 7. Below is
a section-by-section map so you can drop each piece into Overleaf, plus the
reviewer-coverage table.

Everything in `~/Desktop/new theory/` compiles: `pdflatex new.tex → bibtex new →
pdflatex ×2` gives `new.pdf` (23 pp, no undefined refs, no fatal errors).

---

## Section-by-section changes

Legend: **[NEW]** added this round · **[EDIT]** I changed content · **[KEEP]**
unchanged from the merged base you pasted.

| Section | Status | What it is / what I changed |
|---|---|---|
| **Abstract** | **[EDIT]** | Fixed a real contradiction: it claimed correlation "inflates Gram eigenvalues and **weakens** privacy," but Thm 11 proves correlation *reduces* MI. Rewrote to the accurate story: correlation reduces MI **but** enables an aggregation attack, fixed by correlated noise. Added the measured `α≈0.84`. |
| **1 Introduction** | **[EDIT]** | Same fix in the contribution bullet on correlation. Rest kept (quantitative preview 31×/5.6×, weighted-scoring intuition, "why theoretical bounds"). |
| **2 Related Work** | **[KEEP]** | Includes the new §2.5 "DP for Temporally Correlated Data" (Cao, Pufferfish, Song, Bozkir) — addresses **B3**. |
| **3 Preliminary** | **[KEEP]** | Self-contained recap of [50] (**B1**), "Role of uniform sampling" (**B4**), "Weight selection" (**B5**), "Role of σ" (**B6**), Algorithm 1 & 2 boxes (**B6**). |
| **4 Main Results** | **[KEEP]** | Lemma 2 (hardness) + Lemmas 3–4 (half-integer, 1/m priors). |
| **5 Noise Calibration** | **[KEEP]** | Theorems 5–6 (half-integer, 1/m calibration). |
| **6.2 Numerical Eval / Table 1** | **[KEEP]** | Table 1 now labels `C_int`/`C_half` (**A4**). |
| **6.3 Empirical MIA / Table 2** | **[EDIT]** | **Filled the binary-scoring rows** (were `---`) from the committed result CSVs; added the weighted-vs-binary finding (Attack 1: weighted 0.44–0.51 vs binary 0.18–0.23 — binary misses ~half the leakage). Fixed one stale weighted value ((1,0.10) Attack 1: 0.404→**0.454**, matching the artifact). Clarified the baseline columns in the caption. Addresses **B7 + A2**. |
| **6.5 Utility** | **[KEEP]** | (k,σ) accuracy sweep. |
| **7.1–7.5 Correlation theory** | **[KEEP]** | The advisor's 5 theorems: 11 (MI reduction), 13 (block prior), 14 (combined calibration), 15 (noise averaging), 16 (correlated noise). I verified the AR(1) factor `R̄_b` closed form is algebraically correct (matches the direct sum exactly). |
| **7.5 Correlated noise** | **[EDIT]** | Removed the "we leave empirical evaluation to future work" line — it's now done in §7.7, so I point there. |
| **7.6 Autocorrelation** | **[NEW]** | Filled the TODO: measured real ResNet-18 embedding autocorrelation on UCF-101 (`fig_embedding_autocorr.pdf`). `ρ(1)=0.78`, AR(1) `α=0.84`, `R̄_16=7.4` → noise reduced 2.7× under aggregation. This is the real-data instance of Theorem 15. |
| **7.7 Empirical Validation** | **[NEW]** | New subsection instantiating Theorem 15 (the correlation-aware **aggregation attack** vs the blind attack, Table 4 on UCF-101) and Theorem 16 (**correlated noise** cuts the attack 0.29→0.17), plus the controlled AR(1) study (`fig_synthetic_correlation.pdf`). Directly answers **A1/A2/A3** empirically. |
| **8 Conclusion** | **[EDIT]** | Rewrote the correlation paragraph to the accurate Thm-11/13/15/16 story; updated "future work" (correlated-noise now validated; only the downstream utility eval remains). |
| **Appendix A.1 / Table 5** | **[EDIT]** | Regenerated to match the committed CSVs (only change: (1,0.10) Attack 1 0.404→0.454). |
| **Appendix B/C** | **[KEEP]** | Open Science, Ethics. |

**Figures to upload alongside `new.tex`** (already in this folder):
`fig_embedding_autocorr.pdf` (Fig 3), `fig_synthetic_correlation.pdf` (Fig 4),
plus the existing `fig1_prior_success.pdf`, `figure5_visual_obfuscation_pixel.pdf`.

---

## Reviewer coverage (all 15 points)

| # | Reviewer point | Where addressed |
|---|---|---|
| **A1** | data not i.i.d., correlations ignored | §7 (theory) **+ §7.6/7.7 (measured + attack)** |
| **A2** | MI attack testing if [50] overestimated | §6.3 weighted-vs-binary + **§7.7 aggregation attack** |
| **A3** | independent noise wrong for joint-processing | Thm 15/16 + **§7.7 correlated-noise result** |
| **A4** | mark C_half in Table 1 | Table 1 (labeled) |
| **A5** | intro quantitative + #nearby-frames impact | Intro "Quantitative preview" + weight window |
| **A6** | why bounds vs attacks | Intro "Why theoretical bounds?" + now both run |
| **A7** | explain half-integer/1/m in intro | Intro "Weighted scoring: intuition" |
| **B1** | self-contained, don't assume [50] | §3.1 recap |
| **B2** | experimental section underspecified | §6.1 expanded + algorithm boxes |
| **B3** | DP for video/correlated data | §2.5 |
| **B4** | why uniform sampling in Def 1 | §3.1 "Role of uniform sampling" |
| **B5** | how weights decided | §3.2 "Weight selection" |
| **B6** | algorithm boxes + σ role | Alg 1 & 2 + "Role of σ" |
| **B7** | Table 2 only one (k,σ) | Table 2 now 4 points × weighted/binary |

---

## Key decisions / fixes you should know about

1. **Abstract/intro correction (important).** The merged draft's abstract said
   correlation "weakens privacy," but the advisor's own Theorem 11 proves
   correlation *reduces* MI (helps privacy on that axis). A reviewer would catch
   this. I rewrote abstract + intro + conclusion to the accurate net story:
   correlation reduces MI, **but** the aggregation attack (Thm 15) is the real
   danger, eliminated by correlated noise (Thm 16). **Please sanity-check you're
   happy with this framing.**

2. **Table consistency.** Table 2 and Appendix Table 5 now come from the same
   committed result CSVs (the reproducible artifact). Only one number moved:
   (1,0.10) Attack 1, 0.404 → 0.454 (the old value was a stale run).

3. **Attack 2 weighted = binary** — not a bug: the subsampled pool has only exact
   target rows, so there are no near-miss frames for partial credit. Stated in
   the text.

4. **Real-data honesty.** The aggregation attack's gain over the blind attack is
   *modest* on UCF-101 (random per-clip sampling mutes the correlation); the full
   effect is shown synthetically. Said plainly in §7.7.

## Section 7 theorem corrections (please review — I edited the advisor's math)

An automated proof-check (5 independent reviewers + adversarial verification)
found two genuine errors in the Section 7 theorems, which I corrected. **Please
confirm you agree:**

1. **Theorem 14 (combined calibration), `eq:sigma_combined_bound`: inequality was
   backwards.** It stated `σ_comb ≥ σ_int·√(C_block/C_int)`, but the theorem's own
   proof ("no harder to satisfy," because `I_R ≤ I_{I_n}`) implies `≤`. Since
   correlation *reduces* the MI bound, the correlated mechanism needs *no more*
   noise than the weighted-scoring inflation — so the √ factor is an **upper**
   bound, not a lower bound. Fixed `≥`→`≤` and rewrote the interpretation.

2. **Theorem 15 numeric example wrong.** For `α=0.9, b=16` the text claimed
   `R̄_b ≈ 11.4` (noise reduction `3.4×`). The correct value from the (correct)
   closed form is `9.8` (`3.1×`). The closed form itself is right — I verified it
   equals the direct sum for all `b`; only the plugged-in number was off. Fixed.

3. **"Competing forces" interpretation after Theorem 11 was wrong.** It said
   correlation might *raise* `C` and outweigh the MI reduction. But
   Theorem 13 proves block-correlation *lowers* `C` vs scattered; the `C`
   increase is from *weighted scoring*, not correlation. Rewrote: correlation is
   privacy-favorable on both factors; the real residual threat is the aggregation
   attack (Thm 15). This now matches the abstract/intro/conclusion.

4. Minor wording: conclusion said correlation "reduces the mutual information";
   Thm 11 only bounds it — changed to "mutual-information **bound**".

5. Table 4 caption trial count corrected (75 = 15×5, not 60 = 15×4).

The math reviewer confirmed the AR(1) closed form and Theorems 11, 13, 15(i–ii),
16 are otherwise sound.

## Open items for you / the advisor

- **AAAI page limit.** The paper is 23 pp; AAAI main text is typically ~7–9 pp +
  refs. If submitting to AAAI, a lot will need to move to appendix/supplement.
- **`log 2` vs `(log 2)/n`** inconsistency across Lemma 2 / Theorems 5–6 (pre-
  existing in the calibration theorems) — still worth reconciling.
- **Downstream utility of correlated noise** — §7.7 validates it against the
  attack; a UCF-101 accuracy run under clip-correlated noise would complete it.

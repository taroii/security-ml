# Additional experiments (address the "empirics don't close the loop" critique)

Two new experiments target the reviewer's two strongest objections. Both use the
existing artifact; scripts + figures + CSVs are in the repo.

## 1. The MI bound, evaluated on real embeddings (`fig_mi_bound_real.pdf`)

**Critique it answers:** "the central claim rests on bounding `MI(X;M(X))`, but it
is never computed." The paper only evaluated the prior constant `C` on a toy
universe; the MI side was never grounded in real data.

`src/mi_bound_evaluation.py` computes the Theorem-5 mutual-information upper bound
on the real UCF-101 (ResNet/R3D-18) embeddings across `(k, σ)`, using the
Sylvester identity `log det(I_n + XX^T/σ²) = log det(I_{512} + X^TX/σ²)` (a fast
512×512 determinant). Findings:

- The bound is **finite and non-vacuous** on the embeddings (≈240 → 26 nats/point
  as σ: 0.01 → 1.0), unlike the raw-pixel case.
- It **decreases monotonically with σ** and is **roughly halved by class-1 mixing**
  (k=0: 240 → k=1: 118 → k=5: 101 nats/pt at σ=0.01), directly confirming the
  paper's claim that mixing reduces `MI(X;M(X))`.
- **Honest refinement:** in the operable band the bound decays like ~`1/σ`, *not*
  the `1/σ²` the calibration assumes (measured ratio ≈0.53 vs 0.25 predicted for a
  2× σ increase). So the √-based `5.6×` noise-inflation figure is a **lower bound**
  on the true required inflation — the real calibration is *more* demanding, not
  less. This tightens (and slightly corrects the optimism of) the paper's headline.

Caveat to state in the paper: this is the MI *upper bound*, not a direct estimate
of the MI itself; a PAC-privacy Monte-Carlo estimate would sandwich it from below.

## 2. Privacy–utility Pareto: mixing gives privacy for free (`fig_privacy_utility.pdf`)

**Critique it answers:** "there is no demonstrated `(k,σ)` that is simultaneously
private and useful; obfuscation only helps accuracy where σ is negligible."

Merging the attack CSVs (Attack 2, the realistic uninformed adversary) with the
accuracy sweep shows the no-obfuscation point is **Pareto-dominated**:

| `(k,σ)` | Attack-2 leakage | vs no-obf (0.835) | test acc | vs baseline (60.19) |
|---|---|---|---|---|
| **(1, 0.10)** | **0.44** | **−47%** | **66.6%** | **+6.4 pp** |
| **(1, 0.01)** | **0.56** | **−32%** | **70.0%** | **+9.8 pp** |
| (5, 0.10) | 0.65 | −22% | 64.9% | +4.7 pp |

`(k=1, σ=0.10)` cuts leakage nearly in half **and** raises accuracy — a genuine win
over no obfuscation. The lever is **mixing (k), not noise (σ)**: mixing reduces
both the MI bound (Exp 1) and the empirical attack at no utility cost, while σ
trades utility for privacy only past the operable band. The paper currently buries
this as a "regularizer curiosity" in the utility section; it is the strongest
privacy–utility result in the paper and should be foregrounded with this figure.

Honest caveat: even at `(1,0.10)`, leakage 0.44 is still ≈55× the chance baseline
(0.008) — this is substantial leakage *reduction* plus a utility gain, not a claim
of strong absolute privacy. Frame it as a Pareto improvement, not as "private."

## What's still worth doing (in priority order)

3. **Report the attack as TPR@low-FPR / AUC** (the modern LiRA metric security
   reviewers expect) instead of only the weighted-overlap score. Needs logging the
   raw member vs non-member LiRA scores and computing an ROC. High credibility gain.
4. **Direct MI estimate** via the PAC-privacy Monte-Carlo estimator, to sandwich
   the bound of Exp 1 from below with the empirical attack advantage.
5. **Multi-seed error bars** on the headline cells (currently single-seed).
6. **Downstream utility of correlated noise** (the §7.7 open item): run the
   classifier under clip-correlated noise to show the Thm-16 defense keeps utility.

Files: `src/mi_bound_evaluation.py`, `results_revision/mi_bound_real.csv`,
`results_revision/privacy_utility_pareto.csv`, `images/fig_mi_bound_real.pdf`,
`images/fig_privacy_utility.pdf`.

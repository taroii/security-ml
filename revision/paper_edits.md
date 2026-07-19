# Paper Edits — drafted LaTeX for Paper #1783 revision

Paste-ready text for each change. Section numbers refer to the submitted
version; adjust to taste. Notation follows the submission (`M(X)=Π₁ M_mix X W + B`,
truth size `n`/`T`, calibration constant `C = Σ_j 1/D_j`).

---

## 1. Introduction — new quantitative paragraph (addresses A4, A6)

> **Our guarantees, concretely.** We instantiate two graded-correctness models.
> Under *half-integer* weights an adversary earns full credit for naming an exact
> member frame and half credit for naming any frame in a declared temporal
> neighborhood of a member; under *`1/m`-resolution* weights credit is graded in
> `m` steps by similarity. Relative to the exact-match (integer) baseline of
> prior work, half-integer credit raises the adversary's prior-success constant
> by ≈31× on our reference universe, which by our calibration (Theorem 7)
> demands ≈5.6× more noise standard deviation to preserve the same
> mutual-information guarantee. The size of this penalty is governed by the
> *temporal reach* of partial credit: as the credited neighborhood widens from
> ±1 to ±10 frames, the adversary's expected recovered fraction of the training
> set grows monotonically from 0.08 to 0.27 (§4.x, Fig. X). We further show that
> once membership is *clustered* into clips — as it is in every real video
> dataset — the required noise depends on the clip structure itself, and an
> i.i.d. calibration is optimistic in the upper tail that privacy actually
> bounds (§4.y).

---

## 2. Preliminaries — self-contained recap of [50] (addresses B1)

> **§2. Preliminaries: learnable obfuscation, recalled.** We summarize the
> machinery of Xiao et al. [50] that we build on, so this paper is
> self-contained. A data owner holds a dataset `X ∈ ℝ^{n×d}` and releases a
> transformed version `M(X)` to an untrusted trainer. The mechanism we adopt is
> the linear masking construction of [50, Alg. 1],
> \[ M(X) = Π₁\, M_{\mathrm{mix}}\, X\, W + B, \]
> where `M_mix` averages `k` same- and cross-class samples (class-`k` mixing),
> `Π₁` is a random row permutation, `W ∈ ℝ^{d×d'}` is a Gaussian random
> projection, and `B` is i.i.d. Gaussian noise with per-entry standard deviation
> `σ`. Privacy is measured information-theoretically: [50] bounds an adversary's
> posterior inference advantage by the mutual information `I(X; M(X))`, and for
> the isotropic-noise mechanism gives `I(X; M(X)) ≤ (d'/2)·min{(I),(II)}`
> (Theorem 5 here), which decreases in `σ`. The membership-inference difficulty
> is summarized by a single calibration constant
> \[ C = Σ_{j=1}^{n} 1/D_j, \qquad D_j = -\log(1-δ_0^{\ge j}), \]
> where `1-δ_0^{≥j}` is the prior probability that a random guess achieves
> weighted overlap at least `j`. Larger `C` means the prior success is high
> across thresholds, so more noise is needed; the calibration `σ ∝ sqrt(C)`
> (Theorem 7) converts a change in `C` into a change in required noise. All new
> results below are expressed through this same `C`.

---

## 3. Threat model / Definition 1 — the uniform-sampling clarification (addresses A1, B2)

> **On the sampling of `X`.** Definition 1 draws `X` uniformly from the universe
> `U`. We stress that this is a *modeling baseline chosen for comparability with
> [50]*, not a requirement on the data owner. It is, moreover, the conservative
> choice for the prior: uniform sampling scatters members across `U`, whereas a
> real owner assembles `X` from whole clips, which *concentrates* members and —
> as we show in §4.y — makes the adversary strictly stronger in the upper tail.
> Retaining the uniform baseline therefore does not weaken our threat model; the
> clustered model of §4.y strengthens it.

---

## 4. New §4.y — structured (two-level) sampling and the compound prior (addresses A1, B2)

> **§4.y. Clustered membership and the two-level prior.** Real video datasets are
> assembled at the *clip* level: the owner selects videos, and each contributes a
> block of temporally adjacent frames. We model this with two-level sampling: the
> universe is `G` clips of `L` frames each (`N = GL`); the owner selects `g`
> clips and `ell` frames from each, so `n = g·ell` members are *clustered* into
> `g` clips. A correlation-aware adversary guesses `g` clips and `ell` frames per
> clip.

> **Lemma (two-level compound prior).** *Under the two-level model with the
> half-integer rule, the adversary's weighted overlap is*
> \[ W = Σ_{i=1}^{H} W_i, \qquad H ∼ \mathrm{Hypergeometric}(G, g, g), \]
> *where the `W_i` are i.i.d. within-clip overlaps distributed as in Lemma 3
> with universe `L`, truth `ell`, and half-credit zone determined by the
> temporal window. Its distribution is the `H`-mixture of the `H`-fold
> convolution of the within-clip law, and it reduces to Lemma 3 when `G=1`.*

> *(Proof: partition guessed clips into matched/unmatched; unmatched clips
> contribute zero weight; matched clips contribute i.i.d. within-clip overlaps;
> `H` is the hypergeometric match count. Full proof in App. X; validated
> numerically against Monte Carlo to `2×10⁻⁴` and against Lemma 3 at `G=1` to
> `10⁻¹³`.)*

> **Consequence.** At an identical marginal membership rate `n/N` and identical
> total half-credit mass, the clustered and flat priors share the same *mean*
> overlap but differ sharply in the *tail*. On the reference universe
> (`G=100, g=50, L=100, ell=8`), `P(W ≥ 130)` rises from `6×10⁻⁵` (flat) to
> `0.024` (clustered), a ≈400× increase, and `P(W ≥ 140)` by ≈`10⁵×` (Fig. X,
> left). Since a guarantee bounds the tail, the flat i.i.d. calibration is
> optimistic exactly where privacy is defined. The gap grows with the universe:
> at `g/G = 0.5` the clustered-to-flat ratio of `C` crosses one and exceeds
> `10⁴×` as `G` grows (Fig. X). Thus the required noise is a function of the clip
> structure `(G, g, L, window)`, which no `(N, n)`-only analysis can capture.

---

## 5. New subsection — choosing the weights in practice (addresses B3)

> **Choosing the weights.** The weights are a *policy*, fixed a priori by the
> data owner, that declares what counts as a near-miss worth protecting. Two
> natural instantiations: (i) a **temporal window** `±w`, set to the duration of
> the event the owner wishes to hide (e.g. if a 0.5 s action at 30 fps must be
> concealed, `w ≈ 15` frames); every frame within `w` of a member earns half
> credit. (ii) a **semantic threshold** on an embedding distance, when the owner
> wishes to protect *what* is depicted rather than *when*. The choice is a
> privacy/utility knob, not a free parameter: our window sweep (Fig. X, right)
> shows the adversary's expected recovery growing monotonically with `w`, so a
> wider protected zone is strictly more expensive in noise. The owner picks `w`
> from the threat they care about and reads the required `σ` off Theorem 8.

---

## 6. Algorithm boxes (addresses B4)

```latex
\begin{algorithm}[t]
\caption{Obfuscation mechanism $\mathcal{M}$ (per [50], made explicit)}
\label{alg:mech}
\begin{algorithmic}[1]
\Require dataset $X\in\mathbb{R}^{n\times d}$, labels $Y$, mixing $k$,
         projection dim $d'$, noise level $\sigma$
\Ensure  released $(\mathcal{M}(X), \tilde Y)$
\State $M_{\mathrm{mix}} \gets$ class-$k$ mixing matrix from $Y$
       \Comment{averages $k$ same- and $k$ cross-class rows}
\State $W \sim \mathcal{N}(0, 1/d')^{d\times d'}$
       \Comment{fixed random projection}
\State $\Pi_1 \gets$ random row permutation;\; $\Pi_2 \gets$ random label perm.
\State $B \sim \mathcal{N}(0,\sigma^2)^{m\times d'}$
       \Comment{$\sigma$ enters HERE: additive Gaussian noise}
\State \Return $\mathcal{M}(X) = \Pi_1 M_{\mathrm{mix}} X W + B$,\;\;
       $\tilde Y = \Pi_1 M_{\mathrm{mix}} Y \Pi_2$
\end{algorithmic}
\end{algorithm}
```

```latex
\begin{algorithm}[t]
\caption{Blind temporal MIA (Attack 3, the i.i.d. adversary)}
\label{alg:blind}
\begin{algorithmic}[1]
\Require release $\mathcal{M}(X)$, candidate pool $P$ of $(\text{clip},\text{index})$
         rows, target size $T$
\Ensure  $T$ guessed $(\text{clip},\text{index})$ pairs
\State for each candidate $r\in P$: $\ell_r \gets \textsc{LiRA}(r; \mathcal{M}(X))$
       \Comment{per-frame likelihood ratio, scored independently}
\State \Return the $T$ candidates with the largest $\ell_r$
\end{algorithmic}
\end{algorithm}
```

```latex
\begin{algorithm}[t]
\caption{Correlation-aware temporal MIA (new)}
\label{alg:aware}
\begin{algorithmic}[1]
\Require release $\mathcal{M}(X)$, candidate pool $P$ grouped by clip, target $T$
\Ensure  $T$ guessed $(\text{clip},\text{index})$ pairs
\State for each $r\in P$: $\ell_r \gets \textsc{LiRA}(r; \mathcal{M}(X))$
       \Comment{same per-frame scores as Alg.~\ref{alg:blind}}
\State for each clip $c$: $s_c \gets \textstyle\sum_{r\in c}\ell_r$
       \Comment{pool correlated evidence across the clip}
\State $c^\star \gets \arg\max_c s_c$
       \Comment{identify the source clip}
\State \Return the $T$ highest-$\ell_r$ rows within $c^\star$ (fill from next
       clips if fewer than $T$)
\end{algorithmic}
\end{algorithm}
```

Table 3 caption addition: *"σ is the standard deviation of the Gaussian noise
`B` added in line 4 of Algorithm 1; larger σ increases `I(X;M(X))` protection at
the cost of downstream accuracy."*

---

## 7. New §5.x — correlated noise for joint-processing models (addresses A2)

> **§5.x. Noise design under dependence.** Independent per-frame noise is
> natural when frames are classified individually, but a model that jointly
> processes a clip — and, dually, an adversary that pools evidence across a
> clip's frames — is not best countered by independent noise. Consider a
> mechanism whose noise is `B_r = √(1-f)·ε_r + √f·η_{c(r)}`, where `ε_r` is
> per-frame and `η_c` is a per-clip shared draw; the total per-frame variance is
> `σ²` for any `f`, so the noise budget is unchanged. Under the correlation-aware
> attack (Alg. 3), the shared component `η_c` sums *coherently* across a clip's
> frames and cannot be averaged out, whereas independent noise partially
> cancels. Empirically (Experiment C), at matched budget and `rho=0.8`, moving
> from `f=0` (i.i.d.) to `f=1` (fully clip-shared) reduces the pooling
> adversary's success from 0.29 to 0.17, while the blind adversary is unaffected.
> Correlated noise is therefore the appropriate design for the joint-processing
> threat, at no extra budget.

---

## 8. Related work — DP for video and correlated data (addresses B6)

> **DP for video and correlated data.** Differentially private video analytics
> has been studied via query-release systems: Privid enforces event-duration
> privacy over untrusted analytic queries [Cangialosi22], and VideoDP releases a
> utility-driven private surrogate video for downstream analyses [Wang20]. These
> protect *query answers* over a fixed video; we instead protect the *training
> set* while preserving learnability, and our leakage measure is
> information-theoretic rather than DP. Our clustered-sampling analysis connects
> to the line showing DP degrades under data correlation: Dependent DP
> [Liu16] demonstrates Bayesian attacks that exploit tuple correlation,
> Pufferfish [Kifer14] gives a framework for privacy under correlated data, and
> DP under temporal correlations quantifies the amplified leakage in sequential
> data [Cao17]. Our two-level prior is the analogous statement for
> learnable obfuscation: correlation inflates the adversary's tail success and
> the required noise, and must be modeled in the sampling, not only the reward.

```bibtex
@inproceedings{Cangialosi22,
  title={Privid: Practical, Privacy-Preserving Video Analytics Queries},
  author={Cangialosi, Frank and Agarwal, Neil and Arun, Venkat and Narayana,
          Srinivas and Sarwate, Anand and Netravali, Ravi},
  booktitle={19th USENIX Symp. on Networked Systems Design and Implementation
             (NSDI 22)}, year={2022}}

@article{Wang20,
  title={VideoDP: A Flexible Platform for Video Analytics with Differential
         Privacy},
  author={Wang, Han and Xie, Shangyu and Hong, Yuan},
  journal={Proc. on Privacy Enhancing Technologies (PoPETs)},
  volume={2020}, number={4}, pages={277--296}, year={2020}}

@inproceedings{Liu16,
  title={Dependence Makes You Vulnerable: Differential Privacy Under Dependent
         Tuples},
  author={Liu, Changchang and Chakraborty, Supriyo and Mittal, Prateek},
  booktitle={Network and Distributed System Security Symp. (NDSS)}, year={2016}}

@article{Kifer14,
  title={Pufferfish: A Framework for Mathematical Privacy Definitions},
  author={Kifer, Daniel and Machanavajjhala, Ashwin},
  journal={ACM Trans. on Database Systems (TODS)},
  volume={39}, number={1}, year={2014}}

@inproceedings{Cao17,
  title={Quantifying Differential Privacy under Temporal Correlations},
  author={Cao, Yang and Yoshikawa, Masatoshi and Xiao, Yonghui and Xiong, Li},
  booktitle={IEEE Int. Conf. on Data Engineering (ICDE)}, year={2017}}
```

---

## 9. Exposition fixes (addresses A3, B5)

- **Table 1:** add a `$C_{\mathrm{half}}$` row label and mark the half-integer
  column, e.g. `\multicolumn{1}{c}{$C_{\mathrm{half}}$}`; the value is emitted by
  `theorem_calibration_table.py` (`C_half = 651.8`, `C_half/C_int = 31.38`,
  `σ_half/σ_int = 5.60`).
- **Table 2 / appendix sweep:** promote a condensed `(k,σ)` grid into the main
  text beside Table 2 with an explicit `\Cref{tab:full-sweep}` cross-reference to
  the appendix; add one sentence: *"The full sweep across all `(k,σ)` is in
  App.~A.1, Table 4."*
- **Figure 1 caption:** state the parameters inline (`N=1000, n=50, m=100,
  q=0.05`) and note the half-integer curve dominates the integer curve at every
  threshold.

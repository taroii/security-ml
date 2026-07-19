# Response to Reviewers — Paper #1783

We thank both reviewers. The reviews converge on one substantive point — that
our formal model motivates itself with temporal correlation but does not place
that correlation in the data distribution — and we have restructured the paper
around answering it. Concretely, we (1) introduce a two-level *clip-then-frame*
sampling model that puts dependence into the generative process and derive the
adversary's prior in closed form under it; (2) build the correlation-exploiting
membership-inference attack Reviewer A asked for and show whether it defeats the
i.i.d.-calibrated defense; and (3) analyze correlated noise as a mechanism-design
response. All new results are implemented and reproducible in the artifact.

Below we respond to each point. Quoted text is from the reviews.

---

## Reviewer A

> **"While the paper explicitly acknowledges that the data are not i.i.d., it
> appears to largely ignore the correlations present in the dataset ... the
> resulting guarantees seem disconnected from the practical security of the
> mechanism."**

We agree this was the central weakness, and we have addressed it in the model,
in theory, and empirically.

*In the model.* Real video datasets are assembled at the clip level: the data
owner picks videos, and each contributes a block of temporally adjacent frames.
We now model exactly this with a two-level scheme — select `g` clips from `G`,
then `ell` frames from each — so that membership is clustered rather than
scattered i.i.d. (new §4.x, Definition of structured sampling).

*In theory.* Under this model the adversary's weighted overlap is a **compound
distribution** `W = Σ_{i=1}^{H} W_i` with `H ~ Hypergeometric(G,g,g)` and `W_i`
the i.i.d. within-clip half-integer overlap of our Lemma 3. This **recovers the
paper's original Lemma 3 exactly when `G = 1`** (verified to 1.3e-13), so it
generalizes rather than replaces the prior analysis. The consequence is the key
point the review anticipated: at an *identical* marginal membership rate and
*identical* total half-credit mass, the clustered prior has the same mean
overlap as the flat i.i.d. prior but a far heavier **upper tail** — the
probability of recovering ≥130 of 400 units of overlap is ≈400× larger under
clustering, and ≥140 is ≈10⁵× larger. Because a privacy guarantee bounds the
tail, not the mean, the i.i.d. calibration is optimistic precisely where it
matters, and the required noise now depends on the clip structure that the flat
analysis cannot see (new Figure `fig_two_level_prior`).

> **"I expected the paper to evaluate privacy leakage using a standard
> membership inference attack, which could help demonstrate whether the privacy
> guarantees reported in [50] are substantially overestimated."**

Done. Our original Attack 3 scored candidate frames independently — the
i.i.d.-blind adversary. We add a **correlation-aware** adversary that pools the
same per-frame likelihood-ratio evidence across a clip before deciding (new
§6.x). Because uncontrolled real-video correlation makes the effect hard to
isolate, we complement the UCF-101 evaluation with a controlled study on
synthetic AR(1) clips whose intra-clip correlation is a single dial `rho`. We
find: (i) the likelihood-ratio score of a *non-member* frame rises as its
temporal distance to a member shrinks, with reach growing in `rho` — direct
evidence that near-miss proximity leaks; and (ii) at a fixed noise level the
correlation-aware attack's source-clip identification climbs from 0.75 at
`rho=0` to 1.0 at `rho≥0.9`, while the blind adversary the theory assumes lags
(0.05→0.87). This is the requested evidence that an i.i.d.-calibrated guarantee
can be optimistic for correlated video.

We also report honestly a limitation this exposed: on UCF-101 the effect is
*muted* because our per-clip **random frame sampling** scatters the members
across the clip, weakening the very intra-clip correlation the attack exploits.
The correlation-aware attack still identifies the source clip far above chance
(top-1 ≈ 0.47–0.57 of 400 candidates, ≈190–230× the 1/400 baseline) and gives a
small consistent edge over the blind attack, and the synthetic study isolates
the full effect. We now state this design tension explicitly rather than leaving
it implicit.

> **"Adding independent noise to each frame ... it is less clear that this
> approach is appropriate for models that must jointly process correlated
> frames."**

We add a correlated-noise mechanism variant (`noise_mode="clip"`) that injects
a shared per-clip noise component at matched total variance, and evaluate it
against the pooling adversary. At matched budget, clip-correlated noise
**reduces** the correlation-aware attack's success from 0.29 to 0.17 (fully
shared), while leaving the blind adversary unchanged: the shared component adds
coherently under the adversary's cross-frame pooling and cannot be averaged
out. Independent per-frame noise is therefore *not* the right design for
joint-processing models, and we now say so and quantify the better alternative
(new Experiment C).

> **Constructive: mark `C_half` in Table 1; state the half-integer and 1/m
> guarantees and the window dependence in the Intro; explain half-integer and
> 1/m in the Intro.**

All done. Table 1 now labels the `C_half` row and column explicitly. The
Introduction states the quantitative guarantees in plain terms (half-integer
credit inflates the prior constant ≈31× and the required noise ≈5.6× over the
integer baseline) and defines half-integer and `1/m` weights on first use. A new
**window sweep** quantifies how the number of credited nearby frames moves the
numbers (expected recovered fraction 0.08 at ±1 frame → 0.27 at ±10).

---

## Reviewer B

We appreciate the candor that the paper was hard to follow; the revision is
substantially more self-contained.

> **"Does not lay out its preliminaries very well, seems to assume a lot of
> familiarity with reference [50]."**

We added a self-contained Preliminaries section that states everything imported
from [50] — the mechanism `M(X)=Π₁ M_mix X W + B`, the mutual-information bound,
and the calibration constant `C` — with all notation defined before use.

> **"Why does Definition 1 necessitate that X is drawn uniform randomly from U?
> ... does the data owner need to assemble the dataset this way?"**

This is the same modeling gap Reviewer A identified, and we now treat it
head-on. Uniform sampling is retained only as the conservative baseline that
matches [50]; the realistic **clustered** sampling (clip-then-frame) is the new
default, and a threat-model paragraph explains that clustering makes the
adversary *stronger* (heavier tail), so uniform sampling is if anything
optimistic. It is not a restriction we impose on the data owner.

> **"How are the weights in Definition 1 decided in practice?"**

New subsection "Choosing the weights in practice": the weights encode the data
owner's declared protection target, fixed a priori — e.g. a temporal window
equal to the event duration the owner wishes to hide, or a semantic-similarity
threshold. The window sweep shows the privacy cost of widening the protected
zone, giving the owner an explicit knob.

> **"Please include clearly demarcated algorithm boxes ... I do not understand
> how sigma impacts the mechanism, and as a result I find Table 3 hard to
> interpret."**

We added three algorithm boxes (mechanism, blind Attack 3, correlation-aware
Attack 3) with explicit inputs/outputs, and marked the exact line where the
Gaussian noise `B` (scaled by σ) enters the release. Table 3's caption now
points to that line so the σ dependence is unambiguous.

> **"Why in Table 2 do you only display a single (k, σ)?"**

The full `(k, σ)` sweep was in Appendix Table 4; we have promoted a condensed
version into the main text beside Table 2 and fixed the cross-reference (the
appendix pointer was easy to miss).

> **"Compare your approach to DP approaches for video data, not just general
> DP."**

Added a related-work paragraph covering Privid (NSDI 2022), VideoDP (PoPETs
2020), and the correlated-data DP line — Dependent DP (NDSS 2016), Pufferfish
(TODS 2014), and DP under temporal correlations (ICDE 2017) — situating our
information-theoretic, learnability-preserving approach against these
query-release and DP-based methods.

---

We believe these changes convert the review's central criticism into the
paper's contribution: we now model the dependence, attack it, and calibrate
against it, and we show both where an i.i.d. analysis is optimistic and how to
fix the mechanism.

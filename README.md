# Learnable Obfuscation for Temporally Related Video Data

Code supplement for the paper. The pipeline sweeps the obfuscation knobs
`(k, sigma)`, scores membership-inference attacks on the resulting embeddings,
and measures downstream classifier accuracy.

**This repository ships code only.** Every CSV, figure, and table in the paper
and the technical supplement is produced by running the scripts below, so there
is exactly one source of truth for each number. Outputs land in `results/`,
`results_revision/`, `accuracy_results/`, and `images/`, all of which are
gitignored.

## Attack naming: code vs. paper

The code implements three attacks; the paper reports two of them. **Read this
table before opening any output CSV** — the column whose name contains
`attack2` is *not* the paper's Attack 2.

| Paper | CSV column | Description |
| --- | --- | --- |
| Attack 1 (index inference), weighted | `attack1_score_window` | exact (clip, index) match = 1; same clip within ±2 indices of a true frame = 0.5 |
| Attack 1 (index inference), binary | `attack1_score_int` | exact match only |
| Attack 2 (frame-level membership), weighted | `attack3_score_half` | top-`T` over a subsampled universe pool ∪ the target rows |
| Attack 2 (frame-level membership), binary | `attack3_score_int` | exact match only |
| *not reported* | `attack2_top1` | clip-level top-1 over 50 same-class candidate clips |
| *not reported* | `attack1_score_half` | a **different** scoring rule: same-clip-*any*-index earns 0.5, with no ±2 window |

Two consequences worth stating explicitly:

- At `(k, sigma) = (1, 0.10)` the CSV shows `attack2_top1 = 0.860` next to
  `attack3_score_half = 0.439`. The paper's "Attack 2 = 0.439" is the latter.
  `attack2_top1` is a clip-identification diagnostic that the paper does not
  report.
- `attack1_score_half` is retained only as the `window = 0` degenerate case of
  the same scoring function. It is **not** the rule behind any reported number.

Attack 2's weighted and binary scores coincide in every cell
(`attack3_score_half == attack3_score_int`) because its subsampled pool
contains only exact target rows, with no near-miss frames to earn partial
credit — this is the effect explained in the caption of main-paper Table 2.

### Baseline columns

| Column | Meaning |
| --- | --- |
| `attack3_baseline_half` | **operational** chance level, `T / \|pool\|` = 16/2016 ≈ `0.0079`. This is the ≈0.008 baseline the paper reports for Attack 2. |
| `attack3_baseline_universe` | universe-level chance level over all (clip, index) pairs, ≈ `7.3e-05`. Not reported in the paper. |
| `attack2_baseline` | `1 / n_clip_candidates` = 0.02, for the unreported `attack2_top1`. |
| `attack2_baseline_universe` | `1 / n_universe_clips` ≈ `1.26e-04`. |
| `attack1_baseline_window` | closed-form ±2-window chance level, ≈ `0.373`. |

## Setup

```
conda create -n security python=3.11.14
conda activate security
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions listed in Appendix E.1 of the
technical supplement. The reported runs used the PyTorch MPS backend on Apple
silicon; no CUDA GPU is required. On a CUDA machine, install
`torch`/`torchvision` from https://pytorch.org/get-started/locally/ first.

UCF-101 is downloaded automatically on first run (`src/main_video.py` and
`src/membership_inference.py` both call `download_ucf101`), into `data/`.

## Running the pipeline

```
bash run_temporal_mia.sh
```

Four phases, individually runnable:

```
bash run_temporal_mia.sh smoke      # 1-cell, 2-trial sanity check
bash run_temporal_mia.sh mia        # MIA sweep across (k, sigma)
bash run_temporal_mia.sh accuracy   # downstream accuracy sweep
bash run_temporal_mia.sh plots      # regenerate figures from existing CSVs
```

- **`mia`** — runs `src/membership_inference.py` per `k`, sweeping all
  `SIGMAS`. Per-cell CSVs land in `results/`; `src/aggregate.py` collates them.
  Backs main-paper Table 2 and supplement Table 6.
- **`accuracy`** — runs `src/main_video.py` per `(k, sigma)` cell for
  downstream Transformer accuracy on obfuscated UCF-101. The first cell builds
  the embedding cache (slow); the rest reuse it. CSVs land in
  `accuracy_results/`. Backs supplement Table 1.
- **`plots`** — re-aggregates both sweeps, builds the privacy–utility frontier
  (supplement Table 4 + Figure 6), and runs the closed-form numerics
  (main Table 1, main Figure 1, supplement Table 7).

The remaining artifacts are produced by running their script directly; see the
table below.

### Compute knobs

Override via env vars:

| Variable | Default | Meaning |
| --- | --- | --- |
| `N_TARGETS` | `10` | MIA targets per `(k, sigma)` cell |
| `N_TRIALS` | `5` | MC trials per target |
| `N_CLIP_CANDIDATES` | `50` | clip-level candidate pool size |
| `N_FRAME_CANDIDATES_A3` | `2000` | frame-level universe subsample size |
| `K_VALUES` | `"0 1 5"` | space-separated `k` list |
| `SIGMAS` | `"0.01 0.05 0.10 0.50"` | space-separated sigma list |
| `ACC_PARALLEL` | `1` | accuracy jobs in parallel |
| `SKIP_ACCURACY` | unset | skip the accuracy sweep if set |
| `CONDA_ENV` | `security` | conda env to run python in |

```
N_TARGETS=50 N_TRIALS=10 bash run_temporal_mia.sh mia
K_VALUES="0 1" SIGMAS="0.01 0.10" bash run_temporal_mia.sh
```

Note that `run_attack` is seeded with `SEED + i` for the *i*-th sigma in
`--sigmas`, so changing the order of the sigma list changes the target and
candidate-pool draws.

### Multi-seed accuracy runs

`src/main_video.py --seed S` reseeds every stochastic component of the utility
pipeline (frame-index sampling, the stratified subset, class-`k` mixing, the
projection and the additive noise) and tags the output filename with `S` for
any `S != 42`, so seeds do not overwrite one another:

```
for s in 42 43 44; do
  for k in 0 1 5; do
    for sig in 0.01 0.05 0.10; do
      python src/main_video.py --k $k --sigma $sig --seed $s
    done
  done
done
python src/aggregate.py --mode accuracy   # writes merged_accuracy_by_seed.csv
```

## The ten modules

Every table and figure in the paper and supplement is produced by one of these.

| Module | Produces |
| --- | --- |
| `membership_inference.py` | The mechanism and the three attacks. `results/attack_k*_sigma*.csv` — **main Table 2**, **supplement Table 6**. |
| `main_video.py` | Utility pipeline: embedding extraction, obfuscation, Transformer training. `accuracy_results/*.csv` — **supplement Table 1**. |
| `theory_numerics.py` | Closed-form prior/calibration numerics, no dataset needed. **main Table 1** + `images/table_calibration.tex`, **main Figure 1**, **supplement Table 7**. `--mode {calibration,prior-figure,frame-survival}`. |
| `mi_bound_evaluation.py` | Log-determinant functional on real embeddings. **supplement Table 2 + Figure 2**. |
| `lira_roc.py` | LiRA ROC / TPR at low FPR. **supplement Table 3 + Figure 3**. Modes `run` \| `plot` \| `verify`. |
| `correlation_aware_attack.py` | Blind vs. correlation-aware adversary. **supplement Table 5**. |
| `synthetic_correlation.py` | Controlled AR(1) study. **supplement Figure 4**. Needs no dataset. |
| `embedding_autocorrelation.py` | ρ(τ) on real frame embeddings. **supplement §E.6 + Figure 1**. |
| `visual_obfuscation.py` | Pixel-space visualization of the mechanism. **supplement Figure 5**. |
| `aggregate.py` | Collates per-cell CSVs and joins them into the privacy–utility frontier. `merged_results.csv`, `merged_accuracy.csv`, **supplement Table 4 + Figure 6**. `--mode {attacks,accuracy,pareto}`. |

Only `membership_inference.py` and `main_video.py` require UCF-101;
`theory_numerics.py` and `synthetic_correlation.py` need no data at all, and
the rest read artifacts the first two produce.

### Re-rendering without recomputing

```
python src/mi_bound_evaluation.py --replot-only   # Figure 2 from the CSV
python src/lira_roc.py plot                       # Figure 3 from saved curves
python src/aggregate.py --mode pareto             # Table 4 + Figure 6 from CSVs
```

### Checking results

```
python src/lira_roc.py verify
```

Sanity-checks the ROC output: AUCs in range and consistent between the pooled
and per-query estimators, AUC non-increasing in σ, the hardness ordering
(informed ≤ same-class ≤ cross-class) intact, TPR@0.1% ≤ TPR@1%, and ROC curves
monotone from 0 to 1. Exits non-zero on a hard failure.

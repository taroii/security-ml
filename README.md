# Learnable Obfuscation for Temporally Related Video Data

This repo contains the experiments backing our paper on temporal membership-inference attacks (MIA) against learnable obfuscation of video data. The pipeline sweeps the obfuscation knobs `(k, sigma)`, scores three MIA variants on the resulting embeddings, and measures the downstream classifier accuracy gap.

## CCS revision (round-1 response)

The `revision/` directory and the new scripts below address the CCS 2026-B
reviews. The central criticism was that the paper motivated itself with
temporal correlation but modeled it only in the reward, not the sampling.
The revision adds:

- **`src/two_level_prior.py`** — a two-level clip-then-frame *compound prior*
  (validated against Monte Carlo and recovering the paper's Lemma 3 at `G=1`),
  showing clustered membership inflates the adversary's upper-tail success.
- **`src/correlation_aware_attack.py`** — a clip-aggregated MI attack that pools
  per-frame evidence, vs. the paper's blind per-frame attack, on UCF-101.
- **`src/synthetic_correlation.py`** — a controlled AR(1) study (dial `rho`)
  isolating graded leakage, the aware-vs-blind gap, and correlated noise.
- **`src/membership_inference.py`** — now supports `--noise-mode {iid,clip}`
  (correlated per-clip noise; backward compatible, default `iid`).

Deliverables: `revision/REVISION_PLAN.md` (comment-by-comment mapping),
`revision/response_to_reviewers.md`, and `paper/main.tex` (revised paper source;
see `paper/REVISION_NOTES.md` for how to integrate). New figures:
`images/fig_two_level_prior.pdf`, `images/fig_synthetic_correlation.pdf`.

## Setup

### 1. Create a conda environment

```
conda create -n security python=3.11
conda activate security
```

### 2. Install PyTorch with CUDA

Follow https://pytorch.org/get-started/locally/ for the right command for your platform. For example:

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

If you don't have a GPU, install `torch` and `torchvision` normally.

### 3. Install remaining dependencies

```
pip3 install -r requirements.txt
```

## Running the pipeline

The full sweep (membership-inference attacks, downstream classification accuracy sweep, and figures) is driven by a single script:

```
bash run_temporal_mia.sh
```

The script has four phases — you can run them individually:

```
bash run_temporal_mia.sh smoke      # 1-cell, 2-trial sanity check
bash run_temporal_mia.sh mia        # MIA sweep across (k, sigma)
bash run_temporal_mia.sh accuracy   # downstream accuracy sweep
bash run_temporal_mia.sh plots      # regenerate figures from existing CSVs
```

### What each phase produces

- **`mia`** — runs `src/membership_inference.py` for each `k` in `K_VALUES`, sweeping all `SIGMAS`. Three attacks are scored per cell (Attack 1: half-integer detection on a single clip; Attack 2: same-class clip identification; Attack 3: frame-of-origin identification across the universe). Per-cell CSVs land in `results/`, and `src/merge_results.py` collates them into `merged_results.csv`.
- **`accuracy`** — runs `src/main_video.py` for each `(k, sigma)` cell to measure downstream Transformer-classifier accuracy on obfuscated UCF-101. The first cell builds the embedding cache (slow); the rest reuse it. Per-cell CSVs land in `accuracy_results/`, merged into `merged_accuracy.csv`.
- **`plots`** — produces the paper figures into `images/`:
  - `fig2_mia_robustness.pdf` (Figure 2): MIA scores vs. sigma for each k, all three attacks.
  - `fig3_int_vs_half.pdf` (Figure 3): integer-only vs. half-integer Attack 1 scores.
  - `fig4_pareto.pdf` (Figure 4): privacy/utility Pareto frontier (MIA score vs. accuracy gap).

### Standalone figures

Two figures live outside the main sweep:

- **Figure 1** (`fig1_prior_success.pdf`) — analytical plot of prior attacker success vs. obfuscation parameters. Regenerate with:
  ```
  python src/figure1_prior_success.py
  ```
- **Figure 5** (`figure5_visual_obfuscation_pixel.pdf`) — qualitative pixel-space visualization (Basketball vs. Skiing under `k=5`, `sigma=0.1`). Regenerate with:
  ```
  python src/figure5_visual_obfuscation.py
  ```

### Compute knobs

The defaults in `run_temporal_mia.sh` are intentionally small for quick iteration. Override via env vars to scale up:

| Variable | Default | Meaning |
| --- | --- | --- |
| `N_TARGETS` | `10` | MIA targets per `(k, sigma)` cell |
| `N_TRIALS` | `5` | MC trials per target |
| `N_CLIP_CANDIDATES` | `50` | Attack 2 same-class candidate pool size |
| `N_FRAME_CANDIDATES_A3` | `2000` | Attack 3 universe subsample size |
| `K_VALUES` | `"0 1 5"` | space-separated `k` list |
| `SIGMAS` | `"0.01 0.05 0.10 0.50"` | space-separated sigma list |
| `ACC_PARALLEL` | `1` | accuracy jobs in parallel |
| `SKIP_ACCURACY` | unset | skip the accuracy sweep if set |
| `CONDA_ENV` | `security` | conda env to run python in |

Examples:

```
N_TARGETS=50 N_TRIALS=10 bash run_temporal_mia.sh mia
K_VALUES="0 1" SIGMAS="0.01 0.10" bash run_temporal_mia.sh
```

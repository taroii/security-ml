# Learnable Obfuscation for Temporally Related Video Data

This repo contains the experiments backing our paper on temporal membership-inference attacks (MIA) against learnable obfuscation of video data. The pipeline sweeps the obfuscation knobs `(k, sigma)`, scores three MIA variants on the resulting embeddings, and measures the downstream classifier accuracy gap.

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

The full sweep (membership-inference attacks, accuracy sweep, and figures) is driven by a single script:

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

## Repository layout

```
src/                            pipeline source
  membership_inference.py       Attack 1/2/3 implementations
  main_video.py                 downstream Transformer classifier (accuracy)
  figure1_prior_success.py      Figure 1 (standalone analytical plot)
  figure2_mia_robustness.py     Figure 2
  figure3_integer_vs_half.py    Figure 3
  figure4_pareto.py             Figure 4
  figure5_visual_obfuscation.py Figure 5 (standalone pixel-space visualization)
  merge_results.py              merge per-cell MIA CSVs
  merge_accuracy.py             merge per-cell accuracy CSVs
results/                        per-cell MIA CSVs (attack_k*_sigma*.csv)
accuracy_results/               per-cell accuracy CSVs (acc_k*_sigma*.csv, baseline.csv)
images/                         paper figures (fig2/3/4)
logs/                           per-cell stdout/stderr from each run
old/                            superseded scripts and figures (kept for reference)
run_temporal_mia.sh             pipeline driver
```

# Revision package — Paper #1783

Everything produced to address the CCS 2026-B reviews, in one place.

## Documents (read in this order)
1. **[REVISION_PLAN.md](REVISION_PLAN.md)** — every reviewer comment → the change
   that answers it, with the result numbers. Start here.
2. **[response_to_reviewers.md](response_to_reviewers.md)** — the rebuttal letter.
3. **[paper_edits.md](paper_edits.md)** — paste-ready LaTeX for each paper change
   (intro, preliminaries, threat model, two-level prior, algorithm boxes,
   correlated-noise section, related work, exposition fixes).

## New / changed code (`../src/`)
| file | what it adds | reviewer point |
|------|--------------|----------------|
| `two_level_prior.py` | clip-then-frame compound prior; validated vs Monte Carlo and vs Lemma 3 at G=1; window sweep; figure+table | A1, A4 |
| `correlation_aware_attack.py` | clip-aggregated Attack 3 vs blind Attack 3 on UCF-101 | A1 |
| `synthetic_correlation.py` | controlled-ρ study: graded leakage, aware-vs-blind, iid-vs-clip noise | A1, A2 |
| `membership_inference.py` | `noise_mode={iid,clip}` correlated-noise option (backward compatible; default iid) | A2 |

## Figures (`../images/`)
- `fig_two_level_prior.pdf` — clustered vs i.i.d. prior tail; window sweep.
- `fig_synthetic_correlation.pdf` — leakage-vs-distance; aware-vs-blind-vs-ρ; noise modes.

## Result data
- `../results_revision/*.csv` — machine-readable attack/leakage tables.
- `*_results.txt` here — raw run logs (two-level, synthetic, real-data).

## Reproduce
```bash
# env (Python 3.11); dataset already under ../data/UCF-101
python src/two_level_prior.py --G 100 --g 50 --L 100 --ell 8 --mc
python src/synthetic_correlation.py           # ~2 min, no dataset needed
python src/correlation_aware_attack.py --k 0 --sigmas 0.05 0.10 0.25 0.50 \
       --n-universe-clips 400 --agg sum        # needs UCF-101 frame-pool cache
python src/membership_inference.py --k 0 --noise-mode clip --noise-clip-frac 1.0
```

## Headline numbers (all validated)
- Two-level prior recovers Lemma 3 at G=1 (Δ=1.3e-13); matches Monte Carlo (Δ=2e-4).
- Clustered vs i.i.d. at matched mean: `P(recover ≥130/400)` = 0.024 vs 6e-5 (**≈400×**).
- Half-integer vs integer: `C` ratio 31.4×, noise ratio 5.60× (confirms Table 1).
- Synthetic: source-clip ID rises 0.75→1.00 as ρ 0→0.95 (blind adversary 0.05→0.87).
- Correlated noise cuts pooling-attack success 0.29→0.17 at matched budget.
- Real UCF-101: clip top-1 ≈ 0.47–0.57 of 400 (≈190–230× chance); modest gain over
  blind, muted by the paper's random per-clip sampling (reported honestly).

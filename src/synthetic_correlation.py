"""Synthetic controlled-correlation study of temporal membership leakage.

Two empirical questions motivate this study:

  (1) the analysis credits near-miss guesses, but does temporal proximity
      actually leak membership? and
  (2) is independent per-frame noise appropriate when a model jointly
      processes correlated frames?

On real video the correlation is uncontrolled, so the effect cannot be
isolated. Here we generate synthetic clips whose intra-clip temporal
correlation is a single dial rho in [0, 1): each clip is an AR(1) sequence

    x_t = rho * x_{t-1} + sqrt(1 - rho^2) * eps_t,   eps_t ~ N(0, I_{d0}),

so that Corr(x_s, x_t) = rho^{|s - t|} exactly. rho = 0 is the i.i.d. frame
model the paper's theory assumes; rho -> 1 is a nearly static video. We push
these clips through the paper's own mechanism (M(X) = Pi_1 M_mix X W + B, via
membership_inference.precompute_trial) and its LiRA scoring, and measure:

  Experiment A -- graded leakage:  the likelihood-ratio score of a NON-member
     frame as a function of its temporal distance to the nearest true member.
     If proximity leaks, the score decays with distance; the theory predicts
     the decay tracks rho^{distance}. rho = 0 should be flat (no leakage),
     directly demonstrating that the partial-credit reward the paper defines
     corresponds to real, measurable leakage.

  Experiment B -- correlation-aware vs blind attack across rho:  the
     clip-aggregated adversary's advantage over the i.i.d. adversary as a
     function of rho, showing the flat analysis is increasingly optimistic as
     correlation grows.

  Experiment C -- i.i.d. vs temporally-correlated noise:  at matched noise
     budget, whether shaping the mechanism noise to be correlated within a
     clip defends the joint-processing adversary better than independent
     per-frame noise -- the design question raised by (2) above, and the
     empirical counterpart of Theorem S8 (privacy-preserving correlated
     noise) in Appendix B.5 of the supplement.

Self-contained: needs no dataset, runs in a couple of minutes, reuses the
audited mechanism and scoring from membership_inference.py.
"""
import argparse
import os
from typing import Tuple

import numpy as np

from membership_inference import (
    precompute_trial,
    precompute_target_lira,
    _lira_score_external_uw,
    lira_score_frame_candidates,
    compute_weighted_score,
)
from correlation_aware_attack import clip_aggregated_scores


def make_ar1_universe(
    n_clips: int,
    L: int,
    d0: int,
    rho: float,
    n_classes: int,
    T: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic universe of AR(1) clips.

    Each clip is a length-L AR(1) sequence in R^{d0} with lag-1 correlation
    rho (so Corr(frame_s, frame_t) = rho^{|s-t|}). We sample T frame indices
    per clip into the pooled universe U (mirroring the real pipeline), and
    also return the FULL clips so Experiment A can score every index by its
    temporal distance to a member.

    Returns
      U            : (n_clips*T, d0)  pooled frames (T per clip)
      clip_ids     : (n_clips*T,)     clip id of each pooled row
      frame_indices: (n_clips*T,)     index in [0,L) of each pooled row
      labels       : (n_clips*T,)     class label of each pooled row
      full_clips   : (n_clips, L, d0) every frame of every clip
    """
    full_clips = np.empty((n_clips, L, d0), dtype=np.float32)
    a = float(rho)
    b = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    for cflip in range(n_clips):
        eps = rng.standard_normal((L, d0)).astype(np.float32)
        x = np.empty((L, d0), dtype=np.float32)
        x[0] = eps[0]
        for t in range(1, L):
            x[t] = a * x[t - 1] + b * eps[t]
        full_clips[cflip] = x

    U_rows, cids, fidx, labs = [], [], [], []
    for cflip in range(n_clips):
        idx = np.sort(rng.choice(L, size=T, replace=False))
        U_rows.append(full_clips[cflip, idx])
        cids.append(np.full(T, cflip, dtype=int))
        fidx.append(idx.astype(int))
        labs.append(np.full(T, cflip % n_classes, dtype=int))
    U = np.concatenate(U_rows, axis=0).astype(np.float32)
    clip_ids = np.concatenate(cids)
    frame_indices = np.concatenate(fidx)
    labels = np.concatenate(labs)
    return U, clip_ids, frame_indices, labels, full_clips


# ----------------------------------------------------------------------------
# Experiment A: leakage vs temporal distance to nearest member.
# ----------------------------------------------------------------------------
def experiment_A(
    rhos, n_clips, L, d0, T, n_classes, d, sigma,
    n_targets, n_trials, max_dist, seed,
):
    """Mean (standardized) LiRA score of a frame vs its temporal distance to
    the nearest member, for each rho. Distance 0 = member."""
    print("\n=== Experiment A: graded leakage vs temporal distance ===")
    # accumulate per (rho, distance): sum of standardized scores and count
    curves = {}
    for rho in rhos:
        rng = np.random.default_rng(seed + int(rho * 1000))
        U, clip_ids, frame_indices, labels, full = make_ar1_universe(
            n_clips, L, d0, rho, n_classes, T, rng
        )
        clip_rows = [np.where(clip_ids == v)[0] for v in range(n_clips)]
        class_to_rows = {int(l): np.where(labels == l)[0]
                         for l in np.unique(labels)}
        c = n_classes

        dist_sum = np.zeros(max_dist + 1)
        dist_cnt = np.zeros(max_dist + 1)

        tgt_vids = rng.choice(n_clips, size=min(n_targets, n_clips),
                              replace=False)
        for vid in tgt_vids:
            target_rows = clip_rows[vid]
            member_idx = set(frame_indices[target_rows].tolist())
            member_pos = np.array(sorted(member_idx))
            label = int(labels[target_rows][0])
            for trial_idx in range(n_trials):
                trng = np.random.default_rng(
                    seed * 10**6 + int(vid) * 100 + trial_idx)
                trial = precompute_trial(U, labels, c, 0, d, sigma, trng)
                W = trial["W"]
                non_target = np.setdiff1d(class_to_rows[label], target_rows)
                repl = len(non_target) < T
                cf = trng.choice(non_target, size=T, replace=repl)
                tpre = precompute_target_lira(trial, target_rows, cf)

                # score every frame of the target clip's FULL sequence
                full_W = full[vid].astype(np.float32) @ W        # (L, d)
                scores = _lira_score_external_uw(trial, tpre, full_W, sigma)
                # standardize within this clip/trial so curves are comparable
                s = (scores - scores.mean()) / (scores.std() + 1e-9)
                for t in range(L):
                    dnear = int(np.min(np.abs(member_pos - t)))
                    if dnear <= max_dist:
                        dist_sum[dnear] += s[t]
                        dist_cnt[dnear] += 1
        mean_curve = dist_sum / np.maximum(dist_cnt, 1)
        curves[rho] = mean_curve
        # print a compact summary
        head = "  ".join(f"d{d_}={mean_curve[d_]:+.2f}"
                         for d_ in [0, 1, 2, 3, 5, 10] if d_ <= max_dist)
        print(f"  rho={rho:.2f}:  {head}")
    return curves


# ----------------------------------------------------------------------------
# Experiment B: correlation-aware vs blind attack across rho.
# ----------------------------------------------------------------------------
def experiment_B(
    rhos, n_clips, L, d0, T, n_classes, d, sigma,
    n_targets, n_trials, agg, seed,
):
    print("\n=== Experiment B: correlation-aware vs blind attack vs rho ===")
    out = []
    for rho in rhos:
        rng = np.random.default_rng(seed + 7 + int(rho * 1000))
        U, clip_ids, frame_indices, labels, _ = make_ar1_universe(
            n_clips, L, d0, rho, n_classes, T, rng
        )
        clip_rows = [np.where(clip_ids == v)[0] for v in range(n_clips)]
        class_to_rows = {int(l): np.where(labels == l)[0]
                         for l in np.unique(labels)}
        c = n_classes
        blind, aware, top1 = [], [], []
        tgt_vids = rng.choice(n_clips, size=min(n_targets, n_clips),
                              replace=False)
        pool = np.arange(U.shape[0])  # whole universe; each clip has T rows
        for vid in tgt_vids:
            target_rows = clip_rows[vid]
            label = int(labels[target_rows][0])
            truth = list(zip(np.full(T, vid).tolist(),
                             frame_indices[target_rows].tolist()))
            for trial_idx in range(n_trials):
                trng = np.random.default_rng(
                    seed * 10**6 + int(vid) * 100 + trial_idx + 1)
                trial = precompute_trial(U, labels, c, 0, d, sigma, trng)
                non_target = np.setdiff1d(class_to_rows[label], target_rows)
                repl = len(non_target) < T
                cf = trng.choice(non_target, size=T, replace=repl)
                tpre = precompute_target_lira(trial, target_rows, cf)
                lr = lira_score_frame_candidates(trial, tpre, pool, sigma)

                top_blind = pool[np.argsort(-lr)[:T]]
                gb = [(int(clip_ids[r]), int(frame_indices[r]))
                      for r in top_blind]
                blind.append(compute_weighted_score(gb, truth, "half") / T)

                cs = clip_aggregated_scores(pool, lr, clip_ids, agg=agg)
                ranked = sorted(cs.items(), key=lambda kv: -kv[1])
                best = ranked[0][0]
                top1.append(1.0 if best == vid else 0.0)
                brows = pool[clip_ids[pool] == best]
                blr = lr[clip_ids[pool] == best]
                chosen = brows[np.argsort(-blr)[:T]]
                ga = [(int(clip_ids[r]), int(frame_indices[r]))
                      for r in chosen]
                aware.append(compute_weighted_score(ga, truth, "half") / T)
        row = {
            "rho": rho, "blind_half": float(np.mean(blind)),
            "aware_half": float(np.mean(aware)),
            "clip_top1": float(np.mean(top1)),
            "gain": float(np.mean(aware) - np.mean(blind)),
        }
        out.append(row)
        print(f"  rho={rho:.2f}:  blind={row['blind_half']:.3f}  "
              f"aware={row['aware_half']:.3f}  gain={row['gain']:+.3f}  "
              f"clip_top1={row['clip_top1']:.3f}")
    return out


# ----------------------------------------------------------------------------
# Experiment C: i.i.d. vs clip-correlated noise at matched budget.
# ----------------------------------------------------------------------------
def experiment_C(
    rho, n_clips, L, d0, T, n_classes, d, sigma,
    clip_fracs, n_targets, n_trials, agg, seed,
):
    print(f"\n=== Experiment C: i.i.d. vs clip-correlated noise (rho={rho}) ===")
    rng0 = np.random.default_rng(seed + 99)
    U, clip_ids, frame_indices, labels, _ = make_ar1_universe(
        n_clips, L, d0, rho, n_classes, T, rng0
    )
    clip_rows = [np.where(clip_ids == v)[0] for v in range(n_clips)]
    class_to_rows = {int(l): np.where(labels == l)[0]
                     for l in np.unique(labels)}
    c = n_classes
    pool = np.arange(U.shape[0])
    tgt_vids = rng0.choice(n_clips, size=min(n_targets, n_clips),
                           replace=False)

    out = []
    for mode, frac in ([("iid", 0.0)] +
                       [("clip", f) for f in clip_fracs]):
        aware, blind, top1 = [], [], []
        for vid in tgt_vids:
            target_rows = clip_rows[vid]
            label = int(labels[target_rows][0])
            truth = list(zip(np.full(T, vid).tolist(),
                             frame_indices[target_rows].tolist()))
            for trial_idx in range(n_trials):
                trng = np.random.default_rng(
                    seed * 10**6 + int(vid) * 100 + trial_idx + 3)
                trial = precompute_trial(
                    U, labels, c, 0, d, sigma, trng,
                    noise_mode=mode, clip_ids=clip_ids, noise_clip_frac=frac)
                non_target = np.setdiff1d(class_to_rows[label], target_rows)
                repl = len(non_target) < T
                cf = trng.choice(non_target, size=T, replace=repl)
                tpre = precompute_target_lira(trial, target_rows, cf)
                lr = lira_score_frame_candidates(trial, tpre, pool, sigma)
                top_blind = pool[np.argsort(-lr)[:T]]
                gb = [(int(clip_ids[r]), int(frame_indices[r]))
                      for r in top_blind]
                blind.append(compute_weighted_score(gb, truth, "half") / T)
                cs = clip_aggregated_scores(pool, lr, clip_ids, agg=agg)
                ranked = sorted(cs.items(), key=lambda kv: -kv[1])
                best = ranked[0][0]
                top1.append(1.0 if best == vid else 0.0)
                brows = pool[clip_ids[pool] == best]
                blr = lr[clip_ids[pool] == best]
                chosen = brows[np.argsort(-blr)[:T]]
                ga = [(int(clip_ids[r]), int(frame_indices[r]))
                      for r in chosen]
                aware.append(compute_weighted_score(ga, truth, "half") / T)
        tag = "iid" if mode == "iid" else f"clip f={frac:.2f}"
        row = {"noise": tag, "aware_half": float(np.mean(aware)),
               "blind_half": float(np.mean(blind)),
               "clip_top1": float(np.mean(top1))}
        out.append(row)
        print(f"  {tag:>12}:  aware={row['aware_half']:.3f}  "
              f"blind={row['blind_half']:.3f}  clip_top1={row['clip_top1']:.3f}")
    return out


def make_figures(curves, expB, expC, max_dist, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[skip figures] {e}")
        return
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # A: leakage vs distance. Restrict to distances with enough support that
    # the standardized mean is stable (far tails have few frames and are noisy).
    ax = axes[0]
    plot_d = min(8, max_dist)
    ds = np.arange(0, plot_d + 1)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(curves)))
    for (rho, curve), col in zip(sorted(curves.items()), cmap):
        ax.plot(ds, curve[:plot_d + 1], "o-", color=col, ms=3,
                label=f"$\\rho$={rho:.2f}")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("temporal distance to nearest member")
    ax.set_ylabel("standardized LiRA score")
    ax.set_title("A. Graded leakage vs proximity\n"
                 "($\\rho$=0: only members leak; higher $\\rho$ leaks farther)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # B: aware vs blind vs rho
    ax = axes[1]
    rr = [r["rho"] for r in expB]
    ax.plot(rr, [r["blind_half"] for r in expB], "s-", color="#c0392b",
            label="blind (i.i.d. adversary)")
    ax.plot(rr, [r["aware_half"] for r in expB], "o-", color="#2c3e50",
            label="correlation-aware")
    ax.plot(rr, [r["clip_top1"] for r in expB], "^--", color="#27ae60",
            label="aware clip top-1 acc")
    ax.set_xlabel(r"intra-clip correlation $\rho$")
    ax.set_ylabel("attack success (half-integer / acc)")
    ax.set_title("B. Correlation-aware attack defeats noise\n"
                 "that blinds the i.i.d. adversary")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # C: noise modes
    ax = axes[2]
    tags = [r["noise"] for r in expC]
    aw = [r["aware_half"] for r in expC]
    ax.bar(range(len(tags)), aw, color="#2c3e50")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("correlation-aware success")
    ax.set_title("C. Clip-correlated noise defends better\n"
                 "(matched budget; shared noise foils pooling)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    p = os.path.join(out_dir, "fig_synthetic_correlation.pdf")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nFigure -> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Synthetic correlation study.")
    ap.add_argument("--n-clips", type=int, default=150)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--d0", type=int, default=256)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--n-classes", type=int, default=10)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="noise for Experiment A (leakage profile).")
    ap.add_argument("--sigma-attack", type=float, default=4.0,
                    help="noise for Experiment B, chosen so the attack is "
                         "off the ceiling and the correlation trend is visible.")
    ap.add_argument("--sigma-c", type=float, default=12.0,
                    help="noise for Experiment C; higher so the aware attack "
                         "is off the ceiling and the iid-vs-clip-noise "
                         "differential is visible.")
    ap.add_argument("--rhos", type=float, nargs="+",
                    default=[0.0, 0.3, 0.6, 0.8, 0.9, 0.95])
    ap.add_argument("--n-targets", type=int, default=10)
    ap.add_argument("--n-trials", type=int, default=3)
    ap.add_argument("--max-dist", type=int, default=12)
    ap.add_argument("--agg", type=str, default="sum")
    ap.add_argument("--out-dir", type=str, default="./images")
    ap.add_argument("--results-dir", type=str, default="./results_revision")
    args = ap.parse_args()

    import pandas as pd

    curves = experiment_A(
        args.rhos, args.n_clips, args.L, args.d0, args.T, args.n_classes,
        args.d, args.sigma, args.n_targets, args.n_trials, args.max_dist,
        seed=42,
    )
    expB = experiment_B(
        args.rhos, args.n_clips, args.L, args.d0, args.T, args.n_classes,
        args.d, args.sigma_attack, args.n_targets, args.n_trials, args.agg,
        seed=42,
    )
    expC = experiment_C(
        0.8, args.n_clips, args.L, args.d0, args.T, args.n_classes,
        args.d, args.sigma_c, [0.25, 0.5, 1.0], args.n_targets, args.n_trials,
        args.agg, seed=42,
    )

    make_figures(curves, expB, expC, args.max_dist, args.out_dir)

    os.makedirs(args.results_dir, exist_ok=True)
    # save curves + tables
    dd = np.arange(0, args.max_dist + 1)
    cdf = pd.DataFrame({"distance": dd})
    for rho, curve in sorted(curves.items()):
        cdf[f"rho_{rho:.2f}"] = curve[:args.max_dist + 1]
    cdf.to_csv(os.path.join(args.results_dir, "synth_leakage_vs_distance.csv"),
               index=False, float_format="%.5f")
    pd.DataFrame(expB).to_csv(
        os.path.join(args.results_dir, "synth_aware_vs_blind.csv"),
        index=False, float_format="%.5f")
    pd.DataFrame(expC).to_csv(
        os.path.join(args.results_dir, "synth_noise_modes.csv"),
        index=False, float_format="%.5f")
    print(f"\nWrote synthetic-experiment CSVs to {args.results_dir}/")

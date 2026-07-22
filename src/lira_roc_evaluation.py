"""LiRA ROC / TPR@low-FPR evaluation of the temporal membership-inference attacks.

The paper (and Table 2) reports the attacks with a *weighted-overlap* recovery
score. Modern membership-inference practice (Carlini et al., "Membership
Inference Attacks From First Principles", S&P'22) argues that an average-case
score hides what matters: an attack is only meaningful if it is confident on a
*few* examples, i.e. it achieves high true-positive rate at *low* false-positive
rate. This script re-expresses our exact same LiRA scores as an ROC and reports
AUC, TPR@FPR=1%, TPR@FPR=0.1%, and the attack advantage max(TPR-FPR).

It imports and calls the *unmodified* scoring functions used by run_attack
(precompute_trial / precompute_target_lira / _lira_score_external_uw /
lira_score_frame_candidates), so the per-candidate likelihood ratios are
identical to the paper's; only the *aggregation* into a metric differs.

Two membership tests, both on the real UCF-101 frame universe:

  A1  within-clip index inference.  Candidate set = the CLIP_LEN=100 frames of
      the target clip.  MEMBERS = the 16 frames that were actually sampled into
      the universe U; NON-MEMBERS = the other 84 frames of the *same* clip.
      This is exactly the reviewer's "nearby correlated frames" concern made
      quantitative: can the adversary separate the 16 real members from their
      84 temporal neighbours?

  A3  universe-scale frame MIA.  Candidate set = a subsampled universe pool plus
      the 16 target rows.  MEMBERS = the 16 target rows; NON-MEMBERS = the rest
      (~2000, mostly other clips).  The standard "is this record in the dataset"
      test, in the heavily-imbalanced regime where TPR@low-FPR is the right axis.

Methodology (offline-LiRA, non-parametric).  Absolute LR scores are not
comparable across queries (different targets/trials have different residual
scales), so for each query we standardise every candidate's score by the mean
and std of that query's NON-MEMBER scores (the empirical null of that query;
members are excluded from the null estimate), then pool the standardised scores
across all queries and sweep one global threshold.  We also report the raw
(unstandardised) pooled AUC and the per-query mean AUC as cross-checks.

Efficiency.  run_attack redraws the mechanism (W,B) -- and thus the dominant
U@W matmul -- once per (target,trial).  For an ROC we instead draw one mechanism
per trial and probe every target against it, cutting the matmul count from
n_targets*n_trials down to n_trials with no loss of validity (each query is
standardised against its own null).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from membership_inference import (          # noqa: E402
    precompute_trial,
    precompute_target_lira,
    _lira_score_external_uw,
    lira_score_frame_candidates,
    load_full_clip_gray,
)


# ----------------------------------------------------------------------------
# Metric helpers (numpy only -- no sklearn dependency).
# ----------------------------------------------------------------------------
def auc_mann_whitney(scores, labels):
    """Exact AUC = P(score_member > score_nonmember), ties counted as 0.5.

    Rank-based (Mann-Whitney U), so it is invariant to any monotone transform
    of the scores and needs no thresholding.
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg_rank_by_group = (start + csum + 1) / 2.0
    ranks = avg_rank_by_group[inv]
    sum_pos = ranks[labels == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def roc_curve(scores, labels):
    """Return (fpr, tpr, thr) sweeping the threshold from +inf downward.

    fpr/tpr are step functions evaluated just after admitting each successive
    highest-scoring candidate; both start at 0 and end at 1.
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    n_pos = max(1, int((labels == 1).sum()))
    n_neg = max(1, int((labels == 0).sum()))
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    # collapse ties: keep only the last index of each distinct score
    keep = np.ones(len(s), bool)
    keep[:-1] = s[1:] != s[:-1]
    tpr = np.concatenate([[0.0], tp[keep] / n_pos])
    fpr = np.concatenate([[0.0], fp[keep] / n_neg])
    thr = np.concatenate([[np.inf], s[keep]])
    return fpr, tpr, thr


def tpr_at_fpr(fpr, tpr, target_fpr):
    """TPR at the largest threshold whose FPR does not exceed target_fpr
    (conservative), plus the linearly-interpolated value at exactly target_fpr."""
    fpr = np.asarray(fpr, float)
    tpr = np.asarray(tpr, float)
    below = np.where(fpr <= target_fpr)[0]
    conservative = float(tpr[below[-1]]) if len(below) else 0.0
    interp = float(np.interp(target_fpr, fpr, tpr))
    return conservative, interp


def attack_advantage(fpr, tpr):
    """max(TPR - FPR) over all thresholds -- the standard MIA advantage."""
    return float(np.max(np.asarray(tpr) - np.asarray(fpr)))


# ----------------------------------------------------------------------------
# Score collection for one (k, sigma) cell.
# ----------------------------------------------------------------------------
def collect_k_scores(
    U, labels, frame_indices, clip_ids, clip_paths, video_root,
    c, d, k, sigmas, targets_meta, full_clips_cache,
    n_trials, n_frame_candidates_a3, num_frames, clip_len, size, seed,
):
    """Sweep ALL sigmas for one mixing level k, reusing the expensive U@W matmul.

    Since sigma affects only the additive noise B (not W, UW, the permutation, or
    the mixing records), we draw the mechanism once per (k, trial) -- paying the
    one 127k x 12.5k x d matmul -- and then, for each sigma, redraw only B and
    rescore. This cuts the dominant cost by a factor of len(sigmas).

    Returns {sigma: {"a1":store, "a3s":store, "a3x":store}}.
    """
    rng = np.random.default_rng(seed)
    N_total = U.shape[0]
    class_to_rows = {int(l): np.where(labels == l)[0] for l in np.unique(labels)}

    # one fixed universe subsample for A3-crossclass candidates
    if n_frame_candidates_a3 < N_total:
        a3_pool = rng.choice(N_total, size=n_frame_candidates_a3, replace=False)
    else:
        a3_pool = np.arange(N_total)

    stores = {s: {"a1":  dict(z=[], y=[], q=[], auc_q=[]),
                  "a3s": dict(z=[], y=[], q=[], auc_q=[]),
                  "a3x": dict(z=[], y=[], q=[], auc_q=[])} for s in sigmas}
    qid = 0

    for trial_idx in range(n_trials):
        trial_rng = np.random.default_rng(seed * 10**6 + trial_idx)
        # One matmul (UW = U@W) reused across every sigma below.
        trial = precompute_trial(U, labels, c, k, d, 1.0, trial_rng)
        W = trial["W"]
        m, d_out = trial["m"], W.shape[1]
        print(f"    [k={k}] trial {trial_idx+1}/{n_trials}: matmul done, "
              f"sweeping {len(sigmas)} sigmas", flush=True)

        for tm in targets_meta:
            target_rows = tm["rows"]
            target_label = tm["label"]
            target_vid = tm["vid_idx"]

            # sigma-independent per-(trial,target) setup, computed ONCE:
            same_class_rows = class_to_rows[target_label]
            non_target = np.setdiff1d(same_class_rows, target_rows, assume_unique=False)
            if len(non_target) >= num_frames:
                cf_rows = trial_rng.choice(non_target, size=num_frames, replace=False)
            else:
                cf_rows = trial_rng.choice(non_target, size=num_frames, replace=True)
            full = full_clips_cache.get(target_vid)
            full_W = full.astype(np.float32) @ W if full is not None else None
            cf_set = set(cf_rows.tolist())
            same_pool_src = np.array([r for r in non_target if r not in cf_set], dtype=int)
            if len(same_pool_src) > n_frame_candidates_a3:
                same_pool_src = trial_rng.choice(
                    same_pool_src, size=n_frame_candidates_a3, replace=False)
            pool_s = np.unique(np.concatenate([same_pool_src, target_rows]))
            pool_x = np.unique(np.concatenate([a3_pool, target_rows]))
            y_s = np.isin(pool_s, target_rows).astype(int)
            y_x = np.isin(pool_x, target_rows).astype(int)

            for sigma in sigmas:
                # redraw ONLY the noise for this sigma (deterministic per cell)
                brng = np.random.default_rng(seed * 10**6 + trial_idx * 1000
                                             + int(round(sigma * 1e6)))
                trial["B"] = (brng.standard_normal((m, d_out)).astype(np.float32) * sigma)
                target_pre = precompute_target_lira(trial, target_rows, cf_rows)
                st = stores[sigma]

                # A1: within-clip in/out MIA (16 members vs 84 same-clip neighbors)
                if full_W is not None:
                    s_a1 = np.asarray(
                        _lira_score_external_uw(trial, target_pre, full_W, sigma), float)
                    y_a1 = np.zeros(clip_len, int)
                    y_a1[tm["true_idx"]] = 1
                    _accumulate(st["a1"], s_a1, y_a1, qid)

                # A3-sameclass: attribution vs same-class other-clip rows
                s_s = np.asarray(
                    lira_score_frame_candidates(trial, target_pre, pool_s, sigma), float)
                _accumulate(st["a3s"], s_s, y_s, qid)

                # A3-crossclass: identification vs broad universe
                s_x = np.asarray(
                    lira_score_frame_candidates(trial, target_pre, pool_x, sigma), float)
                _accumulate(st["a3x"], s_x, y_x, qid)

            qid += 1

    for s in sigmas:
        for key in ("a1", "a3s", "a3x"):
            D = stores[s][key]
            D["z"] = np.concatenate(D["z"]) if D["z"] else np.array([])
            D["y"] = np.concatenate(D["y"]) if D["y"] else np.array([])
            D["q"] = np.concatenate(D["q"]) if D["q"] else np.array([])
    return stores


def _accumulate(store, scores, labels, qid):
    """Per-query offline-LiRA standardisation against the non-member null."""
    null = scores[labels == 0]
    if len(null) < 2 or (labels == 1).sum() == 0:
        return
    mu, sd = float(null.mean()), float(null.std())
    sd = sd if sd > 1e-12 else 1.0
    z = (scores - mu) / sd
    store["z"].append(z)
    store["y"].append(labels)
    store["q"].append(np.full(len(scores), qid))
    store["auc_q"].append(auc_mann_whitney(scores, labels))   # per-query, raw


def summarise(store, k, sigma, attack):
    z, y = store["z"], store["y"]
    if len(z) == 0 or (y == 1).sum() == 0:
        return None
    fpr, tpr, _ = roc_curve(z, y)
    auc = auc_mann_whitney(z, y)
    t1_c, t1_i = tpr_at_fpr(fpr, tpr, 0.01)
    t01_c, t01_i = tpr_at_fpr(fpr, tpr, 0.001)
    per_q = np.array(store["auc_q"], float)
    return dict(
        attack=attack, k=k, sigma=sigma,
        auc_pooled=round(auc, 4),
        auc_perquery_mean=round(float(np.nanmean(per_q)), 4),
        auc_perquery_std=round(float(np.nanstd(per_q)), 4),
        tpr_at_fpr1pct=round(t1_c, 4),
        tpr_at_fpr1pct_interp=round(t1_i, 4),
        tpr_at_fpr0p1pct=round(t01_c, 4),
        tpr_at_fpr0p1pct_interp=round(t01_i, 4),
        advantage=round(attack_advantage(fpr, tpr), 4),
        n_members=int((y == 1).sum()),
        n_nonmembers=int((y == 0).sum()),
        n_queries=int(len(np.unique(store["q"]))),
    ), (fpr, tpr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="ucf101_frame_pool_gray_random_len100.npz")
    ap.add_argument("--video-root", default="./data/UCF-101")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 5])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.01, 0.10, 0.50])
    ap.add_argument("--n-targets", type=int, default=15)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--n-frame-candidates-a3", type=int, default=2000)
    ap.add_argument("--d", type=int, default=500)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--clip-len", type=int, default=100)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="./results_revision")
    ap.add_argument("--img-dir", default="./images")
    args = ap.parse_args()

    obj = np.load(args.pool, allow_pickle=True)
    U = obj["U"]; labels = obj["labels"].astype(int)
    frame_indices = obj["frame_indices"].astype(int)
    clip_ids = obj["clip_ids"].astype(int)
    clip_paths = list(obj["clip_paths"])
    present = np.unique(labels)
    labels = np.searchsorted(present, labels).astype(int)   # dense [0,c), like main
    c = int(len(present))
    n_clips = len(clip_paths)
    print(f"U={U.shape} clips={n_clips} classes={c}")

    # Fixed target set across all cells (same 15 clips), full clips cached once.
    rng = np.random.default_rng(args.seed)
    clip_row_indices = [np.where(clip_ids == v)[0] for v in range(n_clips)]
    target_vids = rng.choice(n_clips, size=args.n_targets, replace=False)
    targets_meta, full_clips_cache = [], {}
    for v in target_vids:
        rows = clip_row_indices[v]
        if len(rows) != args.num_frames:
            continue
        full = load_full_clip_gray(
            os.path.join(args.video_root, clip_paths[v]),
            clip_len=args.clip_len, size=args.size,
        )
        if full is None:
            continue
        full_clips_cache[int(v)] = full
        targets_meta.append(dict(
            vid_idx=int(v), rows=rows,
            true_idx=frame_indices[rows].astype(int),
            label=int(labels[rows][0]),
        ))
    print(f"targets usable: {len(targets_meta)}/{args.n_targets}")

    rows_out, curves = [], {}
    for k in args.ks:
        print(f"\n=== k={k} (sweeping sigmas {args.sigmas}) ===", flush=True)
        stores = collect_k_scores(
            U, labels, frame_indices, clip_ids, clip_paths, args.video_root,
            c, args.d, k, args.sigmas, targets_meta, full_clips_cache,
            args.n_trials, args.n_frame_candidates_a3, args.num_frames,
            args.clip_len, args.size, args.seed + k * 100,
        )
        for sigma in args.sigmas:
            for tag, key in (("A1_within_clip", "a1"),
                             ("A3_sameclass", "a3s"),
                             ("A3_crossclass", "a3x")):
                res = summarise(stores[sigma][key], k, sigma, tag)
                if res is None:
                    continue
                row, (fpr, tpr) = res
                rows_out.append(row)
                curves[f"{tag}_k{k}_s{sigma}_fpr"] = fpr
                curves[f"{tag}_k{k}_s{sigma}_tpr"] = tpr
                print(f"  k={k} s={sigma} {tag}: AUC={row['auc_pooled']:.3f} "
                      f"TPR@1%={row['tpr_at_fpr1pct']:.3f} "
                      f"TPR@0.1%={row['tpr_at_fpr0p1pct']:.3f} "
                      f"adv={row['advantage']:.3f} "
                      f"(n_mem={row['n_members']}, n_non={row['n_nonmembers']})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows_out)
    csv_path = os.path.join(args.out_dir, "lira_roc_summary.csv")
    df.to_csv(csv_path, index=False)
    np.savez(os.path.join(args.out_dir, "lira_roc_curves.npz"), **curves)
    print(f"\nWrote {csv_path}")
    print(df.to_string(index=False))

    _plot(curves, args)


def _plot(curves, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[skip figure] {e}")
        return
    os.makedirs(args.img_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    # Panel A: same-class attribution MIA, k=0, across sigma (log-log LiRA ROC)
    axA = axes[0]
    cols = ["#7f8c8d", "#c0392b", "#2c3e50", "#27ae60", "#8e44ad"]
    for sigma, col in zip(args.sigmas, cols):
        fk, tk = f"A3_sameclass_k0_s{sigma}_fpr", f"A3_sameclass_k0_s{sigma}_tpr"
        if fk in curves:
            f, t = curves[fk], curves[tk]
            f = np.clip(f, 1e-5, 1); t = np.clip(t, 1e-5, 1)
            axA.loglog(f, t, "-", color=col, label=rf"$\sigma={sigma}$")
    axA.plot([1e-5, 1], [1e-5, 1], "k--", lw=0.8, alpha=0.6)
    axA.set_xlim(1e-4, 1); axA.set_ylim(1e-3, 1)
    axA.set_xlabel("False positive rate"); axA.set_ylabel("True positive rate")
    axA.set_title(r"Same-class frame MIA ($k=0$), vs.\ noise")
    axA.legend(fontsize=7, loc="lower right"); axA.grid(True, which="both", alpha=0.25)
    # Panel B: mixing at fixed sigma=0.10 (same-class)
    axB = axes[1]
    for k, col in zip(args.ks, ["#c0392b", "#2c3e50", "#27ae60"]):
        fk, tk = f"A3_sameclass_k{k}_s0.1_fpr", f"A3_sameclass_k{k}_s0.1_tpr"
        if fk in curves:
            f, t = curves[fk], curves[tk]
            f = np.clip(f, 1e-5, 1); t = np.clip(t, 1e-5, 1)
            axB.loglog(f, t, "-", color=col, label=rf"$k={k}$ mixing")
    axB.plot([1e-5, 1], [1e-5, 1], "k--", lw=0.8, alpha=0.6)
    axB.set_xlim(1e-4, 1); axB.set_ylim(1e-3, 1)
    axB.set_xlabel("False positive rate"); axB.set_ylabel("True positive rate")
    axB.set_title(r"Same-class frame MIA ($\sigma=0.10$), vs.\ mixing")
    axB.legend(fontsize=7, loc="lower right"); axB.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(args.img_dir, "fig_lira_roc.pdf")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"Figure -> {p}")


if __name__ == "__main__":
    main()

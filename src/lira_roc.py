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
      This makes the "nearby correlated frames" question quantitative: can the
      adversary separate the 16 real members from their 84 temporal
      neighbours?

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

Produces supplement Table 3 and Figure 3 (Appendix E.4).

Modes:
    python src/lira_roc.py            # run: compute, write CSV + curves + figure
    python src/lira_roc.py plot       # re-render the figure from saved curves
    python src/lira_roc.py verify     # self-checks on the computed results

`run` writes lira_roc_summary.csv (Table 3) and lira_roc_curves.npz to
--out-dir, and fig_lira_roc.pdf (Figure 3) to --img-dir. `plot` needs only the
npz, so the figure can be restyled without recomputing. `verify` exits non-zero
if a hard check fails.
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

OUT_DIR = "./results_revision"
IMG_DIR = "./images"


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


def run_evaluation():
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
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--img-dir", default=IMG_DIR)
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

    plot_roc(curves, args.out_dir, args.img_dir)




# ----------------------------------------------------------------------------
# Figure: supplement Figure 3 (fig_lira_roc.pdf)
# ----------------------------------------------------------------------------

def plot_roc(curves, out_dir=OUT_DIR, img_dir=IMG_DIR):
    """Render supplement Figure 3 from ROC curves.

    Panel A: the three adversaries at (k=0, sigma=0.10) -- the hardness
             gradient (informed within-clip near chance; the uninformed
             identification adversaries confident).
    Panel B: the uninformed cross-class adversary at sigma=0.10 across mixing k.

    `curves` may be the in-memory dict from run_evaluation or a loaded npz.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(os.path.join(out_dir, "lira_roc_summary.csv"))

    def auc_of(tag, k, s):
        r = df[(df.attack == tag) & (df.k == k) & (df.sigma == s)]
        return float(r["auc_pooled"].iloc[0]) if len(r) else float("nan")

    def curve(tag, k, s):
        fk, tk = f"{tag}_k{k}_s{s}_fpr", f"{tag}_k{k}_s{s}_tpr"
        if fk not in curves:
            return None
        return np.clip(curves[fk], 1e-5, 1.0), np.clip(curves[tk], 1e-5, 1.0)

    os.makedirs(img_dir, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.2))
    sA = 0.1

    specs = [
        ("A3_crossclass",  "Uninformed (vs. unrelated)",   "#c0392b"),
        ("A3_sameclass",   "Uninformed (vs. same class)",  "#e08e0b"),
        ("A1_within_clip", "Informed (vs. own neighbors)", "#2c3e50"),
    ]
    for tag, lab, col in specs:
        c = curve(tag, 0, sA)
        if c is None:
            continue
        axA.loglog(c[0], c[1], "-", color=col, lw=1.8,
                   label=f"{lab}, AUC={auc_of(tag, 0, sA):.2f}")
    axA.plot([1e-5, 1], [1e-5, 1], "k--", lw=0.8, alpha=0.6, label="chance")
    axA.set_xlim(1e-4, 1); axA.set_ylim(1e-3, 1)
    axA.set_xlabel("False positive rate"); axA.set_ylabel("True positive rate")
    axA.set_title("Three adversaries at $(k{=}0,\\ \\sigma{=}0.10)$")
    axA.legend(fontsize=7.5, loc="lower right")
    axA.grid(True, which="both", alpha=0.25)

    for k, col in zip([0, 1, 5], ["#c0392b", "#2c3e50", "#27ae60"]):
        c = curve("A3_crossclass", k, sA)
        if c is None:
            continue
        axB.loglog(c[0], c[1], "-", color=col, lw=1.8,
                   label=f"$k={k}$, AUC={auc_of('A3_crossclass', k, sA):.2f}")
    axB.plot([1e-5, 1], [1e-5, 1], "k--", lw=0.8, alpha=0.6, label="chance")
    axB.set_xlim(1e-4, 1); axB.set_ylim(1e-3, 1)
    axB.set_xlabel("False positive rate"); axB.set_ylabel("True positive rate")
    axB.set_title("Uninformed adversary vs. mixing $(\\sigma{=}0.10)$")
    axB.legend(fontsize=7.5, loc="lower right")
    axB.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    p = os.path.join(img_dir, "fig_lira_roc.pdf")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"Figure -> {p}")


def replot(out_dir=OUT_DIR, img_dir=IMG_DIR):
    """Re-render the figure from saved curves, with no recomputation."""
    curves = np.load(os.path.join(out_dir, "lira_roc_curves.npz"))
    plot_roc(curves, out_dir, img_dir)


# ----------------------------------------------------------------------------
# Self-checks on the computed results
# ----------------------------------------------------------------------------

def verify(out_dir=OUT_DIR):
    """Sanity-check the ROC results. Returns the number of hard failures.

    From lira_roc_summary.csv + lira_roc_curves.npz:
      (C1) every AUC in [0,1]; pooled AUC within 0.06 of per-query-mean AUC
           (two ways of measuring the same thing -> must roughly agree).
      (C2) monotone privacy: for a fixed attack & k, AUC is non-increasing in
           sigma (more noise cannot HELP the attack, MC tolerance 0.03).
      (C3) hardness ordering at each (k,sigma): A1_within_clip <= A3_sameclass
           <= A3_crossclass (same members, progressively easier non-members).
      (C4) TPR@FPR sanity: 0 <= TPR@0.1% <= TPR@1% <= 1; advantage in [0,1].
      (C5) ROC curve shape: fpr,tpr monotone nondecreasing, running 0 -> 1.

    C1/C2/C3 and the C5 endpoint check are reported as warnings, since each has
    a legitimate Monte-Carlo explanation; range and ordering violations, which
    do not, are hard failures.
    """
    import pandas as pd

    df = pd.read_csv(os.path.join(out_dir, "lira_roc_summary.csv"))
    curves = np.load(os.path.join(out_dir, "lira_roc_curves.npz"))
    fails, warns = [], []

    for _, r in df.iterrows():
        tag = f"{r['attack']} k{r['k']} s{r['sigma']}"
        if not (0.0 <= r["auc_pooled"] <= 1.0):
            fails.append(f"C1 AUC out of range: {tag} auc={r['auc_pooled']}")
        if abs(r["auc_pooled"] - r["auc_perquery_mean"]) > 0.06:
            warns.append(f"C1 pooled vs per-query AUC differ: {tag} "
                         f"{r['auc_pooled']} vs {r['auc_perquery_mean']}")

    for attack in df["attack"].unique():
        for k in sorted(df["k"].unique()):
            sub = df[(df.attack == attack) & (df.k == k)].sort_values("sigma")
            a = sub["auc_pooled"].to_numpy()
            s = sub["sigma"].to_numpy()
            for i in range(1, len(a)):
                if a[i] > a[i - 1] + 0.03:
                    warns.append(f"C2 AUC rises with sigma: {attack} k={k} "
                                 f"sigma {s[i-1]}->{s[i]} "
                                 f"{a[i-1]:.3f}->{a[i]:.3f}")

    order = ("A1_within_clip", "A3_sameclass", "A3_crossclass")
    for (k, s), g in df.groupby(["k", "sigma"]):
        m = {row["attack"]: row["auc_pooled"] for _, row in g.iterrows()}
        if all(a in m for a in order):
            seq = [m[a] for a in order]
            for i in range(1, len(seq)):
                if seq[i] < seq[i - 1] - 0.05:
                    warns.append(f"C3 hardness order broken at k={k},s={s}: "
                                 f"A1={seq[0]:.3f} A3s={seq[1]:.3f} "
                                 f"A3x={seq[2]:.3f}")

    for _, r in df.iterrows():
        tag = f"{r['attack']} k{r['k']} s{r['sigma']}"
        if not (0 <= r["tpr_at_fpr0p1pct"]
                <= r["tpr_at_fpr1pct"] + 1e-9 <= 1 + 1e-9):
            fails.append(f"C4 TPR ordering: {tag} "
                         f"0.1%={r['tpr_at_fpr0p1pct']} "
                         f"1%={r['tpr_at_fpr1pct']}")
        if not (-1e-9 <= r["advantage"] <= 1 + 1e-9):
            fails.append(f"C4 advantage out of range: {tag} "
                         f"adv={r['advantage']}")

    for key in curves.files:
        if not key.endswith("_fpr"):
            continue
        f = curves[key]
        t = curves[key[:-4] + "_tpr"]
        if len(f) < 2:
            continue
        if np.any(np.diff(f) < -1e-9) or np.any(np.diff(t) < -1e-9):
            fails.append(f"C5 non-monotone ROC: {key[:-4]}")
        if (abs(f[0]) > 1e-6 or abs(t[0]) > 1e-6
                or abs(f[-1] - 1) > 1e-6 or abs(t[-1] - 1) > 1e-6):
            warns.append(f"C5 ROC endpoints off: {key[:-4]} "
                         f"f[{f[0]:.3g},{f[-1]:.3g}] "
                         f"t[{t[0]:.3g},{t[-1]:.3g}]")

    print("=== WARNINGS ===")
    for w in warns:
        print(" ", w)
    print("=== FAILURES ===")
    for x in fails:
        print(" ", x)
    print(f"\n{len(warns)} warnings, {len(fails)} failures")
    return len(fails)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else "run"
    if mode == "run":
        run_evaluation()
    elif mode == "plot":
        replot()
    elif mode == "verify":
        sys.exit(1 if verify() else 0)
    else:
        sys.exit(f"unknown mode {mode!r}; expected run | plot | verify")

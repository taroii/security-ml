"""Correlation-aware temporal membership inference.

Directly answers Reviewer A's central request:

  "Given these correlations, I expected the paper to evaluate privacy leakage
   using a standard membership inference attack, which could help demonstrate
   whether the privacy guarantees ... are substantially overestimated ...
   it is less clear that [independent per-frame noise] is appropriate for
   models that must jointly process correlated frames."

The paper's Attack 3 scores every candidate frame INDEPENDENTLY and takes the
top-T -- a correlation-BLIND adversary that treats frames as i.i.d., exactly
the assumption the reviewer says is disconnected from practice. This module
adds a correlation-AWARE adversary that pools the per-frame likelihood-ratio
evidence across frames of the same clip before deciding, mirroring the
two-level prior (winning one clip delivers a block of correlated credit).

Both adversaries see the SAME released mechanism output and use the SAME
per-frame LiRA scores; they differ only in whether they exploit the clip
structure. Comparing them at a fixed noise level isolates the value of the
correlation the flat analysis ignores, and tests whether the i.i.d.-
calibrated noise the paper certifies actually protects against an adversary
that uses the dependence.

Reported per (k, sigma) cell:
  * blind Attack 3        : top-T independent frames (paper's adversary)
  * clip-aware Attack 3   : rank clips by aggregated LR, then allocate the T
                            guesses within the best clip(s)
  * target-clip rank      : where the true target clip lands under aggregation
                            (top-1 accuracy = correlation-aware clip inference)
  * weighted-overlap gain : aware minus blind, half-integer rule

Everything reuses the audited LiRA machinery in membership_inference.py; no
mechanism or scoring code is duplicated.
"""
import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from membership_inference import (
    build_frame_pool,
    precompute_trial,
    precompute_target_lira,
    lira_score_frame_candidates,
    compute_weighted_score,
)
from main_video import (
    set_seed,
    parse_ucf101_split,
    download_ucf101,
    filter_long_clips,
    _long_clips_cache_path,
    CLIP_LEN,
)


def clip_aggregated_scores(
    pool_rows: np.ndarray,
    per_frame_lr: np.ndarray,
    clip_ids: np.ndarray,
    agg: str = "sum",
) -> Dict[int, float]:
    """Aggregate per-frame LiRA scores into a per-clip evidence score.

    pool_rows[i] is a universe row index; per_frame_lr[i] its LiRA score.
    Returns {clip_id: aggregated_score}. The correlation-aware adversary
    exploits that a member clip contributes SEVERAL elevated-LR frames, so
    pooling their evidence separates member from non-member clips far better
    than any single frame does -- precisely the joint-processing leakage the
    reviewer flagged.

    agg:
      "sum"  -- total evidence (favours clips with many pooled frames)
      "mean" -- average evidence per pooled frame (size-normalized)
      "topk" -- mean of the clip's top-min(4,n) frame scores (robust to a
                few noisy frames; models an adversary that needs only a
                handful of confident frames to flag a clip)
    """
    clip_to_scores: Dict[int, List[float]] = {}
    for r, s in zip(pool_rows.tolist(), per_frame_lr.tolist()):
        clip_to_scores.setdefault(int(clip_ids[r]), []).append(float(s))
    out: Dict[int, float] = {}
    for cid, scores in clip_to_scores.items():
        arr = np.asarray(scores)
        if agg == "sum":
            out[cid] = float(arr.sum())
        elif agg == "mean":
            out[cid] = float(arr.mean())
        elif agg == "topk":
            kk = min(4, len(arr))
            out[cid] = float(np.sort(arr)[-kk:].mean())
        else:
            raise ValueError(f"unknown agg {agg}")
    return out


def run_correlation_aware(
    U: np.ndarray,
    labels: np.ndarray,
    frame_indices: np.ndarray,
    clip_ids: np.ndarray,
    clip_paths: List[str],
    c: int,
    d: int,
    sigma: float,
    k: int,
    n_targets: int = 12,
    n_trials: int = 5,
    n_frame_candidates: int = 2000,
    num_frames: int = 16,
    agg: str = "sum",
    seed: int = 0,
    noise_mode: str = "iid",
    noise_clip_frac: float = 0.5,
) -> dict:
    """Blind vs correlation-aware Attack 3 on the shared LiRA scores."""
    rng = np.random.default_rng(seed)
    n_clips = len(clip_paths)
    N_total = U.shape[0]

    clip_row_indices = [np.where(clip_ids == v)[0] for v in range(n_clips)]
    class_to_rows = {int(l): np.where(labels == l)[0] for l in np.unique(labels)}

    target_vids = rng.choice(
        n_clips, size=min(n_targets, n_clips), replace=False
    )
    targets_meta = []
    for vid in target_vids:
        rows = clip_row_indices[vid]
        if len(rows) != num_frames:
            continue
        targets_meta.append({
            "vid": int(vid), "rows": rows,
            "true_idx": frame_indices[rows].astype(int),
            "label": int(labels[rows][0]),
        })

    # Candidate pool = EVERY clip's full set of pooled frames, so each clip is
    # represented by exactly num_frames rows. This removes any clip-size
    # confound from the aggregation: a clip cannot rank first merely for having
    # more rows in the pool, only for carrying more membership evidence. (If the
    # universe is large we subsample CLIPS, never rows, to preserve this.)
    if n_frame_candidates < N_total:
        n_pool_clips = max(2, n_frame_candidates // num_frames)
        pool_clips = rng.choice(n_clips, size=min(n_pool_clips, n_clips),
                                replace=False)
        base_pool = np.concatenate([clip_row_indices[v] for v in pool_clips])
    else:
        base_pool = np.arange(N_total)

    blind_scores, aware_scores = [], []
    blind_int, aware_int = [], []
    target_clip_ranks = []          # rank (1=best) of target clip under aggregation
    aware_top1 = []                 # 1 if target clip is #1 under aggregation
    blind_top1_clip = []            # 1 if blind top-T's plurality clip is target

    n_target_clips = len(targets_meta)
    print(f"  correlation-aware: {n_trials} trials x {n_target_clips} targets "
          f"(k={k}, sigma={sigma}, agg={agg})")

    for ti, tm in enumerate(targets_meta):
        target_rows = tm["rows"]
        target_vid = tm["vid"]
        target_label = tm["label"]
        truth = list(zip(
            np.full(num_frames, target_vid, dtype=int).tolist(),
            tm["true_idx"].tolist(),
        ))

        for trial_idx in range(n_trials):
            trial_rng = np.random.default_rng(
                seed * 10**6 + ti * 1000 + trial_idx
            )
            trial = precompute_trial(
                U, labels, c, k, d, sigma, trial_rng,
                noise_mode=noise_mode, clip_ids=clip_ids,
                noise_clip_frac=noise_clip_frac,
            )

            same_class_rows = class_to_rows[target_label]
            non_target = np.setdiff1d(same_class_rows, target_rows)
            # In a restricted universe some classes can have a single clip, so
            # same-class non-target rows may be too few (or empty). Fall back
            # to cross-class non-target rows -- the counterfactual only needs
            # to be plausible non-member frames for the LiRA baseline.
            if len(non_target) < num_frames:
                all_non_target = np.setdiff1d(np.arange(N_total), target_rows)
                non_target = all_non_target
            replace = len(non_target) < num_frames
            counterfactual_rows = trial_rng.choice(
                non_target, size=num_frames, replace=replace
            )
            target_pre = precompute_target_lira(
                trial, target_rows, counterfactual_rows
            )

            # Shared candidate pool of whole clips; always include the target
            # clip's full rows so every clip (incl. target) has exactly
            # num_frames rows -> aggregation is size-balanced.
            pool = np.unique(np.concatenate([base_pool, target_rows]))
            lr = lira_score_frame_candidates(trial, target_pre, pool, sigma)

            # ---- BLIND (paper's adversary): top-T independent frames ----
            top_blind = pool[np.argsort(-lr)[:num_frames]]
            g_blind = [(int(clip_ids[r]), int(frame_indices[r])) for r in top_blind]
            blind_scores.append(compute_weighted_score(g_blind, truth, "half"))
            blind_int.append(compute_weighted_score(g_blind, truth, "integer"))
            # plurality clip among the blind top-T
            bl_clips, bl_counts = np.unique(
                [clip_ids[r] for r in top_blind], return_counts=True
            )
            blind_top1_clip.append(
                1.0 if int(bl_clips[np.argmax(bl_counts)]) == target_vid else 0.0
            )

            # ---- CORRELATION-AWARE: aggregate LR by clip, pick best clip ----
            clip_scores = clip_aggregated_scores(pool, lr, clip_ids, agg=agg)
            ranked = sorted(clip_scores.items(), key=lambda kv: -kv[1])
            ranked_clips = [cid for cid, _ in ranked]
            rank = ranked_clips.index(target_vid) + 1 if target_vid in ranked_clips else len(ranked_clips) + 1
            target_clip_ranks.append(rank)
            aware_top1.append(1.0 if rank == 1 else 0.0)

            # Allocate the T guesses within the top-ranked clip: take that
            # clip's pooled rows, ranked by their own per-frame LR, up to T.
            best_clip = ranked_clips[0]
            best_rows = pool[clip_ids[pool] == best_clip]
            best_lr = lr[clip_ids[pool] == best_clip]
            order = np.argsort(-best_lr)
            chosen = best_rows[order[:num_frames]]
            # if the top clip has < T pooled frames, fill from next clips
            fill_i = 1
            chosen = list(chosen)
            while len(chosen) < num_frames and fill_i < len(ranked_clips):
                nxt = ranked_clips[fill_i]
                nxt_rows = pool[clip_ids[pool] == nxt]
                nxt_lr = lr[clip_ids[pool] == nxt]
                for r in nxt_rows[np.argsort(-nxt_lr)]:
                    chosen.append(r)
                    if len(chosen) >= num_frames:
                        break
                fill_i += 1
            chosen = np.asarray(chosen[:num_frames])
            g_aware = [(int(clip_ids[r]), int(frame_indices[r])) for r in chosen]
            aware_scores.append(compute_weighted_score(g_aware, truth, "half"))
            aware_int.append(compute_weighted_score(g_aware, truth, "integer"))

        if (ti + 1) % max(1, n_target_clips // 4) == 0:
            print(f"    target {ti+1}/{n_target_clips}")

    def _n(x):
        return float(np.mean(x) / num_frames)

    blind_h, aware_h = _n(blind_scores), _n(aware_scores)
    blind_i, aware_i = _n(blind_int), _n(aware_int)
    ranks = np.asarray(target_clip_ranks, dtype=float)

    print(f"\n  --- k={k}, sigma={sigma} ---")
    print(f"  Attack 3 blind (half)     : {blind_h:.4f}")
    print(f"  Attack 3 clip-aware (half): {aware_h:.4f}   "
          f"(gain {aware_h - blind_h:+.4f}, "
          f"{aware_h / max(blind_h,1e-9):.2f}x)")
    print(f"  Attack 3 blind (int)      : {blind_i:.4f}")
    print(f"  Attack 3 clip-aware (int) : {aware_i:.4f}   "
          f"(gain {aware_i - blind_i:+.4f})")
    print(f"  target-clip top-1 acc     : {np.mean(aware_top1):.4f}   "
          f"(blind plurality-clip acc {np.mean(blind_top1_clip):.4f})")
    print(f"  target-clip mean rank     : {ranks.mean():.1f}   "
          f"(median {np.median(ranks):.0f}, of {n_clips} clips)")

    return {
        "k": k, "sigma": sigma, "agg": agg,
        "n_targets": n_target_clips, "n_trials": n_trials,
        "n_universe_clips": int(n_clips), "n_universe_frames": int(N_total),
        "pool_size": int(n_frame_candidates), "num_frames": int(num_frames),
        "attack3_blind_half": blind_h,
        "attack3_aware_half": aware_h,
        "attack3_gain_half": aware_h - blind_h,
        "attack3_gain_ratio": aware_h / max(blind_h, 1e-9),
        "attack3_blind_int": blind_i,
        "attack3_aware_int": aware_i,
        "target_clip_top1_acc": float(np.mean(aware_top1)),
        "blind_plurality_clip_acc": float(np.mean(blind_top1_clip)),
        "target_clip_mean_rank": float(ranks.mean()),
        "target_clip_median_rank": float(np.median(ranks)),
        "clip_top1_chance": 1.0 / n_clips,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Blind vs correlation-aware temporal MIA (Attack 3)."
    )
    ap.add_argument("--k", type=int, default=0, choices=[0, 1, 5])
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.05, 0.10])
    ap.add_argument("--n-targets", type=int, default=12)
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--n-frame-candidates", type=int, default=2000)
    ap.add_argument("--n-universe-clips", type=int, default=400,
                    help="Restrict the attack universe to this many clips "
                         "(candidate pool). Keeps U @ W tractable on CPU; "
                         "0 = use all clips.")
    ap.add_argument("--agg", type=str, default="sum",
                    choices=["sum", "mean", "topk"])
    ap.add_argument("--noise-mode", type=str, default="iid",
                    choices=["iid", "clip"])
    ap.add_argument("--noise-clip-frac", type=float, default=0.5)
    ap.add_argument("--results-dir", type=str, default="./results_revision")
    args = ap.parse_args()

    SEED, NUM_FRAMES, SIZE, D, SPLIT = 42, 16, 112, 500, 1
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    VIDEO_ROOT = os.path.join(PROJECT_ROOT, "data", "UCF-101")
    ANNOT_ROOT = os.path.join(PROJECT_ROOT, "data", "ucfTrainTestlist")
    CACHE_PATH = os.path.join(
        PROJECT_ROOT, f"ucf101_frame_pool_gray_random_len{CLIP_LEN}.npz"
    )

    download_ucf101(os.path.join(PROJECT_ROOT, "data"), VIDEO_ROOT, ANNOT_ROOT)
    set_seed(SEED)
    _, train_list, _ = parse_ucf101_split(ANNOT_ROOT, SPLIT)
    train_cache = _long_clips_cache_path(ANNOT_ROOT, SPLIT, CLIP_LEN) \
        .replace(".json", "_train.json")
    train_list, _ = filter_long_clips(
        VIDEO_ROOT, train_list, CLIP_LEN, cache_path=train_cache, desc="train"
    )
    U, labels, frame_indices, clip_ids, clip_paths = build_frame_pool(
        video_root=VIDEO_ROOT, train_list=train_list, cache_path=CACHE_PATH,
        num_frames=NUM_FRAMES, size=SIZE, seed=SEED, clip_len=CLIP_LEN,
    )

    # Restrict to a candidate universe of clips so U @ W is tractable on CPU.
    # We keep whole clips (all their pooled frames) and remap clip_ids dense.
    if args.n_universe_clips and args.n_universe_clips < len(clip_paths):
        sub_rng = np.random.default_rng(SEED)
        keep_clips = np.sort(sub_rng.choice(
            len(clip_paths), size=args.n_universe_clips, replace=False))
        keep_set = set(keep_clips.tolist())
        row_mask = np.array([cid in keep_set for cid in clip_ids])
        U = U[row_mask]
        labels = labels[row_mask]
        frame_indices = frame_indices[row_mask]
        old_clip_ids = clip_ids[row_mask]
        remap = {old: new for new, old in enumerate(keep_clips.tolist())}
        clip_ids = np.array([remap[c_] for c_ in old_clip_ids.tolist()])
        clip_paths = [clip_paths[c_] for c_ in keep_clips.tolist()]
        print(f"Restricted universe to {len(clip_paths)} clips, "
              f"{U.shape[0]} frames (d0={U.shape[1]}).")

    present = np.unique(labels)
    labels = np.searchsorted(present, labels).astype(int)
    c = int(len(present))

    os.makedirs(args.results_dir, exist_ok=True)
    rows = []
    for i, sigma in enumerate(args.sigmas):
        res = run_correlation_aware(
            U, labels, frame_indices, clip_ids, clip_paths,
            c=c, d=D, sigma=sigma, k=args.k,
            n_targets=args.n_targets, n_trials=args.n_trials,
            n_frame_candidates=args.n_frame_candidates,
            num_frames=NUM_FRAMES, agg=args.agg, seed=SEED + i,
            noise_mode=args.noise_mode, noise_clip_frac=args.noise_clip_frac,
        )
        res["noise_mode"] = args.noise_mode
        res["noise_clip_frac"] = args.noise_clip_frac
        rows.append(res)
    noise_tag = "" if args.noise_mode == "iid" else f"_{args.noise_mode}{args.noise_clip_frac:g}"
    out = os.path.join(
        args.results_dir, f"corr_aware_k{args.k}_{args.agg}{noise_tag}.csv"
    )
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    print(f"\nWrote {out}")

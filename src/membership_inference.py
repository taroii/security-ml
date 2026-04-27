import argparse
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

# reuse utilities from main pipeline
from main_video import (
    set_seed,
    get_device,
    parse_ucf101_split,
    download_ucf101,
    filter_long_clips,
    _long_clips_cache_path,
    CLIP_LEN,  # common trimming length, single source of truth
)


# Frame loading: grayscale, no embedding model
def load_video_frames_gray(
    path:       str,
    num_frames: int = 16,
    size:       int = 112,
    rng:        np.random.Generator = None,
    clip_len:   int = CLIP_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load num_frames random indices from [0, clip_len) of the clip.
    Caller must ensure the clip has at least clip_len frames (use
    filter_long_clips upstream). Returns (frames, indices).
        frames: (num_frames, clip_len_pixels)  flat float32 in [0, 1]
        indices: (num_frames,) int sorted
    """
    if rng is None:
        rng = np.random.default_rng()

    indices = np.sort(
        rng.choice(clip_len, size=num_frames, replace=False)
    ).astype(int)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    sampled = [None] * num_frames
    next_target = 0
    cur = 0
    while next_target < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if cur == int(indices[next_target]):
            sampled[next_target] = frame
            next_target += 1
        cur += 1
        if cur >= clip_len:
            break
    cap.release()

    if next_target < num_frames:
        raise RuntimeError(
            f"Only read {next_target}/{num_frames} frames from {path} "
            f"(clip shorter than CLIP_LEN={clip_len}?)"
        )

    out = np.empty((num_frames, size * size), dtype=np.float32)
    for t, frame in enumerate(sampled):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (size, size))
        out[t] = gray.astype(np.float32).reshape(-1) / 255.0
    return out, indices


def load_full_clip_gray(
    path:     str,
    clip_len: int = CLIP_LEN,
    size:     int = 112,
) -> np.ndarray:
    """
    Read the first clip_len frames of a clip as flat-grayscale (clip_len, size*size).
    Used by Attack 1 (index inference) where the adversary ranks all
    indices in [0, clip_len) by per-candidate likelihood.
    Returns None if clip is shorter than clip_len.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    out = np.empty((clip_len, size * size), dtype=np.float32)
    for i in range(clip_len):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (size, size))
        out[i] = gray.astype(np.float32).reshape(-1) / 255.0
    cap.release()
    return out


# Build frame pool U over the trimmed dataset. Per-clip rng is seeded
# deterministically so the universe is reproducible from (clip_idx, seed).
def build_frame_pool(
    video_root: str,
    train_list: List[Tuple[str, int]],
    cache_path: str,
    num_frames: int = 16,
    size:       int = 112,
    seed:       int = 0,
    clip_len:   int = CLIP_LEN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if os.path.exists(cache_path):
        obj = np.load(cache_path, allow_pickle=True)
        print(f"Loaded frame pool from cache: {cache_path}")
        return (
            obj["U"],
            obj["labels"],
            obj["frame_indices"],
            obj["clip_ids"],
            list(obj["clip_paths"]),
        )

    U_list, labels_list, fidx_list, cid_list, paths_list = [], [], [], [], []

    for src_idx, (rel_path, label) in enumerate(train_list):
        video_path = os.path.join(video_root, rel_path)
        # Per-clip rng deterministic in (src_idx, seed) — required for the
        # cache-validity invariant: the universe must be identical across
        # candidate evaluations within a trial. We seed off the train_list
        # position so skips don't shift the rng for later clips.
        clip_rng = np.random.default_rng(seed * (1 << 20) + src_idx)
        try:
            frames, fidxs = load_video_frames_gray(
                video_path, num_frames=num_frames, size=size,
                rng=clip_rng, clip_len=clip_len,
            )
        except RuntimeError as e:
            print(f"  Skipping {rel_path}: {e}")
            continue

        # Stored clip_id is dense in [0, n_kept) so downstream indexing
        # (per_clip aggregation, clip_row_indices) stays in-bounds.
        clip_id = len(paths_list)
        T = frames.shape[0]
        U_list.append(frames)
        labels_list.append(np.full(T, label, dtype=int))
        fidx_list.append(fidxs)
        cid_list.append(np.full(T, clip_id, dtype=int))
        paths_list.append(rel_path)

        if (src_idx + 1) % 200 == 0:
            print(f"  Loaded {src_idx + 1}/{len(train_list)} videos")

    U = np.concatenate(U_list, axis=0).astype(np.float32)
    labels = np.concatenate(labels_list, axis=0)
    frame_indices = np.concatenate(fidx_list, axis=0)
    clip_ids = np.concatenate(cid_list, axis=0)
    clip_paths = np.array(paths_list, dtype=object)

    np.savez(
        cache_path,
        U=U, labels=labels, frame_indices=frame_indices,
        clip_ids=clip_ids, clip_paths=clip_paths,
    )
    print(f"Frame pool cached to {cache_path}")
    print(f"Total frames: {U.shape[0]}, d0={U.shape[1]}, n_clips={len(paths_list)}")
    return U, labels, frame_indices, clip_ids, paths_list


# Mechanism: with or without mixing
# We expose the M_mix as explicit row-selection records so the
# linearity-of-mechanism amortization in run_attack can identify which
# mixed rows touch the target's universe rows.
def class_k_mix_records(
    labels: np.ndarray,
    c:      int,
    k:      int,
    rng:    np.random.Generator,
) -> List[np.ndarray]:
    """
    Build the row-selection records for the class-k mixing matrix M_mix.
    Returns a list of length c*c, each entry a (2k,) array of universe
    indices selected for that mixed row. Mixed-row i averages frames at
    those indices with weight 1/(2k) each.
    """
    assert k >= 1, "class_k_mix_records requires k >= 1"
    cls_to_idx = [np.where(labels == cls)[0] for cls in range(c)]
    records = []
    for i in range(c):
        for j in range(c):
            idx_i = rng.choice(cls_to_idx[i], size=k, replace=True)
            idx_j = rng.choice(cls_to_idx[j], size=k, replace=True)
            records.append(np.concatenate([idx_i, idx_j]))
    return records


def apply_mechanism(
    U:      np.ndarray,
    labels: np.ndarray,
    c:      int,
    k:      int,
    d:      int,
    sigma:  float,
    rng:    np.random.Generator,
) -> np.ndarray:
    """
    Naive (non-amortized) reference implementation of T_X(X) = Pi1 [M] X W + B.
    Used to seed the cached observation o (computed once per cell), to
    validate the amortized path in tests, and as a fallback.

    For k >= 1: shape (c^2, d). For k == 0: shape (N_total, d).
    """
    if k >= 1:
        records = class_k_mix_records(labels, c, k, rng)
        m = c * c
        d0 = U.shape[1]
        Xm = np.empty((m, d0), dtype=np.float32)
        coef = 1.0 / (2.0 * k)
        for i, sel in enumerate(records):
            Xm[i] = U[sel].sum(axis=0) * coef
    else:
        Xm = U

    m, d0 = Xm.shape
    W = rng.standard_normal((d0, d)).astype(np.float32) / np.sqrt(d)
    B = rng.standard_normal((m, d)).astype(np.float32) * sigma
    perm = rng.permutation(m)
    return Xm[perm] @ W + B


# Linearity-of-mechanism amortization
#
# T_X(X) = Pi_1 M_mix X W + B is linear in X up to the additive noise B.
# Decompose X = X_base + Delta where X_base zeros out the target's T rows
# and Delta is supported only on those T rows. Then
#     T_X(X) = (Pi_1 M_mix X_base W + B) + Pi_1 M_mix Delta W
#            = o_base + Pi_1 (M_mix Delta) W
# For a candidate hypothesis "all T target rows = r" (a single d0 vector):
#     M_mix Delta_r [i] = (sum_{tr in target} M_mix[i, tr]) * r = alpha[i] * r
# so Pi_1 (M_mix Delta_r) W = alpha_perm * (r W)^T  (rank-1 in (m, d)).
#
# This makes per-candidate likelihood a few O(d) ops once W has been
# applied to the candidate batch, so whole-universe ranking is feasible.
def precompute_trial(
    U:      np.ndarray,
    labels: np.ndarray,
    c:      int,
    k:      int,
    d:      int,
    sigma:  float,
    rng:    np.random.Generator,
) -> Dict:
    """
    Sample (W, B, perm, M_mix) once for this MC trial; precompute UW and
    a partial obfuscation o_full that omits the per-target zeroing. All
    per-target work later only adjusts for the target rows.
    """
    N_total, d0 = U.shape

    if k >= 1:
        records = class_k_mix_records(labels, c, k, rng)
        m = c * c
        coef = 1.0 / (2.0 * k)
    else:
        records = None
        m = N_total
        coef = None  # not used

    perm = rng.permutation(m)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(m)

    W = rng.standard_normal((d0, d)).astype(np.float32) / np.sqrt(d)
    B = rng.standard_normal((m, d)).astype(np.float32) * sigma

    UW = U @ W   # (N_total, d): the dominant per-trial cost

    if k >= 1:
        Xm_full_W = np.empty((m, d), dtype=np.float32)
        for i, sel in enumerate(records):
            Xm_full_W[i] = UW[sel].sum(axis=0) * coef
    else:
        Xm_full_W = UW

    o_full = Xm_full_W[perm] + B

    return {
        "UW": UW, "W": W, "B": B, "perm": perm, "inv_perm": inv_perm,
        "records": records, "coef": coef, "m": m, "k": k, "c": c,
        "sigma": sigma, "Xm_full_W": Xm_full_W, "o_full": o_full,
    }


def precompute_target(trial: Dict, target_rows: np.ndarray) -> Dict:
    """
    Per-(trial, target) precompute: subtract the target rows' contribution
    out of o_full to obtain o_base, and build the alpha vector recording
    target-row presence in each mixed row.
    """
    UW = trial["UW"]
    perm = trial["perm"]
    inv_perm = trial["inv_perm"]
    records = trial["records"]
    coef = trial["coef"]
    m = trial["m"]
    k = trial["k"]
    o_full = trial["o_full"]
    d = UW.shape[1]

    target_rows = np.asarray(target_rows, dtype=int)
    target_set = set(int(t) for t in target_rows.tolist())

    if k >= 1:
        alpha = np.zeros(m, dtype=np.float32)
        beta_W = np.zeros((m, d), dtype=np.float32)
        for i, sel in enumerate(records):
            mask = np.isin(sel, target_rows)
            count = int(mask.sum())
            if count > 0:
                alpha[i] = count * coef
                # contributions from target rows to mixed row i (handles
                # repeated target indices via summation)
                beta_W[i] = UW[sel[mask]].sum(axis=0) * coef
    else:
        alpha = np.zeros(m, dtype=np.float32)
        alpha[target_rows] = 1.0
        beta_W = np.zeros_like(UW)
        beta_W[target_rows] = UW[target_rows]

    o_base = o_full - beta_W[perm]
    alpha_perm = alpha[perm]

    return {
        "o_base":      o_base,
        "alpha_perm":  alpha_perm,
        "alpha_norm2": float(alpha_perm @ alpha_perm),
        "target_rows": target_rows,
    }


def candidate_scores(
    o:           np.ndarray,
    candidates_W: np.ndarray,
    target_pre:  Dict,
    sigma:       float,
) -> np.ndarray:
    """
    Per-candidate likelihood-ranking score under the hypothesis
    "target's T rows are all r" for candidate r. Higher = more likely.

    Drops constants (||e||^2 / 2sigma^2 in particular) since only ranks
    matter. candidates_W must equal candidates @ trial["W"] for the same
    trial that produced target_pre.
    """
    o_base      = target_pre["o_base"]
    alpha_perm  = target_pre["alpha_perm"]
    alpha_norm2 = target_pre["alpha_norm2"]

    e = o - o_base                              # (m, d)
    z = alpha_perm @ e                          # (d,)
    # score(r) = z . v - 0.5 * ||alpha_perm||^2 * ||v||^2,  v = r W
    scores = candidates_W @ z - 0.5 * alpha_norm2 * (candidates_W ** 2).sum(axis=1)
    return scores / (sigma ** 2)


# Weighted scoring on (clip, index) pairs (no proximity window)
def compute_weighted_score(
    guesses:     List[Tuple[int, int]],
    truth:       List[Tuple[int, int]],
    weight_rule: str = "half",
) -> float:
    """
    Score a guess set against the ground-truth set under the
    half-integer rule on (clip, index) pairs:
        exact (clip, index) match -> 1
        same clip, any index      -> 0.5  (only when rule == "half")
        different clip            -> 0
    """
    truth_set = set((int(c), int(i)) for c, i in truth)
    truth_clips = set(c for c, _ in truth_set)
    total = 0.0
    for c_hat, i_hat in guesses:
        key = (int(c_hat), int(i_hat))
        if key in truth_set:
            total += 1.0
        elif weight_rule == "half" and int(c_hat) in truth_clips:
            total += 0.5
    return total


# Closed-form random-guess baselines for the three attacks
def lemma3_random_baseline_attacks(
    n_universe_clips: int,
    num_frames:       int,
    clip_len:         int = CLIP_LEN,
) -> Dict[str, float]:
    """
    Closed-form per-guess baselines for the three temporal MIA attacks
    under the half-integer rule. Universe is n_universe_clips clips,
    each contributing num_frames random sampled indices in [0, clip_len).

    Attack 1 (index inference, clip given): every guess is same-clip,
        score >= 0.5. Exact-match probability is num_frames / clip_len.
        Baseline normalized score = 0.5 * (1 - num_frames/clip_len)
                                    + 1.0 * (num_frames/clip_len).
    Attack 2 (clip inference, top-1): 1 / n_universe_clips.
    Attack 3 (frame-level MIA, half-integer): per-guess
        P(exact) = 1 / (n_universe_clips * clip_len)
        P(same-clip, not exact) = (num_frames * (clip_len - 1))
                                  / (n_universe_clips * clip_len * clip_len)
                                  ~ 1 / n_universe_clips for clip_len >> 1
        Per-guess score ~ 0.5 / n_universe_clips, dominated by the
        same-clip half-credit term.
    """
    p_exact = num_frames / clip_len
    a1_per_guess = 0.5 * (1.0 - p_exact) + 1.0 * p_exact

    a2_top1 = 1.0 / max(1, n_universe_clips)

    a3_p_exact = 1.0 / (n_universe_clips * clip_len)
    a3_p_same_clip_only = (
        (clip_len - 1.0) / (n_universe_clips * clip_len)
    )
    # adversary emits one (clip, idx) per guess uniformly at random over
    # the n_universe_clips * clip_len candidate space; per-guess expected
    # score is dominated by the half-credit same-clip term
    a3_per_guess = 1.0 * a3_p_exact + 0.5 * a3_p_same_clip_only

    return {
        "attack1_per_guess": float(a1_per_guess),
        "attack2_top1":      float(a2_top1),
        "attack3_per_guess": float(a3_per_guess),
    }


# Cached observed release o = T_X(X) — frozen per (k, sigma, seed)
def compute_or_load_o(
    U:         np.ndarray,
    labels:    np.ndarray,
    c:         int,
    k:         int,
    d:         int,
    sigma:     float,
    seed:      int,
    cache_dir: str,
) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"o_k{k}_sigma{sigma:.4f}_seed{seed}.npy"
    )
    if os.path.exists(cache_path):
        print(f"  Loaded cached o from {cache_path}")
        return np.load(cache_path)

    print(f"  Computing o = T_X(X) for k={k}, sigma={sigma}...")
    rng = np.random.default_rng(seed)
    o = apply_mechanism(U, labels, c, k, d, sigma, rng)
    np.save(cache_path, o)
    print(f"  Cached o to {cache_path}, shape={o.shape}")
    return o


# Three-attack run
def run_attack(
    U:             np.ndarray,
    labels:        np.ndarray,
    frame_indices: np.ndarray,
    clip_ids:      np.ndarray,
    clip_paths:    List[str],
    video_root:    str,
    c:             int,
    d:             int,
    sigma:         float,
    k:             int,
    n_targets:     int = 50,
    n_trials:      int = 10,
    num_frames:    int = 16,
    size:          int = 112,
    clip_len:      int = CLIP_LEN,
    seed:          int = 0,
    cache_dir:     str = "./cache",
) -> dict:
    """
    Three-attack temporal membership inference.

      Attack 1: index inference. Adversary is told the target clip ID.
                Ranks all clip_len candidate indices for that clip by
                per-candidate log-likelihood; emits top-T indices.
                Score: half-integer on (clip, index) pairs (clip fixed
                so floor at 0.5/guess).

      Attack 2: clip inference. Per-(clip, frame) likelihoods aggregated
                by clip; top-1 clip emitted. Score: top-1 accuracy.

      Attack 3: frame-level MIA. Top-T (clip, index) pairs over the
                whole universe by per-row likelihood. Score: half-integer.

    All three attacks share the candidate-likelihood machinery in
    candidate_scores; they differ only in the candidate set ranked and
    how guesses are scored.
    """
    rng = np.random.default_rng(seed)
    n_clips = len(clip_paths)
    N_total = U.shape[0]

    target_vids = rng.choice(n_clips, size=n_targets, replace=False)

    # Frozen observation o (one per cell). The MC averaging in run_attack
    # then samples fresh (W, B, perm, M_mix) per trial against this fixed o.
    o = compute_or_load_o(
        U, labels, c, k, d, sigma, seed=seed, cache_dir=cache_dir
    )

    clip_row_indices = [np.where(clip_ids == vid)[0] for vid in range(n_clips)]

    # Pre-load full-clip frames for Attack 1 candidates per target
    full_clips_cache: Dict[int, np.ndarray] = {}

    # We accumulate raw scores across trials per (target, candidate);
    # ranks are taken on the trial-averaged scores. This matches the
    # MC estimator on the (W, B, perm, M_mix) randomness while keeping
    # o frozen as the observation.
    a1_scores = []
    a2_scores = []
    a3_scores = []
    a2_correct = 0
    a3_exact_match_counts = []

    # Pre-compute target metadata (rows, indices, clip-id) for all targets
    targets_meta = []
    for vid_idx in target_vids:
        rows = clip_row_indices[vid_idx]
        if len(rows) != num_frames:
            print(f"  [warn] target clip {vid_idx} has {len(rows)} rows != {num_frames}; skipping")
            continue
        targets_meta.append({
            "vid_idx":  int(vid_idx),
            "rows":     rows,
            "true_idx": frame_indices[rows].astype(int),
            "label":    int(labels[rows][0]),
        })

    # Load full target clips (CLIP_LEN frames) once for Attack 1
    for tm in targets_meta:
        vid_idx = tm["vid_idx"]
        if vid_idx in full_clips_cache:
            continue
        rel = clip_paths[vid_idx]
        full = load_full_clip_gray(
            os.path.join(video_root, rel), clip_len=clip_len, size=size
        )
        if full is None:
            print(f"  [warn] could not load full clip for target {vid_idx} ({rel})")
            continue
        full_clips_cache[vid_idx] = full

    print(f"  Running {n_trials} MC trials over {len(targets_meta)} targets...")

    # Per-target accumulators (trial-averaged score per candidate)
    sum_universe = [
        np.zeros(N_total, dtype=np.float64) for _ in targets_meta
    ]
    sum_index = [
        np.zeros(clip_len, dtype=np.float64) for _ in targets_meta
    ]

    for trial_idx in range(n_trials):
        trial_rng = np.random.default_rng(seed * 10**6 + trial_idx)
        trial = precompute_trial(U, labels, c, k, d, sigma, trial_rng)
        UW = trial["UW"]
        W = trial["W"]

        for ti, tm in enumerate(targets_meta):
            target_pre = precompute_target(trial, tm["rows"])

            # Attack 2 / 3: candidates are universe rows
            scores_uni = candidate_scores(o, UW, target_pre, sigma)
            sum_universe[ti] += scores_uni

            # Attack 1: candidates are CLIP_LEN frames of the target clip
            full = full_clips_cache.get(tm["vid_idx"])
            if full is not None:
                full_W = full @ W
                scores_idx = candidate_scores(o, full_W, target_pre, sigma)
                sum_index[ti] += scores_idx

        if (trial_idx + 1) % max(1, n_trials // 5) == 0:
            print(f"    trial {trial_idx + 1}/{n_trials} done")

    # Score each target on the trial-averaged candidates
    for ti, tm in enumerate(targets_meta):
        avg_uni = sum_universe[ti] / float(n_trials)
        avg_idx = sum_index[ti] / float(n_trials)

        truth = list(zip(
            np.full(num_frames, tm["vid_idx"], dtype=int).tolist(),
            tm["true_idx"].tolist(),
        ))

        # Attack 1: top-T indices in [0, clip_len)
        if tm["vid_idx"] in full_clips_cache:
            top_idx = np.argsort(-avg_idx)[:num_frames]
            a1_guesses = [(tm["vid_idx"], int(i)) for i in top_idx]
            s1 = compute_weighted_score(a1_guesses, truth, weight_rule="half")
        else:
            s1 = float("nan")
        a1_scores.append(s1)

        # Attack 2: aggregate per-clip, emit top-1
        per_clip = np.zeros(n_clips, dtype=np.float64)
        np.add.at(per_clip, clip_ids, avg_uni)
        top1_clip = int(np.argmax(per_clip))
        s2 = 1.0 if top1_clip == tm["vid_idx"] else 0.0
        a2_scores.append(s2)
        if s2 > 0:
            a2_correct += 1

        # Attack 3: top-T (clip, index) pairs from the whole universe
        top_uni = np.argsort(-avg_uni)[:num_frames]
        a3_guesses = [
            (int(clip_ids[r]), int(frame_indices[r])) for r in top_uni
        ]
        s3 = compute_weighted_score(a3_guesses, truth, weight_rule="half")
        a3_scores.append(s3)

        # also report integer-rule (exact-only) score on Attack 3, for §6.5
        s3_int = compute_weighted_score(a3_guesses, truth, weight_rule="integer")
        a3_exact_match_counts.append(s3_int)

    a1_scores = np.array(a1_scores, dtype=float)
    a2_scores = np.array(a2_scores, dtype=float)
    a3_scores = np.array(a3_scores, dtype=float)
    a3_int    = np.array(a3_exact_match_counts, dtype=float)

    base = lemma3_random_baseline_attacks(
        n_universe_clips=n_clips, num_frames=num_frames, clip_len=clip_len,
    )

    # Normalize: per-guess for Attack 1 and Attack 3 (divide by num_frames),
    # already binary for Attack 2.
    n_a1_valid = int((~np.isnan(a1_scores)).sum())
    a1_norm = float(np.nanmean(a1_scores) / num_frames) if n_a1_valid > 0 else float("nan")
    a1_std  = float(np.nanstd(a1_scores)  / num_frames) if n_a1_valid > 0 else float("nan")
    a2_norm = float(a2_scores.mean())
    a2_std  = float(a2_scores.std())
    a3_norm = float(a3_scores.mean() / num_frames)
    a3_std  = float(a3_scores.std()  / num_frames)
    a3_int_norm = float(a3_int.mean() / num_frames)
    a3_int_std  = float(a3_int.std()  / num_frames)

    print(f"\n--- Attack Results (k={k}, sigma={sigma}) ---")
    print(f"Attack 1 (index inference, half) : {a1_norm:.4f}   "
          f"(baseline {base['attack1_per_guess']:.4f})")
    print(f"Attack 2 (clip inference, top-1) : {a2_norm:.4f}   "
          f"(baseline {base['attack2_top1']:.6f})")
    print(f"Attack 3 (frame-level MIA, half) : {a3_norm:.4f}   "
          f"(baseline {base['attack3_per_guess']:.6f})")
    print(f"Attack 3 (frame-level MIA, int)  : {a3_int_norm:.4f}")

    return {
        "k": k, "sigma": sigma,
        "n_targets":         n_targets,
        "n_trials":          n_trials,
        "n_universe_clips":  int(n_clips),
        "n_universe_frames": int(N_total),
        "clip_len":          int(clip_len),
        "num_frames":        int(num_frames),
        # Attack 1
        "attack1_score_half":      a1_norm,
        "attack1_score_half_std":  a1_std,
        "attack1_baseline_half":   base["attack1_per_guess"],
        # Attack 2
        "attack2_top1":            a2_norm,
        "attack2_top1_std":        a2_std,
        "attack2_baseline":        base["attack2_top1"],
        # Attack 3
        "attack3_score_half":      a3_norm,
        "attack3_score_half_std":  a3_std,
        "attack3_score_int":       a3_int_norm,
        "attack3_score_int_std":   a3_int_std,
        "attack3_baseline_half":   base["attack3_per_guess"],
    }


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Three-attack temporal MIA: index inference, clip inference, "
            "and frame-level membership inference."
        )
    )
    parser.add_argument(
        "--k", type=int, required=True, choices=[0, 1, 5],
        help="Mixing parameter: 0 (no mixing), 1, or 5",
    )
    parser.add_argument(
        "--n-targets", type=int, default=50,
        help="Number of target videos to attack (default: 50)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=10,
        help="Number of MC trials per cell (default: 10)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="./results",
        help="Output directory for per-cell CSVs (default: ./results)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./cache",
        help="Cache directory for observed release o (default: ./cache)",
    )
    parser.add_argument(
        "--sigmas", type=float, nargs="+",
        default=[0.01, 0.05, 0.10, 0.50],
        help="Noise standard deviations to sweep",
    )
    args = parser.parse_args()

    SEED = 42
    NUM_FRAMES = 16
    SIZE = 112
    D = 500
    SPLIT = 1

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    VIDEO_ROOT = os.path.join(PROJECT_ROOT, "data", "UCF-101")
    ANNOT_ROOT = os.path.join(PROJECT_ROOT, "data", "ucfTrainTestlist")
    CACHE_PATH = os.path.join(
        PROJECT_ROOT, f"ucf101_frame_pool_gray_random_len{CLIP_LEN}.npz"
    )

    data_root = os.path.join(PROJECT_ROOT, "data")
    download_ucf101(data_root, VIDEO_ROOT, ANNOT_ROOT)
    set_seed(SEED)

    _, train_list, _ = parse_ucf101_split(ANNOT_ROOT, SPLIT)
    print(f"Training videos (raw): {len(train_list)}")

    train_cache = _long_clips_cache_path(ANNOT_ROOT, SPLIT, CLIP_LEN) \
        .replace(".json", "_train.json")
    train_list, dropped = filter_long_clips(
        VIDEO_ROOT, train_list, CLIP_LEN, cache_path=train_cache, desc="train"
    )
    print(
        f"After CLIP_LEN={CLIP_LEN} filter: kept {len(train_list)}, "
        f"dropped {dropped}"
    )
    c = 101

    print("\nBuilding frame pool...")
    U, labels, frame_indices, clip_ids, clip_paths = build_frame_pool(
        video_root=VIDEO_ROOT,
        train_list=train_list,
        cache_path=CACHE_PATH,
        num_frames=NUM_FRAMES,
        size=SIZE,
        seed=SEED,
        clip_len=CLIP_LEN,
    )
    print(f"U shape: {U.shape}, n_clips_in_universe: {len(clip_paths)}")

    os.makedirs(args.results_dir, exist_ok=True)

    for i, sigma in enumerate(args.sigmas):
        out_path = os.path.join(
            args.results_dir, f"attack_k{args.k}_sigma{sigma:.4f}.csv"
        )
        if os.path.exists(out_path):
            print(f"\n[skip] {out_path} already exists")
            continue

        print(f"\n{'=' * 60}")
        print(f"Running attack: k={args.k}, sigma={sigma}")
        print(f"{'=' * 60}")

        result = run_attack(
            U=U, labels=labels, frame_indices=frame_indices,
            clip_ids=clip_ids, clip_paths=clip_paths,
            video_root=VIDEO_ROOT,
            c=c, d=D, sigma=sigma, k=args.k,
            n_targets=args.n_targets, n_trials=args.n_trials,
            num_frames=NUM_FRAMES, size=SIZE, clip_len=CLIP_LEN,
            seed=SEED + i, cache_dir=args.cache_dir,
        )

        df = pd.DataFrame([result])
        df.to_csv(out_path, index=False, float_format="%.6f")
        print(f"Wrote {out_path}")

    print(f"\nAll sigmas complete for k={args.k}.")

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


# LiRA-style hypothesis-test scoring.
#
# The previous formulation built o_base by subtracting the TRUE target-row
# contribution from o_full. Both o_full and o_base then contained the same
# B, so e = o_full - o_base = beta_truth_W[perm] had no B term and the
# candidate ranking was sigma-independent (the score was a constant rescale
# by 1/sigma^2). That made every figure with sigma on the x-axis flat.
#
# The LiRA formulation instead constructs a HYPOTHETICAL baseline: o_base_lira
# is what o would be if the target's rows were replaced by counterfactual
# (same-class, non-target) rows, computed from W/perm/M_mix WITHOUT adding B.
# Then
#     residual := o_full - o_base_lira = M_mix(delta_truth)[perm] W + B
# where delta_truth = (true target rows) - (counterfactual rows). B does
# not cancel here, so candidate scoring becomes a real likelihood ratio
# whose distribution across trials depends on sigma.
#
# Per-candidate score: for candidate hypothesis "target's contribution to
# the mechanism is delta_r" (an N_total x d0 matrix supported only at the
# target's row positions),
#     log P(o | H_r) - log P(o | H_counterfactual)
#       = (residual . pred_diff_r) / sigma^2 - 0.5 ||pred_diff_r||^2 / sigma^2
# where pred_diff_r = M_mix(delta_r - delta_counterfactual)[perm] W.
def precompute_target_lira(
    trial:               Dict,
    target_rows:         np.ndarray,
    counterfactual_rows: np.ndarray,
) -> Dict:
    """
    Build the LiRA residual = o_full - o_base_lira and pre-cache the
    M_mix-touch metadata so per-candidate predicted_diff vectors can be
    formed cheaply.

    target_rows[j] -> counterfactual_rows[j] is a 1-1 replacement; both
    arrays have length T = num_frames.
    """
    UW = trial["UW"]
    perm = trial["perm"]
    records = trial["records"]
    coef = trial["coef"]
    m = trial["m"]
    k = trial["k"]
    o_full = trial["o_full"]
    d = UW.shape[1]

    target_rows = np.asarray(target_rows, dtype=int)
    counterfactual_rows = np.asarray(counterfactual_rows, dtype=int)
    T = len(target_rows)
    assert T == len(counterfactual_rows), "target/counterfactual length mismatch"

    # row -> counterfactual replacement (target rows only; others are
    # identity / not replaced).
    cf_map = dict(zip(target_rows.tolist(), counterfactual_rows.tolist()))
    # Map each target row -> its position in target_rows (for slot lookup
    # in per-clip-candidate scoring).
    row_to_slot = {int(r): s for s, r in enumerate(target_rows.tolist())}

    if k >= 1:
        # For each mixed row i, the counterfactual contribution at that row
        # equals the original M_mix coefficients with target rows substituted
        # for counterfactual rows. The DIFFERENCE
        #     truth_diff_W[i] = (sum_{s in records[i] ∩ target_rows}
        #                        (UW[s] - UW[cf_map[s]]) * coef)
        # is the row-i column of M_mix(delta_truth)[i] @ W.
        truth_diff_W = np.zeros((m, d), dtype=np.float32)
        # Per-row affected-positions metadata used by per-candidate scoring.
        # records_target_hits[i] = list of (slot_in_target, cf_row) tuples,
        # one per occurrence of a target row inside records[i].
        records_target_hits = [[] for _ in range(m)]
        for i, sel in enumerate(records):
            for s in sel.tolist():
                if s in cf_map:
                    cf = cf_map[s]
                    truth_diff_W[i] += (UW[s] - UW[cf]) * coef
                    records_target_hits[i].append((row_to_slot[s], cf))
    else:
        # k=0: M_mix is identity, m = N_total. Each target row maps to
        # exactly one mixed row; the counterfactual swap is row-by-row.
        truth_diff_W = np.zeros_like(UW)
        truth_diff_W[target_rows] = UW[target_rows] - UW[counterfactual_rows]
        records_target_hits = None

    # residual = o_full - o_base_lira = (Xm_full_W - Xm_replaced_W)[perm] + B
    # Xm_full_W - Xm_replaced_W = truth_diff_W (by the construction above),
    # so residual[perm-index j] = truth_diff_W[perm[j]] + B[j].
    residual = truth_diff_W[perm] + trial["B"]

    return {
        "residual":            residual,
        "truth_diff_W":        truth_diff_W,
        "target_rows":         target_rows,
        "counterfactual_rows": counterfactual_rows,
        "records_target_hits": records_target_hits,
    }


def lira_score_clip_candidates(
    trial:    Dict,
    target_pre: Dict,
    candidate_clip_rows: np.ndarray,
    sigma:    float,
) -> np.ndarray:
    """
    Score candidate clips for Attack 2.

    candidate_clip_rows: (n_candidates, T) array of universe row indices,
        one row of T positions per candidate clip. The candidate's
        hypothesis is "this clip's T rows occupy the target's slots"
        (so its delta vs the counterfactual baseline is the candidate's
        rows minus the counterfactual rows, slot-by-slot).

    Returns: (n_candidates,) LR scores, divided by sigma^2.
    """
    UW = trial["UW"]
    perm = trial["perm"]
    inv_perm = trial["inv_perm"]
    records = trial["records"]
    coef = trial["coef"]
    m = trial["m"]
    k = trial["k"]
    d = UW.shape[1]

    residual = target_pre["residual"]
    target_rows = target_pre["target_rows"]
    counterfactual_rows = target_pre["counterfactual_rows"]
    records_target_hits = target_pre["records_target_hits"]

    n_cand = candidate_clip_rows.shape[0]
    T = len(target_rows)
    assert candidate_clip_rows.shape[1] == T

    scores = np.zeros(n_cand, dtype=np.float32)

    # k=0: pred is sparse with only T nonzero rows (target_rows). The
    # post-perm row positions of the target slots are inv_perm[target_rows].
    # Use that to compute the residual-pred dot and the squared-norm in
    # O(T*d) instead of O(N_total*d).
    if k == 0:
        target_perm_idx = inv_perm[target_rows]                  # (T,)
        residual_at_target = residual[target_perm_idx]           # (T, d)
        for ci in range(n_cand):
            c_rows = candidate_clip_rows[ci]
            slot_diff_W = (
                UW[c_rows].astype(np.float32)
                - UW[counterfactual_rows].astype(np.float32)
            )                                                     # (T, d)
            dot = float((residual_at_target * slot_diff_W).sum())
            nrm = float((slot_diff_W * slot_diff_W).sum())
            scores[ci] = (dot - 0.5 * nrm) / (sigma ** 2)
        return scores

    # k>=1: pred has nonzero entries only at mixed rows that touch a target
    # slot (records_target_hits). Per-candidate cost is the total slot-hit
    # count across all mixed rows (~ T*c for typical M_mix).
    for ci in range(n_cand):
        c_rows = candidate_clip_rows[ci]
        slot_diff_W = (
            UW[c_rows].astype(np.float32) - UW[counterfactual_rows].astype(np.float32)
        )  # (T, d)
        pred = np.zeros((m, d), dtype=np.float32)
        for i, hits in enumerate(records_target_hits):
            if not hits:
                continue
            acc = np.zeros(d, dtype=np.float32)
            for slot_in_target, _ in hits:
                acc += slot_diff_W[slot_in_target] * coef
            pred[i] = acc
        pred_perm = pred[perm]

        dot = float((residual * pred_perm).sum())
        nrm = float((pred_perm * pred_perm).sum())
        scores[ci] = (dot - 0.5 * nrm) / (sigma ** 2)

    return scores


def lira_score_frame_candidates(
    trial:               Dict,
    target_pre:          Dict,
    candidate_rows:      np.ndarray,
    sigma:               float,
) -> np.ndarray:
    """
    Score single-frame candidates for Attacks 1 and 3. Each candidate
    hypothesizes "this single universe row r occupies one of the target's
    T slots, replacing one counterfactual row," marginalized uniformly
    over the T slots (sum of slot contributions, with 1/T normalization
    folded into ||pred||^2 only).

    candidate_rows: (n_candidates,) array of universe row indices.
    Returns: (n_candidates,) LR scores divided by sigma^2.
    """
    UW = trial["UW"]
    perm = trial["perm"]
    inv_perm = trial["inv_perm"]
    records = trial["records"]
    coef = trial["coef"]
    m = trial["m"]
    k = trial["k"]
    d = UW.shape[1]

    residual = target_pre["residual"]
    target_rows = target_pre["target_rows"]
    counterfactual_rows = target_pre["counterfactual_rows"]
    records_target_hits = target_pre["records_target_hits"]

    n_cand = candidate_rows.shape[0]
    T = len(target_rows)
    inv_T = 1.0 / float(T)

    # Sparse summary statistics drive every per-candidate score:
    #   residual_alpha = sum_j alpha_perm[j] * residual[j]
    #   cf_dot_residual = sum_j residual[j] . cf_perm[j]
    #   alpha2_sum = sum_j alpha_perm[j]^2
    #   cf_norm2_sum = sum_j ||cf_perm[j]||^2
    #   a_cf = sum_j alpha_perm[j] * cf_perm[j]
    # For k=0, alpha_perm is 0/1 valued at inv_perm[target_rows] only and
    # cf_perm is nonzero only there too — we compute everything by indexing
    # those T positions, no full (N_total, d) materialization.
    if k == 0:
        target_perm_idx = inv_perm[target_rows]              # (T,)
        residual_at_target = residual[target_perm_idx]       # (T, d)
        cf_at_target = UW[counterfactual_rows].astype(np.float32)  # (T, d)

        residual_alpha = residual_at_target.sum(axis=0)      # (d,)
        cf_dot_residual = float((residual_at_target * cf_at_target).sum())
        alpha2_sum = float(T)
        cf_norm2_sum = float((cf_at_target * cf_at_target).sum())
        a_cf = cf_at_target.sum(axis=0)                      # (d,)
    else:
        # k>=1: alpha_perm and cf_perm have nonzeros only at mixed rows
        # that touched a target slot. Build the full (m,) and (m, d)
        # vectors then collapse — typical |records_target_hits with hits|
        # << m, so this is the dominant cost only for very large c.
        alpha_rows = np.zeros(m, dtype=np.float32)
        cf_contrib = np.zeros((m, d), dtype=np.float32)
        for i, hits in enumerate(records_target_hits):
            if not hits:
                continue
            acc_a = 0.0
            acc_v = np.zeros(d, dtype=np.float32)
            for _, cf_row in hits:
                acc_a += coef
                acc_v += UW[cf_row] * coef
            alpha_rows[i] = acc_a
            cf_contrib[i] = acc_v
        alpha_perm = alpha_rows[perm]
        cf_perm = cf_contrib[perm]

        residual_alpha = (alpha_perm[:, None] * residual).sum(axis=0)  # (d,)
        cf_dot_residual = float((residual * cf_perm).sum())
        alpha2_sum = float((alpha_perm ** 2).sum())
        cf_norm2_sum = float((cf_perm * cf_perm).sum())
        a_cf = (alpha_perm[:, None] * cf_perm).sum(axis=0)             # (d,)

    cand_UW = UW[candidate_rows]                                       # (n_cand, d)
    cand_norm2 = (cand_UW * cand_UW).sum(axis=1)                       # (n_cand,)
    cross = cand_UW @ a_cf                                             # (n_cand,)

    dot_terms = (cand_UW @ residual_alpha) * inv_T - cf_dot_residual * inv_T
    nrm_terms = (
        (alpha2_sum * cand_norm2 - 2.0 * cross + cf_norm2_sum) * (inv_T ** 2)
    )

    return (dot_terms - 0.5 * nrm_terms) / (sigma ** 2)


def _lira_score_external_uw(
    trial:      Dict,
    target_pre: Dict,
    cand_UW:    np.ndarray,
    sigma:      float,
) -> np.ndarray:
    """
    Same LR formula as lira_score_frame_candidates, but the candidates'
    W-projections are passed in directly rather than indexed from UW.
    Used by Attack 1 where candidates are CLIP_LEN frames of the
    target's full clip, which are not represented as rows in U.
    """
    UW = trial["UW"]
    perm = trial["perm"]
    inv_perm = trial["inv_perm"]
    coef = trial["coef"]
    m = trial["m"]
    k = trial["k"]
    d = UW.shape[1]

    residual = target_pre["residual"]
    target_rows = target_pre["target_rows"]
    counterfactual_rows = target_pre["counterfactual_rows"]
    records_target_hits = target_pre["records_target_hits"]
    T = len(target_rows)
    inv_T = 1.0 / float(T)

    if k == 0:
        target_perm_idx = inv_perm[target_rows]
        residual_at_target = residual[target_perm_idx]
        cf_at_target = UW[counterfactual_rows].astype(np.float32)

        residual_alpha = residual_at_target.sum(axis=0)
        cf_dot_residual = float((residual_at_target * cf_at_target).sum())
        alpha2_sum = float(T)
        cf_norm2_sum = float((cf_at_target * cf_at_target).sum())
        a_cf = cf_at_target.sum(axis=0)
    else:
        alpha_rows = np.zeros(m, dtype=np.float32)
        cf_contrib = np.zeros((m, d), dtype=np.float32)
        for i, hits in enumerate(records_target_hits):
            if not hits:
                continue
            acc_a = 0.0
            acc_v = np.zeros(d, dtype=np.float32)
            for _, cf_row in hits:
                acc_a += coef
                acc_v += UW[cf_row] * coef
            alpha_rows[i] = acc_a
            cf_contrib[i] = acc_v
        alpha_perm = alpha_rows[perm]
        cf_perm = cf_contrib[perm]

        residual_alpha = (alpha_perm[:, None] * residual).sum(axis=0)
        cf_dot_residual = float((residual * cf_perm).sum())
        alpha2_sum = float((alpha_perm ** 2).sum())
        cf_norm2_sum = float((cf_perm * cf_perm).sum())
        a_cf = (alpha_perm[:, None] * cf_perm).sum(axis=0)

    cand_norm2 = (cand_UW * cand_UW).sum(axis=1)
    cross = cand_UW @ a_cf

    dot_terms = (cand_UW @ residual_alpha) * inv_T - cf_dot_residual * inv_T
    nrm_terms = (
        (alpha2_sum * cand_norm2 - 2.0 * cross + cf_norm2_sum) * (inv_T ** 2)
    )

    return (dot_terms - 0.5 * nrm_terms) / (sigma ** 2)


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
    Attack 3 (frame-level MIA, half-integer): truth set has T = num_frames
        (clip, index) pairs, all in one clip. For one uniformly random
        guess over the n_universe_clips * clip_len candidate-pair space:
            P(exact) = T / (n_universe_clips * clip_len)
            P(same-clip, not exact) = (clip_len - T) / (n_universe_clips * clip_len)
        Per-guess expected score = 1 * P(exact) + 0.5 * P(same-clip-not-exact),
        dominated by the half-credit term for clip_len >> T.
    """
    p_exact = num_frames / clip_len
    a1_per_guess = 0.5 * (1.0 - p_exact) + 1.0 * p_exact

    a2_top1 = 1.0 / max(1, n_universe_clips)

    a3_p_exact = num_frames / (n_universe_clips * clip_len)
    a3_p_same_clip_only = (
        (clip_len - num_frames) / (n_universe_clips * clip_len)
    )
    a3_per_guess = 1.0 * a3_p_exact + 0.5 * a3_p_same_clip_only

    return {
        "attack1_per_guess": float(a1_per_guess),
        "attack2_top1":      float(a2_top1),
        "attack3_per_guess": float(a3_per_guess),
    }


# Three-attack run (LiRA-style scoring)
#
# Per (target, trial):
#   1. Sample fresh (W, B, perm, M_mix). Compute o = M(U)[perm] W + B.
#   2. Sample T counterfactual rows (same-class, not from target clip).
#   3. Build LiRA target_pre: residual = o - M(U_replaced)[perm] W
#      where U_replaced has target's rows swapped for counterfactuals.
#      Critically, M(U_replaced)[perm] W is computed WITHOUT B, so the
#      residual carries the B noise term and ranks become sigma-sensitive
#      across trials.
#   4. For each attack, score candidate set under the LiRA likelihood
#      ratio and emit top-K guesses.
#
# Per-trial scoring (not per-target-trial-averaged) — each trial is one
# MC sample of attack success against this cell's mechanism family.
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
    n_targets:        int = 10,
    n_trials:         int = 5,
    n_clip_candidates:    int = 50,
    n_frame_candidates_a3: int = 2000,
    num_frames:    int = 16,
    size:          int = 112,
    clip_len:      int = CLIP_LEN,
    seed:          int = 0,
) -> dict:
    """LiRA-style three-attack temporal MIA.

    Compute-saving knobs:
      n_targets         — number of target clips evaluated per cell.
      n_trials          — MC trials per (target, cell).
      n_clip_candidates — Attack 2 ranks this many same-class candidate
                          clips (always includes the true target).
      n_frame_candidates_a3 — Attack 3 subsamples this many universe
                          rows as candidate (clip, index) pairs.
    """
    rng = np.random.default_rng(seed)
    n_clips = len(clip_paths)
    N_total = U.shape[0]

    clip_row_indices = [np.where(clip_ids == vid)[0] for vid in range(n_clips)]

    # Pre-compute target metadata
    target_vids = rng.choice(n_clips, size=min(n_targets, n_clips), replace=False)
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

    # Group rows by class for counterfactual / Attack 2 sampling
    class_to_rows = {int(lbl): np.where(labels == lbl)[0] for lbl in np.unique(labels)}
    class_to_clips: Dict[int, np.ndarray] = {}
    for vid in range(n_clips):
        lbl = int(labels[clip_row_indices[vid][0]])
        class_to_clips.setdefault(lbl, []).append(vid)
    class_to_clips = {k_: np.asarray(v) for k_, v in class_to_clips.items()}

    # Pre-load full target clips for Attack 1
    full_clips_cache: Dict[int, np.ndarray] = {}
    for tm in targets_meta:
        rel = clip_paths[tm["vid_idx"]]
        full = load_full_clip_gray(
            os.path.join(video_root, rel), clip_len=clip_len, size=size
        )
        if full is None:
            print(f"  [warn] could not load full clip for target {tm['vid_idx']} ({rel})")
            continue
        full_clips_cache[tm["vid_idx"]] = full

    # Subsample universe candidate set for Attack 3 (compute-saving)
    if n_frame_candidates_a3 < N_total:
        a3_pool = rng.choice(N_total, size=n_frame_candidates_a3, replace=False)
    else:
        a3_pool = np.arange(N_total)

    a1_scores, a1_int_scores = [], []
    a2_scores = []
    a2_actual_n_candidates = []   # may be < n_clip_candidates for sparse classes
    a3_scores, a3_int_scores = [], []

    print(
        f"  Running {n_trials} trials x {len(targets_meta)} targets "
        f"(k={k}, sigma={sigma}, |U_clips|={n_clips}, |U_rows|={N_total})..."
    )

    for ti, tm in enumerate(targets_meta):
        target_rows = tm["rows"]
        target_label = tm["label"]
        target_vid = tm["vid_idx"]

        # Same-class candidate clips for Attack 2 (always include true target)
        same_class = class_to_clips[target_label]
        same_class_no_target = same_class[same_class != target_vid]
        n_to_pick = max(0, n_clip_candidates - 1)
        if len(same_class_no_target) > n_to_pick:
            picks = rng.choice(same_class_no_target, size=n_to_pick, replace=False)
        else:
            picks = same_class_no_target
        candidate_clip_ids = np.concatenate([picks, [target_vid]])
        # candidate_clip_rows[ci]: T universe rows of candidate clip ci
        candidate_clip_rows = np.stack(
            [clip_row_indices[int(cid)] for cid in candidate_clip_ids],
            axis=0,
        )
        true_target_index_among_candidates = len(candidate_clip_ids) - 1
        a2_actual_n_candidates.append(len(candidate_clip_ids))

        truth = list(zip(
            np.full(num_frames, target_vid, dtype=int).tolist(),
            tm["true_idx"].tolist(),
        ))

        for trial_idx in range(n_trials):
            trial_rng = np.random.default_rng(
                seed * 10**6 + ti * 1000 + trial_idx
            )
            trial = precompute_trial(U, labels, c, k, d, sigma, trial_rng)
            UW = trial["UW"]
            W = trial["W"]

            # Sample counterfactual rows: T same-class frames not in target clip
            same_class_rows = class_to_rows[target_label]
            non_target = np.setdiff1d(same_class_rows, target_rows, assume_unique=False)
            if len(non_target) >= num_frames:
                counterfactual_rows = trial_rng.choice(
                    non_target, size=num_frames, replace=False
                )
            else:
                counterfactual_rows = trial_rng.choice(
                    non_target, size=num_frames, replace=True
                )

            target_pre = precompute_target_lira(
                trial, target_rows, counterfactual_rows
            )

            # ---- Attack 2: top-1 over candidate clips ----
            a2_lr = lira_score_clip_candidates(
                trial, target_pre, candidate_clip_rows, sigma
            )
            top1 = int(np.argmax(a2_lr))
            a2_scores.append(1.0 if top1 == true_target_index_among_candidates else 0.0)

            # ---- Attack 3: top-T over subsampled (clip, index) pairs ----
            # Always include the true target rows in the pool so the test is
            # well-defined. (If subsampling missed them they'd never be in
            # the top-T.)
            a3_pool_with_truth = np.unique(
                np.concatenate([a3_pool, target_rows])
            )
            a3_lr = lira_score_frame_candidates(
                trial, target_pre, a3_pool_with_truth, sigma
            )
            top_ranks = np.argsort(-a3_lr)[:num_frames]
            top_rows_a3 = a3_pool_with_truth[top_ranks]
            a3_guesses = [
                (int(clip_ids[r]), int(frame_indices[r])) for r in top_rows_a3
            ]
            a3_scores.append(
                compute_weighted_score(a3_guesses, truth, weight_rule="half")
            )
            a3_int_scores.append(
                compute_weighted_score(a3_guesses, truth, weight_rule="integer")
            )

            # ---- Attack 1: top-T frames of target's CLIP_LEN frames ----
            full = full_clips_cache.get(target_vid)
            if full is None:
                a1_scores.append(float("nan"))
                a1_int_scores.append(float("nan"))
                continue

            # Each candidate frame f at index j contributes a single-frame
            # delta. We score by treating each candidate as if it occupied
            # one of the target's T slots (slot-marginalized; same machinery
            # as Attack 3 but with the candidate's W-projection placed at
            # the target's rows in U "virtually" — we form a synthetic
            # universe extension to plug in lira_score_frame_candidates).
            #
            # Implementation: append the CLIP_LEN frames to UW as virtual
            # rows. Compute their per-row LR via the same closed form as
            # Attack 3, treating them as "candidate rows" replacing target
            # slots. The hits / counterfactual structure is unchanged.
            full_W = full.astype(np.float32) @ W   # (clip_len, d)

            # Inline a copy of lira_score_frame_candidates that takes UW
            # rows directly (we don't have universe row indices for the
            # virtual frames, so we pass their UW vectors).
            scores_a1 = _lira_score_external_uw(
                trial, target_pre, full_W, sigma
            )
            top_idx_a1 = np.argsort(-scores_a1)[:num_frames]
            a1_guesses = [(target_vid, int(i)) for i in top_idx_a1]
            a1_scores.append(
                compute_weighted_score(a1_guesses, truth, weight_rule="half")
            )
            a1_int_scores.append(
                compute_weighted_score(a1_guesses, truth, weight_rule="integer")
            )

        if (ti + 1) % max(1, len(targets_meta) // 5) == 0:
            print(f"    target {ti + 1}/{len(targets_meta)} done")

    a1_scores = np.array(a1_scores, dtype=float)
    a1_int_scores = np.array(a1_int_scores, dtype=float)
    a2_scores = np.array(a2_scores, dtype=float)
    a3_scores = np.array(a3_scores, dtype=float)
    a3_int    = np.array(a3_int_scores, dtype=float)

    base = lemma3_random_baseline_attacks(
        n_universe_clips=n_clips, num_frames=num_frames, clip_len=clip_len,
    )

    # Normalize: per-guess for Attack 1 and Attack 3 (divide by num_frames),
    # already binary for Attack 2.
    n_a1_valid = int((~np.isnan(a1_scores)).sum())
    a1_norm = float(np.nanmean(a1_scores) / num_frames) if n_a1_valid > 0 else float("nan")
    a1_std  = float(np.nanstd(a1_scores)  / num_frames) if n_a1_valid > 0 else float("nan")
    a1_int_norm = float(np.nanmean(a1_int_scores) / num_frames) if n_a1_valid > 0 else float("nan")
    a2_norm = float(a2_scores.mean()) if len(a2_scores) else float("nan")
    a2_std  = float(a2_scores.std())  if len(a2_scores) else float("nan")
    a3_norm = float(a3_scores.mean() / num_frames) if len(a3_scores) else float("nan")
    a3_std  = float(a3_scores.std()  / num_frames) if len(a3_scores) else float("nan")
    a3_int_norm = float(a3_int.mean() / num_frames) if len(a3_int) else float("nan")
    a3_int_std  = float(a3_int.std()  / num_frames) if len(a3_int) else float("nan")

    # Use the *actual* mean candidate count for the baseline. With
    # ~75 clips/class on UCF-101 this matches n_clip_candidates exactly,
    # but small classes (e.g. when filter_long_clips trimmed a class hard)
    # cap below n_clip_candidates and the baseline shifts accordingly.
    a2_actual_mean = (
        float(np.mean(a2_actual_n_candidates))
        if a2_actual_n_candidates else float(n_clip_candidates)
    )
    a2_baseline_topN = 1.0 / max(1.0, a2_actual_mean)

    print(f"\n--- Attack Results (k={k}, sigma={sigma}) ---")
    print(f"Attack 1 (index inference, half) : {a1_norm:.4f}   "
          f"(baseline {base['attack1_per_guess']:.4f})")
    print(f"Attack 2 (clip inference, top-1) : {a2_norm:.4f}   "
          f"(baseline {a2_baseline_topN:.4f}, |cands|~{a2_actual_mean:.1f})")
    print(f"Attack 3 (frame-level MIA, half) : {a3_norm:.4f}   "
          f"(baseline {base['attack3_per_guess']:.6f})")
    print(f"Attack 3 (frame-level MIA, int)  : {a3_int_norm:.4f}")

    return {
        "k": k, "sigma": sigma,
        "n_targets":              n_targets,
        "n_trials":               n_trials,
        "n_clip_candidates":      n_clip_candidates,
        "n_frame_candidates_a3":  n_frame_candidates_a3,
        "n_universe_clips":       int(n_clips),
        "n_universe_frames":      int(N_total),
        "clip_len":               int(clip_len),
        "num_frames":             int(num_frames),
        # Attack 1
        "attack1_score_half":      a1_norm,
        "attack1_score_half_std":  a1_std,
        "attack1_score_int":       a1_int_norm,
        "attack1_baseline_half":   base["attack1_per_guess"],
        # Attack 2 (top-1 over n_clip_candidates same-class clips, restricted)
        "attack2_top1":            a2_norm,
        "attack2_top1_std":        a2_std,
        "attack2_baseline":        a2_baseline_topN,
        "attack2_actual_n_candidates": a2_actual_mean,
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
        "--n-targets", type=int, default=10,
        help="Number of target videos to attack (default: 10; small "
             "for fast iteration. Scale up for paper figures.)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=5,
        help="MC trials per (target, cell) (default: 5).",
    )
    parser.add_argument(
        "--n-clip-candidates", type=int, default=50,
        help="Attack 2 same-class candidate set size (default: 50). "
             "Top-1 chance baseline = 1/n_clip_candidates.",
    )
    parser.add_argument(
        "--n-frame-candidates-a3", type=int, default=2000,
        help="Attack 3 universe subsample size (default: 2000). "
             "True target rows are always added on top.",
    )
    parser.add_argument(
        "--results-dir", type=str, default="./results",
        help="Output directory for per-cell CSVs (default: ./results)",
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

    # Some UCF-101 classes can lose all their training clips to the
    # CLIP_LEN filter; class_k_mix_records would then sample an empty
    # class. Remap labels to dense [0, c) over surviving classes.
    present = np.unique(labels)
    if len(present) < 101:
        print(
            f"[warn] {101 - len(present)} class(es) have no surviving "
            f"training clips after CLIP_LEN={CLIP_LEN}; remapping labels "
            f"to dense [0, {len(present)})."
        )
    labels = np.searchsorted(present, labels).astype(int)
    c = int(len(present))

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
            n_clip_candidates=args.n_clip_candidates,
            n_frame_candidates_a3=args.n_frame_candidates_a3,
            num_frames=NUM_FRAMES, size=SIZE, clip_len=CLIP_LEN,
            seed=SEED + i,
        )

        df = pd.DataFrame([result])
        df.to_csv(out_path, index=False, float_format="%.6f")
        print(f"Wrote {out_path}")

    print(f"\nAll sigmas complete for k={args.k}.")

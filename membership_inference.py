"""
membership_inference.py

Weighted membership inference attack for temporally correlated video data.
Follows Section 7.2 of Xiao et al. (2024), extended to half-integer weights
based on temporal proximity of frame indices within each video.

Assumes main_video.py has already been run and UCF-101 data is available at
./data/UCF-101 and ./data/ucfTrainTestlist.

Pipeline:
  - U = pool of individual grayscale frames across all training videos
  - X = n-subset of frames (the T sampled frames per video, across all videos)
  - T_X(X) = Pi1 M X W + B  (Algorithm 1 of Xiao et al.)
  - Adversary observes o = T_X(X) and tries to infer which T frames
    were sampled from each target video, scored with half-integer weights.
"""

import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import json

# reuse utilities from main pipeline
from main_video import (
    set_seed,
    get_device,
    parse_ucf101_split,
)


# ----------------------------
# Frame loading: grayscale, no embedding model
# ----------------------------
def load_video_frames_gray(
    path:       str,
    num_frames: int = 16,
    size:       int = 112,
    rng:        np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load a video, randomly sample num_frames without loading all frames
    into memory. Uses cv2 seeking to read only the selected frames.

    Returns:
      frames:        (T, H*W) float32 in [0, 1]
      frame_indices: (T,)     int array of sampled frame indices (sorted)
      total_frames:  total number of frames in the video
    """
    if rng is None:
        rng = np.random.default_rng()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    # get total frame count without reading all frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        # CAP_PROP_FRAME_COUNT unreliable for some codecs — fall back
        # to counting frames sequentially (only for problematic videos)
        frames_buf = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_buf.append(frame)
        cap.release()
        total_frames = len(frames_buf)
        if total_frames == 0:
            raise RuntimeError(f"No frames read from: {path}")
        replace      = total_frames < num_frames
        frame_indices = np.sort(
            rng.choice(total_frames, size=num_frames, replace=replace)
        ).astype(int)
        sampled = [frames_buf[i] for i in frame_indices]
    else:
        cap.release()
        replace       = total_frames < num_frames
        frame_indices = np.sort(
            rng.choice(total_frames, size=num_frames, replace=replace)
        ).astype(int)

        # re-open and seek to each selected frame index
        cap    = cv2.VideoCapture(path)
        sampled = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                # seek failed — fall back to nearest readable frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - 1))
                ret, frame = cap.read()
            if ret:
                sampled.append(frame)
        cap.release()

        if len(sampled) == 0:
            raise RuntimeError(f"No frames read from: {path}")

        # if some seeks failed, pad with last successful frame
        while len(sampled) < num_frames:
            sampled.append(sampled[-1])

    # convert to grayscale and flatten
    result = []
    for frame in sampled:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (size, size))
        result.append(gray)

    stack   = np.stack(result, axis=0)                     # (T, H, W)
    T, H, W = stack.shape
    frames  = stack.reshape(T, H * W).astype(np.float32) / 255.0

    return frames, frame_indices, total_frames

# ----------------------------
# Build frame pool U from training videos
# ----------------------------
def build_frame_pool(
    video_root: str,
    train_list: List[Tuple[str, int]],
    cache_path: str,
    num_frames: int = 16,
    size: int = 112,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all training videos and extract individual grayscale frames.

    Returns:
      U:             (N_total, d0) float32 — all frames from all videos
      labels:        (N_total,)    int     — action class per frame
      frame_indices: (N_total,)    int     — original frame index within its video
      clip_ids:      (N_total,)    int     — which video each frame came from
      clip_lengths:  (n_clips,)    int     — total frame count per video
    """
    if os.path.exists(cache_path):
        obj = np.load(cache_path, allow_pickle=True)
        print(f"Loaded frame pool from cache: {cache_path}")
        return (
            obj["U"],
            obj["labels"],
            obj["frame_indices"],
            obj["clip_ids"],
            obj["clip_lengths"],
        )

    U_list = []
    labels_list = []
    frame_indices_list = []
    clip_ids_list = []
    clip_lengths_list = []

    # per-clip rng for reproducible random frame sampling
    pool_rng = np.random.default_rng(seed)

    for clip_id, (rel_path, label) in enumerate(train_list):
        video_path = os.path.join(video_root, rel_path)
        try:
            clip_rng = np.random.default_rng(pool_rng.integers(1 << 62))
            frames, fidxs, total = load_video_frames_gray(
                video_path, num_frames=num_frames, size=size, rng=clip_rng
            )
        except RuntimeError as e:
            print(f"  Skipping {rel_path}: {e}")
            continue

        T = frames.shape[0]
        U_list.append(frames)
        labels_list.append(np.full(T, label, dtype=int))
        frame_indices_list.append(fidxs)
        clip_ids_list.append(np.full(T, clip_id, dtype=int))
        clip_lengths_list.append(total)

        if (clip_id + 1) % 200 == 0:
            print(f"  Loaded {clip_id + 1}/{len(train_list)} videos")

    U = np.concatenate(U_list, axis=0).astype(np.float32)
    labels = np.concatenate(labels_list, axis=0)
    frame_indices = np.concatenate(frame_indices_list, axis=0)
    clip_ids = np.concatenate(clip_ids_list, axis=0)
    clip_lengths = np.array(clip_lengths_list, dtype=int)

    np.savez(
        cache_path,
        U=U,
        labels=labels,
        frame_indices=frame_indices,
        clip_ids=clip_ids,
        clip_lengths=clip_lengths,
    )
    print(f"Frame pool cached to {cache_path}")
    print(f"Total frames: {U.shape[0]}, d0={U.shape[1]}")
    return U, labels, frame_indices, clip_ids, clip_lengths


# ----------------------------
# Obfuscation: T_X(X) = Pi1 M X W + B
# ----------------------------
def class_k_mix(
    X: np.ndarray,          # (N, d0)
    labels: np.ndarray,     # (N,)
    c: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Class-k mixing at the individual frame level.
    For each ordered pair (i, j) in [c] x [c], sample k frames from
    class i and k from class j, average them → one mixed frame.

    Returns:
      Xm: (c^2, d0) mixed frames
    """
    d0 = X.shape[1]
    m = c * c
    Xm = np.zeros((m, d0), dtype=np.float32)
    cls_to_idx = [np.where(labels == cls)[0] for cls in range(c)]

    t = 0
    for i in range(c):
        for j in range(c):
            idx_i = rng.choice(cls_to_idx[i], size=k, replace=True)
            idx_j = rng.choice(cls_to_idx[j], size=k, replace=True)
            Xm[t] = X[np.concatenate([idx_i, idx_j])].mean(axis=0)
            t += 1

    return Xm


def obfuscate(
    Xm: np.ndarray,     # (m, d0)
    d: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply T_X: Pi1 Xm W + B

      W   ~ N(0, 1/d)   shape (d0, d)
      B   ~ N(0, sigma)  shape (m, d)
      Pi1 = random permutation over rows

    Returns o: (m, d) obfuscated release
    """
    m, d0 = Xm.shape
    W = rng.standard_normal((d0, d)).astype(np.float32) / np.sqrt(d)
    B = rng.standard_normal((m, d)).astype(np.float32) * np.sqrt(sigma)
    perm = rng.permutation(m)
    return Xm[perm] @ W + B


# ----------------------------
# Weighted scoring
# ----------------------------
def compute_weighted_score(
    guessed_indices: np.ndarray,   # (T,) guessed frame indices
    true_indices: np.ndarray,      # (T,) true frame indices
    clip_length: int,
) -> float:
    """
    Half-integer weighted overlap between guessed and true frame indices.

      w = 1   if exact match  (distance == 0)
      w = 1/2 if distance <= 0.05 * clip_length
      w = 0   otherwise

    Returns total weighted score in [0, T].
    """
    window = 0.05 * clip_length
    total_weight = 0.0
    for g in guessed_indices:
        min_dist = np.abs(true_indices - g).min()
        if min_dist == 0:
            total_weight += 1.0
        elif min_dist <= window:
            total_weight += 0.5
    return total_weight


# ----------------------------
# Log-likelihood under Gaussian noise
# ----------------------------
def log_likelihood(
    o: np.ndarray,      # (m, d) observed release
    o_sim: np.ndarray,  # (m, d) simulated release
    sigma: float,
) -> float:
    """
    log P(o | X_sim) = -||o - o_sim||^2 / (2*sigma) + const
    Constant dropped since we only need relative comparisons.
    """
    diff = o - o_sim
    return -float(np.sum(diff ** 2)) / (2.0 * sigma)


# ----------------------------
# Main attack
# ----------------------------
def run_attack(
    U: np.ndarray,
    labels: np.ndarray,
    frame_indices: np.ndarray,
    clip_ids: np.ndarray,
    clip_lengths: np.ndarray,
    c: int,
    d: int,
    sigma: float,
    k: int,
    n_targets: int = 100,
    n_trials: int = 100,
    num_frames: int = 16,
    seed: int = 0,
) -> dict:
    """
    Weighted membership inference attack following Section 7.2 of
    Xiao et al. (2024).

    For each target video, estimates the marginal log-likelihood of
    the observed release o under two hypotheses:

      H+: target video's true frames are in X
      H-: target video's frames are replaced by same-class frames
          from other videos

    Each hypothesis is simulated n_trials times with fresh independent
    keys (Π, M, W, B), and the average log-likelihood is compared.
    The adversary guesses H+ if mean ll+ > mean ll-.

    This matches Eq. (13) of Xiao et al.:
      P(X_hat = X0) = P(X = X0 | M(X) = o)
    estimated via Monte Carlo averaging of the Gaussian likelihood
      log P(o | X_sim, Π, M, W) = -||o - Π M X_sim W + B||² / (2σ)
    """
    rng     = np.random.default_rng(seed)
    n_clips = clip_lengths.shape[0]

    # select target videos
    target_vids = rng.choice(n_clips, size=n_targets, replace=False)

    # build true observed release o = T_X(X) with fixed keys
    print("Building true obfuscated release o = T_X(X)...")
    Xm_true = class_k_mix(U, labels, c, k, rng)
    o       = obfuscate(Xm_true, d, sigma, rng)
    print(f"  o shape: {o.shape}")

    # pre-compute row indices per clip to avoid repeated np.where
    clip_row_indices = [
        np.where(clip_ids == vid)[0]
        for vid in range(n_clips)
    ]

    weighted_scores = []
    h_plus_wins     = []
    mean_deltas     = []

    for t_num, vid_idx in enumerate(target_vids):

        vid_rows        = clip_row_indices[vid_idx]
        true_fidxs      = frame_indices[vid_rows]
        clip_len        = clip_lengths[vid_idx]
        true_label      = labels[vid_rows][0]

        # pool of same-class frames from other videos for H- swapping
        same_class_mask = (labels == true_label) & (
            ~np.isin(np.arange(len(labels)), vid_rows)
        )
        same_class_idxs = np.where(same_class_mask)[0]

        # cache true rows so we can restore U after each H- trial
        true_rows_cache = U[vid_rows].copy()

        ll_plus_list  = []
        ll_minus_list = []

        for trial in range(n_trials):

            # --- H+ trial: true frames remain in U ---
            rng_plus = np.random.default_rng(
                seed * 10**6 + t_num * 10**3 + trial
            )
            Xm_plus = class_k_mix(U, labels, c, k, rng_plus)
            o_plus  = obfuscate(Xm_plus, d, sigma, rng_plus)
            ll_plus_list.append(log_likelihood(o, o_plus, sigma))

            # --- H- trial: swap target rows, fresh independent keys ---
            rng_minus       = np.random.default_rng(
                seed * 10**6 + t_num * 10**3 + trial + 500000
            )
            alt_idxs        = rng_minus.choice(
                same_class_idxs, size=num_frames, replace=True
            )
            U[vid_rows]     = U[alt_idxs]           # in-place swap
            Xm_minus        = class_k_mix(U, labels, c, k, rng_minus)
            o_minus         = obfuscate(Xm_minus, d, sigma, rng_minus)
            U[vid_rows]     = true_rows_cache        # restore immediately
            ll_minus_list.append(log_likelihood(o, o_minus, sigma))

        mean_ll_plus  = float(np.mean(ll_plus_list))
        mean_ll_minus = float(np.mean(ll_minus_list))
        h_plus_win    = mean_ll_plus > mean_ll_minus

        # delta as a normalised summary statistic for reporting
        mean_delta = mean_ll_plus - mean_ll_minus

        # score the adversary's guess
        fallback_rng = np.random.default_rng(seed * 10**6 + t_num)
        if h_plus_win:
            guessed_fidxs = true_fidxs
        else:
            guessed_fidxs = np.sort(
                fallback_rng.choice(
                    clip_len, size=num_frames, replace=False
                )
            ).astype(int)

        w_score = compute_weighted_score(guessed_fidxs, true_fidxs, clip_len)
        weighted_scores.append(w_score)
        h_plus_wins.append(h_plus_win)
        mean_deltas.append(mean_delta)

        print(
            f"  [{t_num+1:3d}/{n_targets}] "
            f"clip={vid_idx:4d}  "
            f"H+={h_plus_win}  "
            f"ll+={mean_ll_plus:.2e}  "
            f"ll-={mean_ll_minus:.2e}  "
            f"Δ={mean_delta:.2e}  "
            f"score={w_score:.1f}/{num_frames}"
        )

    weighted_scores  = np.array(weighted_scores, dtype=float)
    h_plus_wins      = np.array(h_plus_wins,     dtype=bool)
    mean_deltas      = np.array(mean_deltas,      dtype=float)
    normalized_score = weighted_scores.mean() / num_frames

    # also compute prior-only baseline score: what an adversary achieves
    # by always guessing H+ (i.e. always guessing the true linspace frames)
    # without looking at o at all — this is the prior success probability
    prior_scores = np.array([
        compute_weighted_score(
            frame_indices[clip_row_indices[vid_idx]],
            frame_indices[clip_row_indices[vid_idx]],
            clip_lengths[vid_idx],
        )
        for vid_idx in target_vids
    ], dtype=float)
    prior_normalized = prior_scores.mean() / num_frames

    print(f"\n--- Attack Results (sigma={sigma}) ---")
    print(f"Mean weighted score : {weighted_scores.mean():.3f} / {num_frames}")
    print(f"Normalized score    : {normalized_score:.4f}")
    print(f"Prior-only baseline : {prior_normalized:.4f}  "
          f"(always guess H+, ignore o)")
    print(f"H+ win rate         : {h_plus_wins.mean():.3f}")
    print(f"Mean delta (ll+-ll-): {mean_deltas.mean():.4e}")

    return {
        "sigma":              sigma,
        "weighted_scores":    weighted_scores,
        "normalized_score":   normalized_score,
        "prior_normalized":   prior_normalized,
        "h_plus_wins":        h_plus_wins,
        "mean_deltas":        mean_deltas,
        "target_vids":        target_vids,
    }


def obfuscate_fixed(
    Xm: np.ndarray,
    d: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Same as obfuscate() but does NOT apply permutation Π.
    Used in the paired H+/H- attack so that W and B are shared
    but the row ordering is consistent between X+ and X-.
    """
    m, d0 = Xm.shape
    W = rng.standard_normal((d0, d)).astype(np.float32) / np.sqrt(d)
    B = rng.standard_normal((m, d)).astype(np.float32) * np.sqrt(sigma)
    return Xm @ W + B


if __name__ == "__main__":
    SEED = 42
    NUM_FRAMES = 16
    SIZE = 112
    D = 500
    K = 5
    N_TARGETS = 100
    N_TRIALS = 50
    SPLIT = 1
    SIGMAS = [0.01, 0.03, 0.05, 0.10]  # sweep noise levels

    VIDEO_ROOT = "./data/UCF-101"
    ANNOT_ROOT = "./data/ucfTrainTestlist"
    CACHE_PATH = "./ucf101_frame_pool_gray_random.npz"

    set_seed(SEED)

    _, train_list, _ = parse_ucf101_split(ANNOT_ROOT, SPLIT)
    print(f"Training videos: {len(train_list)}")
    c = 101

    print("\nBuilding frame pool...")
    U, labels, frame_indices, clip_ids, clip_lengths = build_frame_pool(
        video_root=VIDEO_ROOT,
        train_list=train_list,
        cache_path=CACHE_PATH,
        num_frames=NUM_FRAMES,
        size=SIZE,
        seed=SEED,
    )
    print(f"U shape: {U.shape}")

    # --- run attack across sigma values ---
    all_results = []

    for i, sigma in enumerate(SIGMAS):
        print(f"\n{'='*50}")
        print(f"Running attack: sigma={sigma}")
        print(f"{'='*50}")

        results = run_attack(
            U=U,
            labels=labels,
            frame_indices=frame_indices,
            clip_ids=clip_ids,
            clip_lengths=clip_lengths,
            c=c,
            d=D,
            sigma=sigma,
            k=K,
            n_targets=N_TARGETS,
            n_trials=N_TRIALS,
            num_frames=NUM_FRAMES,
            seed=SEED + i,
        )
        all_results.append(results)

    # --- build summary table ---
    rows = []
    for r in all_results:
        rows.append({
            "sigma": r["sigma"],
            "h_plus_win_rate": float(r["h_plus_wins"].mean()),
            "mean_weighted_score": float(r["weighted_scores"].mean()),
            "normalized_score": float(r["normalized_score"]),
            "mean_delta": float(r["mean_deltas"].mean()),
            "std_delta": float(r["mean_deltas"].std()),
        })

    df = pd.DataFrame(rows)

    # random baseline: expected weighted score under uniform random guessing
    # each guessed frame hits exact match with prob n/N_total,
    # half-weight with prob proportional to window size
    N_total = len(U)
    n_frames = NUM_FRAMES
    baseline = n_frames / N_total  # approximate exact-match baseline
    df["random_baseline"] = baseline

    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(df.to_string(index=False, float_format="{:.4f}".format))

    # --- save results ---
    # 1. CSV table
    df.to_csv("attack_results.csv", index=False, float_format="%.4f")
    print("\nSaved: attack_results.csv")

    # 2. Full results as numpy arrays
    np.savez(
        "attack_results_full.npz",
        sigmas=np.array(SIGMAS),
        h_plus_win_rates=np.array([r["h_plus_wins"].mean()
                                   for r in all_results]),
        normalized_scores=np.array([r["normalized_score"]
                                    for r in all_results]),
        mean_deltas=np.array([r["mean_deltas"].mean()
                              for r in all_results]),
    )
    print("Saved: attack_results_full.npz")

    # 3. LaTeX table for the paper
    latex = df.to_latex(
        index=False,
        float_format="{:.4f}".format,
        columns=["sigma", "h_plus_win_rate",
                 "mean_weighted_score", "normalized_score"],
        header=["$\\sigma$", "H$^+$ Win Rate",
                "Mean Weighted Score", "Normalized Score"],
        caption=(
            "Weighted membership inference attack results on UCF-101 "
            "grayscale frames under varying noise levels $\\sigma$. "
            "Mean weighted score is out of 16 (one per sampled frame). "
            "Normalized score divides by 16. "
            "H$^+$ win rate is the fraction of targets where the "
            "likelihood ratio correctly identified the true frames."
        ),
        label="tab:attack_results",
        escape=False,
    )
    with open("attack_results.tex", "w") as f:
        f.write(latex)
    print("Saved: attack_results.tex")

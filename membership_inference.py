"""
membership_inference.py

Weighted membership inference attack for temporally correlated video data.
Follows Section 7.2 of Xiao et al. (2024), extended to half-integer weights
based on temporal proximity of frame indices within each video.

Pipeline:
  - U = pool of individual grayscale frames across all training videos
  - X = n-subset of frames (the T sampled frames per video, across all videos)
  - T_X(X) = Pi1 M X W + B  (Algorithm 1 of Xiao et al.)
  - For k=0, the mixing step M is skipped: T_X(X) = Pi1 X W + B
  - Adversary observes o = T_X(X) and tries to infer which T frames
    were sampled from each target video, scored with half-integer weights.

Usage (one process per k value, parallelize across k):
    python membership_inference.py --k 0
    python membership_inference.py --k 1
    python membership_inference.py --k 5

Each process iterates over all sigma values internally and writes one CSV
per (k, sigma) cell to ./results/. Use merge_results.py to combine.
"""

import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

# reuse utilities from main pipeline
from main_video import (
    set_seed,
    get_device,
    parse_ucf101_split,
    download_ucf101,        # add this
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
    """
    if rng is None:
        rng = np.random.default_rng()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
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
        replace = total_frames < num_frames
        frame_indices = np.sort(
            rng.choice(total_frames, size=num_frames, replace=replace)
        ).astype(int)
        sampled = [frames_buf[i] for i in frame_indices]
    else:
        cap.release()
        replace = total_frames < num_frames
        frame_indices = np.sort(
            rng.choice(total_frames, size=num_frames, replace=replace)
        ).astype(int)

        cap = cv2.VideoCapture(path)
        sampled = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - 1))
                ret, frame = cap.read()
            if ret:
                sampled.append(frame)
        cap.release()

        if len(sampled) == 0:
            raise RuntimeError(f"No frames read from: {path}")

        while len(sampled) < num_frames:
            sampled.append(sampled[-1])

    result = []
    for frame in sampled:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (size, size))
        result.append(gray)

    stack = np.stack(result, axis=0)
    T, H, W = stack.shape
    frames = stack.reshape(T, H * W).astype(np.float32) / 255.0

    return frames, frame_indices, total_frames


# ----------------------------
# Build frame pool U
# ----------------------------
def build_frame_pool(
    video_root: str,
    train_list: List[Tuple[str, int]],
    cache_path: str,
    num_frames: int = 16,
    size: int = 112,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    U_list, labels_list, fidx_list, cid_list, clen_list = [], [], [], [], []
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
        fidx_list.append(fidxs)
        cid_list.append(np.full(T, clip_id, dtype=int))
        clen_list.append(total)

        if (clip_id + 1) % 200 == 0:
            print(f"  Loaded {clip_id + 1}/{len(train_list)} videos")

    U = np.concatenate(U_list, axis=0).astype(np.float32)
    labels = np.concatenate(labels_list, axis=0)
    frame_indices = np.concatenate(fidx_list, axis=0)
    clip_ids = np.concatenate(cid_list, axis=0)
    clip_lengths = np.array(clen_list, dtype=int)

    np.savez(
        cache_path,
        U=U, labels=labels, frame_indices=frame_indices,
        clip_ids=clip_ids, clip_lengths=clip_lengths,
    )
    print(f"Frame pool cached to {cache_path}")
    print(f"Total frames: {U.shape[0]}, d0={U.shape[1]}")
    return U, labels, frame_indices, clip_ids, clip_lengths


# ----------------------------
# Mechanism: with or without mixing
# ----------------------------
def class_k_mix(
    X: np.ndarray,
    labels: np.ndarray,
    c: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Class-k mixing at the individual frame level.
    For each ordered pair (i, j) in [c] x [c], sample k frames from
    class i and k from class j, average them -> one mixed frame.

    Requires k >= 1. For no mixing (k=0), apply_mechanism handles that
    branch directly without calling this function.
    """
    assert k >= 1, "class_k_mix requires k >= 1; k=0 is handled in apply_mechanism"
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


def apply_mechanism(
    U: np.ndarray,
    labels: np.ndarray,
    c: int,
    k: int,
    d: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply the obfuscation T_X(X) = Pi1 [M] X W + B.

    For k >= 1: mixing is applied via class_k_mix; output shape (c^2, d).
    For k == 0: no mixing; output shape (N_total, d) -- each row is one
                original frame after permutation, projection, and noise.
    """
    if k >= 1:
        Xm = class_k_mix(U, labels, c, k, rng)
    else:
        Xm = U  # k=0: no mixing, operate on raw frame pool

    m, d0 = Xm.shape
    W = rng.standard_normal((d0, d)).astype(np.float32) / np.sqrt(d)
    B = rng.standard_normal((m, d)).astype(np.float32) * sigma
    perm = rng.permutation(m)
    return Xm[perm] @ W + B


# ----------------------------
# Weighted scoring
# ----------------------------
def compute_weighted_score(
    guessed_indices: np.ndarray,
    true_indices: np.ndarray,
    clip_length: int,
    weight_rule: str = "half",
    window_frac: float = 0.01,
) -> float:
    """
    Weighted overlap between guessed and true frame indices.

    weight_rule:
      "half"    : w=1 if exact, w=1/2 if dist <= 0.05*clip_length, else 0
      "integer" : w=1 if exact, else 0  (binary baseline)
    """
    window = window_frac * clip_length
    total = 0.0
    for g in guessed_indices:
        min_dist = np.abs(true_indices - g).min()
        if min_dist == 0:
            total += 1.0
        elif weight_rule == "half" and min_dist <= window:
            total += 0.5
    return total


def lemma3_random_baseline(
    num_frames: int,
    clip_length: int,
    N_total: int,
    weight_rule: str = "half",
    window_frac: float = 0.01,
) -> float:
    """
    Analytical random baseline per Lemma 3 / Lemma 2 (integer case),
    normalized to [0, 1].

    For half-integer weights:
      Approximate per-clip universe partition relative to the n=num_frames
      true frames: |A|=num_frames exact-match, |B| ~ 2*window*num_frames
      half-weight neighbours (capped at clip_length - num_frames).
      baseline = (|A| + |B|/2) / N_total

    For integer weights: baseline = num_frames / N_total.
    """
    if weight_rule == "integer":
        return float(num_frames) / float(N_total)

    window = window_frac * clip_length
    half_per_true = min(2.0 * window, max(0, clip_length - 1))
    m_half = min(num_frames * half_per_true, max(0, clip_length - num_frames))
    return float((num_frames + m_half / 2.0) / float(N_total))


# ----------------------------
# Log-likelihood under Gaussian noise
# ----------------------------
def log_likelihood(
    o: np.ndarray,
    o_sim: np.ndarray,
    sigma: float,
) -> float:
    """log P(o | X_sim) ~ -||o - o_sim||^2 / (2*sigma)  (constant dropped)."""
    diff = o - o_sim
    return -float(np.sum(diff ** 2)) / (2.0 * sigma**2 )


# ----------------------------
# Cached observed release o = T_X(X)
# ----------------------------
def compute_or_load_o(
    U: np.ndarray,
    labels: np.ndarray,
    c: int,
    k: int,
    d: int,
    sigma: float,
    seed: int,
    cache_dir: str,
) -> np.ndarray:
    """Cache o per (k, sigma, seed) to make plot iteration cheap."""
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
    n_targets: int = 50,
    n_trials: int = 10,
    num_frames: int = 16,
    seed: int = 0,
    cache_dir: str = "./cache",
) -> dict:
    """
    Weighted MIA per Xiao et al. 2024, Section 7.2.

    For each target video, estimate log P(o | H+) vs log P(o | H-) by
    Monte-Carlo, then guess H+ if mean ll+ > mean ll-. Score the guess
    under both integer and half-integer weighting rules, and report
    along with two reference baselines.
    """
    rng = np.random.default_rng(seed)
    n_clips = clip_lengths.shape[0]
    N_total = U.shape[0]

    target_vids = rng.choice(n_clips, size=n_targets, replace=False)

    o = compute_or_load_o(
        U, labels, c, k, d, sigma, seed=seed, cache_dir=cache_dir
    )

    clip_row_indices = [
        np.where(clip_ids == vid)[0]
        for vid in range(n_clips)
    ]

    weighted_scores_half = []
    weighted_scores_int = []
    h_plus_wins = []
    mean_deltas = []
    always_h_plus_half = []
    always_h_plus_int = []
    random_baseline_half = []
    random_baseline_int = []

    for t_num, vid_idx in enumerate(target_vids):
        vid_rows = clip_row_indices[vid_idx]
        true_fidxs = frame_indices[vid_rows]
        clip_len = clip_lengths[vid_idx]
        true_label = labels[vid_rows][0]

        same_class_mask = (labels == true_label) & (
            ~np.isin(np.arange(len(labels)), vid_rows)
        )
        same_class_idxs = np.where(same_class_mask)[0]

        true_rows_cache = U[vid_rows].copy()

        ll_plus_list = []
        ll_minus_list = []

        for trial in range(n_trials):
            # H+ trial: true frames remain in U
            rng_plus = np.random.default_rng(
                seed * 10**6 + t_num * 10**3 + trial
            )
            o_plus = apply_mechanism(U, labels, c, k, d, sigma, rng_plus)
            ll_plus_list.append(log_likelihood(o, o_plus, sigma))

            # H- trial: swap target rows, fresh independent keys
            rng_minus = np.random.default_rng(
                seed * 10**6 + t_num * 10**3 + trial + 500000
            )
            alt_idxs = rng_minus.choice(
                same_class_idxs, size=num_frames, replace=True
            )
            U[vid_rows] = U[alt_idxs]
            o_minus = apply_mechanism(U, labels, c, k, d, sigma, rng_minus)
            U[vid_rows] = true_rows_cache
            ll_minus_list.append(log_likelihood(o, o_minus, sigma))

        mean_ll_plus = float(np.mean(ll_plus_list))
        mean_ll_minus = float(np.mean(ll_minus_list))
        h_plus_win = mean_ll_plus > mean_ll_minus
        mean_delta = mean_ll_plus - mean_ll_minus

        # adversary's guess: true linspace if H+ wins, else uniform random
        fallback_rng = np.random.default_rng(seed * 10**6 + t_num)
        if h_plus_win:
            guessed_fidxs = true_fidxs
        else:
            guessed_fidxs = np.sort(
                fallback_rng.choice(
                    clip_len, size=num_frames, replace=False
                )
            ).astype(int)

        # score adversary under both weighting rules
        w_half = compute_weighted_score(
            guessed_fidxs, true_fidxs, clip_len, weight_rule="half"
        )
        w_int = compute_weighted_score(
            guessed_fidxs, true_fidxs, clip_len, weight_rule="integer"
        )
        weighted_scores_half.append(w_half)
        weighted_scores_int.append(w_int)

        # always-H+ baseline = score returned by always guessing true frames
        b_half = compute_weighted_score(
            true_fidxs, true_fidxs, clip_len, weight_rule="half"
        )
        b_int = compute_weighted_score(
            true_fidxs, true_fidxs, clip_len, weight_rule="integer"
        )
        always_h_plus_half.append(b_half)
        always_h_plus_int.append(b_int)

        # analytical Lemma-3 random baseline (per-clip)
        random_baseline_half.append(
            lemma3_random_baseline(
                num_frames, clip_len, N_total, weight_rule="half"
            ) * num_frames
        )
        random_baseline_int.append(
            lemma3_random_baseline(
                num_frames, clip_len, N_total, weight_rule="integer"
            ) * num_frames
        )

        h_plus_wins.append(h_plus_win)
        mean_deltas.append(mean_delta)

        if (t_num + 1) % 10 == 0 or t_num == n_targets - 1:
            print(
                f"  [{t_num+1:3d}/{n_targets}] H+={h_plus_win}  "
                f"score(half)={w_half:.1f}/{num_frames}  "
                f"score(int)={w_int:.1f}/{num_frames}"
            )

    weighted_scores_half = np.array(weighted_scores_half, dtype=float)
    weighted_scores_int = np.array(weighted_scores_int, dtype=float)
    h_plus_wins = np.array(h_plus_wins, dtype=bool)
    mean_deltas = np.array(mean_deltas, dtype=float)
    always_h_plus_half = np.array(always_h_plus_half, dtype=float)
    always_h_plus_int = np.array(always_h_plus_int, dtype=float)
    random_baseline_half = np.array(random_baseline_half, dtype=float)
    random_baseline_int = np.array(random_baseline_int, dtype=float)

    print(f"\n--- Attack Results (k={k}, sigma={sigma}) ---")
    print(f"H+ win rate            : {h_plus_wins.mean():.3f}")
    print(f"Adversary score (half) : {weighted_scores_half.mean()/num_frames:.4f}")
    print(f"Adversary score (int)  : {weighted_scores_int.mean()/num_frames:.4f}")
    print(f"Always-H+ (half)       : {always_h_plus_half.mean()/num_frames:.4f}")
    print(f"Always-H+ (int)        : {always_h_plus_int.mean()/num_frames:.4f}")
    print(f"Random baseline (half) : {random_baseline_half.mean()/num_frames:.4f}")
    print(f"Random baseline (int)  : {random_baseline_int.mean()/num_frames:.4f}")
    print(f"Mean delta             : {mean_deltas.mean():.4e}")

    return {
        "k": k,
        "sigma": sigma,
        "n_targets": n_targets,
        "n_trials": n_trials,
        "h_plus_win_rate": float(h_plus_wins.mean()),
        "h_plus_win_se": float(
            np.sqrt(h_plus_wins.mean() * (1 - h_plus_wins.mean()) / n_targets)
        ),
        "score_half": float(weighted_scores_half.mean() / num_frames),
        "score_int": float(weighted_scores_int.mean() / num_frames),
        "score_half_std": float(weighted_scores_half.std() / num_frames),
        "score_int_std": float(weighted_scores_int.std() / num_frames),
        "always_h_plus_half": float(always_h_plus_half.mean() / num_frames),
        "always_h_plus_int": float(always_h_plus_int.mean() / num_frames),
        "random_baseline_half": float(random_baseline_half.mean() / num_frames),
        "random_baseline_int": float(random_baseline_int.mean() / num_frames),
        "mean_delta": float(mean_deltas.mean()),
        "std_delta": float(mean_deltas.std()),
    }


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run weighted MIA for one k value across all sigmas."
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
        help="Number of MC trials per H+/H- estimate (default: 10)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="./results",
        help="Output directory for per-cell CSVs (default: ./results)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./cache",
        help="Cache directory for observed release o (default: ./cache)",
    )
    args = parser.parse_args()

    SEED = 42
    NUM_FRAMES = 16
    SIZE = 112
    D = 500
    SPLIT = 1
    SIGMAS = [0.01, 0.05, 0.10, 0.50]

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    VIDEO_ROOT = os.path.join(SCRIPT_DIR, "data", "UCF-101")
    ANNOT_ROOT = os.path.join(SCRIPT_DIR, "data", "ucfTrainTestlist")
    CACHE_PATH = os.path.join(SCRIPT_DIR, "ucf101_frame_pool_gray_random.npz")

    data_root = os.path.join(SCRIPT_DIR, "data")
    download_ucf101(data_root, VIDEO_ROOT, ANNOT_ROOT)   # no-op if already present
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

    for i, sigma in enumerate(SIGMAS):
        out_path = os.path.join(
            args.results_dir, f"attack_k{args.k}_sigma{sigma:.4f}.csv"
        )
        if os.path.exists(out_path):
            print(f"\n[skip] {out_path} already exists")
            continue

        print(f"\n{'='*60}")
        print(f"Running attack: k={args.k}, sigma={sigma}")
        print(f"{'='*60}")

        result = run_attack(
            U=U,
            labels=labels,
            frame_indices=frame_indices,
            clip_ids=clip_ids,
            clip_lengths=clip_lengths,
            c=c,
            d=D,
            sigma=sigma,
            k=args.k,
            n_targets=args.n_targets,
            n_trials=args.n_trials,
            num_frames=NUM_FRAMES,
            seed=SEED + i,
            cache_dir=args.cache_dir,
        )

        df = pd.DataFrame([result])
        df.to_csv(out_path, index=False, float_format="%.6f")
        print(f"Wrote {out_path}")

    print(f"\nAll sigmas complete for k={args.k}.")

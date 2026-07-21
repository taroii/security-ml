"""Empirical temporal autocorrelation of ResNet-18 frame embeddings on UCF-101.

This fills the measurement proposed in the theory's Section 7.6: it computes the
per-clip embedding autocorrelation

    rho_clip(tau) = (1 / (L - tau)) * sum_t  <z_t, z_{t+tau}> / (||z_t|| ||z_{t+tau}||),

averaged over clips, where z_t is the centered ResNet-18 embedding of frame t.
This is the concrete counterpart of the AR(1) correlation model (Definition 9)
and gives a real alpha to plug into the theory's AR(1) predictions
(Theorems 11/15). Consecutive frames are used (not the random sampling of the
MIA pipeline), since autocorrelation is a function of temporal lag.

Self-contained: reuses only the data utilities from main_video.py.
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from main_video import (
    set_seed, get_device, parse_ucf101_split, download_ucf101,
    filter_long_clips, _long_clips_cache_path, CLIP_LEN,
)


def load_consecutive_frames(path, L, size=224):
    """Read the first L frames of a clip as (L, 3, size, size) float in [0,1]."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    frames = []
    for _ in range(L):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)
    cap.release()
    clip = np.stack(frames, 0).astype(np.float32) / 255.0     # (L,H,W,3)
    clip = torch.from_numpy(clip).permute(0, 3, 1, 2)          # (L,3,H,W)
    return clip


@torch.no_grad()
def embed_frames(model, clip, mean, std, device, bs=64):
    """(L,3,H,W) -> (L,512) ResNet-18 embeddings."""
    feats = []
    for i in range(0, clip.shape[0], bs):
        b = ((clip[i:i+bs] - mean) / std).to(device)
        feats.append(model(b).float().cpu())
    return torch.cat(feats, 0).numpy()


def clip_autocorr(Z, max_lag):
    """Per-clip embedding autocorrelation rho(tau), tau=0..max_lag.
    Z: (L, d) embeddings. Center by clip mean, then cosine over lag."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Zc, axis=1) + 1e-12
    L = Zc.shape[0]
    out = np.full(max_lag + 1, np.nan)
    for tau in range(max_lag + 1):
        if tau >= L:
            break
        a = Zc[:L - tau]
        b = Zc[tau:]
        num = (a * b).sum(axis=1)
        den = norms[:L - tau] * norms[tau:]
        out[tau] = float(np.mean(num / den))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=200)
    ap.add_argument("--L", type=int, default=CLIP_LEN)
    ap.add_argument("--max-lag", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="./results_revision")
    ap.add_argument("--img-dir", type=str, default="./images")
    args = ap.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    VIDEO_ROOT = os.path.join(PROJECT_ROOT, "data", "UCF-101")
    ANNOT_ROOT = os.path.join(PROJECT_ROOT, "data", "ucfTrainTestlist")
    download_ucf101(os.path.join(PROJECT_ROOT, "data"), VIDEO_ROOT, ANNOT_ROOT)
    set_seed(args.seed)
    device = get_device()

    _, train_list, _ = parse_ucf101_split(ANNOT_ROOT, 1)
    cache = _long_clips_cache_path(ANNOT_ROOT, 1, args.L).replace(".json", "_train.json")
    train_list, _ = filter_long_clips(VIDEO_ROOT, train_list, args.L, cache_path=cache, desc="train")

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(train_list), size=min(args.n_clips, len(train_list)), replace=False)

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights); model.fc = nn.Identity(); model.eval().to(device)
    mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1)
    std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1)

    curves = []
    used = 0
    for i, idx in enumerate(pick):
        rel, _ = train_list[int(idx)]
        clip = load_consecutive_frames(os.path.join(VIDEO_ROOT, rel), args.L)
        if clip is None:
            continue
        Z = embed_frames(model, clip, mean, std, device)
        curves.append(clip_autocorr(Z, args.max_lag))
        used += 1
        if used % 25 == 0:
            print(f"  embedded {used} clips")
    curves = np.array(curves)
    rho = np.nanmean(curves, axis=0)
    rho_se = np.nanstd(curves, axis=0) / np.sqrt(max(1, curves.shape[0]))

    # Fit AR(1) alpha: rho(tau) ~ alpha^tau  =>  alpha = exp(mean_{tau>=1} log rho / tau)
    taus = np.arange(1, args.max_lag + 1)
    valid = (rho[1:] > 1e-3)
    if valid.sum() >= 2:
        alpha_fit = float(np.exp(np.mean(np.log(rho[1:][valid]) / taus[valid])))
    else:
        alpha_fit = float("nan")
    # lag-1 correlation is the most interpretable single number
    alpha_lag1 = float(rho[1])

    print("\n=== Empirical ResNet-18 embedding autocorrelation (UCF-101) ===")
    print(f"clips used: {used},  L={args.L},  max_lag={args.max_lag}")
    for t in [0, 1, 2, 3, 5, 10, 20, 30]:
        if t <= args.max_lag:
            print(f"  rho({t:2d}) = {rho[t]:.4f}  (SE {rho_se[t]:.4f})")
    print(f"  lag-1 correlation alpha ~= {alpha_lag1:.4f}")
    print(f"  AR(1) geometric-fit alpha ~= {alpha_fit:.4f}")

    # AR(1) aggregation factor R_bar_b from the theory (Thm 15) for b=16:
    # R_bar_b = 1 + (2/b) sum_{tau=1}^{b-1} (b - tau) rho(tau)
    for b in [8, 16]:
        s = sum((b - t) * rho[t] for t in range(1, min(b, args.max_lag + 1)))
        Rbar = 1.0 + (2.0 / b) * s
        print(f"  aggregation factor R_bar_{b} = {Rbar:.3f}  "
              f"(=> noise reduced by sqrt = {np.sqrt(Rbar):.2f}x under averaging)")

    os.makedirs(args.out_dir, exist_ok=True)
    import pandas as pd
    pd.DataFrame({"lag": np.arange(args.max_lag + 1), "rho": rho, "se": rho_se}).to_csv(
        os.path.join(args.out_dir, "embedding_autocorrelation.csv"), index=False, float_format="%.5f")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        os.makedirs(args.img_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        lags = np.arange(args.max_lag + 1)
        ax.plot(lags, rho, "o-", color="#2c3e50", ms=3, label="measured $\\rho(\\tau)$")
        ax.fill_between(lags, rho - rho_se, rho + rho_se, color="#2c3e50", alpha=0.2)
        if np.isfinite(alpha_fit):
            ax.plot(lags, alpha_fit ** lags, "--", color="#c0392b",
                    label=f"AR(1) fit $\\alpha={alpha_fit:.2f}$")
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("temporal lag $\\tau$ (frames)")
        ax.set_ylabel("embedding autocorrelation $\\rho(\\tau)$")
        ax.set_title("ResNet-18 frame-embedding autocorrelation (UCF-101)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); fig.tight_layout()
        p = os.path.join(args.img_dir, "fig_embedding_autocorr.pdf")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        fig.savefig(p.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
        print(f"\nFigure -> {p}")
    except Exception as e:
        print(f"[skip figure] {e}")


if __name__ == "__main__":
    main()

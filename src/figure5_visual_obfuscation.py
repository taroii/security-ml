import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from main_video import parse_ucf101_split

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    CACHE_TRAIN = os.path.join(PROJECT_ROOT, "ucf101_resnet18_perframe_train.pt")
    ANNOT_ROOT = os.path.join(PROJECT_ROOT, "data", "ucfTrainTestlist")
    OUT_PATH = os.path.join(PROJECT_ROOT, "images", "figure5_visual_obfuscation.pdf")

    CLASS_A_NAME = "Basketball"
    CLASS_B_NAME = "Skiing"
    K = 5
    SIGMA = 0.1
    SEED = 42
    SPLIT = 1
    C = 101


    if not os.path.exists(CACHE_TRAIN):
        print(f"Embedding cache not found: {CACHE_TRAIN}", file=sys.stderr)
        print("Run main_video.py first to build the ResNet-18 embedding cache.", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)

    obj = torch.load(CACHE_TRAIN, map_location="cpu")
    X, y = obj["X"], obj["y"]

    class_to_idx, _, _ = parse_ucf101_split(ANNOT_ROOT, SPLIT)
    name_lookup = {n.lower(): i for n, i in class_to_idx.items()}

    CLASS_A = name_lookup.get(CLASS_A_NAME.lower())
    CLASS_B = name_lookup.get(CLASS_B_NAME.lower())
    if CLASS_A is None or CLASS_B is None:
        print(f"Could not find {CLASS_A_NAME} or {CLASS_B_NAME} in classInd.txt.", file=sys.stderr)
        print("Available classes:", sorted(class_to_idx.keys()), file=sys.stderr)
        sys.exit(1)

    idx_a = torch.where(y == CLASS_A)[0]
    idx_b = torch.where(y == CLASS_B)[0]

    # Original: one class-A clip
    orig = X[idx_a[0]]

    # M: k-mix of k class-A + k class-B clips
    pick_a = idx_a[torch.randint(0, len(idx_a), (K,), generator=g)]
    pick_b = idx_b[torch.randint(0, len(idx_b), (K,), generator=g)]
    mixed = X[torch.cat([pick_a, pick_b])].mean(dim=0)

    # B: Gaussian noise
    noise = torch.randn(mixed.shape, generator=g) * SIGMA
    noised = mixed + noise

    # Labels: one-hot -> soft -> column-permuted
    perm2 = torch.randperm(C, generator=g)
    y_orig = torch.zeros(C)
    y_orig[CLASS_A] = 1.0
    y_mix = torch.zeros(C)
    y_mix[CLASS_A] = 0.5
    y_mix[CLASS_B] = 0.5
    y_perm = y_mix[perm2]


    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    vmin = min(orig.min().item(), mixed.min().item(), noised.min().item())
    vmax = max(orig.max().item(), mixed.max().item(), noised.max().item())

    heat_data = [orig, mixed, noised]
    heat_titles = [
        f"(a) Original $X_0$: {CLASS_A_NAME}",
        f"(b) After $M$: mix with {CLASS_B_NAME} ($k$={K})",
        f"(c) After $+B$: Gaussian noise ($\\sigma$={SIGMA})",
    ]

    for ax, data, title in zip(axes[0], heat_data, heat_titles):
        im = ax.imshow(
            data.numpy(), aspect="auto", cmap="viridis",
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Feature dim (0–511)")
        ax.set_ylabel("Frame (0–15)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    label_data = [y_orig, y_mix, y_perm]
    label_titles = [
        "(d) Original label $Y_0$: one-hot",
        "(e) After $M$: soft label",
        "(f) After $\\Pi_2$: columns permuted",
    ]
    bar_colors = ["#1f77b4", "#8c6bb1", "#d62728"]

    for ax, data, title, color in zip(axes[1], label_data, label_titles, bar_colors):
        ax.bar(range(C), data.numpy(), width=1.0, color=color, edgecolor="none")
        ax.set_title(title)
        ax.set_xlabel("Class index")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.5, C - 0.5)

    for ax, (cls, name) in zip(
        [axes[1, 0], axes[1, 1]],
        [(CLASS_A, CLASS_A_NAME), (CLASS_A, CLASS_A_NAME)],
    ):
        ax.annotate(
            name, xy=(cls, 1.0 if ax is axes[1, 0] else 0.5),
            xytext=(cls + 15, 0.85),
            fontsize=9, color="black",
            arrowprops=dict(arrowstyle="->", lw=0.7, color="gray"),
        )
    axes[1, 1].annotate(
        CLASS_B_NAME, xy=(CLASS_B, 0.5),
        xytext=(CLASS_B - 35, 0.85),
        fontsize=9, color="black",
        arrowprops=dict(arrowstyle="->", lw=0.7, color="gray"),
    )

    fig.suptitle(
        "Obfuscation pipeline on a single UCF-101 clip "
        "($\\Pi_1$ row permutation acts across the dataset and is not shown)",
        fontsize=12, y=1.02,
    )

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


    # ---- Pixel-space MIA visualization ----
    # Mirrors (a)/(b)/(c) above but in the MIA pipeline (112x112 grayscale
    # pixels, per-frame mixing, no per-clip constraint). Projection W is
    # omitted so the output stays pixel-shaped and visually interpretable.
    MIA_CACHE = os.path.join(SCRIPT_DIR, "ucf101_frame_pool_gray_random.npz")
    MIA_OUT_PATH = os.path.join(
        PROJECT_ROOT, "images", "figure5_visual_obfuscation_pixel.pdf"
    )
    SIZE = 112
    N_DISPLAY = 8

    if not os.path.exists(MIA_CACHE):
        print(f"Frame pool cache not found: {MIA_CACHE}", file=sys.stderr)
        print(
            "Skipping pixel-space figure. Build it via "
            "`python src/membership_inference.py --k 0` once.",
            file=sys.stderr,
        )
        sys.exit(0)

    print(f"Loading frame pool from {MIA_CACHE}...")
    pool = np.load(MIA_CACHE)
    U_pool = pool["U"].astype(np.float32)
    pool_labels = pool["labels"]
    pool_clip_ids = pool["clip_ids"]

    a_rows = np.where(pool_labels == CLASS_A)[0]
    b_rows = np.where(pool_labels == CLASS_B)[0]

    # Original row: 8 frames from a single Basketball clip.
    clip_A_id = int(pool_clip_ids[a_rows][0])
    clip_A_rows = np.where(pool_clip_ids == clip_A_id)[0][:N_DISPLAY]
    orig_px = U_pool[clip_A_rows].reshape(-1, SIZE, SIZE)

    # Mixed row: replicate class_k_mix for the (Basketball, Skiing) pair,
    # N_DISPLAY independent draws (MIA has no per-clip constraint).
    pix_rng = np.random.default_rng(SEED)
    mixed_px = np.zeros((N_DISPLAY, SIZE, SIZE), dtype=np.float32)
    for t in range(N_DISPLAY):
        idx_a = pix_rng.choice(a_rows, size=K, replace=True)
        idx_b = pix_rng.choice(b_rows, size=K, replace=True)
        mixed_px[t] = (
            U_pool[np.concatenate([idx_a, idx_b])].mean(axis=0).reshape(SIZE, SIZE)
        )

    # Noised row: +B Gaussian noise; W is intentionally skipped so output
    # stays in 112x112 pixel space for display.
    noise_px = pix_rng.standard_normal(mixed_px.shape).astype(np.float32) * SIGMA
    noised_px = mixed_px + noise_px

    vmin_px = float(min(orig_px.min(), mixed_px.min(), noised_px.min()))
    vmax_px = float(max(orig_px.max(), mixed_px.max(), noised_px.max()))

    fig2, axes2 = plt.subplots(3, N_DISPLAY, figsize=(N_DISPLAY * 1.4, 5.2))

    row_data = [orig_px, mixed_px, noised_px]
    row_labels = [
        f"(a) Original\nframes\n({CLASS_A_NAME})",
        f"(b) After $M$\nper-frame mix\nwith {CLASS_B_NAME}\n($k$={K})",
        f"(c) After $+B$\nGaussian noise\n($\\sigma$={SIGMA})",
    ]

    for r, (data, label) in enumerate(zip(row_data, row_labels)):
        for col in range(N_DISPLAY):
            ax = axes2[r, col]
            ax.imshow(data[col], cmap="gray", vmin=vmin_px, vmax=vmax_px)
            ax.set_xticks([])
            ax.set_yticks([])
        axes2[r, 0].set_ylabel(
            label, fontsize=9, rotation=0, ha="right", va="center", labelpad=6,
        )

    fig2.suptitle(
        "MIA pipeline: pixel-space obfuscation of grayscale $112{\\times}112$ "
        "frames ($W$ omitted for visualization)",
        fontsize=11,
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(MIA_OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Wrote {MIA_OUT_PATH}")

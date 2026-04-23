import os
import sys

import cv2
import numpy as np
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
    K = 1
    SIGMA = 0.1
    SEED = 42
    SPLIT = 1
    C = 101


    # --- Embedding-space figure disabled; only the pixel figure is built. ---
    # if not os.path.exists(CACHE_TRAIN):
    #     print(f"Embedding cache not found: {CACHE_TRAIN}", file=sys.stderr)
    #     print("Run main_video.py first to build the ResNet-18 embedding cache.", file=sys.stderr)
    #     sys.exit(1)
    #
    # torch.manual_seed(SEED)
    # g = torch.Generator().manual_seed(SEED)
    #
    # obj = torch.load(CACHE_TRAIN, map_location="cpu")
    # X, y = obj["X"], obj["y"]

    class_to_idx, _, _ = parse_ucf101_split(ANNOT_ROOT, SPLIT)
    name_lookup = {n.lower(): i for n, i in class_to_idx.items()}

    CLASS_A = name_lookup.get(CLASS_A_NAME.lower())
    CLASS_B = name_lookup.get(CLASS_B_NAME.lower())
    if CLASS_A is None or CLASS_B is None:
        print(f"Could not find {CLASS_A_NAME} or {CLASS_B_NAME} in classInd.txt.", file=sys.stderr)
        print("Available classes:", sorted(class_to_idx.keys()), file=sys.stderr)
        sys.exit(1)

    # idx_a = torch.where(y == CLASS_A)[0]
    # idx_b = torch.where(y == CLASS_B)[0]
    #
    # # Original: one class-A clip
    # orig = X[idx_a[0]]
    #
    # # M: k-mix of k class-A + k class-B clips
    # pick_a = idx_a[torch.randint(0, len(idx_a), (K,), generator=g)]
    # pick_b = idx_b[torch.randint(0, len(idx_b), (K,), generator=g)]
    # mixed = X[torch.cat([pick_a, pick_b])].mean(dim=0)
    #
    # # B: Gaussian noise
    # noise = torch.randn(mixed.shape, generator=g) * SIGMA
    # noised = mixed + noise
    #
    # # Labels: one-hot -> soft -> column-permuted
    # perm2 = torch.randperm(C, generator=g)
    # y_orig = torch.zeros(C)
    # y_orig[CLASS_A] = 1.0
    # y_mix = torch.zeros(C)
    # y_mix[CLASS_A] = 0.5
    # y_mix[CLASS_B] = 0.5
    # y_perm = y_mix[perm2]
    #
    #
    # plt.rcParams.update({
    #     "font.size": 10,
    #     "axes.titlesize": 11,
    #     "axes.labelsize": 10,
    # })
    #
    # fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    #
    # vmin = min(orig.min().item(), mixed.min().item(), noised.min().item())
    # vmax = max(orig.max().item(), mixed.max().item(), noised.max().item())
    #
    # heat_data = [orig, mixed, noised]
    # heat_titles = [
    #     f"(a) Original $X_0$: {CLASS_A_NAME}",
    #     f"(b) After $M$: mix with {CLASS_B_NAME} ($k$={K})",
    #     f"(c) After $+B$: Gaussian noise ($\\sigma$={SIGMA})",
    # ]
    #
    # for ax, data, title in zip(axes[0], heat_data, heat_titles):
    #     im = ax.imshow(
    #         data.numpy(), aspect="auto", cmap="viridis",
    #         vmin=vmin, vmax=vmax,
    #     )
    #     ax.set_title(title)
    #     ax.set_xlabel("Feature dim (0–511)")
    #     ax.set_ylabel("Frame (0–15)")
    #     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #
    # label_data = [y_orig, y_mix, y_perm]
    # label_titles = [
    #     "(d) Original label $Y_0$: one-hot",
    #     "(e) After $M$: soft label",
    #     "(f) After $\\Pi_2$: columns permuted",
    # ]
    # bar_colors = ["#1f77b4", "#8c6bb1", "#d62728"]
    #
    # for ax, data, title, color in zip(axes[1], label_data, label_titles, bar_colors):
    #     ax.bar(range(C), data.numpy(), width=1.0, color=color, edgecolor="none")
    #     ax.set_title(title)
    #     ax.set_xlabel("Class index")
    #     ax.set_ylabel("Probability")
    #     ax.set_ylim(0, 1.05)
    #     ax.set_xlim(-0.5, C - 0.5)
    #
    # for ax, (cls, name) in zip(
    #     [axes[1, 0], axes[1, 1]],
    #     [(CLASS_A, CLASS_A_NAME), (CLASS_A, CLASS_A_NAME)],
    # ):
    #     ax.annotate(
    #         name, xy=(cls, 1.0 if ax is axes[1, 0] else 0.5),
    #         xytext=(cls + 15, 0.85),
    #         fontsize=9, color="black",
    #         arrowprops=dict(arrowstyle="->", lw=0.7, color="gray"),
    #     )
    # axes[1, 1].annotate(
    #     CLASS_B_NAME, xy=(CLASS_B, 0.5),
    #     xytext=(CLASS_B - 35, 0.85),
    #     fontsize=9, color="black",
    #     arrowprops=dict(arrowstyle="->", lw=0.7, color="gray"),
    # )
    #
    # fig.suptitle(
    #     "Obfuscation pipeline on a single UCF-101 clip "
    #     "($\\Pi_1$ row permutation acts across the dataset and is not shown)",
    #     fontsize=12, y=1.02,
    # )
    #
    # plt.tight_layout()
    # plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    # print(f"Wrote {OUT_PATH}")


    # ---- Pixel-space obfuscation visualization ----
    # Load one color frame each from a Basketball and Skiing video at 224x224,
    # average them (k=1), and add Gaussian noise. Visualization-only: the real
    # MIA pipeline operates on 112x112 grayscale frames.
    VIDEO_ROOT = os.path.join(PROJECT_ROOT, "data", "UCF-101")
    MIA_OUT_PATH = os.path.join(
        PROJECT_ROOT, "images", "figure5_visual_obfuscation_pixel.pdf"
    )
    SIZE = 224

    def load_color_mid_frame(path: str, size: int) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame from {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        return frame.astype(np.float32) / 255.0

    _, train_list, _ = parse_ucf101_split(ANNOT_ROOT, SPLIT)
    a_paths = [rp for rp, lab in train_list if lab == CLASS_A]
    b_paths = [rp for rp, lab in train_list if lab == CLASS_B]
    if not a_paths or not b_paths:
        print(
            f"No training videos for {CLASS_A_NAME} or {CLASS_B_NAME} in "
            f"train_list. Check {ANNOT_ROOT}.",
            file=sys.stderr,
        )
        sys.exit(1)

    orig_px = load_color_mid_frame(os.path.join(VIDEO_ROOT, a_paths[0]), SIZE)
    skiing_px = load_color_mid_frame(os.path.join(VIDEO_ROOT, b_paths[0]), SIZE)

    # Mixed: reuse the Basketball frame from (a), averaged with one Skiing
    # frame. class_k_mix normally samples both sides fresh; fixing the
    # class-A contribution here keeps the (a)->(b)->(c) comparison direct.
    mixed_px = (orig_px + skiing_px) / 2.0

    pix_rng = np.random.default_rng(SEED)
    noise_px = pix_rng.standard_normal(mixed_px.shape).astype(np.float32) * SIGMA
    noised_px = mixed_px + noise_px

    fig2, axes2 = plt.subplots(1, 3, figsize=(9, 3.6))
    panels = [orig_px, mixed_px, noised_px]
    captions = [
        f"(a) Original {CLASS_A_NAME} frame",
        f"(b) After $M$: mix with {CLASS_B_NAME} ($k$={K})",
        f"(c) After $+B$: Gaussian noise ($\\sigma$={SIGMA})",
    ]
    for ax, panel, caption in zip(axes2, panels, captions):
        ax.imshow(np.clip(panel, 0.0, 1.0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(caption, fontsize=10, labelpad=8)

    plt.tight_layout()
    plt.savefig(MIA_OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Wrote {MIA_OUT_PATH}")

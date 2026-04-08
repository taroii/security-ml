import os
import math
import argparse
import random
import shutil
import ssl
import subprocess
import sys
import urllib.request
import zipfile
from typing import Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights


# ----------------------------
# Model (3-layer FC as in paper experiments)
# ----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with soft targets (rows sum to 1)."""
    logp = F.log_softmax(logits, dim=1)
    return -(soft_targets * logp).sum(dim=1).mean()


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------
# Dataset download
# ----------------------------
UCF101_VIDEO_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
UCF101_ANNOT_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"


def _download_with_progress(url: str, dest: str):
    """Download a file with a simple progress indicator."""
    print(f"Downloading {url}")
    print(f"  -> {dest}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)
        else:
            mb = downloaded / (1024 * 1024)
            print(f"\r  {mb:.1f} MB", end="", flush=True)

    # UCF server has SSL cert issues in some environments (e.g. conda)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    urllib.request.install_opener(opener)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()  # newline after progress


def _extract_rar(rar_path: str, dest_dir: str):
    """Extract a RAR archive using rarfile, 7z, or unrar."""
    # Try rarfile package first
    try:
        import rarfile
        print(f"Extracting {rar_path} with rarfile...")
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(dest_dir)
        return
    except ImportError:
        pass
    except Exception as e:
        print(f"  rarfile failed: {e}")

    # Try 7z (very common on Windows)
    for cmd_7z in ["7z", r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"]:
        if shutil.which(cmd_7z) or os.path.isfile(cmd_7z):
            print(f"Extracting {rar_path} with 7z...")
            result = subprocess.run(
                [cmd_7z, "x", rar_path, f"-o{dest_dir}", "-y"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                return
            print(f"  7z failed: {result.stderr.strip()}")

    # Try unrar
    if shutil.which("unrar"):
        print(f"Extracting {rar_path} with unrar...")
        result = subprocess.run(
            ["unrar", "x", "-o+", rar_path, dest_dir],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return
        print(f"  unrar failed: {result.stderr.strip()}")

    print("ERROR: Cannot extract RAR archive. Install one of:")
    print("  pip install rarfile   (+ unrar binary)")
    print("  7-Zip: https://www.7-zip.org/")
    print("  unrar: available via package manager")
    sys.exit(1)


def download_ucf101(data_root: str, video_root: str, annot_root: str):
    """Download and extract UCF-101 videos and annotations if not present."""
    os.makedirs(data_root, exist_ok=True)

    # Download and extract annotations (ZIP)
    if not os.path.isdir(annot_root):
        zip_path = os.path.join(data_root, "UCF101TrainTestSplits-RecognitionTask.zip")
        if not os.path.isfile(zip_path):
            _download_with_progress(UCF101_ANNOT_URL, zip_path)
        print(f"Extracting annotations to {data_root}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
        if not os.path.isdir(annot_root):
            print(f"ERROR: Expected {annot_root} after extraction but not found.")
            sys.exit(1)
        print("Annotations ready.\n")

    # Download and extract videos (RAR)
    if not os.path.isdir(video_root):
        rar_path = os.path.join(data_root, "UCF101.rar")
        if not os.path.isfile(rar_path):
            _download_with_progress(UCF101_VIDEO_URL, rar_path)
        _extract_rar(rar_path, data_root)
        if not os.path.isdir(video_root):
            print(f"ERROR: Expected {video_root} after extraction but not found.")
            sys.exit(1)
        print("Videos ready.\n")


# ----------------------------
# Video loading
# ----------------------------
def load_video_frames(path: str, num_frames: int = 16, size: int = 112) -> torch.Tensor:
    """
    Read a video file with OpenCV (sequential reads, avoids seek bugs on Windows).
    Uniformly subsample to `num_frames` frames, resize to (size, size).
    Returns a float tensor of shape (3, T, H, W) in [0, 1].
    Pads by repeating the last frame if the video has fewer than `num_frames` frames.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from video: {path}")

    # Pad by repeating last frame if too short
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # Uniformly subsample to num_frames
    total = len(frames)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = [frames[i] for i in indices]

    # Stack to (T, H, W, 3), then to (3, T, H, W) float in [0, 1]
    clip = np.stack(frames, axis=0)                  # (T, H, W, 3)
    clip = torch.from_numpy(clip).float() / 255.0    # (T, H, W, 3)
    clip = clip.permute(3, 0, 1, 2)                  # (3, T, H, W)
    return clip


# ----------------------------
# UCF-101 annotation parsing
# ----------------------------
def parse_ucf101_split(
    annot_root: str, split: int = 1
) -> Tuple[dict, List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Parse UCF-101 split files.
    Returns:
      class_to_idx: dict mapping class name -> 0-based index
      train_list: list of (relative_path, class_idx) for training
      test_list: list of (relative_path, class_idx) for testing
    """
    # Parse classInd.txt: "1 ApplyEyeMakeup" -> 0-based
    class_ind_path = os.path.join(annot_root, "classInd.txt")
    class_to_idx = {}
    with open(class_ind_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            idx_1based = int(parts[0])
            class_name = parts[1]
            class_to_idx[class_name] = idx_1based - 1  # 0-based

    # Parse trainlistXX.txt: "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi 1"
    train_path = os.path.join(annot_root, f"trainlist{split:02d}.txt")
    train_list = []
    with open(train_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            rel_path = parts[0]
            label_1based = int(parts[1])
            train_list.append((rel_path, label_1based - 1))

    # Parse testlistXX.txt: "ApplyEyeMakeup/v_ApplyEyeMakeup_g05_c02.avi" (no label)
    test_path = os.path.join(annot_root, f"testlist{split:02d}.txt")
    test_list = []
    with open(test_path, "r") as f:
        for line in f:
            rel_path = line.strip()
            if not rel_path:
                continue
            # Infer label from directory name
            class_name = rel_path.split("/")[0]
            label = class_to_idx[class_name]
            test_list.append((rel_path, label))

    return class_to_idx, train_list, test_list


# ----------------------------
# UCF-101 Dataset
# ----------------------------
class UCF101Dataset(Dataset):
    def __init__(
        self,
        video_root: str,
        samples: List[Tuple[str, int]],
        num_frames: int = 16,
        size: int = 112,
        transform=None,
    ):
        self.video_root = video_root
        self.samples = samples
        self.num_frames = num_frames
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        video_path = os.path.join(self.video_root, rel_path)
        clip = load_video_frames(video_path, self.num_frames, self.size)
        if self.transform is not None:
            clip = self.transform(clip)
        return clip, label


# ----------------------------
# Pre-embedding obfuscation: load clips, mix, add noise, permute, then embed
# ----------------------------
def load_all_clips(
    video_root: str,
    samples: List[Tuple[str, int]],
    num_frames: int = 16,
    size: int = 112,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load all video clips into memory as tensors.
    Returns:
      clips: (N, 3, T, H, W) float32 in [0, 1]
      labels: (N,) int64
    """
    clips = []
    labels = []
    total = len(samples)
    for i, (rel_path, label) in enumerate(samples):
        video_path = os.path.join(video_root, rel_path)
        clip = load_video_frames(video_path, num_frames, size)
        clips.append(clip)
        labels.append(label)
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  Loaded {i+1}/{total} clips")
    clips = torch.stack(clips, dim=0)       # (N, 3, T, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    return clips, labels


def stratified_subset_clips(
    clips: torch.Tensor, labels: torch.Tensor, n: int, c: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick n clips with n/c per class (assumes divisible)."""
    assert n % c == 0, "n must be divisible by number of classes"
    n0 = n // c
    g = torch.Generator().manual_seed(seed)
    idxs = []
    for cls in range(c):
        cls_idx = torch.where(labels == cls)[0]
        if len(cls_idx) < n0:
            raise ValueError(
                f"Class {cls} has only {len(cls_idx)} samples, need {n0}. "
                f"Reduce --n or check the dataset."
            )
        perm = cls_idx[torch.randperm(len(cls_idx), generator=g)]
        idxs.append(perm[:n0])
    idx = torch.cat(idxs, dim=0)
    return clips[idx], labels[idx]


def make_k_mixed_clips(
    clips: torch.Tensor, labels: torch.Tensor, c: int, m: int, k: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Class-k-mixing on raw video clips: for each (i,j), produce m0 mixed clips
    by averaging k random clips from class i and k random clips from class j.
    Returns:
      mixed_clips: (m, 3, T, H, W) float32
      soft_labels: (m, c) float32
    """
    assert m % (c * c) == 0, "m must be divisible by c^2"
    m0 = m // (c * c)

    g = torch.Generator().manual_seed(seed)
    cls_to_idx = [torch.where(labels == cls)[0] for cls in range(c)]

    mixed_clips = torch.empty((m, *clips.shape[1:]), dtype=clips.dtype)
    soft_labels = torch.zeros((m, c), dtype=torch.float32)

    t = 0
    for i in range(c):
        for j in range(c):
            for _ in range(m0):
                idx_i = cls_to_idx[i][torch.randint(0, len(cls_to_idx[i]), (k,), generator=g)]
                idx_j = cls_to_idx[j][torch.randint(0, len(cls_to_idx[j]), (k,), generator=g)]
                mix = torch.cat([idx_i, idx_j], dim=0)

                mixed_clips[t] = clips[mix].mean(dim=0)

                if i == j:
                    soft_labels[t, i] = 1.0
                else:
                    soft_labels[t, i] = 0.5
                    soft_labels[t, j] = 0.5
                t += 1

    soft_labels = soft_labels / (soft_labels.sum(dim=1, keepdim=True) + 1e-12)
    return mixed_clips, soft_labels


def obfuscate_clips(
    clips: torch.Tensor,
    soft_labels: torch.Tensor,
    sigma: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pre-embedding obfuscation on raw video clips:
      - Add Gaussian noise to clips
      - Permute sample order (Pi1)
      - Permute label columns (Pi2)
    Returns:
      noisy_clips: (m, 3, T, H, W) with noise and permuted order
      permuted_labels: (m, c) with permuted rows and columns
      perm2: label column permutation
      inv_perm2: inverse of perm2
    """
    m = clips.shape[0]
    g = torch.Generator().manual_seed(seed)

    # Gaussian noise on raw clips
    noise = torch.randn_like(clips) * sigma
    noisy_clips = (clips + noise).clamp(0.0, 1.0)

    # Sample permutation (Pi1)
    perm1 = torch.randperm(m, generator=g)
    noisy_clips = noisy_clips[perm1]

    # Label column permutation (Pi2)
    perm2 = torch.randperm(soft_labels.shape[1], generator=g)
    inv_perm2 = torch.empty_like(perm2)
    inv_perm2[perm2] = torch.arange(len(perm2))

    permuted_labels = soft_labels[:, perm2]
    permuted_labels = permuted_labels[perm1]

    return noisy_clips, permuted_labels, perm2, inv_perm2


@torch.no_grad()
def embed_clips(
    clips: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """
    Embed pre-obfuscated clips using r3d_18 pretrained on Kinetics-400.
    Returns (N, 512) L2-normalized embeddings.
    """
    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)

    kinetics_mean = torch.tensor(weights.transforms().mean).view(3, 1, 1, 1)
    kinetics_std = torch.tensor(weights.transforms().std).view(3, 1, 1, 1)

    feats = []
    total_batches = math.ceil(clips.shape[0] / batch_size)
    for i in range(0, clips.shape[0], batch_size):
        batch = clips[i:i+batch_size]
        batch = (batch - kinetics_mean) / kinetics_std
        batch = batch.to(device)
        f = model(batch).float()
        f = f / (f.norm(dim=1, keepdim=True) + 1e-12)
        feats.append(f.cpu())
        batch_idx = i // batch_size + 1
        if batch_idx % 10 == 0 or batch_idx == total_batches:
            print(f"  Embedding: batch {batch_idx}/{total_batches}")

    return torch.cat(feats, dim=0)


@torch.no_grad()
def embed_dataset(
    video_root: str,
    samples: List[Tuple[str, int]],
    device: torch.device,
    batch_size: int,
    num_frames: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and embed a set of video clips (no obfuscation, for test set / baseline).
    Returns (N, 512) embeddings and (N,) labels.
    """
    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)

    kinetics_mean = torch.tensor(weights.transforms().mean).view(3, 1, 1, 1)
    kinetics_std = torch.tensor(weights.transforms().std).view(3, 1, 1, 1)

    dataset = UCF101Dataset(video_root, samples, num_frames=num_frames)

    def collate_fn(batch):
        clips, labels = zip(*batch)
        clips = torch.stack(clips, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return clips, labels

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    feats = []
    labs = []
    total_batches = len(loader)
    for i, (clips, y) in enumerate(loader):
        clips = (clips - kinetics_mean) / kinetics_std
        clips = clips.to(device)
        f = model(clips).float()
        f = f / (f.norm(dim=1, keepdim=True) + 1e-12)
        feats.append(f.cpu())
        labs.append(y.cpu())
        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(f"  Embedding: batch {i+1}/{total_batches}")

    return torch.cat(feats, dim=0), torch.cat(labs, dim=0)


# ----------------------------
# Post-embedding: W projection only
# ----------------------------
def project_embeddings(
    X: torch.Tensor, d: int, seed: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random projection W from embedding dim to d.
    Returns projected embeddings and W.
    """
    d0 = X.shape[1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.randn(d0, d, generator=g, dtype=torch.float32) * (1.0 / math.sqrt(d))
    W = W.to(device)
    X = X.to(device, dtype=torch.float32)
    Xw = (X @ W).cpu()
    return Xw, W.cpu()


@torch.no_grad()
def eval_model(model, Xte, yte, W, inv_perm2, device, batch_size=512):
    """Evaluate obfuscated model (applies W projection and inverts label permutation)."""
    model.eval()
    W = W.to(device)
    correct = 0
    total = 0
    for s in range(0, Xte.shape[0], batch_size):
        x = Xte[s:s+batch_size].to(device, dtype=torch.float32)
        y = yte[s:s+batch_size].to(device)
        xw = x @ W
        logits = model(xw)
        logits = logits[:, inv_perm2]
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


@torch.no_grad()
def eval_model_baseline(model, Xte, yte, device, batch_size=512):
    """Evaluate non-obfuscated baseline model (no transforms)."""
    model.eval()
    correct = 0
    total = 0
    for s in range(0, Xte.shape[0], batch_size):
        x = Xte[s:s+batch_size].to(device, dtype=torch.float32)
        y = yte[s:s+batch_size].to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Video obfuscation pipeline (pre-embedding): UCF-101 + r3d_18"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)

    # Paper-style knobs
    ap.add_argument("--n", type=int, default=5050,
                    help="training subset size (50 per class * 101 classes)")
    ap.add_argument("--m", type=int, default=10201,
                    help="number of mixed samples (c^2 = 101^2)")
    ap.add_argument("--k", type=int, default=5, help="mix number")
    ap.add_argument("--sigma", type=float, default=0.03,
                    help="std of Gaussian noise added to video clips before embedding")

    ap.add_argument("--d", type=int, default=256,
                    help="output dim after projection")
    ap.add_argument("--embed_bs", type=int, default=16,
                    help="batch size for embedding extraction (video clips are large)")
    ap.add_argument("--train_bs", type=int, default=256)

    # Data paths
    ap.add_argument("--video_root", type=str, default="./data/UCF-101",
                    help="root directory containing UCF-101 class folders")
    ap.add_argument("--annot_root", type=str, default="./data/ucfTrainTestlist",
                    help="root directory containing UCF-101 split files")
    ap.add_argument("--split", type=int, default=1, help="UCF-101 split (1, 2, or 3)")
    args = ap.parse_args()

    # Download dataset if not present
    data_root = os.path.dirname(args.video_root)  # ./data
    download_ucf101(data_root, args.video_root, args.annot_root)

    set_seed(args.seed)
    device = get_device()
    c = 101
    print(f"device: {device}")
    print(f"UCF-101: {c} classes, split {args.split}")

    # Parse split annotations
    _, train_list, test_list = parse_ucf101_split(args.annot_root, args.split)
    print(f"UCF-101 split {args.split}: {len(train_list)} train, {len(test_list)} test")

    # 1) Embed test set (clean, no obfuscation) for evaluation
    print("\nEmbedding test clips...")
    Xte, yte = embed_dataset(
        args.video_root, test_list, device, args.embed_bs, num_frames=16,
    )
    d0 = Xte.shape[1]
    print(f"Test: {Xte.shape[0]} clips, d0={d0}")

    # 2) Load all training clips into memory
    print("\nLoading training clips...")
    train_clips, train_labels = load_all_clips(args.video_root, train_list, num_frames=16)
    print(f"Train clips loaded: {train_clips.shape}")

    # 3) Subsample n clips (balanced per class)
    sub_clips, sub_labels = stratified_subset_clips(
        train_clips, train_labels, n=args.n, c=c, seed=args.seed
    )
    print(f"Subset: {sub_clips.shape}")

    # --- Baseline: embed clean subset, train without obfuscation ---
    print("\nEmbedding clean subset for baseline...")
    Xn_clean = embed_clips(sub_clips, device, args.embed_bs)
    yn_clean = sub_labels

    print("\nTraining baseline model (no obfuscation):")
    model_baseline = MLP(input_dim=d0, num_classes=c).to(device)
    opt_baseline = optim.Adam(model_baseline.parameters(), lr=args.lr)

    Xn_dev = Xn_clean.to(device, dtype=torch.float32)
    yn_dev = yn_clean.to(device)
    n_train = Xn_clean.shape[0]

    for ep in range(args.epochs):
        model_baseline.train()
        perm = torch.randperm(n_train, device=device)
        Xn_ep = Xn_dev[perm]
        yn_ep = yn_dev[perm]

        for s in range(0, n_train, args.train_bs):
            xb = Xn_ep[s:s+args.train_bs]
            yb = yn_ep[s:s+args.train_bs]
            opt_baseline.zero_grad()
            logits = model_baseline(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt_baseline.step()

        acc_baseline = eval_model_baseline(model_baseline, Xte, yte, device=device)
        print(f"Epoch {ep}: test acc = {acc_baseline:.2f}%")

    # --- Obfuscated: mix clips, add noise, permute, then embed ---
    print("\nApplying pre-embedding obfuscation (k-mixing on raw clips)...")
    mixed_clips, soft_labels = make_k_mixed_clips(
        sub_clips, sub_labels, c=c, m=args.m, k=args.k, seed=args.seed
    )
    print(f"Mixed clips: {mixed_clips.shape}, soft labels: {soft_labels.shape}")

    print("Adding Gaussian noise, permuting samples and labels...")
    noisy_clips, permuted_labels, perm2, inv_perm2 = obfuscate_clips(
        mixed_clips, soft_labels, sigma=args.sigma, seed=args.seed
    )
    print(f"Obfuscated clips: {noisy_clips.shape}")

    # Free mixed_clips to save memory
    del mixed_clips

    print("\nEmbedding obfuscated clips...")
    Xm_emb = embed_clips(noisy_clips, device, args.embed_bs)
    del noisy_clips
    print(f"Obfuscated embeddings: {Xm_emb.shape}")

    # 4) Apply W projection (post-embedding)
    Xt, W = project_embeddings(Xm_emb, d=args.d, seed=args.seed, device=device)
    Yt = permuted_labels
    print(f"Projected: Xt={Xt.shape}, Yt={Yt.shape}")

    # 5) Train MLP on transformed data
    print("\nTraining obfuscated model:")
    model = MLP(input_dim=args.d, num_classes=c).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    Xt = Xt.to(device, dtype=torch.float32)
    Yt = Yt.to(device, dtype=torch.float32)
    W = W.to(device)
    m_total = Xt.shape[0]

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(m_total, device=device)
        Xt_ep = Xt[perm]
        Yt_ep = Yt[perm]

        for s in range(0, m_total, args.train_bs):
            xb = Xt_ep[s:s+args.train_bs]
            yb = Yt_ep[s:s+args.train_bs]
            opt.zero_grad()
            logits = model(xb)
            loss = soft_ce_loss(logits, yb)
            loss.backward()
            opt.step()

        acc_obf = eval_model(model, Xte, yte, W, inv_perm2, device=device)
        print(f"Epoch {ep}: test acc = {acc_obf:.2f}%")

    print("\nFinal results:")
    final_baseline = eval_model_baseline(model_baseline, Xte, yte, device=device)
    final_obf = eval_model(model, Xte, yte, W, inv_perm2, device=device)
    print(f"Baseline (no obfuscation): {final_baseline:.2f}%")
    print(f"Obfuscated:                {final_obf:.2f}%")
    print(f"Accuracy gap:              {final_baseline - final_obf:.2f}%")

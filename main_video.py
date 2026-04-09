"""Video obfuscation pipeline: per-frame ResNet-50 embeddings + temporal model.

Embeds each frame of a UCF-101 video independently with ResNet-50, then
classifies the frame sequence with a Transformer or LSTM. Compares a
baseline (no obfuscation) against the paper's Algorithm 1 obfuscation
(projection W, sample permutation Π₁, label permutation Π₂, frame-order
permutation Π₃, and Gaussian noise B).

Usage:
    # Transformer (default) with 250 frames per video
    python main_video.py

    # LSTM with 128 frames
    python main_video.py --model lstm --num_frames 128

    # Custom obfuscation parameters
    python main_video.py --model transformer --k 10 --sigma 0.05

    # Use a different UCF-101 split and data location
    python main_video.py --split 2 --video_root /path/to/UCF-101 --annot_root /path/to/ucfTrainTestlist
"""

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
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights


# ----------------------------
# Temporal models
# ----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  # (B, T, d)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, num_frames: int = 16,
                 nhead: int = 8, num_layers: int = 2, dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_frames + 1, input_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):  # (B, T, d)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)              # (B, T+1, d)
        x = x + self.pos_embed[:, :x.shape[1]]
        x = self.encoder(x)
        return self.fc(x[:, 0])                      # CLS token


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
# Video loading (per-frame)
# ----------------------------

def count_video_frames(path: str) -> int:
    """Return the number of frames in a video, falling back to manual counting."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0:
        cap.release()
        return frame_count

    frame_count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1
    cap.release()
    return frame_count


def analyze_frame_counts(
    video_root: str,
    train_list: List[Tuple[str, int]],
    test_list: List[Tuple[str, int]],
    split: int,
) -> Tuple[int, Dict[str, int], str]:
    """
    Compute frame-count statistics over the active split, save a histogram,
    and choose a fixed clip length as mean + 2 * std.
    """
    all_samples = train_list + test_list
    frame_counts: Dict[str, int] = {}
    counts = []

    print("Scanning frame counts to choose a fixed clip length...")
    for idx, (rel_path, _) in enumerate(all_samples, start=1):
        video_path = os.path.join(video_root, rel_path)
        n_frames = count_video_frames(video_path)
        frame_counts[rel_path] = n_frames
        counts.append(n_frames)
        if idx % 500 == 0 or idx == len(all_samples):
            print(f"  counted {idx}/{len(all_samples)} videos")

    counts_np = np.array(counts, dtype=np.float32)
    mean_frames = float(counts_np.mean())
    std_frames = float(counts_np.std())
    frame_limit = max(1, int(math.ceil(mean_frames + 2.0 * std_frames)))

    plot_path = os.path.abspath(f"./ucf101_frame_count_distribution_split{split}.png")
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 5))
        plt.hist(counts_np, bins=50, color="#1f77b4", alpha=0.85, edgecolor="black")
        plt.axvline(mean_frames, color="orange", linestyle="--", linewidth=2, label=f"mean={mean_frames:.1f}")
        plt.axvline(frame_limit, color="red", linestyle="-", linewidth=2, label=f"mean+2std={frame_limit}")
        plt.title(f"UCF-101 Split {split} Frame Count Distribution")
        plt.xlabel("Frames per video")
        plt.ylabel("Number of videos")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()
        print(f"Saved frame-count distribution plot to {plot_path}")
    except Exception as e:
        print(f"Skipping histogram plot because matplotlib is unavailable: {e}")

    print(f"Frame-count stats: mean={mean_frames:.2f}, std={std_frames:.2f}, limit={frame_limit}")
    return frame_limit, frame_counts, plot_path


def filter_samples_by_frame_limit(
    samples: List[Tuple[str, int]],
    frame_counts: Dict[str, int],
    frame_limit: int,
) -> List[Tuple[str, int]]:
    """Keep only videos whose frame count is at most the chosen fixed length."""
    return [sample for sample in samples if frame_counts[sample[0]] <= frame_limit]


def load_video_frames(path: str, num_frames: int, size: int = 224) -> torch.Tensor:
    """
    Read a video file. If shorter than `num_frames`, pad with black frames.
    Videos longer than `num_frames` should already have been filtered out,
    but we keep uniform subsampling as a safety fallback.
    Returns a float tensor of shape (num_frames, 3, H, W) in [0, 1].
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

    # Pad with black frames if shorter than num_frames
    while len(frames) < num_frames:
        frames.append(np.zeros_like(frames[0]))

    # Uniformly subsample if longer than num_frames
    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]

    clip = np.stack(frames, axis=0)                  # (T, H, W, 3)
    clip = torch.from_numpy(clip).float() / 255.0    # (T, H, W, 3)
    clip = clip.permute(0, 3, 1, 2)                  # (T, 3, H, W)
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
# UCF-101 Dataset (per-frame)
# ----------------------------
class UCF101Dataset(Dataset):
    def __init__(
        self,
        video_root: str,
        samples: List[Tuple[str, int]],
        num_frames: int = 16,
        size: int = 224,
    ):
        self.video_root = video_root
        self.samples = samples
        self.num_frames = num_frames
        self.size = size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        video_path = os.path.join(self.video_root, rel_path)
        clip = load_video_frames(video_path, self.num_frames, self.size)
        return clip, label  # clip: (T, 3, H, W)


# ----------------------------
# Raw-video obfuscation + per-frame embedding with ResNet-50
# ----------------------------
def apply_raw_video_obfuscation(
    clips: torch.Tensor,
    temporal_proj: torch.Tensor,
    sigma: float,
    perm_frames: torch.Tensor,
) -> torch.Tensor:
    """
    Obfuscate raw clips before ResNet embedding.
    Applies a shared random projection along the temporal axis so each output
    frame stays image-shaped, adds Gaussian noise, and permutes frame order
    with a fixed permutation shared across samples.
    """
    obf = torch.einsum("btchw,ts->bschw", clips, temporal_proj)
    obf = obf + torch.randn_like(obf) * sigma
    obf = obf[:, perm_frames]
    # Keep values in the range expected by the pretrained ResNet.
    return obf.clamp(0.0, 1.0)


@torch.no_grad()
def embed_ucf101_per_frame(
    video_root: str,
    samples: List[Tuple[str, int]],
    device: torch.device,
    batch_size: int,
    cache_path: str,
    num_frames: int = 250,
    sigma: float = 0.0,
    obfuscate_before_embedding: bool = False,
    obf_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Embed each frame of each UCF-101 video independently with ResNet-50.
    Returns:
      X: (N, T, 2048) float32
      y: (N,) int64
    """
    if os.path.exists(cache_path):
        obj = torch.load(cache_path, map_location="cpu")
        metadata_ok = (
            obj.get("num_frames") == num_frames
            and obj.get("sigma") == sigma
            and obj.get("obfuscate_before_embedding") == obfuscate_before_embedding
            and obj.get("obf_seed") == obf_seed
            and obj.get("sample_paths") == [rel_path for rel_path, _ in samples]
        )
        if metadata_ok:
            print("Using cached per-frame embeddings.\n")
            return obj["X"], obj["y"]
        print(f"Cache mismatch at {cache_path}; rebuilding embeddings.\n")

    # Load pretrained ResNet-50 and remove FC head
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)

    # ImageNet normalization stats
    img_mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1).to(device)
    img_std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1).to(device)

    temporal_proj = None
    perm_frames = None
    if obfuscate_before_embedding:
        if obf_seed is None:
            raise ValueError("obf_seed must be provided when obfuscate_before_embedding=True")
        g = torch.Generator(device="cpu").manual_seed(obf_seed)
        temporal_proj = torch.randn(num_frames, num_frames, generator=g, dtype=torch.float32)
        temporal_proj *= 1.0 / math.sqrt(num_frames)
        perm_frames = torch.randperm(num_frames, generator=g)
        temporal_proj = temporal_proj.to(device)
        perm_frames = perm_frames.to(device)

    def collate_fn(batch):
        clips, labels = zip(*batch)
        clips = torch.stack(clips, dim=0)       # (B, T, 3, H, W)
        labels = torch.tensor(labels, dtype=torch.long)
        return clips, labels

    dataset = UCF101Dataset(video_root, samples, num_frames=num_frames, size=224)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    feats = []
    labs = []
    total_batches = len(loader)
    for i, (clips, y) in enumerate(loader):
        B, T = clips.shape[:2]
        clips = clips.to(device, dtype=torch.float32)
        if obfuscate_before_embedding:
            clips = apply_raw_video_obfuscation(clips, temporal_proj, sigma, perm_frames)

        # Flatten videos into individual frames
        frames = clips.view(B * T, 3, 224, 224)
        frames = (frames - img_mean) / img_std
        f = model(frames)                       # (B*T, 2048)
        f = f.float().view(B, T, -1)            # (B, T, 2048)
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)  # L2 normalize per frame
        feats.append(f.cpu())
        labs.append(y.cpu())
        if (i + 1) % 100 == 0 or (i + 1) == total_batches:
            print(f"  embedding batch {i+1}/{total_batches}")

    X = torch.cat(feats, dim=0)
    y = torch.cat(labs, dim=0)

    torch.save(
        {
            "X": X,
            "y": y,
            "num_frames": num_frames,
            "sigma": sigma,
            "obfuscate_before_embedding": obfuscate_before_embedding,
            "obf_seed": obf_seed,
            "sample_paths": [rel_path for rel_path, _ in samples],
        },
        cache_path,
    )
    print(f"Per-frame embeddings cached to {cache_path}\n")
    return X, y


# ----------------------------
# Dataset-agnostic pipeline (adapted for frame sequences)
# ----------------------------
def stratified_subset(X: torch.Tensor, y: torch.Tensor, n: int, c: int, seed: int):
    """Pick n points with n/c per class (assumes divisible).
    Works for X of any shape via first-dim indexing."""
    assert n % c == 0, "n must be divisible by number of classes"
    n0 = n // c
    g = torch.Generator().manual_seed(seed)
    idxs = []
    for cls in range(c):
        cls_idx = torch.where(y == cls)[0]
        if len(cls_idx) < n0:
            raise ValueError(
                f"Class {cls} has only {len(cls_idx)} samples, need {n0}. "
                f"Reduce --n or check the dataset."
            )
        perm = cls_idx[torch.randperm(len(cls_idx), generator=g)]
        idxs.append(perm[:n0])
    idx = torch.cat(idxs, dim=0)
    return X[idx], y[idx]


def make_k_mixed_dataset(
    X: torch.Tensor, y: torch.Tensor, c: int, m: int, k: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Class-k-mixing on frame sequences.
    X: (N, T, d0) — each sample is a sequence of frame embeddings.
    Frame-by-frame averaging preserves temporal structure.
    Returns: Xm (m, T, d0), Ym (m, c)
    """
    assert m % (c * c) == 0, "m must be divisible by c^2"
    m0 = m // (c * c)

    g = torch.Generator().manual_seed(seed)
    _, T, d0 = X.shape

    cls_to_idx = [torch.where(y == cls)[0] for cls in range(c)]

    Xm = torch.empty((m, T, d0), dtype=X.dtype)
    Ym = torch.zeros((m, c), dtype=X.dtype)

    t = 0
    for i in range(c):
        for j in range(c):
            for _ in range(m0):
                idx_i = cls_to_idx[i][torch.randint(0, len(cls_to_idx[i]), (k,), generator=g)]
                idx_j = cls_to_idx[j][torch.randint(0, len(cls_to_idx[j]), (k,), generator=g)]
                mix = torch.cat([idx_i, idx_j], dim=0)

                Xm[t] = X[mix].mean(dim=0)  # (T, d0) frame-by-frame average

                if i == j:
                    Ym[t, i] = 1.0
                else:
                    Ym[t, i] = 0.5
                    Ym[t, j] = 0.5
                t += 1

    Ym = Ym / (Ym.sum(dim=1, keepdim=True) + 1e-12)
    return Xm, Ym


def permute_training_labels(
    Xm: torch.Tensor,
    Ym: torch.Tensor,
    seed: int,
    device: torch.device,
):
    """
    After raw-video obfuscation and embedding, keep the label-side privacy step:
    permute sample order (Π₁) and permute label columns (Π₂).
    """
    m = Xm.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    perm1 = torch.randperm(m, generator=g)

    perm2 = torch.randperm(Ym.shape[1], generator=g)
    inv_perm2 = torch.empty_like(perm2)
    inv_perm2[perm2] = torch.arange(len(perm2))

    # Π₃: fixed frame-order permutation (shared across all samples)
    Xm = Xm.to(device, dtype=torch.float32)
    Ym = Ym.to(device, dtype=torch.float32)
    perm1 = perm1.to(device)
    perm2 = perm2.to(device)
    inv_perm2 = inv_perm2.to(device)

    Xt = Xm[perm1]       # Π₁ on rows only; raw clips were already obfuscated

    Yp = Ym[:, perm2]    # Π₂ on columns
    Yt = Yp[perm1]       # Π₁ on rows

    return Xt, Yt, perm2, inv_perm2


@torch.no_grad()
def eval_model(model, Xte, yte, inv_perm2, device, batch_size=64):
    """Evaluate obfuscated temporal model after raw-video obfuscation."""
    model.eval()
    correct = 0
    total = 0
    for s in range(0, Xte.shape[0], batch_size):
        x = Xte[s:s+batch_size].to(device, dtype=torch.float32)  # (B, T, d0)
        y = yte[s:s+batch_size].to(device)
        logits = model(x)
        logits = logits[:, inv_perm2]  # invert Π₂
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


@torch.no_grad()
def eval_model_baseline(model, Xte, yte, device, batch_size=64):
    """Evaluate non-obfuscated baseline temporal model."""
    model.eval()
    correct = 0
    total = 0
    for s in range(0, Xte.shape[0], batch_size):
        x = Xte[s:s+batch_size].to(device, dtype=torch.float32)  # (B, T, d0)
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
        description="Video obfuscation pipeline: per-frame ResNet-50 embeddings + temporal model"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model", type=str, default="transformer",
                    choices=["lstm", "transformer"],
                    help="temporal classifier architecture")
    ap.add_argument("--num_frames", type=int, default=250,
                    help="fallback frame count if filtering is skipped; otherwise overwritten by mean+2std")

    # Paper-style knobs
    ap.add_argument("--n", type=int, default=5050,
                    help="training subset size (50 per class * 101 classes)")
    ap.add_argument("--m", type=int, default=10201,
                    help="number of mixed samples (c^2 = 101^2)")
    ap.add_argument("--k", type=int, default=5, help="mix number")
    ap.add_argument("--sigma", type=float, default=0.03)

    ap.add_argument("--d", type=int, default=None,
                    help="output dim after projection (defaults to d0, i.e. square W per Theorem 7)")
    ap.add_argument("--embed_bs", type=int, default=4,
                    help="batch size for embedding extraction (videos × frames are large)")
    ap.add_argument("--train_bs", type=int, default=64)
    ap.add_argument("--cache", type=str, default="./ucf101_perframe_resnet50_embed.pt")

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
    print(f"model: {args.model}")
    print(f"UCF-101: {c} classes, split {args.split}")

    # Parse split annotations and choose a fixed clip length from the frame-count distribution.
    _, train_list, test_list = parse_ucf101_split(args.annot_root, args.split)
    print(f"UCF-101 split {args.split}: {len(train_list)} train, {len(test_list)} test")
    frame_limit, frame_counts, plot_path = analyze_frame_counts(
        args.video_root, train_list, test_list, args.split
    )
    train_list = filter_samples_by_frame_limit(train_list, frame_counts, frame_limit)
    test_list = filter_samples_by_frame_limit(test_list, frame_counts, frame_limit)
    args.num_frames = frame_limit
    print(f"Using fixed clip length {args.num_frames} from mean + 2*std")
    print(f"Filtered split: {len(train_list)} train, {len(test_list)} test")
    print(f"Frame-count plot: {plot_path}")

    cache_base = Path(args.cache)
    clean_cache = str(cache_base)
    obf_cache = str(cache_base.with_name(f"{cache_base.stem}_rawobf{cache_base.suffix}"))

    # 1) Per-frame embeddings with ResNet-50 on clean clips
    print("Extracting clean per-frame embeddings...")
    Xtr, ytr = embed_ucf101_per_frame(
        args.video_root, train_list, device, args.embed_bs, clean_cache,
        num_frames=args.num_frames,
    )
    Xte, yte = embed_ucf101_per_frame(
        args.video_root, test_list, device, args.embed_bs,
        clean_cache.replace(cache_base.suffix, f"_test{cache_base.suffix}"),
        num_frames=args.num_frames,
    )
    d0 = Xtr.shape[2]  # 2048
    if args.d is None:
        args.d = d0  # square W per Theorem 7
    print(f"Train: {Xtr.shape[0]} videos × {Xtr.shape[1]} frames × {d0}d")
    print(f"Test:  {Xte.shape[0]} videos × {Xte.shape[1]} frames × {d0}d")

    # 2) Per-frame embeddings with raw-video obfuscation applied first
    print("Extracting obfuscated per-frame embeddings...")
    Xtr_obf, ytr_obf = embed_ucf101_per_frame(
        args.video_root, train_list, device, args.embed_bs, obf_cache,
        num_frames=args.num_frames,
        sigma=args.sigma,
        obfuscate_before_embedding=True,
        obf_seed=args.seed,
    )
    Xte_obf, yte_obf = embed_ucf101_per_frame(
        args.video_root, test_list, device, args.embed_bs,
        obf_cache.replace(cache_base.suffix, f"_test{cache_base.suffix}"),
        num_frames=args.num_frames,
        sigma=args.sigma,
        obfuscate_before_embedding=True,
        obf_seed=args.seed,
    )

    # 3) Subsample n points (balanced per class) from the obfuscated embedding set
    Xn, yn = stratified_subset(Xtr_obf, ytr_obf, n=args.n, c=c, seed=args.seed)
    print(f"subset: X={Xn.shape}, y={yn.shape}")

    # Helper to build the selected temporal model
    def make_model(input_dim):
        if args.model == "lstm":
            return LSTMClassifier(input_dim, c).to(device)
        else:
            return TransformerClassifier(
                input_dim, c, num_frames=args.num_frames,
            ).to(device)

    ### Baseline training for comparison
    print("\nTraining baseline model (no obfuscation):")

    model_baseline = make_model(args.d)
    opt_baseline = optim.Adam(model_baseline.parameters(), lr=args.lr)

    Xtr_dev = Xtr.to(device, dtype=torch.float32)
    ytr_dev = ytr.to(device)
    n_train = Xtr.shape[0]

    for ep in range(args.epochs):
        model_baseline.train()
        perm = torch.randperm(n_train, device=device)
        Xtr_ep = Xtr_dev[perm]
        ytr_ep = ytr_dev[perm]

        for s in range(0, n_train, args.train_bs):
            xb = Xtr_ep[s:s+args.train_bs]
            yb = ytr_ep[s:s+args.train_bs]
            opt_baseline.zero_grad()
            logits = model_baseline(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt_baseline.step()

        acc_baseline = eval_model_baseline(model_baseline, Xte, yte, device=device)
        print(f"Epoch {ep}: test acc = {acc_baseline:.2f}%")

    ### Obfuscated model training
    print("\nTraining obfuscated model:")

    # 4) Build mixed dataset using class-k-mixing (frame-by-frame averaging)
    Xm, Ym = make_k_mixed_dataset(Xn, yn, c=c, m=args.m, k=args.k, seed=args.seed)
    print(f"mixed: Xm={Xm.shape}, Ym={Ym.shape}")

    # 5) Keep the row/label permutations after raw-video obfuscation + embedding
    Xt, Yt, perm2, inv_perm2 = permute_training_labels(
        Xm, Ym, seed=args.seed, device=device
    )
    print(f"obfuscated: Xt={Xt.shape}, Yt={Yt.shape}")

    # 6) Train temporal model on transformed data
    model = make_model(args.d)
    opt = optim.Adam(model.parameters(), lr=args.lr)

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

        acc_obf = eval_model(model, Xte_obf, yte_obf, inv_perm2, device=device)
        print(f"Epoch {ep}: test acc = {acc_obf:.2f}%")

    print("\nFinal results:")
    final_baseline = eval_model_baseline(model_baseline, Xte, yte, device=device)
    final_obf = eval_model(model, Xte_obf, yte_obf, inv_perm2, device=device)
    print(f"Baseline (no obfuscation): {final_baseline:.2f}%")
    print(f"Obfuscated:                {final_obf:.2f}%")
    print(f"Accuracy gap:              {final_baseline - final_obf:.2f}%")

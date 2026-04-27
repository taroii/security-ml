import argparse
import json
import os
import random
import shutil
import ssl
import subprocess
import sys
import urllib.request
import zipfile
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights


# Common trimming length: clips with fewer than CLIP_LEN frames are dropped
# from the dataset; longer clips are truncated to their first CLIP_LEN frames.
# Both downstream and MIA pipelines must agree on this constant.
CLIP_LEN = 100



# Temporal classifier
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

    def forward(self, x):
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1]]
        x = self.encoder(x)
        return self.fc(x[:, 0])


def soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return -(soft_targets * logp).sum(dim=1).mean()



# Utilities
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



# Dataset download
UCF101_VIDEO_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
UCF101_ANNOT_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"


def _download_with_progress(url: str, dest: str):
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

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()


def _extract_rar(rar_path: str, dest_dir: str):
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
    os.makedirs(data_root, exist_ok=True)

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

    if not os.path.isdir(video_root):
        rar_path = os.path.join(data_root, "UCF101.rar")
        if not os.path.isfile(rar_path):
            _download_with_progress(UCF101_VIDEO_URL, rar_path)
        _extract_rar(rar_path, data_root)
        if not os.path.isdir(video_root):
            print(f"ERROR: Expected {video_root} after extraction but not found.")
            sys.exit(1)
        print("Videos ready.\n")



# Video loading
def load_video_frames(
    path: str,
    num_frames: int = 16,
    size: int = 224,
    rng: np.random.Generator = None,
    clip_len: int = CLIP_LEN,
) -> torch.Tensor:
    """
    Load a clip and sample num_frames random indices from [0, clip_len).
    Caller must guarantee the clip has at least clip_len frames (use
    filter_long_clips upstream). Frames after clip_len are ignored.
    """
    if rng is None:
        rng = np.random.default_rng()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    indices = np.sort(
        rng.choice(clip_len, size=num_frames, replace=False)
    ).astype(int)

    frames = [None] * num_frames
    next_target = 0  # position in indices we need next
    cur = 0  # current frame number being read
    while next_target < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if cur == int(indices[next_target]):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size))
            frames[next_target] = frame
            next_target += 1
        cur += 1
        if cur >= clip_len:
            break
    cap.release()

    if next_target < num_frames:
        raise RuntimeError(
            f"Only read {next_target}/{num_frames} requested frames "
            f"from {path} (clip shorter than CLIP_LEN={clip_len}?)"
        )

    clip = np.stack(frames, axis=0)
    clip = torch.from_numpy(clip).float() / 255.0
    clip = clip.permute(3, 0, 1, 2)
    return clip


# Cache file for the filtered (clips ≥ CLIP_LEN) sample list. Both
# main_video.py and membership_inference.py read this cache.
def _long_clips_cache_path(annot_root: str, split: int, clip_len: int) -> str:
    return os.path.join(
        annot_root, f"long_clips_split{split:02d}_len{clip_len}.json"
    )


def filter_long_clips(
    video_root: str,
    samples: List[Tuple[str, int]],
    clip_len: int,
    cache_path: str = None,
    desc: str = "samples",
) -> Tuple[List[Tuple[str, int]], int]:
    """
    Drop clips with total frame count < clip_len. Returns (kept, dropped).
    Reads CAP_PROP_FRAME_COUNT (metadata only, fast). Caches kept paths
    in cache_path keyed by clip_len for reuse.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
        if int(data.get("clip_len", -1)) == clip_len:
            kept_set = set(data["kept_paths"])
            kept = [(rel, lbl) for rel, lbl in samples if rel in kept_set]
            dropped = len(samples) - len(kept)
            print(
                f"[filter] {desc}: loaded cached length filter "
                f"({len(kept)} kept, {dropped} dropped, clip_len={clip_len})"
            )
            return kept, dropped

    print(
        f"[filter] {desc}: scanning {len(samples)} clips for length>={clip_len}..."
    )
    kept = []
    dropped = 0
    for n, (rel_path, label) in enumerate(samples):
        video_path = os.path.join(video_root, rel_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            dropped += 1
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total >= clip_len:
            kept.append((rel_path, label))
        else:
            dropped += 1
        if (n + 1) % 1000 == 0:
            print(f"  scanned {n + 1}/{len(samples)}")
    print(
        f"[filter] {desc}: kept {len(kept)} / dropped {dropped} (total {len(samples)})"
    )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {"clip_len": clip_len, "kept_paths": [k[0] for k in kept]}, f
            )
    return kept, dropped



# UCF-101 annotation parsing
def parse_ucf101_split(
    annot_root: str, split: int = 1
) -> Tuple[dict, List[Tuple[str, int]], List[Tuple[str, int]]]:
    class_ind_path = os.path.join(annot_root, "classInd.txt")
    class_to_idx = {}
    with open(class_ind_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            idx_1based = int(parts[0])
            class_name = parts[1]
            class_to_idx[class_name] = idx_1based - 1

    train_path = os.path.join(annot_root, f"trainlist{split:02d}.txt")
    train_list = []
    with open(train_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            rel_path = parts[0]
            label_1based = int(parts[1])
            train_list.append((rel_path, label_1based - 1))

    test_path = os.path.join(annot_root, f"testlist{split:02d}.txt")
    test_list = []
    with open(test_path, "r") as f:
        for line in f:
            rel_path = line.strip()
            if not rel_path:
                continue
            class_name = rel_path.split("/")[0]
            label = class_to_idx[class_name]
            test_list.append((rel_path, label))

    return class_to_idx, train_list, test_list



# UCF-101 Dataset
class UCF101Dataset(Dataset):
    def __init__(self, video_root, samples, num_frames=16, size=224,
                 clip_len=CLIP_LEN, sampling_seed=0):
        self.video_root = video_root
        self.samples = samples
        self.num_frames = num_frames
        self.size = size
        self.clip_len = clip_len
        self.sampling_seed = sampling_seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        video_path = os.path.join(self.video_root, rel_path)
        # deterministic per-clip rng so the embedding cache and any future
        # rebuild use the same indices for the same (sample idx, sampling_seed)
        rng = np.random.default_rng(self.sampling_seed * (1 << 20) + idx)
        clip = load_video_frames(
            video_path, self.num_frames, self.size, rng=rng,
            clip_len=self.clip_len,
        )
        return clip, label



# Per-frame embedding with ResNet-18
@torch.no_grad()
def embed_dataset(
    video_root: str,
    samples: List[Tuple[str, int]],
    device: torch.device,
    batch_size: int,
    cache_path: str,
    num_frames: int = 16,
    clip_len: int = CLIP_LEN,
    sampling_seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if os.path.exists(cache_path):
        obj = torch.load(cache_path, map_location="cpu")
        print(f"Using cached embeddings from {cache_path}\n")
        return obj["X"], obj["y"]

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)

    img_mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1).to(device)
    img_std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1).to(device)

    dataset = UCF101Dataset(
        video_root, samples, num_frames=num_frames, size=224,
        clip_len=clip_len, sampling_seed=sampling_seed,
    )

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
        B, C, T, H, W = clips.shape
        frames = clips.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        frames = frames.to(device)
        frames = (frames - img_mean) / img_std
        f = model(frames)
        f = f.float().view(B, T, -1)
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
        feats.append(f.cpu())
        labs.append(y.cpu())
        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(f"  Embedding: batch {i+1}/{total_batches}")

    X = torch.cat(feats, dim=0)
    y = torch.cat(labs, dim=0)

    torch.save({"X": X, "y": y}, cache_path)
    print(f"Embeddings cached to {cache_path}\n")
    return X, y



# Dataset-agnostic pipeline
def stratified_subset(X, y, n, c, seed):
    assert n % c == 0, "n must be divisible by number of classes"
    n0 = n // c
    g = torch.Generator().manual_seed(seed)
    idxs = []
    for cls in range(c):
        cls_idx = torch.where(y == cls)[0]
        if len(cls_idx) < n0:
            raise ValueError(
                f"Class {cls} has only {len(cls_idx)} samples, need {n0}."
            )
        perm = cls_idx[torch.randperm(len(cls_idx), generator=g)]
        idxs.append(perm[:n0])
    idx = torch.cat(idxs, dim=0)
    return X[idx], y[idx]


def make_k_mixed_dataset(X, y, c, m, k, seed):
    """k>=1 mixing on frame sequences. Output shape (m, T, d0)."""
    assert k >= 1, "make_k_mixed_dataset requires k>=1; k=0 handled in main"
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

                Xm[t] = X[mix].mean(dim=0)

                if i == j:
                    Ym[t, i] = 1.0
                else:
                    Ym[t, i] = 0.5
                    Ym[t, j] = 0.5
                t += 1

    Ym = Ym / (Ym.sum(dim=1, keepdim=True) + 1e-12)
    return Xm, Ym


def make_no_mix_dataset(X, y, c):
    """k=0: pass through raw embeddings, expand y to one-hot."""
    n = X.shape[0]
    Ym = torch.zeros((n, c), dtype=X.dtype)
    Ym[torch.arange(n), y] = 1.0
    return X.clone(), Ym


def obfuscate_training(Xm, Ym, sigma, seed, device):
    """Apply Pi1, Pi2, B (W = ResNet-18 already applied)."""
    m = Xm.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    perm1 = torch.randperm(m, generator=g)
    perm2 = torch.randperm(Ym.shape[1], generator=g)
    inv_perm2 = torch.empty_like(perm2)
    inv_perm2[perm2] = torch.arange(len(perm2))

    Xm = Xm.to(device, dtype=torch.float32)
    Ym = Ym.to(device, dtype=torch.float32)
    perm1 = perm1.to(device)
    perm2 = perm2.to(device)
    inv_perm2 = inv_perm2.to(device)

    Xt = Xm[perm1]
    B = torch.randn_like(Xt) * sigma
    Xt = Xt + B

    Yp = Ym[:, perm2]
    Yt = Yp[perm1]

    return Xt, Yt, perm2, inv_perm2


@torch.no_grad()
def eval_model(model, Xte, yte, inv_perm2, device, batch_size=64):
    model.eval()
    correct = 0
    total = 0
    for s in range(0, Xte.shape[0], batch_size):
        x = Xte[s:s+batch_size].to(device, dtype=torch.float32)
        y = yte[s:s+batch_size].to(device)
        logits = model(x)
        logits = logits[:, inv_perm2]
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


@torch.no_grad()
def eval_model_baseline(model, Xte, yte, device, batch_size=64):
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



# Training routines
def train_baseline(Xn, yn, Xte, yte, d0, T, c, device,
                   epochs, lr, train_bs):
    """Train no-obfuscation baseline. Returns final test accuracy."""
    print("\nTraining baseline model (no obfuscation):")
    model = TransformerClassifier(input_dim=d0, num_classes=c, num_frames=T).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    Xn_dev = Xn.to(device, dtype=torch.float32)
    yn_dev = yn.to(device)
    n_train = Xn.shape[0]

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        Xn_ep = Xn_dev[perm]
        yn_ep = yn_dev[perm]

        for s in range(0, n_train, train_bs):
            xb = Xn_ep[s:s+train_bs]
            yb = yn_ep[s:s+train_bs]
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            acc = eval_model_baseline(model, Xte, yte, device=device)
            print(f"  Epoch {ep+1}: test acc = {acc:.2f}%")

    return eval_model_baseline(model, Xte, yte, device=device)


def train_obfuscated(Xn, yn, Xte, yte, d0, T, c, k, sigma, m_mixed,
                     device, epochs, lr, train_bs, seed):
    """Train obfuscated model at given (k, sigma). Returns final test acc."""
    print(f"\nTraining obfuscated model (k={k}, sigma={sigma}):")

    if k >= 1:
        Xm, Ym = make_k_mixed_dataset(Xn, yn, c=c, m=m_mixed, k=k, seed=seed)
    else:
        Xm, Ym = make_no_mix_dataset(Xn, yn, c=c)
    print(f"  Pre-noise data: Xm={tuple(Xm.shape)}, Ym={tuple(Ym.shape)}")

    Xt, Yt, perm2, inv_perm2 = obfuscate_training(
        Xm, Ym, sigma=sigma, seed=seed, device=device
    )
    print(f"  Obfuscated:     Xt={tuple(Xt.shape)}, Yt={tuple(Yt.shape)}")

    model = TransformerClassifier(input_dim=d0, num_classes=c, num_frames=T).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    m_total = Xt.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(m_total, device=device)
        Xt_ep = Xt[perm]
        Yt_ep = Yt[perm]

        for s in range(0, m_total, train_bs):
            xb = Xt_ep[s:s+train_bs]
            yb = Yt_ep[s:s+train_bs]
            opt.zero_grad()
            logits = model(xb)
            loss = soft_ce_loss(logits, yb)
            loss.backward()
            opt.step()

        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            acc = eval_model(model, Xte, yte, inv_perm2, device=device)
            print(f"  Epoch {ep+1}: test acc = {acc:.2f}%")

    return eval_model(model, Xte, yte, inv_perm2, device=device)



# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train UCF-101 classifier under obfuscation; report accuracy."
    )
    parser.add_argument("--k", type=int, required=True, choices=[0, 1, 5],
                        help="Mixing parameter")
    parser.add_argument("--sigma", type=float, required=True,
                        help="Noise standard deviation")
    parser.add_argument("--results-dir", type=str, default="./accuracy_results",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n", type=int, default=5050,
                        help="Training subset size (50 per class * 101 classes)")
    parser.add_argument("--m-mixed", type=int, default=10201,
                        help="Number of mixed samples (used only for k>=1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    EMBED_BS = 16
    TRAIN_BS = 64
    SPLIT = 1
    VIDEO_ROOT = "./data/UCF-101"
    ANNOT_ROOT = "./data/ucfTrainTestlist"
    CACHE_TRAIN = "./ucf101_resnet18_perframe_train.pt"
    CACHE_TEST = "./ucf101_resnet18_perframe_test.pt"

    os.makedirs(args.results_dir, exist_ok=True)

    out_path = os.path.join(
        args.results_dir, f"acc_k{args.k}_sigma{args.sigma:.4f}.csv"
    )
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        sys.exit(0)

    data_root = os.path.dirname(VIDEO_ROOT)
    download_ucf101(data_root, VIDEO_ROOT, ANNOT_ROOT)

    set_seed(args.seed)
    device = get_device()
    c = 101
    print(f"device: {device}")
    print(f"UCF-101: {c} classes, split {SPLIT}")
    print(f"Cell: k={args.k}, sigma={args.sigma}")

    _, train_list, test_list = parse_ucf101_split(ANNOT_ROOT, SPLIT)
    print(f"UCF-101 split {SPLIT}: {len(train_list)} train, {len(test_list)} test (raw)")

    train_cache = _long_clips_cache_path(ANNOT_ROOT, SPLIT, CLIP_LEN) \
        .replace(".json", "_train.json")
    test_cache = _long_clips_cache_path(ANNOT_ROOT, SPLIT, CLIP_LEN) \
        .replace(".json", "_test.json")
    train_list, train_dropped = filter_long_clips(
        VIDEO_ROOT, train_list, CLIP_LEN, cache_path=train_cache, desc="train"
    )
    test_list, test_dropped = filter_long_clips(
        VIDEO_ROOT, test_list, CLIP_LEN, cache_path=test_cache, desc="test"
    )
    print(
        f"After CLIP_LEN={CLIP_LEN} filter: train kept {len(train_list)} "
        f"(dropped {train_dropped}), test kept {len(test_list)} "
        f"(dropped {test_dropped})"
    )

    print("\nEmbedding training clips...")
    Xtr, ytr = embed_dataset(
        VIDEO_ROOT, train_list, device, EMBED_BS, CACHE_TRAIN,
        clip_len=CLIP_LEN, sampling_seed=args.seed,
    )
    print("\nEmbedding test clips...")
    Xte, yte = embed_dataset(
        VIDEO_ROOT, test_list, device, EMBED_BS, CACHE_TEST,
        clip_len=CLIP_LEN, sampling_seed=args.seed,
    )

    d0 = Xtr.shape[2]
    T = Xtr.shape[1]
    print(f"Train: {Xtr.shape[0]} clips x {T} frames x {d0}d")
    print(f"Test:  {Xte.shape[0]} clips x {T} frames x {d0}d")

    Xn, yn = stratified_subset(Xtr, ytr, n=args.n, c=c, seed=args.seed)
    print(f"Subset: {Xn.shape}")

    # Baseline (cached, runs once across all cells)
    baseline_path = os.path.join(args.results_dir, "baseline.csv")
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        baseline_acc = float(baseline_df["accuracy"].iloc[0])
        print(f"\nLoaded cached baseline accuracy: {baseline_acc:.2f}%")
    else:
        baseline_acc = train_baseline(
            Xn, yn, Xte, yte, d0, T, c, device,
            epochs=args.epochs, lr=args.lr, train_bs=TRAIN_BS,
        )
        pd.DataFrame([{
            "accuracy": baseline_acc,
            "n": args.n,
            "epochs": args.epochs,
            "seed": args.seed,
        }]).to_csv(baseline_path, index=False, float_format="%.4f")
        print(f"Cached baseline accuracy: {baseline_acc:.2f}% -> {baseline_path}")

    # Obfuscated for this (k, sigma)
    obf_acc = train_obfuscated(
        Xn, yn, Xte, yte, d0, T, c,
        k=args.k, sigma=args.sigma, m_mixed=args.m_mixed,
        device=device, epochs=args.epochs, lr=args.lr,
        train_bs=TRAIN_BS, seed=args.seed,
    )

    print("\nFinal results:")
    print(f"  Baseline (no obfuscation): {baseline_acc:.2f}%")
    print(f"  Obfuscated (k={args.k}, sigma={args.sigma}): {obf_acc:.2f}%")
    print(f"  Accuracy gap:              {baseline_acc - obf_acc:.2f}%")

    pd.DataFrame([{
        "k": args.k,
        "sigma": args.sigma,
        "accuracy_baseline": baseline_acc,
        "accuracy_obfuscated": obf_acc,
        "accuracy_gap": baseline_acc - obf_acc,
        "n": args.n,
        "epochs": args.epochs,
        "seed": args.seed,
    }]).to_csv(out_path, index=False, float_format="%.4f")
    print(f"Wrote {out_path}")
import os
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
from torchvision.models import resnet18, ResNet18_Weights


# ----------------------------
# Temporal classifier
# ----------------------------
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

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    urllib.request.install_opener(opener)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()


def _extract_rar(rar_path: str, dest_dir: str):
    """Extract a RAR archive using rarfile, 7z, or unrar."""
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
    """Download and extract UCF-101 videos and annotations if not present."""
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


# ----------------------------
# Video loading
# ----------------------------
def load_video_frames(path: str, num_frames: int = 16, size: int = 224) -> torch.Tensor:
    """
    Read a video file with OpenCV.
    Uniformly subsample to `num_frames` frames, resize to (size, size).
    Returns a float tensor of shape (3, T, H, W) in [0, 1].
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

    total = len(frames)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = [frames[i] for i in indices]

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


# ----------------------------
# UCF-101 Dataset
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
        return clip, label


# ----------------------------
# Per-frame embedding with ResNet-18
# ----------------------------
@torch.no_grad()
def embed_dataset(
    video_root: str,
    samples: List[Tuple[str, int]],
    device: torch.device,
    batch_size: int,
    cache_path: str,
    num_frames: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Embed each frame of each UCF-101 video independently with ResNet-18.
    Returns:
      X: (N, T, 512) float32
      y: (N,) int64
    """
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

    dataset = UCF101Dataset(video_root, samples, num_frames=num_frames, size=224)

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
        # Reshape to per-frame: (B*T, 3, H, W)
        frames = clips.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        frames = frames.to(device)
        frames = (frames - img_mean) / img_std
        f = model(frames)                           # (B*T, 512)
        f = f.float().view(B, T, -1)               # (B, T, 512)
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


# ----------------------------
# Dataset-agnostic pipeline
# ----------------------------
def stratified_subset(X: torch.Tensor, y: torch.Tensor, n: int, c: int, seed: int):
    """Pick n points with n/c per class (assumes divisible)."""
    assert n % c == 0, "n must be divisible by number of classes"
    n0 = n // c
    g = torch.Generator().manual_seed(seed)
    idxs = []
    for cls in range(c):
        cls_idx = torch.where(y == cls)[0]
        if len(cls_idx) < n0:
            raise ValueError(
                f"Class {cls} has only {len(cls_idx)} samples, need {n0}. "
                f"Reduce n or check the dataset."
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
    X: (N, T, d0) -- each sample is a sequence of frame embeddings.
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


def obfuscate_training(
    Xm: torch.Tensor,
    Ym: torch.Tensor,
    sigma: float,
    seed: int,
    device: torch.device,
):
    """
    Algorithm 1 (W = embedding model, already applied):
      T_X(X) = Pi1(X) + B
      T_Y(Y) = Pi1(Y Pi2)
    M (k-mixing) is already baked into Xm, Ym.
    W (ResNet-18) is already applied (Xm contains embeddings).
    Remaining transforms: Pi1 (sample perm), Pi2 (label perm), B (noise).
    """
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
    """Evaluate obfuscated model (inverts label permutation Pi2)."""
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
    """Evaluate non-obfuscated baseline model."""
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
    SEED = 42
    EPOCHS = 50
    LR = 1e-3
    N = 5050       # training subset size (50 per class * 101 classes)
    M_MIXED = 10201  # number of mixed samples (c^2 = 101^2)
    K = 5          # mix number
    SIGMA = 0.03
    EMBED_BS = 16
    TRAIN_BS = 64
    SPLIT = 1
    VIDEO_ROOT = "./data/UCF-101"
    ANNOT_ROOT = "./data/ucfTrainTestlist"
    CACHE_TRAIN = "./ucf101_resnet18_perframe_train.pt"
    CACHE_TEST = "./ucf101_resnet18_perframe_test.pt"

    data_root = os.path.dirname(VIDEO_ROOT)
    download_ucf101(data_root, VIDEO_ROOT, ANNOT_ROOT)

    set_seed(SEED)
    device = get_device()
    c = 101
    print(f"device: {device}")
    print(f"UCF-101: {c} classes, split {SPLIT}")

    _, train_list, test_list = parse_ucf101_split(ANNOT_ROOT, SPLIT)
    print(f"UCF-101 split {SPLIT}: {len(train_list)} train, {len(test_list)} test")

    # 1) Embed all clips with ResNet-18 per-frame
    print("\nEmbedding training clips...")
    Xtr, ytr = embed_dataset(VIDEO_ROOT, train_list, device, EMBED_BS, CACHE_TRAIN)
    print("\nEmbedding test clips...")
    Xte, yte = embed_dataset(VIDEO_ROOT, test_list, device, EMBED_BS, CACHE_TEST)

    d0 = Xtr.shape[2]  # 512
    T = Xtr.shape[1]   # 16
    print(f"Train: {Xtr.shape[0]} clips x {T} frames x {d0}d")
    print(f"Test:  {Xte.shape[0]} clips x {T} frames x {d0}d")

    # 2) Stratified subset
    Xn, yn = stratified_subset(Xtr, ytr, n=N, c=c, seed=SEED)
    print(f"Subset: {Xn.shape}")

    # --- Baseline ---
    print("\nTraining baseline model (no obfuscation):")
    model_baseline = TransformerClassifier(input_dim=d0, num_classes=c, num_frames=T).to(device)
    opt_baseline = optim.Adam(model_baseline.parameters(), lr=LR)

    Xn_dev = Xn.to(device, dtype=torch.float32)
    yn_dev = yn.to(device)
    n_train = Xn.shape[0]

    for ep in range(EPOCHS):
        model_baseline.train()
        perm = torch.randperm(n_train, device=device)
        Xn_ep = Xn_dev[perm]
        yn_ep = yn_dev[perm]

        for s in range(0, n_train, TRAIN_BS):
            xb = Xn_ep[s:s+TRAIN_BS]
            yb = yn_ep[s:s+TRAIN_BS]
            opt_baseline.zero_grad()
            logits = model_baseline(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt_baseline.step()

        acc_baseline = eval_model_baseline(model_baseline, Xte, yte, device=device)
        print(f"Epoch {ep}: test acc = {acc_baseline:.2f}%")

    # --- Obfuscated ---
    print("\nTraining obfuscated model:")

    # 3) k-mix frame-sequence embeddings
    Xm, Ym = make_k_mixed_dataset(Xn, yn, c=c, m=M_MIXED, k=K, seed=SEED)
    print(f"Mixed: Xm={Xm.shape}, Ym={Ym.shape}")

    # 4) Apply Pi1, B, Pi2 (Algorithm 1; W = ResNet-18 already applied)
    Xt, Yt, perm2, inv_perm2 = obfuscate_training(
        Xm, Ym, sigma=SIGMA, seed=SEED, device=device
    )
    print(f"Obfuscated: Xt={Xt.shape}, Yt={Yt.shape}")

    # 5) Train Transformer on obfuscated data
    model = TransformerClassifier(input_dim=d0, num_classes=c, num_frames=T).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)

    m_total = Xt.shape[0]
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(m_total, device=device)
        Xt_ep = Xt[perm]
        Yt_ep = Yt[perm]

        for s in range(0, m_total, TRAIN_BS):
            xb = Xt_ep[s:s+TRAIN_BS]
            yb = Yt_ep[s:s+TRAIN_BS]
            opt.zero_grad()
            logits = model(xb)
            loss = soft_ce_loss(logits, yb)
            loss.backward()
            opt.step()

        acc_obf = eval_model(model, Xte, yte, inv_perm2, device=device)
        print(f"Epoch {ep}: test acc = {acc_obf:.2f}%")

    print("\nFinal results:")
    final_baseline = eval_model_baseline(model_baseline, Xte, yte, device=device)
    final_obf = eval_model(model, Xte, yte, inv_perm2, device=device)
    print(f"Baseline (no obfuscation): {final_baseline:.2f}%")
    print(f"Obfuscated:                {final_obf:.2f}%")
    print(f"Accuracy gap:              {final_baseline - final_obf:.2f}%")

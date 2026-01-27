import os
import math
import argparse
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights


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
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def embed_cifar10_resnet50(
    device: torch.device,
    batch_size: int,
    cache_path: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      Xtr: (50000, d0) float32
      ytr: (50000,) int64
      Xte: (10000, d0) float32
      yte: (10000,) int64
    Embedding: ResNet-50 pretrained on ImageNet; then L2-normalize each embedding.
    """
    if os.path.exists(cache_path):
        obj = torch.load(cache_path, map_location="cpu")
        return obj["Xtr"], obj["ytr"], obj["Xte"], obj["yte"]

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)

    tfm = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=0)

    def run(loader):
        feats = []
        labs = []
        for imgs, y in loader:
            imgs = imgs.to(device)
            f = model(imgs)                   # (B, d0)
            f = f.float()
            f = f / (f.norm(dim=1, keepdim=True) + 1e-12)  # L2 normalize
            feats.append(f.cpu())
            labs.append(y.cpu())
        return torch.cat(feats, dim=0), torch.cat(labs, dim=0)

    Xtr, ytr = run(trainloader)
    Xte, yte = run(testloader)

    torch.save({"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}, cache_path)
    return Xtr, ytr, Xte, yte


def stratified_subset(X: torch.Tensor, y: torch.Tensor, n: int, c: int, seed: int):
    """Pick n points with n/c per class (assumes divisible)."""
    assert n % c == 0, "n must be divisible by number of classes"
    n0 = n // c
    g = torch.Generator().manual_seed(seed)
    idxs = []
    for cls in range(c):
        cls_idx = torch.where(y == cls)[0]
        perm = cls_idx[torch.randperm(len(cls_idx), generator=g)]
        idxs.append(perm[:n0])
    idx = torch.cat(idxs, dim=0)
    return X[idx], y[idx]


def make_k_mixed_dataset(
    X: torch.Tensor, y: torch.Tensor, c: int, m: int, k: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Paper’s class-k-mixing description:
    For each (i,j), produce m0 mixed samples by averaging k random samples from class i
    and k random samples from class j. (Section 5.2)  [oai_citation:8‡People](https://people.csail.mit.edu/devadas/pubs/Learnable_Obfuscation.pdf)

    Label mixing: we use the same averaging on one-hot labels:
      - if i != j: 0.5*onehot(i) + 0.5*onehot(j)
      - if i == j: onehot(i)
    """
    assert m % (c * c) == 0, "m must be divisible by c^2"
    m0 = m // (c * c)

    g = torch.Generator().manual_seed(seed)
    d0 = X.shape[1]

    # indices per class
    cls_to_idx = [torch.where(y == cls)[0] for cls in range(c)]

    Xm = torch.empty((m, d0), dtype=X.dtype)
    Ym = torch.zeros((m, c), dtype=X.dtype)

    t = 0
    for i in range(c):
        for j in range(c):
            for _ in range(m0):
                # sample k from i and k from j (with replacement)
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

    # normalize labels to sum to 1
    Ym = Ym / (Ym.sum(dim=1, keepdim=True) + 1e-12)
    return Xm, Ym


def obfuscate_training(
    Xm: torch.Tensor,
    Ym: torch.Tensor,
    d: int,
    sigma: float,
    seed: int,
    device: torch.device,
):
    """
    Algorithm 1:
      T_X(X)= Π1 M X W + B
      T_Y(Y)= Π1 M Y Π2  [oai_citation:9‡People](https://people.csail.mit.edu/devadas/pubs/Learnable_Obfuscation.pdf)

    Here M is already “baked in” because Xm,Ym are the mixed dataset.
    So we apply:
      X_t = Π1 (Xm W) + B
      Y_t = Π1 (Ym Π2)
    """
    m, d0 = Xm.shape
    g = torch.Generator(device="cpu").manual_seed(seed)

    # W entries ~ N(0, 1/d)  [oai_citation:10‡People](https://people.csail.mit.edu/devadas/pubs/Learnable_Obfuscation.pdf)
    W = torch.randn(d0, d, generator=g, dtype=torch.float32) * (1.0 / math.sqrt(d))

    # Π1 row permutation
    perm1 = torch.randperm(m, generator=g)

    # Π2 label permutation
    perm2 = torch.randperm(Ym.shape[1], generator=g)
    inv_perm2 = torch.empty_like(perm2)
    inv_perm2[perm2] = torch.arange(len(perm2))

    Xm = Xm.to(device, dtype=torch.float32)
    Ym = Ym.to(device, dtype=torch.float32)
    W = W.to(device)

    Xw = Xm @ W                         # (m, d)
    Xw = Xw[perm1]                      # Π1
    B = torch.randn_like(Xw) * sigma    # B ~ N(0, σ^2)  [oai_citation:11‡People](https://people.csail.mit.edu/devadas/pubs/Learnable_Obfuscation.pdf)
    Xt = Xw + B

    Yp = Ym[:, perm2]                   # Π2 on columns
    Yt = Yp[perm1]                      # Π1 on rows

    return Xt, Yt, W, perm2, inv_perm2


@torch.no_grad()
def eval_model(model, Xte, yte, W, inv_perm2, device, batch_size=512):
    """Evaluate obfuscated model (applies W projection and inverts label permutation)."""
    model.eval()
    correct = 0
    total = 0
    for s in range(0, Xte.shape[0], batch_size):
        x = Xte[s:s+batch_size].to(device, dtype=torch.float32)
        y = yte[s:s+batch_size].to(device)
        xw = x @ W
        logits = model(xw)
        logits = logits[:, inv_perm2]   # invert Π2 per Algorithm 1 inference  [oai_citation:12‡People](https://people.csail.mit.edu/devadas/pubs/Learnable_Obfuscation.pdf)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)

    # Paper-style knobs (Table 1 uses n,m,k and sigma)
    ap.add_argument("--n", type=int, default=1000)     # training subset size
    ap.add_argument("--m", type=int, default=4000)     # number of mixed samples
    ap.add_argument("--k", type=int, default=5)        # mix number
    ap.add_argument("--sigma", type=float, default=0.03)

    ap.add_argument("--d", type=int, default=500)      # output dim after masking (paper uses d=500)  [oai_citation:13‡People](https://people.csail.mit.edu/devadas/pubs/Learnable_Obfuscation.pdf)
    ap.add_argument("--embed_bs", type=int, default=256)
    ap.add_argument("--train_bs", type=int, default=256)
    ap.add_argument("--cache", type=str, default="./cifar10_resnet50_embed.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    # 1) Embed + L2 normalize CIFAR10 with ResNet-50 pretrained on ImageNet  [oai_citation:14‡People](https://people.csail.mit.edu/devadas/pubs/Learnable_Obfuscation.pdf)
    Xtr, ytr, Xte, yte = embed_cifar10_resnet50(device, args.embed_bs, args.cache)
    c = 10
    d0 = Xtr.shape[1]
    print(f"embedding dim d0={d0}")

    # 2) Subsample n points (balanced per class)
    Xn, yn = stratified_subset(Xtr, ytr, n=args.n, c=c, seed=args.seed)
    print(f"subset: X={Xn.shape}, y={yn.shape}")

    # ----------------------------
    # Baseline training for comparison
    # ----------------------------
    print("\nTraining baseline model (no obfuscation):")

    model_baseline = MLP(input_dim=d0, num_classes=c).to(device)
    opt_baseline = optim.Adam(model_baseline.parameters(), lr=args.lr)

    Xn_dev = Xn.to(device, dtype=torch.float32)
    yn_dev = yn.to(device)
    n_train = Xn.shape[0]

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

    # ----------------------------
    # Obfuscated model training
    # ----------------------------
    print("\nTraining obfuscated model")

    # 3) Build mixed dataset of size m using paper's class-k-mixing  [oai_citation:15‡People](https://people.csail.mit.edu/devadas/pubs/Learnable_Obfuscation.pdf)
    Xm, Ym = make_k_mixed_dataset(Xn, yn, c=c, m=args.m, k=args.k, seed=args.seed)
    print(f"mixed: Xm={Xm.shape}, Ym={Ym.shape}")

    # 4) Obfuscate via Algorithm 1 transforms (applied to mixed data)  [oai_citation:16‡People](https://people.csail.mit.edu/devadas/pubs/Learnable_Obfuscation.pdf)
    Xt, Yt, W, perm2, inv_perm2 = obfuscate_training(
        Xm, Ym, d=args.d, sigma=args.sigma, seed=args.seed, device=device
    )
    print(f"obfuscated: Xt={Xt.shape}, Yt={Yt.shape}")

    # 5) Train 3-layer FC network on transformed data (paper: “3-layer fully-connected”)  [oai_citation:17‡People](https://people.csail.mit.edu/devadas/pubs/Learnable_Obfuscation.pdf)
    model = MLP(input_dim=args.d, num_classes=c).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # simple batching over Xt/Yt (already in memory)
    m = Xt.shape[0]
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(m, device=device)
        Xt_ep = Xt[perm]
        Yt_ep = Yt[perm]

        for s in range(0, m, args.train_bs):
            xb = Xt_ep[s:s+args.train_bs]
            yb = Yt_ep[s:s+args.train_bs]
            opt.zero_grad()
            logits = model(xb)
            loss = soft_ce_loss(logits, yb)  # soft labels from mixing
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


if __name__ == "__main__":
    # IMPORTANT on macOS: avoids the multiprocessing spawn crash you hit.
    main()
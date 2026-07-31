"""Evaluate the LOG-DETERMINANT FUNCTIONAL of the released Gram matrix on real
UCF-101 embeddings across (k, sigma). Produces Table 4 and Figure 3
(Appendix E.3).

For the isotropic-noise mechanism M(X) = X_mix W + B (B ~ N(0,sigma^2 I_d)) we
compute
    L(sigma) = log det(I_n + sigma^{-2} X_mix X_mix^T)
             = log det(I_{d0} + sigma^{-2} X_mix^T X_mix)     (Sylvester)
via the fast d0 x d0 (512x512) determinant, and report (d/2) * L(sigma)/n so the
numbers are comparable across k (which changes n).

IMPORTANT -- what this is and is not.  L(sigma) shares the
log det(I + Sigma_X / sigma^2) structure that appears in the mutual-information
analysis (Corollary S10 in Appendix D), but it is NOT that bound and NOT a
certified upper bound on MI(X; M(X)):

  * the Corollary S10 Type-(II) term is a Jensen-gap expectation,
    log det E_X[.] - E_X log det[.], whereas L is a single-realization
    log-determinant;
  * no claim is made here that (d/2) * L numerically bounds MI(X; M(X)).

It is used only as a computable PROXY whose qualitative behaviour in (k, sigma)
is what the calibration assumes.  Variable and column names retain the
`mi_bound` prefix for backward compatibility with the shipped CSV; read them as
"log-determinant functional".  See Appendix E.3 of the supplement, which states
these caveats in full, including that L decays like ~1/sigma rather than
1/sigma^2 in the operable band.

What the numbers establish is modest but real: the log-determinant diagnostic is
(i) finite/non-vacuous on real video embeddings, (ii) monotonically decreasing in
sigma, and (iii) reduced by class-k mixing.

Embeddings are 512-d clip-level R3D-18 features, a feature space distinct from
the frame-level ResNet-18 encoder used by the attacks and the downstream
classifier, so this is an independent probe of the mechanism.

Usage:
    python src/mi_bound_evaluation.py                     # recompute from embeddings
    python src/mi_bound_evaluation.py --replot-only       # redraw the figure from
                                                          # the shipped CSV
"""
import argparse, os
import numpy as np, torch


def stratified_subset(X, y, n0, c, seed):
    g = np.random.default_rng(seed)
    idx = []
    for cls in range(c):
        ci = np.where(y == cls)[0]
        if len(ci) >= n0:
            idx.append(g.choice(ci, size=n0, replace=False))
        elif len(ci) > 0:
            idx.append(g.choice(ci, size=n0, replace=True))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


def class_k_mix(X, y, c, k, seed):
    """Average k samples from class i and k from class j for each ordered pair."""
    g = np.random.default_rng(seed)
    cls_idx = [np.where(y == cls)[0] for cls in range(c)]
    cls_idx = [ci for ci in cls_idx if len(ci) > 0]
    cc = len(cls_idx)
    out = np.empty((cc * cc, X.shape[1]), dtype=np.float64)
    t = 0
    for i in range(cc):
        for j in range(cc):
            ii = cls_idx[i][g.integers(0, len(cls_idx[i]), size=k)]
            jj = cls_idx[j][g.integers(0, len(cls_idx[j]), size=k)]
            out[t] = X[np.concatenate([ii, jj])].mean(axis=0)
            t += 1
    return out


def logdet_bound(Xmix, sigma):
    """L(sigma)/n = (1/n) log det(I_d0 + sigma^-2 Xmix^T Xmix)."""
    d0 = Xmix.shape[1]
    G = Xmix.T @ Xmix / (sigma * sigma)             # (d0, d0)
    sign, ld = np.linalg.slogdet(np.eye(d0) + G)
    return ld / Xmix.shape[0]


def plot_curves(curves, img_dir):
    """Draw Figure 3 (Appendix E.3) from {k: (sigmas, values)}.

    Axis labels deliberately say "log-determinant functional", not "MI bound":
    the quantity plotted is a proxy, not a certified mutual-information bound
    (see the module docstring and Appendix E.3).
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(img_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    cols = ["#c0392b", "#2c3e50", "#27ae60"]
    for (k, (sig, v)), col in zip(sorted(curves.items()), cols):
        ax.loglog(sig, v, "o-", color=col, label=f"$k={k}$ mixing")
    ax.set_xlabel(r"noise std. dev. $\sigma$")
    ax.set_ylabel(r"log-det functional per datapoint (nats)")
    ax.set_title("Log-determinant functional on real UCF-101 embeddings")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(img_dir, "fig_mi_bound_real.pdf")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"\nFigure -> {p}")
    return p


def curves_from_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return {int(k): (g["sigma"].tolist(), g["mi_bound_per_n"].tolist())
            for k, g in df.sort_values("sigma").groupby("k")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot-only", action="store_true",
                    help="Skip recomputation; redraw the figure from the "
                         "shipped mi_bound_real.csv (no dataset needed).")
    ap.add_argument("--embed", default="ucf101_r3d18_embed.pt")
    ap.add_argument("--n0", type=int, default=50)
    ap.add_argument("--d", type=int, default=500, help="projection/output dim (MI prefactor d/2)")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 5])
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.01, 0.05, 0.10, 0.25, 0.50, 1.0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="./results_revision")
    ap.add_argument("--img-dir", default="./images")
    args = ap.parse_args()

    if args.replot_only:
        csv_path = os.path.join(args.out_dir, "mi_bound_real.csv")
        plot_curves(curves_from_csv(csv_path), args.img_dir)
        return

    obj = torch.load(args.embed, map_location="cpu", weights_only=False)
    X = obj["Xtr"].numpy().astype(np.float64)
    y = obj["ytr"].numpy().astype(int)
    c = int(y.max()) + 1
    # L2-normalize (matches the pipeline) so trace(XX^T)=n and the bound is scale-free
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Xn, yn = stratified_subset(X, y, args.n0, c, args.seed)
    print(f"subset: {Xn.shape}, classes={c}")

    rows = []
    print(f"\n{'k':>3} {'sigma':>7} {'logdet/n (nats)':>18} {'logdet (nats)':>16}")
    curves = {}
    for k in args.ks:
        Xmix = Xn if k == 0 else class_k_mix(Xn, yn, c, k, args.seed)
        vals = []
        for s in args.sigmas:
            Ln = logdet_bound(Xmix, s)              # per-datapoint core quantity
            mi_per_n = 0.5 * args.d * Ln             # (d/2) * L / n
            mi_total = mi_per_n * Xmix.shape[0]
            vals.append(mi_per_n)
            rows.append(dict(k=k, sigma=s, mi_bound_per_n=mi_per_n, mi_bound_total=mi_total,
                             n_rows=Xmix.shape[0]))
            print(f"{k:>3} {s:>7.2f} {mi_per_n:>18.3f} {mi_total:>16.1f}")
        curves[k] = (args.sigmas, vals)

    os.makedirs(args.out_dir, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "mi_bound_real.csv"),
                              index=False, float_format="%.5f")

    # 1/sigma^2 check at the two largest sigmas
    for k in args.ks:
        sig, v = curves[k]
        if len(sig) >= 2 and v[-2] > 0:
            ratio = v[-1] / v[-2]
            exp = (sig[-2] / sig[-1]) ** 2
            print(f"[k={k}] MI bound ratio at sigma {sig[-2]}->{sig[-1]}: "
                  f"{ratio:.3f}  (1/sigma^2 predicts {exp:.3f})")

    try:
        plot_curves(curves, args.img_dir)
    except Exception as e:
        print(f"[skip figure] {e}")


if __name__ == "__main__":
    main()

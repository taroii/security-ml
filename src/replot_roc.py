"""Re-render the LiRA ROC figure from saved curves (no recompute).

Panel A: the three adversaries at (k=0, sigma=0.10) -- the hardness gradient
         (informed within-clip near chance; uninformed identification confident).
Panel B: the uninformed adversary (cross-class) at sigma=0.10 across mixing k.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "./results_revision"
IMG = "./images"
cur = np.load(os.path.join(OUT, "lira_roc_curves.npz"))
df = pd.read_csv(os.path.join(OUT, "lira_roc_summary.csv"))


def auc_of(tag, k, s):
    r = df[(df.attack == tag) & (df.k == k) & (df.sigma == s)]
    return float(r["auc_pooled"].iloc[0]) if len(r) else float("nan")


def curve(tag, k, s):
    fk, tk = f"{tag}_k{k}_s{s}_fpr", f"{tag}_k{k}_s{s}_tpr"
    if fk not in cur:
        return None
    f = np.clip(cur[fk], 1e-5, 1.0)
    t = np.clip(cur[tk], 1e-5, 1.0)
    return f, t


fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.2))

# Panel A: three adversaries at (k=0, sigma=0.10)
sA = 0.1
specs = [
    ("A3_crossclass", "Uninformed (vs. unrelated)", "#c0392b"),
    ("A3_sameclass",  "Uninformed (vs. same class)", "#e08e0b"),
    ("A1_within_clip", "Informed (vs. own neighbors)", "#2c3e50"),
]
for tag, lab, col in specs:
    c = curve(tag, 0, sA)
    if c is None:
        continue
    axA.loglog(c[0], c[1], "-", color=col, lw=1.8,
               label=f"{lab}, AUC={auc_of(tag,0,sA):.2f}")
axA.plot([1e-5, 1], [1e-5, 1], "k--", lw=0.8, alpha=0.6, label="chance")
axA.set_xlim(1e-4, 1); axA.set_ylim(1e-3, 1)
axA.set_xlabel("False positive rate"); axA.set_ylabel("True positive rate")
axA.set_title("Three adversaries at $(k{=}0,\\ \\sigma{=}0.10)$")
axA.legend(fontsize=7.5, loc="lower right"); axA.grid(True, which="both", alpha=0.25)

# Panel B: uninformed (cross-class) vs mixing at sigma=0.10
for k, col in zip([0, 1, 5], ["#c0392b", "#2c3e50", "#27ae60"]):
    c = curve("A3_crossclass", k, sA)
    if c is None:
        continue
    axB.loglog(c[0], c[1], "-", color=col, lw=1.8,
               label=f"$k={k}$, AUC={auc_of('A3_crossclass',k,sA):.2f}")
axB.plot([1e-5, 1], [1e-5, 1], "k--", lw=0.8, alpha=0.6, label="chance")
axB.set_xlim(1e-4, 1); axB.set_ylim(1e-3, 1)
axB.set_xlabel("False positive rate"); axB.set_ylabel("True positive rate")
axB.set_title("Uninformed adversary vs. mixing $(\\sigma{=}0.10)$")
axB.legend(fontsize=7.5, loc="lower right"); axB.grid(True, which="both", alpha=0.25)

fig.tight_layout()
p = os.path.join(IMG, "fig_lira_roc.pdf")
fig.savefig(p, dpi=200, bbox_inches="tight")
fig.savefig(p.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
print(f"wrote {p}")

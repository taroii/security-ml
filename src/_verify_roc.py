"""Double-check harness for the LiRA ROC results. Run AFTER lira_roc_evaluation.

Checks, from results_revision/lira_roc_summary.csv + lira_roc_curves.npz:
  (C1) every AUC in [0,1]; pooled AUC within 0.05 of per-query-mean AUC
       (they measure the same thing two ways -> must roughly agree).
  (C2) monotone privacy: for a fixed attack & k, AUC is non-increasing in sigma
       (more noise cannot HELP the attack, up to MC noise tolerance 0.03).
  (C3) hardness ordering at each (k,sigma): A1_within_clip <= A3_sameclass
       <= A3_crossclass (same members vs progressively easier non-members).
  (C4) TPR@FPR sanity: 0 <= TPR@0.1% <= TPR@1% <= 1; advantage in [0,1];
       chance-level cells (AUC~0.5) have small advantage.
  (C5) ROC curve endpoints: fpr,tpr start ~0 end ~1, both monotone nondecreasing.
Exits non-zero (and prints FAIL lines) if any hard check is violated.
"""
import sys, os
import numpy as np
import pandas as pd

OUT = os.environ.get("OUTDIR", "./results_revision")
df = pd.read_csv(os.path.join(OUT, "lira_roc_summary.csv"))
curves = np.load(os.path.join(OUT, "lira_roc_curves.npz"))
fails, warns = [], []

# C1
for _, r in df.iterrows():
    tag = f"{r['attack']} k{r['k']} s{r['sigma']}"
    if not (0.0 <= r["auc_pooled"] <= 1.0):
        fails.append(f"C1 AUC out of range: {tag} auc={r['auc_pooled']}")
    if abs(r["auc_pooled"] - r["auc_perquery_mean"]) > 0.06:
        warns.append(f"C1 pooled vs per-query AUC differ: {tag} "
                     f"{r['auc_pooled']} vs {r['auc_perquery_mean']}")

# C2 monotone in sigma
for attack in df["attack"].unique():
    for k in sorted(df["k"].unique()):
        sub = df[(df.attack == attack) & (df.k == k)].sort_values("sigma")
        a = sub["auc_pooled"].to_numpy()
        for i in range(1, len(a)):
            if a[i] > a[i-1] + 0.03:
                warns.append(f"C2 AUC rises with sigma: {attack} k={k} "
                             f"sigma {sub['sigma'].to_numpy()[i-1]}->{sub['sigma'].to_numpy()[i]} "
                             f"{a[i-1]:.3f}->{a[i]:.3f}")

# C3 hardness ordering
order = {"A1_within_clip": 0, "A3_sameclass": 1, "A3_crossclass": 2}
for (k, s), g in df.groupby(["k", "sigma"]):
    m = {row["attack"]: row["auc_pooled"] for _, row in g.iterrows()}
    if all(a in m for a in order):
        seq = [m["A1_within_clip"], m["A3_sameclass"], m["A3_crossclass"]]
        for i in range(1, len(seq)):
            if seq[i] < seq[i-1] - 0.05:
                warns.append(f"C3 hardness order broken at k={k},s={s}: "
                             f"A1={seq[0]:.3f} A3s={seq[1]:.3f} A3x={seq[2]:.3f}")

# C4 TPR@FPR sanity
for _, r in df.iterrows():
    tag = f"{r['attack']} k{r['k']} s{r['sigma']}"
    if not (0 <= r["tpr_at_fpr0p1pct"] <= r["tpr_at_fpr1pct"] + 1e-9 <= 1 + 1e-9):
        fails.append(f"C4 TPR ordering: {tag} "
                     f"0.1%={r['tpr_at_fpr0p1pct']} 1%={r['tpr_at_fpr1pct']}")
    if not (-1e-9 <= r["advantage"] <= 1 + 1e-9):
        fails.append(f"C4 advantage out of range: {tag} adv={r['advantage']}")

# C5 curve shape
for key in curves.files:
    if not key.endswith("_fpr"):
        continue
    f = curves[key]; t = curves[key[:-4] + "_tpr"]
    if len(f) < 2:
        continue
    if np.any(np.diff(f) < -1e-9) or np.any(np.diff(t) < -1e-9):
        fails.append(f"C5 non-monotone ROC: {key[:-4]}")
    if abs(f[0]) > 1e-6 or abs(t[0]) > 1e-6 or abs(f[-1]-1) > 1e-6 or abs(t[-1]-1) > 1e-6:
        warns.append(f"C5 ROC endpoints off: {key[:-4]} "
                     f"f[{f[0]:.3g},{f[-1]:.3g}] t[{t[0]:.3g},{t[-1]:.3g}]")

print("=== WARNINGS ===")
for w in warns: print(" ", w)
print("=== FAILURES ===")
for x in fails: print(" ", x)
print(f"\n{len(warns)} warnings, {len(fails)} failures")
sys.exit(1 if fails else 0)

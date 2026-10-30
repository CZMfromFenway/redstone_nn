#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Filter gain with a NONLINEAR reward, since the linear one failed (AUC 0.543 vs 0.690).

The four knobs do carry the information, but not in a way a hyperplane can cut. This uses a
small quadratic model, which is the cheapest thing that works and is still easy to imagine in
hardware: 4 inputs, 10 pairwise products, 10 squares, one dot product.

Reported per keep fraction, all from the cross-validated scores so no face is scored by a
model that saw it.
"""
import json
import math
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
OUT = os.path.join(ROOT, "results", "style5")
os.makedirs(OUT, exist_ok=True)
STYLE = {4: "类二次元", 3: "类游戏", 2: "类三次元", 1: "中", 0: "差"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
new = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
X = np.array([r["pca"][:4] for r in new], dtype=np.float64)
g = np.array([r["grade"] for r in new])
y = (g >= 2).astype(float)


def quad(Z):
    n, d = Z.shape
    out = [Z]
    for i in range(d):
        out.append((Z[:, i] ** 2)[:, None])
        for j in range(i + 1, d):
            out.append((Z[:, i] * Z[:, j])[:, None])
    return np.hstack(out)


Q = quad(X)
idx = np.random.default_rng(0).permutation(len(X))
pred = np.zeros(len(X))
FOLDS, LAM = 5, 30.0
for f in range(FOLDS):
    te = idx[f::FOLDS]
    tr = np.setdiff1d(idx, te)
    zm = Q[tr].mean(0); zsd = Q[tr].std(0) + 1e-9
    A = np.c_[(Q[tr] - zm) / zsd, np.ones(len(tr))]
    Ae = np.c_[(Q[te] - zm) / zsd, np.ones(len(te))]
    w = np.linalg.solve(A.T @ A + LAM * np.eye(A.shape[1]), A.T @ y[tr])
    pred[te] = Ae @ w
AUC = auc(pred, y)
print("=" * 100)
print(f"QUADRATIC REWARD -- 153 style labels, cv AUC {AUC:.3f}")
print("=" * 100)
print(f"  base rate 可接受 = {100*y.mean():.1f}%")
print()
print(f"  {'keep':>6s} {'n':>5s} {'可接受':>8s} {'95% CI':>17s} {'提升':>8s}")
print("  " + "-" * 50)
out = {}
for keep in (1.0, 0.7, 0.5, 0.35, 0.25):
    k = max(1, int(len(X) * keep))
    top = np.argsort(-pred)[:k]
    good = int(y[top].sum())
    lo, hi = wilson(good, k)
    out[keep] = {"n": k, "good": good, "rate": good / k, "lo": lo, "hi": hi}
    print(f"  {keep*100:5.0f}% {k:5d} {good/k*100:7.1f}% [{lo*100:5.1f}%,{hi*100:5.1f}%] "
          f"{100*(good/k-y.mean()):+7.1f}")

print()
print("  WHAT THE TOP BUCKET IS MADE OF (keep 35%)")
k = max(1, int(len(X) * 0.35))
top = np.argsort(-pred)[:k]
c = {}
for gg in (4, 3, 2, 1, 0):
    c[gg] = int((g[top] == gg).sum())
for gg in (4, 3, 2, 1, 0):
    print(f"    {STYLE[gg]:10s} {c[gg]:3d}  ({100*c[gg]/k:4.1f}%)")

json.dump({"cv_auc": AUC, "filter": {str(k): v for k, v in out.items()},
           "top35_composition": {STYLE[k]: v for k, v in c.items()}},
          open(os.path.join(OUT, "quad_filter.json"), "w"), indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'quad_filter.json')}")
print("QUAD_FILTER_DONE")

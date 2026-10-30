#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Refit the reward on the STYLE labels, and check whether the old reward transfers.

WHY THIS MATTERS
The old reward separates 类二次元 (+1.80) from everything else (~1.29-1.41) but does NOT
separate 中/差 (+1.41/+1.40) from the acceptable styles (+1.29/+1.30). In other words it was
mostly a style detector, and filtering with it would have picked anime faces without
avoiding the buckets the user actually wants to avoid.

Two questions here:
  1. out of sample, how well does the OLD reward rank faces on the new criterion
     (acceptable = 类二次元/类游戏/类三次元, avoid = 中/差)?
  2. how well does a reward fitted on the style labels do, under cross-validation?
If (1) is near 0.5 the old model is useless for the real objective and every earlier
purification number has to be read as "purification for a different target".
"""
import collections
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

recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
new = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
old = [r for r in recs if r.get("model") == "dim4x" and str(r.get("round") or "") in ("2", "3", "4")
       and r.get("grade") in (0, 1, 2, 3)]


def fit(rows, target_fn, lam=10.0):
    X = np.array([r["pca"][:4] for r in rows], dtype=np.float64)
    y = np.array([target_fn(r) for r in rows], dtype=np.float64)
    zm = X.mean(0); zsd = X.std(0) + 1e-9
    Xs = (X - zm) / zsd
    A = np.c_[Xs, np.ones(len(Xs))]
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return {"w4": w[:4] / zsd, "b": float(w[4] - (w[:4] / zsd * zm).sum()),
            "zm": zm, "zsd": zsd}


def score(model, rows):
    Z = np.array([r["pca"][:4] for r in rows], dtype=np.float64)
    return Z @ model["w4"] + model["b"]


def auc(s, y):
    hi = s[y == 1][:, None]; lo = s[y == 0][None, :]
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


# ---------------------------------------------------------------- old model, new task
old_model = fit(old, lambda r: 1.0 if r["grade"] >= 2 else 0.0)
y_new = np.array([1 if r["grade"] >= 2 else 0 for r in new])
s_old = score(old_model, new)
print("=" * 100)
print("DOES THE OLD REWARD SOLVE THE NEW TASK?")
print("=" * 100)
print(f"  new task: acceptable (类二次元/类游戏/类三次元) vs avoid (中/差)")
print(f"  base rate acceptable = {100*y_new.mean():.1f}%")
print(f"  OLD reward's AUC on this task: {auc(s_old, y_new):.3f}")
print("  (0.5 = no better than a coin flip)")
if auc(s_old, y_new) < 0.65:
    print("  -> the old reward is close to useless for the real objective.")

# ---------------------------------------------------------------- new model, CV
print()
print("=" * 100)
print("NEW REWARD FITTED ON THE STYLE LABELS (5-fold cross-validated)")
print("=" * 100)
idx = np.random.default_rng(0).permutation(len(new))
X = np.array([r["pca"][:4] for r in new], dtype=np.float64)
y = np.array([r["grade"] for r in new], dtype=np.float64)
yb = (y >= 2).astype(float)
best = None
for lam in (0.3, 1, 3, 10, 30):
    pred = np.zeros(len(new))
    for f in range(5):
        te = idx[f::5]
        tr = np.setdiff1d(idx, te)
        Xtr, Xte = X[tr], X[te]
        zm = Xtr.mean(0); zsd = Xtr.std(0) + 1e-9
        A = np.c_[(Xtr - zm) / zsd, np.ones(len(tr))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yb[tr])
        pred[te] = np.c_[(Xte - zm) / zsd, np.ones(len(te))] @ w
    a = auc(pred, yb)
    print(f"    lam={lam:<4} cv AUC {a:.3f}")
    if best is None or a > best[0]:
        best = (a, lam, pred)
AUC, LAM, pred = best
print(f"  chosen lam={LAM}  cv AUC = {AUC:.3f}")

m = fit(new, lambda r: 1.0 if r["grade"] >= 2 else 0.0, LAM)
print()
print("  WEIGHTS (raw knob units; positive = the user likes more of it)")
for i, nm in enumerate(["PC0 明度", "PC1 发肤", "PC2 眼型", "PC3 脸型"]):
    print(f"    {nm:10s} {m['w4'][i]:+8.3f}")

# per-category mean predicted score, using the CV predictions (honest)
print()
print(f"  {'category':10s} {'n':>4s} {'cv score':>9s}")
print("  " + "-" * 28)
for g in (4, 3, 2, 1, 0):
    msk = y == g
    if msk.sum():
        print(f"  {STYLE[g]:10s} {int(msk.sum()):4d} {pred[msk].mean():+9.3f}")

# ---------------------------------------------------------------- filter gain
print()
print("=" * 100)
print("FILTER GAIN WITH THE NEW REWARD (cross-validated)")
print("=" * 100)
print(f"  {'keep':>6s} {'n':>5s} {'可接受':>8s} {'提升':>8s}")
print("  " + "-" * 32)
base = yb.mean()
out = {}
for keep in (1.0, 0.7, 0.5, 0.35, 0.25):
    k = max(1, int(len(new) * keep))
    top = np.argsort(-pred)[:k]
    rate = yb[top].mean()
    out[keep] = {"n": k, "rate": float(rate)}
    print(f"  {keep*100:5.0f}% {k:5d} {100*rate:7.1f}% {100*(rate-base):+7.1f}")

json.dump({"old_auc_new_task": auc(s_old, y_new), "new_cv_auc": AUC, "lam": LAM,
           "weights": m["w4"].tolist(), "base": float(base),
           "filter": {str(k): v for k, v in out.items()}},
          open(os.path.join(OUT, "new_reward.json"), "w"), indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'new_reward.json')}")
print("NEW_REWARD_DONE")

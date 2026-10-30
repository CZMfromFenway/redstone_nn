#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
How many more labels are worth collecting?

A learning curve on the 153 style labels: subsample n, run 5-fold CV, repeat with different
subsets, and watch the AUC. Then fit

    AUC(n) = a - b / sqrt(n)

which is the usual shape when the limit is estimation noise rather than model capacity, and
extrapolate. `a` is the ceiling the labels themselves allow -- if the user's own judgements
are inconsistent, no amount of data gets past it.

Two targets are tracked separately because they behave very differently:
    acceptable vs 中/差   the weak link, AUC ~0.69
    类二次元 vs rest      already ~0.92, close to solved
More labels are only worth it for the first one.
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
OUT = os.path.join(ROOT, "results", "deploy")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(6)

recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
lab = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
X = np.array([r["pca"][:4] for r in lab], dtype=np.float32)
g = np.array([r["grade"] for r in lab])
N = len(X)


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    if hi.size == 0 or lo.size == 0:
        return float("nan")
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


def cv_mlp(Xs, y, seed, steps=400, folds=5):
    n = len(Xs)
    idx = np.random.default_rng(seed).permutation(n)
    pred = np.zeros(n)
    ok = True
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            ok = False
            break
        torch.manual_seed(seed * 100 + f)
        net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-3)
        t = torch.tensor(Xs[tr]); v = torch.tensor(y[tr], dtype=torch.float32)[:, None]
        for _ in range(steps):
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(net(t), v).backward()
            opt.step()
        with torch.no_grad():
            pred[te] = net(torch.tensor(Xs[te])).numpy().ravel()
    return (auc(pred, y), True) if ok else (float("nan"), False)


TARGETS = {"可接受 vs 中/差": (g >= 2).astype(float),
           "类二次元 vs 其他": (g == 4).astype(float)}
SIZES = [60, 80, 100, 120, 153]
REPEATS = 8

print("=" * 96)
print(f"LEARNING CURVE on {N} labels (5-fold CV, 4-16-1 network, {REPEATS} subsets per size)")
print("=" * 96)
curve = {}
for tname, y in TARGETS.items():
    print()
    print(f"  target: {tname}")
    print(f"    {'n':>5s} {'cv AUC':>8s} {'sd':>7s}  (repeats)")
    print("    " + "-" * 34)
    curve[tname] = []
    for n in SIZES:
        vals = []
        for rep in range(REPEATS):
            rng = np.random.default_rng(rep * 31 + n)
            sub = rng.choice(N, n, replace=False)
            mu = X[sub].mean(0); sd = X[sub].std(0) + 1e-9
            a, ok = cv_mlp(((X[sub] - mu) / sd).astype(np.float32), y[sub], seed=rep + 1)
            if ok and not np.isnan(a):
                vals.append(a)
        if vals:
            curve[tname].append((n, float(np.mean(vals)), float(np.std(vals))))
            print(f"    {n:5d} {np.mean(vals):8.3f} {np.std(vals):7.3f}  ({len(vals)})")

# ---------------------------------------------------------------- extrapolate
print()
print("=" * 96)
print("EXTRAPOLATION  AUC(n) = a - b/sqrt(n)")
print("=" * 96)
pred = {}
for tname, pts in curve.items():
    if len(pts) < 3:
        continue
    ns = np.array([p[0] for p in pts], dtype=float)
    as_ = np.array([p[1] for p in pts], dtype=float)
    # least squares on 1/sqrt(n)
    A = np.c_[np.ones(len(ns)), -1.0 / np.sqrt(ns)]
    coef = np.linalg.lstsq(A, as_, rcond=None)[0]
    a_, b_ = coef
    print(f"  {tname}")
    print(f"    fitted ceiling a = {a_:.3f},  b = {b_:.2f}")
    line = []
    for n in (300, 500, 800, 1000, 1500):
        v = a_ - b_ / math.sqrt(n) if False else a_ - b_ / np.sqrt(n)
        line.append(f"{n}:{v:.3f}")
    print("    predicted AUC  " + "   ".join(line))
    pred[tname] = {"a": float(a_), "b": float(b_),
                   "at": {n: float(a_ - b_ / np.sqrt(n)) for n in (300, 500, 800, 1000, 1500)}}

print()
print("  the ceiling a is the number that matters: it is what the LABELS allow, not what the")
print("  network could do with infinite data. If your own judgement is inconsistent, a is")
print("  where it stops, and more labels stop helping well before then.")
print()
print("  NOTE: this extrapolation is from 153 labels, so treat it as a direction, not a promise.")

json.dump({"curve": curve, "predicted": pred},
          open(os.path.join(OUT, "learning_curve.json"), "w"), indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'learning_curve.json')}")
print("LEARNING_CURVE_DONE")

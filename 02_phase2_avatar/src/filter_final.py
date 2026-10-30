#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Filtering only: which scorer, and does targeting a style help?

THE DESIGN
A pool of random seeds is generated, each is scored, the low scorers are dropped. Two things
are varied:

    SCORER   linear | quadratic | mlp-1 (4-16-1) | mlp-2 (4-16-16-1)
    TARGET   global  = score P(acceptable)
             anime   = score P(类二次元)

The metric in BOTH modes is the ACCEPTABLE rate of the survivors, because the user's criterion
is that the style that comes out does not matter as long as the yield improves.

HOW IT STAYS HONEST
Every scorer is fitted 5-fold: for each fold the scorer is trained on the other four folds and
scores the fifth. A face is never ranked by a model that saw its label. The same folds are
used for every scorer so the comparison is paired.

Filtering cannot manufacture a degenerate face -- it only reorders what the generator already
produced -- which is why this route does not have the collapse problem the 4->4 transform had.
"""
import json
import math
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
OUT = os.path.join(ROOT, "results", "filter_final")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(4)
KNOB = ["PC0 明度", "PC1 发肤", "PC2 眼型", "PC3 脸型"]


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
    if hi.size == 0 or lo.size == 0:
        return float("nan")
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
new = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
X = np.array([r["pca"][:4] for r in new], dtype=np.float64)
g = np.array([r["grade"] for r in new])
y_acc = (g >= 2).astype(float)
y_anime = (g == 4).astype(float)
N = len(X)
print(f"pool: {N} labelled faces   acceptable {100*y_acc.mean():.1f}%   "
      f"类二次元 {100*y_anime.mean():.1f}%")

mu = X.mean(0); sd = X.std(0) + 1e-9
Xs = (X - mu) / sd
FOLDS = 5
idx = np.random.default_rng(0).permutation(N)
folds = [idx[k::FOLDS] for k in range(FOLDS)]


def quad_np(P):
    d = P.shape[1]
    F = [P]
    for i in range(d):
        F.append((P[:, i] ** 2)[:, None])
        for j in range(i + 1, d):
            F.append((P[:, i] * P[:, j])[:, None])
    return np.hstack(F)


FQ = quad_np(Xs)
FQz = (FQ - FQ.mean(0)) / (FQ.std(0) + 1e-12)


def cv_ridge(Zf, y, lam):
    pred = np.zeros(N)
    for te in folds:
        tr = np.setdiff1d(np.arange(N), te)
        A = np.c_[Zf[tr], np.ones(len(tr))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.c_[Zf[te], np.ones(len(te))] @ w
    return pred


def cv_mlp(y, hidden, steps=500, seed=0):
    pred = np.zeros(N)
    for fi, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(N), te)
        torch.manual_seed(seed * 100 + fi)
        layers = []
        prev = 4
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        net = nn.Sequential(*layers)
        opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-3)
        t = torch.tensor(Xs[tr], dtype=torch.float32)
        v = torch.tensor(y[tr], dtype=torch.float32)[:, None]
        for _ in range(steps):
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(net(t), v).backward()
            opt.step()
        with torch.no_grad():
            pred[te] = net(torch.tensor(Xs[te], dtype=torch.float32)).numpy().ravel()
    return pred


SCORERS = [
    ("linear", lambda y: cv_ridge(Xs, y, 3.0)),
    ("quadratic", lambda y: cv_ridge(FQz, y, 30.0)),
    ("mlp 4-16-1", lambda y: cv_mlp(y, [16], seed=1)),
    ("mlp 4-16-16-1", lambda y: cv_mlp(y, [16, 16], seed=2)),
]
MODES = [("global", y_acc, "不指定风格"), ("anime", y_anime, "指定类二次元")]

cache = {}
print()
print("=" * 100)
print("SCORER QUALITY (5-fold cross-validated AUC)")
print("=" * 100)
print(f"  {'scorer':16s} {'对「可接受」':>12s} {'对「类二次元」':>14s}")
print("  " + "-" * 46)
for name, fn in SCORERS:
    if name not in cache:
        cache[name] = {"acc": fn(y_acc), "anime": fn(y_anime)}
    a1 = auc(cache[name]["acc"], y_acc)
    a2 = auc(cache[name]["anime"], y_anime)
    print(f"  {name:16s} {a1:12.3f} {a2:14.3f}")

print()
print("=" * 100)
print("YIELD AFTER FILTERING -- the metric is always the ACCEPTABLE rate")
print("=" * 100)
print(f"  baseline (no filter): {100*y_acc.mean():.1f}%")
print()
final = {}
for mode, y_target, mode_cn in MODES:
    print(f"  【{mode_cn}】scorer trained to predict "
          + ("可接受" if mode == "global" else "类二次元"))
    print(f"    {'scorer':16s} " + " ".join(f"keep {int(k*100):>3d}%" for k in
                                           (0.50, 0.35, 0.25, 0.15)))
    print("    " + "-" * 62)
    for name, fn in SCORERS:
        sc = cache[name]["acc" if mode == "global" else "anime"]
        line = []
        final[(mode, name)] = {}
        for keep in (0.50, 0.35, 0.25, 0.15):
            k = max(1, int(N * keep))
            sel = np.argsort(-sc)[:k]
            r = float(y_acc[sel].mean())
            final[(mode, name)][keep] = r
            line.append(f"{100*r:8.1f}%")
        print(f"    {name:16s} " + " ".join(line))
    print()

# ---------------------------------------------------------------- at a fixed budget
K = 38
print("=" * 100)
print(f"AT EQUAL COST: keep {K} of {N} (about {N/K:.1f} generated per usable face)")
print("=" * 100)
print(f"  {'mode':12s} {'scorer':16s} {'可接受':>8s} {'95% CI':>17s} {'类二次元':>9s}")
print("  " + "-" * 66)
best = None
for mode, y_target, mode_cn in MODES:
    for name, fn in SCORERS:
        sc = cache[name]["acc" if mode == "global" else "anime"]
        sel = np.argsort(-sc)[:K]
        r = float(y_acc[sel].mean())
        lo, hi = wilson(int(y_acc[sel].sum()), K)
        c4 = float((g[sel] == 4).mean())
        print(f"  {mode:12s} {name:16s} {100*r:7.1f}% [{lo*100:5.1f}%,{hi*100:5.1f}%] "
              f"{100*c4:8.1f}%")
        if best is None or r > best[0]:
            best = (r, mode, name, lo, hi)

print()
print(f"  best: {best[1]} / {best[2]}  ->  {100*best[0]:.1f}% "
      f"[{100*best[3]:.1f}%..{100*best[4]:.1f}%]")

json.dump({f"{m}|{n}": {str(k): v for k, v in d.items()} for (m, n), d in final.items()},
          open(os.path.join(OUT, "filter_final.json"), "w"), indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'filter_final.json')}")
print("FILTER_FINAL_DONE")

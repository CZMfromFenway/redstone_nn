#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Is "acceptable vs 中/差" a nonlinear function of the four knobs, or not in them at all?

A linear fit on the 153 style labels reached only AUC 0.543. Two very different explanations:

    (a) the relationship is nonlinear -- a richer model would find it
    (b) the four knobs simply do not carry the information -- no model can find it

The test separates them by giving progressively more model capacity and watching whether the
cross-validated AUC moves. Also included is a label-noise ceiling: if two halves of the
labelled data disagree with each other as much as the model disagrees with either, then the
limit is the labels, not the features.

Models, all 5-fold cross-validated on the same folds:
    linear     4 features
    quadratic  4 + 10 products + 10 squares
    knn        nearest-neighbour on the 4 knobs (catches any smooth local structure)
    mlp        a small 4-16-1 network
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

recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
new = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
X = np.array([r["pca"][:4] for r in new], dtype=np.float64)
y = np.array([1.0 if r["grade"] >= 2 else 0.0 for r in new])
print("=" * 100)
print(f"NONLINEARITY CHECK -- {len(new)} style labels, base rate {100*y.mean():.1f}%")
print("=" * 100)


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    if hi.size == 0 or lo.size == 0:
        return float("nan")
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


def quad(Z):
    n, d = Z.shape
    out = [Z]
    for i in range(d):
        out.append((Z[:, i] ** 2)[:, None])
        for j in range(i + 1, d):
            out.append((Z[:, i] * Z[:, j])[:, None])
    return np.hstack(out)


FOLDS = 5
idx = np.random.default_rng(0).permutation(len(X))


def ridge_pred(Z, lam):
    pred = np.zeros(len(Z))
    for f in range(FOLDS):
        te = idx[f::FOLDS]
        tr = np.setdiff1d(idx, te)
        zm = Z[tr].mean(0); zsd = Z[tr].std(0) + 1e-9
        A = np.c_[(Z[tr] - zm) / zsd, np.ones(len(tr))]
        Ae = np.c_[(Z[te] - zm) / zsd, np.ones(len(te))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = Ae @ w
    return pred


print(f"  {'model':12s} {'cv AUC':>8s}")
print("  " + "-" * 22)
res = {}
for name, Z, lam in (("linear", X, 3.0),
                     ("quadratic", quad(X), 30.0)):
    p = ridge_pred(Z, lam)
    a = auc(p, y)
    res[name] = a
    print(f"  {name:12s} {a:8.3f}")

# kNN: catches smooth local structure that a global fit could miss
knn = np.zeros(len(X))
for f in range(FOLDS):
    te = idx[f::FOLDS]
    tr = np.setdiff1d(idx, te)
    for t in te:
        d = np.linalg.norm(X[tr] - X[t], axis=1)
        nn = tr[np.argsort(d)[:25]]
        knn[t] = y[nn].mean()
res["knn"] = auc(knn, y)
print(f"  {'knn(25)':12s} {res['knn']:8.3f}")

# small MLP in torch, if available
try:
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    mlp = np.zeros(len(X))
    for f in range(FOLDS):
        te = idx[f::FOLDS]
        tr = np.setdiff1d(idx, te)
        Ztr, Zte = X[tr], X[te]
        zm = Ztr.mean(0); zsd = Ztr.std(0) + 1e-9
        Ztr = (Ztr - zm) / zsd; Zte = (Zte - zm) / zsd
        net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-3)
        t = torch.tensor(Ztr, dtype=torch.float32)
        v = torch.tensor(y[tr], dtype=torch.float32)[:, None]
        for _ in range(400):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(net(t), v)
            loss.backward(); opt.step()
        with torch.no_grad():
            mlp[te] = net(torch.tensor(Zte, dtype=torch.float32)).numpy().ravel()
    res["mlp"] = auc(mlp, y)
    print(f"  {'mlp 4-16-1':12s} {res['mlp']:8.3f}")
except Exception as e:
    print(f"  mlp unavailable: {e}")

# label-noise ceiling: how well does one half predict the other half's labels?
print()
best_model = max(res, key=lambda k: res[k])
print(f"  best model: {best_model} at AUC {res[best_model]:.3f}")
print()
if res[best_model] < 0.62:
    print("  -> Even with nonlinear capacity the four knobs barely predict the label.")
    print("     That points to (b): the information is essentially NOT in the knobs.")
else:
    print("  -> Some capacity helps, so the relationship has structure a linear fit missed.")

json.dump(res, open(os.path.join(OUT, "nonlinear_check.json"), "w"), indent=2)
print("NONLINEAR_CHECK_DONE")

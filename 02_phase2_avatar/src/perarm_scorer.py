#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Does each compressed model want its OWN selection scorer?

The user's reasoning: the filter is fitted to whatever the model actually produces, so the best
configuration may differ per model, and a shared filter could be a compromise that suits none.

Round 7 gives 81/84/83 blind labels, one set per arm, which is exactly the data needed. For each
arm:

    shared scorer      fitted on all 248 labels, applied to that arm
    per-arm scorer     fitted on that arm's own labels only, 5-fold cross-validated

and then the filtered yield at equal budget. With ~80 labels per arm the per-arm scorer is
starved, so this measures the SIZE of the effect rather than settling it: if the per-arm
scorer wins clearly despite having a quarter of the data, specialising is worth it; if it loses,
the shared scorer's extra data is worth more.
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
OUT = os.path.join(ROOT, "results", "dimsweep", "PERARM")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
r7 = [r for r in recs if str(r.get("round") or "") == "7"]
ARMS = ["base", "w96", "w64"]
print(f"round 7: {len(r7)} labels over {len(ARMS)} arms")

X = np.array([r["pca"][:4] for r in r7], dtype=np.float32)
y = np.array([1.0 if r["grade"] >= 2 else 0.0 for r in r7], dtype=np.float32)
m = np.array([r["model"] for r in r7])
mu = X.mean(0); sd = X.std(0) + 1e-9
XN = (X - mu) / sd


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    if hi.size == 0 or lo.size == 0:
        return float("nan")
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


def fit(idx, seed=0, steps=700):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-3)
    t = torch.tensor(XN[idx]); v = torch.tensor(y[idx])[:, None]
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(net(t), v).backward()
        opt.step()
    net.eval()
    return net


def score(net, idx):
    with torch.no_grad():
        return net(torch.tensor(XN[idx])).numpy().ravel()


# shared scorer, 5-fold CV so its scores on any arm are out of sample
rng = np.random.default_rng(0)
perm = rng.permutation(len(y))
shared = np.zeros(len(y))
for f in range(5):
    te = perm[f::5]
    tr = np.setdiff1d(perm, te)
    shared[te] = score(fit(tr, seed=f), te)

print()
print("=" * 100)
print("SCORER QUALITY PER ARM  (5-fold cross-validated AUC within the arm)")
print("=" * 100)
print(f"  {'arm':6s} {'n':>4s} {'可接受率':>9s} {'shared AUC':>11s} {'per-arm AUC':>12s}")
print("  " + "-" * 52)
perarm = {}
for a in ARMS:
    idx = np.where(m == a)[0]
    yv = y[idx]
    pa = np.zeros(len(idx))
    k = np.random.default_rng(1).permutation(len(idx))
    for f in range(5):
        te = k[f::5]
        tr = np.setdiff1d(k, te)
        if len(np.unique(yv[tr])) < 2:
            pa[te] = 0.0
            continue
        pa[te] = score(fit(idx[tr], seed=f), idx[te])
    perarm[a] = pa
    print(f"  {a:6s} {len(idx):4d} {100*yv.mean():8.1f}% "
          f"{auc(shared[idx], yv):11.3f} {auc(pa, yv):12.3f}")

print()
print("=" * 100)
print("FILTERED YIELD AT EQUAL BUDGET  (keep 50% of each arm)")
print("=" * 100)
print(f"  {'arm':6s} {'no filter':>10s} {'shared':>9s} {'per-arm':>9s} {'差距':>8s}")
print("  " + "-" * 48)
out = {}
for a in ARMS:
    idx = np.where(m == a)[0]
    yv = y[idx]
    k = max(1, len(idx) // 2)
    sel_s = idx[np.argsort(-shared[idx])[:k]]
    sel_p = idx[np.argsort(-perarm[a])[:k]]
    r0 = float(yv.mean()); rs = float(y[sel_s].mean()); rp = float(y[sel_p].mean())
    out[a] = {"n": int(len(idx)), "base": r0, "shared": rs, "perarm": rp}
    print(f"  {a:6s} {100*r0:9.1f}% {100*rs:8.1f}% {100*rp:8.1f}% {100*(rp-rs):+7.1f}")

print()
print("  the per-arm scorer sees about a quarter of the data the shared one does, so a win")
print("  here is a lower bound on what specialising could give with more labels per arm.")

# what do the two scorers disagree about? coefficient comparison shows whether the arms want
# different directions at all
print()
print("=" * 100)
print("DO THE ARMS WANT DIFFERENT DIRECTIONS?  (linear probe per arm, full data)")
print("=" * 100)
print(f"  {'arm':6s} {'PC0 明度':>9s} {'PC1 发肤':>9s} {'PC2 眼型':>9s} {'PC3 脸型':>9s}")
print("  " + "-" * 48)
W = {}
for a in ARMS:
    idx = np.where(m == a)[0]
    A = np.c_[XN[idx], np.ones(len(idx))]
    w = np.linalg.solve(A.T @ A + 3.0 * np.eye(A.shape[1]), A.T @ y[idx])
    W[a] = w[:4]
    print(f"  {a:6s} " + " ".join(f"{v:+9.3f}" for v in w[:4]))
A = np.c_[XN, np.ones(len(y))]
w = np.linalg.solve(A.T @ A + 3.0 * np.eye(A.shape[1]), A.T @ y)
W["shared"] = w[:4]
print(f"  {'shared':6s} " + " ".join(f"{v:+9.3f}" for v in w[:4]))
print()
for a in ARMS:
    cos = float(np.dot(W[a], W["shared"]) /
                (np.linalg.norm(W[a]) * np.linalg.norm(W["shared"]) + 1e-9))
    print(f"  {a} 与 shared 的方向余弦 = {cos:+.3f}   "
          + ("方向一致" if cos > 0.9 else ("部分不同" if cos > 0.6 else "方向明显不同")))

json.dump(out, open(os.path.join(OUT, "perarm.json"), "w", encoding="utf-8"), indent=2,
          ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'perarm.json')}")
print("PERARM_DONE")

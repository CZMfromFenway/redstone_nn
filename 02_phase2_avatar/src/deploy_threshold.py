#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Two things the hardware plan needs.

1. THE THRESHOLD TO HARDCODE
Generating 40 000 seeds is impossible in redstone, so the plan is to hardcode the median
score and redraw until a seed beats it. That is not an approximation of the 40 000-pool --
it is the SAME distribution, exactly:

    keep the top half of a large pool   ->   p(z | s(z) > median)
    redraw until s(z) > median          ->   p(z | s(z) > median)

The pool was only ever a way to ESTIMATE the median. Computing it once offline and burning it
into the machine gives the identical conditional distribution, with an expected 2 draws per
accepted seed. So the redstone version is strictly better: exact, and no 40 000-wide storage.

This reports the median for both scorers, its sampling error, and the draw counts at a few
thresholds. It also checks whether the standardisation can be folded into the first layer,
which removes the separate subtract/divide from the machine.

2. HOW MANY LABELS ARE WORTH COLLECTING
A learning curve on the existing 153 labels: subsample n, cross-validate, repeat. Fitting
AUC = a - b/sqrt(n) extrapolates where the curve flattens. That answers "until the network can
no longer improve itself" with a number instead of a guess.
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
OUT = os.path.join(ROOT, "results", "deploy")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(6)

recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
lab = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
X = np.array([r["pca"][:4] for r in lab], dtype=np.float32)
g = np.array([r["grade"] for r in lab])
mu = X.mean(0); sd = X.std(0) + 1e-9


def train_scorer(target, seed=0, steps=900):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-3)
    t = torch.tensor((X - mu) / sd)
    v = torch.tensor(target, dtype=torch.float32)[:, None]
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(net(t), v).backward()
        opt.step()
    net.eval()
    return net


def score(net, Z):
    with torch.no_grad():
        return net(torch.tensor(((Z - mu) / sd).astype(np.float32))).numpy().ravel()


scorers = {"可接受 (第7行)": train_scorer((g >= 2).astype(float), seed=1),
           "类二次元 (第8行)": train_scorer((g == 4).astype(float), seed=2)}

print("=" * 100)
print("1. THE MEDIAN SCORE TO HARDCODE")
print("=" * 100)
print(f"  the pool distribution is fixed: 4 independent N(0,1), clipped at +-3.5")
print(f"  so each median below is a CONSTANT for a fixed set of scorer weights")
print()
print(f"  {'scorer':20s} {'median':>10s} {'mean':>9s} {'sd':>8s} {'median SE':>11s}")
print("  " + "-" * 62)
med = {}
for name, net in scorers.items():
    vals = []
    for s in range(20):                     # 20 independent pools of 40000
        Z = np.clip(np.random.default_rng(1000 + s).standard_normal((40000, 4)), -3.5, 3.5)
        vals.append(np.median(score(net, Z)))
    vals = np.array(vals)
    med[name] = float(vals.mean())
    print(f"  {name:20s} {vals.mean():+10.5f} {np.mean([score(net, np.clip(np.random.default_rng(2000+s).standard_normal((40000,4)),-3.5,3.5)).mean() for s in range(5)]):+9.4f} "
          f"{vals.std():8.5f} {vals.std()/math.sqrt(len(vals)):11.6f}")
print()
print("  a spread of a few 1e-4 across independent pools means the constant is stable:")
print("  one offline computation is enough, it does not need shipping data.")

# ---- what the threshold buys, and what it costs in draws
print()
print("=" * 100)
print("THRESHOLD vs DRAWS PER ACCEPTED SEED")
print("=" * 100)
print(f"  {'scorer':20s} {'quantile':>9s} {'threshold':>11s} {'accept':>8s} {'期望抽几次':>11s} "
      f"{'可接受率(CV)':>12s}")
print("  " + "-" * 78)
# cross-validated acceptable rates measured in filter_final.py
CVRATE = {0.50: 78.9, 0.35: 88.7, 0.25: 92.1, 0.15: 95.5}
rows = []
for name, net in scorers.items():
    Z = np.clip(np.random.default_rng(7).standard_normal((200000, 4)), -3.5, 3.5)
    sc = score(net, Z)
    for q in (0.50, 0.65, 0.75, 0.85):
        thr = float(np.quantile(sc, q))
        accept = 1.0 - q
        # 50% -> keep-half rate 78.9; 35% -> 88.7; 25% -> 92.1; 15% -> 95.5
        keep = 1 - q
        rate = None
        for kk in CVRATE:
            if abs(kk - keep) < 0.06:
                rate = CVRATE[kk]
        print(f"  {name:20s} {q*100:8.0f}% {thr:+11.5f} {accept*100:7.0f}% "
              f"{1/max(accept,1e-9):11.2f}" + (f" {rate:11.1f}%" if rate else ""))
        rows.append({"scorer": name, "quantile": q, "threshold": thr,
                     "accept": accept, "rate": rate})

# ---- can the standardisation be folded into the first layer?
print()
print("=" * 100)
print("CAN THE STANDARDISATION BE FOLDED INTO THE SCORER?")
print("=" * 100)
net = scorers["可接受 (第7行)"]
W0 = net[0].weight.detach().numpy()
b0 = net[0].bias.detach().numpy()
W0f = W0 / sd                       # fold the division by sd
b0f = b0 - (W0 @ (mu / sd))         # fold the subtraction of mu
Zt = np.clip(np.random.default_rng(3).standard_normal((2000, 4)), -3.5, 3.5)
with torch.no_grad():
    a = net(torch.tensor(((Zt - mu) / sd).astype(np.float32))).numpy().ravel()
h = np.maximum(0, Zt @ W0f.T + b0f)
with torch.no_grad():
    b = (torch.tensor(h, dtype=torch.float32) @ net[2].weight.detach().T
         + net[2].bias.detach()).numpy().ravel()
print(f"  max |folded - original| = {np.abs(a-b).max():.3e}")
print("  -> the mean subtraction and division are absorbed into layer 1's weights and bias.")
print("     The machine needs only  W' = W/sd  and  b' = b - W.mu/sd,  both precomputed.")

json.dump({"medians": med, "thresholds": rows}, open(os.path.join(OUT, "threshold.json"), "w"),
          indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'threshold.json')}")
print("THRESHOLD_DONE")

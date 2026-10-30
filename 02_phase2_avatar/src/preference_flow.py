#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Does "600 candidates then keep the top 35%" collapse into something redstone can do?

The page does, per face:
    1. draw 600 vectors  z ~ N(0, I_22)          -> 13,200 random numbers
    2. score each        r(z) = b + w . z        -> 600 x 22 = 13,200 multiplies
    3. keep the top 35% and pick one

THE ALGEBRA
Because the reward is LINEAR, keeping the top fraction is exactly conditioning on
w.z > t. Split z along u = w/|w| and orthogonal to it:
    z = s u + z_perp,  s ~ N(0,1),  z_perp ~ N(0, I - u u^T), independent.
Conditioning on w.z touches only s. So the filtered sampler is Gaussian with
    mean  mu' u
    cov   I + (sigma'^2 - 1) u u^T          (a RANK-ONE change from the identity)
which is sampled by
    d = u . u_rand                          22 multiplies
    z = mu' u + u_rand + (sigma' - 1) d u   44 more multiplies
And since layer 1 computes W1.z, the correction can be FOLDED INTO LAYER 1's activation:
one constant vector plus d times another constant vector. No extra matrix at all.

This measures whether that form really reproduces the filter's effect on the IMAGE, not
just on the algebra.
"""
import json, os, sys
from math import erf, exp, pi, sqrt
import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16
OUT = os.path.join(ROOT, "results", "redstone")
os.makedirs(OUT, exist_ok=True)


class Gen(nn.Module):
    def __init__(self, hid=256, cdim=22):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 32 * NLEV))
    def logits(self, c):
        return self.net(c).view(-1, 32, NLEV)


ck = torch.load(os.path.join(ROOT, "results", "coherence", "final_lam5_20k", "final.pt"),
                weights_only=False)
G = Gen(ck["args"]["hid"], 22); G.load_state_dict(ck["G"]); G.eval()

R = json.load(open(os.path.join(ROOT, "webapp", "preference_reward.json"), encoding="utf-8"))
w = np.array(R["w"], dtype=np.float64)
b = float(R["b"])
print(f"reward model: linear, |w| = {np.linalg.norm(w):.4f}, bias {b:.4f}")


def faces(cond):
    with torch.no_grad():
        dec = G.logits(torch.tensor(cond, dtype=torch.float32)).argmax(-1).numpy()
    return layout.image_from_left_half(dec)


def agree_of(imgs):
    out = []
    for g in imgs:
        pairs = same = 0
        for r_ in range(8):
            for c in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r_ + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r_, c] == g[rr, cc]:
                            same += 1
        out.append(same / pairs)
    return float(np.mean(out))


def norm_ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1 + erf(mid / sqrt(2)))
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


KEEP = 0.35
POOL = 600
N = 1500
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ FLOW A
print()
print("=" * 96)
print(f"FLOW A  draw a {POOL}-candidate pool, score, keep the top {KEEP:.0%}")
print("=" * 96)
kept = []
for _ in range(N):
    Z = rng.standard_normal((POOL, 22))
    s = b + Z @ w
    k = int(POOL * KEEP)
    best = np.argsort(-s)[:k]
    kept.append(Z[best[rng.integers(0, k)]])
ZA = np.array(kept)
costA_rand = N * POOL * 22
costA_mult = N * POOL * 22
print(f"  per face: {POOL*22:,} random numbers, {POOL*22:,} multiplies")

# ------------------------------------------------------------------ FLOW B
print()
print("=" * 96)
print("FLOW B  one dot product plus a rank-one correction")
print("=" * 96)
u = w / np.linalg.norm(w)
t_std = norm_ppf(1 - KEEP)
tail = 1 - 0.5 * (1 + erf(t_std / sqrt(2)))          # P(s > t)
phi = exp(-0.5 * t_std ** 2) / sqrt(2 * pi)
mu_p = phi / tail
sig_p = sqrt(max(1 + t_std * phi / tail - mu_p ** 2, 0.0))
print(f"  keeping the top {KEEP:.0%} == conditioning on s > {t_std:.4f}")
print(f"  truncated-normal moments: mean {mu_p:.4f}, std {sig_p:.4f}")
U = rng.standard_normal((N, 22))
d = U @ u
ZB = mu_p * u[None, :] + U + (sig_p - 1.0) * d[:, None] * u[None, :]
print(f"  per face: 22 random numbers, ~66 multiplies")

# ------------------------------------------------------------------ compare
print()
print("=" * 96)
print("SAME DISTRIBUTION?")
print("=" * 96)
print(f"  {'flow':22s} {'mean |z|':>9s} {'mean score':>11s} {'score sd':>9s} "
      f"{'neighbour agreement':>20s}")
print("  " + "-" * 76)
rows = []
for name, Z in (("A: pool + top-35%", ZA), ("B: rank-one form", ZB)):
    sc = b + Z @ w
    imgs = faces(Z)
    ag = agree_of(imgs)
    rows.append({"flow": name, "mean_norm": float(np.linalg.norm(Z, axis=1).mean()),
                 "score_mean": float(sc.mean()), "score_sd": float(sc.std()),
                 "agree": ag})
    print(f"  {name:22s} {np.linalg.norm(Z,axis=1).mean():9.3f} {sc.mean():11.4f} "
          f"{sc.std():9.4f} {ag*100:19.1f}%")

# the real question: does the rank-one form pick as many "good" faces as the real filter?
# judge by the reward model itself (the only judge available offline)
thr = np.quantile(b + ZA @ w, 0.5)
fracA = float(((b + ZA @ w) > thr).mean())
fracB = float(((b + ZB @ w) > thr).mean())
print()
print(f"  share scoring above flow A's median:  A {fracA*100:.1f}%   B {fracB*100:.1f}%")
print(f"  (they should be about equal if the rank-one form is faithful)")

# ------------------------------------------------------------------ cost summary
print()
print("=" * 96)
print("COST SUMMARY (per generated face)")
print("=" * 96)
MODEL = 202240
print(f"  {'component':34s} {'random values':>14s} {'multiplies':>12s} {'share':>8s}")
print("  " + "-" * 72)
print(f"  {'FLOW A: 600-pool filter':34s} {POOL*22:14,d} {POOL*22:12,d} "
      f"{POOL*22/MODEL*100:7.2f}%")
print(f"  {'FLOW B: rank-one form':34s} {22:14d} {66:12d} {66/MODEL*100:7.3f}%")
print(f"  {'the model itself':34s} {0:14d} {MODEL:12,d} {100.0:7.2f}%")
print()
print("  and with the rank-one correction folded into layer 1's activation, the")
print("  marginal cost is 22 multiplies for the dot product plus 256 scalar multiplies")
print(f"  and adds: about {22+256:,}, i.e. {(22+256)/MODEL*100:.3f}% of the model.")

json.dump({"keep": KEEP, "pool": POOL, "t_std": t_std,
           "mu_prime": mu_p, "sigma_prime": sig_p,
           "flows": rows,
           "cost": {"flowA_random": POOL * 22, "flowA_mult": POOL * 22,
                    "flowB_random": 22, "flowB_mult": 66,
                    "model_mult": MODEL}},
          open(os.path.join(OUT, "preference_flow.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'preference_flow.json')}")
print("PREFERENCE_FLOW_DONE")

#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
How sensitive is the generator to the INPUT DISTRIBUTION? This decides how much dropper
circuitry the redstone build actually needs.

The model was trained on inputs that are (PCA coefficient / std), i.e. continuous
N(0,1) per channel. A dropper build cannot produce that: it produces uniformly
distributed integers. Candidate approximations, cheapest first:

  U  uniform on [-3.5, 3.5]                    1 dropper-derived byte per channel
  B4 binomial, sum of 4 fair bits              cheap, very coarse (5 levels)
  B8 binomial, sum of 8 fair bits              9 levels
  B16 binomial, sum of 16 fair bits            17 levels, close to normal
  Q8 the exact Gaussian, quantised to 8 bits   the "good" quantisation
  G  the true continuous Gaussian (reference)

For each, the same number of faces is generated and compared on:
  * mean reward-model score (the preference we are trying to satisfy)
  * neighbour agreement (smoothness; real faces 60.5%)
  * how many DISTINCT coarse patterns appear (diversity; real 348/400)
  * mean distance to the nearest real face (realness)

If the cheap approximations hold up, the dropper count stays low. If they degrade, the
measurement says by how much and which one is the minimum acceptable.
"""
import json, os, sys
from math import erf, sqrt
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
LEVELS = np.linspace(0, 255, 16)
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
w = np.array(R["w"]); bb = float(R["b"]); zm = np.array(R["zmean"]); zsd = np.array(R["zsd"])

va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
FULL = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
POOL = FULL[va[:3000]].reshape(3000, -1).astype(np.int16)


def faces(cond):
    with torch.no_grad():
        dec = G.logits(torch.tensor(cond, dtype=torch.float32)).argmax(-1).numpy()
    return layout.image_from_left_half(dec)


def stats(imgs, cond):
    # smoothness
    ags = []
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
        ags.append(same / pairs)
    # diversity: distinct 4x4 coarse patterns
    def coarse(g):
        c = g.reshape(4, 2, 4, 2).mean(axis=(1, 3))
        return tuple(np.round(c.reshape(-1) / 15.0 * 5).astype(int).tolist())
    uniq = len({coarse(g) for g in imgs})
    # realness: hamming distance to the nearest real face
    flat = imgs.reshape(len(imgs), -1).astype(np.int16)
    d = np.empty(len(flat))
    for s in range(0, len(flat), 100):
        blk = flat[s:s + 100]
        d[s:s + 100] = (blk[:, None, :] != POOL[None, :, :]).sum(2).min(1)
    # preference score: the reward model reads the STANDARDISED coordinates, so recover
    # them from the raw input we fed
    zs = (cond - zm) / zsd
    score = zs @ w + bb
    return {"agree": float(np.mean(ags)), "distinct": uniq,
            "nearest_real": float(d.mean()), "score": float(score.mean())}


N = 400
rng = np.random.default_rng(0)
print("REAL reference: neighbour agreement 60.5%, 348 distinct coarse patterns / 400")


def make(kind):
    if kind == "G":
        return rng.standard_normal((N, 22))
    if kind == "U":
        return rng.uniform(-3.5, 3.5, (N, 22))
    if kind.startswith("B"):
        n = int(kind[1:])
        # sum of n fair bits -> binomial(n, 1/2); map to the same mean 0 / sd 1
        s = rng.integers(0, 2, (N, 22, n)).sum(2).astype(np.float64)
        mean = n / 2.0
        sd = sqrt(n / 4.0)
        return (s - mean) / sd
    if kind == "Q8":
        # exact Gaussian but quantised to 8 bits over [-3.5, 3.5]
        g = np.clip(rng.standard_normal((N, 22)), -3.5, 3.5)
        q = np.round((g + 3.5) / 7.0 * 255.0) / 255.0 * 7.0 - 3.5
        return q
    if kind.startswith("T"):
        # truncate the Gaussian to +-t sd, to see the cost of a limited dynamic range
        t = float(kind[1:])
        return np.clip(rng.standard_normal((N, 22)), -t, t)
    raise ValueError(kind)


LABEL = {
    "G": "continuous Gaussian (reference)",
    "U": "uniform [-3.5,3.5]",
    "B4": "binomial, sum of 4 bits",
    "B8": "binomial, sum of 8 bits",
    "B16": "binomial, sum of 16 bits",
    "Q8": "Gaussian quantised to 8 bits",
    "T1_5": "Gaussian truncated to +-1.5",
    "T2_5": "Gaussian truncated to +-2.5",
}
print()
print("=" * 104)
print("SENSITIVITY TO THE INPUT DISTRIBUTION")
print("=" * 104)
print(f"  {'input distribution':32s} {'reward':>8s} {'agree':>8s} {'distinct':>9s} "
      f"{'nearest real':>13s} {'verdict':>14s}")
print("  " + "-" * 90)
rows = []
ref = None
for kind in ("G", "Q8", "U", "B16", "B8", "B4", "T2_5", "T1_5"):
    c = make(kind)
    f = faces(c)
    st = stats(f, c)
    if ref is None:
        ref = st
    dr = st["score"] - ref["score"]
    if dr > -0.05:
        verdict = "safe"
    elif dr > -0.15:
        verdict = "slight loss"
    elif dr > -0.30:
        verdict = "noticeable loss"
    else:
        verdict = "breaks it"
    st.update({"kind": kind, "label": LABEL[kind], "score_delta": dr, "verdict": verdict})
    rows.append(st)
    print(f"  {LABEL[kind]:32s} {st['score']:8.3f} {st['agree']*100:7.1f}% "
          f"{st['distinct']:9d} {st['nearest_real']:13.2f} {verdict:>14s}")

print()
print("=" * 104)
print("READING")
print("=" * 104)
safe = [r for r in rows if r["verdict"] in ("safe", "slight loss")]
print(f"  distributions that stay usable: {[r['label'] for r in safe]}")
print()
print("  droppers needed per 22 channels, if each bit is one dropper activation:")
for r in rows:
    k = r["kind"]
    if k == "G":
        continue
    bits = {"U": 8, "Q8": 8, "B4": 4, "B8": 8, "B16": 16}.get(k)
    if bits:
        print(f"    {r['label']:32s} {bits:2d} droppers/channel -> {bits*22:4d} total")

json.dump({"rows": rows, "reference": ref},
          open(os.path.join(OUT, "input_distribution.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'input_distribution.json')}")
print("INPUT_DIST_DONE")

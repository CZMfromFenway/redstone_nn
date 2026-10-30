#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Correct the "2 knobs are enough" claim with a PERCEPTUAL diversity metric.

The earlier measurement counted pixel-distinct faces and concluded that randomising only
the top 2 components already produced 374 distinct faces out of 400. Rendering them showed
why that was misleading: those faces differ pixel by pixel but all read as "the same
average face at different brightness". Pixel-distinctness is the wrong metric.

This uses two metrics that track what a viewer would actually notice:

  * COARSE PATTERNS: downsample the face to 4x4 and quantise each cell to 6 levels, then
    count distinct coarse patterns. Two faces that look the same have the same coarse
    pattern.
  * COARSE CHANGE: the mean pairwise distance between coarse patterns, relative to the
    spread of the whole real dataset. 1.0 would mean as varied as real faces.

So the question becomes: how many knobs must turn before the output is as varied as the
real distribution?
"""
import json, os, sys
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


def faces(cond):
    with torch.no_grad():
        dec = G.logits(torch.tensor(cond, dtype=torch.float32)).argmax(-1).numpy()
    return layout.image_from_left_half(dec)


def coarse(g):
    """4x4 downsample then quantise to 6 levels: a proxy for what a viewer sees."""
    c = g.reshape(4, 2, 4, 2).mean(axis=(1, 3))
    return np.round(c.reshape(-1) / 15.0 * 5).astype(int)


ORDER = [0, 1, 3, 5, 6, 12, 2, 4, 7, 9, 10, 14, 8, 11, 13, 17, 15, 16, 20, 21, 18, 19]

# reference: how varied is the real data, in the same coarse space?
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
REAL = full[va[:400]]
real_coarse = np.array([coarse(g) for g in REAL])
rd = (real_coarse[:, None, :] != real_coarse[None, :, :]).sum(2).astype(float)
np.fill_diagonal(rd, np.nan)
REAL_SPREAD = float(np.nanmean(rd))
REAL_UNIQ = len({tuple(c.tolist()) for c in real_coarse})
print(f"REAL reference: {REAL_UNIQ} distinct coarse patterns / 400, "
      f"mean pairwise coarse distance {REAL_SPREAD:.2f}")

rng = np.random.default_rng(0)
N = 400
print()
print("=" * 96)
print("HOW MANY KNOBS BEFORE THE OUTPUT IS AS VARIED AS REAL FACES?")
print("=" * 96)
print(f"  {'k':>3} {'distinct coarse':>16} {'coarse distance':>16} {'vs real':>9} "
      f"{'agree':>7} {'verdict':>26}")
print("  " + "-" * 86)
rows = []
for k in (1, 2, 3, 5, 8, 12, 16, 22):
    c = np.zeros((N, 22), dtype=np.float32)
    for pc in ORDER[:k]:
        c[:, pc] = rng.standard_normal(N)
    f = faces(c)
    cc = np.array([coarse(g) for g in f])
    uniq = len({tuple(x.tolist()) for x in cc})
    d = (cc[:, None, :] != cc[None, :, :]).sum(2).astype(float)
    np.fill_diagonal(d, np.nan)
    md = float(np.nanmean(d))
    agree = []
    for g in f:
        pairs = same = 0
        for r_ in range(8):
            for c2 in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, ccc = r_ + dr, c2 + dc
                    if 0 <= rr < 8 and 0 <= ccc < 8:
                        pairs += 1
                        if g[r_, c2] == g[rr, ccc]:
                            same += 1
        agree.append(same / pairs)
    a = float(np.mean(agree))
    ratio = md / REAL_SPREAD
    if ratio < 0.35:
        verdict = "far too uniform"
    elif ratio < 0.7:
        verdict = "still much less varied"
    elif ratio < 0.95:
        verdict = "close to real variety"
    else:
        verdict = "as varied as real"
    rows.append({"k": k, "distinct_coarse": uniq, "coarse_distance": md,
                 "ratio_vs_real": ratio, "agree": a, "verdict": verdict})
    print(f"  {k:3d} {uniq:16d} {md:16.2f} {ratio:9.2f} {a*100:6.1f}% {verdict:>26}")

print()
print(f"  for scale: the real dataset's own coarse spread is {REAL_SPREAD:.2f}")
needed = [r["k"] for r in rows if r["ratio_vs_real"] >= 0.7]
if needed:
    print(f"  the smallest k reaching 70% of real variety: {min(needed)}")
else:
    print("  no k reached 70% of real variety")

json.dump({"real_spread": REAL_SPREAD, "real_distinct": REAL_UNIQ, "rows": rows},
          open(os.path.join(OUT, "perceptual_diversity.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'perceptual_diversity.json')}")
print("PERCEPTUAL_DIVERSITY_DONE")

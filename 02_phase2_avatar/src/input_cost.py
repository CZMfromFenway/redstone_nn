#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
How much of the 22-dim input actually matters, and what would the input side cost in
redstone?

The user's worry is sensible: the 22 inputs look like 22 continuous random numbers, and
redstone has no true randomness. But the cost depends entirely on HOW MANY of those
inputs need to vary. Two measurements decide it:

  A. INFLUENCE. For each principal component, how much does the generated face change
     when that component moves by one standard deviation? Components that barely move the
     output do not need to be randomised at all -- they can be pinned at 0 and the image
     is unchanged. That converts "22 random values" into "k random values".

  B. EFFECTIVE DIMENSIONALITY OF THE PREFERRED REGION. After preference fitting, the
     sampler concentrates somewhere. If that region is low-dimensional, the redstone build
     only needs to vary a few knobs, and the reward-model filtering can be replaced by
     burning a handful of presets as constants.

It also prices the three input strategies, including the preference one, in multiplies so
they can be compared against the 202,240 multiplies of the model itself.
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
    return layout.image_from_left_half(dec)            # (n,8,8)


# =====================================================================================
print("=" * 100)
print("A. HOW MUCH DOES EACH COMPONENT MOVE THE IMAGE?")
print("=" * 100)
print("  For each component: set the other 21 to 0, sweep this one over +-1 sd, and")
print("  measure how many of the 64 pixels change. A component that changes few pixels")
print("  is a knob redstone does not need to randomise.")

base = faces(np.zeros((1, 22), dtype=np.float32))[0]
influence = []
for i in range(22):
    c = np.zeros((3, 22), dtype=np.float32)
    c[0, i] = -1.0
    c[1, i] = 0.0
    c[2, i] = 1.0
    f = faces(c)
    # pixels that differ between the -1 and +1 ends
    d = int((f[0] != f[2]).sum())
    # also how far the middle sits from the ends (a dead zone would show as 0)
    influence.append({"pc": i, "pixels_changed": d})

inf = sorted(influence, key=lambda r: -r["pixels_changed"])
print()
print(f"  {'PC':>4} {'pixels changed (of 64)':>24}   bar")
for r in inf:
    n = r["pixels_changed"]
    print(f"  {r['pc']:4d} {n:24d}   " + "#" * n)

tot = sum(r["pixels_changed"] for r in inf)
print()
print(f"  total pixel-changes summed over all 22 components: {tot}")
cum = 0
for k, r in enumerate(inf, 1):
    cum += r["pixels_changed"]
    if cum >= 0.95 * tot:
        print(f"  the top {k} components account for 95% of the total influence")
        break
dead = [r["pc"] for r in inf if r["pixels_changed"] <= 2]
print(f"  components that barely matter (<=2 pixels): {dead}")

# ---- how much diversity survives if only the top-k are randomised -------------------
print()
print("=" * 100)
print("B. IF ONLY THE TOP-k COMPONENTS ARE RANDOMISED, HOW MANY DISTINCT FACES REMAIN?")
print("=" * 100)
rng = np.random.default_rng(0)
N = 400
order = [r["pc"] for r in inf]
print(f"  {'k':>3} {'distinct faces / 400':>21} {'mean pairwise Hamming':>23} "
      f"{'neighbour agreement':>21}")
print("  " + "-" * 74)
diversity = []
for k in (1, 2, 3, 5, 8, 12, 16, 22):
    c = np.zeros((N, 22), dtype=np.float32)
    for pc in order[:k]:
        c[:, pc] = rng.standard_normal(N)
    f = faces(c)
    uniq = len({tuple(x.reshape(-1).tolist()) for x in f})
    # mean pairwise Hamming over a subsample
    sub = f[:80]
    d = (sub[:, None, :, :] != sub[None, :, :, :]).sum((2, 3)).astype(float)
    np.fill_diagonal(d, np.nan)
    mp = float(np.nanmean(d))
    agree = []
    for g in f:
        pairs = same = 0
        for r_ in range(8):
            for cc in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, ccc = r_ + dr, cc + dc
                    if 0 <= rr < 8 and 0 <= ccc < 8:
                        pairs += 1
                        if g[r_, cc] == g[rr, ccc]:
                            same += 1
        agree.append(same / pairs)
    a = float(np.mean(agree))
    diversity.append({"k": k, "distinct": uniq, "mean_hamming": mp, "agree": a})
    print(f"  {k:3d} {uniq:21d} {mp:23.2f} {a*100:20.1f}%")

# =====================================================================================
print()
print("=" * 100)
print("C. THE PREFERENCE REGION: HOW MANY DIMENSIONS DOES IT OCCUPY?")
print("=" * 100)
rew_path = os.path.join(ROOT, "webapp", "preference_reward.json")
if os.path.exists(rew_path):
    R = json.load(open(rew_path, encoding="utf-8"))
    lab = []
    for line in open(os.path.join(ROOT, "results", "preference", "labels.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("grade") in (2, 3):
                lab.append(r["pca"])
    A = np.array(lab)
    print(f"  {len(A)} labelled 优+良 samples")
    mu = A.mean(0)
    C = np.cov(A.T)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    ev = np.clip(ev, 0, None)
    frac = ev / ev.sum()
    print(f"  variance explained by the leading components of the preferred cloud:")
    cum = np.cumsum(frac)
    for i in range(min(8, len(frac))):
        print(f"    dim {i+1}: {frac[i]*100:5.1f}%   cumulative {cum[i]*100:5.1f}%")
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    print(f"  -> {k95} dimensions capture 95% of the variation among the preferred faces")

    # the reward weights: which axes actually drive the score?
    w = np.array(R["w"])
    print()
    print(f"  reward-model weights (|w| in descending order):")
    o = np.argsort(-np.abs(w))
    for i in o[:8]:
        print(f"    PC{i:<2d}  {w[i]:+.4f}")
    print(f"  norm of the reward weight vector: {np.linalg.norm(w):.3f}")
    print(f"  (a mean sample scores ~{R['b']:.3f}; a 1-sd move along the strongest axis")
    print(f"   changes the score by {np.abs(w).max():.3f})")
else:
    print("  no reward model yet")

# =====================================================================================
print()
print("=" * 100)
print("D. WHAT EACH INPUT STRATEGY WOULD COST")
print("=" * 100)
MODEL_MULT = 202240
print(f"  the model itself: {MODEL_MULT:,} multiplies  (the reference)")
print()
print(f"  {'strategy':34s} {'multiplies':>12s} {'random values needed':>22s}")
print("  " + "-" * 72)
strategies = [
    ("fixed preset (no randomness at all)", 0, 0),
    ("22 uniform randoms (approximate)", 22, 22),
    ("22 Gaussian randoms (sum of 8 uniforms)", 22 * 8, 22),
    ("k randoms, k from measurement B", None, None),
    ("preference filter (600 candidates x 22)", 600 * 22, 600 * 22),
]
for name, m, rv in strategies:
    if m is None:
        continue
    print(f"  {name:34s} {m:12,d} {rv:22d}")
print()
print("  Note the preference filter costs 13,200 multiplies: only 6.5% of the model.")
print("  So the input side is NOT the bottleneck while the model is this large. It only")
print("  becomes significant if the model is shrunk, which is the plan.")

json.dump({"influence": inf, "diversity": diversity,
           "model_multiplies": MODEL_MULT,
           "preference_filter_multiplies": 600 * 22},
          open(os.path.join(OUT, "input_cost.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'input_cost.json')}")
print("INPUT_COST_DONE")

#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Why the output must be 4x8x16 (a distribution per position) and not 4x8 (one value).

The claim to test: the conditional distribution of a pixel, given its context, is
NOT a single sharp value -- it is multi-modal. If so, one number per position is
provably unable to represent the data, no matter how well it is trained.

We measure, on the real quantized-grayscale data:
  * the empirical entropy of p(pixel | its 3x3 neighbourhood)   -> how uncertain is it?
  * how often that distribution has >=2 modes with real mass   -> is it multi-modal?
  * what a single-number (conditional mean) prediction does to such a position

Relative paths only.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results", "L0_probe")
os.makedirs(RES, exist_ok=True)

rgb = np.load(os.path.join(DATA, "faces8_rgb_v2.npy"))
rgbi = np.load(os.path.join(DATA, "faces8_v2.npy"))
pal = np.load(os.path.join(DATA, "palette_avatar16_v2.npy")).astype(np.float64)
LUM = np.array([0.299, 0.587, 0.114])
pl = (pal * LUM).sum(1)

# ---- 16-level gray, per-position, on the symmetric subset --------------------
N = len(rgb)
g = (rgb.astype(np.float64) * LUM).sum(-1)
LEVELS = np.linspace(0, 255, 16)
gi = np.abs(g[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)   # (N,8,8) 0..15

LEFT = [(r, c) for r in range(8) for c in range(4)]
RIGHT = [(r, 7 - c) for r in range(8) for c in range(4)]
_lr = np.array([r for r, _ in LEFT]); _lc = np.array([c for _, c in LEFT])
_rr = np.array([r for r, _ in RIGHT]); _rc = np.array([c for _, c in RIGHT])
is_sym = (gi[:, _lr, _lc] == gi[:, _rr, _rc]).mean(1) >= 0.95
sub = gi[is_sym]
print(f"symmetric subset: {len(sub):,} avatars")

# ---- empirical conditional distribution p(pixel | 3x3 neighbourhood) ---------
# Quantize the neighbourhood into a discrete context so the conditional is countable.
# 8 neighbour pixels x 16 levels is too many buckets, so use a coarse context:
# (sum of the 8 neighbours, discretized) + position. This is a *coarse* context, which
# makes the measured conditional ENTROPY an UPPER bound on the true one (coarser
# context = more residual uncertainty). Stated honestly below.
pos_list = LEFT
H_pos, H_all = [], []
multimodal = 0
total = 0
example = None

for k, (r, c) in enumerate(pos_list):
    # neighbour values (wrap-safe with edge padding)
    rr = np.clip(r + np.array([-1, -1, -1, 0, 0, 1, 1, 1]), 0, 7)
    cc = np.clip(c + np.array([-1, 0, 1, -1, 1, -1, 0, 1]), 0, 7)
    nb = sub[:, rr, cc]                                  # (n,8)
    ctx = nb.sum(1)                                      # 0..120
    ctx_bin = (ctx // 8).astype(int)                     # 16 bins

    ent_sum, ent_n = 0.0, 0
    for b in np.unique(ctx_bin):
        vals = sub[ctx_bin == b, r, c]
        if len(vals) < 50:
            continue
        p = np.bincount(vals, minlength=16) / len(vals)
        p = p[p > 0]
        H = float(-(p * np.log2(p)).sum())
        ent_sum += H * len(vals)
        ent_n += len(vals)
        # multi-modal? count modes holding >=20% mass each
        q = np.bincount(vals, minlength=16) / len(vals)
        modes = int((q >= 0.20).sum())
        multimodal += int(modes >= 2) * len(vals)
        total += len(vals)
        if example is None and k == 0 and modes >= 2:
            example = (r, c, int(b), q.tolist())
    H_pos.append(ent_sum / max(ent_n, 1))
    H_all.extend([ent_sum / max(ent_n, 1)] * ent_n)

H_pos = np.array(H_pos)
print()
print("conditional entropy H(pixel | coarse neighbourhood, position):")
print(f"  mean over the 32 positions : {H_pos.mean():.3f} bits   (max possible 4.0)")
print(f"  min / max over positions   : {H_pos.min():.3f} / {H_pos.max():.3f} bits")
print()
print(f"positions where the conditional has >=2 modes with >=20% mass: {100*multimodal/max(total,1):.2f}% of cases")

# ---- what the conditional MEAN does -----------------------------------------
print()
print("=" * 74)
print("What a single-number output predicts, versus what the data contains")
print("=" * 74)
# global: the mean level is a value that may barely occur
allv = sub.reshape(-1)
p_all = np.bincount(allv, minlength=16) / len(allv)
mean_lvl = float((p_all * np.arange(16)).sum())
print(f"  data: mean gray level = {mean_lvl:.2f}  (level {int(round(mean_lvl))} holds "
      f"{100*p_all[int(round(mean_lvl))]:.2f}% of pixels)")
print(f"  a conditional-mean predictor outputs ~{mean_lvl:.2f} for a typical position,")
print(f"  i.e. level {int(round(mean_lvl))} -- an intermediate gray, while the true")
print(f"  distribution puts most of its mass at the extremes.")

# entropy of the marginal, for reference
pm = p_all[p_all > 0]
print(f"\n  marginal entropy H(pixel) = {float(-(pm*np.log2(pm)).sum()):.3f} bits")
print(f"  mean conditional entropy  = {H_pos.mean():.3f} bits")
print(f"  -> the context removes only {float(-(pm*np.log2(pm)).sum()) - H_pos.mean():.3f} bits;")
print(f"     the remaining {H_pos.mean():.3f} bits of uncertainty is REAL and must be")
print(f"     represented by the model, not averaged away.")

# ---- worked example ----------------------------------------------------------
if example:
    r, c, b, q = example
    print()
    print("=" * 74)
    print(f"Worked example: position (row {r}, col {c}), neighbourhood-sum bin {b}")
    print("=" * 74)
    print("  empirical p(level) over that context:")
    for k in range(16):
        if q[k] > 0.005:
            print(f"    level {k:2d} (gray {LEVELS[k]:5.1f})  {q[k]*100:5.1f}%  {'#'*int(q[k]*60)}")
    mean_here = float(sum(k * q[k] for k in range(16)))
    top = int(np.argmax(q))
    print(f"  -> conditional mean would output {mean_here:.2f} (level ~{mean_here:.1f})")
    print(f"  -> the most likely single level is {top} ({q[top]*100:.1f}%)")
    print(f"  -> but level {top} itself only covers {q[top]*100:.1f}% of the cases")
    print("  -> a single number CANNOT express this; a 16-way distribution can.")

json.dump({
    "subset_n": int(len(sub)),
    "H_position_mean_bits": float(H_pos.mean()),
    "H_position_min": float(H_pos.min()),
    "H_position_max": float(H_pos.max()),
    "H_marginal_bits": float(-(pm * np.log2(pm)).sum()),
    "multimodal_share": float(multimodal / max(total, 1)),
    "note": "context is deliberately coarse (sum of 8 neighbours, 16 bins), so these "
            "entropies are UPPER bounds on the true conditional entropy",
}, open(os.path.join(RES, "why_distribution.json"), "w"), indent=2)
print("\nWROTE why_distribution.json")
print("WHY_DONE")

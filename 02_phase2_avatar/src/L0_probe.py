#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
L0 feasibility: "learn nearly-symmetric GRAYSCALE avatars, generate perfectly symmetric ones".

Answers, with numbers:
  Q1 Does grayscale + 16-level quantization make symmetry MORE common or less?
  Q2 How big is the usable nearly-symmetric subset at several symmetry thresholds?
  Q3 How much does forcing perfect symmetry cost (per-pixel flip rate)?
  Q4 Do the full 16 gray levels actually get used? (over-provisioned or not?)
  Q5 What is the effective output space size?
  Q6 Does the whole thing still look like a face? -> writes preview PNGs

Everything is RELATIVE to this script. D: is never touched; no existing file is modified.
"""
import os, json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results", "L0_probe")
os.makedirs(RES, exist_ok=True)

rgb = np.load(os.path.join(DATA, "faces8_rgb_v2.npy"))     # (N,8,8,3) uint8
N = len(rgb)
print(f"input: {rgb.shape}  N={N:,}")

# ---- luminance (perceptual Rec.601), then 16 uniform levels ------------------
LUM_W = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM_W).sum(-1)            # (N,8,8) float 0..255
print(f"luminance range: {gray.min():.1f} .. {gray.max():.1f}")

LEVELS = np.linspace(0, 255, 16)                            # 16 uniform gray levels
gi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1).astype(np.uint8)  # (N,8,8)

# reconstruction error of the 16-level grayscale representation itself
rec = LEVELS[gi]
g_err = ((rec - gray) ** 2).mean()
print(f"16-level gray quantization: per-pixel MSE {g_err:.2f}, RMS {np.sqrt(g_err):.2f}/255")

# ---- mirror structure --------------------------------------------------------
# pairs within a row: (r,c) <-> (r,7-c); the left half is (r,c) for c in 0..3
LEFT = np.array([(r * 8 + c) for r in range(8) for c in range(4)])
RIGHT = np.array([(r * 8 + (7 - c)) for r in range(8) for c in range(4)])

g2 = gi.reshape(N, 64)
L = g2[:, LEFT]
R = g2[:, RIGHT]

pairs_eq = (L == R)                                    # (N,32) bool
frac_eq = pairs_eq.mean(1)                             # per-avatar fraction of equal pairs
print(f"\ngray mirror pairs equal (all avatars): {pairs_eq.mean()*100:.2f}%")
print(f"gray fully-symmetric avatars        : {frac_eq.mean() and (frac_eq==1).mean()*100:.2f}%"
      f"  ({int((frac_eq==1).sum()):,})")

# compare with the colour version, to answer Q1
r_col = np.load(os.path.join(DATA, "faces8_v2.npy")).reshape(N, 64)
col_eq = (r_col[:, LEFT] == r_col[:, RIGHT])
print(f"colour mirror pairs equal           : {col_eq.mean()*100:.2f}%")
print(f"colour fully-symmetric avatars      : {(col_eq.mean(1)==1).mean()*100:.2f}%")
print("  -> grayscale+quantization " +
      ("INCREASES" if pairs_eq.mean() > col_eq.mean() else "DECREASES") + " symmetry")

# ---- usable subsets at several thresholds -----------------------------------
print("\n" + "=" * 74)
print("Q2/Q3  subset sizes and the cost of forcing perfect symmetry")
print("=" * 74)
print(f"{'threshold':>12} {'avatars kept':>13} {'share':>8} {'exact-sym projection flips':>28}")
forced = np.concatenate([L, L], axis=1)                # perfect mirror of the left half
# rebuild in (N,64) order: left half then its mirror
forced64 = np.empty((N, 64), dtype=np.uint8)
forced64[:, LEFT] = L
forced64[:, RIGHT] = L
flip = (forced64 != g2).mean(1)
rows = []
for t in (1.00, 0.99, 0.95, 0.90, 0.85, 0.75, 0.00):
    keep = frac_eq >= t
    n = int(keep.sum())
    fl = float(flip[keep].mean()) if n else float("nan")
    rows.append({"threshold": t, "kept": n, "share": n / N, "flip_rate": fl})
    print(f"{t:>12.2f} {n:>13,} {100*n/N:>7.2f}% {100*fl:>27.2f}%")

# ---- Q4: how many of the 16 levels are actually used? -----------------------
sym_mask = frac_eq >= 0.95
sub = g2[sym_mask]
print("\n" + "=" * 74)
print(f"Q4  gray level usage  (on the >=95% symmetric subset, n={len(sub):,})")
print("=" * 74)
counts = np.bincount(sub.reshape(-1), minlength=16)
share = counts / counts.sum()
for k in range(16):
    print(f"  level {k:2d} (value {LEVELS[k]:6.1f})  {100*share[k]:6.2f}%  {'#'*int(share[k]*200)}")
print(f"  levels with <0.1% share: {int((share<0.001).sum())}")
# distinct levels per avatar
dist = np.array([len(np.unique(s)) for s in sub[:20000]])
print(f"  distinct gray levels per avatar: mean {dist.mean():.2f}  max {dist.max()}")

# ---- Q5: output space -------------------------------------------------------
print("\n" + "=" * 74)
print("Q5  output space size")
print("=" * 74)
print(f"  half image (4x8, 16 levels) : 16^32 = 2^{32*4} = 2^{128}")
print(f"  full image (8x8, 16 levels) : 16^64 = 2^256")
print(f"  -> a feed-forward network can only ever reach a subset; the GAN must pick a good one")

# ---- Q6: previews ------------------------------------------------------------
def grid(imgs, cols=16, sc=8):
    rows_ = (len(imgs) + cols - 1) // cols
    cv = np.full((rows_ * 8, cols * 8), 40, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r_, c_ = divmod(i, cols)
        # Force uint8 + mode 'L'. Feeding PIL a float array silently creates mode 'F',
        # which PNG saving then rescales (0..255 -> 0..1) and the image comes out black.
        cv[r_*8:(r_+1)*8, c_*8:(c_+1)*8] = np.clip(np.asarray(im), 0, 255).astype(np.uint8)
    big = np.kron(cv, np.ones((sc, sc), dtype=np.uint8))
    for r_ in range(rows_ + 1):
        big[r_*8*sc:r_*8*sc+2, :] = 200
    for c_ in range(cols + 1):
        big[:, c_*8*sc:c_*8*sc+2] = 200
    return Image.fromarray(big, mode="L").convert("RGB")


rng = np.random.default_rng(0)
pick = rng.choice(np.flatnonzero(sym_mask), 64, replace=False)
# IMPORTANT: render the gray LEVELS, not the palette indices. Plotting an index array
# (0..15) as an 'L' image yields a near-black picture that looks like a data disaster.
grid([LEVELS[gi[i]] for i in pick]).save(os.path.join(RES, "gray_real_quantized.png"))
grid([gray[i] for i in pick]).save(os.path.join(RES, "gray_real_continuous.png"))
grid([LEVELS[forced64[i].reshape(8, 8)] for i in pick]).save(os.path.join(RES, "gray_forced_symmetric.png"))
print(f"\npreviews -> {RES}")
print("  gray_real_continuous.png   : true grayscale (no quantization)")
print("  gray_real_quantized.png    : 16-level grayscale")
print("  gray_forced_symmetric.png  : 16-level + perfect mirror (this is the training target)")

json.dump({
    "n_input": int(N),
    "gray_quant": {"per_pixel_mse": float(g_err), "rms": float(np.sqrt(g_err))},
    "symmetry": {
        "gray_pairs_equal": float(pairs_eq.mean()),
        "gray_fully_symmetric": float((frac_eq == 1).mean()),
        "colour_pairs_equal": float(col_eq.mean()),
        "colour_fully_symmetric": float((col_eq.mean(1) == 1).mean()),
    },
    "subsets": rows,
    "gray_levels_used": int((share > 0.001).sum()),
    "distinct_levels_per_avatar_mean": float(dist.mean()),
}, open(os.path.join(RES, "L0_probe.json"), "w"), indent=2)
print("L0_PROBE_DONE")

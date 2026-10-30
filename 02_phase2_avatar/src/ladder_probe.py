#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Answer three questions with numbers, for the "output ladder" plan.

Q1  What is the intrinsic dimensionality of an 8x8 face?
      -> PCA on RGB (continuous, well-posed) and on palette indices (discrete, indicative).
Q2  How much colour information survives at each rung of the ladder?
      -> 16-colour / RGB(3ch) / luminance-only(1ch) / 4x8 mirrored
Q3  Is a *learned* 3->16 or 1->16 mapper worth anything, versus an exact palette LUT?
      -> measure the exact nearest-palette rounding error, which is the LUT's error floor.

All paths are RELATIVE to this script (per the project rule: never touch D:, never
hardcode absolutes).
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # 02_phase2_avatar
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results", "ladder_probe")
os.makedirs(RES, exist_ok=True)

lab = np.load(os.path.join(DATA, "faces8_v2.npy"))            # (N,64) uint8 0..15
rgb = np.load(os.path.join(DATA, "faces8_rgb_v2.npy"))        # (N,8,8,3) uint8
pal = np.load(os.path.join(DATA, "palette_avatar16_v2.npy")).astype(np.int32)  # (16,3)
N = len(lab)
print(f"data: {lab.shape} rgb {rgb.shape} palette {pal.shape}  N={N:,}")

PAIRS = np.array([(r * 8 + c, r * 8 + (7 - c)) for r in range(8) for c in range(4)])

# ----------------------------------------------------------------- PCA
def pca_report(X, name, ks=(1, 2, 4, 8, 12, 16, 24, 32, 48, 64)):
    X = X - X.mean(0, keepdims=True)
    # economy SVD on the (n,d) matrix; d <= 192 so this is cheap
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = (S ** 2) / max(len(X) - 1, 1)
    ev = var / var.sum()
    cum = np.cumsum(ev)
    print(f"\n[{name}] total variance {var.sum():.1f}")
    for k in ks:
        if k <= len(cum):
            print(f"    k={k:3d}  explained {100*cum[k-1]:6.2f}%")
    # how many components for 90/95/99%
    for t in (0.90, 0.95, 0.99):
        k = int(np.searchsorted(cum, t) + 1)
        print(f"    -> {int(t*100)}% of variance needs k={k}")
    return ev

# RGB: use a subsample, it's continuous and well-posed
sub = np.random.default_rng(0).choice(N, 100000, replace=False)
ev_rgb = pca_report(rgb[sub].reshape(len(sub), -1).astype(np.float64), "RGB 8x8x3 = 192 dims")

# Palette indices: discrete, treat as ordered numbers (a heuristic, stated as such)
sub2 = np.random.default_rng(1).choice(N, 100000, replace=False)
ev_idx = pca_report(lab[sub2].astype(np.float64), "palette index 64 dims (heuristic)")

# Luminance only
lum_all = (rgb.astype(np.float64) * np.array([0.299, 0.587, 0.114])).sum(-1)  # (N,8,8)
ev_lum = pca_report(lum_all[sub].reshape(len(sub), -1), "luminance 8x8 = 64 dims")

# ----------------------------------------------------------------- ladder rungs
print("\n" + "=" * 76)
print("LADDER RUNGS: how much information each output keeps")
print("=" * 76)

def quantize_to_palette(X3):
    """Exact nearest-palette LUT: this IS what redstone would do."""
    d = ((X3[:, None, :].astype(np.int32) - pal[None, :, :]) ** 2).sum(-1)
    j = d.argmin(1)
    return j, float(d[np.arange(len(X3)), j].mean())

flat = rgb.reshape(N, -1, 3).reshape(-1, 3).astype(np.int32)

# rung 1: full 16-colour classification (the target)
print("\n[rung] 8x8x16  discrete palette classes")
print(f"    exact: reconstructs the data with 0 error (it IS the label set)")

# rung 2: 8x8x3 RGB (continuous)
print("\n[rung] 8x8x3  continuous RGB")
j_rgb, err_rgb = quantize_to_palette(flat)
print(f"    nearest-palette rounding: per-pixel SSE {err_rgb:.2f}, per-channel MSE {err_rgb/3:.2f}, RMS {np.sqrt(err_rgb/3):.2f}/255")
print(f"    discrete classes actually used after rounding: {len(np.unique(j_rgb))}/16")

# rung 3: 8x8 luminance only
print("\n[rung] 8x8  luminance only (1 channel)")
L = lum_all.reshape(-1, 1)
# the best a 1-D 'luminance -> palette' mapper can do: pick the palette entry with
# closest luminance (this is the LUT's floor for a 1-channel output)
pl = (pal.astype(np.float64) * np.array([0.299, 0.587, 0.114])).sum(-1)   # (16,)
nearest_l = np.abs(L - pl[None, :]).argmin(1)
err_lum = ((flat - pal[nearest_l]) ** 2).sum(1).mean()
print(f"    luminance->palette LUT: per-pixel SSE {err_lum:.2f}, per-channel MSE {err_lum/3:.2f}, RMS {np.sqrt(err_lum/3):.2f}/255")
print(f"    (compare: exact 3-channel LUT = {err_rgb:.2f} SSE)")
print(f"    -> keeping only luminance costs {err_lum/err_rgb:.2f}x the error")
print(f"    palette luminance spread: min {pl.min():.1f} max {pl.max():.1f}, "
      f"distinct-to-1-level pairs closer than 2: {int((np.abs(pl[:,None]-pl[None,:])<2).sum()//2)}")

# rung 4: 4x8 half face with mirroring
print("\n[rung] 4x8  left half + mirror")
le, ri = PAIRS[:, 0], PAIRS[:, 1]
identical = (lab[:, le] == lab[:, ri])
print(f"    mirror pairs exactly equal : {identical.mean()*100:.2f}%")
print(f"    avatars fully symmetric    : {identical.all(1).mean()*100:.2f}%")
pairs_ok = identical.mean(1)          # fraction of the 32 pairs equal, per avatar
for t in (1.0, 0.99, 0.95, 0.90, 0.80):
    keep = int((pairs_ok >= t).sum())
    # error introduced by forcing the left half onto the right
    forced = lab.copy(); forced[:, ri] = forced[:, le]
    err = float((forced != lab).mean())
    print(f"    keep avatars with >= {int(t*100):3d}% mirror pairs equal: {keep:>7,} ({100*keep/N:5.2f}%)   "
          f"mirror-forcing flips {100*err:5.2f}% of all pixels")

json.dump({
    "n": int(N),
    "pca": {"rgb_192": [float(x) for x in ev_rgb[:16]],
            "index_64": [float(x) for x in ev_idx[:16]],
            "lum_64": [float(x) for x in ev_lum[:16]]},
    "lut_error_per_pixel_sse": {"rgb3_exact": err_rgb, "luminance1": err_lum},
    "mirror": {"pairs_equal": float(identical.mean()),
               "fully_symmetric": float(identical.all(1).mean())},
}, open(os.path.join(RES, "ladder_probe.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(RES, 'ladder_probe.json')}")
print("LADDER_DONE")

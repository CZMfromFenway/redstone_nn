#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Colour feasibility: can saturation and hue be inferred from the lightness map?

THE PLAN BEING TESTED
The user proposed two ways to add colour:
  (a) fit lightness, saturation and hue directly from extra principal components
  (b) keep the existing lightness stage, then train a SECOND stage that maps the lightness
      map to saturation and hue

Option (b) is far cheaper -- it reuses the whole luminance pipeline, and in redstone it means
one more network fed by outputs that already exist. But it only works if saturation and hue
are actually predictable from lightness. That is what this measures: a linear readout from
the 32 lightness values to S and to the hue channels, plus the R^2 ceiling.

If R^2 is high, (b) is the way. If it is near zero, colour needs its own conditioning
information and (b) cannot work.
"""
import json
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "colour")
os.makedirs(OUT, exist_ok=True)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
LEVELS = np.linspace(0, 255, 16)

rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
print(f"rgb {rgb.shape}   canonical {full.shape}")

# align: the exact tie-down quantisation reproduces L0_full_idx row for row
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.clip(np.ceil(gray / 17.0 - 0.5), 0, 15).astype(np.int64)
kn = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
col = rgb[kn]
print(f"symmetry filter keeps {len(col):,} rows; canonical has {len(full):,}")
assert len(col) == len(full), "colour and luminance data are not aligned"
assert np.array_equal(np.clip(np.ceil(
    (col.astype(np.float64) * LUM).sum(-1) / 17.0 - 0.5), 0, 15).astype(np.int64), full), \
    "colour rows do not match the canonical luminance"
print("alignment verified: colour rows match the luminance rows exactly")

# ---------------------------------------------------------------- HSV
c = col.astype(np.float64) / 255.0
mx = c.max(-1); mn = c.min(-1)
v = mx
s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
r, g, b = c[..., 0], c[..., 1], c[..., 2]
delta = np.maximum(mx - mn, 1e-9)
h = np.zeros_like(mx)
mask = mx == r
h[mask] = ((g - b)[mask] / delta[mask]) % 6
mask = (mx == g) & (mx != r)
h[mask] = ((b - r)[mask] / delta[mask]) + 2
mask = (mx == b) & (mx != r) & (mx != g)
h[mask] = ((r - g)[mask] / delta[mask]) + 4
h = h / 6.0                       # 0..1
chroma = (mx - mn)

print()
print("=" * 100)
print("WHAT THE COLOUR DATA LOOKS LIKE")
print("=" * 100)
print(f"  value  V : mean {v.mean():.3f}  sd {v.std():.3f}")
print(f"  saturation S: mean {s.mean():.3f}  sd {s.std():.3f}  "
      f"share S<0.1 = {100*np.mean(s<0.1):.1f}%")
print(f"  chroma   : mean {chroma.mean():.3f}  sd {chroma.std():.3f}")
print(f"  share of pixels that are essentially grey (chroma<0.05): "
      f"{100*np.mean(chroma<0.05):.1f}%")

# ---------------------------------------------------------------- predictability
SL = full[:, layout.POS_ROW, layout.MIRROR_COL].astype(np.float64)   # (N,32) lightness
S32 = s[:, layout.POS_ROW, layout.MIRROR_COL]                        # (N,32) saturation
H32 = h[:, layout.POS_ROW, layout.MIRROR_COL]
CH32 = chroma[:, layout.POS_ROW, layout.MIRROR_COL]

n = 40000
rng = np.random.default_rng(0)
sub = rng.choice(len(SL), n, replace=False)
A = np.c_[SL[sub], np.ones(n)]


def r2(Xd, Y):
    W = np.linalg.lstsq(Xd, Y, rcond=None)[0]
    res = Y - Xd @ W
    return float(1 - (res ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


print()
print("=" * 100)
print("CAN SATURATION AND HUE BE READ OFF THE LIGHTNESS MAP?  (linear readout, R^2)")
print("=" * 100)
r2_s = r2(A, S32[sub])
r2_c = r2(A, CH32[sub])
r2_h_sin = r2(A, np.sin(2 * np.pi * H32[sub]))
r2_h_cos = r2(A, np.cos(2 * np.pi * H32[sub]))
print(f"  saturation S      R^2 = {r2_s:6.3f}")
print(f"  chroma (mx-mn)    R^2 = {r2_c:6.3f}")
print(f"  hue sin           R^2 = {r2_h_sin:6.3f}")
print(f"  hue cos           R^2 = {r2_h_cos:6.3f}")
print(f"  hue (both)        R^2 = {(r2_h_sin + r2_h_cos) / 2:6.3f}")
print()
print("  reading: R^2 near 0 means lightness says nothing about that channel, so a second")
print("  stage taking only the lightness map could not produce it.")

# how much of the colour variance is explained by the SOURCE rgb, i.e. is it even stable?
print()
print("=" * 100)
print("IS COLOUR LOW-RANK? (how many components does the colour field need)")
print("=" * 100)
M = np.concatenate([S32[sub], CH32[sub], (H32[sub] - 0.5) * 2], axis=1)
Mu = M - M.mean(0)
U, Sg, Vt = np.linalg.svd(Mu, full_matrices=False)
ev = Sg ** 2 / (Sg ** 2).sum()
for k in (4, 8, 16, 22, 32):
    print(f"    first {k:2d} components explain {100*ev[:k].sum():5.1f}% of colour variance")

json.dump({"r2_sat": r2_s, "r2_chroma": r2_c, "r2_hue": (r2_h_sin + r2_h_cos) / 2,
           "sat_mean": float(s.mean()), "grey_share": float(np.mean(chroma < 0.05)),
           "colour_pca": [float(x) for x in ev[:32]]},
          open(os.path.join(OUT, "colour_feasibility.json"), "w"), indent=2)
print(f"\nwrote {os.path.join(OUT, 'colour_feasibility.json')}")
print("COLOUR_FEASIBILITY_DONE")

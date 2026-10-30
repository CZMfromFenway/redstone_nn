#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Colour stage 2: how many colour components are needed, and does the greyscale map help?

THE DESIGN THE USER PROPOSED
  stage 1 : 4 greyscale PCs -> 32-level greyscale avatar        (already built, 74k mults)
  stage 2 : [greyscale avatar + a few colour PCs] -> saturation and hue
The earlier finding was that greyscale ALONE explains almost nothing about colour
(R^2 0.06-0.11). The point of stage 2 is that it also receives colour PCs, so the question
changes to:

    how many colour components are enough, and does adding the greyscale map on top of them
    buy anything?

If (greyscale + colour PCs) is no better than (colour PCs alone), then the greyscale input is
dead weight and stage 2 can be fed colour components directly -- which is cheaper and removes
the 32-wide dependency between the stages.
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

rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.clip(np.ceil(gray / 17.0 - 0.5), 0, 15).astype(np.int64)
kn = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
col = rgb[kn].astype(np.float64) / 255.0
assert len(col) == len(full)
print(f"aligned colour rows: {len(col):,}")

# ---- colour PCA on the 192-dim RGB image (64 pixels x 3)
Xr = col.reshape(len(col), -1)
n = 60000
sub = np.random.default_rng(0).choice(len(Xr), n, replace=False)
mu = Xr[sub].mean(0)
U, S, Vt = np.linalg.svd(Xr[sub] - mu, full_matrices=False)
ev = S ** 2 / (S ** 2).sum()
CPC = Vt[:32]
CSD = ((Xr[sub] - mu) @ CPC.T).std(0) + 1e-9
print(f"colour PCA: first 8 components explain {100*ev[:8].sum():.1f}%, "
      f"16 -> {100*ev[:16].sum():.1f}%, 22 -> {100*ev[:22].sum():.1f}%")

# ---- targets: saturation and hue of the 32 canonical positions
c = col
mx = c.max(-1); mn = c.min(-1)
s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
r_, g_, b_ = c[..., 0], c[..., 1], c[..., 2]
delta = np.maximum(mx - mn, 1e-9)
h = np.zeros_like(mx)
m1 = mx == r_; h[m1] = ((g_ - b_)[m1] / delta[m1]) % 6
m2 = (mx == g_) & ~m1; h[m2] = ((b_ - r_)[m2] / delta[m2]) + 2
m3 = (mx == b_) & ~m1 & ~m2; h[m3] = ((r_ - g_)[m3] / delta[m3]) + 4
h = h / 6.0

SL = full[:, layout.POS_ROW, layout.MIRROR_COL].astype(np.float64)     # (N,32)
S32 = s[:, layout.POS_ROW, layout.MIRROR_COL]
H32 = h[:, layout.POS_ROW, layout.MIRROR_COL]

nn = 30000
bs = np.random.default_rng(1).choice(len(Xr), nn, replace=False)
CF = ((Xr[bs] - mu) @ CPC.T) / CSD      # (nn,32) standardized colour components
G = SL[bs] / 15.0                        # (nn,32) greyscale 0..1


def r2(A, Y):
    W = np.linalg.lstsq(A, Y, rcond=None)[0]
    res = Y - A @ W
    return float(1 - (res ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def add_one(A):
    return np.c_[A, np.ones(len(A))]


Y_S = S32[bs]
Y_Hs = np.sin(2 * np.pi * H32[bs])
Y_Hc = np.cos(2 * np.pi * H32[bs])

print()
print("=" * 104)
print("STAGE 2: PREDICTING SATURATION AND HUE  (linear readout, R^2 on a held-out-ish sample)")
print("=" * 104)
print(f"  {'inputs':34s} {'sat R^2':>9s} {'hue sin':>9s} {'hue cos':>9s} {'hue avg':>9s}")
print("  " + "-" * 76)


def show(label, A):
    a = add_one(A)
    rs = r2(a, Y_S)
    rh1 = r2(a, Y_Hs); rh2 = r2(a, Y_Hc)
    print(f"  {label:34s} {rs:9.3f} {rh1:9.3f} {rh2:9.3f} {(rh1+rh2)/2:9.3f}")
    return {"sat": rs, "hue": (rh1 + rh2) / 2}


out = {}
out["greyscale only (32)"] = show("灰阶图 32 维（无颜色信息）", G)
for k in (4, 8, 12, 16, 22):
    out[f"colour {k}"] = show(f"颜色 PCA {k} 维", CF[:, :k])
for k in (4, 8, 12, 16):
    out[f"grey+colour {k}"] = show(f"灰阶 32 + 颜色 {k} 维", np.c_[G, CF[:, :k]])

print()
print("  reading: if 'grey+colour k' is no better than 'colour k', the greyscale map adds")
print("  nothing once colour components are present, and stage 2 should take colour PCs only.")

json.dump(out, open(os.path.join(OUT, "stage2_readout.json"), "w"), indent=2,
          ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'stage2_readout.json')}")
print("STAGE2_DONE")

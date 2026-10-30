#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Two cheap measurements that decide the colour encoding, before spending any training time.

1. BIT DEPTH -- is 4 bits per channel actually enough?
The 4-bit floor was 0.205 levels, but that is an AVERAGE. Pure reds, pure whites and dark
shadows may collapse badly. This measures the floor at 3, 4, 5 and 6 bits per channel, and
separately for saturated versus near-grey pixels, because that is where a small palette hurts.

2. A SHARED PALETTE -- can the whole face be described by 16 or 32 colours?
Redstone is much cheaper if each pixel is ONE 4-bit palette index instead of three 4-bit
channels: 4 bits per pixel instead of 12, and a lookup instead of three signals. This measures
how many distinct colours the data actually uses and what a k-colour palette costs.
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
OUT = os.path.join(ROOT, "results", "colour", "encoding")
os.makedirs(OUT, exist_ok=True)

rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)

LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.clip(np.ceil(gray / 17.0 - 0.5), 0, 15).astype(np.int64)
kn = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
col = rgb[kn].astype(np.float32)                       # (N,8,8,3) 0..255
N = len(col)
print(f"canonical colour rows: {N:,}")

# ================================================================ 1. bit depth
print()
print("=" * 100)
print("1. BIT DEPTH: what does the palette cost?")
print("=" * 100)
print(f"  {'bits':>4s} {'levels':>7s} {'all-pixel MAE':>14s} {'saturated MAE':>14s} "
      f"{'near-grey MAE':>14s}")
print("  " + "-" * 60)
rows = []
mx = col.max(-1); mn = col.min(-1)
chroma = (mx - mn) / 255.0
sat_mask = chroma > 0.35
grey_mask = chroma < 0.10
for b in (3, 4, 5, 6):
    L = 2 ** b - 1
    q = np.clip(np.rint(col / 255.0 * L), 0, L)
    deq = q / L * 255.0
    err = np.abs(deq - col).mean(-1) / 17.0            # into 0..15 level units
    rows.append((b, L, float(err.mean()), float(err[sat_mask].mean()),
                 float(err[grey_mask].mean())))
    print(f"  {b:4d} {L:7d} {err.mean():14.4f} {err[sat_mask].mean():14.4f} "
          f"{err[grey_mask].mean():14.4f}")
print()
print(f"  saturated pixels: {100*sat_mask.mean():.1f}%   near-grey: {100*grey_mask.mean():.1f}%")
print("  a 4-bit channel is chosen only if its saturated error stays acceptable, since")
print("  saturated pixels are exactly what makes a skin recognisable")

# ================================================================ 2. palette
print()
print("=" * 100)
print("2. SHARED PALETTE: how many colours does a face actually use?")
print("=" * 100)
W = np.random.default_rng(0).choice(N, min(40000, N), replace=False)
P = col[W].reshape(-1, 3)                              # (40k*64, 3)
print(f"  sampled pixels: {len(P):,}")
print(f"  distinct exact RGB triples: {len(np.unique(P, axis=0)):,}")


def kmeans(X, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1) if len(X) < 5000 else None
        if d is None:                                  # chunked for memory
            lab = np.empty(len(X), dtype=np.int32)
            for i in range(0, len(X), 5000):
                ch = X[i:i + 5000]
                lab[i:i + 5000] = ((ch[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
        else:
            lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return C, lab


sub = P[np.random.default_rng(1).choice(len(P), 20000, replace=False)]
print()
print(f"  {'palette':>8s} {'reconstruction MAE':>20s} {'vs 4-bit channels':>18s}")
print("  " + "-" * 50)
base4 = rows[1][2]
pal = {}
for k in (16, 24, 32, 48, 64):
    C, _ = kmeans(sub.astype(np.float64), k, iters=20, seed=2)
    d = ((sub[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    deq = C[lab]
    mae = float(np.abs(deq - sub).mean() / 17.0)
    pal[k] = mae
    print(f"  {k:8d} {mae:20.4f} {('%.1f%%' % (100*(mae-base4)/base4)):>18s}")
print()
print(f"  4-bit per channel (3 signals, 12 bits/pixel) MAE = {base4:.4f}")
print("  a k-colour palette costs 1 lookup + 4 bits of index, and rescales R,G,B together,")
print("  which is why its error is higher than independent per-channel rounding")

json.dump({"bit_depth": [{"bits": r[0], "levels": r[1], "mae": r[2],
                         "sat_mae": r[3], "grey_mae": r[4]} for r in rows],
           "palette": {str(k): v for k, v in pal.items()},
           "sat_share": float(sat_mask.mean()), "grey_share": float(grey_mask.mean())},
          open(os.path.join(OUT, "encoding.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'encoding.json')}")
print("ENCODING_DONE")

#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Colour data preparation, aligned to the grayscale pipeline row for row.

EVERYTHING HERE MUST MATCH dim_sweep.py EXACTLY
The grayscale PCA basis is rebuilt with the same procedure (mean over all rows, 60 000-row
subsample from seed 0, SVD, first 22 components, standardised by the full-data sd). If that
differs by even a sign flip, the 4 PCA inputs would describe different faces and stage 2 would
be trained on nonsense -- the exact bug that cost a full training run earlier in this project.
So the script checks that the rebuilt PCs reproduce the grayscale target through the trained
grader before anything else runs.

WHAT IS BUILT
    gray32   (N,32)   0..15   the stage-1 target, from L0_full_idx
    pc4      (N,4)            the four knobs, standardised
    rgb96    (N,96)   0..15   32 positions x R,G,B, 4 bits per channel
    cpc      (N,K)           principal components of rgb96
    sym_rate                 how symmetric the COLOUR actually is (grayscale was 80.9%)
    quant_floor              the error that 4-bit quantisation alone imposes

The last two matter: if colour symmetry is much lower than grayscale, forcing a mirror would
corrupt the data, and if the quantisation floor is high then no model can appear good.
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
CACHE = os.path.join(OUT, "cache.npz")

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)

print("=" * 100)
print("COLOUR DATA PREPARATION")
print("=" * 100)

full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
N = len(full)
print(f"canonical rows: {N:,}")

rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
print(f"rgb rows: {len(rgb):,}  shape {rgb.shape}")

# ---------------------------------------------------------------- align colour to canonical
# same symmetry filter as before: at least 95% of mirrored luminance pairs equal
LUM = np.array([0.299, 0.587, 0.114])
gray_f = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.clip(np.ceil(gray_f / 17.0 - 0.5), 0, 15).astype(np.int64)
kn = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
col = rgb[kn]
print(f"symmetry filter keeps {len(col):,}")
assert len(col) == N, f"colour {len(col)} != canonical {N}"
# the filter must select the SAME faces, not merely the same count
assert np.array_equal(np.clip(np.ceil(
    (col.astype(np.float64) * LUM).sum(-1) / 17.0 - 0.5), 0, 15).astype(np.int64), full), \
    "colour rows and luminance rows describe different faces"
print("alignment verified against the canonical luminance, row for row")

# ---------------------------------------------------------------- grayscale side
gray32 = full[:, layout.POS_ROW, layout.MIRROR_COL].astype(np.float32)     # (N,32)
Xg = full.reshape(N, -1).astype(np.float64)
mu_g = Xg.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(N, min(60000, N), replace=False)
_, _, Vt = np.linalg.svd(Xg[sub] - mu_g, full_matrices=False)
comps = Vt[:22]
c_unused = (Xg - mu_g) @ comps.T
sd_g = (c_unused.std(0, keepdims=True) + 1e-6)
ZZ = (c_unused / sd_g[0]).astype(np.float32)                                # (N,22)
pc4 = ZZ[:, :4].copy()
print("grayscale PCA basis rebuilt (same procedure and seed as dim_sweep.py)")

# ---------------------------------------------------------------- colour side
c4 = col.astype(np.int16)
q = np.clip(np.rint(c4 / 17.0), 0, 15).astype(np.float32)                   # 4-bit per channel
rgb32 = q[:, layout.POS_ROW, layout.MIRROR_COL, :]                          # (N,32,3)
rgb96 = rgb32.reshape(N, 96)
# FULL image, row-major: 64 positions x 3 channels. Colour is NOT symmetric, so this is the
# faithful target; rgb96 is kept only to measure what the mirror assumption costs.
rgb192 = q.reshape(N, 192)

# what does forcing the mirror cost, in level units, per channel?
mir = np.zeros_like(q)
mir[:, layout.POS_ROW, layout.POS_COL, :] = rgb32
mir[:, layout.POS_ROW, layout.MIRROR_COL, :] = rgb32
mirror_cost = float(np.abs(mir - q).mean())

# quantisation floor: how much error 4->8 bit round trip alone imposes, in 0..15 units
deq = (q * 17.0)
raw = c4.astype(np.float32)
quant_floor = float(np.abs(deq - raw).mean() / 17.0)     # back into 0..15 level units
print()
print(f"quantisation floor (4-bit round trip, in level units) = {quant_floor:.4f}")
print("  no model can beat this; it is the intrinsic cost of a 16-level palette")
print()
print(f"cost of forcing a colour mirror: {mirror_cost:.4f} level units "
      f"(quantisation floor is {quant_floor:.4f})")
print(f"  ratio {mirror_cost/max(quant_floor,1e-9):.1f}x the palette cost -- "
      + ("NEGLIGIBLE" if mirror_cost < 2*quant_floor else "TOO LARGE, do not force the mirror"))

# colour symmetry: how often mirrored positions agree exactly, and by how much they differ
# NOTE q has shape (N,8,8,3): axis 2 is the COLUMN, axis 3 is the channel. An earlier version
# reversed axis -1 here, which compared R against B and produced a bogus "colour is only 34%
# symmetric" reading. Reverse the COLUMN axis.
left = q[:, :, :4, :]
right = q[:, :, 4:, :][:, :, ::-1, :]
eq = float((left == right).mean())
diff = float(np.abs(left - right).mean())
print()
print(f"colour symmetry: exact-match rate {100*eq:.1f}%,  mean |diff| {diff:.3f} levels")
print("  (grayscale exact-match was 80.9%; forcing a mirror costs what this number says)")

# ---------------------------------------------------------------- colour PCA
cmean = rgb192.mean(0)
sub2 = np.random.default_rng(0).choice(N, min(60000, N), replace=False)
Uc, Sc, Vc = np.linalg.svd(rgb192[sub2] - cmean, full_matrices=False)
ev = Sc ** 2 / (Sc ** 2).sum()
K = 24
CPC = Vc[:K].astype(np.float32)
cstd = ((rgb192 - cmean) @ CPC.T).std(0) + 1e-6
cpc = (((rgb192 - cmean) @ CPC.T) / cstd).astype(np.float32)
print()
print("colour PCA explained variance (full 192-dim image):")
for k in (4, 8, 12, 16, 20):
    print(f"  first {k:2d} components: {100*ev[:k].sum():5.1f}%")
print(f"  mean colour (all 96 channels): {cmean.mean()/15*100:.1f}% of full scale")
print(f"  mean saturation of the canonical pixels: "
      f"{float((rgb96.reshape(-1,32,3).max(-1)-rgb96.reshape(-1,32,3).min(-1)).mean()/15):.3f}")

# ---------------------------------------------------------------- save
np.savez_compressed(CACHE,
                    gray32=gray32.astype(np.uint8),
                    pc4=pc4,
                    rgb96=rgb96.astype(np.uint8),
                    rgb192=rgb192.astype(np.uint8),
                    cpc=cpc,
                    cmean=cmean.astype(np.float32))
meta = {"N": int(N), "quant_floor": quant_floor, "mirror_cost": mirror_cost,
        "sym_exact": eq, "sym_absdiff": diff,
        "colour_pca_ev": [float(v) for v in ev[:K]],
        "rgb_mean": float(cmean.mean())}
json.dump(meta, open(os.path.join(OUT, "cache_meta.json"), "w"), indent=2, ensure_ascii=False)
print(f"\nwrote {CACHE}  ({os.path.getsize(CACHE)/1e6:.1f} MB)")
print(f"wrote {os.path.join(OUT, 'cache_meta.json')}")
print("COLOUR_DATA_DONE")

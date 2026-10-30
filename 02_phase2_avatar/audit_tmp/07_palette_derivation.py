#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit step 3: independently re-derive the concrete/wool/terracotta 16-colour
palettes from the local Minecraft textures (same rule as build_palettes.py) and
compare with data/palettes.json.  Also check the 48-colour interleave."""
import os, json, collections, time
import numpy as np
from PIL import Image

ROOT = r"E:\redstone_nn\02_phase2_avatar\archive_phase1_attempts"
DATA = os.path.join(ROOT, "data")
MC = r"D:\czm我的世界\.minecraft"
ORDER16 = ['white', 'light_gray', 'gray', 'black', 'brown', 'red', 'orange', 'yellow',
           'lime', 'green', 'cyan', 'light_blue', 'blue', 'purple', 'magenta', 'pink']

res = {f: {} for f in ("concrete", "wool", "terracotta")}
path_of = {}
t0 = time.time()
n = 0
for dp, dn, fn in os.walk(MC):
    for f in fn:
        if not f.endswith(".png"):
            continue
        n += 1
        base = f[:-4]
        for nm in ORDER16:
            for fam in ("concrete", "wool", "terracotta"):
                if base in (f"{nm}_{fam}", f"{nm}_{fam}_top") and nm not in res[fam]:
                    try:
                        im = Image.open(os.path.join(dp, f)).convert("RGB")
                        a = np.array(im).reshape(-1, 3)
                        v, c = np.unique(a, axis=0, return_counts=True)
                        res[fam][nm] = v[c.argmax()].tolist()
                        path_of[(fam, nm)] = os.path.join(dp, f)
                    except Exception:
                        pass
print(f"walked {n} PNGs under {MC} in {time.time()-t0:.0f}s")
for fam in res:
    print(f"  {fam}: found {len(res[fam])}/16")

J = json.load(open(os.path.join(DATA, "palettes.json")))
pal16 = [res["concrete"][k] for k in ORDER16 if k in res["concrete"]]
pal48 = [res[fam][k] for k in ORDER16 for fam in ("concrete", "wool", "terracotta") if k in res[fam]]
print(f"\nre-derived pal16 len={len(pal16)}  matches palettes.json pal16: {pal16 == J['pal16']}")
for a, b, k in zip(pal16, J['pal16'], ORDER16):
    if a != b:
        print(f"   MISMATCH {k}: re-derived {a} vs json {b}")
print(f"re-derived pal48 len={len(pal48)}  matches palettes.json pal48: {pal48 == J['pal48']}")
print(f"by_family matches: "
      f"{ {f: {k: v for k, v in res[f].items()} == J['by_family'][f] for f in res} }")
print("\nsource texture for a few concrete colours:")
for k in ("white", "black", "lime", "purple"):
    print(f"   {k}_concrete -> {path_of.get(('concrete', k))} = {res['concrete'].get(k)}")

print("\n### 16-colour custom palette derived in custom_palette.py")
PA = json.load(open(os.path.join(DATA, "palette_avatar16.json")))
P = np.array(PA["palette_ordered"])
print(f"  len={len(P)} distinct rows={len(np.unique(P, axis=0))}")
print(f"  names: {PA['names']}")
print(f"  stored quant_err={PA['quant_err']:.3f} free_kmeans={PA['free_kmeans_err']:.3f} mc16={PA['mc16_err']:.3f}")
rgb = np.load(os.path.join(DATA, "faces8_rgb.npy")).astype(np.int64)
F = rgb.reshape(-1, 3)
e_custom = float(((F[:, None] - P[None]) ** 2).sum(2).min(1).mean())
e_mc16 = float(((F[:, None] - np.array(J['pal16'], dtype=np.int64)[None]) ** 2).sum(2).min(1).mean())
e_pal48 = float(((F[:, None] - np.array(J['pal48'], dtype=np.int64)[None]) ** 2).sum(2).min(1).mean())
print(f"  recomputed on the CURRENT 50000-sample faces8_rgb (per-pixel squared err, sum over 3 ch):")
print(f"    custom16 = {e_custom:.2f} (per-channel MSE {e_custom/3:.2f})")
print(f"    mc16     = {e_mc16:.2f}   ratio mc16/custom = {e_mc16/e_custom:.2f}")
print(f"    pal48    = {e_pal48:.2f}   ratio pal48/custom = {e_pal48/e_custom:.2f}")
print(f"  custom palette is the exact 16-colour Voronoi fit? recompute MSE with k-means(16) on current data:")
rng = np.random.default_rng(1)
cen = F[rng.choice(len(F), 16, replace=False)].copy()
for _ in range(25):
    lb = ((F[:, None] - cen[None]) ** 2).sum(2).argmin(1)
    for kk in range(16):
        m = lb == kk
        if m.sum():
            cen[kk] = F[m].mean(0)
e_km = float(((F - cen[lb]) ** 2).sum(1).mean())
print(f"    free k-means16 = {e_km:.2f} (per-channel MSE {e_km/3:.2f}); custom/optimal = {e_custom/e_km:.3f}")

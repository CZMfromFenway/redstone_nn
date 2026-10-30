#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit step 3c/6: palette-file cross-checks, MC16-vs-custom labelling of faces8.npy,
48-colour palette distinctness, hypercube assignment sanity."""
import os, json
import numpy as np

ROOT = r"E:\redstone_nn\02_phase2_avatar\archive_phase1_attempts"
DATA = os.path.join(ROOT, "data")
ORDER16 = ['white', 'light_gray', 'gray', 'black', 'brown', 'red', 'orange', 'yellow',
           'lime', 'green', 'cyan', 'light_blue', 'blue', 'purple', 'magenta', 'pink']
mc16j = json.load(open(os.path.join(DATA, "mc16_palette.json")))
pals = json.load(open(os.path.join(DATA, "palettes.json")))
mc16_from_json = [mc16j[k][0] for k in ORDER16]
print(f"mc16_palette.json[k][0] == palettes.json pal16 : {mc16_from_json == pals['pal16']}")
print(f"mc16_palette.json[k][1] (secondary) example: {mc16j['white']}")

P16 = np.array(pals["pal16"], np.int64)
P48 = np.array(pals["pal48"], np.int64)
for nm, P in (("pal16", P16), ("pal48", P48)):
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(d, 1e9)
    print(f"{nm}: n={len(P)} distinct={len(np.unique(P,axis=0))} min pairwise RGB dist={d.min():.2f} "
          f"argmin pair={np.unravel_index(d.argmin(), d.shape)}")

rgb = np.load(os.path.join(DATA, "faces8_rgb.npy"))
lab = np.load(os.path.join(DATA, "faces8.npy")).reshape(-1, 64)
F = rgb.reshape(-1, 3).astype(np.int64)
P16C = np.array(json.load(open(os.path.join(DATA, "palette_avatar16.json")))["palette_ordered"], np.int64)

for nm, P in (("MC16 concrete pal16", P16), ("custom palette_avatar16", P16C)):
    idx = ((F[:, None] - P[None]) ** 2).sum(2).argmin(1).astype(np.uint8)
    mism = int((idx != lab.reshape(-1)).sum())
    mse = float(((P[idx] - F) ** 2).mean())
    print(f"re-derive labels with {nm:24s}: pixel mismatches vs faces8.npy = {mism:9d} "
          f"({mism/lab.size*100:6.2f}%)  per-channel MSE = {mse:8.2f}")
print("-> faces8.npy is labelled with the CUSTOM 16-colour palette, not the MC16 concrete one")

# hypercube assignment sanity
H = json.load(open(os.path.join(DATA, "palette_hypercube.json")))
ass = H["assignment"]
print(f"\npalette_hypercube.json: {len(ass)} vertices; bits unique: {len(set(a['bits'] for a in ass))}")
cols = [tuple(a["rgb"]) for a in ass]
print(f"  all colours drawn from palette_avatar16: {all(c in set(map(tuple, P16C.tolist())) for c in cols)}")
print(f"  edge_dist={H['edge_dist']:.2f} opp_dist={H['opp_dist']:.2f} bit_effect={[round(x,1) for x in H['bit_effect']]}")

# per-index usage entropy of the label distribution
cnt = np.array([(lab == k).sum() for k in range(16)], float)
p = cnt / cnt.sum()
print(f"\nlabel usage entropy = {-(p*np.log2(p)).sum():.3f} bits / 4.0 max; "
      f"min share {p.min()*100:.2f}% (index {int(p.argmin())}), max share {p.max()*100:.2f}% (index {int(p.argmax())})")
print(f"uniform-avatar rows: {int((np.array([len(np.unique(s)) for s in lab])==1).sum())}")

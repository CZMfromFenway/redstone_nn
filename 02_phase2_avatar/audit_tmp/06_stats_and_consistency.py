#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit step 5/6: reproduce the project's own reference statistics with the same
RNG draws, exact whole-dataset diversity, dedup consistency for skins/, CIFAR batch check."""
import os, json, tarfile, io, zipfile
import numpy as np
from PIL import Image

ROOT = r"E:\redstone_nn\02_phase2_avatar\archive_phase1_attempts"
DATA = os.path.join(ROOT, "data")
SK = os.path.join(DATA, "skins")
rgb = np.load(os.path.join(DATA, "faces8_rgb.npy"))
lab = np.load(os.path.join(DATA, "faces8.npy")).reshape(-1, 64).astype(np.int64)

def diversity(S):
    A = S[:, None, :] != S[None, :, :]
    m = len(S)
    return float(A.sum(2).sum() / (m * (m - 1)))

PAIRS = np.array([(r*8+c, r*8+(7-c)) for r in range(8) for c in range(4)])

print("### reference stats exactly as the project computes them (mixture_50k.py)")
n = len(lab)
POOL = 20000
rng0 = np.random.default_rng(0)
pool = lab[rng0.choice(n, POOL, replace=False)]
real_div = diversity(pool[:120])
real_uc = np.mean([len(np.unique(s)) for s in pool[:3000]])
print(f"  pool[:120] diversity                = {real_div:.4f}   (project claim: 57.8)")
print(f"  pool[:3000] mean distinct colours   = {real_uc:.4f}   (project claim: 5.29 real)")
print(f"  whole 50000 mean distinct colours   = {np.mean([len(np.unique(s)) for s in lab]):.4f}  (claim 5.31)")
print(f"  pool[:120] mean distinct colours    = {np.mean([len(np.unique(s)) for s in pool[:120]]):.4f}")
print(f"  pool[:120] symmetry rate            = {float((pool[:120][:,PAIRS[:,0]]==pool[:120][:,PAIRS[:,1]]).mean()):.4f}")

print("\n### exact whole-dataset diversity (no sampling), via per-position colour counts")
tot = 0.0
for p in range(64):
    _, c = np.unique(lab[:, p], return_counts=True)
    pk = c / len(lab)
    tot += 1.0 - float((pk ** 2).sum())
print(f"  mean pairwise Hamming distance over all 50000 avatars = {tot:.4f}")
print(f"  symmetry rate over all 50000 = {float((lab[:,PAIRS[:,0]]==lab[:,PAIRS[:,1]]).mean()):.4f}")
print(f"  fully symmetric avatars      = {float((lab[:,PAIRS[:,0]]==lab[:,PAIRS[:,1]]).all(1).mean()):.4f}")

print("\n### per-position palette occupancy / degenerate positions")
occ = np.zeros((64, 16), np.int64)
for p in range(64):
    v, c = np.unique(lab[:, p], return_counts=True)
    occ[p, v] = c
print(f"  positions where one palette index covers >50%: "
      f"{[(p, int(occ[p].argmax()), float(occ[p].max()/len(lab))) for p in range(64) if occ[p].max()/len(lab) > 0.5]}")
print(f"  palette indices never used anywhere: "
      f"{[k for k in range(16) if (lab == k).sum() == 0]}")
print(f"  usage of each palette index (pixel counts): {[int((lab==k).sum()) for k in range(16)]}")

print("\n### dedup consistency: unmatched skins/ inside the scanned window must be duplicates")
zf = zipfile.ZipFile(os.path.join(DATA, "minecraft_skins_64x64.zip"))
names = [n for n in zf.namelist() if n.lower().endswith(".png")]
idx = {nm: i for i, nm in enumerate(names)}
# rebuild the set of accepted face byte keys with their row, in zip order
acc_key_row, seen, k = {}, set(), 0
for nm in names:
    if k >= 50000:
        break
    with zf.open(nm) as f:
        im = Image.open(io.BytesIO(f.read())).convert("RGBA")
    key = np.array(im.crop((8, 8, 16, 16)))[:, :, :3].tobytes()
    if key in seen:
        continue
    seen.add(key)
    acc_key_row[key] = k
    k += 1
zf.close()
skfiles = sorted(f for f in os.listdir(SK) if f.lower().endswith(".png"))
unmatched_inside = [f for f in skfiles if f not in acc_key_row and idx[f] < 54441]
print(f"  skins/ files with zip position < 54441 that are NOT an accepted row: {len(unmatched_inside)}")
dupe_of = []
for f in unmatched_inside[:200]:
    with Image.open(os.path.join(SK, f)) as im:
        key = np.array(im.convert("RGBA").crop((8, 8, 16, 16)))[:, :, :3].tobytes()
    dupe_of.append((f, acc_key_row.get(key), key in seen))
print(f"  ... of those, how many are byte-duplicates of an accepted face: "
      f"{sum(1 for _, r, d in dupe_of if d)}/{len(dupe_of)}; example {dupe_of[:3]}")

print("\n### CIFAR-10 batch integrity")
p = os.path.join(DATA, "cifar-10-binary.tar.gz")
with tarfile.open(p, "r:gz") as tf:
    for m in tf.getmembers():
        if m.name.endswith(".bin"):
            d = tf.extractfile(m).read()
            print(f"  {m.name}: {len(d)} bytes, expected 30730000 -> {len(d)==30730000}, "
                  f"record size ok -> {len(d)%3073==0}, first byte label {d[0]}")

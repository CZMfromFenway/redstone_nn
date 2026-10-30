#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit step 6b (full dataset): separate TRANSPARENT blank faces from genuinely
opaque black faces, and check whole-file duplicates (the 'deduped' claim)."""
import os, zipfile, io, hashlib, collections, time
import numpy as np
from PIL import Image

ROOT = r"E:\redstone_nn\02_phase2_avatar\archive_phase1_attempts"
DATA = os.path.join(ROOT, "data")
ZIP = os.path.join(DATA, "minecraft_skins_64x64.zip")

zf = zipfile.ZipFile(ZIP)
names = zf.namelist()
t0 = time.time()
n = 0
fully_transparent = 0
any_transparent = 0
opaque_allblack = 0
file_digests = collections.Counter()
rgb_digests = collections.Counter()
trans_rgb_digests = collections.Counter()
for nm in names:
    n += 1
    with zf.open(nm) as f:
        raw = f.read()
    file_digests[hashlib.blake2b(raw, digest_size=16).digest()] += 1
    a4 = np.array(Image.open(io.BytesIO(raw)).convert("RGBA").crop((8, 8, 16, 16)))
    al = a4[:, :, 3]
    rgb = a4[:, :, :3]
    d = hashlib.blake2b(rgb.tobytes(), digest_size=16).digest()
    rgb_digests[d] += 1
    if al.min() == 0:
        any_transparent += 1
    if (al == 0).all():
        fully_transparent += 1
        trans_rgb_digests[d] += 1
    elif (rgb == 0).all():
        opaque_allblack += 1
    if n % 200000 == 0:
        print(f"  {n} {time.time()-t0:.0f}s  fully_transparent={fully_transparent} "
              f"any_transparent={any_transparent} opaque_allblack={opaque_allblack}", flush=True)
zf.close()
print(f"\nDONE {n} entries in {time.time()-t0:.0f}s")
print(f"FULLY transparent face crop (all 64 px alpha==0) : {fully_transparent} "
      f"({fully_transparent/n*100:.2f}%)")
print(f"ANY transparent pixel in the face crop          : {any_transparent} "
      f"({any_transparent/n*100:.2f}%)")
print(f"opaque but all-black face crop                  : {opaque_allblack} "
      f"({opaque_allblack/n*100:.3f}%)")
print(f"distinct face-crop RGB maps (incl. transparent) : {len(rgb_digests)}")
print(f"distinct transparent-blank RGB maps             : {len(trans_rgb_digests)} "
      f"(top counts {[c for _, c in trans_rgb_digests.most_common(5)]})")
opaque_distinct = len(rgb_digests) - len(trans_rgb_digests)
print(f"distinct NON-transparent face maps              : ~{opaque_distinct} "
      f"({opaque_distinct/n*100:.1f}% of entries are distinct visible faces)")
print(f"\nwhole-file duplicates: distinct PNG files {len(file_digests)}/{n} "
      f"-> duplicate files {n-len(file_digests)} ({(n-len(file_digests))/n*100:.3f}%)")
print(f"  whole-file multiplicity histogram: "
      f"{dict(sorted(collections.Counter(file_digests.values()).items())[:8])}")

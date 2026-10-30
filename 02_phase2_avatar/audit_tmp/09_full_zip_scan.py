#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit step 6 (full-dataset): open EVERY PNG in the 1.1M zip, verify it decodes
as 64x64, and measure face-crop redundancy across the whole dataset."""
import os, zipfile, io, hashlib, collections, time
import numpy as np
from PIL import Image

ROOT = r"E:\redstone_nn\02_phase2_avatar\archive_phase1_attempts"
DATA = os.path.join(ROOT, "data")
ZIP = os.path.join(DATA, "minecraft_skins_64x64.zip")

zf = zipfile.ZipFile(ZIP)
names = zf.namelist()
print(f"entries: {len(names)}", flush=True)
t0 = time.time()
digests = collections.Counter()
sizes = collections.Counter()
modes = collections.Counter()
bad = []
face_digest_of_name = {}
n = 0
for nm in names:
    n += 1
    try:
        with zf.open(nm) as f:
            im = Image.open(io.BytesIO(f.read()))
        sizes[im.size] += 1
        modes[im.mode] += 1
        a = np.array(im.convert("RGBA").crop((8, 8, 16, 16)))[:, :, :3]
        d = hashlib.blake2b(a.tobytes(), digest_size=16).digest()
        digests[d] += 1
        face_digest_of_name[nm] = d
    except Exception as e:
        bad.append((nm, type(e).__name__, str(e)[:60]))
    if n % 100000 == 0:
        el = time.time() - t0
        print(f"  {n}  bad={len(bad)}  distinct_faces={len(digests)}  {el:.0f}s "
              f"({n/max(el,1e-9):.0f}/s)", flush=True)
zf.close()
el = time.time() - t0
print(f"\nDONE {n} entries in {el:.0f}s ({n/max(el,1e-9):.0f}/s)")
print(f"unreadable / non-PNG: {len(bad)}  {bad[:10]}")
print(f"image sizes: {dict(sizes.most_common(6))}")
print(f"image modes: {dict(modes.most_common(6))}")
print(f"TOTAL entries: {n}")
print(f"DISTINCT face crops (8x8 RGB of head-front): {len(digests)}")
print(f"face-crop duplicate fraction: {1 - len(digests)/n:.4f}")
mult = collections.Counter(digests.values())
print(f"multiplicity histogram (count -> how many faces): {dict(sorted(mult.items())[:12])} ...")
print(f"largest multiplicities: {sorted(digests.values(), reverse=True)[:10]}")
# what is the single most duplicated face?
top = digests.most_common(1)[0][0]
ex = [nm for nm, d in face_digest_of_name.items() if d == top][:5]
print(f"most duplicated face appears {digests[top]} times; example skins {ex}")
with zf if False else zipfile.ZipFile(ZIP) as z2:
    for nm in ex[:2]:
        a = np.array(Image.open(io.BytesIO(z2.read(nm))).convert("RGBA").crop((8, 8, 16, 16)))[:, :, :3]
        print(f"   {nm}:")
        for r in range(8):
            print("     " + " ".join("%02X%02X%02X" % tuple(a[r, c]) for c in range(8)))

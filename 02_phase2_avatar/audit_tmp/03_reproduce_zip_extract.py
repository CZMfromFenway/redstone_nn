#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit step 4/6: reproduce extract_from_zip.py exactly, in the same zip order,
and check row-by-row whether faces8_rgb.npy / faces8.npy equal the crop of the
corresponding zip entry.  This recovers the row -> skin filename mapping."""
import os, json, zipfile, io, time
import numpy as np
from PIL import Image

ROOT = r"E:\redstone_nn\02_phase2_avatar\archive_phase1_attempts"
DATA = os.path.join(ROOT, "data")
ZIP = os.path.join(DATA, "minecraft_skins_64x64.zip")

PAL = np.array(json.load(open(os.path.join(DATA, "palette_avatar16.json")))["palette_ordered"],
               dtype=np.float64)
rgb = np.load(os.path.join(DATA, "faces8_rgb.npy"))
lab = np.load(os.path.join(DATA, "faces8.npy")).reshape(-1, 64)
N = len(rgb)
print(f"reference npy: rgb {rgb.shape} {rgb.dtype}  lab {lab.shape} {lab.dtype}  N={N}", flush=True)

zf = zipfile.ZipFile(ZIP)
names = [n for n in zf.namelist() if n.lower().endswith(".png")]
print(f"zip png entries: {len(names)}", flush=True)

t0 = time.time()
k = 0                # index into npy rows
seen = set()
skipped_dup = 0
bad = 0
scanned = 0
mismatch_rgb = []
mismatch_lab = []
mapping = {}         # row -> filename (a sample of them)
trunc_small = 0
alpha_stats = []    # alpha min per face crop
for nm in names:
    if k >= N:
        break
    scanned += 1
    try:
        with zf.open(nm) as f:
            im = Image.open(io.BytesIO(f.read())).convert("RGBA")
        if im.size[0] < 16 or im.size[1] < 16:
            trunc_small += 1
            continue
        face = im.crop((8, 8, 16, 16))
        arr4 = np.array(face)
        a = arr4[:, :, :3]
        if a.tobytes() in seen:
            skipped_dup += 1
            continue
        seen.add(a.tobytes())
    except Exception as e:
        bad += 1
        continue
    if not np.array_equal(a, rgb[k]):
        mismatch_rgb.append((k, nm, int(np.abs(a.astype(int) - rgb[k].astype(int)).max())))
    if k in (0, 1, 2, 3, 4, 10, 100, 1000, 10000, 25000, 49999) or k % 10000 == 0:
        mapping[k] = nm
        alpha_stats.append((k, nm, int(arr4[:, :, 3].min()), int(arr4[:, :, 3].max())))
    k += 1
    if scanned % 20000 == 0:
        print(f"  scanned {scanned}  kept {k}  dup {skipped_dup}  bad {bad}  "
              f"{time.time()-t0:.0f}s", flush=True)

zf.close()
el = time.time() - t0
print(f"\nfinished: scanned={scanned} kept={k} (target {N}) dup_skipped={skipped_dup} "
      f"bad={bad} too_small={trunc_small}  in {el:.0f}s ({scanned/max(el,1e-9):.0f}/s)")
print(f"RGB mismatches vs npy: {len(mismatch_rgb)}")
for m in mismatch_rgb[:10]:
    print("   row", m[0], m[1], "maxabsdiff", m[2])
print("\nrow -> zip entry name (recovered mapping, exact-match rows):")
for kk, nm, amin, amax in alpha_stats:
    print(f"   row {kk:6d} <- {nm}   (crop alpha min={amin} max={amax})")
print(f"\nrows beyond recovered mapping: npy N={N}, recovered={k}; extra npy rows = {N-k}")

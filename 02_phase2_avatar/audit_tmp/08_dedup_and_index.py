#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit step 6: dedup accounting, zip_index.json vs real zip central directory,
face-crop duplicate multiplicity."""
import os, json, zipfile, io, collections
import numpy as np
from PIL import Image

ROOT = r"E:\redstone_nn\02_phase2_avatar\archive_phase1_attempts"
DATA = os.path.join(ROOT, "data")
SK = os.path.join(DATA, "skins")
ZIP = os.path.join(DATA, "minecraft_skins_64x64.zip")

zf = zipfile.ZipFile(ZIP)
names = [n for n in zf.namelist() if n.lower().endswith(".png")]
infos = zf.infolist()
idx = {n: i for i, n in enumerate(names)}

# --- A. full scan of the first 54441 zip entries: multiplicity of face crops
seen = {}
mult = collections.Counter()
k = 0
for i, nm in enumerate(names):
    if k >= 50000:
        break
    with zf.open(nm) as f:
        im = Image.open(io.BytesIO(f.read())).convert("RGBA")
    key = np.array(im.crop((8, 8, 16, 16)))[:, :, :3].tobytes()
    if key in seen:
        seen[key] += 1
    else:
        seen[key] = 1
        k += 1
    if i >= 54441:
        pass
last_scanned = i
print(f"scanned {last_scanned+1} entries, accepted {k}")
print(f"distinct face crops: {len(seen)}; multiplicity histogram: "
      f"{dict(sorted(collections.Counter(seen.values()).items()))}")

# --- B. skins/ files inside the scanned window: accepted by name vs deduped away
k = 0
keyset = set()
acc_order = []
for nm in names:
    if k >= 50000:
        break
    with zf.open(nm) as f:
        im = Image.open(io.BytesIO(f.read())).convert("RGBA")
    key = np.array(im.crop((8, 8, 16, 16)))[:, :, :3].tobytes()
    if key in keyset:
        continue
    keyset.add(key)
    acc_order.append(nm)
    k += 1
acc_by_name = {n: i for i, n in enumerate(acc_order)}
skfiles = sorted(f for f in os.listdir(SK) if f.lower().endswith(".png"))
inwin = [f for f in skfiles if idx[f] <= last_scanned]
acc = [f for f in inwin if f in acc_by_name]
dup = [f for f in inwin if f not in acc_by_name]
print(f"\nskins/ files located inside the scanned window: {len(inwin)}")
print(f"  accepted as their own npy row: {len(acc)}")
print(f"  skipped by the byte-dedup        : {len(dup)}")
dupmatch = 0
for f in dup[:40]:
    with Image.open(os.path.join(SK, f)) as im:
        key = np.array(im.convert("RGBA").crop((8, 8, 16, 16)))[:, :, :3].tobytes()
    if key in keyset:
        dupmatch += 1
print(f"  of the first {min(40,len(dup))} skipped files, byte-duplicates of an accepted face: {dupmatch}")
print(f"  observed overlap fraction {len(acc)}/{len(skfiles)} = {len(acc)/len(skfiles):.4f}; "
      f"expected if skins/ were a uniform sample of 1107411 entries and npy covers the first "
      f"{last_scanned+1}: {(last_scanned+1)/len(names):.4f}")

# --- C. zip_index.json vs the real central directory
J = json.load(open(os.path.join(DATA, "zip_index.json")))
print(f"\nzip_index.json: {type(J).__name__} of {len(J)} entries")
mism_name = mism_csz = mism_lho = mism_crc = 0
for a, b in zip(J, infos):
    if a["name"] != b.filename:
        mism_name += 1
    if a["csz"] != b.compress_size:
        mism_csz += 1
    if a["lho"] != b.header_offset:
        mism_lho += 1
    if a["crc"] != (b.CRC & 0xFFFFFFFF):
        mism_crc += 1
print(f"  compared first {len(J)} entries: name mismatches={mism_name} csz={mism_csz} "
      f"lho={mism_lho} crc={mism_crc}")
print(f"  json entry [0]={J[0]}")
print(f"  zip  info [0]= name={infos[0].filename} csz={infos[0].compress_size} "
      f"lho={infos[0].header_offset} crc={infos[0].CRC & 0xFFFFFFFF} usize={infos[0].file_size}")
print(f"  json covers {len(J)}/{len(infos)} zip entries ({len(J)/len(infos)*100:.1f}%)")
zf.close()

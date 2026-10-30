#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit step 1/4a: zip readability, entry inventory, skins/ folder inventory."""
import os, json, zipfile, io, collections
import numpy as np
from PIL import Image

ROOT = r"E:\redstone_nn\02_phase2_avatar\archive_phase1_attempts"
DATA = os.path.join(ROOT, "data")
ZIP = os.path.join(DATA, "minecraft_skins_64x64.zip")
SK = os.path.join(DATA, "skins")

print("=== zip central directory ===")
zf = zipfile.ZipFile(ZIP)
names = zf.namelist()
print(f"entries: {len(names)}")
ext = collections.Counter(os.path.splitext(n)[1].lower() for n in names)
print("extensions:", dict(ext.most_common(10)))
print("first 5 names:", names[:5])
print("last 5 names:", names[-5:])
pngs = [n for n in names if n.lower().endswith(".png")]
print(f"png entries: {len(pngs)}")

# test-decompress a spread of entries
info = zf.infolist()
idx = [0, 1, 100, 1000, 10000, 500000, len(info) - 1]
for i in idx:
    it = info[i]
    try:
        with zf.open(it) as f:
            raw = f.read()
        im = Image.open(io.BytesIO(raw))
        print(f"  [{i}] {it.filename!r} compress={it.compress_type} csize={it.compress_size} "
              f"usize={it.file_size} -> PNG {im.size} {im.mode}")
    except Exception as e:
        print(f"  [{i}] {it.filename!r} FAILED: {type(e).__name__}: {e}")

print()
print("=== zip_index.json ===")
with open(os.path.join(DATA, "zip_index.json"), "rb") as f:
    head = f.read(400)
print("head bytes:", head[:300])
try:
    J = json.load(open(os.path.join(DATA, "zip_index.json")))
    print("type:", type(J), "len:", len(J))
    if isinstance(J, dict):
        print("keys:", list(J.keys())[:10])
        for k in list(J.keys())[:5]:
            print("  ", k, "->", str(J[k])[:200])
    else:
        print("first 3:", J[:3])
except Exception as e:
    print("json parse failed:", e)
    # try line-based
    with open(os.path.join(DATA, "zip_index.json"), "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            print(f"  line {i}: {line.rstrip()[:200]}")
            if i > 4:
                break

print()
print("=== skins/ folder ===")
files = sorted(f for f in os.listdir(SK) if f.lower().endswith(".png"))
print(f"png files: {len(files)}  total bytes: {sum(os.path.getsize(os.path.join(SK,f)) for f in files)}")
print("first 5:", files[:5])
print("last 5:", files[-5:])
sizes = collections.Counter()
modes = collections.Counter()
bad = []
for f in files:
    try:
        with Image.open(os.path.join(SK, f)) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
    except Exception as e:
        bad.append((f, repr(e)))
print("sizes:", dict(sizes.most_common(10)))
print("modes:", dict(modes))
print(f"unreadable PNGs: {len(bad)}  {bad[:5]}")
inzip = sum(1 for f in files if f in set(pngs))
print(f"skins/ files that also appear as zip entry names: {inzip}/{len(files)}")
print("sample of skins/ names present in zip:", [f for f in files[:2000] if f in set(pngs)][:5])
zf.close()

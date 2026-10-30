#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit step 4b/6: full row->name mapping for all 50000 rows, alpha check on the
crop, label re-derivation over all rows, alternative-crop comparison on the
5820 skins/ PNGs, and skins/ <-> npy row matching."""
import os, json, zipfile, io, time, collections
import numpy as np
from PIL import Image

ROOT = r"E:\redstone_nn\02_phase2_avatar\archive_phase1_attempts"
DATA = os.path.join(ROOT, "data")
ZIP = os.path.join(DATA, "minecraft_skins_64x64.zip")
SK = os.path.join(DATA, "skins")

PAL = np.array(json.load(open(os.path.join(DATA, "palette_avatar16.json")))["palette_ordered"],
               dtype=np.float64)
rgb = np.load(os.path.join(DATA, "faces8_rgb.npy"))
lab = np.load(os.path.join(DATA, "faces8.npy")).reshape(-1, 64)

print("### A. rerun extraction, keep every accepted name, check alpha + labels")
zf = zipfile.ZipFile(ZIP)
names = [n for n in zf.namelist() if n.lower().endswith(".png")]
t0 = time.time()
row_names, seen, k, alpha_non255, lab_mismatch = [], set(), 0, [], 0
for nm in names:
    if k >= 50000:
        break
    with zf.open(nm) as f:
        im = Image.open(io.BytesIO(f.read())).convert("RGBA")
    a4 = np.array(im.crop((8, 8, 16, 16)))
    a = a4[:, :, :3]
    if a.tobytes() in seen:
        continue
    seen.add(a.tobytes())
    if not np.array_equal(a, rgb[k]):
        print("  RGB MISMATCH row", k, nm)
    if a4[:, :, 3].min() != 255:
        alpha_non255.append((k, nm, int(a4[:, :, 3].min())))
    d = ((a.reshape(-1, 3).astype(np.int64)[:, None] - PAL.astype(np.int64)[None]) ** 2).sum(2)
    if not np.array_equal(d.argmin(1).astype(np.uint8), lab[k]):
        lab_mismatch += 1
    row_names.append(nm)
    k += 1
zf.close()
print(f"  accepted rows: {len(row_names)} in {time.time()-t0:.0f}s")
print(f"  crops with any alpha != 255: {len(alpha_non255)}  {alpha_non255[:10]}")
print(f"  rows whose re-derived label map differs from faces8.npy: {lab_mismatch}")
print(f"  distinct source skins: {len(set(row_names))}")

# alpha over all 54441 scanned entries (before dedup) as well
print("\n### A2. alpha over the whole head-front region for the scanned window")
zf = zipfile.ZipFile(ZIP)
cnt = collections.Counter(); k2 = 0; n_alpha_bad = 0; ex = []
for nm in names:
    if k2 >= 50000 + 4441:
        break
    with zf.open(nm) as f:
        im = Image.open(io.BytesIO(f.read())).convert("RGBA")
    a4 = np.array(im.crop((8, 8, 16, 16)))[:, :, 3]
    cnt[int(a4.min())] += 1
    if a4.min() != 255 and len(ex) < 8:
        ex.append((nm, int(a4.min()), int((a4 == 0).mean() * 100)))
    k2 += 1
print(f"  scanned {k2}; histogram of min-alpha in the 8x8 face crop: {dict(sorted(cnt.items())[:8])} ...")
print(f"  examples with transparent pixels in the face: {ex}")
zf.close()

print("\n### B. skins/ folder: alternative crops, opacity, symmetry, matching to npy rows")
name2row = {n: i for i, n in enumerate(row_names)}
skfiles = sorted(f for f in os.listdir(SK) if f.lower().endswith(".png"))
in_rows = [(f, name2row[f]) for f in skfiles if f in name2row]
print(f"  skins/ files total {len(skfiles)}; present among the 50000 npy rows: {len(in_rows)}")
print(f"  row indices of matched skins: min={min(r for _, r in in_rows)} max={max(r for _, r in in_rows)}")
print(f"  example matches: {in_rows[:5]}")

REG = {
    "head_front (8,8)-(16,16)  [USED]": (8, 8, 16, 16),
    "head_right (0,8)-(8,16)":          (0, 8, 8, 16),
    "head_left  (16,8)-(24,16)":        (16, 8, 24, 16),
    "head_top   (8,0)-(16,8)":          (8, 0, 16, 8),
    "head_bot   (16,0)-(24,8)":         (16, 0, 24, 8),
    "head_back  (24,8)-(32,16)":        (24, 8, 32, 16),
    "hat_front  (40,8)-(48,16)":        (40, 8, 48, 16),
    "body_front (20,20)-(28,28)":       (20, 20, 28, 28),
    "shifted    (9,9)-(17,17)":         (9, 9, 17, 17),
}
stats = {k: {"opaque": 0, "full": 0, "clear": 0, "sym": 0.0, "uc": [], "std": []} for k in REG}
nf = 0
for f in skfiles:
    with Image.open(os.path.join(SK, f)) as im:
        im = im.convert("RGBA")
        if im.size != (64, 64):
            continue
        nf += 1
        for k, (x0, y0, x1, y1) in REG.items():
            a4 = np.array(im.crop((x0, y0, x1, y1)))
            al = a4[:, :, 3]
            s = stats[k]
            frac = float((al > 8).mean())
            s["opaque"] += frac
            if frac == 1.0:
                s["full"] += 1
            if frac == 0.0:
                s["clear"] += 1
            px = a4[:, :, :3]
            s["sym"] += float((px[:, :4] == px[:, 4:][:, ::-1]).all(-1).mean())
            s["uc"].append(len(np.unique(px.reshape(-1, 3), axis=0)))
            s["std"].append(float(px.reshape(-1, 3).std(0).mean()))
print(f"\n  skins analysed: {nf}")
print(f"  {'region':<34} {'mean opaque':>11} {'fully opaque':>12} {'fully clear':>11} "
      f"{'mirror-sym':>10} {'uniq RGB':>9} {'pix std':>8}")
for k in REG:
    s = stats[k]
    print(f"  {k:<34} {s['opaque']/nf:>11.3f} {s['full']/nf:>12.3f} {s['clear']/nf:>11.3f} "
          f"{s['sym']/nf:>10.3f} {np.mean(s['uc']):>9.2f} {np.mean(s['std']):>8.1f}")

print("\n### C. a few real faces rendered as ASCII (palette index -> glyph), plus the crop pixels")
GLYPH = " .:-=+*#%@ABCDEFGH"   # by luminance rank, purely for eyeballing
pal_lum = 0.299 * PAL[:, 0] + 0.587 * PAL[:, 1] + 0.114 * PAL[:, 2]
rank = np.argsort(np.argsort(pal_lum))
for row in (0, 1, 2, 1000):
    nm = row_names[row]
    print(f"  row {row} <- {nm}")
    for r in range(8):
        line = "".join(GLYPH[min(15, rank[lab[row, r * 8 + c]])] for c in range(8))
        px = " ".join("%02X%02X%02X" % tuple(rgb[row, r, c]) for c in range(8))
        print(f"    |{line}|  {px}")

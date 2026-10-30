#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit step 4c/6: alpha handling quantified, disk-based verification of the
matched skins/, CIFAR archive readability, mc8 sanity, axis-order (transpose) test."""
import os, json, tarfile, gzip, collections
import numpy as np
from PIL import Image

ROOT = r"E:\redstone_nn\02_phase2_avatar\archive_phase1_attempts"
DATA = os.path.join(ROOT, "data")
SK = os.path.join(DATA, "skins")
rgb = np.load(os.path.join(DATA, "faces8_rgb.npy"))
lab = np.load(os.path.join(DATA, "faces8.npy")).reshape(-1, 64)

print("### A. axis order / transpose test (row-major r*8+c vs column-major)")
def mirr_sym(X):           # X (N,8,8,3) ; vertical mirror symmetry of the *viewed* image
    return float((X[:, :, :4] == X[:, :, 4:][:, :, ::-1]).all(-1).mean())
print(f"  vertical-mirror symmetry, row-major view  : {mirr_sym(rgb):.4f}")
print(f"  vertical-mirror symmetry, transposed view : {mirr_sym(rgb.transpose(0,2,1,3)):.4f}")
# same at label level with the project's PAIRS definition p = r*8+c
PAIRS = np.array([(r*8+c, r*8+(7-c)) for r in range(8) for c in range(4)])
print(f"  label-level pair equality (r*8+c convention): "
      f"{float((lab[:, PAIRS[:,0]] == lab[:, PAIRS[:,1]]).mean()):.4f}")
# if the flat array were column-major, the mirror pairs would be (r*8+c, (7-r)*8+c)
PAIRS_T = np.array([(r*8+c, (7-r)*8+c) for r in range(8) for c in range(4)])
print(f"  label-level pair equality (transpose convention): "
      f"{float((lab[:, PAIRS_T[:,0]] == lab[:, PAIRS_T[:,1]]).mean()):.4f}")

print("\n### B. alpha handling: transparency inside the extracted face crop")
zf_names_ok = True
# recompute alpha for the exact 50000 accepted rows (same deterministic order)
import zipfile, io
zf = zipfile.ZipFile(os.path.join(DATA, "minecraft_skins_64x64.zip"))
names = [n for n in zf.namelist() if n.lower().endswith(".png")]
row_names, seen, k = [], set(), 0
alpha_min = np.empty(50000, np.uint8)
alpha_zero_frac = np.empty(50000, np.float32)
rgba_rows = np.empty((50000, 8, 8, 4), np.uint8)
for nm in names:
    if k >= 50000:
        break
    with zf.open(nm) as f:
        im = Image.open(io.BytesIO(f.read())).convert("RGBA")
    a4 = np.array(im.crop((8, 8, 16, 16)))
    if a4[:, :, :3].tobytes() in seen:
        continue
    seen.add(a4[:, :, :3].tobytes())
    rgba_rows[k] = a4
    alpha_min[k] = a4[:, :, 3].min()
    alpha_zero_frac[k] = float((a4[:, :, 3] == 0).mean())
    row_names.append(nm)
    k += 1
zf.close()
print(f"  rows: {k}")
print(f"  rows whose crop is fully opaque (alpha==255 everywhere): {int((alpha_min==255).sum())}")
print(f"  rows with >=1 fully transparent pixel: {int((alpha_min<255).sum())}")
print(f"  rows with >=1 pixel of alpha==0       : {int((alpha_zero_frac>0).sum())}")
print(f"  rows fully transparent (all 64 px alpha==0): {int((alpha_zero_frac==1).sum())}")
print(f"  rows with >50% transparent pixels: {int((alpha_zero_frac>0.5).sum())}")
print(f"  total transparent pixel fraction in the 50000 crops: {alpha_zero_frac.mean():.5f}")
# RGB kept for transparent pixels -> is it black?
tr = rgba_rows[:, :, :, 3] == 0
print(f"  RGB of transparent pixels: mean={rgba_rows[:,:,:,:3][tr].mean(0) if tr.any() else None}, "
      f"all-zero fraction={float((rgba_rows[:,:,:,:3][tr]==0).all(-1).mean()) if tr.any() else 0:.4f}")
print(f"  compare: alpha dropped -> stored rgb equals crop's RGB channel exactly "
      f"(ignoring alpha): {np.array_equal(rgba_rows[:,:,:,:3], rgb)}")
# effect of alpha on the palette: how many rows become (4,4,4)-ish black because of transparency
fully_tr = np.where(alpha_zero_frac == 1)[0]
print(f"  fully transparent rows: {fully_tr[:20]}  -> stored rgb {rgb[fully_tr[:5]].reshape(-1,3)[:6].tolist() if len(fully_tr) else []}")
badblack = [i for i in fully_tr if not (rgb[i] == 0).all()]
print(f"  fully transparent rows whose stored RGB is all zero: "
      f"{sum(1 for i in fully_tr if (rgb[i]==0).all())}/{len(fully_tr)}")

print("\n### C. direct disk verification for skins/ that are among the 50000 rows")
name2row = {n: i for i, n in enumerate(row_names)}
skfiles = sorted(f for f in os.listdir(SK) if f.lower().endswith(".png"))
matched = [(f, name2row[f]) for f in skfiles if f in name2row]
ok, bad = 0, []
for f, r in matched:
    with Image.open(os.path.join(SK, f)) as im:
        a = np.array(im.convert("RGBA").crop((8, 8, 16, 16)))[:, :, :3]
    if np.array_equal(a, rgb[r]):
        ok += 1
    else:
        bad.append((f, r, int(np.abs(a.astype(int) - rgb[r].astype(int)).max())))
print(f"  matched skins: {len(matched)}   byte-identical to npy row: {ok}   mismatches: {len(bad)}")
print(f"  mismatch examples: {bad[:5]}")
# how many skins/ files are NOT in the 50000: check they are simply not in the scanned window
notm = [f for f in skfiles if f not in name2row]
print(f"  skins/ files not among the 50000 rows: {len(notm)}  (e.g. {notm[:3]})")
idx = {n: i for i, n in enumerate(names)}
pos = [idx[f] for f in notm[:20]]
print(f"  their zip positions: {pos} (npy covers zip positions < {idx[row_names[-1]]+1} roughly)")
print(f"  rank of last accepted name in zip order: {idx[row_names[-1]]}")
far = [idx[f] for f in notm]
print(f"  min zip position among unmatched skins: {min(far)}; count unmatched with position < 54441: "
      f"{sum(1 for p in far if p < 54441)}")

print("\n### D. CIFAR-10 archive")
p = os.path.join(DATA, "cifar-10-binary.tar.gz")
with open(p, "rb") as f:
    print("  magic bytes:", f.read(4))
try:
    with tarfile.open(p, "r:gz") as tf:
        mem = tf.getmembers()
        print(f"  tar members: {len(mem)}")
        print(f"  first 8: {[m.name for m in mem[:8]]}")
        tot = sum(m.size for m in mem)
        print(f"  total uncompressed size: {tot/1e6:.1f} MB")
        # read one batch fully to prove integrity
        b = [m for m in mem if m.name.endswith('data_batch_1')]
        if b:
            d = tf.extractfile(b[0]).read()
            print(f"  data_batch_1 bytes={len(d)} (expect 30730000): {len(d)==30730000}")
except Exception as e:
    print("  tarfile FAILED:", type(e).__name__, e)

print("\n### E. mc8.npy (texture-patch dataset, different provenance)")
mc8 = np.load(os.path.join(DATA, "mc8.npy"))
print(f"  mc8 shape={mc8.shape} dtype={mc8.dtype} min={mc8.min()} max={mc8.max()}")
sm = mc8[:20000].reshape(20000, -1)
fr = rgb.reshape(50000, -1)
inter = len(set(map(bytes, sm[:5000])) & set(map(bytes, fr)))
print(f"  exact 8x8 matches between mc8[:5000] and faces8_rgb: {inter}")
u = np.unique(mc8[:20000].reshape(-1, 3), axis=0)
print(f"  distinct colours in mc8[:20000]: {len(u)}; avatars uniform: "
      f"{int((mc8[:20000] == mc8[:20000][:, :1]).all((1,2,3)).sum())}")

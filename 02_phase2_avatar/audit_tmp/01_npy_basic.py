#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit step 1+2+3+5: inspect faces8.npy / faces8_rgb.npy, palette consistency, stats."""
import json, os, sys
import numpy as np

ROOT = r"E:\redstone_nn\02_phase2_avatar\archive_phase1_attempts"
DATA = os.path.join(ROOT, "data")

def hdr(p):
    with open(p, "rb") as f:
        magic = f.read(6)
        f.seek(0)
        v = np.lib.format.read_magic(f)
        shape, fo, dt = np.lib.format.read_array_header_1_0(f) if v == (1, 0) else np.lib.format.read_array_header_2_0(f)
    return magic, shape, fo, dt

for nm in ("faces8.npy", "faces8_rgb.npy", "mc8.npy"):
    p = os.path.join(DATA, nm)
    if not os.path.exists(p):
        print(f"{nm}: MISSING"); continue
    m, s, fo, dt = hdr(p)
    print(f"{nm}: magic={m!r} shape={s} fortran_order={fo} dtype={dt} bytes={os.path.getsize(p)}")

lab = np.load(os.path.join(DATA, "faces8.npy"))
rgb = np.load(os.path.join(DATA, "faces8_rgb.npy"))
print()
print(f"faces8      shape={lab.shape} dtype={lab.dtype} min={lab.min()} max={lab.max()} nbytes={lab.nbytes}")
print(f"faces8_rgb  shape={rgb.shape} dtype={rgb.dtype} min={rgb.min()} max={rgb.max()} nbytes={rgb.nbytes}")

pal = np.array(json.load(open(os.path.join(DATA, "palette_avatar16.json")))["palette_ordered"], dtype=np.float64)
P = len(pal)
print(f"custom palette (palette_avatar16.json) len={P}")
print("palette rows:", [tuple(int(x) for x in c) for c in pal])
print(f"distinct palette rows: {len(np.unique(pal, axis=0))}")
d = np.linalg.norm(pal[:, None] - pal[None], axis=2)
np.fill_diagonal(d, 1e9)
print(f"min pairwise palette distance: {d.min():.3f}")

# ---- shape normalisation
lab2 = lab.reshape(len(lab), -1) if lab.ndim == 2 else lab.reshape(len(lab), 64)
print(f"\nlab reshaped to {lab2.shape}; labels >= {P}: {int((lab2 >= P).sum())}; labels < 0: {int((lab2 < 0).sum())}")

# ---- quantization consistency: recompute nearest palette entry
F = rgb.reshape(-1, 3).astype(np.int64)
# exact integer argmin (squared distance)
dist = ((F[:, None, :] - pal[None].astype(np.int64)) ** 2).sum(2)
nn = dist.argmin(1).astype(np.uint8)
stored = lab2.reshape(-1)
mism = int((nn != stored).sum())
print(f"\nnearest-palette re-derivation: total pixels={nn.size}  mismatches={mism}  ({mism/nn.size*100:.4f}%)")

# ties between nearest and second nearest?
dsort = np.sort(dist, axis=1)
ties = int((dsort[:, 0] == dsort[:, 1]).sum())
print(f"pixels with a tie between best and 2nd-best palette entry: {ties}")

# what error would stored labels give vs re-derived labels
err_stored = float(((pal[stored] - F) ** 2).sum(1).mean())
err_nn = float(((pal[nn] - F) ** 2).sum(1).mean())
print(f"mean per-pixel squared error (sum over 3 ch), stored labels : {err_stored:.4f}")
print(f"mean per-pixel squared error (sum over 3 ch), re-derived NN: {err_nn:.4f}")
print(f"  -> per-CHANNEL MSE (divide by 3): stored {err_stored/3:.4f}  NN {err_nn/3:.4f}")
print(f"  -> RMS per channel sqrt(x/3)    : stored {np.sqrt(err_stored/3):.4f}/255  NN {np.sqrt(err_nn/3):.4f}/255")
print(f"  -> RMS per channel sqrt(x)      : stored {np.sqrt(err_stored):.4f}/255  NN {np.sqrt(err_nn):.4f}/255")
print(f"  -> max abs per-channel error    : {np.abs(pal[stored]-F).max():.1f}")

# same numbers with the MC16 concrete palette for reference
pal16 = np.array(json.load(open(os.path.join(DATA, "palettes.json")))["pal16"], dtype=np.int64)
d16 = ((F[:, None, :] - pal16[None]) ** 2).sum(2)
e16 = float((d16.min(1)).mean())
print(f"MC16 concrete palette reference: mean per-pixel squared err {e16:.4f}  per-channel MSE {e16/3:.4f}")

# ---- per-avatar distinct colours
sub = lab2
uc = np.array([len(np.unique(s)) for s in sub])
print(f"\n--- distinct palette colours per avatar (n={len(sub)}) ---")
print(f"mean={uc.mean():.4f} median={np.median(uc):.1f} min={uc.min()} max={uc.max()}")
vals, cnt = np.unique(uc, return_counts=True)
print("histogram:", {int(k): int(v) for k, v in zip(vals, cnt)})
print(f"uniform (1 colour) avatars: {int((uc == 1).sum())}  ({float((uc==1).mean())*100:.3f}%)")

# same on raw RGB (distinct RGB colours)
ucr = np.array([len(np.unique(s.reshape(-1, 3), axis=0)) for s in rgb])
print(f"distinct RAW RGB colours per avatar: mean={ucr.mean():.4f} median={np.median(ucr):.1f} max={ucr.max()}")
print(f"raw avatars that are a single flat colour: {int((ucr==1).sum())}")

# ---- diversity: mean pairwise hamming distance over 64 positions
def diversity(S):
    A = S[:, None, :] != S[None, :, :]
    m = len(S)
    return float(A.sum(2).sum() / (m * (m - 1)))

print(f"\n--- diversity (mean pairwise Hamming over 64 positions) ---")
print(f"first 120 samples (project definition uses pool[:120]): {diversity(sub[:120]):.4f}")
print(f"first 400  samples: {diversity(sub[:400]):.4f}")
print(f"first 3000 samples: {diversity(sub[:3000]):.4f}")
rng = np.random.default_rng(0)
for m in (120, 400):
    vals_ = [diversity(sub[rng.choice(len(sub), m, replace=False)]) for _ in range(5)]
    print(f"5x random {m}: {['%.4f' % v for v in vals_]}")

# ---- symmetry rate
PAIRS = np.array([(r * 8 + c, r * 8 + (7 - c)) for r in range(8) for c in range(4)])
same = float((sub[:, PAIRS[:, 0]] == sub[:, PAIRS[:, 1]]).mean())
allsame = float((sub[:, PAIRS[:, 0]] == sub[:, PAIRS[:, 1]]).all(1).mean())
print(f"\n--- symmetry (mirror pairs (r,c)<->(r,7-c), 32 pairs) ---")
print(f"pair-level equal rate over all {len(sub)} samples: {same:.4f}")
print(f"fully symmetric avatars (all 32 pairs equal): {allsame:.4f}  ({int((sub[:,PAIRS[:,0]]==sub[:,PAIRS[:,1]]).all(1).sum())} samples)")
print(f"same on raw RGB pixels: {float((rgb[:, PAIRS[:,0]//8, PAIRS[:,0]%8] == rgb[:, PAIRS[:,1]//8, PAIRS[:,1]%8]).all(-1).mean()):.4f}")

# ---- duplicates / blank
print(f"\n--- duplicates & degenerate rows ---")
u, c = np.unique(sub, axis=0, return_counts=True)
print(f"unique label rows: {len(u)} / {len(sub)}   duplicated rows: {len(sub)-len(u)}   max multiplicity {c.max()}")
ur, cr = np.unique(rgb.reshape(len(rgb), -1), axis=0, return_counts=True)
print(f"unique RGB rows  : {len(ur)} / {len(rgb)}   duplicated rows: {len(rgb)-len(ur)}   max multiplicity {cr.max()}")
# label collision: two different RGB faces with identical label map (expected lossy)
ul, cl = np.unique(sub, axis=0, return_counts=True)
print(f"distinct label maps: {len(ul)}")
# blank images: all pixels same RGB
flat = rgb.reshape(len(rgb), -1, 3)
isblank = (flat == flat[:, :1, :]).all((1, 2))
print(f"blank (uniform RGB) avatars in faces8_rgb: {int(isblank.sum())}")
# black / white dominated
print(f"all-zero RGB avatars: {int((flat.sum((1,2))==0).sum())}   all-255: {int((flat==255).all((1,2)).sum())}")
print(f"\nRGB value histogram (top 20): {sorted(zip(*np.unique(rgb, return_counts=True)), key=lambda t:-t[1])[:20]}")
print(f"RGB mean over all pixels: {rgb.mean(axis=(0,1,2))}")

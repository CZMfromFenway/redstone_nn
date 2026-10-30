#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Build an INDEXED gallery of real faces and several style embeddings for them.

Why an indexed gallery: the user's 24-image contact sheet has no file names, so its
indices come from the grid order. To let the user VERIFY any clustering later, the
gallery must be reproducible, numbered, and printable as a contact sheet with indices.

Embeddings computed (each is a candidate "style space"):
  pca22    : the 22 PCA coefficients the generator is conditioned on
  k3       : responses of the learned 3x3 kernels, mean+max pooled per kernel
  k2       : responses of the learned 2x2 kernels, mean+max pooled per kernel
  stats    : hand-made interpretable features (brightness, vertical gradient, contrast,
             edge density, level histogram, upper/lower half means, eye-row darkness)

Outputs under results/style/:
  gallery.npy           (G,64) uint8   the gallery faces, levels 0..15
  gallery_meta.json     provenance: which val-split rows, and the embeddings' shapes
  emb_<name>.npy        (G,D) float32  each candidate style space
  contact_sheet.png     12 wide, each tile labelled with its index
"""
import json, os
import numpy as np
from PIL import Image, ImageDraw

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)

OUT = os.path.join(ROOT, "results", "style")
os.makedirs(OUT, exist_ok=True)
LEVELS = np.linspace(0, 255, 16)
NLEV = 16
G_SIZE = 240
COLS = 12
SEED = 2024

# ---------------------------------------------------------------- gallery rows
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
rows = va[np.random.default_rng(SEED).choice(len(va), G_SIZE, replace=False)]
gallery = full[rows]                                  # (G,8,8) uint8 levels
np.save(os.path.join(OUT, "gallery.npy"), gallery)
print(f"gallery: {gallery.shape} from val rows (seed {SEED})")

# the user's 24-image sheet used a DIFFERENT draw; keep its rows too so the labels
# they gave can be matched exactly.
user_rows = va[np.random.default_rng(11).choice(len(va), 24, replace=False)]
np.save(os.path.join(OUT, "user24_rows.npy"), user_rows)
print(f"user sheet rows: {user_rows.shape}")

# ---------------------------------------------------------------- embeddings
def pca_basis():
    rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
    LUM = np.array([0.299, 0.587, 0.114])
    gray = (rgb.astype(np.float64) * LUM).sum(-1)
    gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
    m = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
         gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
    Gg = gray[m]
    assert len(Gg) == len(full)
    X = Gg.reshape(len(Gg), -1)
    mu = X.mean(0, keepdims=True)
    sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
    U, S, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
    comps = Vt[:22]
    sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
    return mu, comps, sd, X


mu, comps, sd, Xall = pca_basis()
emb = {}

# --- PCA coefficients, in the generator's own units
emb["pca22"] = ((Xall[rows] - mu[0]) @ comps.T) / sd[0]

# --- learned convolution-kernel responses (from the kernel_style run) ----------------
kpath = os.path.join(ROOT, "results", "kernel_style", "kernel_style", "kernel_info.json")
if os.path.exists(kpath):
    info = json.load(open(kpath, encoding="utf-8"))
    E = info["embed"]
    k3 = np.array(info["kernels_3x3"], dtype=np.float32).reshape(E, 3, 3)
    k2 = np.array(info["kernels_2x2"], dtype=np.float32).reshape(E, 2, 2)
    G01 = gallery.astype(np.float32) / 15.0 * 2 - 1          # (G,8,8) in [-1,1]

    def conv_same(img, k):
        kh, kw = k.shape
        ph, pw = kh // 2, kw // 2
        P = np.pad(img, ((0, 0), (ph, ph), (pw, pw)), mode="edge")
        out = np.zeros((img.shape[0], 8, 8), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                out += k[i, j] * P[:, i:i + 8, j:j + 8]
        return out

    feats3, feats2 = [], []
    for e in range(E):
        r = np.maximum(conv_same(G01, k3[e]), 0)
        feats3 += [r.mean(axis=(1, 2)), r.max(axis=(1, 2)), (r > 0).mean(axis=(1, 2))]
        r2 = np.maximum(conv_same(G01, k2[e]), 0)
        feats2 += [r2.mean(axis=(1, 2)), r2.max(axis=(1, 2)), (r2 > 0).mean(axis=(1, 2))]
    emb["k3"] = np.stack(feats3, 1)
    emb["k2"] = np.stack(feats2, 1)
    emb["kernels_all"] = np.concatenate([emb["k3"], emb["k2"]], 1)
    print(f"kernel features: k3 {emb['k3'].shape}  k2 {emb['k2'].shape}")
else:
    print("!! kernel_info.json missing; skipping kernel embeddings")

# --- hand-made interpretable statistics ---------------------------------------------
g = gallery.astype(np.float64)
hist = np.stack([(g == k).mean(axis=(1, 2)) for k in range(NLEV)], 1)      # (G,16)
top = g[:, :3, :].mean(axis=(1, 2))
bot = g[:, 5:, :].mean(axis=(1, 2))
stats = np.stack([
    g.mean(axis=(1, 2)),                       # brightness
    g.std(axis=(1, 2)),                        # contrast
    top, bot, top - bot,                       # vertical structure
    g[:, 3, :].mean(1),                        # eye-row brightness
    g[:, 0, :].mean(1),                        # top-row brightness
    g[:, 7, :].mean(1),                        # bottom-row brightness
    np.abs(np.diff(g, axis=2)).mean(axis=(1, 2)),   # horizontal edge density
    np.abs(np.diff(g, axis=1)).mean(axis=(1, 2)),   # vertical edge density
    (g[:, :, :4] == g[:, :, 4:][:, :, ::-1]).mean(axis=(1, 2)),   # actual mirror rate
], 1)
emb["stats"] = stats
emb["hist"] = hist
emb["stats_hist"] = np.concatenate([stats, hist], 1)
print(f"hand-made features: stats {stats.shape}  hist {hist.shape}")

for k, v in emb.items():
    np.save(os.path.join(OUT, f"emb_{k}.npy"), v.astype(np.float32))
    print(f"  saved emb_{k}.npy  {v.shape}")

json.dump({
    "gallery_size": int(G_SIZE), "cols": COLS, "seed": SEED,
    "gallery_rows_hash": int(rows.sum()),
    "user24_rows_hash": int(user_rows.sum()),
    "splits": "uniformly sampled from the val split of data_v2_canonical",
    "embeddings": {k: list(v.shape) for k, v in emb.items()},
    "note": "index i of the gallery = row i of the contact sheet (row-major, 12 per row)",
}, open(os.path.join(OUT, "gallery_meta.json"), "w"), indent=2)

# ---------------------------------------------------------------- contact sheet with indices
TILE, PAD, LAB = 48, 6, 14
rowsn = (G_SIZE + COLS - 1) // COLS
W = COLS * (TILE + PAD) + PAD
H = rowsn * (TILE + PAD + LAB) + PAD
cv = Image.new("RGB", (W, H), (24, 26, 30))
dr = ImageDraw.Draw(cv)
for i in range(G_SIZE):
    r, c = divmod(i, COLS)
    x = PAD + c * (TILE + PAD)
    y = PAD + r * (TILE + PAD + LAB)
    tile = np.kron(LEVELS[gallery[i]], np.ones((TILE // 8, TILE // 8), dtype=np.uint8))
    tile = np.clip(tile, 0, 255).astype(np.uint8)
    cv.paste(Image.fromarray(tile, mode="L").convert("RGB"), (x, y))
    dr.text((x + 2, y + TILE + 1), str(i), fill=(180, 190, 200))
cv.save(os.path.join(OUT, "contact_sheet_indexed.png"))
print(f"wrote contact_sheet_indexed.png  ({G_SIZE} tiles, {COLS} per row)")
print("STYLE_EMBED_DONE")

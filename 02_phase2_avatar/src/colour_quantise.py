#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
The gap that the loss function cannot close: the models use 17-23 distinct colours per face,
the real data uses 7.

WHY THIS MATTERS MORE THAN THE LOSS
Measured on identical axes:

    real data          7.2 distinct colours   edge p90 15.56
    MSE model         17.5                   9.53
    GAN variants      19.8 - 23.3           8.3 - 9.4

So the models produce many slightly-different colours where the source art uses a handful of
clean ones. That single fact explains both complaints: a face made of twenty near-neighbours of
the same brown reads as dirty, and because the steps between them are small the edges read as
soft. The user's own labels point the same way -- fewer distinct colours is better (AUC 0.718)
and sharper edges is better (AUC 0.838).

WHAT IS TESTED
Snapping each face's colours onto a small set, two ways:

    per-face k-means   the best k colours for THIS face -- an upper bound on what any palette
                       can do, but k-means per face is expensive in hardware
    fixed palette       one shared set of k colours for every face -- cheap: an index plus a
                       lookup, and exactly what a redstone build would use

Both are post-processes, so they apply to any generator without retraining.
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "colour")
GAN = os.path.join(OUT, "gan")
WEB = os.path.join(ROOT, "webapp")
QOUT = os.path.join(OUT, "quantise")
os.makedirs(QOUT, exist_ok=True)
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def to_imgs(flat):
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    img = np.zeros((len(a), 8, 8, 3), dtype=np.float64)
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return img


def metrics(flat):
    img = to_imgs(flat)
    a = np.clip(flat, 0, 15).reshape(len(flat), 32, 3)
    x = img.reshape(len(img), 64, 3)
    mx = x.max(-1); mn = x.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    d = np.linalg.norm(a[:, :, None, :] - a[:, None, :, :], axis=-1)
    iu = np.triu_indices(32, 1)
    sep = d[:, iu[0], iu[1]].mean(1)
    dh = np.linalg.norm(img[:, :, 1:] - img[:, :, :-1], axis=-1)
    dv = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    ed = np.concatenate([dh.reshape(len(img), -1), dv.reshape(len(img), -1)], 1)
    ncol = np.array([len({tuple(np.rint(c).astype(int)) for c in a[i]}) for i in range(len(a))])
    return {"sat": float(sat.mean()), "sep": float(sep.mean()),
            "edge_p90": float(np.percentile(ed, 90)), "edge_mean": float(ed.mean()),
            "n_colours": float(ncol.mean())}


def kmeans(X, k, iters=20, seed=0):
    """X: (n, 32, 3). Returns (centres, labels) per face."""
    rng = np.random.default_rng(seed)
    n = len(X)
    C = np.empty((n, k, 3))
    L = np.empty((n, 32), dtype=int)
    for i in range(n):
        c = X[i][rng.choice(32, k, replace=False)].copy()
        for _ in range(iters):
            d = ((X[i][:, None, :] - c[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            for j in range(k):
                m = lab == j
                if m.any():
                    c[j] = X[i][m].mean(0)
        C[i] = c; L[i] = lab
    return C, L


z = np.load(os.path.join(OUT, "cache.npz"))
rgb96 = z["rgb96"].astype(np.float64)
cpc = z["cpc"]
M = 4000
sel = np.random.default_rng(3).choice(len(rgb96), M, replace=False)
REAL = rgb96[sel]
CT = cpc[sel][:, :8].astype(np.float32)


def mse_model():
    j = json.load(open(os.path.join(WEB, "colour_e6_model.json"), encoding="utf-8"))
    Ls = j["gen"]["layers"]
    seq = []
    for i, L in enumerate(Ls):
        W = torch.tensor(L["W"], dtype=torch.float32); b = torch.tensor(L["b"], dtype=torch.float32)
        lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            lin.weight.copy_(W); lin.bias.copy_(b)
        seq.append(lin)
        if i != len(Ls) - 1:
            seq.append(nn.ReLU())
    return nn.Sequential(*seq).eval()


m6 = mse_model()
with torch.no_grad():
    GEN = m6(torch.tensor(CT)).numpy() * 15.0

print("=" * 104)
print("QUANTISING THE COLOURS: can a small palette restore the real data's look?")
print("=" * 104)
print(f"  {'variant':30s} {'颜色数':>7s} {'饱和度':>8s} {'分离度':>8s} {'边缘p90':>9s} "
      f"{'边缘均值':>9s}")
print("  " + "-" * 76)
rows = []


def show(name, flat, store=True):
    m = metrics(flat)
    if store:
        rows.append((name, m))
    print(f"  {name:30s} {m['n_colours']:7.1f} {m['sat']:8.3f} {m['sep']:8.2f} "
          f"{m['edge_p90']:9.2f} {m['edge_mean']:9.2f}")
    return m


show("真实数据（参照）", REAL)
show("模型原始输出", GEN)

# ---- per-face k-means
X = np.clip(GEN, 0, 15).reshape(M, 32, 3)
for k in (4, 6, 8, 12, 16):
    C, L = kmeans(X, k, seed=1)
    q = C[np.arange(M)[:, None], L].reshape(M, 96)
    show(f"逐脸 k-means 到 {k} 色", q)

# ---- fixed palette (fit on real data, applied to everything)
PAL = REAL.reshape(M, 32, 3)
flat_pal = PAL.reshape(-1, 3)
for k in (8, 16, 32, 64):
    rng = np.random.default_rng(5)
    c = flat_pal[rng.choice(len(flat_pal), k, replace=False)].copy()
    for _ in range(30):
        d = ((flat_pal[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            mm = lab == j
            if mm.any():
                c[j] = flat_pal[mm].mean(0)
    d = ((X[:, :, None, :] - c[None, None, :, :]) ** 2).sum(-1)
    lab = d.argmin(-1)
    q = c[lab].reshape(M, 96)
    show(f"固定调色板 {k} 色", q)

print()
best = min(rows[1:], key=lambda r: abs(r[1]["n_colours"] - 7.2) + r[1]["sat"] * 10)
print(f"  最接近真实数据的一行：{best[0]}")
print("  注意彩色数是向下靠近 7 时，边缘锐度是否同时上升——两者应当一起变好。")

# ---- sheet
COLS, TILE, PAD = 14, 50, 5
keep = [r for r in rows if "k-means 到 8" in r[0] or "k-means 到 16" in r[0]
        or "调色板 16" in r[0] or "调色板 64" in r[0] or r[0] in
        ("真实数据（参照）", "模型原始输出")]
W = 290 + COLS * (TILE + PAD) + PAD
H = 92 + len(keep) * (TILE + 26) + 24
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(12), font(10)
dr.text((12, 12), "把颜色量化成少数几个：能否救回「清晰」", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), f"真实脸平均只用 7.2 种颜色，模型用 17.5 种。逐脸 k-means 是上限，"
        "固定调色板是红石可实现的版本。", font=f_s, fill=(150, 162, 178))
FLATS = {"真实数据（参照）": REAL, "模型原始输出": GEN}
C8, L8 = kmeans(X, 8, seed=1)
FLATS["逐脸 k-means 到 8 色"] = C8[np.arange(M)[:, None], L8].reshape(M, 96)
C16, L16 = kmeans(X, 16, seed=1)
FLATS["逐脸 k-means 到 16 色"] = C16[np.arange(M)[:, None], L16].reshape(M, 96)
y = 74
for name, m in keep:
    dr.text((12, y + 4), name[:24], font=f_l, fill=(110, 168, 254))
    dr.text((12, y + 20), f"{m['n_colours']:.1f} 色  边缘 {m['edge_p90']:.1f}", font=f_s,
            fill=(150, 162, 178))
    if name in FLATS:
        imgs = to_imgs(FLATS[name])
        for i in range(COLS):
            t = np.kron(np.clip(imgs[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                        np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
            cv.paste(Image.fromarray(t), (290 + i * (TILE + PAD), y))
    y += TILE + 26
p = os.path.join(QOUT, "QUANTISE.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

json.dump({n: m for n, m in rows}, open(os.path.join(QOUT, "quantise.json"), "w",
                                       encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(QOUT, 'quantise.json')}")
print("QUANTISE_DONE")

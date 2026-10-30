#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
The 4-PCA map: sample each of the four grayscale knobs at eight levels and show every face.

THE GRID
Each knob takes -3.5, -2.5, ..., +3.5, so eight levels each and 8^4 = 4096 faces. Four dimensions
are shown by splitting them across the page:

    rows        PC0  (8)
    columns     PC1  x  PC2   (8 x 8 = 64), drawn as eight blocks of eight
    one image   per PC3 value  (8 images)

Each image therefore holds 8 x 64 = 512 faces and the eight images together cover all 4096.

WHY SPLIT THIS WAY RATHER THAN ONE GIANT SHEET
4096 faces on one sheet means 20 px per face, which is too small to judge. Sixty-four faces wide
by eight tall at 36 px keeps every face legible, and the PC3 slices are cheap to flip through.

ZERO IS SKIPPED
The eight levels are the half-integers, not 0, 1, 2, ... -- sampling exactly 0 puts a face at the
distribution mean, which is the flattest possible output, and four of those would waste a tenth of
the grid.
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
WEB = os.path.join(ROOT, "webapp")
OUT = os.path.join(ROOT, "results", "pca4map")
os.makedirs(OUT, exist_ok=True)
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


j = json.load(open(os.path.join(WEB, "pca22_dim4w64L1_model.json"), encoding="utf-8"))
Ls = j["gen"]["layers"]
seq = []
for i, L in enumerate(Ls):
    W = torch.tensor(L["W"], dtype=torch.float32)
    b = torch.tensor(L["b"], dtype=torch.float32)
    lin = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        lin.weight.copy_(W); lin.bias.copy_(b)
    seq.append(lin)
    if i != len(Ls) - 1:
        seq.append(nn.ReLU())
net = nn.Sequential(*seq).eval()

LEVELS = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]


def render_grid(p3):
    """512 faces: rows = PC0, cols = PC1 x PC2."""
    pts = []
    for c0 in LEVELS:
        for c1 in LEVELS:
            for c2 in LEVELS:
                pts.append([c0, c1, c2, p3])
    Z = np.array(pts, dtype=np.float32)
    with torch.no_grad():
        v = np.clip(net(torch.tensor(Z)).numpy(), 0, 15)
    lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
    lv = np.rint(np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)).astype(int)
    img = np.zeros((len(lv), 8, 8, 3))
    a = np.clip(lv, 0, 15).reshape(len(lv), 32)          # grayscale: one value per position
    img[:, layout.POS_ROW, layout.POS_COL] = a[:, :, None]
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a[:, :, None]
    return img.reshape(8, 8, 8, 8, 8, 3)     # PC0, PC1, PC2, row, col, rgb


TILE, PAD = 36, 2
BLOCKGAP = 10
BW = 8 * (TILE + PAD)                 # one PC1 block = 8 faces wide
W = 90 + 8 * BW + 7 * BLOCKGAP + 16
H = 96 + 8 * (TILE + PAD) + 18
f_t, f_l, f_s = font(20), font(13), font(10)

for pi, p3 in enumerate(LEVELS):
    g = render_grid(p3)
    cv = Image.new("RGB", (W, H), (18, 20, 24))
    dr = ImageDraw.Draw(cv)
    dr.text((12, 12), f"4-PCA 地图　PC3 = {p3:+.1f}", font=f_t, fill=(234, 238, 244))
    dr.text((12, 40), "行 = PC0（8 值），横轴 = PC1（8 组）× PC2（每组 8 张）；"
                      "每张都是同一批等级 −3.5…+3.5。", font=f_s, fill=(150, 162, 178))
    dr.text((12, 56), "模型 4-64-32（2,304 乘法），min-max 拉伸到 16 级。",
            font=f_s, fill=(130, 142, 158))
    for c1 in range(8):
        x0 = 90 + c1 * (BW + BLOCKGAP)
        dr.text((x0 + 2, 74), f"PC1={LEVELS[c1]:+.1f}", font=f_s, fill=(110, 168, 254))
    for c0 in range(8):
        y0 = 88 + c0 * (TILE + PAD)
        dr.text((12, y0 + 10), f"PC0={LEVELS[c0]:+.1f}", font=f_l, fill=(200, 210, 226))
        for c1 in range(8):
            for c2 in range(8):
                x = 90 + c1 * (BW + BLOCKGAP) + c2 * (TILE + PAD)
                t = np.kron(np.clip(g[c0, c1, c2] / 15.0 * 255, 0, 255).astype(np.uint8),
                            np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
                cv.paste(Image.fromarray(t), (x, y0))
    tag = f"{p3:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
    p = os.path.join(OUT, f"MAP_p3_{tag}.png")
    cv.save(p)
    print(f"  wrote {os.path.basename(p)}  {cv.size}")

# ---- candidate presets proposed by the 二次元 probe, to be verified by eye, not trusted
sc = json.load(open(os.path.join(WEB, "scorer_anime.json"), encoding="utf-8"))


def score(x):
    v = x
    for L in sc["layers"]:
        v = v @ np.array(L["W"]).T + np.array(L["b"])
        if L["relu"]:
            v = np.maximum(0, v)
    return v.ravel()


grid = np.array([[a, b, c, d] for a in LEVELS for b in LEVELS for c in LEVELS for d in LEVELS],
                dtype=np.float32)
s = score(grid)
order = np.argsort(-s)[:24]
with torch.no_grad():
    v = np.clip(net(torch.tensor(grid[order])).numpy(), 0, 15)
lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
lv = np.rint(np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)).astype(int)
a = np.clip(lv, 0, 15).reshape(len(lv), 32)
img = np.zeros((len(a), 8, 8, 3))
img[:, layout.POS_ROW, layout.POS_COL] = a[:, :, None]
img[:, layout.POS_ROW, layout.MIRROR_COL] = a[:, :, None]

T2 = 72
W2 = 300 + 12 * (T2 + 6) + 6
H2 = 76 + 2 * (T2 + 26) + 20
cv2 = Image.new("RGB", (W2, H2), (18, 20, 24))
dr2 = ImageDraw.Draw(cv2)
dr2.text((12, 12), "二次元打分器最高分的 24 个网格坐标", font=font(18), fill=(234, 238, 244))
dr2.text((12, 38), "只是候选，不是结论——这个打分器的 AUC 只有约 0.6，必须自己看。",
         font=f_s, fill=(224, 192, 90))
for k, i in enumerate(order[:24]):
    r, c = divmod(k, 12)
    x0 = 300 + c * (T2 + 6)
    y0 = 66 + r * (T2 + 26)
    t = np.kron(np.clip(img[k] / 15.0 * 255, 0, 255).astype(np.uint8),
                np.ones((T2 // 8, T2 // 8, 1))).astype(np.uint8)
    cv2.paste(Image.fromarray(t), (x0, y0))
    dr2.text((x0, y0 + T2 + 2),
             f"{grid[i][0]:+.1f},{grid[i][1]:+.1f},{grid[i][2]:+.1f},{grid[i][3]:+.1f}",
             font=font(9), fill=(150, 162, 178))
p2 = os.path.join(OUT, "CANDIDATES.png")
cv2.save(p2)
print(f"  wrote {os.path.basename(p2)}  {cv2.size}")
json.dump({"levels": LEVELS,
           "presets": [{"pc": [float(v) for v in grid[i]],
                        "score": float(s[i])} for i in order]},
          open(os.path.join(OUT, "presets.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("PCA4MAP_DONE")

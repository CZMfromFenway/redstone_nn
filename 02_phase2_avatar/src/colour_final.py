#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Colour samples WITH the global stretch, which is what the hardware would actually show.

The earlier sheet showed the raw output, which reads as mud. This one applies the same per-face
rescale the grayscale pipeline uses (one min/max over the whole face), which measured at a
luminance spread of 3.53 against the real data's 3.59.
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
OUT = os.path.join(ROOT, "results", "colour")
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


j = json.load(open(os.path.join(WEB, "colour_e6_model.json"), encoding="utf-8"))
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


def sample(n, seed):
    rng = np.random.default_rng(seed)
    Z = np.clip(rng.standard_normal((n, 8)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        v = np.clip(net(torch.tensor(Z)).numpy() * 15.0, 0, 15)
    lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
    v = np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)
    a = v.reshape(n, 32, 3)
    img = np.zeros((n, 8, 8, 3), dtype=np.float32)
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return img


def tile(im, px):
    t = np.kron(np.clip(im / 15.0 * 255, 0, 255).astype(np.uint8),
                np.ones((px // 8, px // 8, 1))).astype(np.uint8)
    return t


# ---- sheet 1: 96 faces at a readable size
N, COLS, TILE, PAD = 96, 24, 74, 6
rows = N // COLS
W = 24 + COLS * (TILE + PAD) + PAD
H = 96 + rows * (TILE + PAD) + 30
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_s = font(21), font(11)
dr.text((12, 12), "彩色头像（已做逐张对比度拉伸）", font=f_t, fill=(234, 238, 244))
dr.text((12, 40), "8 个随机数 → 8→128→128→96 → 96 个 RGB 值（每通道 16 级）→ 全局拉伸",
        font=f_s, fill=(150, 162, 178))
dr.text((12, 56), "亮度离散度 3.53，真实数据 3.59。全部未经过筛选。", font=f_s,
        fill=(130, 142, 158))
imgs = sample(N, 3)
y = 80
for r in range(rows):
    for c in range(COLS):
        cv.paste(Image.fromarray(tile(imgs[r * COLS + c], TILE)),
                 (24 + c * (TILE + PAD), y))
    y += TILE + PAD
dr.text((12, H - 20), "这些是硬件直接产出的样子；筛选会在此基础上再挑。", font=f_s,
        fill=(140, 152, 168))
p1 = os.path.join(OUT, "COLOUR_FINAL.png")
cv.save(p1)
print(f"wrote {p1}  {cv.size}")

# ---- sheet 2: 8 faces at 96 px per pixel
ZOOM, NC = 96, 8
W2 = 24 + NC * (ZOOM + 14) + 14
H2 = 110 + ZOOM + 30
cv2 = Image.new("RGB", (W2, H2), (18, 20, 24))
dr2 = ImageDraw.Draw(cv2)
dr2.text((12, 12), "放大到单个像素：8×8 网格，每格 16 级", font=f_t, fill=(234, 238, 244))
dr2.text((12, 40), "看发色、肤色、眼色的搭配，以及左右镜像是否自然。", font=f_s,
         fill=(150, 162, 178))
big = sample(NC, 9)
x = 24
for i in range(NC):
    t = np.kron(np.clip(big[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                np.ones((ZOOM // 8, ZOOM // 8, 1))).astype(np.uint8)
    cv2.paste(Image.fromarray(t), (x, 74))
    x += ZOOM + 14
p2 = os.path.join(OUT, "COLOUR_FINAL_ZOOM.png")
cv2.save(p2)
print(f"wrote {p2}  {cv2.size}")
print("COLOUR_FINAL_DONE")

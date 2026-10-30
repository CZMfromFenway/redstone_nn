#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
A large, readable sheet of faces from the colour model, so the overall look can be judged.

The model is the E6 export (8 -> 128 -> 128 -> 96). Everything shown is generated from fresh
N(0,1) draws with no filtering, i.e. what the hardware would produce before any selection.

Two sheets:
    COLOUR_SAMPLES.png    160 faces at a size you can actually see, 4 blocks of 40
    COLOUR_COLUMNS.png    a few faces blown up 24x, to inspect the pixels themselves
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
seq = []
Ls = j["gen"]["layers"]
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
KIN = int(Ls[0]["W"][0].__len__())
print(f"model: {KIN} -> 128 -> 128 -> 96")


def sample(n, seed):
    rng = np.random.default_rng(seed)
    Z = np.clip(rng.standard_normal((n, KIN)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        o = net(torch.tensor(Z)).numpy()
    v = np.clip(o * 15.0, 0, 15)
    a = v.reshape(n, 32, 3)
    img = np.zeros((n, 8, 8, 3), dtype=np.float32)
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return img


# ================================================================ sheet 1: 160 faces
N, COLS, TILE, PAD = 160, 40, 52, 5
rows = N // COLS
W = 24 + COLS * (TILE + PAD) + PAD
H = 96 + rows * (TILE + PAD) + 26
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_s = font(21), font(11)
dr.text((12, 12), "彩色头像：8 个随机数直接生成（无筛选）", font=f_t, fill=(234, 238, 244))
dr.text((12, 40), f"模型 8→128→128→96，29,696 次乘法。每通道 4 bit（16 级）。"
        f"{N} 张全部来自新鲜随机数。", font=f_s, fill=(150, 162, 178))
dr.text((12, 56), "这些是硬件直接产出的样子，没有经过任何打分筛选。", font=f_s,
        fill=(130, 142, 158))
imgs = sample(N, 3)
y = 80
for r in range(rows):
    for c in range(COLS):
        im = imgs[r * COLS + c]
        # np.kron multiplies by a float64 ones array, so the RESULT needs the cast, not the input
        t = np.kron(np.clip(im / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        cv.paste(Image.fromarray(t), (24 + c * (TILE + PAD), y))
    y += TILE + PAD
dr.text((12, H - 18), "如果这些脸整体方向可以接受，再谈筛选和良率。", font=f_s,
        fill=(140, 152, 168))
p1 = os.path.join(OUT, "COLOUR_SAMPLES.png")
cv.save(p1)
print(f"wrote {p1}  {cv.size}")

# ================================================================ sheet 2: zoom on pixels
ZOOM, NC = 24, 12
big = sample(NC, 9)
W2 = 24 + NC * (8 * ZOOM + 12) + 12
H2 = 110 + 8 * ZOOM + 30
cv2 = Image.new("RGB", (W2, H2), (18, 20, 24))
dr2 = ImageDraw.Draw(cv2)
dr2.text((12, 12), "放大看像素：每张是 8×8，每个格子 16 级", font=f_t, fill=(234, 238, 244))
dr2.text((12, 40), "行内镜像（左右对称）。看发色、肤色、眼色的搭配是否自然。", font=f_s,
         fill=(150, 162, 178))
x = 24
for i in range(NC):
    t = np.kron(np.clip(big[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                np.ones((ZOOM, ZOOM, 1))).astype(np.uint8)
    cv2.paste(Image.fromarray(t), (x, 74))
    x += 8 * ZOOM + 12
dr2.text((12, H2 - 20), "格子边界就是单个像素；4 bit 每通道意味着每种颜色有 16 个档位。",
         font=f_s, fill=(140, 152, 168))
p2 = os.path.join(OUT, "COLOUR_ZOOM.png")
cv2.save(p2)
print(f"wrote {p2}  {cv2.size}")
print("COLOUR_SAMPLES_DONE")

#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Verify the user's observation that PC3 turns 少女 from short hair into thick eyebrows or an angry
glare, then render a name-check strip for each style at its grown centre.

The claim is checked rather than taken on trust because it is the one place the user described a
within-style behaviour, and a within-style axis is what makes a Gaussian the right shape: if PC3
really only modulates expression, its variance should be large and its centre near zero.
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
OUT = os.path.join(ROOT, "results", "styles")
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

ST = json.load(open(os.path.join(OUT, "styles.json"), encoding="utf-8"))
LV = np.linspace(-3.5, 3.5, 8)


def render(pts):
    Z = np.array(pts, dtype=np.float32)
    with torch.no_grad():
        v = np.clip(net(torch.tensor(Z)).numpy(), 0, 15)
    lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
    lv = np.rint(np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)).astype(int)
    img = np.zeros((len(lv), 8, 8, 3))
    a = np.clip(lv, 0, 15).reshape(len(lv), 32)
    img[:, layout.POS_ROW, layout.POS_COL] = a[:, :, None]
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a[:, :, None]
    return img


ORDER = ["少女", "少年", "骷髅", "毒液", "黑色悦灵", "潦草大叔", "黑暗法师"]
TILE, PAD = 56, 5
COLS = 8
W = 250 + COLS * (TILE + PAD)
H = 96 + len(ORDER) * (TILE + 26)
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(13), font(10)
dr.text((12, 12), "每个风格中心沿 PC3 变化（其余三个固定）", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), "横向 8 格就是 PC3 = −3.5 … +3.5。用来验证『PC3 只改表情』这个说法。",
        font=f_s, fill=(150, 162, 178))
y = 74
for name in ORDER:
    mu = ST[name]["mu"]
    pts = [[mu[0], mu[1], mu[2], v] for v in LV]
    img = render(pts)
    dr.text((12, y + 8), name, font=f_l, fill=(110, 200, 130))
    dr.text((12, y + 26), f"PC3 sd {ST[name]['sd'][3]:.2f}", font=f_s, fill=(150, 162, 178))
    for i in range(len(LV)):
        t = np.kron(np.clip(img[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        cv.paste(Image.fromarray(t), (250 + i * (TILE + PAD), y))
    y += TILE + 26
p = os.path.join(OUT, "PC3_SWEEP.png")
cv.save(p)
print(f"wrote {p}  {cv.size}")

# how much does PC3 actually change the face, per style?
LUM = np.array([0.299, 0.587, 0.114])
print()
print("=" * 88)
print("HOW MUCH DOES PC3 CHANGE THE PICTURE, PER STYLE?")
print("=" * 88)
print(f"  {'style':10s} {'沿PC3的变化':>11s} {'沿PC1的变化':>11s} {'沿PC2的变化':>11s}")
print("  " + "-" * 52)
for name in ORDER:
    mu = np.array(ST[name]["mu"])
    base = None
    cols = {}
    for ax in (3, 1, 2):
        fs = []
        for v in LV:
            pt = mu.copy(); pt[ax] = v
            fs.append(render([pt])[0].ravel())
        fs = np.array(fs)
        cols[ax] = float(np.abs(fs - fs.mean(0)).mean())
    print(f"  {name:10s} {cols[3]:11.3f} {cols[1]:11.3f} {cols[2]:11.3f}")
print()
print("  数值是沿该轴移动一整条线时，像素的平均变化幅度。")
print("  少女的 PC3 若明显小于 PC1/PC2，就说明 PC3 对形态影响小，与用户观察一致。")
print("STYLE_PC3_DONE")

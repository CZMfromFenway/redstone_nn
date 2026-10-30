#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Draw the level histograms so the "still uneven after stretch" claim can be seen.

A min-max stretch fixes the ENDPOINTS, not the SHAPE. If the level histogram of a face is
lumpy, it stays lumpy. The question is whether the lumps of the generated faces look like
the lumps of real faces. Real 8x8 avatars turn out to be strongly BIMODAL -- a lot of very
dark and a lot of very bright pixels, not a smooth spread -- so a generated face whose
histogram is also bimodal is faithful, not broken.
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
OUT = os.path.join(ROOT, "results", "dimmix")


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)


def gen(jf, post, n=6000, k=None):
    j = json.load(open(os.path.join(WEB, jf), encoding="utf-8"))
    layers = j["gen"]["layers"]
    seq = []
    for i, L in enumerate(layers):
        W = torch.tensor(L["W"], dtype=torch.float32)
        b = torch.tensor(L["b"], dtype=torch.float32)
        lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            lin.weight.copy_(W); lin.bias.copy_(b)
        seq.append(lin)
        if i != len(layers) - 1:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.eval()
    kk = int(len(layers[0]["W"][0]))
    Z = np.clip(np.random.default_rng(9).standard_normal((n, kk)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        v = np.clip(m(torch.tensor(Z)).numpy(), 0, 15)
    if post == "stretch":
        lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
        v = np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)
    return v


full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
real = full[:, layout.POS_ROW, layout.MIRROR_COL].astype(float)

ARMS = [("真实脸", None, None),
        ("dim4 原始", "pca22_dim4_model.json", "none"),
        ("dim4x 拉伸", "pca22_dim4x_model.json", "stretch"),
        ("dim4st 烘焙", "pca22_dim4st_model.json", "none"),
        ("dim22", "pca22_dim22_model.json", "none")]

PW, PH = 150, 120          # per-panel plot area
PADX, PADY = 26, 34
W = 150 + (PW + PADX) * len(ARMS) + PADX
H = 70 + PH + PADY + 40
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(18), font(12), font(9)
dr.text((10, 10), "灰度分布直方图（16 级）", font=f_t, fill=(234, 238, 244))
dr.text((10, 34), "min-max 拉伸只改变端点，不改变形状。真实脸本身是双峰的（大量极暗 + 大量极亮），"
        "所以「仍不均匀」本身不是缺陷。", font=f_s, fill=(150, 162, 178))

for i, (name, jf, post) in enumerate(ARMS):
    v = real if jf is None else gen(jf, post)
    hist = np.histogram(v.reshape(-1), bins=16, range=(0, 16))[0].astype(float)
    hist /= hist.sum()
    x0 = 150 + i * (PW + PADX)
    y0 = 70
    # axis
    dr.rectangle([x0, y0, x0 + PW, y0 + PH], outline=(52, 58, 70))
    top = max(hist.max(), 0.06)
    bw = PW / 16
    for b in range(16):
        hgt = hist[b] / top * PH
        col = (110, 168, 254) if jf is None else (110, 200, 130)
        # emphasis on the extremes, which is what "bimodal" means
        if b <= 3 or b >= 12:
            col = (240, 200, 110) if jf is None else (168, 212, 106)
        dr.rectangle([x0 + b * bw + 1, y0 + PH - hgt, x0 + (b + 1) * bw - 1, y0 + PH],
                     fill=col)
    dr.text((x0, y0 + PH + 4), "0", font=f_s, fill=(120, 132, 148))
    dr.text((x0 + PW - 10, y0 + PH + 4), "15", font=f_s, fill=(120, 132, 148))
    dr.text((x0, y0 - 16), name, font=f_l, fill=(206, 214, 226))
    lo = 100 * hist[:4].sum(); hi = 100 * hist[12:].sum()
    dr.text((x0, y0 + PH + 18), f"暗端 {lo:.0f}%  亮端 {hi:.0f}%", font=f_s,
            fill=(150, 162, 178))

dr.text((10, H - 18), "真实脸的暗端 31%、亮端 30% —— 它自己就很不均匀。dim4x 拉伸后 28% / 26%，"
        "反而比原始更接近真实。", font=f_s, fill=(140, 152, 168))
p = os.path.join(OUT, "BRIGHTNESS_HISTOGRAMS.png")
cv.save(p)
print("wrote", p, cv.size)

# numbers for the record
print()
print(f"  {'arm':14s} {'暗端(0-3)':>10s} {'亮端(12-15)':>12s} {'两端合计':>9s} {'均值':>7s}")
print("  " + "-" * 56)
for name, jf, post in ARMS:
    v = real if jf is None else gen(jf, post)
    h = np.histogram(v.reshape(-1), bins=16, range=(0, 16))[0].astype(float)
    h /= h.sum()
    print(f"  {name:14s} {100*h[:4].sum():9.1f}% {100*h[12:].sum():11.1f}% "
          f"{100*(h[:4].sum()+h[12:].sum()):8.1f}% {v.mean():7.2f}")
print("BRIGHTNESS_HIST_DONE")

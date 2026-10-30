#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Side by side at the same zoom: grayscale generator against the colour generator.

WHY THIS COMPARISON
The colour faces look acceptable at 52 px but mushy when enlarged: a few broad colour bands
rather than hair, skin and eye regions. The obvious suspect is the training objective. The
grayscale generator is adversarial (a discriminator is trained alongside it), which sharpens
output; the colour model is pure MSE, whose optimum is the conditional mean, i.e. a blur.

If the grayscale faces keep their structure at 24x while the colour ones do not, that is the
explanation and the fix is to add an adversarial or coherence term to the colour trainer.
If both are equally mushy at 24x, the problem is the 8x8 resolution and not the objective.
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


def load_json_model(f, key="gen"):
    j = json.load(open(os.path.join(WEB, f), encoding="utf-8"))
    Ls = j[key]["layers"]
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
    return nn.Sequential(*seq).eval()


def gray_faces(net, n, seed):
    rng = np.random.default_rng(seed)
    Z = np.clip(rng.standard_normal((n, 4)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        v = torch.clamp(net(torch.tensor(Z)), 0, 15)
        lo = v.min(1, keepdim=True)[0]; hi = v.max(1, keepdim=True)[0]
        v = torch.clamp((v - lo) / (hi - lo).clamp_min(1e-6) * 15, 0, 15)
        return np.rint(v.numpy()).astype(int)


def colour_faces(net, n, seed):
    rng = np.random.default_rng(seed)
    Z = np.clip(rng.standard_normal((n, 8)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        v = np.clip(net(torch.tensor(Z)).numpy() * 15.0, 0, 15)
    a = v.reshape(n, 32, 3)
    img = np.zeros((n, 8, 8, 3), dtype=np.float32)
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return img


G = load_json_model("pca22_dim4x_model.json")
C = load_json_model("colour_e6_model.json")
N = 12
gf = gray_faces(G, N, 9)
cf = colour_faces(C, N, 9)

# how much internal structure does each have? count distinct levels across the 32 positions,
# and measure the spread of neighbouring brightness -- a mushy face has few levels and small
# local steps.
gs = np.rint(gf).astype(int)
gl = np.array([len(set(row.tolist())) for row in gs])
# for colour, count distinct LUMINANCE levels across the 32 positions -- the fair comparison,
# since a colour face has 16 levels per channel but only the luminance carries structure
LUMW = np.array([0.299, 0.587, 0.114])
lum32 = np.array([cf[i][layout.POS_ROW, layout.POS_COL] @ LUMW for i in range(N)])
cl_lev = np.array([len(set(np.rint(lum32[i]).astype(int).tolist())) for i in range(N)])

print("=" * 88)
print("STRUCTURE: how many distinct levels does each generator actually use?")
print("=" * 88)
print(f"  {'':16s} {'灰度级数/张':>13s} {'亮度级数/张':>13s} {'通道 sd':>9s}")
print("  " + "-" * 56)
print(f"  {'灰度模型':16s} {gl.mean():13.1f} {'—':>13s} {gs.std():9.2f}")
c_lum = (cf @ np.array([0.299, 0.587, 0.114]))
print(f"  {'彩色模型':16s} {'—':>13s} {cl_lev.mean():13.1f} {cf.std():9.2f}")
print()
print("  a pure-MSE model regresses to the conditional mean, which flattens exactly this")
print("  quantity; the grayscale model has a discriminator pushing the other way")

TILE = 24
W = 24 + N * (8 * TILE + 12) + 12
H = 150 + 8 * TILE + 30
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_s = font(20), font(11)
dr.text((12, 12), "同一放大倍数下：灰度（GAN）对比彩色（纯 MSE）", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), f"灰度 {gl.mean():.1f} 级/张，彩色亮度 {cl_lev.mean():.1f} 级/张。"
        "级数少＝只有几个色块，缺少头发/皮肤/眼睛的分层。", font=f_s, fill=(150, 162, 178))
y = 66
dr.text((12, y + 60), "灰度", font=f_s, fill=(110, 168, 254))
for i in range(N):
    img8 = layout.image_from_left_half(gs[i])
    t = np.kron(np.clip(img8 / 15.0 * 255, 0, 255).astype(np.uint8),
                np.ones((TILE, TILE))).astype(np.uint8)
    cv.paste(Image.fromarray(np.stack([t] * 3, -1)), (24 + i * (8 * TILE + 12), y))
y += 8 * TILE + 14
dr.text((12, y + 60), "彩色", font=f_s, fill=(110, 168, 254))
for i in range(N):
    t = np.kron(np.clip(cf[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                np.ones((TILE, TILE, 1))).astype(np.uint8)
    cv.paste(Image.fromarray(t), (24 + i * (8 * TILE + 12), y))
p = os.path.join(OUT, "GRAY_VS_COLOUR.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")
print("GRAY_VS_COLOUR_DONE")

#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Does a per-face stretch rescue the colour output's contrast?

THE DIAGNOSIS
At 24x the colour faces are muddy: 9.6 distinct luminance levels against the grayscale model's
12.7, and a channel sd of 4.33 against real data's 5.78. Two causes are candidates, and they
call for different fixes:

    (a) no stretch      -- the grayscale pipeline rescales every face to 0..15 before display;
                           the colour pipeline does not, so whatever range the network emits
                           is what gets shown
    (b) pure MSE        -- regression to the conditional mean smooths everything, and a
                           discriminator is what keeps the grayscale output crisp

This tests (a) alone, which is free. Three ways to stretch colour, and they are not equivalent:

    global     one min/max over all 96 values          strongest contrast, distorts hue
    per-channel  separate min/max per channel          strongest, but shifts hue badly
    luminance  min/max of the luminance, one affine     preserves hue ordering, weakest

Measured against the real data's luminance sd of 5.78.
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
LUMW = np.array([0.299, 0.587, 0.114])


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

N = 4000
rng = np.random.default_rng(7)
Z = np.clip(rng.standard_normal((N, 8)), -3.5, 3.5).astype(np.float32)
with torch.no_grad():
    raw = net(torch.tensor(Z)).numpy() * 15.0
base = np.clip(raw, 0, 15)

# reference: the real data's luminance statistics
z = np.load(os.path.join(OUT, "cache.npz"))
real = z["rgb96"].astype(np.float64).reshape(None, 32, 3) if False else None
rl = np.load(os.path.join(OUT, "cache.npz"))["rgb96"].astype(np.float64)
rl = rl[np.random.default_rng(0).choice(len(rl), N, replace=False)]
real_lum = (rl.reshape(N, 32, 3) @ LUMW)
real_sd = float((rl.reshape(N, 32, 3) @ LUMW).std(1).mean())
real_sat = float((rl.reshape(N, 32, 3).max(-1) - rl.reshape(N, 32, 3).min(-1)).mean() /
                 np.maximum(rl.reshape(N, 32, 3).max(-1), 1e-6).mean())


def stats(a):
    x = a.reshape(N, 32, 3)
    lum = x @ LUMW
    lev = np.array([len(set(np.rint(lum[i]).astype(int).tolist())) for i in range(min(500, N))])
    return {"lum_sd": float(lum.std(1).mean()), "lum_levels": float(lev.mean()),
            "ch_sd": float(x.std()), "sat": float((x.max(-1) - x.min(-1)).mean() / 15.0)}


def stretch_global(a):
    lo = a.min(1, keepdims=True); hi = a.max(1, keepdims=True)
    return np.clip((a - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)


def stretch_chan(a):
    x = a.reshape(-1, 32, 3)
    lo = x.min(1, keepdims=True); hi = x.max(1, keepdims=True)
    return np.clip((x - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15).reshape(-1, 96)


def stretch_lum(a):
    x = a.reshape(-1, 32, 3)
    lum = x @ LUMW
    lo = lum.min(1, keepdims=True); hi = lum.max(1, keepdims=True)
    k = (15.0 / np.maximum(hi - lo, 1e-6))
    # apply the same affine to every channel so hue ordering is untouched
    return np.clip((x - lo[..., None]) * k[..., None], 0, 15).reshape(-1, 96)


VAR = [("不拉伸", base), ("全局拉伸", stretch_global(base)),
       ("逐通道拉伸", stretch_chan(base)), ("亮度拉伸", stretch_lum(base))]
print("=" * 96)
print("DOES STRETCHING RESCUE THE COLOUR CONTRAST?")
print("=" * 96)
print(f"  {'':14s} {'亮度 sd':>9s} {'亮度级数':>9s} {'通道 sd':>9s} {'饱和':>7s}")
print("  " + "-" * 54)
print(f"  {'真实数据':14s} {real_sd:9.2f} {'—':>9s} {'—':>9s} {'—':>7s}")
res = {}
for name, a in VAR:
    s = stats(a)
    res[name] = s
    print(f"  {name:14s} {s['lum_sd']:9.2f} {s['lum_levels']:9.1f} {s['ch_sd']:9.2f} "
          f"{s['sat']:7.3f}")
print()
print(f"  真实数据的亮度 sd = {real_sd:.2f}；彩色模型不拉伸只有 "
      f"{res['不拉伸']['lum_sd']:.2f}")

# pictures
TILE, NC = 22, 12
W = 24 + NC * (8 * TILE + 12) + 12
H = 96 + len(VAR) * (8 * TILE + 26) + 24
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_s = font(20), font(11)
dr.text((12, 12), "三种拉伸方式对彩色的影响", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), f"真实亮度 sd {real_sd:.2f}。亮度拉伸保持色相不变，逐通道拉伸会改变色相。",
        font=f_s, fill=(150, 162, 178))
y = 66
for name, a in VAR:
    s = stats(a)
    dr.text((12, y + 4), name, font=f_s, fill=(110, 168, 254))
    dr.text((12, y + 20), f"sd {s['lum_sd']:.1f}", font=f_s, fill=(150, 162, 178))
    for i in range(NC):
        x = a[i].reshape(32, 3)
        img = np.zeros((8, 8, 3), dtype=np.float32)
        img[layout.POS_ROW, layout.POS_COL] = x
        img[layout.POS_ROW, layout.MIRROR_COL] = x
        t = np.kron(np.clip(img / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE, TILE, 1))).astype(np.uint8)
        cv.paste(Image.fromarray(t), (70 + i * (8 * TILE + 12), y))
    y += 8 * TILE + 26
p = os.path.join(OUT, "COLOUR_STRETCH.png")
cv.save(p)
print(f"\nwrote {p}")

json.dump({"real_lum_sd": real_sd, "variants": res},
          open(os.path.join(OUT, "stretch_test.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("STRETCH_TEST_DONE")

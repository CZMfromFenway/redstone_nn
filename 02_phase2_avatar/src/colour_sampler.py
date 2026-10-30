#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Where do the colour components come from? This decides the whole colour design.

WHAT THE NUMBERS SAID
Reconstruction (feed the TRUE 4 PCs and the TRUE colour components):
    E3  gray + colour 8    RGB MAE 1.369   hue 0.044
    E6  colour 8 ONLY      RGB MAE 1.349   hue 0.043
E6 matches E3 with no gray input at all, because the colour PCA was computed on the full RGB
image, so its leading components already carry the structure. Stage 1's output is redundant
once colour components are present.

WHAT THE GENERATION PICTURES SAID
The two-stage rows (E2..E5) produced neon colours: the colour components handed to them came
from a DIFFERENT face than the gray, so the colour and the structure contradict each other.
E6 produced coherent, natural faces -- because its colour components and its output describe
the same face.

THE CONSEQUENCE
Colour components cannot be derived from the four grayscale PCs (a linear readout from
lightness to hue had R^2 = 0.06). So they must be SAMPLED. This script tests whether sampling
them from a fitted Gaussian gives usable faces, and compares three samplers:

    joint      z ~ N(0, Sigma_c)   with Sigma_c the empirical 8x8 covariance
    independent  z ~ N(0, I)        ignores the correlations between components
    real rows   rows drawn from the training set (the ceiling)

If `joint` is as good as `real rows`, the deployable colour generator is just: eight random
numbers, an 8x8 covariance, then the 8->96 network.
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


z = np.load(os.path.join(OUT, "cache.npz"))
cpc = z["cpc"]
rgb96 = z["rgb96"].astype(np.float32)
K = 8
C = cpc[:, :K].astype(np.float64)
Sig = np.cov(C.T)
ev = np.linalg.eigvalsh(Sig)
print("=" * 100)
print("SAMPLING THE COLOUR COMPONENTS")
print("=" * 100)
print(f"  covariance of the first {K} colour components: eigenvalues "
      f"{', '.join(f'{v:.2f}' for v in ev)}")
print(f"  condition number {ev.max()/max(ev.min(),1e-9):.1f}")
off = Sig - np.diag(np.diag(Sig))
print(f"  mean |off-diagonal correlation| = {np.mean(np.abs(off/np.sqrt(np.outer(np.diag(Sig),np.diag(Sig))))):.3f}")
print("  (near 0 would mean the components are already independent and N(0,I) would do)")

m = torch.load(os.path.join(OUT, "stage2_E6.pt"), weights_only=False)
net = nn.Sequential(nn.Linear(m["d_in"], m["hid"]), nn.ReLU(),
                    nn.Linear(m["hid"], m["hid"]), nn.ReLU(), nn.Linear(m["hid"], 96))
net.load_state_dict(m["state"]); net.eval()


def gen(Z):
    with torch.no_grad():
        return np.clip(net(torch.tensor(Z.astype(np.float32))).numpy() * 15.0, 0, 15)


def rgb_image(a):
    a = np.clip(a, 0, 15).reshape(-1, 32, 3)
    img = np.zeros((len(a), 8, 8, 3), dtype=np.float32)
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    return np.clip(img / 15.0 * 255.0, 0, 255).astype(np.uint8)


def stats(a):
    x = np.clip(a, 0, 15).reshape(-1, 32, 3)
    mx = x.max(-1); mn = x.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    lum = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    return float(sat.mean()), float(lum.mean()), float(x.std())


NS = 22
rng = np.random.default_rng(5)
L = np.linalg.cholesky(Sig + 1e-9 * np.eye(K))
Z_joint = (rng.standard_normal((NS, K)) @ L.T)
Z_indep = rng.standard_normal((NS, K))
Z_real = C[rng.choice(len(C), NS, replace=False)]

rows = [("真实数据（参考）", rgb96[rng.choice(len(rgb96), NS, replace=False)]),
        ("z ~ N(0, Σc) 联合高斯", gen(Z_joint)),
        ("z ~ N(0, I) 独立", gen(Z_indep)),
        ("z 取自真实样本", gen(Z_real))]

print()
print(f"  {'sampler':24s} {'平均饱和':>9s} {'平均明度':>9s} {'通道 sd':>8s}")
print("  " + "-" * 56)
for name, a in rows:
    s_, l_, d_ = stats(a)
    print(f"  {name:24s} {s_:9.3f} {l_:9.3f} {d_:8.3f}")

TILE = 44
COLS = NS
W = 280 + COLS * (TILE + 4) + 4
H = 92 + len(rows) * (TILE + 26) + 26
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font = (font(20), font(12), font(10)) if False else (font(20), font(12), font(10))
dr.text((10, 10), "彩色生成：颜色分量从哪来？（E6 网络，8 个输入 → 96 个 RGB 值）",
        font=f_t, fill=(234, 238, 244))
dr.text((10, 36), "真实数据的饱和度和明度作为参考。联合高斯保留了分量之间的相关性。",
        font=f_s, fill=(150, 162, 178))
dr.text((10, 52), "如果「联合高斯」和「取自真实样本」看起来一样，部署时只需要 8 个随机数 + 一个 8×8 矩阵。",
        font=f_s, fill=(130, 142, 158))
y = 76
for name, a in rows:
    dr.text((10, y + 4), name, font=f_l, fill=(110, 168, 254))
    s_, l_, d_ = stats(a)
    dr.text((10, y + 20), f"饱和 {s_:.2f}  明度 {l_:.1f}", font=f_s, fill=(150, 162, 178))
    imgs = rgb_image(a)
    for i, im in enumerate(imgs):
        tile = np.kron(im, np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        cv.paste(Image.fromarray(tile), (280 + i * (TILE + 4), y))
    y += TILE + 26
p = os.path.join(OUT, "COLOUR_SAMPLER.png")
cv.save(p)
print(f"\nwrote {p}")

json.dump({"cov_eigenvalues": [float(v) for v in ev],
           "stats": {n: stats(a) for n, a in rows}},
          open(os.path.join(OUT, "sampler_stats.json"), "w"), indent=2, ensure_ascii=False)
print("COLOUR_SAMPLER_DONE")

#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
A proper showcase for PCA-4 and PCA-8, since they looked unexpectedly good.

Three panels, because "looks good" has to be separated from "controls the face":

PANEL 1  many samples. 64 faces from each model, drawn with random parameters as the
         progress sheet did, at a size where the detail is visible.

PANEL 2  what each of PCA-4's four knobs does. Optionally this is the most useful panel:
         it shows that the model really is steered by those four numbers, by sweeping one
         component at a time with the others held at zero.

PANEL 3  the noise is nearly inert. With the 8 PCA values FIXED and only the 64 noise
         values changing, the output barely moves -- measured earlier as a mean Hamming
         distance of 5.1 pixels out of 64. That is why "random parameters" produce
         sensible faces: the randomness is happening in the PCA slots, not the noise.
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

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16
LEVELS = np.linspace(0, 255, 16)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "pca_sweep")
os.makedirs(OUT, exist_ok=True)


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def load_net(path):
    ck = torch.load(path, weights_only=False)
    st = ck.get("G") or ck.get("S")
    norm = {(k.replace("net.", "", 1) if k.startswith("net.") else k): v
            for k, v in st.items()}
    idx = sorted({int(k.split(".")[0]) for k in norm if k.endswith(".weight")})
    seq, dims = [], []
    for j in idx:
        W = norm[f"{j}.weight"]
        seq.append(nn.Linear(int(W.shape[1]), int(W.shape[0])))
        dims.append((int(W.shape[1]), int(W.shape[0])))
        if j != idx[-1]:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq)
    m.load_state_dict(norm)
    m.eval()
    return m, dims[0][0]


def run(net, cond):
    with torch.no_grad():
        raw = net(torch.tensor(cond, dtype=torch.float32))
    dec = raw.view(-1, 32, NLEV).argmax(-1).numpy()
    return layout.image_from_left_half(dec)


def paste_row(cv, imgs, x0, y0, tile, pad=3, labels=None, fnt=None, col=(150, 162, 178)):
    dr = ImageDraw.Draw(cv)
    for i, g in enumerate(imgs):
        gg = np.clip(g / 15.0 * 255.0, 0, 255)
        t = np.kron(gg, np.ones((tile // 8, tile // 8)))
        cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                 (x0 + i * (tile + pad), y0))
        if labels and fnt:
            dr.text((x0 + i * (tile + pad) + 1, y0 + tile + 1), labels[i], font=fnt,
                    fill=col)


M4 = load_net(os.path.join(RES, "sweep", "sweep_pca4", "final.pt"))
M8 = load_net(os.path.join(RES, "sweep", "sweep_pca8", "final.pt"))
M22 = load_net(os.path.join(RES, "sweep", "sweep_pca22", "final.pt"))

TILE = 44
PAD = 3
COLS = 32
W = 150 + COLS * (TILE + PAD) + PAD
H = 2 * (TILE + 26) + 4 * (TILE + 20) + 3 * (TILE + 22) + 90
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_m, f_s = font(19), font(12), font(11), font(9)

y = 12
dr.text((6, y), "PCA-4 与 PCA-8 的表现", font=f_t, fill=(234, 238, 244))
y += 26

# ---------------- panel 1: many samples
dr.text((6, y), "① 随机参数下的样本（每个模型各自的随机抽样）", font=f_m, fill=(110, 168, 254))
y += 20
for name, (net, indim), k in (("PCA-4", M4, 4), ("PCA-8", M8, 8)):
    rng = np.random.default_rng(2024)
    pca = np.clip(rng.standard_normal((COLS, k)), -3.5, 3.5)
    noise = np.clip(rng.standard_normal((COLS, indim - k)), -3.5, 3.5)
    imgs = run(net, np.concatenate([pca, noise], 1).astype(np.float32))
    dr.text((6, y + 6), name, font=f_l, fill=(206, 214, 226))
    dr.text((6, y + 20), f"{k} PCA + {indim-k} 噪声", font=f_s, fill=(126, 138, 154))
    paste_row(cv, imgs, 150, y, TILE)
    y += TILE + 26

# ---------------- panel 2: the four knobs of PCA-4
y += 8
dr.text((6, y), "② PCA-4 的四个旋钮各自控制什么（一次只动一个，其余归零，噪声固定）",
        font=f_m, fill=(110, 168, 254))
y += 20
net4, indim4 = M4
fixed_noise = np.clip(np.random.default_rng(5).standard_normal((1, indim4 - 4)), -3.5, 3.5)
vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
for comp in range(4):
    rows_pca = []
    for v in vals:
        p = np.zeros((1, 4)); p[0, comp] = v
        rows_pca.append(p)
    block = np.concatenate(rows_pca, 0)
    cond = np.concatenate([block, fixed_noise.repeat(len(vals), 0)], 1).astype(np.float32)
    imgs = run(net4, cond)
    dr.text((6, y + 12), f"PC{comp}", font=f_l, fill=(206, 214, 226))
    dr.text((6, y + 26), "值 −2…+2", font=f_s, fill=(126, 138, 154))
    paste_row(cv, imgs, 150, y, TILE, labels=[f"{v:+.0f}" for v in vals], fnt=f_s)
    y += TILE + 20

# ---------------- panel 3: the noise is nearly inert
y += 8
dr.text((6, y), "③ 把 8 个 PCA 值固定，只换 64 维噪声（输出几乎不动 → 变化来自 PCA，不是噪声）",
        font=f_m, fill=(110, 168, 254))
y += 20
net8, indim8 = M8
rng = np.random.default_rng(9)
pca_fixed = np.clip(rng.standard_normal((1, 8)), -3.5, 3.5)
noise_var = np.clip(rng.standard_normal((COLS, indim8 - 8)), -3.5, 3.5)
cond = np.concatenate([pca_fixed.repeat(COLS, 0), noise_var], 1).astype(np.float32)
imgs = run(net8, cond)
A = imgs.reshape(len(imgs), -1).astype(np.int16)
d = np.abs(A[:, None, :] - A[None, :, :]).__gt__(0).sum(2).astype(float)
np.fill_diagonal(d, np.nan)
dr.text((6, y + 6), "PCA 固定", font=f_l, fill=(206, 214, 226))
dr.text((6, y + 20), f"32 次换噪声", font=f_s, fill=(126, 138, 154))
paste_row(cv, imgs, 150, y, TILE)
dr.text((6, y + TILE + 4), f"平均差异仅 {np.nanmean(d):.1f} / 64 像素",
        font=f_s, fill=(224, 192, 90))
y += TILE + 22

dr.text((6, H - 16), "结论：PCA-4/8 看起来不错，主要因为它们更平滑（同色率 74%/69%，而 PCA-22 是 65%），"
        "不是更准确（acc 0.21/0.27 vs 0.38）。",
        font=f_s, fill=(146, 158, 174))
p = os.path.join(OUT, "PCA48_SHOWCASE.png")
cv.save(p)
print(f"wrote {p}  {cv.size}")
print("noise-only mean Hamming:", round(float(np.nanmean(d)), 2))
print("PCA48_SHOWCASE_DONE")

#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Name the four surviving PCA knobs.

Two independent kinds of evidence, because either one alone can mislead:

1. THE EIGENFACE. Each principal component is a 64-number direction in image space. Drawn
   as an 8x8 picture it shows WHERE that knob acts: red where raising the knob brightens,
   blue where it darkens. A knob that only touches the eye rows is doing something local; a
   knob that is uniformly red has to be doing overall brightness.

2. CORRELATION WITH MEASURABLE STATISTICS. For every real face compute a handful of
   interpretable numbers, then ask which knob predicts each one across the whole dataset:
        brightness        mean of all pixels
        contrast          max - min
        vertical skew     mean(top 2 rows) - mean(bottom 2 rows)   (hair vs chin)
        centre vs border  mean(inner 4x4) - mean(outer ring)
        eye band          mean of the two eye rows relative to the rest
        mouth band        mean of the lowest rows relative to the rest
        spatial std       how much the face varies internally
   A knob with |r| near 1 against one of these has an honest name; a knob with no strong
   correlation gets described by its eigenface shape instead.

3. SWEEPS, at several fixed contexts. A sweep with everything else at zero can be deceptive
   because zero is not a typical face. Each knob is therefore also swept while the other
   three sit at three different fixed values.
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
OUT = os.path.join(ROOT, "results", "pcstudy")
os.makedirs(OUT, exist_ok=True)
NLEV = 16
LEVELS = np.linspace(0, 255, 16)

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


# ---------------------------------------------------------------- data + PCA
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
X = full.reshape(len(full), -1).astype(np.float64)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
_, _, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
ZZ = (((X - mu) @ comps.T) / sd[0]).astype(np.float32)
img = full.astype(float)

# ---------------------------------------------------------------- interpretable statistics
def feats(a):
    """a: (N,8,8) float levels."""
    out = {}
    out["brightness"] = a.mean(axis=(1, 2))
    out["contrast"] = a.reshape(len(a), -1).max(1) - a.reshape(len(a), -1).min(1)
    out["vertical skew"] = a[:, :2, :].mean(axis=(1, 2)) - a[:, 6:, :].mean(axis=(1, 2))
    out["centre vs border"] = (a[:, 2:6, 2:6].mean(axis=(1, 2))
                               - (a.sum(axis=(1, 2)) - a[:, 2:6, 2:6].sum(axis=(1, 2))) / 48.0)
    out["eye band"] = a[:, 3:5, :].mean(axis=(1, 2)) - a.mean(axis=(1, 2))
    out["mouth band"] = a[:, 6:, :].mean(axis=(1, 2)) - a.mean(axis=(1, 2))
    out["hair band"] = a[:, :2, :].mean(axis=(1, 2)) - a.mean(axis=(1, 2))
    out["spatial std"] = a.std(axis=(1, 2))
    out["upper vs lower"] = a[:, :4, :].mean(axis=(1, 2)) - a[:, 4:, :].mean(axis=(1, 2))
    out["eye corners"] = (a[:, 3:5, 1:3].mean(axis=(1, 2)) + a[:, 3:5, 5:7].mean(axis=(1, 2))) / 2
    return out


F = feats(img)
names = list(F.keys())

print("=" * 104)
print("WHICH KNOB PREDICTS WHICH MEASURABLE PROPERTY (correlation over 321k real faces)")
print("=" * 104)
corr = np.zeros((4, len(names)))
for k in range(4):
    for j, nm in enumerate(names):
        corr[k, j] = np.corrcoef(ZZ[:, k], F[nm])[0, 1]
print(f"  {'statistic':20s}" + "".join(f"{f'PC{k}':>9s}" for k in range(4)))
print("  " + "-" * 58)
for j, nm in enumerate(names):
    print(f"  {nm:20s}" + "".join(f"{corr[k, j]:+9.3f}" for k in range(4)))
print()
for k in range(4):
    order = np.argsort(-np.abs(corr[k]))
    top = order[:2]
    print(f"  PC{k}: strongest -> {names[top[0]]} ({corr[k, top[0]]:+.3f}), "
          f"then {names[top[1]]} ({corr[k, top[1]]:+.3f})")


# ---------------------------------------------------------------- model for sweeps
def loadj(f):
    j = json.load(open(os.path.join(WEB, f), encoding="utf-8"))
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
    return m, int(len(layers[0]["W"][0]))


def render_model(net, k, Z):
    with torch.no_grad():
        v = np.clip(net(torch.tensor(Z.astype(np.float32))).numpy(), 0, 15)
    lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
    v = np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)
    return np.rint(v).astype(int)


# ---------------------------------------------------------------- figure
TILE, PAD = 46, 4
EIW = 46 + 132                      # eigenface + its two statistic lines
CW = 5 * (TILE + PAD)               # one context's five tiles
LEFT = 12 + EIW
W = LEFT + 3 * CW + 3 * 14 + PAD
ROWH = TILE + 34
H = 92 + 4 * ROWH + 52 + 170
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_m, f_s = font(20), font(13), font(12), font(9)
dr.text((12, 10), "四个剩余旋钮是什么", font=f_t, fill=(234, 238, 244))
dr.text((12, 36), "左：特征脸（红=调高变亮，蓝=调高变暗）   右：在 dim4x 模型上扫 −2 / −1 / +0 / +1 / +2，"
        "每种固定三种不同上下文", font=f_s, fill=(150, 162, 178))
dr.text((12, 54), "每个旋钮单独一行；三组上下文用来看效果是否只在原点附近才成立", font=f_s,
        fill=(130, 142, 158))

NAMES = ["PC0", "PC1", "PC2", "PC3"]
net, kk = loadj("pca22_dim4x_model.json")
info = {}
corrrows = {}
print()
print("=" * 104)
print("SWEEPS ON dim4x (post-stretch applied, as the page shows it)")
print("=" * 104)

for k in range(4):
    y = 92 + k * ROWH
    e = comps[k].reshape(8, 8)
    m = np.abs(e).max()
    e = e / m if m > 0 else e
    rgb = np.zeros((8, 8, 3))
    rgb[..., 0] = np.where(e > 0, 55 + 200 * e, 55 + 35 * -e)
    rgb[..., 1] = 55 + 35 * (1 - np.abs(e))
    rgb[..., 2] = np.where(e < 0, 55 + 200 * -e, 55 + 35 * e)
    t = np.kron(rgb, np.ones((TILE // 8, TILE // 8, 1)))
    cv.paste(Image.fromarray(np.clip(t, 0, 255).astype(np.uint8)), (12, y))
    dr.text((12, y + TILE + 3), NAMES[k], font=f_l, fill=(110, 168, 254))
    order = np.argsort(-np.abs(corr[k]))[:2]
    for q, j in enumerate(order):
        dr.text((12 + 40, y + 3 + q * 12), f"{names[j]}", font=f_s, fill=(150, 162, 178))
        dr.text((12 + 40, y + 13 + q * 12), f"{corr[k, j]:+.2f}", font=f_s,
                fill=(110, 200, 130) if abs(corr[k, j]) > 0.6 else (150, 162, 178))

    for ctx_i, ctx in enumerate(([0.0, 0.0, 0.0, 0.0],
                                 [0.9, -0.9, 0.7, -0.7],
                                 [-1.2, 1.1, -0.8, 1.0])):
        rows = []
        for v in (-2.0, -1.0, 0.0, 1.0, 2.0):
            z = np.array(ctx, dtype=np.float32).copy()
            z[k] = v
            rows.append(z)
        Z = np.array(rows, dtype=np.float32)
        imgs = layout.image_from_left_half(render_model(net, kk, Z))
        if ctx_i == 0:
            info[k] = feats(imgs.astype(float))
        xx = LEFT + ctx_i * (CW + 14)
        if k == 0:
            dr.text((xx, y - 16), f"上下文 {ctx_i+1}", font=f_s, fill=(130, 142, 158))
        for i, g in enumerate(imgs):
            gg = np.clip(g / 15.0 * 255.0, 0, 255)
            tt = np.kron(gg, np.ones((TILE // 8, TILE // 8)))
            cv.paste(Image.fromarray(tt.astype(np.uint8), mode="L").convert("RGB"),
                     (xx + i * (TILE + PAD), y))
        if ctx_i == 0 and k == 0:
            for i, v in enumerate((-2, -1, 0, 1, 2)):
                dr.text((xx + i * (TILE + PAD) + 2, y + TILE + 2), f"{v:+d}", font=f_s,
                        fill=(120, 132, 148))

# --- measure table + what each knob does
ytab = 92 + 4 * ROWH + 30
dr.text((12, ytab - 18), "旋钮对可解释统计量的控制幅度（上下文 1 的扫描，−2 → +2 的跨度）",
        font=f_l, fill=(110, 168, 254))
ytab += 4
for k in range(4):
    yy = ytab + k * 40
    dr.text((12, yy + 4), NAMES[k], font=f_l, fill=(206, 214, 226))
    tops = [(nm, float(np.max(info[k][nm]) - np.min(info[k][nm]))) for nm in names]
    tops.sort(key=lambda z: -z[1])
    for q, (nm, sp) in enumerate(tops[:3]):
        col = (110, 200, 130) if sp > 6 else ((224, 192, 90) if sp > 2 else (120, 132, 148))
        dr.text((12 + 44 + q * 132, yy + 4), f"{nm} {sp:5.2f}", font=f_s, fill=col)

dr.text((12, H - 20), "绿=跨度大（确实在控制它）  黄=中等  灰=很小（不是它的职责）",
        font=f_s, fill=(140, 152, 168))
p = os.path.join(OUT, "FOUR_KNOBS.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

# ---------------------------------------------------------------- text summary
print()
print("=" * 104)
print("WHAT EACH KNOB DOES, BY THE SWEEP IT PRODUCES")
print("=" * 104)
for k in range(4):
    print(f"  PC{k}")
    for nm in names:
        vals = info[k][nm]
        span = float(np.max(vals) - np.min(vals))
        if span > 0.6:
            print(f"      {nm:20s} range {span:6.2f}   "
                  f"({vals[0]:.2f} @ -2 -> {vals[-1]:.2f} @ +2)")

json.dump({"corr": {f"PC{k}": {names[j]: float(corr[k, j]) for j in range(len(names))}
                    for k in range(4)}},
          open(os.path.join(OUT, "knob_correlations.json"), "w"), indent=2, ensure_ascii=False)
print("KNOBS_DONE")

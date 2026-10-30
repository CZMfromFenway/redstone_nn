#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
用投掷器采样各风格，过整数模型，出图。

链路
    float 与 integer 共用同一个 4→4 变换
        x = mu + L · (S − 128)/36.466,   L = chol
        X = clip(floor((x + 3.5)/7 · 256), 0, 255)
    float 路：S 由拒绝采样的 z 直接给出（等价于原始风格采样）
    整数路：S 由投掷器求和给出
    两边都用同一个整数模型（编译器产物）出图，这样差别只来自采样
"""
import json
import os
import sys
import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from compiler import Layer, compile_network, integer_forward  # noqa: E402

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(P2, "src"))
from layout import image_from_left_half  # noqa: E402

B, NB = 8, 256
XLO, XHI, DELTA = -3.5, 3.5, 7.0
LIMIT = 882
NLEV = 15
SIG_DROP = 36.466

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
W1 = np.array(mj["gen"]["layers"][0]["W"])
TH1 = np.array(mj["gen"]["layers"][0]["b"])
V2W = np.array(mj["gen"]["layers"][1]["W"])
PHI = np.array(mj["gen"]["layers"][1]["b"])
MC = np.array(json.load(open(os.path.join(P2, "webapp", "merge_mc11.json"),
                             encoding="utf-8"))["_lut"])
styles = json.load(open(os.path.join(P2, "webapp", "styles_gauss.json"),
                        encoding="utf-8"))["styles"]

# ---------------- 编译整数模型
comp = compile_network([Layer(W1, TH1, "relu", comparable=False, out_bits=8),
                        Layer(V2W, PHI, "clip", comparable=True,
                              lam=np.full(32, 15.0))],
                        XLO, XHI, B, verbose=False)
L2 = comp[1]
SIG2 = float(L2["sigma"][0])
T2 = int(L2["T"][0])


def int_levels(X):
    Z = integer_forward(comp, np.asarray(X, dtype=np.int64))
    m, n = Z.max(1, keepdims=True), Z.min(1, keepdims=True)
    dZ = np.maximum(m - n, 1)
    return np.clip(np.floor_divide((Z - n) * NLEV + (dZ >> 1), dZ), 0, NLEV)


# ---------------- 投掷器：13 路，权重见文档
WDROP = [1, 6, 7, 7, 7, 9, 10, 11, 11, 12, 14, 15, 18]
ABC = (4, 1, 4)
PV = np.array(ABC) / 9.0
pos = np.array([0])
pr = np.array([1.0])
for wi in WDROP:
    v = np.arange(3) * wi
    npos = (pos[:, None] + v[None, :]).ravel()
    npr = (pr[:, None] * PV[None, :]).ravel()
    up, inv = np.unique(npos, return_inverse=True)
    pos = up
    pr = np.bincount(inv, weights=npr)


def dropper_S(n, rng):
    """抽 n 组，每组 4 个独立的投掷器和。"""
    return rng.choice(pos, size=(n, 4), p=pr)


rng = np.random.default_rng(7)
K = 4

print("=" * 96)
print("投掷器采样 vs 原始拒绝采样")
print("=" * 96)
print(f"  {'风格':10s} {'原始拒绝率':>10s} {'投掷器后超界率':>15s} {'X 的均值':>22s}")
print("  " + "-" * 74)

rows = []
for e in styles:
    mu = np.array(e["mu"], dtype=float)
    L = np.array(e["chol"], dtype=float)
    A = (256.0 / (DELTA * SIG_DROP)) * L
    b = (256.0 / DELTA) * (mu + 3.5) - A @ np.full(4, 128.0)

    # 投掷器采样
    S = dropper_S(20000, rng)
    Xd = np.clip(np.floor(S @ A.T + b[None, :]), 0, NB - 1).astype(np.int64)
    # 原始：z 正态，x = mu + L z，超出 ±3.5 拒绝
    z = rng.standard_normal((20000, 4))
    x = z @ L.T + mu
    keep = np.all(np.abs(x) <= 3.5, axis=1)
    Xf = np.clip(np.floor((x[keep] + 3.5) / DELTA * NB), 0, NB - 1).astype(np.int64)

    over = float(np.mean(np.abs(S - 128.0) / SIG_DROP > 3.5))
    print(f"  {e['name']:10s} {100*(1-keep.mean()):9.1f}% {100*over:14.1f}% "
          f"{np.round(Xf.mean(0), 1)} vs {np.round(Xd.mean(0), 1)}")
    rows.append((e, A, b, Xd, Xf))

# ---------------- 出图
print()
print("=" * 96)
print("出图")
print("=" * 96)

KSHOW = 4
SC, PAD, GAP = 30, 6, 14
Wc = KSHOW * 8 * SC + (KSHOW - 1) * PAD + 2 * GAP
Hc = len(styles) * (2 * 8 * SC + PAD + GAP) + GAP
canvas = np.full((Hc, Wc, 3), 18, dtype=np.uint8)

lat_d = []
lat_f = []
for si, (e, A, b, Xd, Xf) in enumerate(rows):
    ld = MC[int_levels(Xd[:KSHOW])]
    lf = MC[int_levels(Xf[:KSHOW])]
    lat_d.append(ld)
    lat_f.append(lf)
    y0 = GAP + si * (2 * 8 * SC + PAD + GAP)
    for row, lat in ((0, lf), (1, ld)):
        for k in range(KSHOW):
            img = image_from_left_half(lat[k])
            px = np.clip(img / 15.0 * 255, 0, 255).astype(np.uint8)
            big = np.repeat(np.repeat(px, SC, 0), SC, 1)
            yy = y0 + row * (8 * SC + PAD)
            xx = GAP + k * (8 * SC + PAD)
            canvas[yy:yy + 8 * SC, xx:xx + 8 * SC] = np.stack([big, big, big], -1)
    canvas[y0 + 8 * SC:y0 + 8 * SC + PAD - 1,
           GAP:GAP + KSHOW * 8 * SC + (KSHOW - 1) * PAD] = (60, 60, 60)

out = os.path.join(RESULTS, "style_dropper.png")
Image.fromarray(canvas).save(out)
print(f"  wrote {out}  ({os.path.getsize(out)/1e3:.0f} KB)")
print("  每个风格两行：上=原始拒绝采样，下=投掷器采样；两条路都过同一个整数模型")

# ---------------- 统计：两条路的 16 级差
print()
print("=" * 96)
print("同一风格下，投掷器采样与原始采样的输出差异（各 300 张）")
print("=" * 96)
print(f"  {'风格':10s} {'像素不同':>9s} {'平均|差|':>9s} {'最大差':>7s}")
print("  " + "-" * 42)
for e, A, b, _, _ in rows:
    mu = np.array(e["mu"], dtype=float)
    L = np.array(e["chol"], dtype=float)
    S = dropper_S(300, rng)
    Xd = np.clip(np.floor(S @ A.T + b[None, :]), 0, NB - 1).astype(np.int64)
    z = rng.standard_normal((300, 4))
    x = z @ L.T + mu
    keep = np.all(np.abs(x) <= 3.5, axis=1)
    Xf = np.clip(np.floor((x[keep] + 3.5) / DELTA * NB), 0, NB - 1).astype(np.int64)
    ld = int_levels(Xd)
    lf = int_levels(Xf)
    n = min(len(ld), len(lf))
    d = ld[:n] - lf[:n]
    print(f"  {e['name']:10s} {100*(d!=0).mean():8.1f}% {np.abs(d).mean():9.3f} "
          f"{np.abs(d).max():7d}")
print()
print("  注：两者是不同的分布（投掷器路是截断，原始路是拒绝），")
print("      所以这里的差异主要是「截断 vs 拒绝」，不是数值精度。")
print("STYLE_FIG_DONE")

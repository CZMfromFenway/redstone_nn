#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
最终对比：实数管线 与 整数管线，都映射到 MC 11 档配色。

两条路
    实数：x -> 浮点两层 -> clip -> 逐张 min-max -> 16 级 -> MC11 查表
    整数：X -> 位权重 -> H,F -> RELU -> Y -> 位权重 -> G -> Z -> ΔZ,L -> MC11 查表

输入完全一致：实数路吃的是 X 的格心重建值。
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
sys.path.insert(0, os.path.join(HERE, "intpipe_lib") if False else HERE)

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(P2, "src"))
from layout import image_from_left_half  # noqa: E402

B = 8
NB = 1 << B
XLO, XHI = -3.5, 3.5
DELTA = XHI - XLO
LIMIT = 882
NLEV = 15

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
U1W = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
TH1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
V2W = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
PHI = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)
MC = np.array(json.load(open(os.path.join(P2, "webapp", "merge_mc11.json"),
                             encoding="utf-8"))["_lut"], dtype=np.float64)
styles = json.load(open(os.path.join(P2, "webapp", "styles_gauss.json"),
                        encoding="utf-8"))["styles"]

# ---------- 编译期常数
alpha = LIMIT / (np.abs(U1W).max(1) * DELTA / 4)
U = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB)
            * (1 << np.arange(B))[None, None, :]).astype(np.int64)
Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
Hmax = np.where(U > 0, U, 0).sum(axis=(1, 2))
A = np.rint(-alpha * (TH1 + XLO * U1W.sum(1)
                      + DELTA / (2 ** (B + 1)) * U1W.sum(1)) - Hmin).astype(np.int64)
c = np.maximum(np.ceil((Hmax - Hmin - A) / (NB - 1)), 1).astype(np.int64)   # D = 2^b-1
delta = c / alpha
beta = LIMIT / ((np.abs(V2W) * delta[None, :]).max() * (1 << (B - 2)))
V = np.rint(beta * V2W[:, :, None] * delta[None, :, None]
            * (1 << np.arange(B))[None, None, :]).astype(np.int64)
Gmin = np.where(V < 0, V, 0).sum(axis=(1, 2))
Bp = np.rint(-beta * (PHI + 0.5 * (V2W * delta[None, :]).sum(1)) - Gmin).astype(np.int64)
WIDTH = int(round(15 * beta))


def integer_path(X):
    bits = ((X[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
    H = np.einsum("nik,jik->nj", bits, U)
    Y = np.clip(np.floor_divide(H - Hmin[None, :] - A[None, :], c[None, :]), 0, NB - 1)
    bitsY = ((Y[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
    G = np.einsum("njk,pjk->np", bitsY, V)
    Z = np.clip(G - Gmin[None, :] - Bp[None, :], 0, WIDTH)
    mx, mn = Z.max(1, keepdims=True), Z.min(1, keepdims=True)
    dZ = np.maximum(mx - mn, 1)
    L = np.clip(np.floor_divide((Z - mn) * NLEV + (dZ >> 1), dZ), 0, NLEV)
    return L


def float_path(xq):
    xr = XLO + (xq + 0.5) * DELTA / NB
    z = np.maximum(0.0, xr @ U1W.T + TH1) @ V2W.T + PHI
    v = np.clip(z, 0, 15)
    lo, hi = v.min(1, keepdims=True), v.max(1, keepdims=True)
    sp = np.maximum(hi - lo, 1e-9)
    return np.clip(np.rint((v - lo) / sp * NLEV), 0, NLEV).astype(int)


# ---------- 采样
K = 4
rng = np.random.default_rng(7)
xs = []
for e in styles:
    mu, cho = np.array(e["mu"]), np.array(e["chol"])
    got = []
    while len(got) < K:
        z = rng.standard_normal(4)
        v = mu + cho @ z
        if np.all(np.abs(v) <= 3.5):
            got.append(v)
    xs.append(np.array(got))
x_all = np.concatenate(xs, 0)
NS = len(x_all)
Xq = np.clip(np.floor((x_all - XLO) / DELTA * NB), 0, NB - 1).astype(np.int64)

Lf = float_path(Xq)
Li = integer_path(Xq)

# MC11 映射
Gf = MC[Lf]          # 实数管线，MC 亮度
Gi = MC[Li]          # 整数管线，MC 亮度

print("=" * 100)
print(f"实数 vs 整数，均映射到 MC 11 档       {NS} 张（6 风格 x {K}）")
print("=" * 100)
d16 = Li - Lf
dMC = Gi - Gf
print(f"  16 级下：不同像素 {100*(d16!=0).mean():.2f}%   平均|差| {np.abs(d16).mean():.3f}  "
      f"最大 {np.abs(d16).max()} 级")
print(f"  MC11 下：不同像素 {100*(dMC!=0).mean():.2f}%   平均|亮度差| {np.abs(dMC).mean():.3f}  "
      f"最大 {np.abs(dMC).max():.2f}")
print(f"  逐张：16 级最大差 " + " ".join(f"{int(v)}" for v in np.abs(d16).max(1)))
print(f"  MC11 的相异像素里，落在调色板宽缝上的（亮度差 > 1）："
      f"{int((np.abs(dMC) > 1).sum())} 个")

# ---------- 出图
SC, PAD, GAP = 30, 6, 14
Wc = K * 8 * SC + (K - 1) * PAD + 2 * GAP
Hc = len(styles) * (2 * 8 * SC + PAD + GAP) + GAP
canvas = np.full((Hc, Wc, 3), 18, dtype=np.uint8)
for s in range(len(styles)):
    for k in range(K):
        n = s * K + k
        y0 = GAP + s * (2 * 8 * SC + PAD + GAP)
        for row, gg in ((0, Gf[n]), (1, Gi[n])):
            img = image_from_left_half(gg)
            px = np.clip(img / 15.0 * 255, 0, 255).astype(np.uint8)
            big = np.repeat(np.repeat(px, SC, 0), SC, 1)
            yy = y0 + row * (8 * SC + PAD)
            xx = GAP + k * (8 * SC + PAD)
            canvas[yy:yy + 8 * SC, xx:xx + 8 * SC] = np.stack([big, big, big], -1)
        # 每对的上下用细线分开
        canvas[y0 + 8 * SC:y0 + 8 * SC + PAD - 1,
               GAP + k * (8 * SC + PAD):GAP + k * (8 * SC + PAD) + 8 * SC] = (60, 60, 60)
out = os.path.join(RESULTS, "final_mc11.png")
Image.fromarray(canvas).save(out)
print(f"\n  wrote {out}  ({os.path.getsize(out)/1e3:.0f} KB)")
print("  每张脸的上半是实数管线，下半是整数管线，都映射到 MC 11 档")

# ---------- 另一张：16 级对照 + 差值
SC2, PAD2 = 20, 5
Wc2 = NS * 8 * SC2 + (NS + 1) * PAD2
Hc2 = 3 * 8 * SC2 + 4 * PAD2
cv = np.full((Hc2, Wc2, 3), 18, dtype=np.uint8)
dmax = max(1.0, float(np.abs(dMC).max()))
for r in range(3):
    for n in range(NS):
        if r == 0:
            img = image_from_left_half(Lf[n] / NLEV * 15)
        elif r == 1:
            img = image_from_left_half(Li[n] / NLEV * 15)
        else:
            img = image_from_left_half(np.abs(dMC[n]))
        px = np.clip(img / 15.0 * 255, 0, 255).astype(np.uint8)
        b2 = np.repeat(np.repeat(px, SC2, 0), SC2, 1)
        y0 = PAD2 + r * (8 * SC2 + PAD2)
        x0 = PAD2 + n * (8 * SC2 + PAD2)
        if r < 2:
            cv[y0:y0 + 8 * SC2, x0:x0 + 8 * SC2] = np.stack([b2, b2, b2], -1)
        else:
            cv[y0:y0 + 8 * SC2, x0:x0 + 8 * SC2, 0] = b2
            cv[y0:y0 + 8 * SC2, x0:x0 + 8 * SC2, 1] = (b2 * 0.25).astype(np.uint8)
            cv[y0:y0 + 8 * SC2, x0:x0 + 8 * SC2, 2] = (b2 * 0.25).astype(np.uint8)
        cv[y0 - 2:y0, x0:x0 + 8 * SC2] = ((235, 235, 235), (90, 190, 255),
                                          (255, 110, 110))[r]
out2 = os.path.join(RESULTS, "final_16level.png")
Image.fromarray(cv).save(out2)
print(f"  wrote {out2}   三行：实数16级 / 整数16级 / MC11 亮度差（红）")
print("FINAL_FIG_DONE")

#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
3 倍体积能换多少像素一致性？

三条路线对比（4→64→32 头像网络）
    A 位权重 @ W_ref=882   1 环   体积 1.00×    ← 定稿
    B 位权重 @ W_ref=1764  2 环   体积 ?
    C 强模 LUT（等效于位权重 @ W_ref≈4×882，但不需要环）  体积 2.72×

LUT 那一路没有可直接编译的实现，用「位权重在等效 W_ref 下的精度」代理，
再标注它实际需要的环数（LUT 是 0 环，位权重是 N 环）——这个差异要显式说明。
"""
import json
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
U1W = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
TH1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
V2W = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
PHI = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)

DELTA, XLO, CAP, NLEV = 7.0, -3.5, 882, 15
N1, M1 = 4, 64
N2, M2 = 64, 32


def compile_run(bx, by, W_ref, rng, NF=6000):
    NX, NY = 1 << bx, 1 << by
    alpha = W_ref / (np.abs(U1W).max(1) * DELTA / 4)
    U = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NX)
                * (1 << np.arange(bx))[None, None, :]).astype(np.int64)
    Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
    Hmax = np.where(U > 0, U, 0).sum(axis=(1, 2))
    kappa = TH1 + XLO * U1W.sum(1) + DELTA / (2 ** (bx + 1)) * U1W.sum(1)
    P = np.rint(-alpha * kappa - Hmin).astype(np.int64)
    c = np.maximum(np.ceil((Hmax - Hmin - P) / (NY - 1)), 1).astype(np.int64)
    delta = c / alpha
    beta = W_ref / ((np.abs(V2W) * delta[None, :]).max() * (1 << (by - 2)))
    V = np.rint(beta * V2W[:, :, None] * delta[None, :, None]
                * (1 << np.arange(by))[None, None, :]).astype(np.int64)
    Gmin = np.where(V < 0, V, 0).sum(axis=(1, 2))
    Bp = np.rint(-beta * (PHI + 0.5 * (V2W * delta[None, :]).sum(1)) - Gmin).astype(np.int64)
    T = int(round(NLEV * beta))
    # 环数：逐位统计该位最大权重是否 > 897
    r1 = sum(N1 for k in range(bx) if int(np.abs(U[:, :, k]).max()) > 897)
    r2 = sum(M2 for k in range(by) if int(np.abs(V[:, :, k]).max()) > 897)
    # 前向
    xs = np.clip(rng.standard_normal((NF, 4)), -3.5, 3.5)
    X = np.clip(np.floor((xs - XLO) / DELTA * NX), 0, NX - 1).astype(np.int64)
    b1 = ((X[:, :, None] >> np.arange(bx)[None, None, :]) & 1).astype(np.int64)
    Nn = np.einsum("nik,jik->nj", b1, U) - Hmin[None, :]
    Y = np.clip(np.floor_divide(Nn - P[None, :], c[None, :]), 0, NY - 1)
    b2 = ((Y[:, :, None] >> np.arange(by)[None, None, :]) & 1).astype(np.int64)
    G = np.einsum("njk,pjk->np", b2, V)
    Z = np.clip(G - Gmin[None, :] - Bp[None, :], 0, T)
    mx, mn = Z.max(1, keepdims=True), Z.min(1, keepdims=True)
    dZ = np.maximum(mx - mn, 1)
    L = np.clip(np.floor_divide((Z - mn) * NLEV + (dZ >> 1), dZ), 0, NLEV)
    # 参照
    Xq = np.clip(np.floor((xs + 3.5) / 7.0 * 256), 0, 255).astype(np.int64)
    zr = (np.maximum(0.0, (XLO + (Xq + 0.5) * DELTA / 256) @ U1W.T + TH1) @ V2W.T + PHI)
    v = np.clip(zr, 0, 15)
    lo, hi = v.min(1, keepdims=True), v.max(1, keepdims=True)
    Lr = np.clip(np.rint((v - lo) / np.maximum(hi - lo, 1e-9) * NLEV), 0, NLEV).astype(int)
    diff = 100 * float((L != Lr).mean())
    return diff, r1 + r2, U.size + V.size


def v_bits(n, m, b, rings):
    return (3 * b) * (2 * n) * (2 * m) * (b + rings) / b


rng = np.random.default_rng(41)
print("=" * 92)
print("一、位权重：W_ref 换精度（b_x=8, b_y=10）")
print("=" * 92)
print(f"  {'W_ref':>7s} {'倍数×882':>8s} {'环数':>5s} {'体积(层1+层2)':>14s} {'体积×':>7s} "
      f"{'不同像素':>9s}")
print("  " + "-" * 62)
base_vol = None
rows = []
for wr, mult in ((882, 1), (1323, 1.5), (1764, 2), (2646, 3), (3528, 4), (5292, 6)):
    d, rings, ncnt = compile_run(8, 10, wr, rng)
    r1 = min(64, rings)
    r2 = rings - r1
    v = v_bits(N1, M1, 8, 8 if r1 else 0) + v_bits(N2, M2, 10, 10 if r2 else 0)
    if base_vol is None:
        base_vol = v
    rows.append((wr, mult, rings, v, v / base_vol, d))
    print(f"  {wr:7d} {mult:8.1f} {rings:5d} {v:14,.0f} {v/base_vol:7.2f} {d:8.2f}%")

print()
print("=" * 92)
print("二、三条路线并排")
print("=" * 92)
LUT_VOL = 2.72          # 前面算的：811,008 / 297,984
print(f"  {'路线':>34s} {'体积×':>7s} {'环':>4s} {'不同像素':>9s} {'每 1% 体积换':>12s}")
print("  " + "-" * 76)
d882 = rows[0][5]
d1764 = [r for r in rows if r[0] == 1764][0]
dd = [r for r in rows if r[0] == 3528][0]
print(f"  {'A 位权重 W_ref=882（定稿）':>34s} {1.00:7.2f} {96:4d} {d882:8.2f}% {'—':>12s}")
print(f"  {'B 位权重 W_ref=1764（双环）':>34s} {d1764[4]:7.2f} {d1764[2]:4d} "
      f"{d1764[5]:8.2f}% {((d882-d1764[5])/(d1764[4]-1)):11.3f} 点/倍")
print(f"  {'C 强模 LUT（等效 W_ref≈4×882）':>34s} {LUT_VOL:7.2f} {0:4d} "
      f"{dd[5]:8.2f}% {((d882-dd[5])/(LUT_VOL-1)):11.3f} 点/倍")

print()
print("=" * 92)
print("三、判决")
print("=" * 92)
print(f"  定稿基线 {d882:.2f}%")
print(f"  双环路线 {d1764[5]:.2f}%（体积 {d1764[4]:.2f}×，环 {d1764[2]}）")
print(f"  LUT 路线 {dd[5]:.2f}%（体积 {LUT_VOL:.2f}×，环 0）")
print()
gain_B = (d882 - d1764[5]) / (d1764[4] - 1)
gain_C = (d882 - dd[5]) / (LUT_VOL - 1)
print(f"  每多 1 倍体积换来的误差下降：")
print(f"    双环路线 {gain_B:.3f} 个百分点")
print(f"    LUT 路线 {gain_C:.3f} 个百分点")
print()
print(f"  ⇒ LUT 的『性价比』是双环路线的 {gain_C/gain_B:.2f} 倍"
      f"（{'更差' if gain_C < gain_B else '更好'}）")

json.dump({"rows": [{"W_ref": r[0], "rings": r[2], "vol_x": r[4], "diff": r[5]}
                    for r in rows],
           "lut_vol_x": LUT_VOL},
          open(os.path.join(RESULTS, "volume_vs_pixels.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  已写入 results/volume_vs_pixels.json")
print("VVP_DONE")

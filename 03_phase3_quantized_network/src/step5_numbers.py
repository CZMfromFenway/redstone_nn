#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
按 G_min（全局最小值）重算层 2 的 Z_p，并为第 5 步取出数字。

用户定义
    G_{p,min} = sum min(0, V)          逐神经元的最小值
    G_min     = min_p G_{p,min}        32 个里最小的那个
    Z_p       = G_p - G_min            >= 0，且保持可比（全局平移不改差值）
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

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
U1W = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
TH1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
V2W = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
PHI = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)

B = 8
NB = 1 << B
XLO, XHI = -3.5, 3.5
DELTA = XHI - XLO
LIMIT = 882

# 第 2 步
alpha = LIMIT / (np.abs(U1W).max(1) * DELTA / 4)
U1 = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB)
             * (1 << np.arange(B))[None, None, :]).astype(np.int64)
Hmin = np.where(U1 < 0, U1, 0).sum(axis=(1, 2))
Hmax = np.where(U1 > 0, U1, 0).sum(axis=(1, 2))

# 第 3 步
A1 = np.rint(-alpha * (TH1 + XLO * U1W.sum(1)
                       + DELTA / (2 ** (B + 1)) * U1W.sum(1)) - Hmin).astype(np.int64)
c1 = np.maximum(np.ceil((Hmax - Hmin - A1) / NB), 1).astype(np.int64)

# 第 4 步
delta = c1 / alpha
g = np.abs(V2W) * delta[None, :]
beta = LIMIT / (g.max() * (1 << (B - 2)))
V2 = np.rint(beta * V2W[:, :, None] * delta[None, :, None]
             * (1 << np.arange(B))[None, None, :]).astype(np.int64)
Gmin_p = np.where(V2 < 0, V2, 0).sum(axis=(1, 2))     # G_{p,min}
GMAX_p = np.where(V2 > 0, V2, 0).sum(axis=(1, 2))     # G_{p,max}
G_min = int(Gmin_p.min())

print("=" * 100)
print("第 4 步：G 与 Z")
print("=" * 100)
print(f"  G_p,min 逐神经元范围 [{Gmin_p.min()}, {Gmin_p.max()}]")
print(f"  G_min = min_p G_p,min = {G_min}")
print(f"  Z_p = G_p - G_min 的可行范围 [0, {int(GMAX_p.max()-G_min)}]（理论上界）")
print(f"  beta = {beta:.4f}，15*beta = {15*beta:.2f}")

# 第 5 步需要用到的：B_p（z_p = 0 处的 Z_p）
Bp = np.rint(-beta * (PHI + 0.5 * (V2W * delta[None, :]).sum(1)) - G_min).astype(np.int64)
UPPER = int(round(15 * beta))
print()
print("=" * 100)
print("第 5 步：层 2 的零点与上界")
print("=" * 100)
print(f"  B_p（z_p = 0 处的 Z_p）范围 [{Bp.min()}, {Bp.max()}]，极差 {int(Bp.max()-Bp.min())}")
print(f"  上界 = B_p + 15*beta，而 15*beta = {UPPER} 是全局常数")
print(f"  B_p 极差 / beta = {(Bp.max()-Bp.min())/beta:.2f} 个 z 单位（显示区间只有 15）")

# 实测
rng = np.random.default_rng(41)
NF = 20000
xt = np.clip(rng.standard_normal((NF, 4)), XLO, XHI)
X1 = np.clip(np.floor((xt - XLO) / DELTA * NB), 0, NB - 1).astype(np.int64)
bits = ((X1[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
H = np.einsum("nik,jik->nj", bits, U1)
F = H - Hmin[None, :]
Y = np.clip(np.floor_divide(F - A1[None, :], c1[None, :]), 0, NB - 1)
bitsY = ((Y[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
G = np.einsum("njk,pjk->np", bitsY, V2)
Z = G - G_min

print()
print("=" * 100)
print("实测 Z_p（随机 20000 张脸）")
print("=" * 100)
print(f"  Z_p 全体范围 [{Z.min()}, {Z.max()}]")
print(f"  Z_p 最小值为 0 的比例 {100*(Z==0).mean():.3f}%")
print()
print(f"  {'像素':>4s} {'Z 最小':>8s} {'p5':>7s} {'中位':>7s} {'p95':>7s} {'Z 最大':>8s}")
print("  " + "-" * 50)
for p in (0, 3, 7, 12, 14, 23, 29, 31):
    v = Z[:, p]
    print(f"  {p:4d} {v.min():8d} {int(np.percentile(v,5)):7d} {int(np.median(v)):7d} "
          f"{int(np.percentile(v,95)):7d} {v.max():8d}")

# 验证 Z - B_p 约等于 beta * z_p
Zt = np.clip(Z - Bp[None, :], 0, UPPER)
yh = (Y + 0.5) * delta[None, :]
z2 = yh @ V2W.T + PHI
pred = beta * np.clip(z2, 0, 15)
e = np.abs(Zt - pred)
print()
print("=" * 100)
print("验证：clip(Z_p - B_p, 0, 15β) ≈ β·clip(z_p, 0, 15)")
print("=" * 100)
print(f"  误差 平均 {e.mean():.2f}  最大 {e.max():.0f}  占上界 {100*e.max()/UPPER:.3f}%")
print(f"  clip 后 Z~ 的范围 [{Zt.min()}, {Zt.max()}]")
print("STEP5_NUMBERS_DONE")

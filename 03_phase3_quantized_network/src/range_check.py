#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
按用户的最终结构：
    c_j = ceil((Fmax_j - A_j) / (2^b - 1))          D 已并入，开关删除
    B_p = round(-β(φ_p + ½Σ v δ) - G_p,min)
    Z_p = CLIP(G_p - G_p,min - B_p, 0, 上界)

要回答
    1 B_p 是否恒正
    2 G_p - G_p,min - B_p 的理论范围
    3 上界取 15 还是 15β
    4 Z_p 与 ΔZ 各需要多少位
    5 哪些量可能有负号
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
D = NB - 1                       # 已定为 2^b - 1
XLO, XHI = -3.5, 3.5
DELTA = XHI - XLO
LIMIT = 882

# 第 2 ~ 3 步
alpha = LIMIT / (np.abs(U1W).max(1) * DELTA / 4)
U = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB)
            * (1 << np.arange(B))[None, None, :]).astype(np.int64)
Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
Hmax = np.where(U > 0, U, 0).sum(axis=(1, 2))
A = np.rint(-alpha * (TH1 + XLO * U1W.sum(1)
                      + DELTA / (2 ** (B + 1)) * U1W.sum(1)) - Hmin).astype(np.int64)
c = np.maximum(np.ceil((Hmax - Hmin - A) / D), 1).astype(np.int64)
Ymax = np.clip(np.floor_divide(Hmax - Hmin - A, c), 0, NB - 1)

# 第 4 步
delta = c / alpha
beta = LIMIT / ((np.abs(V2W) * delta[None, :]).max() * (1 << (B - 2)))
V = np.rint(beta * V2W[:, :, None] * delta[None, :, None]
            * (1 << np.arange(B))[None, None, :]).astype(np.int64)
Gmin = np.where(V < 0, V, 0).sum(axis=(1, 2))
Gmax = np.where(V > 0, V, 0).sum(axis=(1, 2))

# 第 5 步
Bp = np.rint(-beta * (PHI + 0.5 * (V2W * delta[None, :]).sum(1)) - Gmin).astype(np.int64)
WIDTH = int(round(15 * beta))

print("=" * 100)
print("1  B_p 是否恒正")
print("=" * 100)
print(f"  B_p 范围 [{Bp.min()}, {Bp.max()}]，全部为正：{bool((Bp > 0).all())}")
print(f"  它的两个成分：")
print(f"    -G_p,min      范围 [{(-Gmin).min()}, {(-Gmin).max()}]   （正）")
print(f"    -β(φ+½Σvδ)   范围 "
      f"[{(-beta*(PHI+0.5*(V2W*delta[None,:]).sum(1))).min():.0f}, "
      f"{(-beta*(PHI+0.5*(V2W*delta[None,:]).sum(1))).max():.0f}]   （负，量级小）")

print()
print("=" * 100)
print("2  G_p - G_p,min - B_p 的理论范围")
print("=" * 100)
# 可达的 y_j 上界：格心 (Ymax + 1/2) δ
ymax = (Ymax + 0.5) * delta
zmin = PHI + (V2W.clip(max=0) * ymax[None, :]).sum(1)
zmax = PHI + (V2W.clip(min=0) * ymax[None, :]).sum(1)
print(f"  可达的 z_p 上界 z_max：范围 [{zmax.min():.3f}, {zmax.max():.3f}]")
print(f"  可达的 z_p 下界 z_min：范围 [{zmin.min():.3f}, {zmin.max():.3f}]")
print()
print(f"  β z_p 的范围：下 [{ (beta*zmin).min():.0f}, { (beta*zmin).max():.0f}]  "
      f"上 [{ (beta*zmax).min():.0f}, { (beta*zmax).max():.0f}]")
lo = int(np.floor((beta * zmin).min()))
hi = int(np.ceil((beta * zmax).max()))
print(f"  合并：[{lo}, {hi}]  跨度 {hi-lo}")

print()
print("=" * 100)
print("3  上界取 15 还是 15β")
print("=" * 100)
rng = np.random.default_rng(41)
NF = 20000
xt = np.clip(rng.standard_normal((NF, 4)), XLO, XHI)
X = np.clip(np.floor((xt - XLO) / DELTA * NB), 0, NB - 1).astype(np.int64)
bits = ((X[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
H = np.einsum("nik,jik->nj", bits, U)
Y = np.clip(np.floor_divide(H - Hmin[None, :] - A[None, :], c[None, :]), 0, NB - 1)
bitsY = ((Y[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
G = np.einsum("njk,pjk->np", bitsY, V)
inner = G - Gmin[None, :] - Bp[None, :]

print(f"  β = {beta:.4f}，15β = {15*beta:.2f} → round 成 {WIDTH}")
print()
print(f"  {'上界':>8s} {'饱和比例':>10s} {'Z 的不同取值个数':>16s} {'能否分辨 16 级':>14s}")
print("  " + "-" * 56)
for ub, nm in ((15, "15"), (WIDTH, "15β")):
    Z = np.clip(inner, 0, ub)
    nz = len(np.unique(Z))
    print(f"  {nm:>8s} {100*(inner>ub).mean():9.2f}% {nz:16d} "
          f"{'否' if nz < 16 else '是':>14s}")
print()
print(f"  → 上界取 15 时，$\\beta z_p$ 的 1 个单位就占 15 的 {beta/15:.1f} 倍，")
print(f"    只要 z_p > 15/β = {15/beta:.4f} 就饱和。整个画面会糊成一片。")
print(f"  → 上界必须取 15β = {WIDTH}。")

print()
print("=" * 100)
print("4  位宽")
print("=" * 100)
Z = np.clip(inner, 0, WIDTH)
dz = Z.max(1) - Z.min(1)
for nm, v in (("Z_p 上界 15β", WIDTH), ("ΔZ 实测最大", int(dz.max())),
              ("ΔZ 理论上界", WIDTH)):
    bits_needed = int(np.ceil(np.log2(v + 1)))
    print(f"  {nm:18s} = {v:6d}   需要 {bits_needed:2d} 位"
          f"（b = {B}，多出 {bits_needed - B} 位）")
print(f"  Z_p 实测范围 [{Z.min()}, {Z.max()}]，ΔZ 实测中位 {int(np.median(dz))}")
print(f"  G_p - G_p,min 的范围 [0, {int((Gmax-Gmin).max())}]，"
      f"需要 {int(np.ceil(np.log2((Gmax-Gmin).max()+1)))} 位（中间量，不必输出）")

print()
print("=" * 100)
print("5  哪些量可能有负号")
print("=" * 100)
rows = [
    ("u_{j,i}, v_{p,j}", "实数权重", "可负", "编译器给定"),
    ("U^(k)_{j,i}, V^(k)_{p,j}", "整数位权重", "可负", "由实数权重符号决定"),
    ("η_j, z_p", "实数预激活", "可负", "数学量，不参与硬件存储"),
    ("H_j, G_p", "原始累加", "可负", "硬件里靠加常数抬正"),
    ("H_{j,min}, G_{p,min}", "最小值", "负", "与 H、G 同族"),
    ("x_i, X_i", "输入", "非负", ""),
    ("F_j, Y_j", "层 1 中间与输出", "非负", ""),
    ("A_j, B_p", "零点", "非负", "B_p 实测恒正"),
    ("c_j, α_j, β, δ_j", "步长与缩放", "非负", ""),
    ("Z_p, ΔZ, L_p", "层 2 输出与级数", "非负", ""),
]
print(f"  {'符号':26s} {'是什么':14s} {'符号':6s} {'备注'}")
print("  " + "-" * 78)
for a, bb, cc, dd in rows:
    print(f"  {a:26s} {bb:14s} {cc:6s} {dd}")
print()
print("  → 需要带符号的只有：权重（实数与整数）与 H、G 这一族。")
print("    其余全部非负，比较器减法模式自带的截断正好对应 CLIP 的下界。")
print("RANGE_DONE")

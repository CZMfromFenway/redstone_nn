#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
检验：Z_p = CLIP(G_p - G_p,min, B_p, B_p + 15β) 能否直接做 min-max。

问题的来源
    B_p = round(-β(φ_p + ½Σ v δ) - G_p,min)
    它含 -G_p,min，而 G_p,min 逐通道从 -12180 到 -5358，跨度 6822。
    15β 只有 3641。
    于是 Z_p = B_p + clip(β z_p, 0, 15β) 的主体是 B_p，不是 β z_p。
    min-max 被 B_p 的逐通道差异主导，Z 与 z 完全不同序。

修法
    从零点量起：Z_p = CLIP(G_p - G_p,min - B_p, 0, 15β)
    等价于 Z_p = CLIP(G_p - K_p, 0, 15β)，K_p = G_p,min + B_p，
    仍是一个逐通道常数，运算量不变。
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
NLEV = 15


def pipeline(D, x):
    alpha = LIMIT / (np.abs(U1W).max(1) * DELTA / 4)
    U = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB)
                * (1 << np.arange(B))[None, None, :]).astype(np.int64)
    Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
    Hmax = np.where(U > 0, U, 0).sum(axis=(1, 2))
    A = np.rint(-alpha * (TH1 + XLO * U1W.sum(1)
                          + DELTA / (2 ** (B + 1)) * U1W.sum(1)) - Hmin).astype(np.int64)
    c = np.maximum(np.ceil((Hmax - Hmin - A) / D), 1).astype(np.int64)

    X = np.clip(np.floor((x - XLO) / DELTA * NB), 0, NB - 1).astype(np.int64)
    bits = ((X[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
    H = np.einsum("nik,jik->nj", bits, U)
    Y = np.clip(np.floor_divide(H - Hmin[None, :] - A[None, :], c[None, :]), 0, NB - 1)

    delta = c / alpha
    beta = LIMIT / ((np.abs(V2W) * delta[None, :]).max() * (1 << (B - 2)))
    V = np.rint(beta * V2W[:, :, None] * delta[None, :, None]
                * (1 << np.arange(B))[None, None, :]).astype(np.int64)
    Gmin = np.where(V < 0, V, 0).sum(axis=(1, 2))

    bitsY = ((Y[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
    G = np.einsum("njk,pjk->np", bitsY, V)

    Bp = np.rint(-beta * (PHI + 0.5 * (V2W * delta[None, :]).sum(1)) - Gmin).astype(np.int64)
    WIDTH = int(round(NLEV * beta))
    Kp = Gmin + Bp                                   # 合并后的逐通道常数
    return dict(G=G, Gmin=Gmin, Bp=Bp, Kp=Kp, WIDTH=WIDTH, beta=beta,
                c=c, A=A, delta=delta, Y=Y, Hmax=Hmax)


def levels(Z):
    m = Z.max(1, keepdims=True)
    n = Z.min(1, keepdims=True)
    dz = np.maximum(m - n, 1)
    return np.clip(np.floor_divide((Z - n) * NLEV + (dz >> 1), dz), 0, NLEV), dz


NF = 20000
rng = np.random.default_rng(41)
xt = np.clip(rng.standard_normal((NF, 4)), XLO, XHI)
Xq = np.clip(np.floor((xt - XLO) / DELTA * NB), 0, NB - 1)
zref = (np.maximum(0.0, (XLO + (Xq + 0.5) * DELTA / NB) @ U1W.T + TH1)) @ V2W.T + PHI
v = np.clip(zref, 0, 15)
lo, hi = v.min(1, keepdims=True), v.max(1, keepdims=True)
ref = np.clip(np.rint((v - lo) / np.maximum(hi - lo, 1e-9) * NLEV), 0, NLEV).astype(int)

print("=" * 104)
print("B_p 的量级：为什么不能直接拿 Z_p 做 min-max")
print("=" * 104)
r = pipeline(NB, xt)
print(f"  G_p,min 逐通道范围 [{r['Gmin'].min()}, {r['Gmin'].max()}]，"
      f"跨度 {int(r['Gmin'].max()-r['Gmin'].min())}")
print(f"  B_p 逐通道范围 [{r['Bp'].min()}, {r['Bp'].max()}]，跨度 {int(r['Bp'].max()-r['Bp'].min())}")
print(f"  15β = {r['WIDTH']}")
print(f"  → B_p 的逐通道差异是显示宽度的 {int(r['Bp'].max()-r['Bp'].min())/r['WIDTH']:.1f} 倍")

print()
print("=" * 104)
print("两种读法的对比")
print("=" * 104)
print(f"  {'读法':46s} {'不同像素':>9s} {'平均|差|':>9s} {'>=2级的脸':>10s}")
print("  " + "-" * 78)
for D in (NB, NB - 1):
    r = pipeline(D, xt)
    W, Bp, Kp, G, Gmin = r["WIDTH"], r["Bp"], r["Kp"], r["G"], r["Gmin"]
    Zl = np.clip(G - Gmin[None, :], Bp[None, :], Bp[None, :] + W)
    Zf = np.clip(G - Kp[None, :], 0, W)
    for nm, Z in ((f"D={D} 照字面：CLIP(G-Gmin,B_p,B_p+15β)", Zl),
                  (f"D={D} 从零点量起：CLIP(G-K_p,0,15β)", Zf)):
        L, _ = levels(Z)
        d = L - ref
        print(f"  {nm:46s} {100*(d!=0).mean():8.2f}% {np.abs(d).mean():9.4f} "
              f"{100*(np.abs(d).max(1)>=2).mean():9.2f}%")

print()
print("=" * 104)
print("在修法下比较 D")
print("=" * 104)
print(f"  {'D':>8s} {'c_j 中位':>9s} {'Y顶到255':>9s} {'beta':>9s} "
      f"{'不同像素':>9s} {'平均|差|':>9s} {'最大差':>7s} {'>=2级脸':>9s} {'完全相同':>9s}")
print("  " + "-" * 90)
for D in (NB, NB - 1):
    r = pipeline(D, xt)
    Ytop = np.clip(np.floor_divide(r["Hmax"] - r["A"], r["c"]), 0, NB - 1)
    Z = np.clip(r["G"] - r["Kp"][None, :], 0, r["WIDTH"])
    L, _ = levels(Z)
    d = L - ref
    print(f"  {D:8d} {int(np.median(r['c'])):9d} {int((Ytop==NB-1).sum()):6d}/64 "
          f"{r['beta']:9.3f} {100*(d!=0).mean():8.2f}% {np.abs(d).mean():9.4f} "
          f"{np.abs(d).max():7d} {100*(np.abs(d).max(1)>=2).mean():8.2f}% "
          f"{100*(np.abs(d).max(1)==0).mean():8.2f}%")

print()
print("=" * 104)
print("D 的影响链")
print("=" * 104)
r0, r1 = pipeline(NB, xt), pipeline(NB - 1, xt)
print(f"  c_j 中位 {int(np.median(r0['c']))} -> {int(np.median(r1['c']))}，"
      f"逐通道变化 {int((r0['c']!=r1['c']).sum())}/64")
print(f"  δ_j 中位 {np.median(r0['delta']):.6f} -> {np.median(r1['delta']):.6f}")
print(f"  β {r0['beta']:.4f} -> {r1['beta']:.4f}   15β {r0['WIDTH']} -> {r1['WIDTH']}")
print(f"  Y 完全相同的比例 {100*np.mean(r0['Y']==r1['Y']):.2f}%")
print("D_TEST2_DONE")

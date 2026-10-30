#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
D = 2^b 与 D = 2^b-1 的端到端对比。D 只出现在 c_j 的分母里，但会一路传下去：
c_j -> delta_j -> beta -> V -> Z -> L。

完整链条
    X_i      = clip(floor((x_i - x_min)/Delta * 2^b), 0, 2^b-1)
    alpha_j  = 882 / (max_i|u_ji| * Delta/4)
    U^(k)_ji = round(alpha_j u_ji Delta/2^b 2^k)
    H_j      = sum U X,  H_min = sum min(0,U),  F_j = H_j - H_min
    A_j      = round(H_j|_{eta=0} - H_min)
    c_j      = ceil((Fmax_j - A_j) / D)
    Y_j      = clip(floor((F_j - A_j)/c_j), 0, 2^b-1)
    delta_j  = c_j / alpha_j
    beta     = 882 / (max_pj|v_pj delta_j| * 2^(b-2))
    V^(k)_pj = round(beta v_pj delta_j 2^k)
    G_p      = sum V Y,  G_p,min = sum min(0,V)
    B_p      = round(-beta(phi_p + 1/2 sum_j v_pj delta_j) - G_p,min)
    Z_p      = CLIP(G_p - G_p,min, B_p, B_p + 15 beta)
    DeltaZ   = max_p Z_p - min_p Z_p
    L_p      = clip(floor((15(Z_p - min) + floor(DeltaZ/2)) / DeltaZ), 0, 15)
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
    """返回 Z_p 与中间量。"""
    alpha = LIMIT / (np.abs(U1W).max(1) * DELTA / 4)
    U = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB)
                * (1 << np.arange(B))[None, None, :]).astype(np.int64)
    Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
    Hmax = np.where(U > 0, U, 0).sum(axis=(1, 2))
    Fmax = Hmax - Hmin

    A = np.rint(-alpha * (TH1 + XLO * U1W.sum(1)
                          + DELTA / (2 ** (B + 1)) * U1W.sum(1)) - Hmin).astype(np.int64)
    c = np.maximum(np.ceil((Fmax - A) / D), 1).astype(np.int64)

    X = np.clip(np.floor((x - XLO) / DELTA * NB), 0, NB - 1).astype(np.int64)
    bits = ((X[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
    H = np.einsum("nik,jik->nj", bits, U)
    F = H - Hmin[None, :]
    Y = np.clip(np.floor_divide(F - A[None, :], c[None, :]), 0, NB - 1)

    delta = c / alpha
    g = np.abs(V2W) * delta[None, :]
    beta = LIMIT / (g.max() * (1 << (B - 2)))
    V = np.rint(beta * V2W[:, :, None] * delta[None, :, None]
                * (1 << np.arange(B))[None, None, :]).astype(np.int64)
    Gmin = np.where(V < 0, V, 0).sum(axis=(1, 2))

    bitsY = ((Y[:, :, None] >> np.arange(B)[None, None, :]) & 1).astype(np.int64)
    G = np.einsum("njk,pjk->np", bitsY, V)

    Bp = np.rint(-beta * (PHI + 0.5 * (V2W * delta[None, :]).sum(1)) - Gmin).astype(np.int64)
    WIDTH = int(round(NLEV * beta))
    Z = np.clip(G - Gmin[None, :], Bp[None, :], Bp[None, :] + WIDTH)

    return dict(Z=Z, Zres=G - Gmin[None, :], Bp=Bp, WIDTH=WIDTH, beta=beta,
                c=c, A=A, delta=delta, Y=Y, alpha=alpha, Fmax=Fmax)


def levels(Z):
    m = Z.max(1, keepdims=True)
    n = Z.min(1, keepdims=True)
    dz = np.maximum(m - n, 1)
    return np.clip(np.floor_divide((Z - n) * NLEV + (dz >> 1), dz), 0, NLEV), dz


def float_levels(z):
    v = np.clip(z, 0, 15)
    lo, hi = v.min(1, keepdims=True), v.max(1, keepdims=True)
    sp = np.maximum(hi - lo, 1e-9)
    return np.clip(np.rint((v - lo) / sp * NLEV), 0, NLEV).astype(int)


NF = 20000
rng = np.random.default_rng(41)
xt = np.clip(rng.standard_normal((NF, 4)), XLO, XHI)
zref = (np.maximum(0.0, (XLO + (np.clip(np.floor((xt - XLO) / DELTA * NB), 0, NB - 1) + 0.5)
                         * DELTA / NB) @ U1W.T + TH1)) @ V2W.T + PHI
ref = float_levels(zref)

print("=" * 104)
print("D 的对比")
print("=" * 104)
print(f"  {'D':>10s} {'c_j 中位':>9s} {'Y 顶到 255 的通道':>16s} {'beta':>9s} "
      f"{'15beta':>8s} {'Z 跨度中位':>11s}")
print("  " + "-" * 74)
for D in (NB, NB - 1):
    r = pipeline(D, xt)
    Ytop = np.clip(np.floor_divide(r["Fmax"] - r["A"], r["c"]), 0, NB - 1)
    L, dz = levels(r["Z"])
    print(f"  {D:10d} {int(np.median(r['c'])):9d} "
          f"{int((Ytop == NB - 1).sum()):9d} / 64     {r['beta']:9.3f} "
          f"{r['WIDTH']:8d} {int(np.median(dz)):11d}")

print()
print("=" * 104)
print("最终 16 级与浮点参照的差")
print("=" * 104)
print(f"  {'D':>10s} {'不同像素':>10s} {'平均|差|':>10s} {'平均偏差':>10s} "
      f"{'最大差':>7s} {'>=2级的脸':>10s} {'完全相同的脸':>12s}")
print("  " + "-" * 78)
for D in (NB, NB - 1):
    r = pipeline(D, xt)
    L, _ = levels(r["Z"])
    d = L - ref
    print(f"  {D:10d} {100*(d!=0).mean():9.2f}% {np.abs(d).mean():10.4f} "
          f"{d.mean():+10.4f} {np.abs(d).max():7d} "
          f"{100*(np.abs(d).max(1)>=2).mean():9.2f}% {100*(np.abs(d).max(1)==0).mean():11.2f}%")

print()
print("  偏差分布：")
for D in (NB, NB - 1):
    r = pipeline(D, xt)
    L, _ = levels(r["Z"])
    d = L - ref
    print(f"    D={D:3d}  " + "  ".join(
        f"{k:+d}:{100*float((d==k).mean()):5.2f}%" for k in range(-2, 3)))

print()
print("=" * 104)
print("D 改动的实际影响链")
print("=" * 104)
r0 = pipeline(NB, xt)
r1 = pipeline(NB - 1, xt)
print(f"  c_j：{int(np.median(r0['c']))} -> {int(np.median(r1['c']))}，"
      f"逐通道不同的 {int((r0['c']!=r1['c']).sum())}/64")
print(f"  delta_j 中位：{np.median(r0['delta']):.6f} -> {np.median(r1['delta']):.6f}")
print(f"  beta：{r0['beta']:.4f} -> {r1['beta']:.4f}")
print(f"  15beta：{r0['WIDTH']} -> {r1['WIDTH']}")
print(f"  Y 的取值：完全相同的比例 {100*np.mean(r0['Y']==r1['Y']):.2f}%")
print("D_TEST_DONE")

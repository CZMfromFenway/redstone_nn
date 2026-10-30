#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
修正环数：每个「超过 897 的位权重」各需要一个环，不是每个神经元一个。

    层1 权重数 = 4×64 = 256      层2 = 64×32 = 2048
    位 k 的权重 W^(k) = 2·W_ref·|w|/max|w| / 2^(b-1-k)
    只有当 |W^(k)| > 897 的那部分权重才要环

所以环数 = Σ_k  #{i,j : |W^(k)_{j,i}| > 897}
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

DELTA, XLO, NLEV, CAPC = 7.0, -3.5, 15, 897
N1, M1 = 4, 64
N2, M2 = 64, 32


def build(bx, by, W_ref):
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
    return dict(bx=bx, by=by, U=U, V=V, Hmin=Hmin, P=P, c=c, delta=delta,
                beta=beta, Gmin=Gmin, Bp=Bp, T=T, NY=NY, NX=NX, alpha=alpha)


def rings_of(W):
    """每个超 897 的位权重一个环。"""
    return int((np.abs(W) > CAPC).sum())


def volume(d, W_ref):
    r1 = rings_of(d["U"])
    r2 = rings_of(d["V"])
    f1 = r1 / (N1 * M1)
    f2 = r2 / (N2 * M2)
    v1 = (3 * d["bx"]) * (2 * N1) * (2 * M1) * (d["bx"] + f1) / d["bx"]
    v2 = (3 * d["by"]) * (2 * N2) * (2 * M2) * (d["by"] + f2) / d["by"]
    return r1, r2, f1, f2, v1 + v2


def run(d, rng, NF=6000):
    xs = np.clip(rng.standard_normal((NF, 4)), -3.5, 3.5)
    X = np.clip(np.floor((xs - XLO) / DELTA * d["NX"]), 0, d["NX"] - 1).astype(np.int64)
    b1 = ((X[:, :, None] >> np.arange(d["bx"])[None, None, :]) & 1).astype(np.int64)
    Nn = np.einsum("nik,jik->nj", b1, d["U"]) - d["Hmin"][None, :]
    Y = np.clip(np.floor_divide(Nn - d["P"][None, :], d["c"][None, :]), 0, d["NY"] - 1)
    b2 = ((Y[:, :, None] >> np.arange(d["by"])[None, None, :]) & 1).astype(np.int64)
    G = np.einsum("njk,pjk->np", b2, d["V"])
    Z = np.clip(G - d["Gmin"][None, :] - d["Bp"][None, :], 0, d["T"])
    mx, mn = Z.max(1, keepdims=True), Z.min(1, keepdims=True)
    dZ = np.maximum(mx - mn, 1)
    L = np.clip(np.floor_divide((Z - mn) * NLEV + (dZ >> 1), dZ), 0, NLEV)
    Xq = np.clip(np.floor((xs + 3.5) / 7.0 * 256), 0, 255).astype(np.int64)
    zr = (np.maximum(0.0, (XLO + (Xq + 0.5) * DELTA / 256) @ U1W.T + TH1) @ V2W.T + PHI)
    v = np.clip(zr, 0, 15)
    lo, hi = v.min(1, keepdims=True), v.max(1, keepdims=True)
    Lr = np.clip(np.rint((v - lo) / np.maximum(hi - lo, 1e-9) * NLEV), 0, NLEV).astype(int)
    return 100 * float((L != Lr).mean())


rng = np.random.default_rng(41)
print("=" * 100)
print("一、环数按「每个超限位权重一个」重算")
print("=" * 100)
print(f"  {'W_ref':>7s} {'层1环':>7s} {'层2环':>8s} {'合计':>8s} "
      f"{'层1占比':>8s} {'层2占比':>8s} {'体积×':>7s} {'不同像素':>9s}")
print("  " + "-" * 78)
rows = []
v0 = None
for wr in (882, 1323, 1764, 2646, 3528):
    d = build(8, 10, wr)
    r1, r2, f1, f2, v = volume(d, wr)
    if v0 is None:
        v0 = v
    diff = run(d, rng)
    rows.append((wr, r1, r2, r1 + r2, f1, f2, v, diff))
    print(f"  {wr:7d} {r1:7d} {r2:8d} {r1+r2:8d} {f1:8.3f} {f2:8.3f} "
          f"{v/v0:7.2f} {diff:8.2f}%")

print()
print("=" * 100)
print("二、三条路线（LUT 体积 2.72× 不变，环 0）")
print("=" * 100)
LUT_VOL = 2.72
a = rows[0]
b = [r for r in rows if r[0] == 1764][0]
c = [r for r in rows if r[0] == 3528][0]
print(f"  {'路线':>32s} {'体积×':>7s} {'环':>7s} {'不同像素':>9s} {'点/倍体积':>10s}")
print("  " + "-" * 72)
print(f"  {'A 位权重 W_ref=882（定稿）':>32s} {1.00:7.2f} {a[3]:7d} {a[7]:8.2f}% {'—':>10s}")
print(f"  {'B 位权重 W_ref=1764（双环）':>32s} {b[6]/v0:7.2f} {b[3]:7d} {b[7]:8.2f}% "
      f"{(a[7]-b[7])/(b[6]/v0-1):10.3f}")
print(f"  {'C 强模 LUT（等效 4×882）':>32s} {LUT_VOL:7.2f} {0:7d} {c[7]:8.2f}% "
      f"{(a[7]-c[7])/(LUT_VOL-1):10.3f}")

print()
print("=" * 100)
print("三、判决")
print("=" * 100)
gB = (a[7] - b[7]) / (b[6] / v0 - 1)
gC = (a[7] - c[7]) / (LUT_VOL - 1)
print(f"  定稿基线        {a[7]:.2f}%   体积 1.00×  环 {a[3]}")
print(f"  双环路线        {b[7]:.2f}%   体积 {b[6]/v0:.2f}×  环 {b[3]}")
print(f"  LUT 路线        {c[7]:.2f}%   体积 {LUT_VOL:.2f}×  环 0")
print()
print(f"  每多 1 倍体积换来的误差下降：双环 {gB:.3f} 点，LUT {gC:.3f} 点")
print(f"  ⇒ LUT 的性价比是双环路线的 {gC/gB:.2f} 倍 "
      f"（{'更差' if gC < gB else '更好'}）")
print()
print(f"  关键数字：3 倍体积换 {a[7]-c[7]:.2f} 个百分点的像素一致性")
print(f"            1.83 倍体积换 {a[7]-b[7]:.2f} 个百分点（双环）")

json.dump({"rows": [{"W_ref": r[0], "rings": r[3], "vol_x": r[6] / v0,
                     "diff": r[7]} for r in rows], "lut_vol_x": LUT_VOL},
          open(os.path.join(RESULTS, "volume_vs_pixels.json"), "w",
               encoding="utf-8"), indent=1)
print("\n  已写入 results/volume_vs_pixels.json")
print("VVP2_DONE")

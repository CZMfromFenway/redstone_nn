#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
隔离权重量化误差：输入取整数（避开输入量化），不经输出量化（不出 Y），
只比较「整数累加逆回的 eta」与「实数权重算的 eta」。

测量对象
    eta_exact = sum_i w_ji * xhat_i + theta_j            实数权重
    eta_int   = (N_j - P_j) / sigma_j                    整数权重
    err = eta_int - eta_exact                            纯粹来自权重量化

两种候选公式
    甲（sqrt(nb) 型）  eps_w = sqrt(n b / 24) / (sigma * Delta_Y)
    乙（你的 2/(4W_ref) 型）相对误差，乘 |eta| 换成绝对
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
DELTA, XLO, CAP, N = 7.0, -3.5, 882, 4


def layer1(b, W_ref):
    NB = 1 << b
    alpha = CAP / (np.abs(U1W).max(1) * DELTA / 4) * (W_ref / CAP)
    U = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB)
                * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
    kappa = TH1 + XLO * U1W.sum(1) + DELTA / (2 ** (b + 1)) * U1W.sum(1)
    P = np.rint(-alpha * kappa - Hmin).astype(np.int64)
    return dict(NB=NB, alpha=alpha, U=U, Hmin=Hmin, P=P, kappa=kappa)


def measure(b, W_ref, rng, NF=20000):
    d = layer1(b, W_ref)
    NB = d["NB"]
    # 输入取精确的 0..255 整数（b 位），避开输入量化误差
    X = rng.integers(0, NB, size=(NF, N))
    xh = XLO + (X + 0.5) * DELTA / NB          # 逆回的实数输入
    eta_exact = xh @ U1W.T + TH1
    bits = ((X[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.int64)
    C = np.einsum("nik,jik->nj", bits, d["U"])
    eta_int = (C - d["Hmin"][None, :] - d["P"][None, :]) / d["alpha"][None, :]
    err = eta_int - eta_exact
    dY = float((np.maximum(0, eta_exact).max(0) - np.maximum(0, eta_exact).min(0)).mean())
    return dict(b=b, W_ref=W_ref, dY=dY,
                std=float(err.std(0).mean()), maxabs=float(np.abs(err).max()),
                eta_rms=float(np.sqrt((eta_exact ** 2).mean())),
                alpha_med=float(np.median(d["alpha"])),
                wmax=float(np.abs(U1W).max()))


print("=" * 104)
print("权重量化误差的实测 vs 两种公式")
print("=" * 104)
rng = np.random.default_rng(3)
rows = []
for W_ref in (441, 882, 1764):
    print(f"\n  W_ref = {W_ref}")
    print(f"  {'b':>3s} {'实测 std(eta)':>13s} {'实测 max|err|':>13s} "
          f"{'甲 sqrt(nb)/sigma':>17s} {'乙 2/(4W_ref)*eta_rms':>21s}")
    print("  " + "-" * 76)
    for b in (6, 7, 8, 9, 10, 11, 12, 13):
        r = measure(b, W_ref, rng)
        rows.append(r)
        A = np.sqrt(N * b / 24) / r["alpha_med"]          # 甲
        Bf = 2.0 / (4 * W_ref) * r["eta_rms"]             # 乙
        print(f"  {b:3d} {r['std']:13.6f} {r['maxabs']:13.6f} "
              f"{A:17.6f} {Bf:21.6f}")

print()
print("=" * 104)
print("谁的标度对：看 std 随 b 怎么变")
print("=" * 104)
print(f"  {'W_ref':>6s} " + " ".join(f"{'b='+str(b):>9s}" for b in range(6, 14)))
print("  " + "-" * 86)
for W_ref in (441, 882, 1764):
    vals = [r["std"] for r in rows if r["W_ref"] == W_ref]
    print(f"  {W_ref:6d} " + " ".join(f"{v:9.6f}" for v in vals))
    print(f"  {'比值':>6s} " + " ".join(f"{v/vals[0]:9.3f}" for v in vals))
    sq = [np.sqrt(b / 6) for b in range(6, 14)]
    print(f"  {'若 sqrt(b)':>6s} " + " ".join(f"{v/sq[0]:9.3f}" for v in sq))
    print()

print("=" * 104)
print("诊断：低位权重的贡献是不是真的可以忽略")
print("=" * 104)
print("  取 W_ref=882, b=10，逐位关掉该位的取整误差，看总误差降多少：")
b, W_ref = 10, 882
d = layer1(b, W_ref)
NB = d["NB"]
X = rng.integers(0, NB, size=(20000, N))
xh = XLO + (X + 0.5) * DELTA / NB
eta_exact = xh @ U1W.T + TH1
bits = ((X[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.int64)
base_std = None
print(f"  {'保留的位（从高到低）':>22s} {'std(eta)':>10s} {'占比':>7s} {'该位实测权重':>12s}")
print("  " + "-" * 58)
for keep in range(1, b + 1):
    U2 = d["U"].copy()
    U2[:, :, :b - keep] = 0
    C = np.einsum("nik,jik->nj", bits, U2)
    Hmin2 = np.where(U2 < 0, U2, 0).sum(axis=(1, 2))
    P2 = np.rint(-d["alpha"] * d["kappa"] - Hmin2).astype(np.int64)
    eta_int = (C - Hmin2[None, :] - P2[None, :]) / d["alpha"][None, :]
    s = float((eta_int - eta_exact).std(0).mean())
    if keep == b:
        base_std = s
    wtop = int(np.abs(d["U"][:, :, b - keep]).max())
    print(f"  {f'最高 {keep} 位':>22s} {s:10.6f} {s/base_std*100:6.1f}% {wtop:12d}")

print()
print("  若低位可忽略，std 应随保留位数迅速饱和；若每位的贡献相等，std 应大致 ∝ sqrt(保留位数)。")
print("ERRW_DONE")

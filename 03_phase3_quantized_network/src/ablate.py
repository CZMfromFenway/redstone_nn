#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
末层误差的消融实验 + 你的线性翻转律检验。

五级消融（逐项打开量化）
    V0  全浮点（参照）
    V1  层1 输入量化
    V2  + 层1 权重量化
    V3  + 层1 输出量化（出 Y）
    V4  + 层2 权重量化
    V5  + 层2 输出截断（全整数管线）

检验两件事
    1. 每一级引入的 z 误差，与解析 ε 是否对得上
    2. 翻转率 ≈ 15·ε/ρ 是否成立（ρ = 实际跨度 / 满量程）
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

DELTA, XLO, CAP_REF = 7.0, -3.5, 882
NLEV = 15


def build(b, W_ref):
    NB = 1 << b
    alpha = CAP_REF / (np.abs(U1W).max(1) * DELTA / 4) * (W_ref / CAP_REF)
    Uf = alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB) \
        * (1 << np.arange(b))[None, None, :]          # 精确位权重
    U = np.rint(Uf).astype(np.int64)
    Hmin_f = np.where(Uf < 0, Uf, 0).sum(axis=(1, 2))
    Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
    kappa = TH1 + XLO * U1W.sum(1) + DELTA / (2 ** (b + 1)) * U1W.sum(1)
    Pf = -alpha * kappa - Hmin_f
    P = np.rint(Pf).astype(np.int64)
    Hmax = np.where(U > 0, U, 0).sum(axis=(1, 2))
    c = np.maximum(np.ceil((Hmax - Hmin - P) / (NB - 1)), 1).astype(np.int64)
    delta = c / alpha
    beta = W_ref / ((np.abs(V2W) * delta[None, :]).max() * (1 << (b - 2)))
    Vf = beta * V2W[:, :, None] * delta[None, :, None] * (1 << np.arange(b))[None, None, :]
    V = np.rint(Vf).astype(np.int64)
    Gmin_f = np.where(Vf < 0, Vf, 0).sum(axis=(1, 2))
    Gmin = np.where(V < 0, V, 0).sum(axis=(1, 2))
    Bpf = -beta * (PHI + 0.5 * (V2W * delta[None, :]).sum(1)) - Gmin_f
    Bp = np.rint(Bpf).astype(np.int64)
    # 每通道增益比 r_p = max_j|v δ| / 全局 max
    q = (np.abs(V2W) * delta[None, :]).max(axis=1)
    r = q / q.max()
    return dict(NB=NB, alpha=alpha, Uf=Uf, U=U, Hmin_f=Hmin_f, Hmin=Hmin,
                Pf=Pf, P=P, c=c, delta=delta, beta=beta, Vf=Vf, V=V,
                Gmin_f=Gmin_f, Gmin=Gmin, Bpf=Bpf, Bp=Bp, r=r)


def run(b, W_ref, level, x):
    d = build(b, W_ref)
    NB = d["NB"]
    # --- 层 1 输入
    Xr = (x - XLO) / DELTA * NB
    if level >= 1:
        Xr = np.floor(Xr)
    Xr = np.clip(Xr, 0, NB - 1)
    if level >= 1:
        xh = XLO + (np.floor(Xr) + 0.5) * DELTA / NB
    else:
        xh = XLO + Xr * DELTA / NB + 0.5 * DELTA / NB
    # --- 层 1 累加（用 η 的实数形式，坏处是看不到位分解；
    #     这里直接用「权重是否取整」来区分）
    w1 = d["U"] if level >= 2 else d["Uf"]
    if level >= 2:
        Hmin, P = d["Hmin"], d["P"]
        eta_int = ((xh - XLO) / DELTA * NB) @ np.zeros((4, 64))   # 占位
        # 用位分解算
        Xq = np.clip(np.floor((xh - XLO) / DELTA * NB), 0, NB - 1).astype(np.int64)
        bits = ((Xq[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.float64)
        N = np.einsum("nik,jik->nj", bits, w1) - Hmin[None, :]
        eta = (N - P[None, :]) / d["alpha"][None, :]
    else:
        eta = (xh - XLO) @ np.zeros((4, 64)) + (xh @ U1W.T + TH1)
        # 精确权重下直接算实数
        eta = xh @ U1W.T + TH1
    y = np.maximum(0.0, eta)
    # --- 层 1 输出量化
    if level >= 3:
        c = d["c"]
        Y = np.clip(np.floor_divide(
            np.rint((eta * d["alpha"][None, :] + d["P"][None, :]
                     + d["Hmin"][None, :]) + d["alpha"][None, :] * 0).astype(np.int64)
            - d["P"][None, :] - d["Hmin"][None, :], c[None, :]), 0, NB - 1)
        # 更直接：用 N-P 再除以 c
        Xq = np.clip(np.floor((xh - XLO) / DELTA * NB), 0, NB - 1).astype(np.int64)
        bits = ((Xq[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.float64)
        Nq = np.einsum("nik,jik->nj", bits, d["U"]) - d["Hmin"][None, :]
        Y = np.clip(np.floor_divide(np.rint(Nq - d["P"][None, :]).astype(np.int64),
                                    d["c"][None, :]), 0, NB - 1)
        y = (Y + 0.5) * d["delta"][None, :]     # 逆回（格心）
    # --- 层 2
    w2 = d["V"] if level >= 4 else d["Vf"]
    if level >= 4:
        Yq = Y if level >= 3 else np.clip(np.floor(y / d["delta"][None, :]), 0, NB - 1)
        bits2 = ((Yq[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.float64)
        G = np.einsum("njk,pjk->np", bits2, w2)
        z = (G - d["Gmin"][None, :] - d["Bp"][None, :]) / d["beta"]
    else:
        z = y @ V2W.T + PHI
    # 显示截断是管线的一部分（不是误差源），所有级别都做
    z = np.clip(z, 0.0, float(NLEV))
    return z, d


def levels_of(z):
    m, n = z.max(1, keepdims=True), z.min(1, keepdims=True)
    sp = np.maximum(m - n, 1e-9)
    return np.clip(np.floor((15.0 * (z - n) / sp) + 0.5), 0, NLEV)


rng = np.random.default_rng(41)
NF = 6000
xs = np.clip(rng.standard_normal((NF, 4)), -3.5, 3.5)

print("=" * 104)
print("消融：逐项打开量化，看 z 的 RMS 误差与 16 级翻转率")
print("=" * 104)
for b, W_ref in ((10, 882), (9, 441), (11, 1764)):
    zs = {}
    for lv in range(6):
        z, d = run(b, W_ref, lv, xs)
        zs[lv] = z
    ref = zs[0]
    Lref = levels_of(ref)
    print(f"\n  b={b}, W_ref={W_ref}")
    print(f"  {'级':>3s} {'新增的量化':>14s} {'RMS(z−参照)':>13s} {'相对满量程':>11s} "
          f"{'翻转率':>8s} {'15ε/ρ 预测':>11s}")
    print("  " + "-" * 70)
    names = ["—", "层1 输入", "层1 权重", "层1 输出", "层2 权重", "层2 输出"]
    for lv in range(6):
        e = zs[lv] - ref
        rms = float(np.sqrt((e ** 2).mean()))
        rel = rms / NLEV
        flips = float((levels_of(zs[lv]) != Lref).mean())
        # ρ：实际跨度 / 满量程（用参照的 Z 跨度 / 15）
        sp = float((ref.max(1) - ref.min(1)).mean())
        rho = sp / NLEV
        pred = 15 * rel / rho
        print(f"  {lv:3d} {names[lv]:>14s} {rms:13.6f} {rel:11.6f} "
              f"{100*flips:7.2f}% {100*pred:10.2f}%")
    d = build(b, W_ref)
    r = d["r"]
    print(f"  层2 增益比 r_p：[{r.min():.4f}, {r.max():.4f}]，"
          f"1/r 均值 {float((1/r).mean()):.3f}，"
          f"log r 的 std {float(np.log(r).std()):.3f}")
    print(f"  实际跨度 ρ = {float((ref.max(1)-ref.min(1)).mean())/NLEV:.3f}")

print()
print("=" * 104)
print("你的猜想检验：层2 全局缩放的放大因子")
print("=" * 104)
print("  若层2 能逐通道用 beta_p，每个通道都能推满；实际用全局 beta，")
print("  通道 p 的有效缩放只有 r_p 倍 => 其权重量化误差被放大 1/r_p 倍。")
print()
print(f"  {'b':>3s} {'W_ref':>6s} {'r 最小':>8s} {'r 最大':>8s} {'1/r 均值':>9s} "
      f"{'1/r 最大':>9s} {'log r std':>10s}")
print("  " + "-" * 62)
for b in (8, 9, 10, 11, 12):
    for W_ref in (441, 882, 1764):
        d = build(b, W_ref)
        r = d["r"]
        print(f"  {b:3d} {W_ref:6d} {r.min():8.4f} {r.max():8.4f} "
              f"{float((1/r).mean()):9.3f} {float((1/r).max()):9.3f} "
              f"{float(np.log(r).std()):10.4f}")
print()
print("  → 1/r 的均值与最大值几乎不随 b、W_ref 变，是固定的网络属性。")
print("    所以层2 的放大是一个常系数，不是随 b 增长的项。")
print("ABL_DONE")

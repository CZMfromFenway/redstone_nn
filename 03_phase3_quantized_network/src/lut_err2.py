#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
强模 LUT 与位权重方案的误差对照（同一张鸢尾花网络，同一口径）。

统一约定
    真实特征值 x 用 0..15 的实数表示（LUT 的整数刻度）
    第 b 位方案：X_b = round(x·2^b/16) ∈ [0, 2^b-1]，重建值 x_b = X_b·16/2^b
    两边都对比「用实数权重算的 u = W·x_q + b1」——输入量化误差被隔离掉
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
P1 = os.path.join(os.path.dirname(ROOT), "01_phase1_digit_recognition")
RESULTS = os.path.join(ROOT, "results")

d = json.load(open(os.path.join(P1, "results", "iris_teacher.json"), encoding="utf-8"))
W1 = np.array(d["W1"], dtype=float)
b1 = np.array(d["b1"], dtype=float)
Xq = np.array(d["Xq"], dtype=int)          # (150, 4) 0..15
M, N = W1.shape
CAP = 868.0


def lut_first(X15):
    """LUT：每（神经元,输入）一列 16 格阶梯。X15 ∈ [0,15] 整数。"""
    rho = CAP * np.abs(W1) / np.abs(W1).max(axis=1, keepdims=True)
    Q = np.zeros((X15.shape[0], M))
    for j in range(M):
        for i in range(N):
            if W1[j, i] >= 0:
                lam = np.rint(X15[:, i] * rho[j, i])
                l0 = 0.0
            else:
                lam = np.rint(15 * rho[j, i] - X15[:, i] * rho[j, i])
                l0 = float(np.rint(15 * rho[j, i]))
            Q[:, j] += lam - l0
    scale = np.abs(W1).max(axis=1) / CAP
    return Q * scale[None, :] + b1[None, :]


def bits_first(Xb, b, W_ref):
    """位权重：Xb 是 b 位整数，真实值 = Xb·16/2^b。"""
    dx = 16.0 / (1 << b)                       # 每个整数步对应多少真实单位
    alpha = W_ref / (np.abs(W1).max(axis=1) * dx * (1 << (b - 2)))
    U = np.rint(alpha[:, None, None] * W1[:, :, None] * dx
                * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
    bits = ((Xb[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.int64)
    C = np.einsum("nik,jik->nj", bits, U)
    # 预激活零点：x=0 处
    kappa = b1 + (W1 * (0.0 + 0.5 * dx)).sum(axis=1)
    P = np.rint(-alpha * kappa - Hmin).astype(np.int64)
    return (C - Hmin[None, :] - P[None, :]) / alpha[None, :]


def u_true_from(x_real):
    return x_real @ W1.T + b1


print("=" * 100)
print("一、第一层预激活误差（隔离输入量化：两边都用各自的重建值做参照）")
print("=" * 100)
print(f"  W1 的 max|W| 逐神经元 = {np.round(np.abs(W1).max(axis=1),4)}")
ut15 = u_true_from(Xq.astype(float))
print(f"  u 的极差 = {np.round(ut15.max(0)-ut15.min(0),3)}")
print()

rows = []
# LUT：4 bit
u_l = lut_first(Xq)
e = np.abs(u_l - ut15)
rows.append(("LUT (16 格, 4bit)", 16, 4, e.max(), e.mean(),
             e.max() / (ut15.max(0) - ut15.min(0)).mean(), 16 * M * N))
# 位权重：各 b
for b in (4, 6, 8, 10, 12, 16):
    Xb = np.rint(Xq.astype(float) * (1 << b) / 16.0).astype(np.int64)
    dx = 16.0 / (1 << b)
    # 编译器约定：逆回是格心 (X+0.5)δ，参照必须用同一约定，否则混入半步偏移
    ut = u_true_from((Xb + 0.5) * dx)
    ub = bits_first(Xb, b, CAP)
    e2 = np.abs(ub - ut)
    rows.append((f"位权重 b={b}", b, b, e2.max(), e2.mean(),
                 e2.max() / (ut.max(0) - ut.min(0)).mean(), b * M * N))

print(f"  {'方案':>18s} {'容器/权重':>9s} {'max|Δu|':>10s} {'mean|Δu|':>10s} "
      f"{'相对 u 极差':>12s} {'一层容器':>9s}")
print("  " + "-" * 76)
for nm, cw, bb, mx, mn, rel, tot in rows:
    print(f"  {nm:>18s} {cw:9d} {mx:10.5f} {mn:10.5f} {rel:12.5f} {tot:9d}")

print()
print("=" * 100)
print("二、误差的标度：LUT 恒定，位权重随 b 增长")
print("=" * 100)
print("  LUT：λ(x) = round(xρ)，整列阶梯只取整一次")
print("        |λ(x) − xρ| ≤ 0.5，对一切 x 成立 ⇒ 与档数无关")
print()
print("  位权重：每个位的权重独立取整，误差按 √(位数) 累加")
print()
rel = {nm: r[5] for nm, *r in [(r[0], *r[1:]) for r in rows]}
for nm, cw, bb, mx, mn, rl, tot in rows:
    print(f"    {nm:>18s}  相对误差 {rl:.2e}   容器 {tot:4d}")

print()
print("=" * 100)
print("三、中间层与输出层：两边完全一样")
print("=" * 100)
print(f"  {'lv':>5s} {'位数':>4s} {'ε_y = 1/(2lv)':>14s} {'容器/输出权重':>13s}")
print("  " + "-" * 40)
for lv, bb in ((15, 4), (255, 8), (4095, 12)):
    print(f"  {lv:5d} {bb:4d} {1/(2*lv):14.6f} {bb:13d}")
print()
print("  r_j = max(0,u_j) → v_j = round(lv·r_j/R_j) → logit_k = Σ A_kj v_j + θ_k")
print("  这两层的实现、误差公式、容器数，两边一致。")

json.dump({"lut": {"containers_per_weight": 16, "max_du": float(rows[0][3])},
           "bits": {str(bb): {"containers_per_weight": bb,
                              "max_du": float(mx)}
                    for nm, cw, bb, mx, mn, rl, tot in rows if bb >= 4}},
          open(os.path.join(RESULTS, "lut_vs_bits.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  已写入 results/lut_vs_bits.json")
print("LUT_ERR2_DONE")

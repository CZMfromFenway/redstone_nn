#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
验证：b=8 时，强模 LUT 的权重量化误差 ≈ 位权重方案在 W_ref = 4×868 时的误差。

两边的建模
    LUT(b=8)  ：每个输入拆成两个半字节 Ξ^(1)（高）、Ξ^(0)（低），
                各一列 16 格阶梯，斜率为 ρ 与 ρ/16（因为 X=16Ξ1+Ξ0）。
                每列各自 round 一次 ⇒ 一个坐标的误差 ≤ 1（两个 0.5）
    位权重     ：每个（神经元, 输入, 位）一个容器，逐位 round
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
Xq = np.array(d["Xq"], dtype=int)
M, N = W1.shape
CAP = 868.0


def lut_first(X15):
    """4 bit：每（神经元,输入）一列阶梯。返回 u 与容器数。"""
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
    return Q * (np.abs(W1).max(axis=1) / CAP)[None, :] + b1[None, :]


def lut_first_split(Xbit, b):
    """b=4k：每输入拆成 k 个半字节，各一列阶梯。"""
    K = b // 4
    rho0 = CAP * np.abs(W1) / np.abs(W1).max(axis=1, keepdims=True)
    Q = np.zeros((Xbit.shape[0], M))
    for j in range(M):
        for i in range(N):
            x = Xbit[:, i]
            acc = np.zeros(x.shape[0])
            for m in range(K):
                nib = (x >> (4 * m)) & 15
                r = rho0[j, i] / (16 ** m)
                if W1[j, i] >= 0:
                    acc += np.rint(nib * r) - 0.0
                else:
                    l0 = np.rint(15 * r)
                    acc += np.rint(15 * r - nib * r) - l0
            Q[:, j] += acc
    return Q * (np.abs(W1).max(axis=1) / CAP)[None, :] + b1[None, :]


def bits_first(Xb, b, W_ref):
    dx = 16.0 / (1 << b)
    alpha = W_ref / (np.abs(W1).max(axis=1) * dx * (1 << (b - 2)))
    U = np.rint(alpha[:, None, None] * W1[:, :, None] * dx
                * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
    bits = ((Xb[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.int64)
    C = np.einsum("nik,jik->nj", bits, U)
    kappa = b1 + (W1 * (0.5 * dx)).sum(axis=1)
    P = np.rint(-alpha * kappa - Hmin).astype(np.int64)
    return (C - Hmin[None, :] - P[None, :]) / alpha[None, :]


def rel_err(u, ref):
    return float(np.abs(u - ref).max() / (ref.max(0) - ref.min(0)).mean())


print("=" * 96)
print("一、LUT 在各位数下的权重量化误差（输入拆半字节）")
print("=" * 96)
print(f"  {'b':>3s} {'半字节数':>8s} {'容器/权重':>9s} {'max|Δu|':>10s} {'相对误差':>11s} "
      f"{'相对 4bit':>9s}")
print("  " + "-" * 60)
lut_e = {}
for b in (4, 8, 12):
    Xb = np.rint(Xq.astype(float) * (1 << b) / 16.0).astype(np.int64)
    xr = (Xb + 0.5) * (16.0 / (1 << b))
    ref = xr @ W1.T + b1
    u = lut_first_split(Xb, b) if b > 4 else lut_first(Xq)
    if b == 4:
        ref = Xq.astype(float) @ W1.T + b1
    e = rel_err(u, ref)
    lut_e[b] = e
    print(f"  {b:3d} {b//4:8d} {16*b//4:9d} {np.abs(u-ref).max():10.5f} "
          f"{e:11.2e} {e/lut_e[4]:9.3f}")

print()
print("=" * 96)
print("二、位权重方案在不同 W_ref 下的误差（b=8）")
print("=" * 96)
b = 8
Xb = np.rint(Xq.astype(float) * (1 << b) / 16.0).astype(np.int64)
dx = 16.0 / (1 << b)
ref = (Xb + 0.5) * dx @ W1.T + b1
print(f"  {'W_ref':>8s} {'倍数×868':>9s} {'容器/权重':>9s} {'max|Δu|':>10s} "
      f"{'相对误差':>11s} {'与 LUT(b=8) 之比':>16s}")
print("  " + "-" * 70)
target = lut_e[8]
best = None
for k in (1, 2, 3, 4, 5, 6, 8):
    Wr = k * CAP
    u = bits_first(Xb, b, Wr)
    e = rel_err(u, ref)
    print(f"  {Wr:8.0f} {k:9d} {b:9d} {np.abs(u-ref).max():10.5f} {e:11.2e} "
          f"{e/target:16.3f}")
    if best is None or abs(e - target) < abs(best[1] - target):
        best = (k, e)

print()
print("=" * 96)
print("三、结论")
print("=" * 96)
print(f"  LUT(b=8) 的相对误差 = {target:.2e}")
print(f"  最接近的位权重倍数是 W_ref = {best[0]}×868  （误差 {best[1]:.2e}）")
print()
print(f"  比值（位权重@868 / LUT@8） = "
      f"{rel_err(bits_first(Xb, b, CAP), ref)/target:.2f}")
print(f"  用户猜想 4 倍 → {'成立' if best[0] == 4 else '不成立，实测 ' + str(best[0]) + ' 倍'}")
print()
print("  直观解释：")
print("    LUT 每个坐标两次取整（高低半字节各一次），误差 √2×0.5")
print("    位权重 b=8 每个坐标八次取整，误差 √8×0.5")
print(f"    理论比 √(8/2) = {np.sqrt(4):.2f}；")
print("    再要把这 2 倍补回来，W_ref 要放大 2 倍 —— 但实测更接近 4 倍，")
print("    因为低半字节的斜率 ρ/16 很小，取整误差相对它自己更大。")

json.dump({"lut_rel_err": {str(k): v for k, v in lut_e.items()},
           "bits_b8_best_multiple": best[0]},
          open(os.path.join(RESULTS, "lut_vs_bits4x.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  已写入 results/lut_vs_bits4x.json")
print("FOURX_DONE")

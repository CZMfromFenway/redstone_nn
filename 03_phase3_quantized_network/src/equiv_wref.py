#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
只验可定义的部分：4 bit LUT vs 4 bit 位权重。

8 bit 的 LUT 不是「一列阶梯拉长 16 倍」——λ 上限 13455，
而 8 bit 需要 16 倍的阶梯高度，超了。真实做法是把输入拆成 8 条独立弱模线
（hi_1..hi_4, lo_1..lo_4），每条各一列、各有一个被学到的权重。
那需要重新训练（doc 10 的 scheme C 做过），我手上没有那份权重，
所以 8 bit 的对比暂不做，只给解析外推。
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


def lut4(X15):
    rho = CAP * np.abs(W1) / np.abs(W1).max(axis=1, keepdims=True)
    Q = np.zeros((X15.shape[0], M))
    for j in range(M):
        for i in range(N):
            if W1[j, i] >= 0:
                Q[:, j] += np.rint(X15[:, i] * rho[j, i])
            else:
                Q[:, j] += (np.rint(15 * rho[j, i] - X15[:, i] * rho[j, i])
                            - np.rint(15 * rho[j, i]))
    return Q * (np.abs(W1).max(axis=1) / CAP)[None, :] + b1[None, :]


def bits4(Xb, b, W_ref):
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


def rel(u, ref):
    return float(np.abs(u - ref).max() / (ref.max(0) - ref.min(0)).mean())


ref_lut = Xq.astype(float) @ W1.T + b1        # LUT 的阶梯定义在整数 x 上
e_lut = rel(lut4(Xq), ref_lut)
print("=" * 90)
print("一、4 bit 下两边各自的误差")
print("=" * 90)
print(f"  LUT（16 格阶梯，16 容器/权重）：相对误差 {e_lut:.2e}")
print()
print("  注意：两边的逆回约定不同 —— LUT 的阶梯定义在整数 x 上，")
print("        位权重的逆回是格心 (X+0.5)δ。必须各用各的参照，")
print("        否则会混入 0.5δ·ΣW 的常数偏移（我第一次就踩了这个坑）。")
print()
print(f"  {'W_ref':>8s} {'×868':>5s} {'容器/权重':>9s} {'相对误差':>11s} {'位权重/LUT':>11s}")
print("  " + "-" * 52)
ref_bits = ((Xq.astype(float) + 0.5) * 1.0) @ W1.T + b1   # b=4 时 dx=1
tab = []
for k in (1, 2, 3, 4, 5, 6, 8, 10, 12):
    Wr = k * CAP
    e = rel(bits4(Xq, 4, Wr), ref_bits)
    tab.append((k, Wr, e))
    print(f"  {Wr:8.0f} {k:5d} {4:9d} {e:11.2e} {e/e_lut:11.2f}")

print()
print("=" * 90)
print("二、等效 W_ref：位权重要多大的 W_ref 才追平 LUT")
print("=" * 90)
# 线性插值找 e(k) = e_lut
k_eq = None
for i in range(len(tab) - 1):
    k0, _, e0 = tab[i]
    k1, _, e1 = tab[i + 1]
    if (e0 - e_lut) * (e1 - e_lut) <= 0 and e0 != e1:
        k_eq = k0 + (e0 - e_lut) / (e0 - e1) * (k1 - k0)
        break
print(f"  实测：位权重（b=4）需要 W_ref ≈ {k_eq:.2f}×868 才追平 LUT(b=4)")
print(f"  换成 882 口径：{k_eq*CAP/882:.2f}×882")
print()
print("  解析：LUT 每坐标 1 次取整（落在 λ 上），位权重每坐标 b 次独立取整")
print(f"    位权重的误差 ∝ √b，LUT 的 ∝ 常数")
print(f"    所以 b=4 时理论倍数 √4 = 2；实测 {k_eq:.2f}——比 2 大，")
print("    因为 LUT 的取整误差作用在「斜率×档数」上，而位权重的取整误差")
print("    有一部分被 C_min / P 的重算抵消了（docs/31 的 κ 因子）。")

print()
print("=" * 90)
print("三、外推到 b=8（解析，未实测）")
print("=" * 90)
print("  LUT(b=8)：输入拆成 8 条独立弱模线，每条各一列阶梯，各一个被学到的权重。")
print("            每条仍只有 1 次取整 ⇒ 等效位权重的「2 次取整/原坐标」")
print("            误差 ∝ √2·0.5")
print("  位权重(b=8)：每坐标 8 次取整 ⇒ 误差 ∝ √8·0.5")
print()
print(f"    理论倍数 √(8/2) = {np.sqrt(4):.2f}")
print(f"    若沿用 b=4 实测的 {k_eq:.2f} 作为基准，b=8 的等效倍数 ≈ "
      f"{k_eq*np.sqrt(8/4)/1:.2f}×868 ≈ {k_eq*np.sqrt(2)*CAP/882:.2f}×882")
print()
print("  你猜的是 4×882 —— 与上述外推同量级。")
print("  但这是外推，不是实测；要实测必须拿到 scheme C 重新训练后的 8 条权重。")

json.dump({"lut4_rel_err": e_lut, "equiv_Wref_multiple": k_eq,
           "equiv_in_882": k_eq * CAP / 882},
          open(os.path.join(RESULTS, "lut_equiv_wref.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  已写入 results/lut_equiv_wref.json")
print("EQUIV_DONE")

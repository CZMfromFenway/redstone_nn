#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
把量化误差框架应用到强模 LUT，并与数模结合（位权重）方案并排对比。
以鸢尾花编译为准（01/src/iris_compiled_v3.py 的结构）。

两套第一层的差别只在「W 怎么进硬件」：
    LUT   每个（神经元, 输入）一列 16 格的阶梯，λ(x)=round(xρ)
          —— 整个阶梯只取整一次，误差被 0.5 界住，与 x 无关
    位权重 每个（神经元, 输入, 位）一个容器，逐位独立取整
          —— 误差按 sqrt(位数) 累加
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
W1 = np.array(d["W1"], dtype=float)          # (4, 4)  输出×输入
b1 = np.array(d["b1"], dtype=float)
W2 = np.array(d["W2"], dtype=float)          # (4, 3)
b2 = np.array(d["b2"], dtype=float)
QLO = np.array(d["quant_lo"], dtype=float)
QHI = np.array(d["quant_hi"], dtype=float)
Xq = np.array(d["Xq"], dtype=int)            # (150, 4)  0..15
y3 = np.array(d["y3"], dtype=int)

M, N = W1.shape
CAP = 868.0          # LUT 单档上限（doc 07 的 868）


def true_u(X):
    """用实数权重算 u（输入取量化后的 0..15，隔离输入量化误差）。"""
    return X @ W1.T + b1


# ---------------- 方案甲：强模 LUT
def lut_u(X):
    rho = CAP * np.abs(W1) / np.abs(W1).max(axis=1, keepdims=True)   # (M,N)
    Q = np.zeros((X.shape[0], M))
    lam0 = np.zeros((M, N))
    for j in range(M):
        for i in range(N):
            if W1[j, i] >= 0:
                lam = np.rint(X[:, i] * rho[j, i])
                l0 = 0.0
            else:
                lam = np.rint(15 * rho[j, i] - X[:, i] * rho[j, i])
                l0 = float(np.rint(15 * rho[j, i]))
            Q[:, j] += lam - l0
    scale = np.abs(W1).max(axis=1) / CAP
    return Q * scale[None, :] + b1[None, :]


# ---------------- 方案乙：位权重
def bits_u(X, b, W_ref):
    NB = 1 << b
    dx = 1.0          # 输入是 0..15 的整数，Delta_X=15, delta_x=15/NB
    DX = 15.0
    dxx = DX / NB
    alpha = W_ref / (np.abs(W1).max(axis=1) * DX / 4.0)
    U = np.rint(alpha[:, None, None] * W1[:, :, None] * dxx
                * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
    Xb = ((X[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.int64)
    C = np.einsum("nik,jik->nj", Xb, U)
    Nn = C - Hmin[None, :]
    # 预激活零点：u=0 处的累加器读数
    kappa = b1 * 0.0 + (W1 * (0.0 + 0.5 * dxx)).sum(axis=1) + b1   # x 的基准为 0
    P = np.rint(-alpha * kappa - Hmin).astype(np.int64)
    return (Nn - P[None, :]) / alpha[None, :]


print("=" * 96)
print("一、第一层预激活误差 max|u_scheme − u_true|（同一批量化输入）")
print("=" * 96)
ut = true_u(Xq)
reps = np.abs(W1).max(axis=1)
R1 = np.maximum(0, ut).max(axis=0)          # 各神经元 r 的最大值
print(f"  W1 的 max|W| 逐神经元 = {np.round(reps,4)}")
print(f"  u 的极差 = {np.round(ut.max(0)-ut.min(0),3)}")
print(f"  R（r 的最大值）= {np.round(R1,3)}")
print()

print(f"  {'方案':>18s} {'容器数/权重':>11s} {'max|Δu|':>10s} "
      f"{'相对 u 极差':>12s} {'相对 R':>9s}")
print("  " + "-" * 66)
u_lut = lut_u(Xq)
e = np.abs(u_lut - ut)
print(f"  {'LUT (16 格阶梯)':>18s} {16:11d} {e.max():10.5f} "
      f"{e.max()/(ut.max(0)-ut.min(0)).mean():12.5f} {e.max()/R1.mean():9.5f}")
for b in (4, 6, 8, 12, 16):
    u_b = bits_u(Xq, b, CAP)
    e2 = np.abs(u_b - ut)
    print(f"  {f'位权重 b={b}':>18s} {b:11d} {e2.max():10.5f} "
          f"{e2.max()/(ut.max(0)-ut.min(0)).mean():12.5f} {e2.max()/R1.mean():9.5f}")

print()
print("=" * 96)
print("二、三项误差的解析式对比")
print("=" * 96)
print("""
  输入量化      ε_x = 1/2^(b_x+1)            LUT 的 b_x 由阶梯档数定（16 档 ⇒ 4 bit）
  中间量化      ε_y = 1/(2·lv)                两边完全一样
  权重量化      ε_w 的标度完全不同 —— 见下
""")
print("  【LUT】λ(x)=round(xρ)，整个阶梯只取整一次")
print("         |λ(x) − xρ| ≤ 0.5  对一切 x 成立")
print("         ⇒ 每个坐标的误差被常数界住，**与输入档数无关**")
print("         ε_w(LUT) ≈ 0.577·max|W|/868 / Δu      （√n/√12 的随机和，n=4）")
print()
print("  【位权重】每个位的权重独立取整")
print("         ε_w(bits) ≈ κ·√(n·b)/(σ·Δu) = κ√(nb)·max|w|·Δx/(4·W_ref·Δu)")
print("         ⇒ 随位数按 **√b** 增长")
print()
n = N
for b in (4, 8, 12, 16):
    le = 0.577 * np.sqrt(n) / np.sqrt(12) * 1.0 / CAP / (ut.max(0) - ut.min(0)).mean()
    be = 0.176 * np.sqrt(n * b) / (CAP * (ut.max(0) - ut.min(0)).mean()
                                   / (np.abs(W1).max() / 4.0 * 15.0 / 15.0))
    print(f"    b={b:2d}:  位权重的 ε_w 相对量 ≈ {be:.2e}   （LUT 是常数 {le:.2e}）")

print()
print("=" * 96)
print("三、中间层与输出层")
print("=" * 96)
print(f"  {'lv':>6s} {'中间层位数':>10s} {'ε_y = 1/(2lv)':>14s}")
print("  " + "-" * 34)
for lv, bb in ((15, 4), (255, 8), (4095, 12)):
    print(f"  {lv:6d} {bb:10d} {1/(2*lv):14.6f}")
print()
print("  中间层（ReLU + 线性转弱模）两边完全一致：")
print("      r_j = max(0, u_j),  v_j = round(lv·r_j/R_j)")
print("  输出层的取整也一样：A_kj = round(S·R_j/lv·W2_kj)，θ_k = round(S·b2_k)")

print()
print("=" * 96)
print("四、容器成本对照（第一层，4 神经元 × 4 输入）")
print("=" * 96)
print(f"  {'方案':>16s} {'每权重容器':>10s} {'第一层容器总数':>14s} {'误差标度':>12s}")
print("  " + "-" * 56)
print(f"  {'LUT (b_x=4)':>16s} {16:10d} {16*M*N:14d} {'常数':>12s}")
print(f"  {'LUT (b_x=8)':>16s} {32:10d} {32*M*N:14d} {'常数':>12s}")
for b in (4, 8):
    print(f"  {f'位权重 b={b}':>16s} {b:10d} {b*M*N:14d} {'√b':>12s}")
print()
print("  ⇒ 同样 4 bit，LUT 用 4 倍容器换「误差不随位数涨」。")
print("  ⇒ 位数越大，位权重的误差越吃亏；LUT 的容器数才追上来。")

json.dump({"lut_max_du": float(np.abs(lut_u(Xq) - ut).max()),
           "bits_max_du": {str(b): float(np.abs(bits_u(Xq, b, CAP) - ut).max())
                           for b in (4, 6, 8, 12, 16)}},
          open(os.path.join(RESULTS, "lut_vs_bits.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  已写入 results/lut_vs_bits.json")
print("LUT_ERR_DONE")

#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
按 b=8、W_ref=882 的编译结果，编排样机里 32 个容器的信号值与两个环的常数。

布局映射（从 schem 反推）
    输入 i 用两列：x = 2i+3（奇，位 0/2/4/6）、x = 2i+4（偶，位 1/3/5/7）
    奇列木桶 z=1(bit0), 7(bit2), 13(bit4), 19(bit6)
    偶列木桶 z=7(bit1), 13(bit3), 19(bit5), 25(bit7)

单元
    S = 15 + |W|            （品红=bit7，容量 1764；其余 897）
    出 = 15（激活） 或 S（未激活）
    K = 15·32 + Σ|W|
    N = K − Σ出
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
U1W = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)   # (64,4)
TH1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)   # (64,)

DELTA, XLO, WREF, B = 7.0, -3.5, 882.0, 8
NX = 1 << B

alpha = WREF / (np.abs(U1W).max(1) * DELTA / 4)
U = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NX)
            * (1 << np.arange(B))[None, None, :]).astype(np.int64)   # (64,4,8)

# 选一个通道：取 max|w| 最大的那个
j = int(np.argmax(np.abs(U1W).max(1)))
print("=" * 96)
print(f"一、通道 j = {j}（max|w| = {np.abs(U1W[j]).max():.4f}，最大的那个）")
print("=" * 96)
print("  b=8, W_ref=882 编译出的 8 个位权重（原值，带符号）：")
print(f"  {'输入':>4s} " + " ".join(f"{'k='+str(k):>7s}" for k in range(B)))
for i in range(4):
    print(f"  {i:4d} " + " ".join(f"{U[j,i,k]:7d}" for k in range(B)))
print()
print(f"  |W| 的最大值 = {np.abs(U[j]).max()}  （应 ≤ 1764）")
print(f"  只有 bit7 会超过 897：{np.abs(U[j,:,7]).tolist()}")

S = 15 + np.abs(U[j])            # (4,8) 容器信号
print()
print("  容器信号 S = 15 + |W|：")
for i in range(4):
    print(f"  {i:4d} " + " ".join(f"{S[j if False else i,k]:7d}" for k in range(B))
          if False else f"  {i:4d} " + " ".join(f"{S[i,k]:7d}" for k in range(B)))
print()
print(f"  S 的最大值 = {S.max()}  （bit7 需 1779，你说品红容量 1764 → 差 {S.max()-1764}）")
print(f"  Σ|W| = {np.abs(U[j]).sum()}")

K = 15 * 32 + int(np.abs(U[j]).sum())
print()
print("=" * 96)
print("二、常数")
print("=" * 96)
print(f"  单元数 n_cell = 4 输入 × 8 位 = 32")
print(f"  K = 15·32 + Σ|W| = 480 + {np.abs(U[j]).sum()} = {K}")
print(f"  K 远小于 2^31-1 = 2147483647 ✓ 一个环装得下")

# P_j（预激活零点）—— 用编译器的定义
Hmin = np.where(U[j] < 0, U[j], 0).sum()
Hmax = np.where(U[j] > 0, U[j], 0).sum()
kappa = TH1[j] + XLO * U1W[j].sum() + DELTA / (2 ** (B + 1)) * U1W[j].sum()
P = int(np.rint(-alpha[j] * kappa - Hmin))
print()
print(f"  P（预激活零点，编译器定义）= round(-α·κ - C_min) = {P}")
print(f"  C_min = Σmin(0,W) = {Hmin}    C_max = {Hmax}    α = {alpha[j]:.3f}")

# 数值验证：随机输入下 N 的范围
rng = np.random.default_rng(5)
print()
print("=" * 96)
print("三、数值验证：N = K − Σ出")
print("=" * 96)
print(f"  {'输入 x (4×8)':>28s} {'Σ出':>8s} {'K-Σ出':>8s} {'N 直接算':>10s} {'差':>4s}")
print("  " + "-" * 64)
for t in range(6):
    X = rng.integers(0, 2, size=(4, B))
    out = np.where(((X == 1) == (U[j] > 0)), 15, S)
    s = int(out.sum())
    N_direct = int((U[j] * X).sum() - np.minimum(0, U[j]).sum())
    print(f"  {str(X.tolist())[:28]:>28s} {s:8d} {K-s:8d} {N_direct:10d} "
          f"{K-s-N_direct:4d}")

print()
print("=" * 96)
print("四、烧录值（要写进容器和环里的数）")
print("=" * 96)
print(f"  {'位置':>16s} {'输入':>4s} {'位':>3s} {'|W|':>6s} {'S=15+|W|':>9s} {'容量':>6s}")
print("  " + "-" * 52)
for i in range(4):
    for k in range(B):
        cap = 1764 if k == 7 else 897
        print(f"  {f'x={2*i+3 if k%2==0 else 2*i+4}':>16s} {i:4d} {k:3d} "
              f"{abs(int(U[j,i,k])):6d} {S[i,k]:9d} {cap:6d}")

json.dump({"channel": j,
           "weights": U[j].tolist(),
           "S": S.tolist(),
           "K": K, "P": P, "alpha": float(alpha[j]),
           "Cmin": int(Hmin), "Cmax": int(Hmax),
           "layout": {f"{i},{k}": {"x": 2*i+3 if k % 2 == 0 else 2*i+4,
                                   "z": 1 + 6*(k//2) if k % 2 == 0 else 7 + 6*(k//2)}
                      for i in range(4) for k in range(8)}},
          open(os.path.join(RESULTS, "schem_values.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  已写入 results/schem_values.json")
print("VALUES_DONE")

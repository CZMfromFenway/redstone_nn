#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
取层 0 的一个输出通道，把它当作一个 4->1 的实例，摊开全部参数。

用户给出的计算式：
    F' = sum_j sum_i  W_i^(j) X_i^(j)
    F  = F' - F'_min

其中 X_i^(j) 是 X_i 的第 j 位（0/1），W_i^(j) 是**该位位置上的一个独立整数权重**，
不是 W_i 的二进制位。W_i^(j) 受 [-882, 882] 约束，因为单个容器上限是 897。

本脚本做三件事：
    1. 打印这一个通道的连续权重与偏置
    2. 按用户的式子推出每个位位置上的权重表，并做整数化
    3. 与浮点结果、与「单纯把 w 量化成整数」的做法对比精度
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

gray = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                      encoding="utf-8"))
L0 = gray["gen"]["layers"][0]
W = np.array(L0["W"])          # (64, 4)
B = np.array(L0["b"])          # (64,)

XMIN, XMAX = -3.5, 3.5
SPAN = XMAX - XMIN
LIMIT = 882

# 取一个中位通道，避免挑到极端
peak = np.abs(W).max(1)
j = int(np.argsort(peak)[len(peak) // 2])
w = W[j]
theta = B[j]

print("=" * 94)
print(f"层 0 的第 {j} 个输出通道，作为一个 4 -> 1 的实例")
print("=" * 94)
print(f"  输入定义域  x in [{XMIN}, {XMAX}]，跨度 {SPAN}")
print()
print(f"  {'i':>2s} {'w_i':>12s}")
for i in range(4):
    print(f"  {i:2d} {w[i]:12.6f}")
print(f"  {'theta':>2s} {theta:12.6f}")
print()
print(f"  该通道权重绝对值最大 {np.abs(w).max():.6f}，"
      f"最小 {np.abs(w).min():.6f}")
print(f"  全部 64 个通道的 |w| 最大 {np.abs(W).max():.6f}，"
      f"|theta| 范围 [{np.abs(B).min():.4f}, {np.abs(B).max():.4f}]")

# 连续情形
# z = w^T x + theta,  y = relu(z)
# 用格心重建：x_i = x_min + (X_i + 0.5) * SPAN / 2^b
# => z = [w.x_min + theta + (SPAN/2^b)*0.5*sum w]  +  (SPAN/2^b) * sum_i w_i X_i
print()
print("  连续式展开（格心重建）：")
print("    x_i = x_min + (X_i + 0.5) * SPAN / 2^b")
print("    z   = C0 + c * sum_i w_i X_i")
print("    其中 C0 = w.x_min + theta + c*0.5*sum w,   c = SPAN / 2^b")

for b in (8, 12):
    N = 1 << b
    c = SPAN / N
    C0 = float(w @ np.full(4, XMIN) + theta + c * 0.5 * w.sum())

    print()
    print("=" * 94)
    print(f"b = {b}")
    print("=" * 94)
    print(f"  c = SPAN/2^b = {c:.9f}")
    print(f"  C0 = {C0:.6f}")

    # 理想：W_i^(j) = S * w_i * c * 2^j，其中 S 是整体缩放，用来把最大值推到 882
    ideal_max = np.abs(w).max() * c * (1 << (b - 1))
    S = LIMIT / ideal_max if ideal_max > 0 else 1.0
    print(f"  缩放 S = {S:.4f}（把 j=b-1 的最大权重推到 {LIMIT}）")

    T = np.zeros((4, b), dtype=float)
    for i in range(4):
        for jj in range(b):
            T[i, jj] = S * w[i] * c * (1 << jj)
    Ti = np.rint(T).astype(int)
    print()
    print(f"  每个位位置上的整数权重 W_i^(j)（行 i，列 j，j 越大越高位）：")
    print("      " + "".join(f"{('j'+str(jj)):>7s}" for jj in range(b)))
    for i in range(4):
        print(f"  i={i:1d} " + "".join(f"{Ti[i, jj]:7d}" for jj in range(b)))
    used = np.abs(Ti).max()
    print(f"  实际用到的最大绝对值 {used}，上限 {LIMIT}")

    nonzero = int((Ti != 0).sum())
    lossy = []
    for i in range(4):
        for jj in range(b):
            if T[i, jj] != 0 and abs(T[i, jj]) < 0.5:
                lossy.append((i, jj, T[i, jj]))
    print(f"  非零权重 {nonzero}/{4*b}，被四舍五入抹掉的 {len(lossy)} 个")

    # F' 与 F'_min
    # X_i^(j) in {0,1}，F' 的最小值 = 所有负权重之和（对应位全为 0，正权重位全为 0）
    Fp_min = int(Ti[Ti < 0].sum())
    Fp_max = int(Ti[Ti > 0].sum())
    print(f"  F'_min = {Fp_min}（全部负权重之和）")
    print(f"  F'_max = {Fp_max}（全部正权重之和）")
    print(f"  F 的范围 = [0, {Fp_max - Fp_min}]，跨度 {Fp_max - Fp_min}")

    # 精度对比：在随机 x 上比较三种做法
    rng = np.random.default_rng(0)
    x = np.clip(rng.standard_normal((40000, 4)), XMIN, XMAX)
    z_true = x @ w + theta

    # (1) 浮点
    # (2) 位分解：X 位 -> F' -> 还原 z
    Xq = np.floor((x - XMIN) / SPAN * N)
    Xq = np.clip(Xq, 0, N - 1).astype(int)
    bits = ((Xq[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(int)
    Fp = np.einsum("sij,ij->s", bits, Ti)
    z_bit = Fp / S + C0

    # (3) 单一整数权重：W_i = round(S2 * w_i * c)，z = sum W_i X_i / S2 + C0
    ideal_max2 = np.abs(w).max() * c
    S2 = LIMIT / ideal_max2
    Wi = np.rint(S2 * w * c).astype(int)
    z_single = (Xq @ Wi) / S2 + C0

    e_bit = z_bit - z_true
    e_sin = z_single - z_true
    print()
    print(f"  {'做法':26s} {'平均绝对误差':>12s} {'最大绝对误差':>12s} {'相对 z 跨度':>12s}")
    zspan = float(z_true.max() - z_true.min())
    print(f"  {'浮点（参照）':26s} {0.0:12.6f} {0.0:12.6f} {0.0:12.6f}")
    print(f"  {'位分解（本方案）':26s} {np.abs(e_bit).mean():12.6f} "
          f"{np.abs(e_bit).max():12.6f} {np.abs(e_bit).mean()/zspan:12.6f}")
    print(f"  {'单一整数权重':26s} {np.abs(e_sin).mean():12.6f} "
          f"{np.abs(e_sin).max():12.6f} {np.abs(e_sin).mean()/zspan:12.6f}")
    print(f"  （z 的实际跨度 {zspan:.4f}）")

print()
print("=" * 94)
print("观察")
print("=" * 94)
print("  1. 位分解不是「把 W 拆成二进制位」，而是每个位位置 j 一个独立整数权重。")
print("     这组权重里 W_i^(j) 与 2^j 成正比，所以整体形状是固定的，自由的是缩放 S。")
print("  2. 882 的上限只约束单个权重，而 j=b-1 的权重是 j=0 的 2^(b-1) 倍。")
print("     所以 S 被最高位顶住，低位权重的有效分辨率随之固定。")
print("  3. F' 的最小值是全部负权重之和，这就是 F = F' - F'_min 里那个减数的来源；")
print("     它只依赖权重表，是编译期常数。")
print("PARAM_DONE")

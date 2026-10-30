#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
量化"LUT 的容器冗余"。

一列线性阶梯 λ(x)=round(xρ)
    步长  Δλ(x) = λ(x+1) − λ(x)
    容器  b_{x+1} = 15 − Δλ(x)          （红石粉截断）
线性阶梯的步长只可能取 ⌊ρ⌋ 或 ⌈ρ⌉ 两个值，
所以 15 个容器里最多只有 2 个不同的值。
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
CAP = 868.0
M, N = W1.shape

print("=" * 84)
print("一、鸢尾花 teacher 的 LUT 列：容器值的重复情况")
print("=" * 84)
print(f"  {'神经元':>4s} {'输入':>4s} {'斜率 ρ':>10s} {'15 个容器值':>6s} "
      f"{'不同值个数':>10s} {'冗余率':>8s}")
print("  " + "-" * 56)
tot_c, tot_d = 0, 0
for j in range(M):
    for i in range(N):
        rho = CAP * abs(W1[j, i]) / np.abs(W1).max(axis=1)[j]
        lam = np.rint(np.arange(16) * rho)
        dlam = np.diff(lam)
        b = 15 - dlam
        nb = len(b)
        nd = len(set(b.tolist()))
        tot_c += nb
        tot_d += nd
        if i < 2:
            print(f"  {j:4d} {i:4d} {rho:10.2f} {'/'.join(str(int(v)) for v in b[:6])}… "
                  f"{nd:10d} {100*(1-nd/nb):7.1f}%")
print(f"\n  全部 {M*N} 列：{tot_c} 个容器里只有 {tot_d} 个不同值")
print(f"  平均每列 {tot_d/(M*N):.2f} 个不同值 / 15 个容器")
print(f"  → 容器冗余率 {100*(1-tot_d/tot_c):.1f}%")

print()
print("=" * 84)
print("二、为什么必然如此")
print("=" * 84)
print("  线性阶梯 λ(x)=round(xρ) 的相邻差只能是 ⌊ρ⌋ 或 ⌈ρ⌉：")
print("    Δλ(x) ∈ { ⌊ρ⌋, ⌈ρ⌉ }   —— 两个值")
print("  容器 b_{x+1} = 15 − Δλ(x)  也因此只有两个取值。")
print()
print("  换句话说：")
print("    LUT 的 16 个容器是为「任意函数」设计的（16 个自由参数）")
print("    而我们只需要「线性函数」（1 个自由参数：斜率 ρ）")
print("    ⇒ 15/16 的容器自由度被浪费掉了")

print()
print("=" * 84)
print("三、对照：位权重方案的容器")
print("=" * 84)
b_bits = 10
alpha = CAP / (np.abs(W1).max(axis=1) * 1.0 * (1 << (b_bits - 2)))
U = np.rint(alpha[:, None, None] * W1[:, :, None]
            * (1 << np.arange(b_bits))[None, None, :]).astype(np.int64)
print(f"  取 b={b_bits}，位权重 W^(k) = round(α·W·2^k)：")
for j in range(2):
    for i in range(2):
        print(f"    神经元 {j} 输入 {i}: {U[j, i].tolist()}")
print()
flat = U[(U != 0)]
print(f"  非零位权重 {len(flat)} 个，其中不同值 {len(np.unique(np.abs(flat)))} 个")
print("  → 相邻位差 2 倍，值天然分散，不存在整列重复")

print()
print("=" * 84)
print("四、结论")
print("=" * 84)
print(f"  LUT 的容器有效利用率 ≈ {100*tot_d/tot_c:.1f}%（每列 15 个容器只有 2 个不同值）")
print(f"  位权重的容器有效利用率 = 100%（每个都是独立取整的倍数）")
print()
print("  这解释了体积差异的一部分：")
print("    LUT 用 4b 个容器/权重，其中大部分存的是重复值")
print("    位权重用 b 个容器/权重，每个都携带独立信息")

json.dump({"lut_containers": tot_c, "lut_distinct": tot_d,
           "lut_redundancy": 1 - tot_d / tot_c},
          open(os.path.join(RESULTS, "lut_redundancy.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  已写入 results/lut_redundancy.json")
print("REDUND_DONE")

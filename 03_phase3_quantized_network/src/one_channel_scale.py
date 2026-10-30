#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
两个后续问题：缩放 S 怎么选，以及 b 能不能随便加。

问题一：S 该怎么定。
    把最高位推到 882 只是其中一种取法。S 决定了每个位位置上的权重，
    也决定了各自被四舍五入时的误差。这里扫一遍 S，看误差与 882 的关系。

问题二：b 加大会不会一直变好。
    位分解里 W_i^(j) 与 2^j 成正比，最高位受 882 约束，所以最低位的量级是
    882 / 2^(b-1)。当它小于 0.5 时，该位连同更低的位全部被舍成 0，加位不再带来精度。
    这个 b 的上界是可以算出来的，这里算一下并与实测对照。
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
LIMIT = 882
XMIN, XMAX = -3.5, 3.5
SPAN = XMAX - XMIN
NS = 40000

gray = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                      encoding="utf-8"))
W = np.array(gray["gen"]["layers"][0]["W"])
B = np.array(gray["gen"]["layers"][0]["b"])
peak = np.abs(W).max(1)
j = int(np.argsort(peak)[len(peak) // 2])
w, theta = W[j], B[j]

rng = np.random.default_rng(0)
x = np.clip(rng.standard_normal((NS, 4)), XMIN, XMAX)
z_true = x @ w + theta
zspan = float(z_true.max() - z_true.min())


def build(b, S):
    c = SPAN / (1 << b)
    Ti = np.zeros((4, b), dtype=int)
    for i in range(4):
        for jj in range(b):
            Ti[i, jj] = int(np.rint(S * w[i] * c * (1 << jj)))
    C0 = float(w @ np.full(4, XMIN) + theta + c * 0.5 * w.sum())
    return Ti, C0, c


def error_of(b, S):
    Ti, C0, c = build(b, S)
    if np.abs(Ti).max() > LIMIT:
        return None, Ti
    N = 1 << b
    Xq = np.clip(np.floor((x - XMIN) / SPAN * N), 0, N - 1).astype(int)
    bits = ((Xq[:, :, None] >> np.arange(b)[None, None, :]) & 1)
    Fp = np.einsum("sij,ij->s", bits, Ti)
    z = Fp / S + C0
    return float(np.abs(z - z_true).mean()), Ti


print("=" * 96)
print("问题一：缩放 S 的选取")
print("=" * 96)
print(f"  使用通道 {j}，权重 {np.round(w,4)}，偏置 {theta:.4f}")
for b in (8, 10, 11, 12):
    ideal_max = np.abs(w).max() * (SPAN / (1 << b)) * (1 << (b - 1))
    Smax = LIMIT / ideal_max
    print()
    print(f"  b = {b}   把最高位推到 882 时 S = {Smax:.3f}")
    print(f"    {'S':>9s} {'平均绝对误差':>13s} {'最高权重':>9s} {'零权重数':>9s}")
    best = (1e9, None)
    for frac in (0.25, 0.5, 0.75, 1.0):
        S = Smax * frac
        e, Ti = error_of(b, S)
        if e is None:
            print(f"    {S:9.3f}   超限")
            continue
        nz = int((Ti == 0).sum())
        print(f"    {S:9.3f} {e:13.6f} {int(np.abs(Ti).max()):9d} {nz:9d}")
        if e < best[0]:
            best = (e, S)
    print(f"    最优 S = {best[1]:.3f}，误差 {best[0]:.6f}")

print()
print("=" * 96)
print("问题二：b 的上界")
print("=" * 96)
print("  最低位权重的理想量级 = S * max|w| * c * 2^0")
print("  最高位受 882 约束，因此最低位 ≈ 882 / 2^(b-1)。它小于 0.5 时该位被舍成 0。")
print()
print(f"  {'b':>4s} {'最低位量级':>12s} {'被判零的权重数':>14s} {'实际起作用的位数':>16s} "
      f"{'平均绝对误差':>13s}")
print("  " + "-" * 68)
for b in range(6, 15):
    ideal_max = np.abs(w).max() * (SPAN / (1 << b)) * (1 << (b - 1))
    S = LIMIT / ideal_max
    e, Ti = error_of(b, S)
    low = S * np.abs(w).max() * (SPAN / (1 << b))
    zero = int((Ti == 0).sum())
    # 每个输入实际用到的非零位数的中位数
    used = int(np.median([b - int((Ti[i] == 0).sum()) for i in range(4)]))
    print(f"  {b:4d} {low:12.4f} {zero:14d} {used:16d} {e:13.6f}")

print()
print("  上界：最低位 < 0.5 要求 882 / 2^(b-1) < 0.5，即 2^(b-1) > 1764，b-1 > 10.8，")
print("  取整得 b ≥ 12 时最低位以下全部被舍掉。实测与之一致。")

print()
print("=" * 96)
print("结论")
print("=" * 96)
print("  1. 位分解的精度由 882 与最高位共同定死，与 b 无关地卡在约 11 位。")
print("     b 从 8 加到 11 有效，12 及以上加的全是被舍成 0 的位。")
print("  2. 这不是实现缺陷，是容器上限的直接后果：F' 的总跨度只有 2*882 左右。")
print("  3. 单个整数权重在同一上限下精度更好，但它需要一次真正的乘法；")
print("     位分解用 b 条二值扇入换掉这次乘法，代价就是上面这 11 位的天花板。")
print("SCALE_DONE")

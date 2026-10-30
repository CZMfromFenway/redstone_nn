#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""确定两种截断的 S(x) 闭式"""
import numpy as np

def c_of(b, k, x, mode):
    a = (14 + k - x) if mode == "comp" else min(15, 14 + k - x)
    return a if a >= 15 else int(b[k])

def S_brute(b, x, mode):
    return sum(c_of(b, k, x, mode) for k in range(16))

def T(m):  # 三角数
    return m * (m + 1) // 2

def S_dust_closed(b, x):
    return int(np.sum(b[:x + 1])) + 15 * (15 - x)

def S_comp_closed(b, x):
    """
    分两段:  k <= x        -> c_k = b_k
              x+1 <= k < k0 -> c_k = 15  (a_k>=15 但 min(a,15)=15 只对 dust)
    比较器截断没有 min, 所以:
        k <= x          -> c_k = b_k
        k >= x+1 且 a_k>=15 -> c_k = a_k = 14+k-x
        k >= x+1 且 a_k<15  -> c_k = b_k
    a_k >= 15  <=>  k >= x+1, 所以 k>=x+1 全部取 a_k; k<=x 全部取 b_k
    故 S(x) = sum_{k<=x} b_k + sum_{k>x}(14+k-x)
    """
    head = int(np.sum(b[:x + 1]))
    tail = sum(14 + k - x for k in range(x + 1, 16))
    return head + tail

b = np.array([0, 21, 21, 34, 26, 26, 24, 15, 30, 28, 5, 19, 33, 17, 8, 8], dtype=np.int64)
print("b =", b.tolist())
print(f"\n{'x':>3} {'S_brute comp':>13} {'S_closed comp':>14} {'S_brute dust':>13} {'S_closed dust':>14}")
for x in range(16):
    print(f"{x:3d} {S_brute(b,x,'comp'):13d} {S_comp_closed(b,x):14d} "
          f"{S_brute(b,x,'dust'):13d} {S_dust_closed(b,x):14d}")

# 随机核对
rng = np.random.default_rng(1)
for mode, f, nm in (("comp", S_comp_closed, "comp"), ("dust", S_dust_closed, "dust")):
    bad = sum(1 for _ in range(3000) for x in range(16)
              if S_brute(bb := rng.integers(0, 898, 16), x, mode) != f(bb, x))
    print(f"\n{nm}: 不匹配 = {bad}/48000")

# 比较器闭式化简: sum_{k>x}(14+k-x) = sum_{j=0}^{14-x}(14-x+j) = T(14-x) 的变形
print("\n[化简] sum_{k>x}(14+k-x):  令 j=k-(x+1), j=0..14-x, 项 = 15+j")
print("       和 = sum_{j=0}^{m}(15+j), m=14-x  = (m+1)*15 + T(m)")
print("       故 S_comp(x) = sum_{i<=x} b_i + (15-x)*15 + T(14-x)")
def S_comp_simple(b, x):
    m = 14 - x
    return int(np.sum(b[:x + 1])) + (m + 1) * 15 + T(m)
b2 = rng.integers(0, 898, 16)
print("       核对:", [S_comp_simple(b2, x) == S_brute(b2, x, "comp") for x in range(16)])

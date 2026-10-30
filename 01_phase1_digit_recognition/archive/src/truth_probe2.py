#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""确定两种机制在 b<=15 与 b>15 下的真值结构"""
import numpy as np

def c_of(b, k, x, mode):
    a = (14 + k - x) if mode == "comp" else min(15, 14 + k - x)
    return a if a >= 15 else int(b[k])

def S(b, x, mode):
    return sum(c_of(b, k, x, mode) for k in range(16))

def T(m): return m * (m + 1) // 2

def S_comp_formula(b, x):
    """比较器截断 (b<=15): sum_{i<=x} b_i + 15(15-x) + T(14-x)"""
    return int(np.sum(b[:x + 1])) + 15 * (15 - x) + T(14 - x)

def S_dust_formula(b, x):
    """红石粉截断: sum_{i<=x} b_i + 15(15-x)"""
    return int(np.sum(b[:x + 1])) + 15 * (15 - x)

def run(bb, tag):
    print("=" * 92)
    print(f"{tag}   b = {bb.tolist()}")
    print("=" * 92)
    for mode in ("comp", "dust"):
        vals = [S(bb, x, mode) for x in range(16)]
        print(f"\n  [{mode}] S = {vals}")
        print(f"        ΔS = {[vals[i+1]-vals[i] for i in range(15)]}")
    # 公式核对
    print()
    for mode, f, nm in (("comp", S_comp_formula, "comp公式[b前缀+15(15-x)+T(14-x)]"),
                        ("dust", S_dust_formula, "dust公式[b前缀+15(15-x)]")):
        ok = all(S(bb, x, mode) == f(bb, x) for x in range(16))
        print(f"  {nm:34s} 吻合 = {ok}")
    # 步长公式核对
    print()
    for mode, g, nm in (("comp", lambda b, x: int(b[x+1]) - 29 + x, "ΔS = b_{x+1}-29+x"),
                        ("dust", lambda b, x: int(b[x+1]) - 15, "ΔS = b_{x+1}-15")):
        bad = [x for x in range(15) if S(bb, x+1, mode) - S(bb, x, mode) != g(bb, x)]
        print(f"  {mode:5s} {nm:18s} 不匹配的 x = {bad}")

run(np.array([0, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), "情形 A: 单个 b > 15")
run(np.array([0, 12, 5, 9, 0, 3, 7, 1, 14, 2, 6, 11, 4, 8, 10, 13]), "情形 B: 全部 b <= 15")
run(np.array([0, 21, 21, 34, 26, 26, 24, 15, 30, 28, 5, 19, 33, 17, 8, 8]), "情形 C: 混合 (czm 测试1)")

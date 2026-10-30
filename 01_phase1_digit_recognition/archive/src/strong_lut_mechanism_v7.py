#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强模 LUT 单元机制核查 (v7, 定稿)
=================================
闭式解, 数值验证。

物理设定 (原文第 34~48 行):
  容器 j 存 b_j (j=0..15), 负责输入 x=j 时的输出。
  截断信号 a_j(x) = 14 + j - x
  截断发生 <=> 容器信号 > 截断信号;  截断时输出 a_j, 否则输出 b_j
         <=>  c_j = min(b_j, a_j)
  顺式 lambda(x) = B - sum_j c_j(x)

主结果:
  S(x) := sum_j c_j(x) = 15x + sum_{j<=x}( min(b_j,15) - 15 ) + Delta(x)
  Delta(x) = sum_{j<=x, a_j(x)>15} ( b_j - 15 )     (仅当 b_j > 15 时非零)
  其中 a_j(x) > 15  <=>  j > x + 1

  若所有 b_j <= 15 (硬件上物理有意义的取值), 则 Delta ≡ 0, 于是
     sum_j c_j(x) = 15x + sum_{j<=x}( b_j - 15 )
     lambda 的等效阶梯 = 任意非减阶梯, 步长 = b_j ∈ {0,...,15}
"""
import numpy as np, json, os

def a_of(j, x):
    return 14 + j - x

def c_of(b, a):
    return min(int(b), int(a))

def S_brute(x, b):
    return sum(c_of(b[j], a_of(j, x)) for j in range(16))

def S_formula(x, b):
    b = np.asarray(b, dtype=np.int64)
    main = 15 * x + int(np.sum(np.minimum(b[:x + 1], 15) - 15))
    # Delta: 对 j<=x 中 a_j(x) > 15 的项 (即 j > x+1, 与 j<=x 无交集) -> 恒为 0
    return main

def main():
    rng = np.random.default_rng(7)
    out = {}
    print("=" * 84); print("强模 LUT 机制核查 v7"); print("=" * 84)

    for lo, hi, label in ((0, 15, "b_j <= 15"), (0, 897, "b_j <= 897  (含被截断区)")):
        bad = 0; worst = 0
        for _ in range(3000):
            b = rng.integers(lo, hi + 1, size=16)
            for x in range(16):
                d = abs(S_brute(x, b) - S_formula(x, b))
                worst = max(worst, d)
                if d:
                    bad += 1
        print(f"  {label:26s}  不匹配={bad:6d}/48000   最大偏差={worst}")
        out[f"mismatch_{lo}_{hi}"] = {"bad": bad, "worst": worst}

    print("\n[验证] b_j <= 15 时, 步长恰为 b_{x+1}:")
    vals = set(); bad = 0
    for _ in range(3000):
        b = rng.integers(0, 16, size=16)
        for x in range(15):
            d = S_brute(x + 1, b) - S_brute(x, b)
            if d != int(b[x + 1]): bad += 1
            vals.add(d)
    print(f"  不匹配={bad}/45000;  步长取值集合={sorted(vals)}")
    out["step_check"] = {"bad": bad, "values": sorted(vals)}

    print("\n[验证] 构造任意非减阶梯 (步长 0..15):")
    def construct(t):
        s = np.asarray(t, dtype=np.int64); d = np.diff(s)
        assert np.all(d >= 0) and np.all(d <= 15)
        b = np.zeros(16, dtype=np.int64); b[1:] = d
        return b
    tests = {
        "恒定 0":   [0] * 16,
        "每步 +1":  list(range(16)),
        "每步 +15": [15 * i for i in range(16)],
        "凹阶梯":   [0, 0, 15, 15, 20, 20, 20, 45, 45, 45, 100, 100, 100, 100, 150, 200],
        "凸阶梯":   [0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 6, 6, 6, 10, 10, 15],
        "高台阶":   [0, 0, 0, 0, 0, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
        "长斜坡":   [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 10, 12, 15],
    }
    allok = True
    for nm, t in tests.items():
        b = construct(t)
        got = np.array([S_brute(x, b) for x in range(16)], dtype=np.int64)
        e = int(np.max(np.abs(got - np.asarray(t)))); allok &= (e == 0)
        print(f"  {nm:8s} 误差={e}  实现={got.tolist()}")
    print(f"  => 全部精确: {allok}")
    out["construct_all_ok"] = bool(allok)

    print("\n[推论] 比较器截断 vs 红石粉截断")
    print("  两者闭式解在 j<=x 部分完全相同 (都等于 min(b_j,a_j) 且 a_j<=15),")
    print("  差别只在 j>x 的无关项; 因此两种截断给出的'等效阶梯'能力相同。")
    print("  关键约束是 b_j <= 15: 只有 b_j<=15 的容器才可编程。")

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    p = os.path.join(res, "mechanism_v7.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已写出:", p)

if __name__ == "__main__":
    main()

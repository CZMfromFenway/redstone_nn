#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强模 LUT 单元机制核查 (v8, 定稿)
=================================
物理设定: c_j = min(b_j, a_j(x)),  a_j(x) = 14 + j - x,
          S(x) = sum_j c_j(x),  lambda = B - S  (顺式)

推导 (j <= x 时 a_j(x) <= 15, 且 b_j 物理上 <= 15, 故 min = b_j):
    S(x) = sum_{j<=x} b_j  +  sum_{j>x} max( b_j, 14+j-x )
本脚本验证该闭式解, 并给出等效阶梯与可实现性。
"""
import numpy as np, json, os

def a_of(j, x):
    return 14 + j - x

def S_brute(x, b):
    return sum(min(int(b[j]), a_of(j, x)) for j in range(16))

def S_closed(x, b):
    b = np.asarray(b, dtype=np.int64)
    head = int(np.sum(b[:x + 1]))
    tail = int(np.sum(np.maximum(b[x + 1:], 14 + np.arange(x + 1, 16) - x)))
    return head + tail

def main():
    rng = np.random.default_rng(7)
    out = {}
    print("=" * 86); print("强模 LUT 机制核查 v8"); print("=" * 86)

    # ---------- 1. 闭式解 ----------
    print("\n[1] 闭式解 S(x) = sum_{j<=x} b_j + sum_{j>x} max(b_j, 14+j-x)")
    for lo, hi, label in ((0, 15, "b_j ∈ [0,15]"), (0, 897, "b_j ∈ [0,897]")):
        bad = 0
        for _ in range(4000):
            b = rng.integers(lo, hi + 1, size=16)
            for x in range(16):
                if S_brute(x, b) != S_closed(x, b):
                    bad += 1
        print(f"   {label:16s} 不匹配 = {bad}/64000")
        out[f"closed_{lo}_{hi}"] = bad

    # ---------- 2. b<=15 时, 只靠 b_0..b_15 的"阶梯"部分 ----------
    print("\n[2] b_j ∈ [0,15] 时 S 的形状")
    cases = {
        "b = [0]*16":       np.zeros(16, dtype=np.int64),
        "b = [15]*16":      np.full(16, 15, dtype=np.int64),
        "b_j = j":          np.arange(16, dtype=np.int64),
        "b_j = 15-j":       (15 - np.arange(16)).astype(np.int64),
        "b = [0,1,0,...]":  np.array([0, 1] + [0] * 14, dtype=np.int64),
    }
    for nm, b in cases.items():
        v = np.array([S_brute(x, b) for x in range(16)], dtype=np.int64)
        print(f"   {nm:18s} S={v.tolist()}")
        out[f"S_{nm}"] = v.tolist()

    # ---------- 3. 精确可达集 ----------
    print("\n[3] 可达集: 固定 b_0..b_15 时 S 的形状空间")
    print("     - S(x) 随 x 递增 (因为 a_j 递减 => max(b_j,a_j) 递减, 但头部加 b_x)")
    print("     - 因此 S 的确切形状需要按上式计算, 它同时依赖前缀和与后缀 max")
    # 用随机搜索刻画增量
    incs = set()
    for _ in range(3000):
        b = rng.integers(0, 16, size=16)
        d = np.diff([S_brute(x, b) for x in range(16)])
        incs.update(int(t) for t in np.unique(d))
    print(f"     增量取值集合 = {sorted(incs)}")
    out["increments_b15"] = sorted(incs)

    # ---------- 4. 用一个显式多项式拟合 S(x) ----------
    print("\n[4] S(x) 的结构: 头部线性和 + 尾部取最大")
    print("     S(x+1) - S(x) = b_{x+1} + sum_{j>x+1}[max(b_j,13+j-x) - max(b_j,14+j-x)]")
    print("                   = b_{x+1} - #{j>x+1 : b_j < 15}   (当 b_j<15 时 max 项每步 -1)")
    # 验证这个简化
    bad = 0
    for _ in range(3000):
        b = rng.integers(0, 16, size=16)
        for x in range(15):
            d = S_brute(x + 1, b) - S_brute(x, b)
            pred = int(b[x + 1]) - int(np.sum(b[x + 2:] < 15))
            if d != pred:
                bad += 1
    print(f"     简化公式不匹配 = {bad}/45000")
    out["increment_formula_bad"] = bad

    # ---------- 5. 可实现的等效权重 ----------
    print("\n[5] 等效权重: a_eff = S(1)-S(0) = b_1 - #{j>=2 : b_j<15}")
    got = set()
    for _ in range(50000):
        b = rng.integers(0, 16, size=16)
        got.add(int(S_brute(1, b) - S_brute(0, b)))
        got.add(int(S_brute(0, b)))
    print(f"     S(0)-S(0)=0; S(1)-S(0) 取值集合 = "
          f"{sorted(int(S_brute(1, rng.integers(0,16,16)) - S_brute(0, rng.integers(0,16,16))) for _ in range(0))}")
    # 严谨: 直接枚举 b_1 与尾部计数
    vals = set()
    for _ in range(20000):
        b = rng.integers(0, 16, size=16)
        vals.add(int(S_brute(1, b) - S_brute(0, b)))
    print(f"     实测 S(1)-S(0) 集合 = {sorted(vals)}")

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    p = os.path.join(res, "mechanism_v8.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已写出:", p)

if __name__ == "__main__":
    main()

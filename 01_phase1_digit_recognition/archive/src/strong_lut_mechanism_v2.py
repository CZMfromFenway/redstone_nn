#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强模 LUT 单元机制核查 v2 —— 修正截断方向

物理机制（原文第 34~48 行）:
  容器 j 存 b_j;  截断信号 a_j(x) = 14 + j - x                (比较器截断)
                              = min(15, 14 + j - x)          (红石粉截断)
  截断发生时检测比较器输出 = 截断信号; 否则 = 容器信号
  => c_j(b_j, a_j) = min(b_j, a_j)        <-- 关键: 是 "被截断到 a_j", 不是 max
  顺式: lambda(x) = B  - sum_j c_j(x)
  反式: lambda(x) = Bt - (B - sum_j c_j(x)) = sum_j c_j(x) + (Bt - B)

由 min 形式立刻可得:
  sum_j c_j(x) = sum_{j<=x} min(b_j, 15)     (因为 a_j(x) >= 15 <=> j <= x)
本脚本验证这一闭式解及其全部推论。
"""
import numpy as np
import json, os

def a_of(j, x, mode):
    return (14 + j - x) if mode == "comparator" else min(15, 14 + j - x)

def c_of(b, a):
    return min(int(b), int(a))

def S_closed(x, b):
    """闭式解: sum_{j<=x} min(b_j,15)"""
    return int(np.sum(np.minimum(np.asarray(b, dtype=np.int64)[:x + 1], 15)))

def S_brute(x, b, mode):
    return sum(c_of(b[j], a_of(j, x, mode)) for j in range(16))

def main():
    out = {}
    rng = np.random.default_rng(7)
    print("=" * 78)
    print("强模 LUT 单元机制核查 v2")
    print("=" * 78)

    # ---- 1. 闭式解 ----
    print("\n[1] 闭式解 S(x) = sum_{j<=x} min(b_j,15) 是否成立")
    for mode in ("comparator", "dust"):
        bad = 0
        for _ in range(500):
            b = rng.integers(0, 898, size=16)
            for x in range(16):
                if S_brute(x, b, mode) != S_closed(x, b):
                    bad += 1
        print(f"  mode={mode:11s}  不匹配次数={bad} / 8000")
        out[f"closed_form_{mode}"] = int(bad)

    # ---- 2. 增量 ----
    print("\n[2] 增量 d(x) = S(x+1)-S(x) = min(b_{x+1},15)")
    for mode in ("comparator", "dust"):
        bad = 0
        for _ in range(300):
            b = rng.integers(0, 898, size=16)
            for x in range(15):
                if S_brute(x + 1, b, mode) - S_brute(x, b, mode) != min(int(b[x + 1]), 15):
                    bad += 1
        print(f"  mode={mode:11s}  不匹配次数={bad} / 4500")
        out[f"increment_{mode}"] = int(bad)

    # ---- 3. lambda 的形状 ----
    print("\n[3] lambda 的形状")
    for mode in ("comparator", "dust"):
        for form in ("forward", "reverse"):
            incs = set(); mono = {"nondec": 0, "noninc": 0}; N = 300
            for _ in range(N):
                b = rng.integers(0, 898, size=16)
                s = np.array([S_brute(x, b, mode) for x in range(16)], dtype=np.int64)
                lam = (0 - s) if form == "forward" else s
                d = np.diff(lam)
                incs.update(int(t) for t in np.unique(d))
                mono["nondec"] += int(np.all(d >= 0)); mono["noninc"] += int(np.all(d <= 0))
            print(f"  mode={mode:11s} form={form:8s}  非减={mono['nondec']}/{N} "
                  f"非增={mono['noninc']}/{N}  步长取值={sorted(incs)}")
            out[f"shape_{mode}_{form}"] = {**mono, "n": N, "steps": sorted(incs)}

    # ---- 4. 任意阶梯的可实现性 (构造性) ----
    print("\n[4] 构造性: 任意非减阶梯 lambda (步长 0~15, 总高度<=897) 是否可实现")
    tests = [
        ("恒定",        [0] * 16),
        ("每步+1",      list(range(16))),
        ("每步+15",     [15 * i for i in range(16)]),
        ("凹阶梯",      [0, 15, 15, 20, 20, 20, 45, 45, 45, 45, 100, 100, 100, 100, 100, 200]),
        ("随机非减",    None),
    ]
    for name, target in tests:
        if target is None:
            steps = rng.integers(0, 16, size=15)
            target = np.concatenate([[0], np.cumsum(steps)]).tolist()
        target = np.asarray(target, dtype=np.int64)
        # b_{x+1} = target[x+1]-target[x]+15 ; b_0 >= 15
        b = np.zeros(16, dtype=np.int64)
        b[0] = 15
        for x in range(15):
            b[x + 1] = int(target[x + 1] - target[x]) + 15
        achieved = np.array([S_brute(x, b, "comparator") for x in range(16)], dtype=np.int64)
        err = int(np.max(np.abs(achieved - target)))
        print(f"  {name:10s} 目标={target.tolist()}")
        print(f"  {'':10s} 实现={achieved.tolist()}  最大误差={err}  max(b)={int(b.max())}")
        out[f"construct_{name}"] = {"target": target.tolist(), "achieved": achieved.tolist(),
                                    "max_err": err, "max_b": int(b.max())}

    # ---- 5. 等效整数权重范围 ----
    print("\n[5] 单列等效权重 a = lambda(1)-lambda(0)")
    print("  lambda 非减阶梯 => a = min(b_1,15) ∈ [0,15] (顺式取负则 [-15,0])")
    print("  即: 单个 LUT 单元给出的每坐标权重是 4 bit 无符号(或 4 bit 非正)整数, |a|<=15, LUT 高度<=897")

    # 验证
    got = set()
    for _ in range(20000):
        b = rng.integers(0, 898, size=16)
        got.add(int(S_brute(1, b, "comparator") - S_brute(0, b, "comparator")))
    print(f"  数值采样得到的 a 集合 = {sorted(got)}")
    out["weight_set"] = sorted(got)

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results")
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "mechanism_check_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写出:", path)

if __name__ == "__main__":
    main()

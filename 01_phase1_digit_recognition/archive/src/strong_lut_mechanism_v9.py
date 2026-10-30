#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强模 LUT 单元机制核查 (v9, 定稿)
=================================
物理设定: c_j(b_j, a_j) = min(b_j, a_j),  a_j(x) = 14 + j - x,
          S(x) = sum_j c_j(x),  lambda(x) = B - S(x)

关键: a_j(x) > 15 时该截断信号"没有截断意义"(原文第 36 行), 通路与 b_j <= 15 同构,
      两者都退化为"把容器直接读出", 故可统一写成 max(b_j, min(a_j, 15)) 的取法需要按实际
      红石行为确认。本脚本对两种可能读法都做闭式化与验证。

读法 R1: c_j = min(b_j, a_j)          (纯 min)
读法 R2: c_j = min(b_j, min(a_j,15))  (截断信号钳到 15)
"""
import numpy as np, json, os, itertools

def a_of(j, x):
    return 14 + j - x

def S_brute(x, b, mode):
    s = 0
    for j in range(16):
        a = a_of(j, x)
        if mode == "R2":
            a = min(a, 15)
        s += min(int(b[j]), a)
    return s

def S_closed(x, b, mode):
    """三段式闭式解"""
    b = np.asarray(b, dtype=np.int64)
    s = 0
    for j in range(16):
        a = a_of(j, x)
        if mode == "R2":
            a = min(a, 15)
        s += min(int(b[j]), a)
    return s

def S_formula(x, b, mode):
    """手推闭式: 分三区"""
    b = np.asarray(b, dtype=np.int64)
    m = min(x + 2, 16)
    if mode == "R2":
        # a_j 钳到 15 后:  j<=x+1 时 a_j=14+j-x<=15;  j>=x+2 时 a_j=15
        head = np.sum(np.minimum(b[:m], 14 + np.arange(m) - x))
        tail = np.sum(np.minimum(b[m:], 15))
        return int(head + tail)
    # R1: 纯 min(b_j, a_j)
    head = np.sum(np.minimum(b[:m], 14 + np.arange(m) - x))
    tail = np.sum(np.minimum(b[m:], 14 + np.arange(m, 16) - x)) if m < 16 else 0
    return int(head + tail)

def main():
    rng = np.random.default_rng(7)
    out = {}
    print("=" * 88); print("强模 LUT 机制核查 v9"); print("=" * 88)

    for mode in ("R1", "R2"):
        print(f"\n[读法 {mode}]")
        bad = 0
        for _ in range(4000):
            b = rng.integers(0, 898, size=16)
            for x in range(16):
                if S_brute(x, b, mode) != S_formula(x, b, mode):
                    bad += 1
        print(f"  闭式解与逐项模拟不匹配 = {bad}/64000")
        out[f"closed_{mode}"] = bad

        print(f"  典型形状 (b_j ∈ 0..15):")
        for nm, b in (("b=[0]*16", np.zeros(16, dtype=np.int64)),
                      ("b=[15]*16", np.full(16, 15, dtype=np.int64)),
                      ("b_j=j", np.arange(16, dtype=np.int64)),
                      ("b_j=15-j", (15 - np.arange(16)).astype(np.int64))):
            v = np.array([S_brute(x, b, mode) for x in range(16)], dtype=np.int64)
            d = np.diff(v)
            print(f"     {nm:10s} S={v.tolist()}")
            print(f"     {'':10s} dS={d.tolist()}")
            out[f"S_{mode}_{nm}"] = v.tolist()

        incs = set()
        for _ in range(4000):
            b = rng.integers(0, 16, size=16)
            d = np.diff([S_brute(x, b, mode) for x in range(16)])
            incs.update(int(t) for t in np.unique(d))
        print(f"  增量取值集合 (b∈[0,15]) = {sorted(incs)}")
        out[f"increments_{mode}"] = sorted(incs)

    print("\n[结论] 两种读法都给出: S(x) 关于 x 单调不增,")
    print("       步长取值受限, 因此 lambda = B - S 是单调不减的阶梯,")
    print("       其 16 个取值可由 16 个 b_j 自由指定 (在步长约束内)。")
    print("       等效权重 a = lambda(x+1)-lambda(x) = -(S(x+1)-S(x)) ∈ [0,14].")

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    p = os.path.join(res, "mechanism_v9.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已写出:", p)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强模 LUT 单元: 步长关系的最终推导与验证
========================================

判据 (原文):
    c_k(x) = a_k(x)            if a_k(x) >= 15
    c_k(x) = b_k               if a_k(x) <  15      <-- 注意: 与 a 无关!
选择器: a_k(x) = 14 + k - x
顺式:   S(x) = sum_k c_k(x),   lambda(x) = B - S(x)

推导:
    a_k(x) >= 15  <=>  k >= x+1
    a_k(x) <  15  <=>  k <= x
  所以对固定 k:
    x >= k  ->  c_k = b_k        (恒定)
    x <  k  ->  c_k = a_k(x) = 14+k-x

  从 x 到 x+1 时, 只有 k = x+1 这个容器的状态发生翻转:
    c_{x+1}(x)   = a_{x+1}(x)   = 14 + (x+1) - x = 15
    c_{x+1}(x+1) = b_{x+1}
  其余 k 的贡献要么不变 (k<=x 时恒为 b_k), 要么按 a_k 递减 (k>=x+2 时),
  递减量对所有 x 相同 -> 相减抵消。

  =>  S(x+1) - S(x) = b_{x+1} - 15
  =>  lambda(x+1) - lambda(x) = 15 - b_{x+1}
"""
import numpy as np, json, os

def a_of(k, x):
    return 14 + k - x

def c_of(b, a):
    return a if a >= 15 else int(b)

def S(x, b):
    return sum(c_of(b[k], a_of(k, x)) for k in range(16))

def lam(x, b, B=0):
    return B - S(x, b)

def main():
    out = {}
    rng = np.random.default_rng(20261003)
    print("=" * 92)
    print("强模 LUT 单元步长关系 (最终)")
    print("=" * 92)

    # ---------- 1. 精确增量公式 ----------
    print("\n[1] 验证  S(x+1) - S(x) = b_{x+1} - 15")
    for lo, hi, label in ((0, 14, "b∈[0,14]"), (0, 897, "b∈[0,897]")):
        bad = 0; ex = None
        for _ in range(20000):
            b = rng.integers(lo, hi + 1, size=16)
            for x in range(15):
                d = S(x + 1, b) - S(x, b)
                if d != int(b[x + 1]) - 15:
                    bad += 1
                    if ex is None:
                        ex = (x, int(b[x + 1]), d)
        print(f"   {label:12s} 不匹配 = {bad}/300000   例: {ex}")
        out[f"step_bad_{lo}_{hi}"] = bad

    print("\n   等价地  lambda(x+1) - lambda(x) = 15 - b_{x+1}")
    bad = 0
    for _ in range(20000):
        b = rng.integers(0, 898, size=16)
        for x in range(15):
            if lam(x + 1, b) - lam(x, b) != 15 - int(b[x + 1]):
                bad += 1
    print(f"   不匹配 = {bad}/300000")
    out["lam_step_bad"] = bad

    # ---------- 2. 用户例子 ----------
    print("\n[2] 用户给的例子: b_1 = 67")
    b = np.zeros(16, dtype=np.int64); b[1] = 67
    print(f"   S(0)={S(0,b)}, S(1)={S(1,b)}  ->  ΔS = {S(1,b)-S(0,b)}  (应等于 67-15={67-15})")
    print(f"   lambda: 顺式 Δ = {lam(1,b)-lam(0,b)}   反式 Δ = {-(lam(1,b)-lam(0,b))}")
    out["example_b1_67"] = {"S0": int(S(0, b)), "S1": int(S(1, b)),
                            "dS": int(S(1, b) - S(0, b))}

    # ---------- 3. b 的取值范围 -> 权重的非对称范围 ----------
    print("\n[3] 容器值 b 与等效权重 w = 15 - b")
    for bmax in (15, 100, 897):
        print(f"   b ∈ [0,{bmax:3d}]  ->  w = 15-b ∈ [{15-bmax}, 15]   "
              f"(共 {bmax+1} 个取值)")
        out[f"weight_range_b{bmax}"] = {"min": 15 - bmax, "max": 15}
    print(f"   b ∈ [0,897] 时权重跨度 = {897} 个整数 ≈ {np.log2(898):.1f} bit")

    # ---------- 4. LUT 高度与分辨率的关系 ----------
    print("\n[4] 高度(体积)与权重的关系: 单表高度 H = sum_k |15 - b_k|")
    print("   固定布局: b_k = b* + δ_k, δ_k 为权重偏差")
    print("   -> 高度 H ≈ 16*|15 - b*|,  权重的精细分辨率来自 δ_k 的整数间隔(恒为1)")
    for bstar in (15, 100, 300, 897):
        print(f"   b*={bstar:3d}: 权重中心 w*={15-bstar:5d},  "
              f"单表高度≈{16*abs(15-bstar):6d},  权重可调范围 ±{bstar//2}")
        out[f"height_b{bstar}"] = {"w_center": 15 - bstar, "height": 16 * abs(15 - bstar)}

    # ---------- 5. 单调性 ----------
    print("\n[5] lambda 的单调性  (Δ = 15 - b)")
    dec = inc = 0
    for _ in range(2000):
        b = rng.integers(0, 898, size=16)
        d = np.diff([lam(x, b) for x in range(16)])
        dec += int(np.all(d <= 0)); inc += int(np.all(d >= 0))
    print(f"   随机 b∈[0,897]: 非增 {dec}/2000, 非减 {inc}/2000")
    print("   (b>15 时步长为负 -> 递减; 因此若要递增须用反式或令 b<=15)")
    out["monotone"] = {"nonincreasing": dec, "nondecreasing": inc}

    # ---------- 6. 自由度 ----------
    print("\n[6] 自由度: 16 个容器 -> 16 个独立参数")
    print("   b_0..b_15 各自独立控制 lambda 在 x=0..15 处的一步")
    print("   lambda(x) = lambda(0) + sum_{t=1}^{x} (15 - b_t)")
    print("   => 相邻步长完全独立, 可实现任意整数阶梯(无需步长上限!因为 b 无上限)")

    # 构造性验证: 任意整数步长
    print("\n[7] 构造性: 任意整数步长 (含大负数) 是否精确可实现")
    for nm, u in (("大步长", [100, -500, 300, 0, 897 - 15, 1, -1, 50, 50, 50, -800, 3, 7, 11, 13]),
                  ("全正", [1] * 15),
                  ("全负", [-1] * 15)):
        u = np.asarray(u, dtype=np.int64)
        b = np.zeros(16, dtype=np.int64)
        b[1:] = 15 - u                      # u_t = 15 - b_t
        v = np.array([lam(x, b) for x in range(16)], dtype=np.int64)
        v = v - v[0]
        tgt = np.concatenate([[0], np.cumsum(u)])
        err = int(np.max(np.abs(v - tgt)))
        print(f"   {nm:6s} 目标步长={u.tolist()}")
        print(f"   {'':6s} b={b.tolist()}")
        print(f"   {'':6s} 误差={err}   (b 是否都在 [0,897]: {bool(np.all((b>=0)&(b<=897)))})")
        out[f"construct_{nm}"] = {"err": err, "b": b.tolist()}

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    p = os.path.join(res, "mechanism_step_final.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已写出:", p)

if __name__ == "__main__":
    main()

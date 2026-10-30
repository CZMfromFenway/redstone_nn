#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
两种截断的步长公式并列验证
============================
比较器截断: a_k(x) = 14 + k - x
红石粉截断: a_k(x) = min(15, 14 + k - x)
判据:       c_k = a_k  if a_k >= 15  else  b_k
"""
import numpy as np, json, os

def a_comp(k, x):
    return 14 + k - x

def a_dust(k, x):
    return min(15, 14 + k - x)

def c_of(b, a):
    return a if a >= 15 else int(b)

def S(x, b, mode):
    f = a_comp if mode == "comp" else a_dust
    return sum(c_of(b[k], f(k, x)) for k in range(16))

def main():
    out = {}
    rng = np.random.default_rng(7)
    print("=" * 92)
    print("两种截断的步长公式")
    print("=" * 92)

    # 自由区/死区结构
    print("\n[结构] 每个 x 的自由区 (a_k(x) < 15, 即 c_k = b_k)")
    for mode in ("comp", "dust"):
        f = a_comp if mode == "comp" else a_dust
        row = []
        for x in range(16):
            ks = [k for k in range(16) if f(k, x) < 15]
            row.append(len(ks))
        print(f"   {mode:5s}: 自由区容器个数 x=0..15 -> {row}")

    # 步长公式
    print("\n[步长] 候选公式对照 (各 300000 样本)")
    cands = {
        "comp: (29-x) - b_{x+1}": ("comp", lambda b, x: (29 - x) - int(b[x + 1])),
        "dust: 15 - b_{x+1}":     ("dust", lambda b, x: 15 - int(b[x + 1])),
    }
    for nm, (mode, f) in cands.items():
        bad = 0; ex = None
        for _ in range(20000):
            b = rng.integers(0, 898, size=16)
            for x in range(15):
                d = S(x + 1, b, mode) - S(x, b, mode)      # = -Δλ (顺式)
                if -d != f(b, x):                            # Δλ = -ΔS
                    bad += 1
                    if ex is None:
                        ex = (x, int(b[x + 1]), int(d), f(b, x))
        print(f"   {nm:26s} 不匹配 = {bad}/300000  例(x,b,dS,预测): {ex}")
        out[nm] = int(bad)

    # 权重可达范围
    print("\n[权重可达范围]  b ∈ [0,897]")
    for mode in ("comp", "dust"):
        got = set()
        for _ in range(20000):
            b = rng.integers(0, 898, size=16)
            for x in range(15):
                got.add(int(S(x, b, mode) - S(x + 1, b, mode)))   # Δλ 顺式
        print(f"   {mode:5s}: Δλ 取值个数={len(got):5d}  范围=[{min(got)}, {max(got)}]")
        out[f"range_{mode}"] = {"n": len(got), "min": min(got), "max": max(got)}

    # 用户例子: 红石粉截断
    print("\n[用户例子] 红石粉截断, b_1=67")
    for mode in ("comp", "dust"):
        b = np.zeros(16, dtype=np.int64); b[1] = 67
        print(f"   {mode:5s}: ΔS(0)={S(1,b,mode)-S(0,b,mode):5d}   "
              f"Δλ顺式={S(0,b,mode)-S(1,b,mode):5d}   Δλ反式={S(1,b,mode)-S(0,b,mode):5d}")

    # 体积模型
    print("\n[体积模型] V ≈ 4n·h·L, L≈40")
    print(f"   {'n':>3} {'h':>4} | {'LUT体积':>10} | {'DT(d=3)':>10} | {'MLP FC':>10} | {'MLP+tanh':>10}")
    for n, h in ((4, 15), (4, 48), (4, 64), (15, 15)):
        lut = 4 * n * h * 40
        dt = 8 * n * h * 3
        fc = 444 * n * h
        fct = fc + 10000 * h
        print(f"   {n:3d} {h:4d} | {lut:10,} | {dt:10,} | {fc:10,} | {fct:10,}")
        out[f"vol_{n}_{h}"] = {"lut": lut, "dt": dt, "mlp_fc": fc, "mlp_fc_tanh": fct}

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    p = os.path.join(res, "two_truncations.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已写出:", p)

if __name__ == "__main__":
    main()

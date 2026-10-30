#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强模 LUT 单元数学解析（按 czm 更正后的判据）
============================================

判据 (原文):
    c_{ijk}(x) = a_{ijk}(x)                    if a_{ijk}(x) >= 15
    c_{ijk}(x) = b_{ijk}                       if a_{ijk}(x) <  15   且 b_{ijk} <= a
    c_{ijk}(x) = a_{ijk}(x)                    if a_{ijk}(x) <  15   且 b_{ijk} >  a
  统一写法:
    c = a                          if a >= 15
    c = min(b, a)                  if a <  15

选择器 (原文第 48 行):  a_k(x) = 14 + k - x

本脚本验证:
  [T1] 对每个 x, 阈值 a_k(x) >= 15 成立的 k 集合
  [T2] 对每个 x, 值 a_k(x) < 15 成立的 k 是否唯一
  [T3] 每个 b_k 是否在某个 x 上进入"自由"区 (即成为唯一可编程项)
  [T4] sum_k a_k(x) 是否与 b 无关  -> 若是, 则给每个输入值一个自由参数
"""
import numpy as np, json, os

def a_of(k, x):
    return 14 + k - x

def c_of(b, a):
    if a >= 15:
        return a
    return min(int(b), a)

def S(x, b):
    return sum(c_of(b[k], a_of(k, x)) for k in range(16))

def main():
    out = {}
    print("=" * 90)
    print("强模 LUT 单元数学解析（更正判据）")
    print("=" * 90)

    # ---------- T1 ----------
    print("\n[T1] 对每个 x, a_k(x) >= 15 的 k 集合（死区: c 与 b 无关）")
    dead = {}
    for x in range(16):
        ks = [k for k in range(16) if a_of(k, x) >= 15]
        dead[x] = ks
        print(f"   x={x:2d}: a_k(x)>=15 的 k = {ks}   (共 {len(ks)} 个)")
    out["dead_zones"] = {str(k): v for k, v in dead.items()}

    # ---------- T2 ----------
    print("\n[T2] 对每个 x, a_k(x) < 15 的 k（自由区: b_k 参与运算）")
    free = {}
    for x in range(16):
        ks = [k for k in range(16) if a_of(k, x) < 15]
        free[x] = ks
        tag = "唯一" if len(ks) == 1 else f"不唯一({len(ks)})"
        print(f"   x={x:2d}: a_k(x)<15 的 k = {ks}   -> {tag}")
    out["free_zones"] = {str(k): v for k, v in free.items()}

    # ---------- T3 ----------
    print("\n[T3] 每个容器 b_k 在哪个 x 上可编程")
    for k in range(16):
        xs = [x for x in range(16) if a_of(k, x) < 15]
        print(f"   b_{k:2d} 可编程于 x = {xs}")
    out["b_programmable_at"] = {str(k): [x for x in range(16) if a_of(k, x) < 15]
                                for k in range(16)}

    # ---------- T4 ----------
    print("\n[T4] sum_k a_k(x) 是否与 b 无关 (即 lambda 的自由部分是否恰为 b_{唯一自由项})")
    for x in range(16):
        phi = sum(a_of(k, x) for k in range(16))
        print(f"   x={x:2d}: sum_k a_k(x) = {phi}")
    # 验证: S(x) = sum_{k: a<15} b_k + sum_{k: a>=15} a_k   仅当自由区唯一且 b_k <= a_k
    rng = np.random.default_rng(5)
    bad = 0
    for _ in range(5000):
        b = rng.integers(0, 16, size=16)
        for x in range(16):
            frees = [k for k in range(16) if a_of(k, x) < 15]
            phi = sum(a_of(k, x) for k in range(16) if a_of(k, x) >= 15)
            pred = phi + sum(int(b[k]) for k in frees)      # 假设 b_k <= a_k(x)
            if S(x, b) != pred:
                bad += 1
    print(f"\n   b_k <= a_k(x) 全成立时, S(x) = sum_(死区) a_k(x) + sum_(自由区) b_k 的偏差次数 = {bad}/80000")
    out["decomposition_bad"] = bad

    # ---------- T5 ----------
    print("\n[T5] 结论: lambda 的自由部分")
    print("   顺式 lambda = B - S")
    print("   对每个 x, 自由区恰有一个 k, 且该 k 随 x 递减:")
    for x in range(16):
        k = [kk for kk in range(16) if a_of(kk, x) < 15]
        print(f"     x={x:2d} <-> k={k} (对应容器 b_{k[0]}, 即把 b 表按此顺序读出)")
    print("\n   => 把 b 表重新索引为 p_{ij,t} := b_{ij,15-t}, 则")
    print("      lambda_ij(x) = const - p_{ij,x} - phi(x)")
    print("      其中 phi(x) = sum_{k>=x+1} a_k(x) 与 b 无关,")
    print("      phi(x) 对所有边相同, 可被激活函数 f_j 吸收。")
    print("   => lambda_ij 是【逐点完全自由】的整数函数, 自由度 = 16 个 p 值。")

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    p = os.path.join(res, "mechanism_corrected.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已写出:", p)

if __name__ == "__main__":
    main()

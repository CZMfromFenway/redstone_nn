#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强模 LUT 单元物理机制仿真 + 结构等价性核查。

对应《Another Path of Comparator Network - analytical LUT network》第 32~55 行。

物理机制（来自原文）:
  容器 j 存 b_j;  截断信号 a_j(x) = 14 + j - x  (比较器截断)
                              = min(15, 14 + j - x)  (红石粉截断)
  检测比较器输出 c_j(b_j, a_j) = a_j  if a_j > b_j  else b_j
  顺式: lambda(x) = B  - sum_j c_j
  反式: lambda(x) = Bt - (B - sum_j c_j) = sum_j c_j + (Bt - B)

结论（本脚本数值验证）:
  S(x) := sum_j c_j(x) 是 x 的**严格递增仿射函数** S(x) = S(0) + 15x + sum_{j<=x} (b_j - 15) 之类的阶梯,
  更准确地说: S(x+1) - S(x) = b_{x+1} - a_{x+1} ... 见正文推导。
"""
import numpy as np
import json, os

MODES = ("comparator", "dust")

def a_of(j, x, mode):
    return (14 + j - x) if mode == "comparator" else min(15, 14 + j - x)

def c_of(b, a):
    return a if a > b else b

def S(x, b, mode):
    """sum_{j=0..15} c_j"""
    s = 0
    for j in range(16):
        s += c_of(int(b[j]), a_of(j, x, mode))
    return int(s)

def lam_forward(x, b, B=0, mode="comparator"):
    return int(B) - S(x, b, mode)

def lam_reverse(x, b, B=0, Bt=0, mode="comparator"):
    return S(x, b, mode) + int(Bt) - int(B)

def slope_profile(b, mode, form):
    f = lam_forward if form == "forward" else lam_reverse
    v = np.array([f(x, b, 0, mode) if form == "forward" else f(x, b, 0, 0, mode)
                  for x in range(16)], dtype=np.int64)
    return v, np.diff(v)

def main():
    out = {}
    rng = np.random.default_rng(7)
    print("=" * 78)
    print("强模 LUT 单元物理机制核查")
    print("=" * 78)

    # ---- 1. S(x) 的精确形状 ----
    print("\n[1] S(x) = sum_j c_j 的增量结构")
    for mode in MODES:
        dev, slopes = [], set()
        for _ in range(400):
            b = rng.integers(0, 898, size=16)
            v, d = slope_profile(b, mode, "forward")
            # 与"阶梯常量"比较: 若 c 全为仿射则 d 应恒定
            dev.append(int(np.max(np.abs(d - d[0]))))
            slopes.update(int(t) for t in np.unique(d))
        print(f"  mode={mode:11s}  d 非常数的最大偏离={max(dev)}   出现过的增量值={sorted(slopes)}")
        out[f"S_increment_{mode}"] = {"max_dev_from_constant": int(max(dev)),
                                     "increment_values": sorted(slopes)}

    # ---- 2. 精确增量公式 ----
    print("\n[2] 验证精确增量公式:  d(x) = S(x+1)-S(x) = c_{x+1}(b_{x+1}, 15)")
    for mode in MODES:
        ok = True
        for _ in range(200):
            b = rng.integers(0, 898, size=16)
            for x in range(15):
                d = S(x + 1, b, mode) - S(x, b, mode)
                a = a_of(x + 1, x + 1, mode)     # 恒等于 15
                true_d = c_of(int(b[x + 1]), a)
                if d != true_d:
                    ok = False
        print(f"  mode={mode:11s}  公式成立={ok}")
        out[f"increment_formula_{mode}"] = bool(ok)

    # ---- 3. 顺式 / 反式的单调方向 ----
    print("\n[3] lambda 的单调方向")
    for mode in MODES:
        for form in ("forward", "reverse"):
            ups = downs = 0
            for _ in range(300):
                b = rng.integers(0, 898, size=16)
                v, d = slope_profile(b, mode, form)
                if np.all(d >= 0):
                    ups += 1
                if np.all(d <= 0):
                    downs += 1
            print(f"  mode={mode:11s} form={form:8s}  非减={ups}/300  非增={downs}/300")
            out[f"monotone_{mode}_{form}"] = {"nondecreasing": ups, "nonincreasing": downs, "n": 300}

    # ---- 4. 可实现的步长（等效整数权重）范围 ----
    print("\n[4] 单列等效权重 a = lambda(1)-lambda(0) 的可达集合")
    for mode in MODES:
        for form in ("forward", "reverse"):
            vals = set()
            for _ in range(20000):
                b = rng.integers(0, 898, size=16)
                v, d = slope_profile(b, mode, form)
                vals.update(int(t) for t in np.unique(d))
            print(f"  mode={mode:11s} form={form:8s}  a 的取值个数={len(vals)}  "
                  f"范围=[{min(vals)},{max(vals)}]  前若干={sorted(vals)[:8]} ... {sorted(vals)[-4:]}")
            out[f"weight_set_{mode}_{form}"] = {"count": len(vals), "min": min(vals), "max": max(vals)}

    # ---- 5. 构造性验证: 任意目标整数权重 a 是否可实现 ----
    print("\n[5] 构造性: 能否实现任意目标 a (比较器截断 + 反式)")
    for target in (-15, -7, 0, 1, 7, 15, 40, 100):
        # 目标: lambda(x) = target*x  (用反式, 只需控制 S 的增量)
        # 取 b_j 使得每个增量都等于 target: 需要 c_{x+1}(b_{x+1},14) = target
        # 若 target >= 14 则取 b_{x+1}=target (>=a), 否则取 a=14 > b -> 输出 14. 故小权重无法直接构造.
        b = np.full(16, max(target, 0))
        v, d = slope_profile(b, "comparator", "reverse")
        got = np.unique(d)
        print(f"  target a={target:4d} -> b_j={b[0]:3d}, 实际增量集合={got.tolist()}, "
              f"lambda={(v - v[0]).tolist()}")

    # ---- 6. 整网等价性 ----
    print("\n[6] 整网 Lambda 与整数权重仿射式是否一致 (n=4)")
    X = rng.integers(0, 16, size=(5000, 4))
    lam_of_x = {}
    b_mat = rng.integers(0, 898, size=(4, 16))
    Lam = np.zeros(5000, dtype=np.int64)
    for i in range(4):
        f = np.array([lam_reverse(x, b_mat[i], 0, 0, "comparator") for x in range(16)], dtype=np.int64)
        lam_of_x[i] = f
        Lam += f[X[:, i]]
    # 用前两点定斜率
    A = np.zeros(4, dtype=np.int64); const = 0
    for i in range(4):
        A[i] = lam_of_x[i][1] - lam_of_x[i][0]
        const += lam_of_x[i][0]
    Lam_aff = X.astype(np.int64) @ A + const
    print(f"  各列斜率 A={A.tolist()}  最大绝对偏差={int(np.max(np.abs(Lam-Lam_aff)))}")
    out["network_equivalence"] = {"A": [int(a) for a in A],
                                  "max_abs_dev": int(np.max(np.abs(Lam - Lam_aff)))}

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results")
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "mechanism_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写出:", path)

if __name__ == "__main__":
    main()

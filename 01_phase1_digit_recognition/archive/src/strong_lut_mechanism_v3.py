#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强模 LUT 单元机制核查 v3 —— 逐字翻译原文条件, 不先验假设形状。

原文 (Another Path of Comparator Network, 第 34~48 行):
  红石比较器存在这样一个性质: 隔一方块检测容器时, 可以被另一信号截断。
  截断信号必须大于等于 15 才有截断的意义。
  当截断发生时, 检测容器的比较器输出的信号等于截断信号, 否则等于容器信号。
  假设检测 b_j 的比较器受到 a_j 的截断, 输出为 c_j。
  lambda(x) = B - sum_j c_j(b_j, a_j(x))
  a_j = 14 + j - x                (比较器截断)
  a_j = min{15, 14 + j - x}       (红石粉截断)
"""
import numpy as np
import json, os

def a_of(j, x, mode):
    return (14 + j - x) if mode == "comparator" else min(15, 14 + j - x)

def c_cut(b, a):
    """截断发生时输出 a, 否则输出 b。  截断 <=> 容器信号超过截断信号。"""
    return a if a < int(b) else int(b)

def c_nocut(b, a):
    """截断发生时输出 a, 否则输出 b。  截断 <=> 截断信号不超过容器信号。"""
    return a if a <= int(b) else int(b)

C_FORMULA = {"cut": c_cut, "nocut": c_nocut}

def S(x, b, mode, key):
    f = C_FORMULA[key]
    return sum(f(b[j], a_of(j, x, mode)) for j in range(16))

def show(name, b, key, mode="comparator"):
    v = np.array([S(x, b, key, mode) for x in range(16)], dtype=np.int64)
    d = np.diff(v)
    print(f"  {name:22s} S = {v.tolist()}")
    print(f"  {'':22s} dS = {d.tolist()}")

def main():
    out = {}
    rng = np.random.default_rng(11)
    print("=" * 90)
    print("强模 LUT 机制核查 v3 (逐字翻译 + 形状可视化)")
    print("=" * 90)

    # 三个典型 b 向量
    cases = {
        "b 全 0":        np.zeros(16, dtype=np.int64),
        "b 全 15":       np.full(16, 15, dtype=np.int64),
        "b 全 100":      np.full(16, 100, dtype=np.int64),
        "b_j = 16j":     np.arange(16, dtype=np.int64) * 16,
        "b_j = 100-6j":  np.maximum(100 - 6 * np.arange(16), 0),
        "b 随机(0..897)": rng.integers(0, 898, size=16),
    }

    for key in ("cut", "nocut"):
        print(f"\n{'#'*90}\n### 截断条件解释 = {key}\n{'#'*90}")
        for name, b in cases.items():
            print(f"\n[{name}] b = {b.tolist()}")
            for mode in ("comparator", "dust"):
                v = np.array([S(x, b, mode, key) for x in range(16)], dtype=np.int64)
                d = np.diff(v)
                print(f"  mode={mode:11s} S={v.tolist()}")
                print(f"  {'':16s} dS={d.tolist()}")
            out[f"{key}_{name}"] = {"b": b.tolist()}

    # 在正确的解释下统计 lambda 的形状
    print(f"\n{'#'*90}\n### 在两种解释下 lambda 的单调性与步长\n{'#'*90}")
    for key in ("cut", "nocut"):
        for mode in ("comparator", "dust"):
            for form in ("forward", "reverse"):
                incs = set(); nd = ni = 0; N = 400
                for _ in range(N):
                    b = rng.integers(0, 898, size=16)
                    s = np.array([S(x, b, mode, key) for x in range(16)], dtype=np.int64)
                    lam = (-s) if form == "forward" else s
                    d = np.diff(lam)
                    incs.update(int(t) for t in np.unique(d))
                    nd += int(np.all(d >= 0)); ni += int(np.all(d <= 0))
                print(f"  {key:6s} mode={mode:11s} form={form:8s} 非减={nd}/{N} 非增={ni}/{N} "
                      f"步长范围=[{min(incs)},{max(incs)}] 个数={len(incs)}")
                out[f"shape_{key}_{mode}_{form}"] = {"nondec": nd, "noninc": ni, "n": N,
                                                     "step_min": min(incs), "step_max": max(incs),
                                                     "n_steps": len(incs)}

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results")
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "mechanism_check_v3.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写出:", path)

if __name__ == "__main__":
    main()

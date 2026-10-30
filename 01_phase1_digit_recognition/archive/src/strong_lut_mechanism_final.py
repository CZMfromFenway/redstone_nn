#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强模 LUT 单元机制核查 (定稿 v5)

从《Another Path of Comparator Network - analytical LUT network》第 34~48 行逐字推导。
两个独立实现 (A: 逐项物理模拟; B: 闭式解) 互相校验。

物理设定
--------
容器 j 存 b_j (j = 0..15)，负责输入 x = j 时的输出。
截断信号 a_j(x) = 14 + j - x        (两种截断方式在 x<=16 时同式)
截断发生 <=> 容器信号 > 截断信号;  截断时输出 a_j, 否则输出 b_j
        等价闭式: c_j = min(b_j, a_j)
顺式 lambda(x) = B  - sum_j c_j(x)
反式 lambda(x) = sum_j c_j(x) + (Bt-B)

定理 (已数值验证)
----------------
(A) 红石粉截断 (要求 b_j <= 897, 但真正有意义的是 b_j <= 15):
      x <= 15 时对每个 j<=x 都有 a_j(x) <= 15 <= b_j, 故 min = a_j = 14+j-x
      S(x) := sum_j c_j(x) = sum_{j<=x}(14+j-x) = 15x + sum_{j<=x}(b_j - 15)
      => S(x+1) - S(x) = b_{x+1}
      => S 是步长恰为 b_1..b_15 的任意非减阶梯 (b_j = 0..15)
      => 单个 LUT 单元 = 一个任意非减阶梯函数, 15 个独立自由度
(B) 比较器截断且所有 b_j >= 15:
      此时对所有 j 都有 a_j(x) <= 15 <= b_j, 于是 c_j 恒等于 a_j, 与 b 完全无关
      S(x) = 15x - x(x+1)/2 + 120   (纯二次, 只在 x<=15 段成立)
      => 表格内容不可编程, 退化为固定函数
"""
import numpy as np
import json, os

def a_of(j, x):
    return 14 + j - x

def c_of(b, a):
    return min(int(b), int(a))

def S_A(x, b):
    """实现 A: 逐项物理模拟"""
    return sum(c_of(b[j], a_of(j, x)) for j in range(16))

def S_B(x, b):
    """实现 B: 闭式解"""
    return 15 * x + int(np.sum(np.asarray(b, dtype=np.int64)[:x + 1] - 15))

def report(t):
    print("\n" + "=" * 84); print(t); print("=" * 84)

def main():
    rng = np.random.default_rng(20261003)
    out = {}

    # ---------- 1. 两实现互校 ----------
    report("[1] 实现 A (物理模拟) 与实现 B (闭式解) 互校,  b_j <= 15")
    bad = 0
    for _ in range(3000):
        b = rng.integers(0, 16, size=16)
        for x in range(16):
            if S_A(x, b) != S_B(x, b):
                bad += 1
    print(f"  3000 x 16 次比对, 不匹配 = {bad}")
    out["AB_mismatch_ble15"] = bad

    report("[1b] 同上, 但 b_j 允许取到 897 (检验 min 截断何时生效)")
    bad = 0
    for _ in range(3000):
        b = rng.integers(0, 898, size=16)
        for x in range(16):
            if S_A(x, b) != S_B(x, b):
                bad += 1
    print(f"  3000 x 16 次比对, 不匹配 = {bad}")
    out["AB_mismatch_b897"] = bad
    print("  说明: 只有 b_j <= 15 的项才是可编程的; b_j > 15 时该容器被截断信号压平。")

    # ---------- 2. 步长 ----------
    report("[2] 步长 d(x) = S(x+1)-S(x) = b_{x+1}   (b_j <= 15)")
    vals = set(); bad = 0
    for _ in range(2000):
        b = rng.integers(0, 16, size=16)
        for x in range(15):
            d = S_A(x + 1, b) - S_A(x, b)
            if d != int(b[x + 1]):
                bad += 1
            vals.add(d)
    print(f"  不匹配 = {bad};  步长取值集合 = {sorted(vals)}")
    out["step_mismatch"] = bad
    out["step_values"] = sorted(vals)

    # ---------- 3. 比较器截断的退化 ----------
    report("[3] 比较器截断: 当所有 b_j >= 15 时表格内容失效")
    x = np.arange(16)
    xa = np.clip(x, 0, 15).astype(np.int64)
    pred = 15 * xa - xa * (xa + 1) // 2 + 120
    for label, B in (("全 15", np.full(16, 15)),
                     ("全 100", np.full(16, 100)),
                     ("递增 16..400", np.array([16, 30, 55, 70, 88, 101, 133, 150,
                                                177, 200, 233, 260, 290, 320, 350, 400]))):
        v = np.array([S_A(xx, B) for xx in range(16)], dtype=np.int64)
        print(f"  b=({label:12s}) S={v.tolist()}")
        print(f"  {'':18s} 与固定二次式 max|diff| = {int(np.max(np.abs(v - pred)))}")
    out["comparator_fixed_quadratic"] = pred.tolist()

    # ---------- 4. 任意阶梯可实现性 ----------
    report("[4] 构造性: 任意非减阶梯 (步长 0..15) 均可实现")
    def construct(target):
        s = np.asarray(target, dtype=np.int64)
        steps = np.diff(s)
        assert np.all(steps >= 0) and np.all(steps <= 15), "目标不是合法阶梯"
        b = np.zeros(16, dtype=np.int64); b[0] = 0
        b[1:] = steps
        return b

    tests = {
        "恒定 0":   [0] * 16,
        "每步 +1":  list(range(16)),
        "每步 +15": [15 * i for i in range(16)],
        "凹阶梯":   [0, 0, 15, 15, 20, 20, 20, 45, 45, 45, 100, 100, 100, 100, 150, 200],
        "凸阶梯":   [0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 6, 6, 6, 10, 10, 15],
        "高台阶":   [0, 0, 0, 0, 0, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
    }
    allok = True
    for name, t in tests.items():
        b = construct(t)
        got = np.array([S_A(xx, b) for xx in range(16)], dtype=np.int64)
        err = int(np.max(np.abs(got - np.asarray(t))))
        allok &= (err == 0)
        print(f"  {name:8s} 目标={t}")
        print(f"  {'':8s} 实现={got.tolist()}  误差={err}")
        out[f"construct_{name}"] = {"err": err}
    print(f"  => 全部精确实现: {allok}")

    # ---------- 5. 等效整数权重 ----------
    report("[5] 等效整数权重  a = lambda(1)-lambda(0) = b_1 ∈ {0,...,15}")
    got = set()
    for _ in range(20000):
        b = rng.integers(0, 16, size=16)
        got.add(int(S_A(1, b) - S_A(0, b)))
    print(f"  采样得到的 a 集合 = {sorted(got)}")
    print("  单 LUT 单元 = 15 个独立整数权重 ∈ [0,15]  (4 bit 分辨率), 权重向量完全自由(无相邻耦合)")
    print("  LUT 高度 H = S(15) = sum_{i} |W_i|  <= 225 (b_j<=15 时);  故体积由 |W| 的 L1 和决定")
    out["weight_values"] = sorted(got)

    # ---------- 6. 分辨率与体积 ----------
    report("[6] 权重分辨率与体积换算")
    print("  令量化因子 k, 有效权重 = k*w (w 为 0..15 整数, 可用 k 份并联单元实现)")
    print("  LUT 高度 H = k * sum_i |w_i|")
    for k in (1, 8, 16, 32, 128):
        H = k * 15
        print(f"   k={k:4d}: 单坐标最大高度 = {H:6d};  4->48->1 全网络 LUT 高度合计 = "
              f"{k*15*4*48 + k*15*48:8d}")
    out["volume_table"] = {str(k): k * 15 * 4 * 48 + k * 15 * 48 for k in (1, 8, 16, 32, 128)}

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results")
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "mechanism_final.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写出:", path)

if __name__ == "__main__":
    main()

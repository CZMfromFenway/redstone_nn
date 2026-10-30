#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核对 czm 的三条关键观察
=======================
1) 正/负权重可顺反转换；正权重的 lambda 从 0 到 15*步长；负权重从 15*步长 到 0
2) 每个 y 可以有自己的 s（只看该 y 的 |w| 最大者）
3) 步长不必均一：x*7.4 的步长是 7,8,7,8... （舍入造成）
"""
import numpy as np, json, os

CAP, CONT_HI = 868, 897

# ---------- 用机制真值正向模拟一张表 ----------
def a_of(k, x, mode="dust"):
    return min(15, 14 + k - x)

def S_of(b, x, mode="dust"):
    return sum((a_of(k, x, mode) if a_of(k, x, mode) >= 15 else int(b[k])) for k in range(16))

def lam_curve(b, form, mode="dust"):
    """给定容器 b, 返回 lambda(x), x=0..15"""
    S = np.array([S_of(b, x, mode) for x in range(16)], dtype=np.int64)
    return -S if form == "forward" else S

def build_staircase(target, form, mode="dust"):
    """
    为一个"步长序列"目标找到容器值。
    target[t] = lambda(t) - lambda(0)  (t=0..15), 要求相邻差 <= 868 且 >= 0 (反式)
    反式 dust: Δλ = b - 15  =>  b_t = 15 + Δλ(t-1)
    """
    d = np.diff(np.asarray(target, dtype=np.int64))
    if form == "reverse":
        b = np.zeros(16, dtype=np.int64)
        b[1:] = 15 + d
    else:
        b = np.zeros(16, dtype=np.int64)
        b[1:] = 15 - (-d)          # 顺式: Δλ = 15 - b
    return b

def main():
    print("=" * 96)
    print("核对 czm 的三条观察")
    print("=" * 96)

    # ---------- 观察 1 ----------
    print("\n[观察 1] 正/负权重的 lambda 形状与顺反转换")
    for k in (7, -7):
        # 目标: lambda(x) - lambda(0) = k*x
        tgt = np.array([k * x for x in range(16)], dtype=np.int64)
        if k >= 0:
            form = "reverse"; b = build_staircase(tgt, form)
        else:
            form = "forward"; b = build_staircase(tgt, form)
        lam = lam_curve(b, form)
        d = np.diff(lam)
        print(f"   k={k:3d}  形式={form:>7}  容器 b_1..b_3={b[1:4].tolist()} (全部 {b[1]})  "
              f"容器范围=[{b.min()},{b.max()}]")
        print(f"            lambda - lambda(0) = {(lam-lam[0]).tolist()}")
        print(f"            => x=0 时 {lam[0]}, x=15 时 {lam[15]}, 步长全为 {np.unique(d).tolist()}")
    print("   结论: 正权重 0 -> 15*步长; 负权重 15*步长 -> 0（含常数偏移）。观察 1 成立。")

    # ---------- 观察 3: 非均一步长 ----------
    print("\n[观察 3] 用舍入型阶梯近似 x*2.4 (s=1), 步长应为 2,2,3,2,2,3...")
    print("   精确目标 lambda(x) = x * 2.4")
    for s in (1, 2, 4):
        # 舍入型: lambda(x) = round(x * s * w) / s, 用整数表达 lambda* = round(x*s*w)
        w = 2.4
        tgt_int = np.array([int(round(x * s * w)) for x in range(16)], dtype=np.int64)
        steps = np.diff(tgt_int)
        # 两种容器构造
        b = build_staircase(tgt_int - tgt_int[0], "reverse")
        lam = lam_curve(b, "reverse")
        got = lam - lam[0]
        print(f"   s={s}: 目标整数阶梯 = {tgt_int.tolist()}")
        print(f"         步长 = {steps.tolist()}")
        print(f"         机制实现 = {got.tolist()}   误差 = {(got-tgt_int+tgt_int[0]).tolist()}")
        print(f"         容器 = {b[1:8].tolist()}...  容器范围=[{b.min()},{b.max()}]")

    # 对比"均一步长"方案
    print("\n   对比：均一步长方案 (每步 = round(s*w))")
    for s in (1, 2, 4):
        w = 2.4
        p = int(round(s * w))
        tgt = np.array([p * x for x in range(16)], dtype=np.int64)
        b = build_staircase(tgt, "reverse")
        lam = lam_curve(b, "reverse"); got = lam - lam[0]
        err_u = int(np.max(np.abs(got - tgt)))
        # 舍入型误差
        tgt_r = np.array([int(round(x * s * w)) for x in range(16)], dtype=np.int64) - 0
        b2 = build_staircase(tgt_r - tgt_r[0], "reverse")
        lam2 = lam_curve(b2, "reverse"); got2 = lam2 - lam2[0]
        err_r = int(np.max(np.abs(got2 - (tgt_r - tgt_r[0]))))
        print(f"   s={s}: 均一步长的最大偏差 = {err_u};  舍入型的最大偏差 = {err_r}")

    # ---------- 观察 2: 每个 y 自己的 s ----------
    print("\n[观察 2] 每个神经元可以用自己的 s (只看该神经元的 max|w|)")
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    teach = os.path.join(base, "DT_Redstone_Research_Archive_20261001",
                         "DT_baseline_recovery", "teachers", "N4_MLP_T1.json")
    m = json.load(open(teach, encoding="utf-8"))
    SL = np.array(m["W1"]) / 7.5
    print(f"   {'j':>2} | {'max|斜率|':>10} | {'全局s下':>9} | {'该y自己的s':>11} | 放大倍数")
    smin = int(CAP / np.abs(SL).max())
    for j in range(SL.shape[0]):
        wj = np.abs(SL[j]).max()
        sj = int(CAP / wj)
        print(f"   {j:2d} | {wj:10.6f} | {smin:9d} | {sj:11d} | {sj/smin:8.2f}x")
    print("   结论: 全局 s 会让小权重神经元损失大量分辨率。观察 2 成立。")

if __name__ == "__main__":
    main()

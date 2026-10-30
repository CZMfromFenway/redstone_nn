#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
用投掷器之和逼近截断正态。

目标
    X ∈ {0..255}，x_k = -3.5 + (k+0.5)·7/256，p_k ∝ exp(-x_k²/2)
    等价于 均值 127.5、标准差 36.57 的离散正态，支持被限制在 ±3.5σ

投掷器能做什么
    N ≤ 9 个已填格子，每个格子概率 1/N。
    同一数值占 k 个格子就得到概率 k/N。
    所以它能实现任何「概率为 1/N 整数倍」的分布。

核心结构约束（这是设计的关键）
    投掷器 i 支撑在 ±a_i 上，方差 ≤ a_i²。
    求和后  支持半径 = Σa_i，方差 = Σσ_i²。
    给定方差 σ² 与 M 个投掷器，支持半径最大是多少？
        Σa_i ≤ √(M · Σa_i²) = √M · σ       （柯西-施瓦茨）
    所以
        支持半径 ≈ ρ·√M·σ,   ρ = 单个投掷器的 支持/σ
    硬币（±a 等概率）ρ = 1；五点三角 (1,2,3,2,1) ρ = √3 ≈ 1.732。
    要覆盖 kσ 就需要 M ≥ (k/ρ)²。
"""
import itertools
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

NLEV = 256
XS = -3.5 + (np.arange(NLEV) + 0.5) * 7.0 / NLEV
P_T = np.exp(-0.5 * XS ** 2)
P_T /= P_T.sum()
SIG_T = float(np.sqrt((P_T * (np.arange(NLEV) - 127.5) ** 2).sum()))
print("=" * 96)
print("目标分布")
print("=" * 96)
print(f"  256 级，均值 127.5，标准差 {SIG_T:.3f}，支持 ±3.5σ = ±128")

# ---------------------------------------------------------------- 形状
# 五点形状：在 {-2,-1,0,1,2} 上的相对权重（用格子数表示，总数 <= 9）
SHAPES = {
    "硬币 ±a":        (np.array([-1, 1]), np.array([1, 1])),
    "三点 {-a,0,a}":  (np.array([-1, 0, 1]), np.array([1, 1, 1])),
    "五点 1,2,3,2,1": (np.array([-2, -1, 0, 1, 2]), np.array([1, 2, 3, 2, 1])),
    "五点 1,1,5,1,1": (np.array([-2, -1, 0, 1, 2]), np.array([1, 1, 5, 1, 1])),
    "五点 2,2,1,2,2": (np.array([-2, -1, 0, 1, 2]), np.array([2, 2, 1, 2, 2])),
    "八点 1,1,1,1,1,1,1,1":
        (np.array([-4, -3, -2, -1, 1, 2, 3, 4]), np.array([1, 1, 1, 1, 1, 1, 1, 1])),
}


def shape_stats(mult, cnt):
    p = cnt / cnt.sum()
    v = float((p * (mult.astype(float)) ** 2).sum())
    sup = float(np.abs(mult).max())
    return v, sup, cnt.sum()


print()
print("=" * 96)
print("单个投掷器的形状：支持/σ 比 ρ 越大，同样方差下能铺得越宽")
print("=" * 96)
print(f"  {'形状':22s} {'格子数':>6s} {'单位方差':>9s} {'支持半径':>9s} {'ρ=支持/σ':>9s}")
print("  " + "-" * 66)
for nm, (m, c) in SHAPES.items():
    v, sup, n = shape_stats(m, c)
    print(f"  {nm:22s} {n:6d} {v:9.4f} {sup:9.1f} {sup/np.sqrt(v):9.3f}")


def pmf_sum(steps, mult, cnt):
    """把若干同形状、不同缩放的投掷器求和，返回 (取值, 概率)。"""
    p1 = cnt / cnt.sum()
    pos = np.array([0])
    pr = np.array([1.0])
    for s in steps:
        v = (mult * s).astype(int)
        newpos = (pos[:, None] + v[None, :]).ravel()
        newpr = (pr[:, None] * p1[None, :]).ravel()
        order = np.argsort(newpos)
        newpos, newpr = newpos[order], newpr[order]
        up, inv = np.unique(newpos, return_inverse=True)
        pos = up
        pr = np.bincount(inv, weights=newpr)
    return pos, pr


def fit_score(steps, shape_name):
    mult, cnt = SHAPES[shape_name]
    pos, pr = pmf_sum(steps, mult, cnt)
    # X = 128 + S，截断到 [0,255]
    X = np.clip(128 + pos, 0, NLEV - 1)
    pk = np.zeros(NLEV)
    np.add.at(pk, X, pr)
    tv = 0.5 * np.abs(pk - P_T).sum()
    cdf = np.abs(np.cumsum(pk) - np.cumsum(P_T)).max()
    mu = float((pk * np.arange(NLEV)).sum())
    sd = float(np.sqrt((pk * (np.arange(NLEV) - mu) ** 2).sum()))
    return dict(tv=tv, cdf=cdf, mu=mu, sd=sd, support=int(np.abs(pos).max()),
                ndist=int(len(np.unique(X))), M=len(steps))


# ---------------------------------------------------------------- 搜索
print()
print("=" * 96)
print("搜索：给定投掷器个数 M，找最优的整数缩放与形状")
print("=" * 96)

rng = np.random.default_rng(0)
best_by_M = {}
for shape_name in SHAPES:
    mult, cnt = SHAPES[shape_name]
    v1, sup1, _ = shape_stats(mult, cnt)
    for M in range(2, 17):
        # 目标：总方差 = 36.57²，总支持半径约 128
        # Σ s_i² = σ²/v1 ；Σ s_i = 支持半径/sup1
        target_sq = SIG_T ** 2 / v1
        target_sum = 128.0 / sup1
        best = None
        for trial in range(4000):
            # 随机整数向量，均值贴近 target_sum/M，平方和贴近 target_sq
            base = max(1, int(round(target_sum / M)))
            s = rng.integers(max(1, base // 2), max(2, base * 2 + 2), size=M)
            ss = s.sum()
            sq = float((s.astype(float) ** 2).sum())
            if abs(sq - target_sq) / target_sq > 0.06:
                continue
            sc = fit_score(list(s), shape_name)
            sc["steps"] = sorted(int(v) for v in s)
            if best is None or sc["tv"] < best["tv"]:
                best = sc
        if best is not None:
            key = M
            if key not in best_by_M or best["tv"] < best_by_M[key]["tv"]:
                best[key] = dict(best, shape=shape_name)

print(f"  {'M':>3s} {'形状':22s} {'支持':>6s} {'σ':>7s} {'不同取值':>8s} "
      f"{'TV 距离':>9s} {'CDF 最大差':>10s}")
print("  " + "-" * 78)
for M in sorted(best_by_M):
    b = best_by_M[M]
    print(f"  {M:3d} {b['shape']:22s} {b['support']:6d} {b['sd']:7.2f} "
          f"{b['ndist']:8d} {b['tv']:9.4f} {b['cdf']:10.4f}")
    if M in (4, 6, 8, 10, 12, 14):
        print(f"      缩放 {b['steps']}")

print()
print("=" * 96)
print("结论")
print("=" * 96)
print("  支持半径 ≈ ρ√M σ。要覆盖 3.5σ：")
for nm, (m, c) in SHAPES.items():
    v, sup, _ = shape_stats(m, c)
    rho = sup / np.sqrt(v)
    need = (3.5 / rho) ** 2
    print(f"    {nm:22s} ρ={rho:5.3f}  →  需要 M ≥ {need:5.1f} 个投掷器")
print()
print("  硬币 ρ=1 最「省方差」，但只铺到 ±a；五点形状 ρ=√3，同样方差能铺 1.73 倍宽。")
print("  代价是形状更尖（0 处堆了 5/9 的格子），CLT 收敛更慢。")
print("DROPPER_SEARCH_DONE")

#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
投掷器之和逼近截断正态——修正版搜索。

关键：给定形状（单位方差 v1、单位支持 sup1）与个数 M，
    要凑出总方差 σ² 与支持半径 R，就要求
        Σ s_i = R/sup1 =: S,     Σ s_i² = σ²/v1 =: Q
    而整数向量存在的前提是
        Q >= S²/M          （柯西-施瓦茨，等号在 s_i 全相等时取到）
    即
        M >= (R/σ)² · (v1/sup1²) = (3.5/ρ)²
    这就是可达律，M 不足时不是拟合差，而是**根本凑不出来**。
"""
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
SIG = float(np.sqrt((P_T * (np.arange(NLEV) - 127.5) ** 2).sum()))
R_T = 3.5 * SIG
print("=" * 94)
print("目标：256 级离散截断正态")
print("=" * 94)
print(f"  标准差 σ = {SIG:.3f}，支持半径 R = 3.5σ = {R_T:.1f}")

SHAPES = {
    "硬币 ±a":        (np.array([-1, 1]), np.array([1, 1])),
    "三点 a,0,a":     (np.array([-1, 0, 1]), np.array([1, 1, 1])),
    "五点 1,2,3,2,1": (np.array([-2, -1, 0, 1, 2]), np.array([1, 2, 3, 2, 1])),
    "五点 1,1,5,1,1": (np.array([-2, -1, 0, 1, 2]), np.array([1, 1, 5, 1, 1])),
    "五点 2,2,1,2,2": (np.array([-2, -1, 0, 1, 2]), np.array([2, 2, 1, 2, 2])),
    "八点均匀":        (np.array([-4, -3, -2, -1, 1, 2, 3, 4]), np.ones(8, int)),
}


def stats(mult, cnt):
    p = cnt / cnt.sum()
    v = float((p * mult.astype(float) ** 2).sum())
    return v, float(np.abs(mult).max()), int(cnt.sum())


print()
print("=" * 94)
print("单个投掷器形状")
print("=" * 94)
print(f"  {'形状':20s} {'格子':>4s} {'单位方差':>8s} {'单位支持':>8s} {'ρ=支持/σ':>8s} "
      f"{'覆盖 3.5σ 所需 M':>16s}")
print("  " + "-" * 76)
for nm, (m, c) in SHAPES.items():
    v, s, n = stats(m, c)
    rho = s / np.sqrt(v)
    print(f"  {nm:20s} {n:4d} {v:8.4f} {s:8.1f} {rho:8.3f} {(3.5/rho)**2:16.1f}")


def find_steps(M, S, Q, rng, iters=40000):
    """找 M 个正整数，和 = S，平方和尽量接近 Q。不可行返回 None。"""
    if Q < S * S / M - 1e-9:
        return None
    if S < M:
        return None
    s = np.full(M, S // M, dtype=int)
    s[: S % M] += 1
    q = int((s * s).sum())
    best = (abs(q - Q), s.copy())
    for _ in range(iters):
        if best[0] == 0:
            break
        i, j = int(rng.integers(M)), int(rng.integers(M))
        if i == j or s[i] <= 1:
            continue
        s2 = s.copy()
        s2[i] -= 1
        s2[j] += 1
        q2 = int((s2 * s2).sum())
        if abs(q2 - Q) <= abs(q - Q):
            s, q = s2, q2
            best = (abs(q - Q), s.copy())
    return best[1]


def pmf_of(steps, mult, cnt):
    p1 = cnt / cnt.sum()
    pos = np.array([0])
    pr = np.array([1.0])
    for s in steps:
        v = (mult * int(s)).astype(int)
        np_ = (pos[:, None] + v[None, :]).ravel()
        npr = (pr[:, None] * p1[None, :]).ravel()
        up, inv = np.unique(np_, return_inverse=True)
        pos = up
        pr = np.bincount(inv, weights=npr)
    return pos, pr


def score(steps, shape):
    mult, cnt = SHAPES[shape]
    pos, pr = pmf_of(steps, mult, cnt)
    X = np.clip(128 + pos, 0, NLEV - 1)
    pk = np.zeros(NLEV)
    np.add.at(pk, X, pr)
    mu = float((pk * np.arange(NLEV)).sum())
    sd = float(np.sqrt((pk * (np.arange(NLEV) - mu) ** 2).sum()))
    return dict(tv=float(0.5 * np.abs(pk - P_T).sum()),
                cdf=float(np.abs(np.cumsum(pk) - np.cumsum(P_T)).max()),
                mu=mu, sd=sd, sup=int(np.abs(pos).max()),
                ndist=int((pk > 1e-12).sum()), M=len(steps))


rng = np.random.default_rng(0)
print()
print("=" * 94)
print("每个 M 下的最优设计")
print("=" * 94)
print(f"  {'M':>3s} {'形状':18s} {'缩放':28s} {'σ':>6s} {'支持':>5s} {'非零级数':>8s} "
      f"{'TV':>7s} {'CDF差':>7s}")
print("  " + "-" * 92)
rows = []
for shape in SHAPES:
    mult, cnt = SHAPES[shape]
    v1, sup1, _ = stats(mult, cnt)
    S = R_T / sup1
    Q = SIG ** 2 / v1
    for M in range(3, 15):
        st = find_steps(M, int(round(S)), int(round(Q)), rng, iters=6000)
        if st is None:
            continue
        sc = score(st, shape)
        sc["shape"] = shape
        sc["steps"] = sorted(int(v) for v in st)
        rows.append(sc)

rows.sort(key=lambda r: r["tv"])
seen = set()
for r in rows:
    if r["M"] in seen:
        continue
    seen.add(r["M"])
    st = r["steps"]
    txt = str(st) if len(st) <= 8 else str(st[:8])[:-1] + ", ...]"
    print(f"  {r['M']:3d} {r['shape']:18s} {txt:28s} {r['sd']:6.2f} {r['sup']:5d} "
          f"{r['ndist']:8d} {r['tv']:7.4f} {r['cdf']:7.4f}")

print()
print("=" * 94)
print("按 TV 距离排前六")
print("=" * 94)
print(f"  {'M':>3s} {'形状':18s} {'TV':>8s} {'CDF差':>8s} {'σ':>7s} {'支持':>5s} {'格子总数':>8s}")
print("  " + "-" * 66)
for r in rows[:6]:
    n = stats(*SHAPES[r["shape"]])[2]
    print(f"  {r['M']:3d} {r['shape']:18s} {r['tv']:8.4f} {r['cdf']:8.4f} "
          f"{r['sd']:7.2f} {r['sup']:5d} {n*r['M']:8d}")
    print(f"      缩放 {r['steps']}")
print()
print("  TV 距离 = 两分布总变差的一半，即「需要搬动的概率质量」。")
print("  0.02 意味着平均每个取值差 2% 的概率，肉眼与统计上都很难分辨。")
print("DROPPER2_DONE")

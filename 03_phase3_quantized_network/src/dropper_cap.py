#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
投掷器之和逼近截断正态，带上「单投掷器输出 ≤ 15」这个硬约束。

单投掷器输出是红石信号，只能 0..15。若用两点分布 {0, v}，则 v ≤ 15。

求和后的分布
    均值 Σp_i v_i，方差 Σp_i(1-p_i)v_i²，支持跨度 Σv_i
要匹配目标（σ=36.46，支持 ±128，即跨度 256）：
    Σ v_i = 256
    Σ p_i(1-p_i) v_i² = σ² = 1329
而 p(1-p) ≤ 1/4，所以
    Σ v_i² ≥ 4·1329 = 5316
但 v_i ≤ 15 限制了 Σv_i² 的上界。
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

print("=" * 92)
print("目标")
print("=" * 92)
print(f"  σ = {SIG:.3f}，支持 ±3.5σ = ±{3.5*SIG:.1f}，跨度 {7*SIG:.1f}")
print(f"  需要的 Σv_i = 跨度 = {7*SIG:.0f}")
print(f"  需要的 Σv_i² ≥ 4σ² = {4*SIG**2:.0f}")

print()
print("=" * 92)
print("v_i ≤ 15 时，Σv_i² 最多能到多少")
print("=" * 92)
print(f"  {'M':>4s} {'Σv_i':>6s} {'Σv_i² 最大':>11s} {'最大 σ':>8s} {'够不够 σ=36.5':>14s}")
print("  " + "-" * 50)
best_sig = 0.0
for M in range(16, 31):
    # Σv 固定 256，v_i ∈ [1,15]，M 个，最大化 Σv²：尽量多取 15，余数摊成 1
    S = 256
    k = (S - M) // 14                    # 15 的个数
    k = min(k, M)
    rem = S - 15 * k
    rest = M - k
    if rest <= 0:
        continue
    base = rem // rest
    extra = rem - base * rest
    v = np.array([15] * k + [base] * (rest - extra) + [base + 1] * extra)
    sq = float((v.astype(float) ** 2).sum())
    smax = np.sqrt(sq / 4)
    best_sig = max(best_sig, smax)
    if M <= 24 or M % 2 == 0:
        print(f"  {M:4d} {int(v.sum()):6d} {sq:11.0f} {smax:8.2f} "
              f"{'够' if smax >= SIG else '不够（差 ' + format(SIG-smax, '.1f') + '）':>14s}")

print()
print("=" * 92)
print("结论")
print("=" * 92)
print(f"  v_i ≤ 15 时，最大可达 σ ≈ {best_sig:.1f}，而目标是 {SIG:.1f}。")
print(f"  也就是说：单投掷器输出限制在 0..15 时，**凑不出目标分布的宽度**。")
print()
print("  为什么：要铺到 ±128 就得 Σv_i = 256；要用有限的 M 做到这一点，")
print("  v_i 就得接近 15。但方差只需要 σ²=1329，而 v_i 接近 15 的两点分布")
print("  方差是 v²/4 ≈ 56，M 个加起来需要 Σv² = 5316，需要 24 个以上的 15。")
print("  可是 24 个 15 加起来是 360，远超所需的 256 —— 于是 v_i 被迫变小，")
print("  方差随之不足。两个约束夹死了。")

print()
print("=" * 92)
print("出路：给每个投掷器的贡献乘一个权重")
print("=" * 92)
print("  若第 i 路贡献是 c_i · v_i（c_i 是容器给的强模权重，v_i 仍是 0..15），")
print("  则支持跨度 = Σc_i v_i，方差 = Σc_i² p(1-p) v_i²，")
print("  v_i ≤ 15 只限制单路的分辨率，不限制跨度。")
print()
print("  这时回到之前那套结果：")

SHAPES = {
    "硬币 ±a": (np.array([-1, 1]), np.array([1, 1])),
    "三点 a,0,a": (np.array([-1, 0, 1]), np.array([1, 1, 1])),
    "五点 1,2,3,2,1": (np.array([-2, -1, 0, 1, 2]), np.array([1, 2, 3, 2, 1])),
}


def pmf_of(steps, shape):
    mult, cnt = SHAPES[shape]
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
    X = np.clip(128 + pos, 0, NLEV - 1)
    pk = np.zeros(NLEV)
    np.add.at(pk, X, pr)
    mu = float((pk * np.arange(NLEV)).sum())
    sd = float(np.sqrt((pk * (np.arange(NLEV) - mu) ** 2).sum())
               ) if (pk > 0).all() else float(np.sqrt((pk * (np.arange(NLEV) - mu) ** 2).sum()))
    return pk, sd


CANDS = [
    ("14 个三点，缩放 1,1,1,1,2,3,7,9,12,14,17,17,20,23", "三点 a,0,a",
     [1, 1, 1, 1, 2, 3, 7, 9, 12, 14, 17, 17, 20, 23]),
    ("6 个五点，缩放 1,5,7,11,19,21", "五点 1,2,3,2,1", [1, 5, 7, 11, 19, 21]),
    ("5 个五点，缩放 3,11,12,18,20", "五点 1,2,3,2,1", [3, 11, 12, 18, 20]),
]
print(f"  {'设计':44s} {'σ':>7s} {'跨度':>6s} {'单路最大':>8s} {'TV':>8s}")
print("  " + "-" * 80)
for nm, sh, st in CANDS:
    pk, sd = pmf_of(st, sh)
    mult = SHAPES[sh][0]
    span = int(np.abs(mult).max() * max(st))
    tv = float(0.5 * np.abs(pk - P_T).sum())
    print(f"  {nm:44s} {sd:7.2f} {int(2*span):6d} {span:8d} {tv:8.4f}")
print()
print("  「单路最大」= 该路贡献的最大绝对值。若它 ≤ 15，就直接可做；")
print("  超过 15 的部分要靠容器权重 c_i 放大。")
print("CAP_DONE")

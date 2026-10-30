#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
投掷器只能输出三个连续整数 {0,1,2}，权重自由，累加可带偏置。

设计变量
    每个投掷器的格子分配 (a, b, c)：a 格取 0，b 格取 1，c 格取 2，a+b+c <= 9
    每路的容器权重 w_i >= 1
    X = clip(Σ w_i v_i, 0, 255)

均值要落在 127.5
    E[v]=(b+2c)/N。若取 a=c（关于 1 对称），则 E[v]=1，X 的均值就是 Σw_i。
    所以 Σw_i = 128 时均值正好 127.5~128，不需要额外偏置。

方差
    Var(v) 关于 (a,b,c) 的最大值：把质量压在两端的 (4,1,4) 给 0.889。
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
print(f"  σ = {SIG:.3f}，支持 ±3.5σ = ±{3.5*SIG:.1f}")

print()
print("=" * 92)
print("投掷器能给的分布：{0,1,2}，N ≤ 9 格")
print("=" * 92)
print(f"  {'(a,b,c)':>10s} {'N':>3s} {'E[v]':>6s} {'Var(v)':>8s} {'ρ=半跨度/σ':>11s} "
      f"{'能否用 w 配平均值':>16s}")
print("  " + "-" * 62)
BEST = None
for a in range(10):
    for b in range(10 - a):
        c = 9 - a - b
        N = a + b + c
        p = np.array([a, b, c]) / N
        ev = float((p * np.arange(3)).sum())
        var = float((p * (np.arange(3) - ev) ** 2).sum())
        sym = abs(a - c) == 0
        rho = 1.0 / np.sqrt(var) if var > 0 else 0
        if a + b + c == 9 and var > 0.3:
            print(f"  {str((a,b,c)):>10s} {N:3d} {ev:6.3f} {var:8.4f} {rho:11.3f} "
                  f"{'是（a=c）' if sym else '否':>16s}")
            if sym and (BEST is None or var > BEST[1]):
                BEST = ((a, b, c), var)

abc, VARV = BEST
print()
print(f"  → 方差最大且均值恰为 1 的是 {abc}，Var(v) = {VARV:.4f}")
print(f"    ρ = 1/√Var = {1/np.sqrt(VARV):.4f}（硬币是 1.000，所以三点略优）")

# ------------------------------------------------------------------ 配权重
print()
print("=" * 92)
print("配权重：Σw=128，Σw²·Var = σ²")
print("=" * 92)
need_sq = SIG ** 2 / VARV
print(f"  需要的 Σw² = σ²/Var = {SIG**2:.1f}/{VARV:.4f} = {need_sq:.1f}")
print(f"  柯西-施瓦茨：Σw² ≥ 128²/M  ⇒  M ≥ 16384/{need_sq:.1f} = {16384/need_sq:.2f}")
print()


def try_weights(M, S, Q, rng, iters=60000):
    if Q < S * S / M - 1e-9 or S < M:
        return None
    w = np.full(M, S // M, dtype=int)
    w[: S % M] += 1
    q = int((w * w).sum())
    best = (abs(q - Q), w.copy())
    for _ in range(iters):
        if best[0] <= 1:
            break
        i, j = int(rng.integers(M)), int(rng.integers(M))
        if i == j or w[i] <= 1:
            continue
        w2 = w.copy()
        w2[i] -= 1
        w2[j] += 1
        q2 = int((w2 * w2).sum())
        if abs(q2 - Q) <= abs(q - Q):
            w, q = w2, q2
            best = (abs(q - Q), w.copy())
    return best[1]


def pmf(w, abc):
    a, b, c = abc
    N = a + b + c
    p = np.array([a, b, c]) / N
    pos = np.array([0])
    pr = np.array([1.0])
    for wi in w:
        v = np.arange(3) * int(wi)
        np_ = (pos[:, None] + v[None, :]).ravel()
        npr = (pr[:, None] * p[None, :]).ravel()
        up, inv = np.unique(np_, return_inverse=True)
        pos = up
        pr = np.bincount(inv, weights=npr)
    X = np.clip(pos, 0, NLEV - 1)
    pk = np.zeros(NLEV)
    np.add.at(pk, X, pr)
    return pk, int(pos.max())


rng = np.random.default_rng(3)
print(f"  {'M':>3s} {'权重':38s} {'σ':>7s} {'支撑':>5s} {'非零级':>7s} {'TV':>8s} {'CDF差':>7s}")
print("  " + "-" * 84)
best_rows = []
for M in range(9, 17):
    w = try_weights(M, 128, int(round(need_sq)), rng)
    if w is None:
        continue
    pk, mx = pmf(w, abc)
    mu = float((pk * np.arange(NLEV)).sum())
    sd = float(np.sqrt((pk * (np.arange(NLEV) - mu) ** 2).sum()))
    tv = float(0.5 * np.abs(pk - P_T).sum())
    cdf = float(np.abs(np.cumsum(pk) - np.cumsum(P_T)).max())
    ws = sorted(int(v) for v in w)
    txt = str(ws) if M <= 12 else str(ws[:12])[:-1] + ", ...]"
    print(f"  {M:3d} {txt:38s} {sd:7.2f} {mx:5d} {int((pk>1e-12).sum()):7d} "
          f"{tv:8.4f} {cdf:7.4f}")
    best_rows.append((tv, M, ws, sd))

best_rows.sort()
print()
print("=" * 92)
print("最优")
print("=" * 92)
tv, M, ws, sd = best_rows[0]
print(f"  {M} 个投掷器，权重 {ws}")
print(f"  每个：9 格，(a,b,c) = {abc}，即 {{0,1,2}} 各占 {abc[0]}/{abc[1]}/{abc[2]} 格")
print(f"  σ = {sd:.3f}（目标 {SIG:.3f}）   TV = {tv:.4f}")
print(f"  Σw = {sum(ws)}，Σw² = {sum(v*v for v in ws)}，gcd = {np.gcd.reduce(ws)}")

print()
print("=" * 92)
print("5 位方案：只在 0..31 上做，低 3 位用均匀补")
print("=" * 92)
SIG_H = SIG / 8
print(f"  高位目标：0..31，σ_H = σ/8 = {SIG_H:.4f}")
print(f"  支持：±3.5σ_H = ±{3.5*SIG_H:.2f}，正好铺满 0..31")
need_sq_h = SIG_H ** 2 / VARV
print(f"  需要 Σw = 16（使 max=32），Σw² = {need_sq_h:.1f}")
print(f"  M ≥ 16²/{need_sq_h:.1f} = {256/need_sq_h:.2f}")
for M in range(9, 15):
    w = try_weights(M, 16, int(round(need_sq_h)), rng)
    if w is None:
        continue
    print(f"    M={M}: 权重 {sorted(int(v) for v in w)}"
          f"  Σw={sum(w)}  Σw²={int((w*w).sum())}")
print()
print("  低 3 位：三个 {0,1} 投掷器，权重 1,2,4")
print("  → 5 位方案总计需要 高位 M 个 + 3 个，比直接 8 位更多，不划算。")
print("THREE_VALUE_DONE")

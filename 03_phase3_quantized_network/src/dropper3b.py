#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
三值投掷器（{0,1,2}）逼近截断正态——用累积分布距离评判。

为什么不用 TV
    M 个三值投掷器之和落在由权重生成的稀疏格点上，点态概率会振荡。
    TV = ½Σ|p−q| 对这种振荡极其敏感，但它不代表「分布不像」。
    CDF 最大差才是分布近似的标准度量，也直接对应分位数误差。
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
ABC = (4, 1, 4)
PV = np.array(ABC) / sum(ABC)
VARV = float((PV * (np.arange(3) - 1.0) ** 2).sum())


def pmf(w, abc=ABC, scale=1, nlev=NLEV):
    p = np.array(abc) / sum(abc)
    pos = np.array([0])
    pr = np.array([1.0])
    for wi in w:
        v = np.arange(3) * int(wi)
        np_ = (pos[:, None] + v[None, :]).ravel()
        npr = (pr[:, None] * p[None, :]).ravel()
        up, inv = np.unique(np_, return_inverse=True)
        pos = up
        pr = np.bincount(inv, weights=npr)
    X = np.clip(pos, 0, nlev - 1)
    pk = np.zeros(nlev)
    np.add.at(pk, X, pr)
    return pk


def metrics(pk, ref):
    cdf = float(np.abs(np.cumsum(pk) - np.cumsum(ref)).max())
    tv = float(0.5 * np.abs(pk - ref).sum())
    mu = float((pk * np.arange(len(pk))).sum())
    sd = float(np.sqrt((pk * (np.arange(len(pk)) - mu) ** 2).sum()))
    return cdf, tv, mu, sd


def qerr(pk, ref, n):
    e = []
    for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
        a = int(np.searchsorted(np.cumsum(ref), q))
        b = int(np.searchsorted(np.cumsum(pk), q))
        e.append(b - a)
    return e


print("=" * 92)
print(f"三值投掷器：{{0,1,2}}，格数分配 {ABC}，Var(v)={VARV:.4f}")
print("=" * 92)
print(f"  需要 Σw = 128，Σw² = σ²/Var = {SIG**2/VARV:.0f}，M ≥ {16384/(SIG**2/VARV):.2f}")

# 手挑几组整数权重：和 128，平方和尽量接近 1496，gcd 1
SETS = {
    "11 路：7×12 + 4×11": [12] * 7 + [11] * 4,
    "13 路：混": [9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11],
    "14 路：混": [8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 11, 12],
    "16 路：全 8": [8] * 16,
}

print()
print("=" * 92)
print("候选（手挑，Σw=128）")
print("=" * 92)
print(f"  {'权重方案':26s} {'Σw':>4s} {'Σw²':>6s} {'σ':>7s} {'CDF 最大差':>10s} "
      f"{'TV':>7s} {'非零级':>6s} {'gcd':>4s}")
print("  " + "-" * 86)
best = None
for nm, w in SETS.items():
    if sum(w) != 128:
        continue
    pk = pmf(w)
    cdf, tv, mu, sd = metrics(pk, P_T)
    print(f"  {nm:26s} {sum(w):4d} {sum(v*v for v in w):6d} {sd:7.2f} {cdf:10.4f} "
          f"{tv:7.4f} {int((pk>1e-12).sum()):6d} {np.gcd.reduce(w):4d}")
    if best is None or cdf < best[0]:
        best = (cdf, nm, w, pk, sd)

# 让平方和更接近目标的改进：把 11 路那组调一调
print()
print("  Σw² 越接近 1496，σ 越准。11 路那组是 1492，最接近，M 也最小。")

cdf, nm, w, pk, sd = best
print()
print("=" * 92)
print(f"选：{nm}")
print("=" * 92)
print(f"  权重 {sorted(w)}")
print(f"  Σw={sum(w)}  Σw²={sum(v*v for v in w)}  σ={sd:.3f}（目标 {SIG:.3f}）")
print(f"  CDF 最大差 {cdf:.4f}   TV {metrics(pk, P_T)[1]:.4f}")
print(f"  分位误差（1,5,25,50,75,95,99 分位）: {qerr(pk, P_T, NLEV)}")

print()
print("=" * 92)
print("5 位方案：先做 0..31，低 3 位用三个硬币均匀补")
print("=" * 92)
SIG_H = SIG / 8
P_TH = np.zeros(32)
for h in range(32):
    P_TH[h] = P_T[h * 8:(h + 1) * 8].sum()
print(f"  高位目标：32 档，σ_H={SIG_H:.4f}")
print(f"  需要 Σw=16，Σw² = σ_H²/Var = {SIG_H**2/VARV:.1f}，M ≥ {256/(SIG_H**2/VARV):.2f}")
print()
HSETS = {
    "12 路：8×1 + 4×2": [1] * 8 + [2] * 4,
    "13 路：11×1 + 2×2": [1] * 11 + [2] * 2,
    "14 路：13×1 + 1×3": [1] * 13 + [3],
}
print(f"  {'方案':22s} {'Σw':>3s} {'Σw²':>5s} {'σ_H':>6s} {'高位 CDF':>9s}")
print("  " + "-" * 50)
for nm, w in HSETS.items():
    if sum(w) != 16:
        continue
    ph = pmf(w, nlev=32)
    mu = float((ph * np.arange(32)).sum())
    sd = float(np.sqrt((ph * (np.arange(32) - mu) ** 2).sum()))
    cdf = float(np.abs(np.cumsum(ph) - np.cumsum(P_TH)).max())
    print(f"  {nm:22s} {sum(w):3d} {sum(v*v for v in w):5d} {sd:6.3f} {cdf:9.4f}")

# 组合：高位 + 均匀低位
print()
print("  把高位分布与 3 位均匀拼成 0..255 后：")
for nm, w in HSETS.items():
    if sum(w) != 16:
        continue
    ph = pmf(w, nlev=32)
    pk8 = np.repeat(ph, 8) / 8.0        # 每档均摊到 8 个子级
    cdf, tv, mu, sd = metrics(pk8, P_T)
    print(f"    {nm:22s} 总投掷器 {len(w)+3:2d} 个   CDF {cdf:.4f}   σ {sd:.2f}")
print()
print("  结论：高位 12~13 个 + 低位 3 个 = 15~16 个，")
print("        而直接做 8 位只要 11 个。五位的简化不省投掷器，反而更多。")
print("THREEV_DONE")

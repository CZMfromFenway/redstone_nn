#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""验证 13 路三值投掷器方案。"""
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
PV = np.array(ABC) / 9.0
VARV = float((PV * (np.arange(3) - 1.0) ** 2).sum())


def pmf(w):
    pos = np.array([0])
    pr = np.array([1.0])
    for wi in w:
        v = np.arange(3) * int(wi)
        npos = (pos[:, None] + v[None, :]).ravel()
        npr = (pr[:, None] * PV[None, :]).ravel()
        up, inv = np.unique(npos, return_inverse=True)
        pos = up
        pr = np.bincount(inv, weights=npr)
    X = np.clip(pos, 0, NLEV - 1)
    pk = np.zeros(NLEV)
    np.add.at(pk, X, pr)
    return pk


def met(pk):
    cdf = float(np.abs(np.cumsum(pk) - np.cumsum(P_T)).max())
    tv = float(0.5 * np.abs(pk - P_T).sum())
    mu = float((pk * np.arange(NLEV)).sum())
    sd = float(np.sqrt((pk * (np.arange(NLEV) - mu) ** 2).sum()))
    return cdf, tv, mu, sd


CANDS = {
    "13 路": [1, 6, 7, 7, 7, 9, 10, 11, 11, 12, 14, 15, 18],
    "15 路": [1, 2, 3, 3, 5, 5, 7, 8, 10, 11, 12, 13, 15, 16, 17],
    "16 路": [1, 1, 1, 2, 3, 4, 6, 7, 9, 11, 12, 12, 12, 14, 15, 18],
}
print("=" * 78)
print("三值投掷器方案的最终对照")
print("=" * 78)
print(f"  {'方案':8s} {'M':>3s} {'sum w':>6s} {'sum w2':>7s} {'sigma':>8s} "
      f"{'CDF':>8s} {'TV':>8s} {'levels':>6s}")
print("  " + "-" * 66)
for nm, w in CANDS.items():
    if sum(w) != 128:
        print(f"  {nm}  和为 {sum(w)}，跳过")
        continue
    pk = pmf(w)
    cdf, tv, mu, sd = met(pk)
    print(f"  {nm:8s} {len(w):3d} {sum(w):6d} {sum(v*v for v in w):7d} {sd:8.3f} "
          f"{cdf:8.4f} {tv:8.4f} {int((pk>1e-12).sum()):6d}")

w = CANDS["13 路"]
pk = pmf(w)
cdf, tv, mu, sd = met(pk)
print()
print("=" * 78)
print("选定的方案：13 路三值投掷器")
print("=" * 78)
print(f"  权重（有序）：{sorted(w)}")
print(f"  sum w = {sum(w)}    sum w2 = {sum(v*v for v in w)}    gcd = {np.gcd.reduce(w)}")
print(f"  sigma = {sd:.3f}（目标 {SIG:.3f}）    CDF 最大差 = {cdf:.4f}    TV = {tv:.4f}")
ct = np.cumsum(P_T)
cb = np.cumsum(pk)
qs = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
err = [int(np.searchsorted(cb, q)) - int(np.searchsorted(ct, q)) for q in qs]
print(f"  分位误差：{dict(zip([f'{q:.2f}' for q in qs], err))}")
print()
print(f"  每个投掷器：9 格，{{0,1,2}} 分别占 {ABC[0]}/{ABC[1]}/{ABC[2]} 格")
print(f"              E[v] = 1，Var(v) = {VARV:.4f}")
print(f"  X = clip( sum_i w_i * v_i , 0, 255 )     均值 = sum w = 128")
print()
print(f"  对比：若投掷器能给五点形状 {{0..4}}，只要 6 个就能达到同样的 CDF 0.0101。")
print(f"        三值约束的代价是把 6 个变成 13 个。")
print("FINAL13_DONE")

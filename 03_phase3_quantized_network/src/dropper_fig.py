#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""把最优设计画出来，并与目标分布对照。"""
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

NLEV = 256
XS = -3.5 + (np.arange(NLEV) + 0.5) * 7.0 / NLEV
P_T = np.exp(-0.5 * XS ** 2)
P_T /= P_T.sum()

SHAPES = {
    "三": (np.array([-1, 0, 1]), np.array([1, 1, 1])),
    "五": (np.array([-2, -1, 0, 1, 2]), np.array([1, 2, 3, 2, 1])),
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
    return pk


CANDIDATES = [
    ("A 14 个三点头\n缩放 [1,1,1,1,2,3,7,9,12,14,17,17,20,23]", "三",
     [1, 1, 1, 1, 2, 3, 7, 9, 12, 14, 17, 17, 20, 23]),
    ("B 6 个五点头\n缩放 [1,5,7,11,19,21]", "五", [1, 5, 7, 11, 19, 21]),
    ("C 5 个五点头\n缩放 [3,11,12,18,20]", "五", [3, 11, 12, 18, 20]),
    ("D 4 个五点头（违反形状要求）\n缩放 [6,15,19,24]", "五", [6, 15, 19, 24]),
]

print("=" * 88)
print("候选设计对照")
print("=" * 88)
print(f"  {'设计':6s} {'σ':>7s} {'非零级数':>8s} {'TV':>8s} {'CDF 最大差':>11s} {'格子总数':>9s}")
print("  " + "-" * 60)
fits = []
for nm, sh, st in CANDIDATES:
    pk = pmf_of(st, sh)
    mu = float((pk * np.arange(NLEV)).sum())
    sd = float(np.sqrt((pk * (np.arange(NLEV) - mu) ** 2).sum()))
    tv = float(0.5 * np.abs(pk - P_T).sum())
    cdf = float(np.abs(np.cumsum(pk) - np.cumsum(P_T)).max())
    n = len(SHAPES[sh][1]) * len(st)
    fits.append((nm, pk, sd, tv, cdf, n))
    print(f"  {nm.split()[0]:6s} {sd:7.2f} {int((pk>1e-12).sum()):8d} {tv:8.4f} "
          f"{cdf:11.4f} {n:9d}")

if HAS_MPL:
    idx = np.arange(NLEV)
    fig, axes = plt.subplots(len(fits), 1, figsize=(11, 2.0 * len(fits)), sharex=True)
    for ax, (nm, pk, sd, tv, cdf, n) in zip(axes, fits):
        ax.fill_between(idx, P_T, color="#bbbbbb", label="目标（截断正态）")
        ax.plot(idx, pk, color="#1f77b4", lw=1.0, label="投掷器之和")
        ax.set_title(f"{nm}      σ={sd:.2f}   TV={tv:.4f}   CDF 最大差={cdf:.4f}",
                     fontsize=9, loc="left")
        ax.set_ylabel("概率", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, 255)
    axes[-1].set_xlabel("X（0..255）", fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS, "dropper_fit.png")
    fig.savefig(out, dpi=130)
    print(f"\n  wrote {out}  ({os.path.getsize(out)/1e3:.0f} KB)")
else:
    print("  matplotlib 不可用，跳过作图")

# 累积分布差的细节
print()
print("=" * 88)
print("最优设计 B 的分位对照")
print("=" * 88)
pk = pmf_of([1, 5, 7, 11, 19, 21], "五")
cb = np.cumsum(pk)
ct = np.cumsum(P_T)
print(f"  {'分位':>6s} {'目标 X':>8s} {'投掷器 X':>9s} {'差':>5s}")
print("  " + "-" * 32)
for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
    a = int(np.searchsorted(ct, q))
    b = int(np.searchsorted(cb, q))
    print(f"  {q:6.2f} {a:8d} {b:9d} {b-a:5d}")
print("DROPPER_FIG_DONE")

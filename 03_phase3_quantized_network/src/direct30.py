#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
直接法：X = c + Σ w_k v_k，M ≤ 30，权重全整数，无定点乘数。

约束
    Var(X) = 0.8889 Σw²  = s²   （s = 该通道 X 的标准差）
    支撑    = 2Σw，需要 ≥ 7s 才能覆盖 ±3.5σ
    mean(X) = c + Σw = m

搜索
    整数权重 + 局部搜索，目标是逐通道边缘分布的 CDF 最大差。
"""
import json
import os
import sys
import time
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
RESULTS = os.path.join(ROOT, "results")
styles = json.load(open(os.path.join(P2, "webapp", "styles_gauss.json"),
                        encoding="utf-8"))["styles"]

DELTA, NB = 7.0, 256
VARV = 0.8889
PV = np.array([4, 1, 4]) / 9.0

rng = np.random.default_rng(19)
NG = 400000
TARGET, MS, SD = {}, {}, {}
for e in styles:
    mu = np.array(e["mu"], dtype=float)
    L = np.array(e["chol"], dtype=float)
    x = rng.standard_normal((NG, 4)) @ L.T + mu
    xa = x[np.all(np.abs(x) <= 3.5, axis=1)]
    X = np.clip(np.floor((xa + 3.5) / DELTA * NB), 0, NB - 1).astype(int)
    nm = e["name"]
    TARGET[nm], MS[nm], SD[nm] = [], [], []
    for d in range(4):
        p = np.bincount(X[:, d], minlength=NB).astype(float)
        p /= p.sum()
        TARGET[nm].append(p)
        pidx = np.arange(NB)
        MS[nm].append(float((p * pidx).sum()))
        SD[nm].append(float(np.sqrt((p * (pidx - MS[nm][-1]) ** 2).sum())))


def pmf_dense(w):
    cur = np.array([1.0])
    for wi in w:
        n = np.zeros(len(cur) + 2 * wi)
        n[:len(cur)] += cur * PV[0]
        n[wi:wi + len(cur)] += cur * PV[1]
        n[2 * wi:2 * wi + len(cur)] += cur * PV[2]
        cur = n
    return cur


def sc_of(w, c, tgt):
    pk = pmf_dense(w)
    idx = np.clip(c + np.arange(len(pk)), 0, NB - 1)
    out = np.zeros(NB)
    np.add.at(out, idx, pk)
    return float(np.abs(np.cumsum(out) - np.cumsum(tgt)).max())


def fit(tgt, m, s, M, rng, iters=260):
    Q = s ** 2 / VARV
    w0 = max(1, int(round(np.sqrt(Q / M))))
    cur = np.full(M, w0, dtype=int)
    q = int((cur * cur).sum())
    best = (9.9, None, None)
    for _ in range(iters):
        i, j = int(rng.integers(M)), int(rng.integers(M))
        cand = cur.copy()
        cand[i] += int(rng.integers(-1, 2))
        cand[j] += int(rng.integers(-1, 2))
        if cand[i] < 1 or cand[j] < 1 or cand.max() > 6 * w0 + 6:
            continue
        qc = int((cand * cand).sum())
        if abs(qc - Q) > max(3.0, 0.10 * Q):
            continue
        base = int(round(m)) - int(cand.sum())
        for dc in range(-2, 3):
            scc = sc_of(cand, base + dc, tgt)
            if scc < best[0]:
                best = (scc, cand.copy(), base + dc)
                cur = cand.copy()
    return best


print("=" * 100)
print("直接法：CDF 最大差（逐风格 4 通道取平均）")
print("=" * 100)
print(f"  {'风格':10s} " + " ".join(f"{'M='+str(M):>7s}" for M in (8, 12, 16, 20, 26, 30)))
print("  " + "-" * 60)
t0 = time.time()
agg = {M: [] for M in (8, 12, 16, 20, 26, 30)}
BEST = {}
for e in styles:
    nm = e["name"]
    line = []
    for M in (8, 12, 16, 20, 26, 30):
        scs = []
        for d in range(4):
            (sc, w, c) = fit(TARGET[nm][d], MS[nm][d], SD[nm][d], M, rng)
            scs.append(sc)
            if sc < BEST.get((nm, d), (9.9,))[0]:
                BEST[(nm, d)] = (sc, M, w, c)
        agg[M].extend(scs)
        line.append(f"{np.mean(scs):7.4f}")
    print(f"  {nm:10s} " + " ".join(line))
print("  " + "-" * 60)
print(f"  {'平均':10s} " + " ".join(f"{np.mean(agg[M]):7.4f}" for M in (8, 12, 16, 20, 26, 30)))
print(f"\n  耗时 {time.time()-t0:.0f} s")

print()
print("=" * 100)
print("对照：INDEP 方案（13 个投掷器 + 16 位定点）的同一指标")
print("=" * 100)
WS = [1, 6, 7, 7, 7, 9, 10, 11, 11, 12, 14, 15, 18]
sig_drop = 36.466
spec = json.load(open(os.path.join(RESULTS, "dropper_spec.json"), encoding="utf-8"))
pos = np.array([0])
pr = np.array([1.0])
for wi in WS:
    v = np.arange(3) * wi
    np_ = (pos[:, None] + v[None, :]).ravel()
    npr = (pr[:, None] * PV[None, :]).ravel()
    up, inv = np.unique(np_, return_inverse=True)
    pos = up
    pr = np.bincount(inv, weights=npr)
ind = []
for e in styles:
    nm = e["name"]
    for d in range(4):
        st = spec["styles"][nm]
        n, B = st["n16"][d], st["B16"][d]
        Xc = np.clip((n * pos + B) >> 16, 0, NB - 1).astype(int)
        out = np.zeros(NB)
        np.add.at(out, Xc, pr)
        ind.append(float(np.abs(np.cumsum(out) - np.cumsum(TARGET[nm][d])).max()))
print(f"  24 组平均 CDF 差 {np.mean(ind):.4f}   最大 {np.max(ind):.4f}")

print()
print("=" * 100)
print("结论")
print("=" * 100)
for M in (8, 12, 16, 20, 26, 30):
    print(f"  M={M:2d}  平均 {np.mean(agg[M]):.4f}")
print(f"  INDEP (13+定点)  平均 {np.mean(ind):.4f}")
print()
print(f"  直接法 30 个仍不如 INDEP 的 13 个：{np.mean(agg[30]):.4f} vs {np.mean(ind):.4f}")
print("  原因：INDEP 的 13 个投掷器先生成一个连续的 z，再乘一个 16 位系数，")
print("        等于把 256 个格点「插值」到任意位置；直接法只能落在整数权重生成的格点上。")
print("DIRECT30_DONE")

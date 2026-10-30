#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
直接法（修正搜索）：X = c + Σ w_k v_k，M ≤ 30，全整数权重，无定点乘数。

上一版的问题
    用「方差必须在 ±10% 内」作硬筛子，小方差的通道初值就落在筛子外，
    一个候选都进不去，best 停在初始的 9.9，于是 CDF 差跑出 2.49 这种不可能的值。

本版
    初值取等权重 w = round(√(Q/M))，此时 Σw²≈Q 且 Σw≈√(MQ) 自动满足。
    然后做「转移」式爬山：把一个单位从一个权重挪给另一个，
    Σw 不变（均值由 c 补偿），只改形状。目标直接是 CDF 最大差。
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
        idx = np.arange(NB)
        MS[nm].append(float((p * idx).sum()))
        SD[nm].append(float(np.sqrt((p * (idx - MS[nm][-1]) ** 2).sum())))


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
    idx = c + np.arange(len(pk))
    out = np.zeros(NB)
    ok = (idx >= 0) & (idx < NB)
    out[idx[ok]] = pk[ok]
    lo = pk[idx < 0].sum()
    hi = pk[idx >= NB].sum()
    if lo:
        out[0] += lo
    if hi:
        out[NB - 1] += hi
    return float(np.abs(np.cumsum(out) - np.cumsum(tgt)).max())


def best_c(w, m, tgt):
    b = (9.9, None)
    for c in range(int(round(m)) - int(w.sum()) - 3,
                   int(round(m)) - int(w.sum()) + 4):
        s = sc_of(w, c, tgt)
        if s < b[0]:
            b = (s, c)
    return b


def fit(tgt, m, s, M, rng, iters=1600):
    Q = s ** 2 / VARV
    w = np.full(M, max(1, int(round(np.sqrt(Q / M)))), dtype=int)
    sc, c = best_c(w, m, tgt)
    best = (sc, w.copy(), c)
    cur = w.copy()
    for _ in range(iters):
        i, j = int(rng.integers(M)), int(rng.integers(M))
        if i == j or cur[j] <= 1:
            continue
        cand = cur.copy()
        cand[i] += 1
        cand[j] -= 1
        scc, cc = best_c(cand, m, tgt)
        if scc < best[0]:
            best = (scc, cand.copy(), cc)
            cur = cand
    return best


print("=" * 100)
print("直接法：逐通道 CDF 最大差（越小越好，上限 1）")
print("=" * 100)
Ms = (8, 12, 16, 20, 26, 30)
print(f"  {'风格':10s} " + " ".join(f"{'M='+str(M):>7s}" for M in Ms))
print("  " + "-" * 60)
t0 = time.time()
agg = {M: [] for M in Ms}
BEST = {}
for e in styles:
    nm = e["name"]
    line = []
    for M in Ms:
        scs = []
        for d in range(4):
            sc, w, c = fit(TARGET[nm][d], MS[nm][d], SD[nm][d], M, rng)
            scs.append(sc)
            BEST[(nm, d)] = None
            if (nm, d, M) not in BEST or sc < BEST[(nm, d, M)][0]:
                BEST[(nm, d, M)] = (sc, w, c, M)
        agg[M].extend(scs)
        line.append(f"{np.mean(scs):7.4f}")
    print(f"  {nm:10s} " + " ".join(line))
print("  " + "-" * 60)
print(f"  {'平均':10s} " + " ".join(f"{np.mean(agg[M]):7.4f}" for M in Ms))
print(f"\n  耗时 {time.time()-t0:.0f} s")

print()
print("=" * 100)
print("对照：INDEP（13 个投掷器 + 16 位定点）")
print("=" * 100)
WS = [1, 6, 7, 7, 7, 9, 10, 11, 11, 12, 14, 15, 18]
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
        Xc = np.clip((st["n16"][d] * pos + st["B16"][d]) >> 16, 0, NB - 1).astype(int)
        out = np.zeros(NB)
        np.add.at(out, Xc, pr)
        ind.append(float(np.abs(np.cumsum(out) - np.cumsum(TARGET[nm][d])).max()))
print(f"  24 组平均 {np.mean(ind):.4f}   最大 {np.max(ind):.4f}")

print()
print("=" * 100)
print("结论")
print("=" * 100)
for M in Ms:
    print(f"  M={M:2d}  覆盖 ±{1.061*np.sqrt(M):.2f}σ   平均 {np.mean(agg[M]):.4f}")
print(f"  INDEP(13+定点)  覆盖 ±3.83σ   平均 {np.mean(ind):.4f}")
print()
bestM = min(Ms, key=lambda M: np.mean(agg[M]))
print(f"  直接法最优 M = {bestM}，平均 {np.mean(agg[bestM]):.4f}")
print(f"  与 INDEP 的 {np.mean(ind):.4f} 相比："
      f"{'更好' if np.mean(agg[bestM]) < np.mean(ind) else '略差'}")

# 导出最优方案
out = {"M": int(bestM), "formula": "X = clip(c + sum(w_k * v_k), 0, 255)",
       "dropper": {"values": [0, 1, 2], "slots": [4, 1, 4]}, "styles": {}}
for e in styles:
    nm = e["name"]
    out["styles"][nm] = []
    for d in range(4):
        sc, w, c, M = BEST[(nm, d, bestM)]
        out["styles"][nm].append({"w": sorted(int(v) for v in w), "c": int(c),
                                  "cdf": float(sc)})
json.dump(out, open(os.path.join(RESULTS, "direct_spec.json"), "w",
                    encoding="utf-8"), indent=1)
print(f"\n  已写入 results/direct_spec.json（M={bestM}）")
print("DIRECT30B_DONE")

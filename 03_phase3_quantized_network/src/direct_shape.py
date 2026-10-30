#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
放开每个投掷器的分布：形状 + 权重都自由。

模型
    X = clip( c + Σ_{k=1..M} w_k · v_k, 0, 255 )
    第 k 个投掷器：{0,1,2} 三值，格数 (a_k,b_k,c_k)，a+b+c=9
                  p = (a,b,c)/9，E[v]=(b+2c)/9，贡献 {0, w_k, 2w_k}

自由度
    M 个权重 + M 个形状（55 种）+ 1 个偏置 = 2M+1 个整数
    要拟合 256 个数 —— 严重超定，所以残差不会归零。
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

# 所有 a+b+c=9 的三值分布
SHAPES = [(a, b, 9 - a - b) for a in range(10) for b in range(10 - a)]
SP = np.array([np.array(s) / 9.0 for s in SHAPES])          # (55,3)
SEV = SP @ np.array([0.0, 1.0, 2.0])                        # 均值
SVAR = SP @ (np.array([0.0, 1.0, 4.0])) - SEV ** 2          # 方差

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


def pmf_of(w, sh):
    cur = np.array([1.0])
    for wi, si in zip(w, sh):
        p = SP[si]
        n = np.zeros(len(cur) + 2 * wi)
        for j in range(3):
            if p[j] > 0:
                n[j * wi:j * wi + len(cur)] += cur * p[j]
        cur = n
    return cur


def cdf_dist(w, sh, c, tgt):
    pk = pmf_of(w, sh)
    idx = c + np.arange(len(pk))
    out = np.zeros(NB)
    lo, hi = pk[idx < 0].sum(), pk[idx >= NB].sum()
    ok = (idx >= 0) & (idx < NB)
    out[idx[ok]] = pk[ok]
    if lo:
        out[0] += lo
    if hi:
        out[NB - 1] += hi
    return float(np.abs(np.cumsum(out) - np.cumsum(tgt)).max())


def bias_for(w, sh, m):
    """使均值对齐的偏置（四舍五入后再微调）。"""
    base = int(round(m - float(np.dot(w, SEV[sh]))))
    return base


def fit(tgt, m, s, M, rng, restarts=3, iters=900):
    best = (9.9, None, None, None)
    for r in range(restarts):
        # 初值：形状全取 (4,1,4)（方差最大），权重等分到目标方差
        sh = np.full(M, SHAPES.index((4, 1, 4)), dtype=int)
        Q = s ** 2 / 0.8889
        w = np.full(M, max(1, int(round(np.sqrt(Q / M)))), dtype=int)
        c = bias_for(w, sh, m)
        cur = cdf_dist(w, sh, c, tgt)
        if r > 0:                    # 后续 restart 随机扰动
            w = np.maximum(1, w + rng.integers(-3, 4, size=M))
            sh = rng.integers(0, len(SHAPES), size=M)
            c = bias_for(w, sh, m)
            cur = cdf_dist(w, sh, c, tgt)
        if cur < best[0]:
            best = (cur, w.copy(), sh.copy(), c)
        for _ in range(iters):
            k = int(rng.integers(M))
            w2, sh2 = w.copy(), sh.copy()
            if rng.random() < 0.5:
                w2[k] = max(1, w2[k] + int(rng.integers(-2, 3)))
            else:
                sh2[k] = int(rng.integers(0, len(SHAPES)))
            c2 = bias_for(w2, sh2, m)
            for dc in (-1, 0, 1):
                v = cdf_dist(w2, sh2, c2 + dc, tgt)
                if v < cur:
                    v_ok, c_ok = v, c2 + dc
                    break
            else:
                continue
            cur, w, sh, c = v_ok, w2, sh2, c_ok
            if cur < best[0]:
                best = (cur, w.copy(), sh.copy(), c)
    return best


print("=" * 100)
print("放开形状：形状 + 权重都自由")
print("=" * 100)
Ms = (8, 10, 12, 14, 16)
print(f"  {'风格':10s} " + " ".join(f"{'M='+str(M):>7s}" for M in Ms))
print("  " + "-" * 52)
t0 = time.time()
agg = {M: [] for M in Ms}
BEST = {}
for e in styles:
    nm = e["name"]
    line = []
    for M in Ms:
        scs = []
        for d in range(4):
            sc, w, sh, c = fit(TARGET[nm][d], MS[nm][d], SD[nm][d], M, rng)
            scs.append(sc)
            if (nm, d) not in BEST or sc < BEST[(nm, d)][0]:
                BEST[(nm, d)] = (sc, w, sh, c, M)
        agg[M].extend(scs)
        line.append(f"{np.mean(scs):7.4f}")
    print(f"  {nm:10s} " + " ".join(line))
print("  " + "-" * 52)
print(f"  {'平均':10s} " + " ".join(f"{np.mean(agg[M]):7.4f}" for M in Ms))
print(f"\n  耗时 {time.time()-t0:.0f} s")
print()
print("  对照：只放开权重的直接法 M=8 是 0.0315；INDEP 是 0.0325")

bestM = min(Ms, key=lambda M: np.mean(agg[M]))
print(f"\n  最优 M = {bestM}，平均 {np.mean(agg[bestM]):.4f}")

out = {"M": int(bestM), "formula": "X = clip(c + sum(w_k*v_k), 0, 255)",
       "dropper": "each dropper: values {0,1,2}, slot counts (a,b,c) with a+b+c=9",
       "warning": "不同 M 的 BEST 是混的，导出时按各通道自己的最优 M 存",
       "styles": {}}
for e in styles:
    nm = e["name"]
    out["styles"][nm] = []
    for d in range(4):
        sc, w, sh, c, M = BEST[(nm, d)]
        out["styles"][nm].append({
            "M": int(M), "c": int(c), "cdf": float(sc),
            "channels": [{"w": int(wi), "abc": list(SHAPES[si])}
                         for wi, si in zip(w, sh)]})
json.dump(out, open(os.path.join(RESULTS, "direct_shape_spec.json"), "w",
                    encoding="utf-8"), indent=1)
print(f"  已写入 results/direct_shape_spec.json")
print("SHAPE_DONE")

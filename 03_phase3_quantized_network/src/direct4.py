#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
直接用投掷器的和作为 X：每通道 M 个投掷器，权重按风格单独拟合。

模型
    X = c + Σ_{k=1..M} w_k · v_k,   v_k ∈ {0,1,2}，格数分配 (4,1,4)/9
    E[v]=1, Var(v)=0.8889
    mean(X) = c + Σw_k
    Var(X)  = 0.8889 · Σw_k²
    支撑     [c, c + 2Σw_k]

可达律
    M 个投掷器最多铺到 ρ√M σ，ρ=1.061
    M=4  →  2.1σ      M=6 → 2.6σ      M=11 → 3.5σ

所以 M=4 天生丢掉 ±2.1σ 以外的质量（约 3.6%）。
本脚本量化每个 M 的拟合误差。
"""
import json
import os
import sys
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
rng = np.random.default_rng(5)

# ---------- 各风格各通道的目标边缘分布
rng0 = np.random.default_rng(19)
NG = 400000
TARGET = {}
for e in styles:
    mu = np.array(e["mu"], dtype=float)
    L = np.array(e["chol"], dtype=float)
    x = rng0.standard_normal((NG, 4)) @ L.T + mu
    xa = x[np.all(np.abs(x) <= 3.5, axis=1)]
    X = np.clip(np.floor((xa + 3.5) / DELTA * NB), 0, NB - 1).astype(int)
    TARGET[e["name"]] = []
    for d in range(4):
        p = np.bincount(X[:, d], minlength=NB).astype(float)
        p /= p.sum()
        TARGET[e["name"]].append(p)


def model_pmf(w, c):
    pos = np.array([0])
    pr = np.array([1.0])
    for wi in w:
        v = np.arange(3) * int(wi)
        np_ = (pos[:, None] + v[None, :]).ravel()
        npr = (pr[:, None] * PV[None, :]).ravel()
        up, inv = np.unique(np_, return_inverse=True)
        pos = up
        pr = np.bincount(inv, weights=npr)
    X = np.clip(c + pos, 0, NB - 1)
    pk = np.zeros(NB)
    np.add.at(pk, X, pr)
    return pk


def score(w, c, tgt):
    pk = model_pmf(w, c)
    return float(np.abs(np.cumsum(pk) - np.cumsum(tgt)).max())


def fit(tgt, M, rng, iters=3000):
    p = np.arange(NB)
    m = float((tgt * p).sum())
    s = float(np.sqrt((tgt * (p - m) ** 2).sum()))
    Q = s ** 2 / VARV
    hi = int(np.sqrt(Q)) + 1
    best = (9.9, None, None)
    # 用均衡初值 + 局部扰动
    base = np.full(M, max(1, int(round(np.sqrt(Q / M)))))
    cur = base.copy()
    curq = int((cur * cur).sum())
    for it in range(iters):
        i, j = int(rng.integers(M)), int(rng.integers(M))
        cand = cur.copy()
        cand[i] += int(rng.integers(-1, 2))
        cand[j] += int(rng.integers(-1, 2))
        if cand[i] < 1 or cand[j] < 1:
            continue
        if cand[i] > hi * 2 or cand[j] > hi * 2:
            continue
        q = int((cand * cand).sum())
        if abs(q - Q) / Q > 0.35:
            continue
        ssum = int(cand.sum())
        for dc in range(-2, 3):
            c = int(round(m)) - ssum + dc
            sc = score(cand, c, tgt)
            if sc < best[0]:
                best = (sc, cand.copy(), c)
                cur, curq = cand.copy(), q
    return best, m, s


print("=" * 96)
print("每通道 M 个投掷器直接生成 X（权重按风格单独拟合）")
print("=" * 96)
print(f"  {'风格':10s} {'M':>3s} {'逐通道 CDF 最大差':>40s} {'平均':>7s}")
print("  " + "-" * 68)
res = {}
for M in (4, 5, 6, 8, 11):
    allsc = []
    for e in styles:
        sc4 = []
        for d in range(4):
            (sc, w, c), m, s = fit(TARGET[e["name"]][d], M, rng,
                                   iters=600 if M <= 6 else 400)
            sc4.append(sc)
        allsc.extend(sc4)
        res[(e["name"], M)] = sc4
        txt = "  ".join(f"{v:.3f}" for v in sc4)
        print(f"  {e['name'] if M == 4 else '':10s} {M:3d} {txt:>40s} "
              f"{np.mean(sc4):7.3f}")
    print(f"  {'' :13s} {'全部平均':>10s} {np.mean(allsc):.4f}")
    print()

print("=" * 96)
print("结论")
print("=" * 96)
for M in (4, 5, 6, 8, 11):
    a = np.mean([np.mean(res[(e['name'], M)]) for e in styles])
    print(f"  M={M:2d}  覆盖 ±{1.061*np.sqrt(M):.2f}σ   平均 CDF 差 {a:.4f}")
print()
print("  M=4 只能铺到 ±2.1σ，丢掉约 3.6% 的尾部质量。")
print("  M=8 能到 ±3.0σ，M=11 才到 ±3.5σ。")
print("CAP4_DONE")

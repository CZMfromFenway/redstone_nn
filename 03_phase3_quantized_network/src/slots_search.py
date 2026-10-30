#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
放开格数（N ≤ 9），并把搜索过程完整记录下来。

形状集合
    (a,b,c) 三值格子数，N = a+b+c ∈ [1,9]
    总共 Σ_{N=1..9} C(N+2,2) = 219 种
    对比：只允许 N=9 时是 55 种
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

# 全部 N<=9 的形状
SHAPES = []
for N in range(1, 10):
    for a in range(N + 1):
        for b in range(N - a + 1):
            SHAPES.append((a, b, N - a - b))
SP = np.array([np.array(s, dtype=float) / sum(s) for s in SHAPES])
SEV = SP @ np.array([0.0, 1.0, 2.0])
print(f"形状总数 {len(SHAPES)}（N<=9）；N=1 的 {sum(1 for s in SHAPES if sum(s)==1)} 种是退化的")

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
    return int(round(m - float(np.dot(w, SEV[sh]))))


def fit(tgt, m, s, M, rng, restarts=3, iters=600, trace=False):
    """返回 (最优, 权重, 形状, 偏置, 统计)。"""
    stats = {"init": [], "final": [], "evals": 0, "accepted": 0, "trace": []}
    best = (9.9, None, None, None)
    for r in range(restarts):
        sh = np.full(M, SHAPES.index((4, 1, 4)), dtype=int)
        Q = s ** 2 / 0.8889
        w = np.full(M, max(1, int(round(np.sqrt(Q / M)))), dtype=int)
        if r > 0:
            w = np.maximum(1, w + rng.integers(-4, 5, size=M))
            sh = rng.integers(0, len(SHAPES), size=M)
        c = bias_for(w, sh, m)
        cur = cdf_dist(w, sh, c, tgt)
        stats["init"].append(cur)
        stats["evals"] += 1
        local = cur
        tr = [cur]
        for it in range(iters):
            k = int(rng.integers(M))
            w2, sh2 = w.copy(), sh.copy()
            if rng.random() < 0.5:
                w2[k] = max(1, w2[k] + int(rng.integers(-2, 3)))
            else:
                sh2[k] = int(rng.integers(0, len(SHAPES)))
            c2 = bias_for(w2, sh2, m)
            acc = None
            for dc in (-1, 0, 1):
                stats["evals"] += 1
                v = cdf_dist(w2, sh2, c2 + dc, tgt)
                if v < local:
                    acc = (v, c2 + dc)
                    break
            if acc is None:
                continue
            local = acc[0]
            cur, w, sh, c = local, w2, sh2, acc[1]
            stats["accepted"] += 1
            if trace:
                tr.append(local)
            if local < best[0]:
                best = (local, w.copy(), sh.copy(), c)
        stats["final"].append(local)
        if trace:
            stats["trace"].append(tr)
    return best, stats


print()
print("=" * 100)
print("放开格数 N <= 9 的结果")
print("=" * 100)
Ms = (8, 10, 12)
print(f"  {'风格':10s} " + " ".join(f"{'M='+str(M):>7s}" for M in Ms))
print("  " + "-" * 44)
t0 = time.time()
agg = {M: [] for M in Ms}
BEST = {}
for e in styles:
    nm = e["name"]
    line = []
    for M in Ms:
        scs = []
        for d in range(4):
            (sc, w, sh, c), st = fit(TARGET[nm][d], MS[nm][d], SD[nm][d], M, rng)
            scs.append(sc)
            if (nm, d) not in BEST or sc < BEST[(nm, d)][0]:
                BEST[(nm, d)] = (sc, w, sh, c, M)
        agg[M].extend(scs)
        line.append(f"{np.mean(scs):7.4f}")
    print(f"  {nm:10s} " + " ".join(line))
print("  " + "-" * 44)
print(f"  {'平均':10s} " + " ".join(f"{np.mean(agg[M]):7.4f}" for M in Ms))
print(f"\n  耗时 {time.time()-t0:.0f} s")
print("  对照：N=9 固定时 0.0152")

bestM = min(Ms, key=lambda M: np.mean(agg[M]))
print(f"\n  最优 M = {bestM}，平均 {np.mean(agg[bestM]):.4f}")
shp = {}
for nm, d in BEST:
    for si in BEST[(nm, d)][2]:
        N = sum(SHAPES[si])
        shp[N] = shp.get(N, 0) + 1
print(f"  用到的格数分布：{dict(sorted(shp.items()))}")
print(f"  权重范围 [{min(BEST[k][1].min() for k in BEST)}, "
      f"{max(BEST[k][1].max() for k in BEST)}]")
print(f"  偏置范围 [{min(BEST[k][3] for k in BEST)}, {max(BEST[k][3] for k in BEST)}]")

out = {"M_note": "每通道各自的 M（8/10/12 里最优的那个）",
       "formula": "X = clip(c + sum(w_k*v_k), 0, 255)",
       "dropper": "each dropper: values {0,1,2}, slot counts (a,b,c), N=a+b+c<=9",
       "styles": {}}
for e in styles:
    nm = e["name"]
    out["styles"][nm] = []
    for d in range(4):
        sc, w, sh, c, M = BEST[(nm, d)]
        out["styles"][nm].append({
            "M": int(M), "c": int(c), "cdf": float(sc),
            "channels": [{"w": int(wi), "abc": list(SHAPES[si]), "N": int(sum(SHAPES[si]))}
                         for wi, si in zip(w, sh)]})
json.dump(out, open(os.path.join(RESULTS, "slots_spec.json"), "w",
                    encoding="utf-8"), indent=1)
print(f"  已写入 results/slots_spec.json")

# -------- 记录一次完整搜索的轨迹
print()
print("=" * 100)
print("搜索过程记录：少女 ch1，M=10，跑一次看轨迹")
print("=" * 100)
(nm, d) = ("少女", 0)
(sc, w, sh, c), st = fit(TARGET[nm][d], MS[nm][d], SD[nm][d], 10, rng,
                         restarts=3, iters=600, trace=True)
print(f"  初值 CDF 差：{[f'{v:.4f}' for v in st['init']]}")
print(f"  终值 CDF 差：{[f'{v:.4f}' for v in st['final']]}")
print(f"  总评估次数 {st['evals']}，接受次数 {st['accepted']}"
      f"（接受率 {100*st['accepted']/st['evals']:.1f}%）")
print(f"  最优 {sc:.4f}")
print(f"  每次重启的下降幅度："
      f"{[f'{a-b:+.4f}' for a, b in zip(st['init'], st['final'])]}")
json.dump({"init": st["init"], "final": st["final"],
           "trace": [list(map(float, t)) for t in st["trace"]]},
          open(os.path.join(RESULTS, "search_trace.json"), "w", encoding="utf-8"))
print("  已写入 results/search_trace.json")
print("SLOTS_DONE")

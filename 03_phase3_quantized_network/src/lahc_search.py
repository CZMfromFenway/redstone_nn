#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
换搜索算法：Late Acceptance Hill Climbing（LAHC）。

为什么不直接用爬山
    纯爬山只接受「更好」的移动，一旦卡在局部极小就永远出不来。
    族越大（55 -> 219 种形状），局部极小越多，爬山反而变差——
    这就是上一轮「放开格数却更差」的真正原因。

LAHC
    维护一个长度 L 的历史队列。
    候选解只要比「L 步之前的那个解」好，就接受（哪怕比当前差）。
    等于给了一个自动衰减的容忍度：早期宽松，后期收紧。
    不需要温度表，只有一个参数 L。
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

FAM = {}
for tag, Nmax in (("N=9", 9), ("N<=9", 9)):
    if tag == "N=9":
        fam = [(a, b, 9 - a - b) for a in range(10) for b in range(10 - a)]
    else:
        fam = [(a, b, N - a - b) for N in range(1, 10)
               for a in range(N + 1) for b in range(N - a + 1)]
    FAM[tag] = (fam, np.array([np.array(s, dtype=float) / sum(s) for s in fam]))

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


def make(tag):
    fam, SP = FAM[tag]
    SEV = SP @ np.array([0.0, 1.0, 2.0])

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

    return fam, SP, pmf_of, cdf_dist, bias_for


def fit(tag, tgt, m, s, M, rng, restarts=2, iters=2500, L=80, trace=False):
    fam, SP, pmf_of, cdf_dist, bias_for = make(tag)
    i414 = fam.index((4, 1, 4))
    best = (9.9, None, None, None)
    stats = {"init": [], "final": [], "evals": 0, "acc": 0, "trace": []}
    for r in range(restarts):
        sh = np.full(M, i414, dtype=int)
        w = np.full(M, max(1, int(round(np.sqrt(s ** 2 / 0.8889 / M)))), dtype=int)
        if r > 0:
            w = np.maximum(1, w + rng.integers(-4, 5, size=M))
            sh = rng.integers(0, len(fam), size=M)
        c = bias_for(w, sh, m)
        cur = cdf_dist(w, sh, c, tgt)
        stats["init"].append(cur)
        stats["evals"] += 1
        hist = [cur] * L
        tr = [cur]
        for it in range(iters):
            k = int(rng.integers(M))
            w2, sh2 = w.copy(), sh.copy()
            if rng.random() < 0.5:
                w2[k] = max(1, w2[k] + int(rng.integers(-2, 3)))
            else:
                sh2[k] = int(rng.integers(0, len(fam)))
            c2 = bias_for(w2, sh2, m)
            v = 9.9
            cb = c2
            for dc in (-1, 0, 1):
                stats["evals"] += 1
                vv = cdf_dist(w2, sh2, c2 + dc, tgt)
                if vv < v:
                    v, cb = vv, c2 + dc
            slot = it % L
            if v <= hist[slot] or v < cur:
                hist[slot] = cur
                cur, w, sh, c = v, w2, sh2, cb
                stats["acc"] += 1
                if trace:
                    tr.append(cur)
                if cur < best[0]:
                    best = (cur, w.copy(), sh.copy(), c)
        stats["final"].append(cur)
        if trace:
            stats["trace"].append(tr)
    return best, stats


print("=" * 100)
print("LAHC 搜索：两种形状族对照")
print("=" * 100)
Ms = (8, 10, 12)
res = {}
t0 = time.time()
for tag in ("N=9", "N<=9"):
    print(f"\n  【{tag}】形状数 {len(FAM[tag][0])}")
    print(f"  {'风格':10s} " + " ".join(f"{'M='+str(M):>7s}" for M in Ms))
    print("  " + "-" * 44)
    for M in Ms:
        res[(tag, M)] = []
    for e in styles:
        nm = e["name"]
        line = []
        for M in Ms:
            scs = []
            for d in range(4):
                (sc, w, sh, c), st = fit(tag, TARGET[nm][d], MS[nm][d], SD[nm][d], M, rng)
                scs.append(sc)
            res[(tag, M)].extend(scs)
            line.append(f"{np.mean(scs):7.4f}")
        print(f"  {nm:10s} " + " ".join(line))
    print("  " + "-" * 44)
    print(f"  {'平均':10s} " + " ".join(f"{np.mean(res[(tag,M)]):7.4f}" for M in Ms))
print(f"\n  耗时 {time.time()-t0:.0f} s")

print()
print("=" * 100)
print("汇总")
print("=" * 100)
for tag in ("N=9", "N<=9"):
    bM = min(Ms, key=lambda M: np.mean(res[(tag, M)]))
    print(f"  {tag:5s}  最优 M={bM}  平均 CDF {np.mean(res[(tag,bM)]):.4f}")
print("  旧（纯爬山，N=9）  0.0152")
print("  旧（纯爬山，N<=9） 0.0184")

# 轨迹
print()
print("=" * 100)
print("搜索轨迹：最佳组合，看下降过程")
print("=" * 100)
tag = "N<=9"
bestM = min(Ms, key=lambda M: np.mean(res[(tag, M)]))
(nm, d) = ("少女", 0)
(sc, w, sh, c), st = fit(tag, TARGET[nm][d], MS[nm][d], SD[nm][d], bestM, rng,
                         trace=True)
print(f"  配置 {tag}, M={bestM}, 通道 {nm} ch1")
print(f"  初值 {[f'{v:.4f}' for v in st['init']]}")
print(f"  终值 {[f'{v:.4f}' for v in st['final']]}")
print(f"  评估 {st['evals']} 次，接受 {st['acc']} 次（{100*st['acc']/st['evals']:.1f}%）")
json.dump({"init": st["init"], "final": st["final"],
           "trace": [list(map(float, t)) for t in st["trace"]],
           "cfg": f"{tag} M={bestM} {nm} ch1"},
          open(os.path.join(RESULTS, "search_trace.json"), "w", encoding="utf-8"))
print("  已写入 results/search_trace.json")
print("LAHC_DONE")

#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
盲测结果分析：任意一轮的标签 → 每臂的良率、均值、"好脸率"、置信区间与两两检验。

用法：
    python analyse_round.py                        # 默认第 1 轮
    python analyse_round.py --round 2 --base k8snap

口径（与二期一致，也和本轮的页面说明一致）：
    grade 0..4；"可接受" = grade >= 2；"好" = grade >= 3
统计：
    Wilson 95% 区间；两两 **置换检验**（10,000 次，均值差与"好脸率"差），
    Holm 校正；另给 Fisher 精确检验（可接受率的 2×2），便于和二期数字直接比。
产物：results/labels/analysis_round<r>.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
REPO = os.path.dirname(P4)
LDIR = os.path.join(P4, "results", "labels")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_p(a, b, c, d):
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def p_tab(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)

    p0 = p_tab(a)
    tot = 0.0
    for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1):
        px = p_tab(x)
        if px <= p0 + 1e-12:
            tot += px
    return min(1.0, tot)


def perm_test(x, y, stat, n_perm=10000, seed=0):
    """双侧置换检验：stat(xy) 的观测值在置换分布中的双尾概率。"""
    rng = np.random.default_rng(seed)
    obs = stat(x, y)
    pool = np.concatenate([x, y])
    nx = len(x)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(stat(pool[:nx], pool[nx:])) >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def holm(pvals):
    """输入 [(key,p)] -> 输出 [(key,p, p_holm)]"""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i][1])
    m = len(pvals)
    out = [None] * m
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i][1])
        adj = max(adj, prev)
        prev = adj
        out[i] = (pvals[i][0], pvals[i][1], adj)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="1")
    ap.add_argument("--base", default="k8snap")
    ap.add_argument("--file", default="")
    A = ap.parse_args()
    path = A.file or os.path.join(LDIR, f"colour_round{A.round}.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"读 {os.path.relpath(path, REPO)}：{len(rows)} 条")

    arms = {}
    for r in rows:
        arms.setdefault(r["model"], []).append(r)
    order = sorted(arms, key=lambda m: -float(np.mean([x["grade"] for x in arms[m]])))

    out = {"file": os.path.basename(path), "n": len(rows), "arms": {}}
    print(f"\n{'臂':10s} {'n':>4s} {'均值':>6s} {'可接受≥2':>16s} {'好脸≥3':>16s} {'分布 0/1/2/3/4'}")
    for m in order:
        g = np.array([x["grade"] for x in arms[m]])
        n = len(g)
        acc = int((g >= 2).sum())
        good = int((g >= 3).sum())
        al, ah = wilson(acc, n)
        gl, gh = wilson(good, n)
        hist = [int((g == i).sum()) for i in range(5)]
        out["arms"][m] = {"n": n, "mean": round(float(g.mean()), 3),
                          "accept": {"k": acc, "rate": round(acc / n, 4),
                                     "ci": [round(al, 4), round(ah, 4)]},
                          "good": {"k": good, "rate": round(good / n, 4),
                                   "ci": [round(gl, 4), round(gh, 4)]},
                          "hist": hist, "median": float(np.median(g))}
        print(f"{m:10s} {n:4d} {g.mean():6.3f}  {acc:3d}/{n:<3d} {acc/n*100:5.1f}% "
              f"[{al*100:4.1f},{ah*100:5.1f}]  {good:3d}/{n:<3d} {good/n*100:5.1f}% "
              f"[{gl*100:4.1f},{gh*100:5.1f}]  {hist}")

    # 两两检验（每个臂对基线 + 相邻维度）
    def stat_mean(x, y):
        return float(x.mean() - y.mean())

    def stat_good(x, y):
        return float((x >= 3).mean() - (y >= 3).mean())

    pairs = []
    for m in order:
        if m == A.base:
            continue
        pairs.append((m, A.base))
    for a, b in (("k8", "k16"), ("k16", "k24"), ("k24", "k32"), ("k8", "k32")):
        if a in arms and b in arms and (a, b) not in pairs and (b, a) not in pairs:
            pairs.append((a, b))

    print(f"\n两两置换检验（10,000 次；A 对 B，正= A 更好）")
    print(f"{'对比':16s} {'Δ均值':>7s} {'p':>7s} {'p_Holm':>7s} | {'Δ好脸率':>8s} {'p':>7s} {'p_Holm':>7s} | {'Δ可接受率':>9s} {'p_Fisher':>8s}")
    res = []
    for a, b in pairs:
        xa = np.array([x["grade"] for x in arms[a]])
        xb = np.array([x["grade"] for x in arms[b]])
        pm = perm_test(xa, xb, stat_mean)
        pg = perm_test(xa, xb, stat_good)
        fa = (int((xa >= 2).sum()), int((xa < 2).sum()))
        fb = (int((xb >= 2).sum()), int((xb < 2).sum()))
        pf = fisher_p(fa[0], fa[1], fb[0], fb[1])
        res.append({"a": a, "b": b,
                    "d_mean": round(float(xa.mean() - xb.mean()), 3), "p_mean": round(pm, 4),
                    "d_good": round(float((xa >= 3).mean() - (xb >= 3).mean()), 3),
                    "p_good": round(pg, 4),
                    "d_accept": round(float((xa >= 2).mean() - (xb >= 2).mean()), 3),
                    "p_fisher_accept": round(pf, 4),
                    "n_a": len(xa), "n_b": len(xb)})
    hm = holm([(f"{r['a']}|{r['b']}|mean", r["p_mean"]) for r in res])
    hg = holm([(f"{r['a']}|{r['b']}|good", r["p_good"]) for r in res])
    for i, r in enumerate(res):
        r["p_mean_holm"] = round(hm[i][2], 4)
        r["p_good_holm"] = round(hg[i][2], 4)
        print(f"{r['a']+' vs '+r['b']:16s} {r['d_mean']:7.3f} {r['p_mean']:7.4f} {r['p_mean_holm']:7.4f} | "
              f"{r['d_good']:8.3f} {r['p_good']:7.4f} {r['p_good_holm']:7.4f} | "
              f"{r['d_accept']:9.3f} {r['p_fisher_accept']:8.4f}")
    out["pairs"] = res

    # 疲劳/漂移：评分顺序与等级的相关
    idx = np.arange(len(rows))
    g = np.array([r["grade"] for r in rows])
    out["order_corr"] = {"pearson_grade_vs_index": round(float(np.corrcoef(idx, g)[0, 1]), 3)}
    print(f"\n顺序漂移：grade 与作答序号的相关系数 {out['order_corr']['pearson_grade_vs_index']:+.3f}"
          f"（≈0 说明标准没漂；明显负值说明越评越严）")

    dst = os.path.join(LDIR, f"analysis_round{A.round}.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"wrote {os.path.relpath(dst, REPO)}")


if __name__ == "__main__":
    main()

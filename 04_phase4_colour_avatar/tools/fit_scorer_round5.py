#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
用第三、四轮的标签拟合「第五轮要用的」旋钮打分器。

为什么要合并两轮：第四轮的两个"筛选"臂因页面 bug 实际是随机的，于是
    两阶段：第三轮 80 + 第四轮 80（two_pal16）+ 80（two_sel）= 240 条
    k8snap：第三轮 80 + 第四轮 80（k8snap）  + 80（snap_sel） = 240 条
是同一模型、同一分布的重复测量，可以合法合并——比第三轮单独拟合（n=80）稳得多。

产物：results/labels/scorers_round5.json（按模型族给出线性/二次两套，附 CV AUC）
用法：python fit_scorer_round5.py
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
P4 = os.path.dirname(HERE)
REPO = os.path.dirname(P4)
LDIR = os.path.join(P4, "results", "labels")

FAMILIES = {
    "two_stage": {"k": 4,
                  "sources": [("3", "two_pal16"), ("4", "two_pal16"), ("4", "two_sel")]},
    "k8snap": {"k": 8,
               "sources": [("3", "k8snap"), ("4", "k8snap"), ("4", "snap_sel")]},
}


def load(round_no):
    p = os.path.join(LDIR, f"colour_round{round_no}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def feats(X, kind):
    cols = [np.ones((len(X), 1)), X]
    if kind == "quad":
        cols.append(X ** 2)
        iu = np.triu_indices(X.shape[1], 1)
        cols.append(X[:, iu[0]] * X[:, iu[1]])
    return np.hstack(cols)


def names(k, kind):
    out = ["const"] + [f"PC{i}" for i in range(k)]
    if kind == "quad":
        out += [f"PC{i}^2" for i in range(k)]
        out += [f"PC{i}*PC{j}" for i in range(k) for j in range(i + 1, k)]
    return out


def ridge(F, y, lam):
    return np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ (y * 2.0 - 1.0))


def auc(s, y):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    p = y == 1
    return float((r[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * (~p).sum()))


def main():
    rows = {r: load(r) for r in ("3", "4")}
    out = {"note": "由第三、四轮同一模型的全部标签合并拟合；分数越高越好", "families": {}}
    for fam, spec in FAMILIES.items():
        X, y = [], []
        for rnd, arm in spec["sources"]:
            for r in rows[rnd]:
                if r["model"] != arm:
                    continue
                X.append([float(v) for v in (r.get("z") or r["pca"])[:spec["k"]]])
                y.append(1.0 if r["grade"] >= 3 else 0.0)
        X = np.array(X)
        y = np.array(y)
        print(f"\n{fam}: 合并后 n={len(y)}，好率 {y.mean()*100:.1f}%，"
              f"来自 {spec['sources']}")
        best = None
        for kind, lam in (("linear", 1.0), ("quad", 20.0), ("quad", 60.0)):
            F = feats(X, kind)
            idx = np.random.default_rng(1).permutation(len(y))
            sc = np.zeros(len(y))
            for i in range(5):
                te = idx[i::5]
                tr = np.setdiff1d(idx, te)
                sc[te] = F[te] @ ridge(F[tr], y[tr], lam)
            a = auc(sc, y)
            order = np.argsort(-sc)
            top50 = float(y[order[:len(y)//2]].mean())
            top30 = float(y[order[:len(y)*3//10]].mean())
            print(f"  {kind:6s} λ={lam:4.0f}  {F.shape[1]:3d} 项  CV AUC {a:.3f}  "
                  f"前50% {top50*100:5.1f}%  前30% {top30*100:5.1f}%")
            cand = {"family": fam, "kind": kind, "lam": lam, "n_terms": int(F.shape[1]),
                    "k": spec["k"], "n": int(len(y)),
                    "base_rate": round(float(y.mean()), 3), "cv_auc": round(a, 4),
                    "in_sample_top50": round(top50, 3), "in_sample_top30": round(top30, 3),
                    "weights": [round(float(v), 6) for v in ridge(F, y, lam)],
                    "feature_names": names(spec["k"], kind),
                    "sources": [f"r{rnd}:{arm}" for rnd, arm in spec["sources"]]}
            if best is None or a > best["cv_auc"]:
                best = cand
        out["families"][fam] = best
        print(f"  → 选用 {best['kind']} λ={best['lam']} CV AUC {best['cv_auc']:.3f}")
    json.dump(out, open(os.path.join(LDIR, "scorers_round5.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\nwrote results/labels/scorers_round5.json")


if __name__ == "__main__":
    main()

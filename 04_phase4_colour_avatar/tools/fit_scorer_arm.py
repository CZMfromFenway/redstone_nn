#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
按「臂」拟合旋钮打分器，供盲测页做拒绝采样。

为什么分臂：第一轮把 k8 与 k8snap 的标签混在一起拟合，第二轮实测证明那样拟合出来的
打分器无效。第三轮的分臂分析显示两边的最优形式不同：

    two_pal16（4 旋钮）：线性 5 项 CV AUC 0.340（无效）→ 二次 15 项 0.654
    k8snap   （8 旋钮）：线性 9 项 CV AUC 0.727 → 二次 45 项 0.679（过拟合）

产物：results/labels/scorer_<arm>.json（含权重顺序说明）
用法：python fit_scorer_arm.py [--round 3]
"""
import argparse
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

JOBS = [
    {"arm": "two_pal16", "k": 4, "kind": "quad", "lam": 10.0},
    {"arm": "k8snap", "k": 8, "kind": "linear", "lam": 10.0},
]


def feat_names(k, kind):
    names = ["const"] + [f"PC{i}" for i in range(k)]
    if kind == "quad":
        names += [f"PC{i}^2" for i in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                names.append(f"PC{i}*PC{j}")
    return names


def feats(X, kind):
    cols = [np.ones((len(X), 1)), X]
    if kind == "quad":
        cols.append(X ** 2)
        iu = np.triu_indices(X.shape[1], 1)
        cols.append(X[:, iu[0]] * X[:, iu[1]])
    return np.hstack(cols)


def ridge(F, y, lam):
    return np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ (y * 2.0 - 1.0))


def auc(s, y):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    p = y == 1
    return float((r[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * (~p).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="3")
    A = ap.parse_args()
    rows = [json.loads(l) for l in
            open(os.path.join(LDIR, f"colour_round{A.round}.jsonl"), encoding="utf-8")
            if l.strip()]
    out = {"round": int(A.round), "scorers": {}}
    for job in JOBS:
        sel = [r for r in rows if r["model"] == job["arm"]]
        if len(sel) < 20:
            print(f"  跳过 {job['arm']}（只有 {len(sel)} 条）")
            continue
        k = job["k"]
        X = np.array([r["pca"][:k] for r in sel], dtype=np.float64)
        y = np.array([1.0 if r["grade"] >= 3 else 0.0 for r in sel])
        F = feats(X, job["kind"])
        idx = np.random.default_rng(1).permutation(len(y))
        sc = np.zeros(len(y))
        for i in range(5):
            te = idx[i::5]
            tr = np.setdiff1d(idx, te)
            sc[te] = F[te] @ ridge(F[tr], y[tr], job["lam"])
        w = ridge(F, y, job["lam"])
        order = np.argsort(-sc)
        top = {f"top{int(f*100)}": round(float(y[order[:max(1, int(f*len(y)))]].mean()), 3)
               for f in (0.5, 0.3)}
        rec = {"arm": job["arm"], "k": k, "kind": job["kind"], "lam": job["lam"],
               "n": int(len(y)), "base_rate": round(float(y.mean()), 3),
               "cv_auc": round(auc(sc, y), 4),
               "weights": [round(float(v), 6) for v in w],
               "feature_names": feat_names(k, job["kind"]),
               "in_sample_top": top,
               "note": "分数越高越好；in_sample_top 是在同一批标签上挑出来的，必然乐观"}
        out["scorers"][job["arm"]] = rec
        print(f"  {job['arm']:10s} {job['kind']:6s} {F.shape[1]:3d} 项  CV AUC {rec['cv_auc']:.3f}  "
              f"基线 {rec['base_rate']*100:.1f}%  前50% {top['top50']*100:.1f}%  "
              f"前30% {top['top30']*100:.1f}%")
    json.dump(out, open(os.path.join(LDIR, "scorers_round3.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("wrote results/labels/scorers_round3.json")


if __name__ == "__main__":
    main()

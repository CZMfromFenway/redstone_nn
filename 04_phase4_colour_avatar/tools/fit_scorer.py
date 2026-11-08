#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
用盲测标签拟合「旋钮 → 好不好」的打分器，供页面做拒绝采样。

主判据（作者 2026-11-02 定）：grade >= 3 算"好"。
只拟合 8 维输入（k8 / k8snap 两个臂，都是 8 个旋钮），线性 9 项——
45 项的二次在 n=121 上会过拟合（CV AUC 0.679 vs 线性 0.708）。

产物：results/labels/scorer_good.json
    {weights:[w0..w8], cv_auc, n, target:"grade>=3", source:..., lam}
用法：python fit_scorer.py [--round 1] [--lam 10]
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
ARMS_8D = ("k8", "k8snap")


def feats(X):
    return np.hstack([np.ones((len(X), 1)), X])


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
    ap.add_argument("--round", default="1")
    ap.add_argument("--lam", type=float, default=10.0)
    A = ap.parse_args()
    path = os.path.join(LDIR, f"colour_round{A.round}.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    sel = [r for r in rows if r["model"] in ARMS_8D]
    X = np.array([r["pca"][:8] for r in sel], dtype=np.float64)
    y = np.array([1.0 if r["grade"] >= 3 else 0.0 for r in sel])
    F = feats(X)

    # 5 折 out-of-fold 分数 → 诚实一点的 AUC
    idx = np.random.default_rng(1).permutation(len(y))
    sc = np.zeros(len(y))
    for i in range(5):
        te = idx[i::5]
        tr = np.setdiff1d(idx, te)
        sc[te] = F[te] @ ridge(F[tr], y[tr], A.lam)
    cv_auc = auc(sc, y)

    w = ridge(F, y, A.lam)
    out = {"target": "grade>=3", "kind": "linear", "n_terms": 9,
           "weights": [round(float(v), 6) for v in w],
           "coef_names": ["const"] + [f"PC{i}" for i in range(8)],
           "n": int(len(y)), "positive_rate": round(float(y.mean()), 4),
           "cv_auc": round(cv_auc, 4),
           "in_sample_auc": round(auc(F @ w, y), 4),
           "lam": A.lam, "source": os.path.basename(path),
           "arms": list(ARMS_8D),
           "note": "分数越高越好；红石成本 = 8 次乘加 + 1 次常数（共 9 项）"}
    json.dump(out, open(os.path.join(LDIR, "scorer_good.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"n={len(y)} 基线好率={y.mean()*100:.1f}%  CV AUC={cv_auc:.3f}  权重={out['weights']}")
    print("wrote results/labels/scorer_good.json")


if __name__ == "__main__":
    main()

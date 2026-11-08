#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
彩色「安全区」试探：只看旋钮坐标，能不能预测一张脸会不会被接受？

数据：二期 `labels.jsonl` 里四个彩色轮的 1,108 条人工标签（round 6/8/11/12），
      每条都带 8 个颜色 PCA 旋钮坐标（`pca` 字段前 8 个，后 14 个是补零的）。
判据：grade >= 2 记为可接受（与二期「可接受 = 类二次元/类游戏/类三次元」同口径）。

做四件事：
  1. 线性（9 项）与二次（45 项）打分器的 5 折 CV AUC；
  2. **留一轮交叉验证**（训练 3 轮、测第 4 轮）——这是"换模型还能不能用"的诚实检验；
  3. 在 N(0,1) 上采样，给出「覆盖率 vs 预测良率」曲线（用于和矩形规则比）；
  4. 出一张 PC0-PC1 截面上的安全区热图（叠加真实标签点）。

产物：results/safe/safe.json、results/safe/safe_map.png
用法：python probe_safe.py
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
REPO = os.path.dirname(P4)
P2 = os.path.join(REPO, "02_phase2_avatar")
OUT = os.path.join(P4, "results", "safe")
os.makedirs(OUT, exist_ok=True)
COLOUR_ROUNDS = {"6", "8", "11", "12"}


def font(sz):
    for fn in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def load_labels():
    rows = [json.loads(l) for l in
            open(os.path.join(P2, "results", "preference", "labels.jsonl"), encoding="utf-8")
            if l.strip()]
    X, y, rnd, mdl = [], [], [], []
    for r in rows:
        if str(r.get("round")) not in COLOUR_ROUNDS:
            continue
        pca = r.get("pca") or []
        if len(pca) < 8:
            continue
        X.append([float(v) for v in pca[:8]])
        y.append(1 if r["grade"] >= 2 else 0)
        rnd.append(str(r.get("round")))
        mdl.append(r.get("model"))
    return np.array(X), np.array(y), np.array(rnd), np.array(mdl)


def feats(X, kind):
    cols = [np.ones((len(X), 1)), X]
    if kind == "quad":
        cols.append(X ** 2)
        iu = np.triu_indices(X.shape[1], 1)
        cols.append((X[:, iu[0]] * X[:, iu[1]]))
    return np.hstack(cols)


def fit_ridge(F, y, lam=1.0):
    w = np.linalg.solve(F.T @ F + lam * np.eye(F.shape[1]), F.T @ (y * 2.0 - 1.0))
    return w


def auc(score, y):
    y = np.asarray(y)
    if y.sum() in (0, len(y)):
        return float("nan")
    o = np.argsort(score)
    r = np.empty(len(score))
    r[o] = np.arange(1, len(score) + 1)
    p = y == 1
    return float((r[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * (~p).sum()))


def kfold(n, k, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::k] for i in range(k)]


def main():
    X, y, rnd, mdl = load_labels()
    print(f"彩色标签 {len(y)} 条：可接受 {int(y.sum())}（{y.mean()*100:.1f}%），"
          f"轮次 {dict((r, int((rnd==r).sum())) for r in sorted(set(rnd)))}")

    out = {"n": int(len(y)), "accept_rate": round(float(y.mean()), 4)}
    for kind in ("linear", "quad"):
        F = feats(X, kind)
        # 5 折 CV
        sc = np.zeros(len(y))
        for te in kfold(len(y), 5, seed=1):
            tr_i = np.setdiff1d(np.arange(len(y)), te)
            w = fit_ridge(F[tr_i], y[tr_i])
            sc[te] = F[te] @ w
        cv_auc = auc(sc, y)
        # 留一轮
        lo = {}
        for r in sorted(set(rnd)):
            te = np.where(rnd == r)[0]
            tr_i = np.setdiff1d(np.arange(len(y)), te)
            if len(set(y[tr_i])) < 2 or len(set(y[te])) < 2:
                lo[r] = None
                continue
            w = fit_ridge(F[tr_i], y[tr_i])
            lo[r] = round(auc(F[te] @ w, y[te]), 4)
        w_all = fit_ridge(F, y)
        out[kind] = {"n_terms": int(F.shape[1]), "cv_auc": round(cv_auc, 4),
                     "in_sample_auc": round(auc(F @ w_all, y), 4),
                     "leave_one_round_out": lo,
                     "weights": np.round(w_all, 4).tolist()}
        print(f"  {kind:6s} {F.shape[1]:2d} 项: CV AUC {cv_auc:.3f}  "
              f"样本内 {auc(F @ w_all, y):.3f}  留一轮 {lo}")

    # 覆盖率 vs 预测良率（在 N(0,1) 上）
    rng = np.random.default_rng(0)
    Z = np.clip(rng.standard_normal((40000, 8)), -3.5, 3.5)
    Fq = feats(Z, "quad")
    w = np.array(out["quad"]["weights"])
    s = Fq @ w
    order = np.argsort(-s)
    cov_curve = []
    for cov in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
        k = int(cov * len(Z))
        cov_curve.append({"coverage": cov,
                          "pred_accept_in_top": round(float((s[order[:k]] > 0).mean()), 4)})
    out["cov_curve_predicted"] = cov_curve
    print("  预测：取分数最高的 x% 时，其中预测为可接受的比例：")
    for c in cov_curve:
        print(f"    {c['coverage']*100:5.0f}% → {c['pred_accept_in_top']*100:5.1f}%")

    # 按轮次的接受率（看标签本身的漂移）
    out["accept_by_round"] = {r: round(float(y[rnd == r].mean()), 3) for r in sorted(set(rnd))}
    out["accept_by_model"] = {m: round(float(y[mdl == m].mean()), 3) for m in sorted(set(mdl))}
    print(f"  各轮接受率 {out['accept_by_round']}")

    json.dump(out, open(os.path.join(OUT, "safe.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ---- 图：PC0-PC1 截面的安全区，叠加真实标签
    LEV = np.linspace(-3.5, 3.5, 29)
    gx, gy = np.meshgrid(LEV, LEV)
    G = np.zeros((gx.size, 8))
    G[:, 0] = gx.ravel()
    G[:, 1] = gy.ravel()
    sg = feats(G, "quad") @ w
    sg = sg.reshape(len(LEV), len(LEV))
    TILE, LW = 18, 300
    size = len(LEV) * TILE
    W, H = LW + size + 40, size + 120
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((14, 12), "彩色旋钮空间的「安全区」：PC0-PC1 截面（其余 6 维取 0）", font=font(19),
            fill=(234, 238, 244))
    dr.text((14, 40), "底色＝二次打分器的分数（绿=预测可接受，红=预测不可接受）；"
                     "圆点＝真实标签（白=可接受，黑=不可接受）。", font=font(11),
            fill=(148, 160, 176))
    dr.text((14, 58), f"1,108 条彩色标签，5 折 CV AUC {out['quad']['cv_auc']:.3f}，"
                      f"留一轮验证见 safe.json。", font=font(11), fill=(130, 142, 158))
    for i in range(len(LEV)):
        for j in range(len(LEV)):
            t = float(np.clip(sg[i, j] / 0.8, -1, 1))       # 归一到 [-1,1]
            col = (int(30 + 200 * max(0.0, -t)),            # 红：不可接受
                   int(30 + 200 * max(0.0, t)),             # 绿：可接受
                   50)
            dr.rectangle([LW + j * TILE, 90 + i * TILE, LW + (j + 1) * TILE - 1,
                          90 + (i + 1) * TILE - 1], fill=col)
    # 叠加标签点（前两维）
    for xi, yi, yi_lab in zip(X[:, 0], X[:, 1], y):
        j = int((xi + 3.5) / 7.0 * (len(LEV) - 1))
        i = int((yi + 3.5) / 7.0 * (len(LEV) - 1))
        cx, cy = LW + j * TILE + TILE // 2, 90 + i * TILE + TILE // 2
        c = (255, 255, 255) if yi_lab else (25, 25, 25)
        dr.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=c)
    dr.text((14, H - 26), "横轴 PC0，纵轴 PC1（其余 6 维固定在均值）；点的大小＝重叠程度，不是权重。",
            font=font(11), fill=(140, 152, 168))
    p = os.path.join(OUT, "safe_map.png")
    cv.save(p)
    print(f"wrote {os.path.relpath(p, REPO)}")
    print("SAFE_DONE")


if __name__ == "__main__":
    main()

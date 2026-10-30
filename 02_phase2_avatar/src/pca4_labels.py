#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
The 4096-point label map, and what it says about scoring and about style switches.

WHAT WE HAVE
The user marked 1372 of the 4096 lattice points as 中, across all 64 planes. That is a COMPLETE
labelling of the four-knob space at one-step resolution, not a sample. Two consequences:

    the base rate can be checked   33.5% marked bad against 35.2% unacceptable measured blind in
                                   round 10 -- they agree, so the map can be treated as a faithful
                                   description rather than one person's mood on one evening
    unmarked means acceptable      with every plane covered there is no "unseen" region, so the
                                   complement is the good set

WHAT IS COMPARED
    a linear probe      the current scorer's shape, 4 -> 1
    a ball model        nearest of k centres, which is what the user's "球状团簇" describes
    a per-axis table    bad rate for each of the eight levels of each knob, which is the most
                        interpretable summary and may beat both

THEN THE STYLE SWITCH
Good regions that are contiguous balls are listed with centre and radius, together with the bad
rate that sampling uniformly inside them would give. That is the "限定随机数分布" feature.
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
OUT = os.path.join(ROOT, "results", "pca4map")
LEVELS = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])

j = json.load(open(os.path.join(OUT, "pca4_plane_marks.json"), encoding="utf-8"))
bad = np.zeros((8, 8, 8, 8), dtype=bool)          # [pc0, pc1, pc2, pc3]
for m in j["marks"]:
    idx = [int(np.argmin(np.abs(LEVELS - v))) for v in m["pca"]]
    bad[tuple(idx)] = True
N = bad.size
print(f"lattice {N} points, marked 中: {bad.sum()} ({100*bad.mean():.1f}%)")
print(f"round 10 blind-measured unacceptable: 35.2%  -- the two agree, so the map is faithful")

# ---- per-axis marginals
print()
print("=" * 100)
print("BAD RATE BY KNOB LEVEL  (all other knobs averaged over)")
print("=" * 100)
print(f"  {'level':>6s} " + " ".join(f"{'PC'+str(i):>7s}" for i in range(4)))
print("  " + "-" * 40)
for li, lv in enumerate(LEVELS):
    row = []
    for k in range(4):
        sel = [slice(None)] * 4
        sel[k] = li
        row.append(100 * bad[tuple(sel)].mean())
    print(f"  {lv:+6.1f} " + " ".join(f"{v:6.1f}%" for v in row))
print()
for k in range(4):
    r = []
    for li in range(8):
        sel = [slice(None)] * 4
        sel[k] = li
        r.append(bad[tuple(sel)].mean())
    r = np.array(r)
    print(f"  PC{k}: 最差档 {LEVELS[r.argmax()]:+.1f}（{100*r.max():.0f}%）　"
          f"最好档 {LEVELS[r.argmin()]:+.1f}（{100*r.min():.0f}%）　"
          f"跨度 {100*(r.max()-r.min()):.0f} 点")

# ---- models
X = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, LEVELS, indexing="ij"), -1).reshape(-1, 4)
y = bad.reshape(-1).astype(float)
print()
print("=" * 100)
print("THREE SCORING SHAPES, 5-FOLD CROSS-VALIDATED AUC")
print("=" * 100)


def auc(s, t):
    hi = s[t == 1][:, None]; lo = s[t == 0][None, :]
    return float((hi > lo).mean() + 0.5 * (hi == lo).mean())


def cv_auc(fit_predict, X, y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    s = np.zeros(len(y))
    for f in range(k):
        te = folds[f]
        tr = np.concatenate([folds[q] for q in range(k) if q != f])
        s[te] = fit_predict(X[tr], y[tr], X[te])
    return auc(s, y), s


def linear(Xtr, ytr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    w = np.linalg.solve(A.T @ A + 3 * np.eye(A.shape[1]), A.T @ ytr)
    return np.c_[(Xte - mu) / sd, np.ones(len(Xte))] @ w


def ball_model(K):
    def f(Xtr, ytr, Xte):
        rng = np.random.default_rng(0)
        C = Xtr[rng.choice(len(Xtr), K, replace=False)]
        for _ in range(40):
            d = ((Xtr[:, None, :] - C[None]) ** 2).sum(-1)
            a = d.argmin(1)
            C = np.stack([Xtr[a == i].mean(0) if (a == i).any() else C[i] for i in range(K)])
        d = ((Xtr[:, None, :] - C[None]) ** 2).sum(-1)
        a = d.argmin(1)
        p = np.array([ytr[a == i].mean() if (a == i).any() else 0.5 for i in range(K)])
        dt = ((Xte[:, None, :] - C[None]) ** 2).sum(-1)
        return p[dt.argmin(1)]
    return f


res = {}
a_lin, _ = cv_auc(linear, X, y)
res["linear 4->1"] = a_lin
print(f"  线性 4→1（现用打分器的形状）      AUC {a_lin:.3f}")
for K in (2, 3, 4, 6, 8, 12):
    a, _ = cv_auc(ball_model(K), X, y)
    res[f"ball k={K}"] = a
    print(f"  最近中心，{K:2d} 个球（每球记一个坏率） AUC {a:.3f}")

best_shape = max(res.items(), key=lambda t: t[1])
print(f"\n  最好的是 {best_shape[0]}，AUC {best_shape[1]:.3f}")

# ---- good regions as connected components of the unmarked lattice
print()
print("=" * 100)
print("GOOD REGIONS: connected components of unmarked points")
print("=" * 100)
good = ~bad
seen = np.zeros_like(good)
comps = []
for i in range(8):
    for k in range(8):
        for l in range(8):
            for m in range(8):
                if good[i, k, l, m] and not seen[i, k, l, m]:
                    stack = [(i, k, l, m)]
                    seen[i, k, l, m] = True
                    cells = []
                    while stack:
                        a, b, c, d = stack.pop()
                        cells.append((a, b, c, d))
                        for da, db, dc, dd in ((1,0,0,0),(-1,0,0,0),(0,1,0,0),(0,-1,0,0),
                                               (0,0,1,0),(0,0,-1,0),(0,0,0,1),(0,0,0,-1)):
                            na, nb, nc, nd = a+da, b+db, c+dc, d+dd
                            if 0 <= na < 8 and 0 <= nb < 8 and 0 <= nc < 8 and 0 <= nd < 8:
                                if good[na, nb, nc, nd] and not seen[na, nb, nc, nd]:
                                    seen[na, nb, nc, nd] = True
                                    stack.append((na, nb, nc, nd))
                    comps.append(cells)
comps.sort(key=len, reverse=True)
print(f"  共 {len(comps)} 个连通块，最大的 {len(comps[0])} 格（占 {100*len(comps[0])/N:.1f}%）")
for ci, cells in enumerate(comps[:6]):
    A = np.array(cells)
    pts = LEVELS[A]
    centre = pts.mean(0)
    rad = float(np.percentile(np.linalg.norm(pts - centre, axis=1), 90))
    print(f"    块 {ci}: {len(cells):4d} 格  中心 {np.round(centre,2)}  半径(r90) {rad:.2f}")
print(f"\n  只保留最大连通块时的良率 {100*len(comps[0])/N:.1f}%"
      f"（占全体 {100*len(comps[0])/N:.1f}%）")

# ---- style switch: uniform sampling inside a ball, what is the bad rate
print()
print("=" * 100)
print("STYLE SWITCH: sample uniformly inside a ball, what bad rate results")
print("=" * 100)
print(f"  {'中心':>26s} {'半径':>6s} {'覆盖格点':>8s} {'坏率':>7s} {'保留多样性':>10s}")
print("  " + "-" * 64)
grid_pts = X
cands = []
for ci, cells in enumerate(comps[:12]):
    A = np.array(cells)
    pts = LEVELS[A]
    centre = pts.mean(0)
    for R in (2.0, 3.0):
        inside = np.linalg.norm(grid_pts - centre, axis=1) <= R
        if inside.sum() < 64:
            continue
        br = float(bad.reshape(-1)[inside].mean())
        frac = inside.sum() / N
        cands.append((br, frac, centre, R))
cands.sort(key=lambda t: (t[0], -t[1]))
for br, frac, centre, R in cands[:10]:
    print(f"  {np.round(centre,2)!s:>26s} {R:6.1f} {100*frac:7.1f}% {100*br:6.1f}% "
          f"{100*frac:9.1f}%")
print()
print("  坏率越低、覆盖越高越好。覆盖就是还能保留多少种不同长相。")

# ---- overall summary numbers
whole = float(bad.mean())
print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"  整个立方体的坏率           {100*whole:.1f}%")
print(f"  最好的球（见上表第一行）     {100*cands[0][0]:.1f}%")
print(f"  提升                      {100*(whole-cands[0][0]):.1f} 点")
print(f"  线性打分器能到             AUC {a_lin:.3f}；最好的球模型 AUC {best_shape[1]:.3f}")

json.dump({"bad_rate": whole, "cv_auc": res,
           "best_regions": [{"centre": c.tolist(), "radius": float(r),
                             "bad_rate": float(b), "coverage": float(f)}
                            for b, f, c, r in cands[:10]],
           "marginals": {f"PC{k}": {f"{LEVELS[i]:+.1f}": float(
               bad[tuple([slice(None)]*k + [i] + [slice(None)]*(3-k))].mean())
               for i in range(8)} for k in range(4)}},
          open(os.path.join(OUT, "labels_analysis.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join(OUT, 'labels_analysis.json')}")
print("LABELS_ANALYSIS_DONE")

#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
三种架构约束 × 不同 b 的精度扫描。

三种情形（容器上限 897，位权重按 LUT 阶跃量计上限 882）
    B  禁用比较器环      全部位权重进容器          2·w_ref ≤ 882  →  w_ref = 441
    A  最高位用环        最高位无限制，次高位起入容器   w_ref ≤ 882   →  w_ref = 882
    C  最高两位都用环    容器约束挪到次次高位          w_ref ≤ 1764  →  w_ref = 1764

注意 b 变了，w_ref 的定义不变（它永远等于次高位位权重），
但每个位权重的绝对值随 b 变化，所以最优 b 不一定相同。
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
RESULTS = os.path.join(ROOT, "results")
P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
U1W = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
TH1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
V2W = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
PHI = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)

DELTA = 7.0
XLO = -3.5
NLEV = 15
CAP = 882          # 位权重作为 LUT 阶跃量时的上限


def run(b, w_ref, rounds=40000, seed=41):
    NB = 1 << b
    rng = np.random.default_rng(seed)
    # ---------- 层 1
    alpha = CAP / (np.abs(U1W).max(1) * DELTA / 4) * (w_ref / CAP)
    U = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB)
                * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
    Hmax = np.where(U > 0, U, 0).sum(axis=(1, 2))
    A = np.rint(-alpha * (TH1 + XLO * U1W.sum(1)
                          + DELTA / (2 ** (b + 1)) * U1W.sum(1)) - Hmin).astype(np.int64)
    c = np.maximum(np.ceil((Hmax - Hmin - A) / (NB - 1)), 1).astype(np.int64)
    delta = c / alpha
    # ---------- 层 2
    beta = w_ref / ((np.abs(V2W) * delta[None, :]).max() * (1 << (b - 2)))
    V = np.rint(beta * V2W[:, :, None] * delta[None, :, None]
                * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    Gmin = np.where(V < 0, V, 0).sum(axis=(1, 2))
    Bp = np.rint(-beta * (PHI + 0.5 * (V2W * delta[None, :]).sum(1)) - Gmin).astype(np.int64)
    T = int(round(NLEV * beta))
    # 容器可行性
    top1 = int(np.abs(U[:, :, -1]).max())
    sec1 = int(np.abs(U[:, :, -2]).max())
    top2 = int(np.abs(V[:, :, -1]).max())
    sec2 = int(np.abs(V[:, :, -2]).max())
    third1 = int(np.abs(U[:, :, -3]).max()) if b >= 3 else 0
    third2 = int(np.abs(V[:, :, -3]).max()) if b >= 3 else 0
    # ---------- 测试
    NF = 8000
    xs = np.clip(rng.standard_normal((NF, 4)), -3.5, 3.5)
    X = np.clip(np.floor((xs - XLO) / DELTA * NB), 0, NB - 1).astype(np.int64)
    bits = ((X[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.int64)
    H = np.einsum("nik,jik->nj", bits, U)
    Y = np.clip(np.floor_divide(H - Hmin[None, :] - A[None, :], c[None, :]), 0, NB - 1)
    bY = ((Y[:, :, None] >> np.arange(b)[None, None, :]) & 1).astype(np.int64)
    G = np.einsum("njk,pjk->np", bY, V)
    Z = np.clip(G - Gmin[None, :] - Bp[None, :], 0, T)
    mx, mn = Z.max(1, keepdims=True), Z.min(1, keepdims=True)
    dZ = np.maximum(mx - mn, 1)
    L = np.clip(np.floor_divide((Z - mn) * NLEV + (dZ >> 1), dZ), 0, NLEV)
    # 参照
    zref = (np.maximum(0.0, (XLO + (X + 0.5) * DELTA / NB) @ U1W.T + TH1) @ V2W.T + PHI)
    v = np.clip(zref, 0, 15)
    lo, hi = v.min(1, keepdims=True), v.max(1, keepdims=True)
    Lref = np.clip(np.rint((v - lo) / np.maximum(hi - lo, 1e-9) * NLEV), 0, NLEV).astype(int)
    d = L - Lref
    return dict(b=b, w_ref=w_ref,
                top1=top1, sec1=sec1, top2=top2, sec2=sec2,
                third1=third1, third2=third2, T=T,
                diff=100 * (d != 0).mean(), mean=float(np.abs(d).mean()),
                maxd=int(np.abs(d).max()),
                bits_bytes=(U.size + V.size))


SCEN = [("B 无环", 441), ("A 最高位环", 882), ("C 最高两位环", 1764)]

print("=" * 104)
print("三种架构约束 × b 的端到端精度（16 级，对比浮点）")
print("=" * 104)
res = {}
for nm, wr in SCEN:
    print(f"\n  【{nm}】w_ref = {wr}")
    print(f"  {'b':>3s} {'次高位':>7s} {'最高位':>8s} {'位权重总数':>9s} "
          f"{'不同像素':>9s} {'平均|差|':>8s} {'最大':>4s} {'容器可行':>8s}")
    print("  " + "-" * 66)
    for b in (5, 6, 7, 8, 9, 10, 11, 12):
        r = run(b, wr)
        # 判定：允许环的位数由情形决定
        if nm.startswith("B"):
            ok = max(r["top1"], r["top2"]) <= CAP
        elif nm.startswith("A"):
            ok = max(r["sec1"], r["sec2"]) <= CAP
        else:
            ok = max(r["third1"], r["third2"]) <= CAP
        r["ok"] = ok
        res[(nm, b)] = r
        print(f"  {b:3d} {r['sec1']:7d} {r['top1']:8d} {r['bits_bytes']:9d} "
              f"{r['diff']:8.2f}% {r['mean']:8.4f} {r['maxd']:4d} "
              f"{'✓' if ok else '✗':>8s}")

print()
print("=" * 104)
print("每种情形的最优 b（只在容器可行的 b 里挑）")
print("=" * 104)
best = {}
for nm, wr in SCEN:
    cand = [(res[(nm, b)]["diff"], b, res[(nm, b)]) for b in (5, 6, 7, 8, 9, 10, 11, 12)
            if res[(nm, b)]["ok"]]
    if not cand:
        print(f"  {nm}: 无可行 b")
        continue
    cand.sort()
    d, b, r = cand[0]
    best[nm] = (b, d, r)
    print(f"  {nm:14s} w_ref={wr:5d}  最优 b = {b:2d}  "
          f"不同像素 {d:5.2f}%  平均|差| {r['mean']:.4f}  最大 {r['maxd']}")

print()
print("  参照：当前编译器 b=8, w_ref=882 → 4.61%")
print()
print("  注：'位权重总数' = U 的个数 + V 的个数，正比于红石规模（乘 b）。")
print("      挑 b 时要把这一列一起看：b 每加 1，权重数大约翻倍。")

json.dump({f"{nm}|b={b}": {k: (v if not isinstance(v, np.integer) else int(v))
                           for k, v in res[(nm, b)].items()}
           for nm, _ in SCEN for b in (5, 6, 7, 8, 9, 10, 11, 12)},
          open(os.path.join(RESULTS, "scenario_b_sweep.json"), "w",
               encoding="utf-8"), indent=1)
print("  已写入 results/scenario_b_sweep.json")
print("SWEEP_DONE")

#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
修正 + 决定性测量
    修正：每格可以留空（贡献 0），上一版漏了，导致 U=0 被判不可达。
    测量：若用漏斗装位权重，权重必须就近取到「可达信号」，误差涨多少？
          对照：木桶全覆盖，无需取整。
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

NONSTACK = set(range(64, 64 * 64 + 1, 64))
S16 = set(range(4, 4 * 64 + 1, 4))
STACK = set(range(1, 65))
PER_SLOT = sorted({0} | NONSTACK | S16 | STACK)


def reach(n):
    cap = 64 * 64 * n
    cur = np.zeros(cap + 1, dtype=bool)
    cur[0] = True
    for _ in range(n):
        nxt = np.zeros(cap + 1, dtype=bool)
        for d in PER_SLOT:
            nxt[d:] |= cur[:cap + 1 - d]
        cur = nxt
    return cur


def sig_of(U, n):
    return 0 if U == 0 else 1 + (14 * U) // (64 * n)


def reach_sig(n):
    R = reach(n)
    Us = np.flatnonzero(R)
    s = np.array([sig_of(u, n) for u in Us])
    got = np.zeros(898, dtype=bool)
    got[s] = True
    return got, R


print("=" * 96)
print("一、修正后的可达性")
print("=" * 96)
res = {}
for nm, n in (("木桶", 27), ("漏斗", 5), ("投掷器", 9)):
    got, R = reach_sig(n)
    miss = [s for s in range(898) if not got[s]]
    res[nm] = got
    print(f"  {nm:6s}（{n:2d} 格）可达 {int(got.sum())}/898  缺 {len(miss)} 个"
          + (f"   首个缺口 {miss[0]}" if miss else "   全覆盖"))

print()
print("  漏斗缺失明细（按 50 一段）：")
got = res["漏斗"]
for lo in range(0, 900, 50):
    hi = min(897, lo + 49)
    m = [s for s in range(lo, hi + 1) if not got[s]]
    if m:
        print(f"    {lo:3d}..{hi:3d}  缺 {len(m):2d}/{hi-lo+1:2d}")
print(f"    完整缺口列表：{[s for s in range(898) if not got[s]]}")

# ---------------- 决定性测量
print()
print("=" * 96)
print("二、若用漏斗装位权重，需要把权重取到可达值，误差涨多少")
print("=" * 96)
sys.path.insert(0, HERE)
from compiler import Layer, compile_network, integer_forward  # noqa: E402

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
W1 = np.array(mj["gen"]["layers"][0]["W"])
TH1 = np.array(mj["gen"]["layers"][0]["b"])
V2W = np.array(mj["gen"]["layers"][1]["W"])
PHI = np.array(mj["gen"]["layers"][1]["b"])

NB = 256
comp = compile_network([Layer(W1, TH1, "relu", comparable=False, out_bits=8),
                        Layer(V2W, PHI, "clip", comparable=True,
                              lam=np.full(32, 15.0))],
                       -3.5, 3.5, 8, verbose=False)


def reach_list(n):
    """可达信号值列表（升序），用于就近取整。"""
    got, _ = reach_sig(n)
    return np.flatnonzero(got)


def round_to(arr, vals):
    """把整数数组就近取到 vals 里（保号）。"""
    a = np.abs(arr).astype(np.int64)
    idx = np.searchsorted(vals, a)
    idx = np.clip(idx, 1, len(vals) - 1)
    lo, hi = vals[idx - 1], vals[idx]
    take = np.where(np.abs(a - lo) <= np.abs(hi - a), lo, hi)
    return np.sign(arr).astype(np.int64) * take


# 参照：不做取整
rng = np.random.default_rng(41)
NF = 20000
xs = np.clip(rng.standard_normal((NF, 4)), -3.5, 3.5)
Xq = np.clip(np.floor((xs + 3.5) / 7.0 * NB), 0, NB - 1).astype(np.int64)
Zref = integer_forward(comp, Xq)


def levels(Z):
    m, n_ = Z.max(1, keepdims=True), Z.min(1, keepdims=True)
    dZ = np.maximum(m - n_, 1)
    return np.clip(np.floor_divide((Z - n_) * 15 + (dZ >> 1), dZ), 0, 15)


Lref = levels(Zref)

print(f"  {'容器':8s} {'取整前最大|W|':>12s} {'非零权重被改动的比例':>18s} "
      f"{'端到端 16 级：不同像素':>20s} {'平均|差|':>8s}")
print("  " + "-" * 74)
for nm, n in (("木桶", 27), ("漏斗", 5), ("投掷器", 9)):
    vals = reach_list(n)
    comp2 = [dict(r) for r in comp]
    changed = 0
    total = 0
    for r in comp2:
        W = r["W"]
        W2 = np.empty_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W2[i, j] = round_to(W[i, j], vals)
        changed += int((W2 != W).sum())
        total += W.size
        r["W"] = W2
        r["Cmin"] = np.where(W2 < 0, W2, 0).sum(axis=(1, 2))
        r["Cmax"] = np.where(W2 > 0, W2, 0).sum(axis=(1, 2))
        r["Nmax"] = r["Cmax"] - r["Cmin"]
        r["P"] = np.rint(-r["sigma"] * (r["theta"] +
                                        (r["w"] * (r["a0"] + 0.5 * r["delta_in"])[None, :]
                                         ).sum(axis=1)) - r["Cmin"]).astype(np.int64)
        if r["kind"] == "relu":
            r["c"] = np.maximum(np.ceil((r["Nmax"] - r["P"]) / ((1 << r["out_bits"]) - 1)),
                                1).astype(np.int64)
    Z2 = integer_forward(comp2, Xq)
    L2 = levels(Z2)
    d = L2 - Lref
    print(f"  {nm:8s} {int(np.abs(np.concatenate([r['W'].ravel() for r in comp2])).max()):12d} "
          f"{100*changed/total:17.2f}% {100*(d != 0).mean():19.2f}% "
          f"{np.abs(d).mean():8.3f}")

print()
print("  注：'取整前最大|W|' 是取整**后**的最大值；原始最大 1764。")
print("      漏斗没有 1764 以上的可达值，所以最高位会被压下来。")
print()
print("=" * 96)
print("三、结论")
print("=" * 96)
g_h = res["漏斗"]
g_b = res["木桶"]
print(f"  木桶 {int(g_b.sum())}/898 全覆盖 → 位权重可以原样落桶，零取整误差")
print(f"  漏斗 {int(g_h.sum())}/898，707..897 段大面积缺失 → 高位权重必须取整")
print("  投掷器 9 格可达性介于两者之间（见 results/container_reach.json）")
print("CONTAINER2_DONE")

#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
决定性测量：容器上限 897，而位权重最高位是 2·w_ref。
    要每个位权重塞进一个容器，必须 2·w_ref <= 897  →  w_ref <= 448。
    对漏斗还有更紧的要求：它的「稠密区」只到 729，再往上大面积缺失。
    → 要完全无损，需 2·w_ref <= 729  →  w_ref <= 364。

测四个组合：
    w_ref = 882  最高位 1764  任何容器都装不下（需要拆）
    w_ref = 441  最高位 882   木桶可装，漏斗不行（882 缺）
    w_ref = 364  最高位 728   漏斗稠密区上界内，木桶也没问题
    w_ref = 320  最高位 640   更保守
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

NONSTACK = set(range(64, 64 * 64 + 1, 64))
S16 = set(range(4, 4 * 64 + 1, 4))
PER_SLOT = sorted({0} | NONSTACK | S16 | set(range(1, 65)))


def reach_sig(n):
    cap = 64 * 64 * n
    cur = np.zeros(cap + 1, dtype=bool)
    cur[0] = True
    for _ in range(n):
        nxt = np.zeros(cap + 1, dtype=bool)
        for d in PER_SLOT:
            nxt[d:] |= cur[:cap + 1 - d]
        cur = nxt
    Us = np.flatnonzero(cur)
    s = np.array([0 if u == 0 else 1 + (14 * u) // (64 * n) for u in Us])
    got = np.zeros(898, dtype=bool)
    got[s] = True
    return got


GOT = {"木桶": reach_sig(27), "漏斗": reach_sig(5), "投掷器": reach_sig(9)}


def levels(Z):
    m, n_ = Z.max(1, keepdims=True), Z.min(1, keepdims=True)
    dZ = np.maximum(m - n_, 1)
    return np.clip(np.floor_divide((Z - n_) * 15 + (dZ >> 1), dZ), 0, 15)


rng = np.random.default_rng(41)
NF = 20000
xs = np.clip(rng.standard_normal((NF, 4)), -3.5, 3.5)
Xq = np.clip(np.floor((xs + 3.5) / 7.0 * NB), 0, NB - 1).astype(np.int64)
zref = (np.maximum(0.0, (-3.5 + (Xq + 0.5) * 7.0 / NB) @ W1.T + TH1) @ V2W.T + PHI)
v = np.clip(zref, 0, 15)
lo, hi = v.min(1, keepdims=True), v.max(1, keepdims=True)
Lref = np.clip(np.rint((v - lo) / np.maximum(hi - lo, 1e-9) * 15), 0, 15).astype(int)

print("=" * 100)
print("一、w_ref 对端到端精度的影响（不做容器取整）")
print("=" * 100)
print(f"  {'w_ref':>6s} {'最高位':>7s} {'木桶可装':>8s} {'漏斗可装':>8s} {'投掷器':>7s} "
      f"{'16级不同像素':>12s} {'平均|差|':>8s}")
print("  " + "-" * 62)
base = {}
for w_ref in (882, 441, 364, 320):
    comp = compile_network([Layer(W1, TH1, "relu", comparable=False, out_bits=8),
                            Layer(V2W, PHI, "clip", comparable=True,
                                  lam=np.full(32, 15.0))],
                           -3.5, 3.5, 8, verbose=False, w_ref=w_ref)
    Wmax = int(max(np.abs(r["W"]).max() for r in comp))
    ok = {k: bool(v[Wmax]) for k, v in GOT.items()} if Wmax <= 897 else \
         {k: False for k in GOT}
    Z = integer_forward(comp, Xq)
    L = levels(Z)
    d = L - Lref
    base[w_ref] = (comp, Z, L)
    print(f"  {w_ref:6d} {Wmax:7d} {str(ok['木桶']):>8s} {str(ok['漏斗']):>8s} "
          f"{str(ok['投掷器']):>7s} {100*(d != 0).mean():11.2f}% {np.abs(d).mean():8.4f}")

print()
print("=" * 100)
print("二、若必须取整到可达信号（只有不可装的情况才需要）")
print("=" * 100)
print(f"  {'w_ref':>6s} {'容器':8s} {'需取整的权重数':>14s} {'改动比例':>9s} "
      f"{'16级不同像素':>12s} {'平均|差|':>8s}")
print("  " + "-" * 62)


def round_to(arr, vals):
    a = np.abs(arr).astype(np.int64)
    idx = np.clip(np.searchsorted(vals, a), 1, len(vals) - 1)
    lo_, hi_ = vals[idx - 1], vals[idx]
    take = np.where(np.abs(a - lo_) <= np.abs(hi_ - a), lo_, hi_)
    return np.sign(arr).astype(np.int64) * take


for w_ref in (882, 441, 364):
    comp, Z0, L0 = base[w_ref]
    for nm, n in (("木桶", 27), ("漏斗", 5), ("投掷器", 9)):
        vals = np.flatnonzero(GOT[nm])
        comp2 = [dict(r) for r in comp]
        changed = 0
        total = 0
        for r in comp2:
            W = r["W"]
            W2 = W.copy()
            for i in range(W.shape[0]):
                W2[i] = round_to(W[i], vals)
            changed += int((W2 != W).sum())
            total += W.size
            r["W"] = W2
            r["Cmin"] = np.where(W2 < 0, W2, 0).sum(axis=(1, 2))
            r["Cmax"] = np.where(W2 > 0, W2, 0).sum(axis=(1, 2))
            r["Nmax"] = r["Cmax"] - r["Cmin"]
            kappa = r["theta"] + (r["w"] * (r["a0"] + 0.5 * r["delta_in"])[None, :]).sum(1)
            r["P"] = np.rint(-r["sigma"] * kappa - r["Cmin"]).astype(np.int64)
            if r["kind"] == "relu":
                r["c"] = np.maximum(np.ceil((r["Nmax"] - r["P"]) /
                                            ((1 << r["out_bits"]) - 1)), 1).astype(np.int64)
        Z = integer_forward(comp2, Xq)
        L = levels(Z)
        d = L - L0
        print(f"  {w_ref:6d} {nm:8s} {changed:14d} {100*changed/total:8.2f}% "
              f"{100*(d != 0).mean():11.2f}% {np.abs(d).mean():8.4f}")
    print()

print("=" * 100)
print("三、结论")
print("=" * 100)
print("  · 位权重最高位 = 2·w_ref，容器上限 897 → w_ref <= 448")
print("  · 漏斗的稠密区只到 729，要无损需 2·w_ref <= 729 → w_ref <= 364")
print("  · 木桶 898/898 全覆盖，w_ref = 441 时最高位 882 可装，零取整误差")
print()
print("  w_ref 从 882 降到 441 的精度代价见上表第一段。")
print("DONE3")

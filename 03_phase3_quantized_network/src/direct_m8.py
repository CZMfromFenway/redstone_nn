#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
最终版：每通道 8 个投掷器直接生成 X，权重全整数（1..15），偏置小整数。
无定点乘数、无缩放网络、无 4→4。

    X = clip( c + Σ_{k=1..8} w_k · v_k, 0, 255 )
    v_k ∈ {0,1,2}，格数 4/1/4（每个投掷器 9 格）
"""
import json
import os
import sys
import time
import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from compiler import Layer, compile_network, integer_forward  # noqa: E402

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(P2, "src"))
from layout import image_from_left_half  # noqa: E402

DELTA, NB, NLEV = 7.0, 256, 15
VARV, M = 0.8889, 8
PV = np.array([4, 1, 4]) / 9.0

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
MC = np.array(json.load(open(os.path.join(P2, "webapp", "merge_mc11.json"),
                             encoding="utf-8"))["_lut"])
styles = json.load(open(os.path.join(P2, "webapp", "styles_gauss.json"),
                        encoding="utf-8"))["styles"]
comp = compile_network([Layer(np.array(mj["gen"]["layers"][0]["W"]),
                              np.array(mj["gen"]["layers"][0]["b"]),
                              "relu", comparable=False, out_bits=8),
                        Layer(np.array(mj["gen"]["layers"][1]["W"]),
                              np.array(mj["gen"]["layers"][1]["b"]),
                              "clip", comparable=True, lam=np.full(32, 15.0))],
                       -3.5, 3.5, 8, verbose=False)


def int_levels(X):
    Z = integer_forward(comp, np.asarray(X, dtype=np.int64))
    m, n = Z.max(1, keepdims=True), Z.min(1, keepdims=True)
    dZ = np.maximum(m - n, 1)
    return np.clip(np.floor_divide((Z - n) * NLEV + (dZ >> 1), dZ), 0, NLEV)


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


def pmf_dense(w):
    cur = np.array([1.0])
    for wi in w:
        n = np.zeros(len(cur) + 2 * wi)
        n[:len(cur)] += cur * PV[0]
        n[wi:wi + len(cur)] += cur * PV[1]
        n[2 * wi:2 * wi + len(cur)] += cur * PV[2]
        cur = n
    return cur


def sc_of(w, c, tgt):
    pk = pmf_dense(w)
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


def best_c(w, m, tgt):
    b = (9.9, None)
    for c in range(int(round(m)) - int(w.sum()) - 3,
                   int(round(m)) - int(w.sum()) + 4):
        s = sc_of(w, c, tgt)
        if s < b[0]:
            b = (s, c)
    return b


def fit(tgt, m, s, rng, iters=2500):
    Q = s ** 2 / VARV
    cur = np.full(M, max(1, int(round(np.sqrt(Q / M)))), dtype=int)
    sc, c = best_c(cur, m, tgt)
    best = (sc, cur.copy(), c)
    for _ in range(iters):
        i, j = int(rng.integers(M)), int(rng.integers(M))
        if i == j or cur[j] <= 1:
            continue
        cand = cur.copy()
        cand[i] += 1
        cand[j] -= 1
        scc, cc = best_c(cand, m, tgt)
        if scc < best[0]:
            best = (scc, cand.copy(), cc)
            cur = cand
    return best


print("=" * 96)
print(f"每通道 {M} 个投掷器直接生成 X")
print("=" * 96)
t0 = time.time()
spec = {"M": M, "formula": "X = clip(c + sum(w_k*v_k), 0, 255)",
        "dropper": {"values": [0, 1, 2], "slots": [4, 1, 4]}, "styles": {}}
allsc = []
for e in styles:
    nm = e["name"]
    print(f"\n  {nm}")
    print(f"    {'ch':>2s} {'权重（8 个）':>42s} {'Σw':>4s} {'c':>5s} {'CDF':>7s}")
    spec["styles"][nm] = []
    for d in range(4):
        sc, w, c = fit(TARGET[nm][d], MS[nm][d], SD[nm][d], rng)
        allsc.append(sc)
        ws = sorted(int(v) for v in w)
        spec["styles"][nm].append({"w": ws, "c": int(c), "cdf": float(sc)})
        print(f"    {d+1:2d} {str(ws):>42s} {sum(ws):4d} {c:+5d} {sc:7.4f}")
print()
print(f"  24 组平均 CDF 差 {np.mean(allsc):.4f}   最大 {np.max(allsc):.4f}")
print(f"  权重范围 [{min(min(c['w']) for s in spec['styles'].values() for c in s)}, "
      f"{max(max(c['w']) for s in spec['styles'].values() for c in s)}]")
print(f"  偏置范围 [{min(c['c'] for s in spec['styles'].values() for c in s)}, "
      f"{max(c['c'] for s in spec['styles'].values() for c in s)}]")
print(f"  耗时 {time.time()-t0:.0f} s")

json.dump(spec, open(os.path.join(RESULTS, "direct_spec_m8.json"), "w",
                     encoding="utf-8"), indent=1)
print(f"\n  已写入 results/direct_spec_m8.json")

# ---------- 出图与平均脸
rngs = np.random.default_rng(3)


def sample_direct(nm, n, rng):
    out = np.zeros((n, 4), dtype=np.int64)
    for d, ch in enumerate(spec["styles"][nm]):
        v = rng.choice([0, 1, 2], size=(n, M), p=PV)
        out[:, d] = np.clip(ch["c"] + v @ np.array(ch["w"]), 0, NB - 1)
    return out


SC, PAD, GAP = 30, 6, 14
KSHOW = 4
Wc = KSHOW * 8 * SC + (KSHOW - 1) * PAD + 2 * GAP
Hc = len(styles) * (8 * SC + GAP) + GAP
cvv = np.full((Hc, Wc, 3), 18, dtype=np.uint8)
mref, mdir = [], []
for si, e in enumerate(styles):
    nm = e["name"]
    g = MC[int_levels(sample_direct(nm, KSHOW, rngs))]
    y0 = GAP + si * (8 * SC + GAP)
    for k in range(KSHOW):
        img = image_from_left_half(g[k])
        px = np.clip(img / 15.0 * 255, 0, 255).astype(np.uint8)
        big = np.repeat(np.repeat(px, SC, 0), SC, 1)
        xx = GAP + k * (8 * SC + PAD)
        cvv[y0:y0 + 8 * SC, xx:xx + 8 * SC] = np.stack([big, big, big], -1)
    if si:
        cvv[y0 - GAP:y0 - GAP + 3, GAP:Wc - GAP] = (70, 70, 70)
    # 平均脸
    mdir.append(MC[int_levels(sample_direct(nm, 600, rngs))].mean(0))
    mu0 = np.array(e["mu"], dtype=float)
    L0 = np.array(e["chol"], dtype=float)
    got = []
    while len(got) < 600:
        z = rngs.standard_normal((600, 4))
        x = z @ L0.T + mu0
        for r in x:
            if np.all(np.abs(r) <= 3.5):
                got.append(r)
                if len(got) >= 600:
                    break
    Xr = np.clip(np.floor((np.array(got) + 3.5) / DELTA * NB), 0, NB - 1).astype(np.int64)
    mref.append(MC[int_levels(Xr)].mean(0))

Image.fromarray(cvv).save(os.path.join(RESULTS, "direct_m8.png"))
print(f"  wrote results/direct_m8.png")

print()
print("=" * 96)
print("平均脸对照（600 张）：直接法 vs 原始拒绝采样")
print("=" * 96)
for e, a, b in zip(styles, mref, mdir):
    print(f"  {e['name']:10s} 最大差 {np.abs(a-b).max():.3f} 级   "
          f"平均差 {np.abs(a-b).mean():.3f} 级")
print("DIRECT_M8_DONE")

#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
最终：LAHC + N=9 形状族，导出参数，并画搜索轨迹对照图。
"""
import json
import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

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
FAM = [(a, b, 9 - a - b) for a in range(10) for b in range(10 - a)]
SP = np.array([np.array(s) / 9.0 for s in FAM])
SEV = SP @ np.array([0.0, 1.0, 2.0])
I414 = FAM.index((4, 1, 4))

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


def pmf_of(w, sh):
    cur = np.array([1.0])
    for wi, si in zip(w, sh):
        p = SP[si]
        n = np.zeros(len(cur) + 2 * wi)
        for j in range(3):
            if p[j] > 0:
                n[j * wi:j * wi + len(cur)] += cur * p[j]
        cur = n
    return cur


def cdf_dist(w, sh, c, tgt):
    pk = pmf_of(w, sh)
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


def bias_for(w, sh, m):
    return int(round(m - float(np.dot(w, SEV[sh]))))


def fit(tgt, m, s, M, rng, restarts=2, iters=2500, L=80, mode="lahc", trace=False):
    best = (9.9, None, None, None)
    stats = {"init": [], "final": [], "evals": 0, "acc": 0, "trace": []}
    for r in range(restarts):
        sh = np.full(M, I414, dtype=int)
        w = np.full(M, max(1, int(round(np.sqrt(s ** 2 / 0.8889 / M)))), dtype=int)
        if r > 0:
            w = np.maximum(1, w + rng.integers(-4, 5, size=M))
            sh = rng.integers(0, len(FAM), size=M)
        c = bias_for(w, sh, m)
        cur = cdf_dist(w, sh, c, tgt)
        stats["init"].append(cur)
        stats["evals"] += 1
        hist = [cur] * L
        tr = [cur]
        for it in range(iters):
            k = int(rng.integers(M))
            w2, sh2 = w.copy(), sh.copy()
            if rng.random() < 0.5:
                w2[k] = max(1, w2[k] + int(rng.integers(-2, 3)))
            else:
                sh2[k] = int(rng.integers(0, len(FAM)))
            c2 = bias_for(w2, sh2, m)
            v, cb = 9.9, c2
            for dc in (-1, 0, 1):
                stats["evals"] += 1
                vv = cdf_dist(w2, sh2, c2 + dc, tgt)
                if vv < v:
                    v, cb = vv, c2 + dc
            slot = it % L
            ok = (v < cur) if mode == "hc" else (v <= hist[slot] or v < cur)
            if ok:
                if mode != "hc":
                    hist[slot] = cur
                cur, w, sh, c = v, w2, sh2, cb
                stats["acc"] += 1
                if trace:
                    tr.append(cur)
                if cur < best[0]:
                    best = (cur, w.copy(), sh.copy(), c)
        stats["final"].append(cur)
        if trace:
            stats["trace"].append(tr)
    return best, stats


print("=" * 100)
print("最终搜索：LAHC，N=9 形状族，M=10")
print("=" * 100)
t0 = time.time()
spec = {"formula": "X = clip(c + sum(w_k*v_k), 0, 255)",
        "dropper": "values {0,1,2}, slot counts (a,b,c), a+b+c=9",
        "search": "LAHC, restarts=2, iters=2500, L=80", "styles": {}}
allsc = []
for e in styles:
    nm = e["name"]
    print(f"\n  {nm}")
    spec["styles"][nm] = []
    for d in range(4):
        (sc, w, sh, c), st = fit(TARGET[nm][d], MS[nm][d], SD[nm][d], 10, rng)
        allsc.append(sc)
        spec["styles"][nm].append({
            "M": 10, "c": int(c), "cdf": float(sc),
            "channels": [{"w": int(wi), "abc": list(FAM[si])} for wi, si in zip(w, sh)]})
        print(f"    ch{d+1}  c={c:+4d}  CDF={sc:.4f}   "
              f"初值 {min(st['init']):.3f} -> {st['final'][-1]:.4f}")
print()
print(f"  24 组平均 CDF {np.mean(allsc):.4f}  最大 {np.max(allsc):.4f}")
print(f"  权重范围 [{min(c['w'] for s in spec['styles'].values() for ch in s for c in ch['channels'])}, "
      f"{max(c['w'] for s in spec['styles'].values() for ch in s for c in ch['channels'])}]")
print(f"  偏置范围 [{min(ch['c'] for s in spec['styles'].values() for ch in s)}, "
      f"{max(ch['c'] for s in spec['styles'].values() for ch in s)}]")
print(f"  耗时 {time.time()-t0:.0f} s")
json.dump(spec, open(os.path.join(RESULTS, "final_spec.json"), "w",
                     encoding="utf-8"), indent=1)
print("  已写入 results/final_spec.json")

# ---------------- 轨迹对照：爬山 vs LAHC
print()
print("=" * 100)
print("轨迹对照：同一个通道、同一个初值，爬山 vs LAHC")
print("=" * 100)
(nm, d) = ("少女", 0)
rs = np.random.default_rng(1234)
_, st_hc = fit(TARGET[nm][d], MS[nm][d], SD[nm][d], 10, rs,
               restarts=1, iters=2500, mode="hc", trace=True)
rs = np.random.default_rng(1234)
_, st_la = fit(TARGET[nm][d], MS[nm][d], SD[nm][d], 10, rs,
               restarts=1, iters=2500, mode="lahc", trace=True)
for nm2, st in (("爬山", st_hc), ("LAHC", st_la)):
    print(f"  {nm2:5s} 初值 {st['init'][0]:.4f}  终值 {st['final'][0]:.4f}  "
          f"接受 {st['acc']:5d}/{st['evals']:6d}（{100*st['acc']/st['evals']:.1f}%）")

W, H = 1000, 420
im = Image.new("RGB", (W, H), (20, 20, 24))
dr = ImageDraw.Draw(im)
bx, by, bw, bh = 70, 40, 880, 300
ymax = max(max(st_hc["trace"][0]), max(st_la["trace"][0])) * 1.05
dr.rectangle([bx, by, bx + bw, by + bh], outline=(70, 70, 80))
for frac in (0.25, 0.5, 0.75, 1.0):
    y = by + bh - int(bh * frac)
    dr.line([(bx, y), (bx + bw, y)], fill=(40, 40, 48))
    dr.text((8, y - 6), f"{ymax*frac:.3f}", fill=(150, 150, 160))
dr.text((bx - 60, by - 26), "CDF 差", fill=(200, 200, 210))
dr.text((bx + bw - 90, by + bh + 10), "迭代", fill=(200, 200, 210))
for st, col, nm2 in ((st_hc, (120, 200, 255), "爬山（只接受更好）"),
                     (st_la, (255, 140, 120), "LAHC（可接受暂时变差）")):
    tr = st["trace"][0]
    for i in range(1, len(tr)):
        x1 = bx + int(bw * (i - 1) / 2500)
        x2 = bx + int(bw * i / 2500)
        y1 = by + bh - int(bh * min(1.0, tr[i - 1] / ymax))
        y2 = by + bh - int(bh * min(1.0, tr[i] / ymax))
        dr.line([(x1, y1), (x2, y2)], fill=col)
dr.text((bx, by + bh + 28), "  ", fill=(255, 255, 255))
dr.line([(bx + 10, by + bh + 40), (bx + 40, by + bh + 40)], fill=(120, 200, 255), width=2)
dr.text((bx + 46, by + bh + 34), "爬山：单调下降，很快停住", fill=(180, 200, 220))
dr.line([(bx + 300, by + bh + 40), (bx + 330, by + bh + 40)], fill=(255, 140, 120), width=2)
dr.text((bx + 336, by + bh + 34), "LAHC：会走上去，但最终低得多", fill=(220, 190, 180))
im.save(os.path.join(RESULTS, "search_compare.png"))
print(f"  wrote results/search_compare.png")

# ---------------- 出图
def sample(nm, n, rng):
    out = np.zeros((n, 4), dtype=np.int64)
    for d, ch in enumerate(spec["styles"][nm]):
        acc = np.full(n, ch["c"], dtype=np.int64)
        for c in ch["channels"]:
            acc += c["w"] * rng.choice([0, 1, 2], size=n, p=np.array(c["abc"]) / 9.0)
        out[:, d] = np.clip(acc, 0, NB - 1)
    return out


rngs = np.random.default_rng(7)
SC, PAD, GAP, K = 30, 6, 14, 4
Wc = K * 8 * SC + (K - 1) * PAD + 2 * GAP
Hc = len(styles) * (8 * SC + GAP) + GAP
cvv = np.full((Hc, Wc, 3), 18, dtype=np.uint8)
mref, mdir = [], []
for si, e in enumerate(styles):
    nm = e["name"]
    g = MC[int_levels(sample(nm, K, rngs))]
    y0 = GAP + si * (8 * SC + GAP)
    for k in range(K):
        img = image_from_left_half(g[k])
        px = np.clip(img / 15.0 * 255, 0, 255).astype(np.uint8)
        big = np.repeat(np.repeat(px, SC, 0), SC, 1)
        xx = GAP + k * (8 * SC + PAD)
        cvv[y0:y0 + 8 * SC, xx:xx + 8 * SC] = np.stack([big, big, big], -1)
    if si:
        cvv[y0 - GAP:y0 - GAP + 3, GAP:Wc - GAP] = (70, 70, 70)
    mdir.append(MC[int_levels(sample(nm, 600, rngs))].mean(0))
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
Image.fromarray(cvv).save(os.path.join(RESULTS, "final_faces.png"))
print(f"  wrote results/final_faces.png")
print()
print("=" * 88)
print("平均脸对照（600 张）")
print("=" * 88)
for e, a, b in zip(styles, mref, mdir):
    print(f"  {e['name']:10s} 最大差 {np.abs(a-b).max():.3f} 级  "
          f"平均差 {np.abs(a-b).mean():.3f} 级")
print("FINAL_DONE")

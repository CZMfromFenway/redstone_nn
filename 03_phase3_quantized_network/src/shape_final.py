#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""导出形状自由的最终方案 + 画拟合图 + 出图。"""
import json
import os
import sys
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
SHAPES = [(a, b, 9 - a - b) for a in range(10) for b in range(10 - a)]
SP = np.array([np.array(s) / 9.0 for s in SHAPES])

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
MC = np.array(json.load(open(os.path.join(P2, "webapp", "merge_mc11.json"),
                             encoding="utf-8"))["_lut"])
styles = json.load(open(os.path.join(P2, "webapp", "styles_gauss.json"),
                        encoding="utf-8"))["styles"]
spec = json.load(open(os.path.join(RESULTS, "direct_shape_spec.json"),
                      encoding="utf-8"))
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


def pmf_of(chs):
    cur = np.array([1.0])
    for ch in chs:
        wi = ch["w"]
        p = np.array(ch["abc"]) / 9.0
        n = np.zeros(len(cur) + 2 * wi)
        for j in range(3):
            if p[j] > 0:
                n[j * wi:j * wi + len(cur)] += cur * p[j]
        cur = n
    return cur


print("=" * 100)
print("形状自由的最终方案")
print("=" * 100)
tot = 0
allcdf = []
shp_use = {}
for nm, chans in spec["styles"].items():
    print(f"\n  {nm}")
    for d, ch in enumerate(chans):
        tot += ch["M"]
        allcdf.append(ch["cdf"])
        for c in ch["channels"]:
            k = tuple(c["abc"])
            shp_use[k] = shp_use.get(k, 0) + 1
        txt = " + ".join(f"{c['w']}x{c['abc']}" for c in ch["channels"])
        print(f"    ch{d+1}  M={ch['M']:2d}  c={ch['c']:+4d}  CDF={ch['cdf']:.4f}")
        print(f"          {txt}")
print()
print(f"  24 组平均 CDF 差 {np.mean(allcdf):.4f}  最大 {np.max(allcdf):.4f}")
print(f"  投掷器总数 {tot}（平均每通道 {tot/24:.1f} 个）")
print(f"  权重范围 [{min(c['w'] for s in spec['styles'].values() for ch in s for c in ch['channels'])}, "
      f"{max(c['w'] for s in spec['styles'].values() for ch in s for c in ch['channels'])}]")
print(f"  偏置范围 [{min(ch['c'] for s in spec['styles'].values() for ch in s)}, "
      f"{max(ch['c'] for s in spec['styles'].values() for ch in s)}]")
print()
print("  用到的形状（(a,b,c) 出现次数，前 10）：")
for k, v in sorted(shp_use.items(), key=lambda kv: -kv[1])[:10]:
    print(f"    {k}  {v:3d} 次   p={np.array(k)/9}")

# ---------- 拟合图：挑 4 个通道
rng0 = np.random.default_rng(19)
NG = 400000
picks = [("少女", 0), ("骷髅", 3), ("毒液", 2), ("黑暗法师", 1)]
rows = []
for nm, d in picks:
    e = [x for x in styles if x["name"] == nm][0]
    mu = np.array(e["mu"], dtype=float)
    L = np.array(e["chol"], dtype=float)
    x = rng0.standard_normal((NG, 4)) @ L.T + mu
    xa = x[np.all(np.abs(x) <= 3.5, axis=1)]
    X = np.clip(np.floor((xa + 3.5) / DELTA * NB), 0, NB - 1).astype(int)
    tgt = np.bincount(X[:, d], minlength=NB).astype(float)
    tgt /= tgt.sum()
    ch = spec["styles"][nm][d]
    pk = pmf_of(ch["channels"])
    idx = ch["c"] + np.arange(len(pk))
    out = np.zeros(NB)
    ok = (idx >= 0) & (idx < NB)
    out[idx[ok]] = pk[ok]
    if (idx < 0).any():
        out[0] += pk[idx < 0].sum()
    if (idx >= NB).any():
        out[NB - 1] += pk[idx >= NB].sum()
    rows.append((f"{nm} ch{d+1}  M={ch['M']}  CDF={ch['cdf']:.4f}", tgt, out))
    # 覆盖全部风格的目标分布（给后面出图用）
# 全部目标缓存
TARGET = {}
for e in styles:
    mu = np.array(e["mu"], dtype=float)
    L = np.array(e["chol"], dtype=float)
    x = rng0.standard_normal((NG, 4)) @ L.T + mu
    xa = x[np.all(np.abs(x) <= 3.5, axis=1)]
    X = np.clip(np.floor((xa + 3.5) / DELTA * NB), 0, NB - 1).astype(int)
    TARGET[e["name"]] = []
    for d in range(4):
        p = np.bincount(X[:, d], minlength=NB).astype(float)
        TARGET[e["name"]].append(p / p.sum())

# 画：左列概率（目标 vs 模型），右列 CDF 差
W, H = 1000, 240 * len(rows)
im = Image.new("RGB", (W, H), (20, 20, 24))
dr = ImageDraw.Draw(im)
for r, (title, tgt, mod) in enumerate(rows):
    y0 = r * 240
    dr.text((10, y0 + 4), title, fill=(230, 230, 240))
    # 概率图
    pw, ph = 620, 170
    bx, by = 20, y0 + 26
    mx = max(tgt.max(), mod.max()) * 1.1
    for xx in range(pw):
        x = int(xx / pw * NB)
        h1 = int(tgt[x] / mx * ph)
        h2 = int(mod[x] / mx * ph)
        if h1:
            dr.line([(bx + xx, by + ph), (bx + xx, by + ph - h1)], fill=(150, 150, 160))
        if h2:
            dr.line([(bx + xx, by + ph), (bx + xx, by + ph - h2)], fill=(90, 190, 255))
    dr.rectangle([bx, by, bx + pw, by + ph], outline=(70, 70, 80))
    dr.text((bx, by + ph + 4), "概率：灰=目标，蓝=投掷器模型", fill=(170, 170, 180))
    # CDF 差
    cx0, cy0, cw, chh = 690, by, 290, ph
    dc = np.cumsum(mod) - np.cumsum(tgt)
    for xx in range(cw):
        x = int(xx / cw * NB)
        v = dc[x]
        yy = cy0 + chh // 2 - int(v / 0.05 * (chh // 2))
        yy = max(cy0, min(cy0 + chh, yy))
        dr.line([(cx0 + xx, cy0 + chh // 2), (cx0 + xx, yy)], fill=(255, 120, 120))
    dr.rectangle([cx0, cy0, cx0 + cw, cy0 + chh], outline=(70, 70, 80))
    dr.line([(cx0, cy0 + chh // 2), (cx0 + cw, cy0 + chh // 2)], fill=(60, 60, 70))
    dr.text((cx0, cy0 + chh + 4), f"CDF 差（满格 = ±0.05）", fill=(170, 170, 180))
im.save(os.path.join(RESULTS, "shape_fit.png"))
print(f"\n  wrote results/shape_fit.png")

# ---------- 出图
def sample_direct(nm, n, rng):
    out = np.zeros((n, 4), dtype=np.int64)
    for d, ch in enumerate(spec["styles"][nm]):
        acc = np.full(n, ch["c"], dtype=np.int64)
        for c in ch["channels"]:
            p = np.array(c["abc"]) / 9.0
            acc += c["w"] * rng.choice([0, 1, 2], size=n, p=p)
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
    g = MC[int_levels(sample_direct(nm, K, rngs))]
    y0 = GAP + si * (8 * SC + GAP)
    for k in range(K):
        img = image_from_left_half(g[k])
        px = np.clip(img / 15.0 * 255, 0, 255).astype(np.uint8)
        big = np.repeat(np.repeat(px, SC, 0), SC, 1)
        xx = GAP + k * (8 * SC + PAD)
        cvv[y0:y0 + 8 * SC, xx:xx + 8 * SC] = np.stack([big, big, big], -1)
    if si:
        cvv[y0 - GAP:y0 - GAP + 3, GAP:Wc - GAP] = (70, 70, 70)
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
Image.fromarray(cvv).save(os.path.join(RESULTS, "shape_faces.png"))
print(f"  wrote results/shape_faces.png")
print()
print("=" * 88)
print("平均脸对照（600 张）")
print("=" * 88)
for e, a, b in zip(styles, mref, mdir):
    print(f"  {e['name']:10s} 最大差 {np.abs(a-b).max():.3f} 级   "
          f"平均差 {np.abs(a-b).mean():.3f} 级")
print("SHAPE_FINAL_DONE")

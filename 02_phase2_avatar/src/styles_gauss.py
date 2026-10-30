#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Turn the seven hand-picked style points into joint Gaussians that avoid the bad region.

WHAT THE USER ASKED FOR
"球有的大，有的小，有的坐标不在球心，你得自己调整使得球变大，然后把球变成一个类似联合高斯
分布的东西。"

So each style starts as a seed point and becomes N(mu, Sigma), with mu free to move and Sigma a
full covariance, grown as large as the safe region allows. Growing matters because a seed the user
picked by eye usually sits near the edge of its own style: the middle of the style is a little
further in, and the shape is not spherical -- PC3 barely matters for 少女 while it matters a lot
elsewhere, which is exactly what a covariance captures and a radius cannot.

HOW THE SAFE REGION IS DEFINED
The quadratic scorer gives a continuous badness over the whole cube. It is thresholded at the
value that marks 33.5% of the volume bad, matching the measured rate on the 4096-point label map,
so the continuous field and the discrete labels agree on the overall balance.

WHAT IS REPORTED
    the grown centre and scales, per style
    the bad rate at one and two sigma
    how much the styles overlap, since the user noted 少女 and 少年 are connected in places
    the sampled faces, because names can only be judged against pictures
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(ROOT, "webapp")
OUT = os.path.join(ROOT, "results", "styles")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


# ---------------------------------------------------------------- model and scorer
j = json.load(open(os.path.join(WEB, "pca22_dim4w64L1_model.json"), encoding="utf-8"))
Ls = j["gen"]["layers"]
seq = []
for i, L in enumerate(Ls):
    W = torch.tensor(L["W"], dtype=torch.float32)
    b = torch.tensor(L["b"], dtype=torch.float32)
    lin = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        lin.weight.copy_(W); lin.bias.copy_(b)
    seq.append(lin)
    if i != len(Ls) - 1:
        seq.append(nn.ReLU())
net = nn.Sequential(*seq).eval()

q = json.load(open(os.path.join(WEB, "scorer_quad.json"), encoding="utf-8"))
C_ = q["coef"]
PAIRS = [(int(k[0]), int(k[1])) for k in C_["cross"].keys()]
CROSS = [C_["cross"][k] for k in C_["cross"].keys()]


def badness(x):
    x = np.atleast_2d(x)
    v = np.full(len(x), C_["c"])
    for i in range(4):
        v = v + C_["linear"][i] * x[:, i] + C_["square"][i] * x[:, i] ** 2
    for (i, k), d in zip(PAIRS, CROSS):
        v = v + d * x[:, i] * x[:, k]
    return v


# SAFETY COMES FROM THE DISCRETE LABELS, NOT FROM THE QUADRATIC FIT.
# The scorer has real local errors: at the user's 潦草大叔 point (3.5, 3.5, 3.5, -0.5) it reports
# badness 0.671, above its own 0.424 threshold, yet that cell was never marked bad. A global fit
# at AUC 0.853 is wrong on roughly one point in seven, and using it to define "safe" would invent
# a wall that is not there. The 4096-cell map is a complete measurement, so it is the authority.
LEVELS = np.linspace(-3.5, 3.5, 8)
marks = json.load(open(os.path.join(ROOT, "results", "pca4map", "pca4_plane_marks.json"),
                       encoding="utf-8"))
bad_cells = np.zeros((8, 8, 8, 8), dtype=bool)
for m in marks["marks"]:
    idx = tuple(int(np.argmin(np.abs(LEVELS - v))) for v in m["pca"])
    bad_cells[idx] = True
G = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, LEVELS, indexing="ij"), -1).reshape(-1, 4)
BADL = bad_cells.reshape(-1)
SAFE = ~BADL
THR = float(np.quantile(badness(G), 0.665))     # kept only for the sample filter
print(f"lattice {len(G)} cells, safe {100*SAFE.mean():.1f}%  "
      f"(quadratic threshold {THR:.4f}, used only to filter samples)")

STYLES = [
    ("少女",     [1.5, -1.5, -1.5, 0.0]),
    ("少年",     [-0.5, -2.5, 2.5, 1.5]),
    ("骷髅",     [2.5, 2.5, 1.5, 2.5]),
    ("毒液",     [-3.5, 1.5, 3.5, 0.0]),
    ("黑色悦灵", [-2.5, -2.5, 3.5, -3.5]),
    ("潦草大叔", [3.5, 3.5, 3.5, -0.5]),
    ("黑暗法师", [-1.5, 3.5, -0.5, 1.5]),
]


def maha(X, mu, Ci):
    d = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, Ci, d))


def fit_style(seed, target_purity=0.92, ladder=None):
    """Fit a Gaussian to the safe lattice points inside the largest clean Euclidean ball."""
    seed = np.array(seed, dtype=np.float64)
    if ladder is None:
        ladder = np.arange(0.8, 3.01, 0.2)
    best = None
    for R in ladder:
        m = np.linalg.norm(G - seed, axis=1) <= R
        if m.sum() < 40:
            continue
        purity = SAFE[m].mean()
        if purity < target_purity:
            break
        best = (R, m)
    if best is None:
        return seed, np.eye(4) * 0.6, None
    R, m = best
    pts = G[m & SAFE]
    mu = pts.mean(0)
    C = np.cov(pts.T) + np.eye(4) * 0.05
    # verify the Gaussian that will be sampled, and shrink if 2 sigma is not clean enough
    for _ in range(12):
        Ci = np.linalg.inv(C)
        mm = maha(G, mu, Ci) <= 2.0
        if mm.sum() < 30:
            break
        if SAFE[mm].mean() >= 0.95:
            break
        C = C * 0.85
    return mu, C, (R, best[1].sum())


def multi_start(seed):
    best = None
    for t in (0.0, 0.15, 0.30):
        cand = (np.array(seed, dtype=np.float64) * (1 - t)).tolist()
        mu, C, info = fit_style(cand)
        Ci = np.linalg.inv(C)
        m2 = maha(G, mu, Ci) <= 2.0
        m1 = maha(G, mu, Ci) <= 1.0
        if m1.sum() < 15:
            continue
        br2 = 1 - SAFE[m2].mean() if m2.sum() else 1.0
        score = float(m1.mean()) * (1.0 if br2 < 0.06 else 0.4)
        if best is None or score > best[0]:
            best = (score, mu, C, info, t, br2)
    if best is None:
        mu, C, info = fit_style(seed)
        return mu, C, info, 0.0
    return best[1], best[2], best[3], best[4]


print()
print("=" * 104)
print("GROWN STYLE GAUSSIANS")
print("=" * 104)
res = {}
print(f"  {'style':10s} {'seed':>22s} {'grown centre':>26s} {'sd':>22s} {'1σ坏率':>8s} {'2σ坏率':>8s} {'内移':>5s}")
print("  " + "-" * 106)
for name, seed in STYLES:
    mu, C, gb, moved = multi_start(seed)
    sd = np.sqrt(np.diag(C))
    Ci = np.linalg.inv(C)
    m1 = maha(G, mu, Ci) <= 1.0
    m2 = maha(G, mu, Ci) <= 2.0
    br1 = 1 - SAFE[m1].mean() if m1.sum() else float("nan")
    br2 = 1 - SAFE[m2].mean() if m2.sum() else float("nan")
    res[name] = {"seed": seed, "mu": mu.tolist(), "cov": C.tolist(), "sd": sd.tolist(),
                 "moved": float(moved),
                 "bad1": float(br1), "bad2": float(br2),
                 "vol1": float(m1.mean()), "vol2": float(m2.mean())}
    print(f"  {name:10s} {np.round(seed,1)!s:>22s} {np.round(mu,2)!s:>26s} "
          f"{np.round(sd,2)!s:>22s} {100*br1:7.1f}% {100*br2:7.1f}% {moved:5.0%}")

# ---------------------------------------------------------------- overlap
print()
print("=" * 104)
print("STYLE OVERLAP (Mahalanobis distance between grown centres, pooled covariance)")
print("=" * 104)
names = list(res.keys())
print("       " + " ".join(f"{n:>10s}" for n in names))
for a in names:
    row = []
    for b in names:
        if a == b:
            row.append(f"{'—':>10s}")
            continue
        Cp = (np.array(res[a]["cov"]) + np.array(res[b]["cov"])) / 2
        d = maha(np.array([res[b]["mu"]]), np.array(res[a]["mu"]), np.linalg.inv(Cp))[0]
        row.append(f"{d:10.2f}")
    print(f"  {a:4s} " + " ".join(row))
print()
print("  小于 2 表示两个高斯的 1σ 区域已经相交。")

# ---------------------------------------------------------------- sample
def sample_style(name, n, seed=0):
    r = res[name]
    mu = np.array(r["mu"]); C = np.array(r["cov"])
    rng = np.random.default_rng(seed)
    keep = []
    for _ in range(200):
        X = rng.multivariate_normal(mu, C, 400)
        X = X[np.all(np.abs(X) <= 3.5, axis=1)]
        if len(X) == 0:
            continue
        ok = badness(X) < THR
        keep.extend(X[ok].tolist())
        if len(keep) >= n * 4:
            break
    keep = np.array(keep[:n])
    if len(keep) == 0:
        return None
    with torch.no_grad():
        v = np.clip(net(torch.tensor(keep.astype(np.float32))).numpy(), 0, 15)
    lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
    lv = np.rint(np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)).astype(int)
    img = np.zeros((len(lv), 8, 8, 3))
    img[:, layout.POS_ROW, layout.POS_COL] = np.clip(lv, 0, 15).reshape(len(lv), 32)[:, :, None]
    img[:, layout.POS_ROW, layout.MIRROR_COL] = np.clip(lv, 0, 15).reshape(len(lv), 32)[:, :, None]
    return img


PER = 20
TILE, PAD = 52, 4
W = 200 + PER * (TILE + PAD)
H = 96 + len(STYLES) * (TILE + 24)
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(13), font(10)
dr.text((12, 12), "七种风格：从给定的种子点撑大成联合高斯后的抽样", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), "每个高斯都在『坏区之外』尽可能撑大；中心已经自动挪到风格内部。", font=f_s,
        fill=(150, 162, 178))
dr.text((12, 54), "样本按二次打分器过滤，只在安全区内抽。", font=f_s, fill=(130, 142, 158))
y = 78
for name in names:
    r = res[name]
    img = sample_style(name, PER, seed=hash(name) % 1000)
    dr.text((12, y + 6), name, font=f_l, fill=(110, 200, 130))
    dr.text((12, y + 22), f"1σ坏率 {100*r['bad1']:.0f}%", font=f_s, fill=(150, 162, 178))
    if img is None:
        dr.text((200, y + 16), "无可用样本", font=f_l, fill=(255, 130, 130))
    else:
        for i in range(min(PER, len(img))):
            t = np.kron(np.clip(img[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                        np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
            cv.paste(Image.fromarray(t), (200 + i * (TILE + PAD), y))
    y += TILE + 24
p = os.path.join(OUT, "STYLES_7.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in res.items()},
          open(os.path.join(OUT, "styles.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'styles.json')}")
print("STYLES_DONE")

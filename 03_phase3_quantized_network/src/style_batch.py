#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
四种采样器的批量对照。

REF     原始 (mu, L) + 拒绝采样            —— 二期的标准答案
INDEP   对角近似（只保留各维标准差）+ 重拟合 + 投掷器 + 截断
TRI     完整下三角 (mu', L') + 重拟合 + 投掷器 + 截断 —— 不要 4→4 的话就没有它
CRASH   原始 (mu, L) + 投掷器 + 粗暴截断到边界   —— 用户要看的「碰壁」

INDEP 与 TRI 都用重拟合参数，差别只在有没有交叉项。
CRASH 不重拟合，直接让原始高斯撞墙。
"""
import json
import os
import sys
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

B, NB = 8, 256
DELTA = 7.0
NLEV = 15
SIG_DROP = 36.466

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
W1 = np.array(mj["gen"]["layers"][0]["W"])
TH1 = np.array(mj["gen"]["layers"][0]["b"])
V2W = np.array(mj["gen"]["layers"][1]["W"])
PHI = np.array(mj["gen"]["layers"][1]["b"])
MC = np.array(json.load(open(os.path.join(P2, "webapp", "merge_mc11.json"),
                             encoding="utf-8"))["_lut"])
styles = json.load(open(os.path.join(P2, "webapp", "styles_gauss.json"),
                        encoding="utf-8"))["styles"]

comp = compile_network([Layer(W1, TH1, "relu", comparable=False, out_bits=8),
                        Layer(V2W, PHI, "clip", comparable=True,
                              lam=np.full(32, 15.0))],
                        -3.5, 3.5, B, verbose=False)


def int_levels(X):
    Z = integer_forward(comp, np.asarray(X, dtype=np.int64))
    m, n = Z.max(1, keepdims=True), Z.min(1, keepdims=True)
    dZ = np.maximum(m - n, 1)
    return np.clip(np.floor_divide((Z - n) * NLEV + (dZ >> 1), dZ), 0, NLEV)


WDROP = [1, 6, 7, 7, 7, 9, 10, 11, 11, 12, 14, 15, 18]
PV = np.array([4, 1, 4]) / 9.0
pos = np.array([0])
pr = np.array([1.0])
for wi in WDROP:
    v = np.arange(3) * wi
    npos = (pos[:, None] + v[None, :]).ravel()
    npr = (pr[:, None] * PV[None, :]).ravel()
    up, inv = np.unique(npos, return_inverse=True)
    pos = up
    pr = np.bincount(inv, weights=npr)


def draw_S(n, rng):
    return rng.choice(pos, size=(n, 4), p=pr)


def quant(x):
    return np.clip(np.floor((x + 3.5) / DELTA * NB), 0, NB - 1).astype(np.int64)


def S_to_z(S):
    return (S - 128.0) / SIG_DROP


rng = np.random.default_rng(2024)
NAMES = ["REF", "INDEP", "TRI", "CRASH"]

# 每个风格先算重拟合参数与对角近似参数
PAR = {}
for e in styles:
    mu = np.array(e["mu"], dtype=float)
    L = np.array(e["chol"], dtype=float)
    z = rng.standard_normal((200000, 4))
    x = z @ L.T + mu
    keep = np.all(np.abs(x) <= 3.5, axis=1)
    xa = x[keep]
    mu2 = xa.mean(0)
    L2 = np.linalg.cholesky(np.cov(xa.T, bias=True))
    PAR[e["name"]] = dict(mu=mu, L=L, mu2=mu2, L2=L2,
                          sd=xa.std(0), mean_diag=xa.mean(0))


def sample(style, mode, n, rng):
    """返回 (n,4) 的 X 整数。"""
    p = PAR[style["name"]]
    if mode == "REF":
        got = []
        while len(got) < n:
            z = rng.standard_normal((n, 4))
            x = z @ p["L"].T + p["mu"]
            for r in x:
                if np.all(np.abs(r) <= 3.5):
                    got.append(r)
                    if len(got) >= n:
                        break
        return quant(np.array(got))
    if mode == "CRASH":
        z = rng.standard_normal((n, 4))
        return quant(z @ p["L"].T + p["mu"])
    S = draw_S(n, rng)
    zz = S_to_z(S)
    if mode == "INDEP":
        x = p["mean_diag"][None, :] + zz * p["sd"][None, :]
    else:
        x = p["mu2"][None, :] + zz @ p["L2"].T
    return quant(x)


def grey(style, mode, n, rng):
    X = sample(style, mode, n, rng)
    return MC[int_levels(X)]


# ---------------- 统计
print("=" * 100)
print("统计特征（各 1000 张）")
print("=" * 100)
print(f"  {'风格':10s} {'采样器':7s} {'平均脸L1':>9s} {'std 比':>7s} "
      f"{'0级占比':>8s} {'15级占比':>9s} {'饱和脸':>7s} {'级均值':>7s}")
print("  " + "-" * 74)
STATS = {}
NS = 1000
for e in styles:
    ref = grey(e, "REF", NS, rng).astype(float)
    mr, sr = ref.mean(0), ref.std(0)
    for mode in NAMES:
        g = ref if mode == "REF" else grey(e, mode, NS, rng).astype(float)
        m, s = g.mean(0), g.std(0)
        l1 = float(np.abs(m - mr).mean())
        ratio = float(s.mean() / sr.mean()) if sr.mean() > 0 else 0.0
        z0 = float((g <= 0.5).mean())
        z15 = float((g >= 14.5).mean())
        sat = float((((g <= 0.5) | (g >= 14.5)).mean(1) > 0.5).mean())
        print(f"  {e['name'] if mode == 'REF' else '':10s} {mode:7s} {l1:9.3f} "
              f"{ratio:7.3f} {100*z0:7.1f}% {100*z15:8.1f}% {100*sat:6.1f}% "
              f"{g.mean():7.3f}")
        STATS[(e["name"], mode)] = (l1, ratio, z0, z15, sat)
    print()

# ---------------- 批量出图：每风格四行
print("=" * 100)
print("批量出图")
print("=" * 100)
KSHOW = 10
SC, PAD, GAP, SEP = 18, 4, 12, 8
for e in styles:
    Wc = KSHOW * 8 * SC + (KSHOW - 1) * PAD + 2 * GAP
    Hc = len(NAMES) * (8 * SC + PAD) + 2 * GAP
    cv = np.full((Hc, Wc, 3), 18, dtype=np.uint8)
    for vi, mode in enumerate(NAMES):
        g = grey(e, mode, KSHOW, np.random.default_rng(900 + vi))
        y0 = GAP + vi * (8 * SC + PAD)
        for k in range(KSHOW):
            img = image_from_left_half(g[k])
            px = np.clip(img / 15.0 * 255, 0, 255).astype(np.uint8)
            big = np.repeat(np.repeat(px, SC, 0), SC, 1)
            xx = GAP + k * (8 * SC + PAD)
            cv[y0:y0 + 8 * SC, xx:xx + 8 * SC] = np.stack([big, big, big], -1)
        cv[y0 + 8 * SC:y0 + 8 * SC + PAD - 1, GAP:Wc - GAP] = \
            {"REF": (90, 190, 255), "INDEP": (120, 230, 120),
             "TRI": (230, 220, 120), "CRASH": (255, 110, 110)}[mode]
    out = os.path.join(RESULTS, f"batch_{e['name']}.png")
    Image.fromarray(cv).save(out)
    print(f"  wrote {out}  ({os.path.getsize(out)/1e3:.0f} KB)")

# ---------------- 四种采样器混排，便于横向比较
Wc = KSHOW * 8 * SC + (KSHOW - 1) * PAD + 2 * GAP
Hc = len(styles) * len(NAMES) * (8 * SC) + len(styles) * (len(NAMES) * PAD + SEP) + 2 * GAP
cv = np.full((Hc, Wc, 3), 18, dtype=np.uint8)
yy = GAP
for e in styles:
    for vi, mode in enumerate(NAMES):
        g = grey(e, mode, KSHOW, np.random.default_rng(900 + vi))
        for k in range(KSHOW):
            img = image_from_left_half(g[k])
            px = np.clip(img / 15.0 * 255, 0, 255).astype(np.uint8)
            big = np.repeat(np.repeat(px, SC, 0), SC, 1)
            xx = GAP + k * (8 * SC + PAD)
            cv[yy:yy + 8 * SC, xx:xx + 8 * SC] = np.stack([big, big, big], -1)
        cv[yy:yy + 8 * SC, GAP - 4:GAP - 1] = \
            {"REF": (90, 190, 255), "INDEP": (120, 230, 120),
             "TRI": (230, 220, 120), "CRASH": (255, 110, 110)}[mode]
        yy += 8 * SC + PAD
    yy += SEP
out = os.path.join(RESULTS, "batch_all.png")
Image.fromarray(cv).save(out)
print(f"  wrote {out}  ({os.path.getsize(out)/1e3:.0f} KB)")
print("  从上到下每四行一种风格，行序 REF / INDEP / TRI / CRASH")
print("BATCH_DONE")

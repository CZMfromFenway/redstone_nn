#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
最终出图：投掷器采样 → 4→4 变换 → 整数模型 → MC11。
参照：原始拒绝采样 → 同一整数模型。
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
XLO, XHI, DELTA = -3.5, 3.5, 7.0
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
refit = json.load(open(os.path.join(RESULTS, "styles_refit.json"), encoding="utf-8"))

comp = compile_network([Layer(W1, TH1, "relu", comparable=False, out_bits=8),
                        Layer(V2W, PHI, "clip", comparable=True,
                              lam=np.full(32, 15.0))],
                        XLO, XHI, B, verbose=False)


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


def sample_dropper(n, rng):
    return rng.choice(pos, size=(n, 4), p=pr)


def affine(mu, L):
    A = (256.0 / (DELTA * SIG_DROP)) * L
    b = (256.0 / DELTA) * (mu + 3.5) - A @ np.full(4, 128.0)
    return A, b


def to_X(S, A, b):
    return np.clip(np.floor(S @ A.T + b[None, :]), 0, NB - 1).astype(np.int64)


rng = np.random.default_rng(101)
K = 4
lat_ref, lat_drop, names = [], [], []

for e in styles:
    nm = e["name"]
    names.append(nm)
    # 参照：原始参数 + 拒绝采样
    mu0 = np.array(e["mu"], dtype=float)
    L0 = np.array(e["chol"], dtype=float)
    got = []
    while len(got) < K:
        z = rng.standard_normal((K, 4))
        x = z @ L0.T + mu0
        for r in x:
            if np.all(np.abs(r) <= 3.5) and len(got) < K:
                got.append(r)
    Xr = np.clip(np.floor((np.array(got) + 3.5) / DELTA * NB), 0, NB - 1).astype(np.int64)
    lat_ref.append(MC[int_levels(Xr)])
    # 投掷器：新参数
    mu1 = np.array(refit[nm]["mu"], dtype=float)
    L1 = np.array(refit[nm]["chol"], dtype=float)
    A, b = affine(mu1, L1)
    S = sample_dropper(K, rng)
    Xd = to_X(S, A, b)
    lat_drop.append(MC[int_levels(Xd)])

# 图一：投掷器采样（主体交付）
SC, PAD, GAP = 30, 6, 14
Wc = K * 8 * SC + (K - 1) * PAD + 2 * GAP
Hc = len(styles) * (8 * SC + GAP) + GAP
cv = np.full((Hc, Wc, 3), 18, dtype=np.uint8)
for si in range(len(styles)):
    y0 = GAP + si * (8 * SC + GAP)
    for k in range(K):
        img = image_from_left_half(lat_drop[si][k])
        px = np.clip(img / 15.0 * 255, 0, 255).astype(np.uint8)
        big = np.repeat(np.repeat(px, SC, 0), SC, 1)
        xx = GAP + k * (8 * SC + PAD)
        cv[y0:y0 + 8 * SC, xx:xx + 8 * SC] = np.stack([big, big, big], -1)
    if si:
        cv[y0 - GAP:y0 - GAP + 3, GAP:Wc - GAP] = (70, 70, 70)
out1 = os.path.join(RESULTS, "style_dropper_main.png")
Image.fromarray(cv).save(out1)
print(f"wrote {out1}  ({os.path.getsize(out1)/1e3:.0f} KB)")
print("  六种风格，每行四张，全部来自投掷器采样 + 整数模型")

# 图二：参照 vs 投掷器
SC2, PAD2, GAP2 = 26, 5, 12
K2 = 2
Wc2 = K2 * 8 * SC2 + (K2 - 1) * PAD2 + 2 * GAP2
Hc2 = len(styles) * (2 * 8 * SC2 + PAD2 + GAP2) + GAP2
cv2 = np.full((Hc2, Wc2, 3), 18, dtype=np.uint8)
for si in range(len(styles)):
    y0 = GAP2 + si * (2 * 8 * SC2 + PAD2 + GAP2)
    for row, lat in ((0, lat_ref), (1, lat_drop)):
        for k in range(K2):
            img = image_from_left_half(lat[si][k])
            px = np.clip(img / 15.0 * 255, 0, 255).astype(np.uint8)
            big = np.repeat(np.repeat(px, SC2, 0), SC2, 1)
            yy = y0 + row * (8 * SC2 + PAD2)
            xx = GAP2 + k * (8 * SC2 + PAD2)
            cv2[yy:yy + 8 * SC2, xx:xx + 8 * SC2] = np.stack([big, big, big], -1)
    cv2[y0 + 8 * SC2:y0 + 8 * SC2 + PAD2 - 1,
        GAP2:GAP2 + K2 * 8 * SC2 + (K2 - 1) * PAD2] = (60, 60, 60)
out2 = os.path.join(RESULTS, "style_dropper_cmp.png")
Image.fromarray(cv2).save(out2)
print(f"wrote {out2}  ({os.path.getsize(out2)/1e3:.0f} KB)")
print("  每个风格两行：上=原始拒绝采样，下=投掷器采样（新参数）")

# 统计
print()
print("=" * 88)
print("投掷器采样 vs 原始拒绝采样：256 张的平均像素差")
print("=" * 88)
print(f"  {'风格':10s} {'像素不同':>9s} {'平均|差|':>9s} {'最大':>5s}")
print("  " + "-" * 38)
for e in styles:
    nm = e["name"]
    mu0 = np.array(e["mu"], dtype=float)
    L0 = np.array(e["chol"], dtype=float)
    z = rng.standard_normal((256, 4))
    x = z @ L0.T + mu0
    keep = np.all(np.abs(x) <= 3.5, axis=1)
    Xr = np.clip(np.floor((x[keep] + 3.5) / DELTA * NB), 0, NB - 1).astype(np.int64)
    mu1 = np.array(refit[nm]["mu"], dtype=float)
    L1 = np.array(refit[nm]["chol"], dtype=float)
    A, b = affine(mu1, L1)
    Xd = to_X(sample_dropper(len(Xr), rng), A, b)
    ld, lr = int_levels(Xd), int_levels(Xr)
    d = ld - lr
    print(f"  {nm:10s} {100*(d!=0).mean():8.1f}% {np.abs(d).mean():9.3f} {np.abs(d).max():5d}")
print()
print("  两张图配同一套整数模型与 MC11 映射；差别只来自采样方式。")
print("STYLE_MAIN_DONE")

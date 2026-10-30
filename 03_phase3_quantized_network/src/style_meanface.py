#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
两种采样器的分布对照（不能逐张配对比较——它们是独立抽样）。

正确做法：各抽 2000 张，比平均脸与逐像素标准差。
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
refit = json.load(open(os.path.join(RESULTS, "styles_refit.json"), encoding="utf-8"))

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


def affine(mu, L):
    A = (256.0 / (DELTA * SIG_DROP)) * L
    b = (256.0 / DELTA) * (mu + 3.5) - A @ np.full(4, 128.0)
    return A, b


rng = np.random.default_rng(555)
NG = 2000

print("=" * 92)
print("两种采样器的平均脸对照（各 2000 张）")
print("=" * 92)
print(f"  {'风格':10s} {'平均脸最大差(级)':>16s} {'平均绝对差(级)':>15s} "
      f"{'标准差之比':>11s}")
print("  " + "-" * 60)
mf_ref, mf_drop, names = [], [], []
for e in styles:
    nm = e["name"]
    # 参照
    mu0 = np.array(e["mu"], dtype=float)
    L0 = np.array(e["chol"], dtype=float)
    got = []
    while len(got) < NG:
        z = rng.standard_normal((NG, 4))
        x = z @ L0.T + mu0
        for r in x:
            if np.all(np.abs(r) <= 3.5):
                got.append(r)
                if len(got) >= NG:
                    break
    Xr = np.clip(np.floor((np.array(got) + 3.5) / DELTA * NB), 0, NB - 1).astype(np.int64)
    Lr = MC[int_levels(Xr)].astype(np.float64)
    # 投掷器
    mu1 = np.array(refit[nm]["mu"], dtype=float)
    L1 = np.array(refit[nm]["chol"], dtype=float)
    A, b = affine(mu1, L1)
    S = rng.choice(pos, size=(NG, 4), p=pr)
    Xd = np.clip(np.floor(S @ A.T + b[None, :]), 0, NB - 1).astype(np.int64)
    Ld = MC[int_levels(Xd)].astype(np.float64)
    mf_ref.append(Lr.mean(0))
    mf_drop.append(Ld.mean(0))
    names.append(nm)
    dm = np.abs(Lr.mean(0) - Ld.mean(0))
    ratio = float(Ld.std(0).mean() / Lr.std(0).mean())
    print(f"  {nm:10s} {dm.max():16.3f} {dm.mean():15.3f} {ratio:11.3f}")

# 平均脸并排
SC, PAD, GAP = 30, 6, 14
Wc = 2 * 8 * SC + PAD + 2 * GAP
Hc = len(styles) * (8 * SC + GAP) + GAP
cv = np.full((Hc, Wc, 3), 18, dtype=np.uint8)
for si in range(len(styles)):
    y0 = GAP + si * (8 * SC + GAP)
    for k, mf in enumerate((mf_ref[si], mf_drop[si])):
        img = image_from_left_half(mf)
        px = np.clip(img / 15.0 * 255, 0, 255).astype(np.uint8)
        big = np.repeat(np.repeat(px, SC, 0), SC, 1)
        xx = GAP + k * (8 * SC + PAD)
        cv[y0:y0 + 8 * SC, xx:xx + 8 * SC] = np.stack([big, big, big], -1)
    cv[y0:y0 + 8 * SC, GAP + 8 * SC:GAP + 8 * SC + PAD - 2] = (60, 60, 60)
out = os.path.join(RESULTS, "style_meanface.png")
Image.fromarray(cv).save(out)
print(f"\n  wrote {out}  ({os.path.getsize(out)/1e3:.0f} KB)")
print("  每行两张平均脸：左=原始拒绝采样，右=投掷器采样")
print()
print("  说明：逐张配对比较无意义（两侧是独立抽样，样本间差异远大于分布差异）。")
print("        平均脸的差才反映分布是否一致。")
print("MEANFACE_DONE")

#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
单色显示器上表现 16 级灰度：单元块尺寸 n 的下限。

一、原理（单色如何表示灰度）
    人眼在足够远处对 n×n 的方块做空间积分，只看到它的平均亮度。
    所以一个 n×n 方块能表现的灰度由「白点数 k」决定：
        density = k / n^2,   k = 0 .. n^2
    可表现灰度级数 = n^2 + 1。

    但「k 个白点怎么摆」决定图案质量：
      · 顺序抖动（ordered dither）：预先给 n×n 每个格子一个等级 0..n^2-1，
        等级 < k 的点亮。同级图案固定，适合红石灯这种静态显示。
      · Bayer 矩阵：2^m × 2^m 的最优分散顺序，4×4 是标准选择。
      · 误差扩散（Floyd–Steinberg）：逐像素依赖邻域，块内做不到，不适用。

二、硬约束
    要 16 级 ⇒ n^2 + 1 >= 16 ⇒ n >= 4
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

NLEV = 15
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


def bayer(n):
    """递归 Bayer 矩阵（n 为 2 的幂）。"""
    if n == 1:
        return np.zeros((1, 1), dtype=int)
    h = bayer(n // 2)
    return np.block([[4 * h + 0, 4 * h + 2],
                     [4 * h + 3, 4 * h + 1]])


def greedy_rank(n, seed=0):
    """任意 n：按「离已放置点最远」的顺序编号（void-and-cluster 的简化版）。"""
    rank = -np.ones((n, n), dtype=int)
    yy, xx = np.mgrid[0:n, 0:n]
    for r in range(n * n):
        if r == 0:
            cy, cx = 0, 0
        else:
            placed = (rank >= 0)
            d = np.full((n, n), 1e9)
            py, px = np.nonzero(placed)
            for a, b in zip(py, px):
                dy = np.minimum(np.abs(yy - a), n - np.abs(yy - a))
                dx = np.minimum(np.abs(xx - b), n - np.abs(xx - b))
                d = np.minimum(d, dy * dy + dx * dx)
            d[placed] = -1
            cy, cx = np.unravel_index(np.argmax(d), d.shape)
        rank[cy, cx] = r
    return rank


RANK = {}
for n in (2, 3, 4, 5, 6, 8):
    RANK[n] = bayer(n) if (n & (n - 1)) == 0 else greedy_rank(n)

print("=" * 88)
print("一、硬计数约束")
print("=" * 88)
print(f"  {'n':>2s} {'白点数 n^2+1':>13s} {'可表现灰度级':>13s} {'够 16 级?':>10s} "
      f"{'相邻级间距(16级制)':>18s}")
print("  " + "-" * 62)
for n in (2, 3, 4, 5, 6, 8):
    k = n * n + 1
    gap = NLEV / (k - 1) if k > 1 else 0
    print(f"  {n:2d} {k:13d} {k:13d} {'是' if k >= 16 else '否':>10s} {gap:18.3f}")

print()
print("=" * 88)
print("二、16 级 → k 个白点的映射误差")
print("=" * 88)
print("  映射 k = round(L·n²/15)，逆回 L̂ = 15k/n²，误差 ≤ 15/(2n²)")
print()
print(f"  {'n':>2s} {'最大级误差':>11s} {'不同 k 的个数':>14s} {'实际可分辨级数':>15s}")
print("  " + "-" * 48)
for n in (2, 3, 4, 5, 6, 8):
    ks = [int(round(L * n * n / NLEV)) for L in range(16)]
    err = max(abs(15 * k / (n * n) - L) for L, k in zip(range(16), ks))
    print(f"  {n:2d} {err:11.3f} {len(set(ks)):14d} {len(set(ks)):15d}")
print()
print("  n=4 时 16 级能一一对应到 17 个密度中的 16 个，最大误差 0.47 级（半级）。")
print("  n<=3 时密度格点太稀，只能表现出更少的级数，必然出现可见条带。")

# ---------------- 出图
rng = np.random.default_rng(7)
XS = np.clip(rng.standard_normal((3, 4)), -3.5, 3.5)
Xq = np.clip(np.floor((XS + 3.5) / 7.0 * 256), 0, 255).astype(np.int64)
L16 = int_levels(Xq)            # (3, 32)

# 取前 3 个通道当一张 8x8（用 layout 排布）
FIGS = np.array([image_from_left_half(L16[i]).reshape(8, 8) for i in range(3)],
                 dtype=float)


def mono_block(img8, n):
    """把 8x8 的 16 级图渲染成 (8n)x(8n) 黑白。"""
    H, W = img8.shape
    out = np.zeros((H * n, W * n), dtype=np.uint8)
    rk = RANK[n]
    for i in range(H):
        for j in range(W):
            L = img8[i, j]
            k = int(round(L * n * n / NLEV))
            k = max(0, min(n * n, k))
            out[i * n:(i + 1) * n, j * n:(j + 1) * n] = (rk < k).astype(np.uint8)
    return out


def blur_block(img_mono, n):
    H, W = img_mono.shape
    a = img_mono.reshape(H // n, n, W // n, n).mean(axis=(1, 3))
    return a


print()
print("=" * 88)
print("三、块内平均误差（模糊后与 16 级的差）")
print("=" * 88)
print(f"  {'n':>2s} {'最大级误差':>11s} {'平均级误差':>11s} {'整张 8x8 的级误差':>18s}")
print("  " + "-" * 50)
stats = {}
for n in (2, 3, 4, 5, 6, 8):
    errs = []
    for f in FIGS:
        mono = mono_block(f, n)
        rec = blur_block(mono, n) * NLEV
        errs.append(np.abs(rec - f))
    e = np.concatenate([x.ravel() for x in errs])
    stats[n] = (float(e.max()), float(e.mean()))
    print(f"  {n:2d} {e.max():11.3f} {e.mean():11.3f} {e.max():18.3f}")

# ---- 图一：单色渲染
NS = (2, 3, 4, 6, 8)
ZOOM = 4
CW = 8 * max(NS) * ZOOM
pad = 10
Wtot = len(NS) * (CW + pad) + pad
Htot = 3 * (CW + pad) + pad
canvas = np.full((Htot, Wtot), 40, dtype=np.uint8)
for ci, n in enumerate(NS):
    for ri in range(3):
        mono = mono_block(FIGS[ri], n)
        big = np.repeat(np.repeat(mono * 255, ZOOM, 0), ZOOM, 1)
        # 居中
        y0 = pad + ri * (CW + pad) + (CW - big.shape[0]) // 2
        x0 = pad + ci * (CW + pad) + (CW - big.shape[1]) // 2
        canvas[y0:y0 + big.shape[0], x0:x0 + big.shape[1]] = big
Image.fromarray(canvas).save(os.path.join(RESULTS, "mono_dither.png"))
print(f"\n  wrote results/mono_dither.png（列 n=2,3,4,6,8；行=三张头像）")

# ---- 图二：模糊后的还原
canvas2 = np.full((3 * (CW + pad) + pad, len(NS) * (CW + pad) + pad), 40,
                  dtype=np.uint8)
for ci, n in enumerate(NS):
    for ri in range(3):
        mono = mono_block(FIGS[ri], n)
        rec = blur_block(mono, n) * (255.0 / NLEV)
        big = np.repeat(np.repeat(rec.astype(np.uint8), ZOOM * n, 0), ZOOM * n, 1)
        y0 = pad + ri * (CW + pad) + (CW - big.shape[0]) // 2
        x0 = pad + ci * (CW + pad) + (CW - big.shape[1]) // 2
        canvas2[y0:y0 + big.shape[0], x0:x0 + big.shape[1]] = big
Image.fromarray(canvas2).save(os.path.join(RESULTS, "mono_blurred.png"))
print(f"  wrote results/mono_blurred.png（同样的排列，模糊后）")

json.dump({"levels_available": {str(n): n * n + 1 for n in NS},
           "level_error": {str(n): stats[n] for n in NS},
           "n_min": 4},
          open(os.path.join(RESULTS, "dither.json"), "w", encoding="utf-8"),
          indent=1)
print("  已写入 results/dither.json")
print("DITHER_DONE")

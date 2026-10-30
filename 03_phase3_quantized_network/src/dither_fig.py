#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""单色抖动的关键图：16 级灰度阶梯在各 n 下的样子。"""
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
RESULTS = os.path.join(ROOT, "results")
NLEV = 15


def bayer(n):
    if n == 1:
        return np.zeros((1, 1), dtype=int)
    h = bayer(n // 2)
    return np.block([[4 * h + 0, 4 * h + 2], [4 * h + 3, 4 * h + 1]])


def greedy_rank(n):
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


RANK = {n: (bayer(n) if (n & (n - 1)) == 0 else greedy_rank(n))
        for n in (2, 3, 4, 5, 6, 8)}


def block(n, L):
    """16 级里的第 L 级 → n×n 黑白块。"""
    k = int(round(L * n * n / NLEV))
    k = max(0, min(n * n, k))
    return (RANK[n] < k).astype(np.uint8)


# ---------------- 图一：16 级阶梯
NS = (2, 3, 4, 5, 6, 8)
ZOOM = 9
PADX, PADY = 6, 6
LEFT = 54
W = LEFT + 16 * (8 * ZOOM + PADX) + PADX
H = PADY + len(NS) * (8 * ZOOM + PADY)
cv = np.full((H, W), 45, dtype=np.uint8)
for ri, n in enumerate(NS):
    y0 = PADY + ri * (8 * ZOOM + PADY)
    for L in range(16):
        b = block(n, L)
        big = np.repeat(np.repeat(b * 255, ZOOM, 0), ZOOM, 1)
        x0 = LEFT + PADX + L * (8 * ZOOM + PADX) + (8 * ZOOM - big.shape[1]) // 2
        yy0 = y0 + (8 * ZOOM - big.shape[0]) // 2
        cv[yy0:yy0 + big.shape[0], x0:x0 + big.shape[1]] = big
im = Image.fromarray(cv).convert("RGB")
dr = ImageDraw.Draw(im)
for ri, n in enumerate(NS):
    y0 = PADY + ri * (8 * ZOOM + PADY)
    dr.text((8, y0 + 30), f"n={n}", fill=(235, 235, 245))
    dr.text((8, y0 + 46), f"{n*n+1} 级", fill=(160, 160, 175))
im.save(os.path.join(RESULTS, "dither_ramp.png"))
print(f"wrote results/dither_ramp.png  ({os.path.getsize(os.path.join(RESULTS,'dither_ramp.png'))/1e3:.0f} KB)")
print("  每行一个 n，16 列是 16 级灰度；按 Bayer/贪心顺序摆放白点")

# ---------------- 图二：真的头像（放亮）
import numpy as np
sys.path.insert(0, HERE)
from compiler import Layer, compile_network, integer_forward  # noqa: E402

P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")
sys.path.insert(0, os.path.join(P2, "src"))
from layout import image_from_left_half  # noqa: E402

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


rng = np.random.default_rng(5)
rows = []
for e in styles[:4]:
    mu = np.array(e["mu"], dtype=float)
    Lc = np.array(e["chol"], dtype=float)
    z = rng.standard_normal((1, 4))
    x = np.clip(z @ Lc.T + mu, -3.5, 3.5)
    Xq = np.clip(np.floor((x + 3.5) / 7.0 * 256), 0, 255).astype(np.int64)
    Lv = int_levels(Xq)[0]
    rows.append(image_from_left_half(Lv).reshape(8, 8))
FIGS = np.array(rows, dtype=float)

NS2 = (3, 4, 5, 6, 8)
ZOOM2 = 7
CW = 8 * ZOOM2
im2 = Image.new("RGB", (60 + (len(NS2) + 1) * (CW + 12) + 12,
                        len(FIGS) * (CW + 14) + 14), (45, 45, 52))
dr2 = ImageDraw.Draw(im2)
for ci, n in enumerate([0] + list(NS2)):
    x0 = 60 + ci * (CW + 12)
    for ri in range(len(FIGS)):
        y0 = 14 + ri * (CW + 14)
        if n == 0:      # 参照：16 级灰度
            g = (FIGS[ri] / NLEV * 255).astype(np.uint8)
            big = np.repeat(np.repeat(g, ZOOM2, 0), ZOOM2, 1)
        else:
            mono = np.zeros((8 * n, 8 * n), dtype=np.uint8)
            for i in range(8):
                for j in range(8):
                    mono[i * n:(i + 1) * n, j * n:(j + 1) * n] = block(n, FIGS[ri][i, j])
            big = np.repeat(np.repeat(mono * 255, max(1, ZOOM2 // n), 0),
                            max(1, ZOOM2 // n), 1)
            big = np.array(Image.fromarray(big).resize((CW, CW), Image.LANCZOS))
        im2.paste(Image.fromarray(big).convert("RGB"), (x0, y0))
    dr2.text((x0 + 4, 0), "16级" if n == 0 else f"n={n}", fill=(235, 235, 245))
im2.save(os.path.join(RESULTS, "dither_faces.png"))
print(f"wrote results/dither_faces.png  ({os.path.getsize(os.path.join(RESULTS,'dither_faces.png'))/1e3:.0f} KB)")
print("  第一列是 16 级灰度参照，后面四列是单色抖动")

# ---------------- 数字
print()
print("=" * 80)
print("块内平均的可达精度")
print("=" * 80)
print(f"  {'n':>2s} {'可达密度级数':>12s} {'相邻级间距':>11s} {'最大级误差':>11s} "
      f"{'相对 n=4':>10s}")
print("  " + "-" * 52)
for n in NS:
    ks = n * n + 1
    err = max(abs(15 * round(L * n * n / NLEV) / (n * n) - L) for L in range(16))
    print(f"  {n:2d} {ks:12d} {NLEV/(ks-1):11.3f} {err:11.3f} "
          f"{err/0.438:10.2f}")
print()
print("  n=4 的最大级误差 0.438，正好是半个灰度级（15/32）。")
print("  再往上加 n 只把这半级继续压小，但面积按 n^2 增长。")
print("DITHER2_DONE")

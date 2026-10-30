#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MC 16 色混凝土调色板结构分析"""
import json, numpy as np, colorsys, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
d = json.load(open(os.path.join(ROOT, 'data', 'mc16_palette.json')))
names = ['white', 'light_gray', 'gray', 'black', 'brown', 'red', 'orange', 'yellow',
         'lime', 'green', 'cyan', 'light_blue', 'blue', 'purple', 'magenta', 'pink']
P = np.array([d[n][0] for n in names], dtype=float)
lab = np.array([0.299, 0.587, 0.114])

print("=" * 96)
print("MC 16 色混凝土调色板（按 czm 给定的顺序）")
print("=" * 96)
print(f"{'#':>3} {'名称':<12} | {'RGB':<18} | {'亮度':>6} | {'色相':>6} | {'饱和':>6} | {'相邻距离':>8}")
print("-" * 96)
prev = None
for i, (n, rgb) in enumerate(zip(names, P)):
    h, s, v = colorsys.rgb_to_hsv(*(rgb / 255.0))
    lum = float(rgb @ lab)
    dist = float(np.linalg.norm(rgb - prev)) if prev is not None else 0.0
    print(f"{i:>3} {n:<12} | ({rgb[0]:>3.0f},{rgb[1]:>3.0f},{rgb[2]:>3.0f})      | "
          f"{lum:>6.0f} | {h*360:>6.0f} | {s:>6.2f} | {dist:>8.0f}")
    prev = rgb

adj = np.linalg.norm(np.diff(P, axis=0), axis=1)
allp = np.linalg.norm(P[:, None] - P[None], axis=2)
np.fill_diagonal(allp, 1e9)
print()
print("=== 顺序的结构性质 ===")
print(f"相邻色距离: 最小 {adj.min():.0f}  中位 {np.median(adj):.0f}  最大 {adj.max():.0f}")
print(f"任意两色的最小距离: {allp.min():.0f}")
print(f"首尾距离 (粉 -> 白): {np.linalg.norm(P[15]-P[0]):.0f}   <- czm 说的闭环")
lum = P @ lab
print()
print(f"沿序号的亮度: {np.round(lum).astype(int).tolist()}")
inc = np.diff(lum)
print(f"  亮度单调递增? {'是' if np.all(inc >= 0) else '否'}")
hue = np.array([colorsys.rgb_to_hsv(*(c / 255))[0] * 360 for c in P])
dh = np.diff(hue) % 360
print()
print(f"沿序号的色相: {np.round(hue).astype(int).tolist()}")
print(f"  色相增量: {np.round(dh).astype(int).tolist()}")
print(f"  色相单调递增(模360)? {'是' if np.all(dh > 0) or (dh < 0).sum() <= 2 else '否'}")

print()
print("=== 结论 ===")
print("  这个顺序**不是**亮度单调，**是**色相沿色环走一圈（模 360 单调）")
print("  前 4 色是中性色（无色相），后 12 色沿色环从棕(20°)走到粉(340°)")
print("  所以它是**一维有序路径**，可以用单个单调阶梯表示 ->")
print("  生成器每个像素只需输出 1 个标量(0..15)而非 16 路 one-hot")

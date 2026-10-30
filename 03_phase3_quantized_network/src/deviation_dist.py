#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
同一路 4 -> 1 通道：实数算出的 Y 与整数方案算出的 Y，整数值偏差的分布。

两条路：
    理想：x -> z -> y = relu(z) -> Y
    整数：x -> X（b 位）-> 位权重求和得 F' -> F -> F_bar -> Y

Y 的标尺有两种取法，两种都算，因为差别本身就是信息：
    标尺 A  用整数方案实际实现的 y_max（即 c*2^b / scale），两条路的标尺一致，
            于是偏差只反映整数运算本身的误差
    标尺 B  用原始的 y_max（max 的 z），偏差里还包含缩放被修正后带来的整体偏移

x 的采样按用户要求：正态分布采样后截断到 [-3.5, 3.5]。
截断有两种做法，一并测：拒绝采样（真正的截断正态）与直接 clamp。
两者只在 ±3.5 处差一个原子，量很小，但既然提了就先量出来。
"""
import json
import os
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")

LIMIT = 882
XMIN, XMAX = -3.5, 3.5
SPAN = XMAX - XMIN
NS = 200000

gray = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                      encoding="utf-8"))
W = np.array(gray["gen"]["layers"][0]["W"])
B = np.array(gray["gen"]["layers"][0]["b"])
peak = np.abs(W).max(1)
ch = int(np.argsort(peak)[len(peak) // 2])
w, theta = W[ch], B[ch]

print("=" * 100)
print(f"通道：层 0 第 {ch} 路")
print("=" * 100)
print(f"  w = {np.round(w, 6)}")
print(f"  theta = {theta:.6f}")
print(f"  x 采样：N(0,1) 后截断到 [{XMIN}, {XMAX}]，n = {NS:,}")

# 两种截断
rng = np.random.default_rng(2026)
raw = rng.standard_normal((NS * 2, 4))
x_rej = raw[np.all(np.abs(raw) <= 3.5, axis=1)][:NS]        # 拒绝采样
x_clip = np.clip(rng.standard_normal((NS, 4)), XMIN, XMAX)  # 直接 clamp
print(f"  拒绝采样保留率 {100*len(x_rej)/(NS*2):.2f}%，"
      f"clamp 在 ±3.5 处的原子占比 {100*np.mean(np.any(np.abs(x_clip) >= 3.5, 1)):.2f}%")


def build(b, limit_top=False):
    """按备选方案定标：次高位受 882 约束（limit_top=False 时次高位即最高位）。"""
    cunit = SPAN / (1 << b)
    capbit = b - 2 if limit_top else b - 1
    scale = LIMIT / (np.abs(w).max() * cunit * (1 << capbit))
    T = np.zeros((4, b), dtype=int)
    for i in range(4):
        for jj in range(b):
            T[i, jj] = int(np.rint(scale * w[i] * cunit * (1 << jj)))
    return T, scale, cunit


def run(x, b, limit_top=True, adjust=True):
    N = 1 << b
    T, scale, cunit = build(b, limit_top)
    zmax = float(np.abs(w).sum() * 3.5 + theta)
    Fbar_max = scale * zmax
    c = int(np.floor(Fbar_max / N)) + 1
    if adjust and c > 1 and (1 - Fbar_max / (c * N)) > 0.20:
        c -= 1
        scale *= (c * N) / Fbar_max
        Fbar_max = scale * zmax
        T = np.zeros((4, b), dtype=int)
        for i in range(4):
            for jj in range(b):
                T[i, jj] = int(np.rint(scale * w[i] * cunit * (1 << jj)))

    z = x @ w + theta
    y = np.maximum(0.0, z)

    Xq = np.clip(np.floor((x - XMIN) / SPAN * N), 0, N - 1).astype(int)
    bits = ((Xq[:, :, None] >> np.arange(b)[None, None, :]) & 1)
    Fp = np.einsum("sij,ij->s", bits, T)
    Fp_min = int(T[T < 0].sum())
    F = Fp - Fp_min
    C0 = float(w @ np.full(4, XMIN) + theta + cunit * 0.5 * w.sum())
    F0 = int(np.rint(-scale * C0 - Fp_min))
    Fbar = F - F0
    Y_int = np.clip(np.maximum(0, np.floor(Fbar / c)).astype(int), 0, N - 1)

    ymax_corr = c * N / scale
    Y_idealA = np.clip(np.floor(y / ymax_corr * N), 0, N - 1).astype(int)
    Y_idealB = np.clip(np.floor(y / zmax * N), 0, N - 1).astype(int)
    return {"b": b, "scale": scale, "c": c, "ymax": zmax, "ymax_corr": ymax_corr,
            "Y_int": Y_int, "A": Y_idealA, "B": Y_idealB, "y": y, "z": z}


def hist(dev, label, top=7):
    vals, cnts = np.unique(dev, return_counts=True)
    order = np.argsort(-cnts)[:top]
    parts = []
    for i in order:
        parts.append(f"{int(vals[i]):+d}:{100*cnts[i]/len(dev):5.2f}%")
    rest = 100 - sum(cnts[i] for i in order) / len(dev) * 100
    print(f"    {label:22s} 均值 {dev.mean():+7.4f}  标准差 {dev.std():6.4f}  "
          f"|偏差|最大 {np.abs(dev).max():2d}  " + "  ".join(parts)
          + (f"  其余 {rest:5.2f}%" if rest > 0.01 else ""))


for tag, x in (("拒绝采样", x_rej), ("clamp", x_clip)):
    print()
    print("=" * 100)
    print(f"x 的截断方式：{tag}")
    print("=" * 100)
    for b in (8, 11, 12):
        r = run(x, b)
        print()
        print(f"  b = {b}   缩放 {r['scale']:.2f}   c = {r['c']}   "
              f"y_max 原 {r['ymax']:.5f} → 修正 {r['ymax_corr']:.5f}"
              f"（高 {100*(r['ymax_corr']/r['ymax']-1):.2f}%）")
        dA = r["Y_int"] - r["A"]
        dB = r["Y_int"] - r["B"]
        hist(dA, "对标尺 A（同一标尺）")
        hist(dB, "对标尺 B（原始 y_max）")
        # 分正负区
        pos = r["z"] > 0
        print(f"    z ≤ 0 的占比 {100*(~pos).mean():.2f}%，其中偏差全为 0："
              f"{bool(np.all(dA[~pos] == 0))}")

print()
print("=" * 100)
print("偏差分布的形状")
print("=" * 100)
r = run(x_rej, 11)
dA = r["Y_int"] - r["A"]
pos = r["z"] > 0
print(f"  b=11，标尺 A，只看 z > 0 的 {int(pos.sum()):,} 个样本：")
print(f"    {'偏差':>6s} {'占比':>8s} {'累计':>8s}")
vals, cnts = np.unique(dA[pos], return_counts=True)
cum = 0
for v, c_ in zip(vals, cnts):
    cum += c_
    print(f"    {int(v):+6d} {100*c_/pos.sum():7.3f}% {100*cum/pos.sum():7.2f}%")
print()
print(f"  同期 y 本身落在 0 的比例 {100*np.mean(r['y']==0):.2f}%（ReLU 截断）")
print(f"  偏差为 0 的样本里，y = 0 的占 {100*np.mean(r['y'][dA==0]==0):.1f}%")
print()
print("  z ≤ 0 时理想 Y 为 0，整数方案的 F_bar < 0 被 ReLU 压成 0，两边必然一致，")
print("  所以偏差全部来自 z > 0 的那部分，且宽度由 |F_bar - scale*z| / c 决定。")
print("DEV_DONE")

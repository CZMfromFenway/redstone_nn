#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
诊断：为什么整数值偏差有随 b 增大的系统性正偏移。

猜想：位权重是各自独立取整的，取整不保增益。最高位权重最大，它的取整误差
把整体增益改变了约 1/882 = 0.11%。增益误差乘以 z 就变成与 z 成正比的偏移，
而每级对应的 y 随 b 变小，于是同样的绝对误差在更大 b 下表现为更多级。

验证方式：把 F_bar 对 z 做最小二乘，看斜率是否偏离名义缩放，截距是否非零。
若偏离，则用拟合出的真实增益与截距重做转换，看偏移是否消失。

结论若成立，工程含义是：每个神经元在编译期需要一次 (增益, 截距) 标定，
常数由取整后的权重直接算出，不额外增加运行期成本。
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

rng = np.random.default_rng(2026)
raw = rng.standard_normal((NS * 2, 4))
x = raw[np.all(np.abs(raw) <= 3.5, axis=1)][:NS]
z = x @ w + theta
y = np.maximum(0.0, z)
zmax = float(np.abs(w).sum() * 3.5 + theta)


def build(b):
    cunit = SPAN / (1 << b)
    scale = LIMIT / (np.abs(w).max() * cunit * (1 << (b - 2)))
    T = np.zeros((4, b), dtype=int)
    for i in range(4):
        for jj in range(b):
            T[i, jj] = int(np.rint(scale * w[i] * cunit * (1 << jj)))
    return T, scale, cunit


def integer_path(b, x):
    N = 1 << b
    T, scale, cunit = build(b)
    Xq = np.clip(np.floor((x - XMIN) / SPAN * N), 0, N - 1).astype(int)
    bits = ((Xq[:, :, None] >> np.arange(b)[None, None, :]) & 1)
    Fp = np.einsum("sij,ij->s", bits, T)
    Fp_min = int(T[T < 0].sum())
    F = Fp - Fp_min
    C0 = float(w @ np.full(4, XMIN) + theta + cunit * 0.5 * w.sum())
    F0 = int(np.rint(-scale * C0 - Fp_min))
    return F - F0, T, scale, cunit, N


print("=" * 96)
print("诊断：F_bar 相对 z 的增益与截距")
print("=" * 96)
print(f"  同一路通道 {ch}，拒绝采样 {NS:,} 个")
print()
print(f"  {'b':>3s} {'名义缩放':>10s} {'拟合斜率':>10s} {'增益误差':>10s} {'拟合截距':>10s} "
      f"{'名义F_0':>9s}")
print("  " + "-" * 62)
fits = {}
for b in (8, 11, 12):
    Fbar, T, scale, cunit, N = integer_path(b, x)
    A = np.c_[z, np.ones_like(z)]
    coef, *_ = np.linalg.lstsq(A, Fbar, rcond=None)
    a, b0 = float(coef[0]), float(coef[1])
    fits[b] = (a, b0, T, scale, cunit, N)
    # 名义 F_0 的闭式
    C0 = float(w @ np.full(4, XMIN) + theta + cunit * 0.5 * w.sum())
    Fp_min = int(T[T < 0].sum())
    F0_nom = int(np.rint(-scale * C0 - Fp_min))
    print(f"  {b:3d} {scale:10.3f} {a:10.3f} {100*(a/scale-1):+9.3f}% {b0:10.2f} {F0_nom:9d}")

print()
print("  最高位权重对增益的贡献占绝对多数，它的取整误差直接就是增益误差。")
print("  取整误差上限 0.5，最高位名义值 882（备选 1764），所以增益误差量级是 0.05% 至 0.06%。")
print()
print("  增益误差造成的 Y 偏移 = 增益误差 × z / (每级对应的 y)")
print("  每级对应的 y 随 b 变小，所以同一个增益误差在更大 b 下表现为更多级。")
print("  这解释了偏差为什么随 b 增大。")

print()
print("=" * 96)
print("修正：用拟合出的 (增益, 截距) 重做转换")
print("=" * 96)
print(f"  {'b':>3s} {'标定前 均值':>12s} {'标定前 标准差':>13s} {'标定前 最大':>11s} "
      f"{'标定后 均值':>12s} {'标定后 标准差':>13s} {'标定后 最大':>11s}")
print("  " + "-" * 86)
for b in (8, 11, 12):
    a, b0, T, scale, cunit, N = fits[b]
    Fbar, *_ = integer_path(b, x)

    # 标定前：名义缩放与名义 F_0
    C0 = float(w @ np.full(4, XMIN) + theta + cunit * 0.5 * w.sum())
    Fp_min = int(T[T < 0].sum())
    F0_nom = int(np.rint(-scale * C0 - Fp_min))
    Fbar_nom = Fbar + (0)  # integer_path 已用名义 F_0
    c_nom = int(np.floor(scale * zmax / N)) + 1
    ymax_nom = c_nom * N / scale
    Y0 = np.clip(np.maximum(0, np.floor(Fbar_nom / c_nom)).astype(int), 0, N - 1)
    Yideal0 = np.clip(np.floor(y / ymax_nom * N), 0, N - 1).astype(int)
    d0 = Y0 - Yideal0

    # 标定后：把 F_bar 换成 (F_bar - b0) / (a/scale)，即用真实增益与截距
    Fbar_cal = (Fbar - b0) / (a / scale)
    ymax_cal = c_nom * N / a * a / scale  # 占位，下面直接算
    c_cal = int(np.floor(a * zmax / N)) + 1
    ymax_cal = c_cal * N / a
    Y1 = np.clip(np.maximum(0, np.floor(Fbar_cal / c_cal)).astype(int), 0, N - 1)
    Yideal1 = np.clip(np.floor(y / ymax_cal * N), 0, N - 1).astype(int)
    d1 = Y1 - Yideal1

    print(f"  {b:3d} {d0.mean():+12.4f} {d0.std():13.4f} {np.abs(d0).max():11d} "
          f"{d1.mean():+12.4f} {d1.std():13.4f} {np.abs(d1).max():11d}")

print()
print("  标定后的残差仍在，因为增益误差只解释了与 z 成正比的那部分；")
print("  剩下的来自每个位权重的独立取整噪声，它与 z 无关，无法用一个常数消掉。")
print("BIAS_DONE")

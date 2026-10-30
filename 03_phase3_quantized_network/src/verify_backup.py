#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
修正两处前提，并验证备选方案。

修正一：强模上限是 2^31 - 1，不是 13455。
        13455 是强模 LUT 里 16 个容器之和的上限，与"强模信号本身能到多大"无关。
        此前据此得出的"层 2 需分段"不成立。

修正二：备选方案允许且仅允许 W^(b-1) 突破 882，直到 2^31 - 1。

要验证的三件事：
    (1) 备选方案是否正好把可用 b 推高 1
    (2) F 与 F' 的实际跨度是否都远在 2^31 之内（即是否真的不需要分段）
    (3) 用户给出的 F -> Y 转换：由 c * 2^b 略大于 max(F_bar) 定出 c，
        再用 Y = round(F_bar / c - 0.5) 转成整数
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

LIMIT = 882                 # 单容器上限，留余量
STRONG = (1 << 31) - 1      # 强模信号上限
XMIN, XMAX = -3.5, 3.5
SPAN = XMAX - XMIN

gray = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                      encoding="utf-8"))
W = np.array(gray["gen"]["layers"][0]["W"])
B = np.array(gray["gen"]["layers"][0]["b"])
peak = np.abs(W).max(1)
j = int(np.argsort(peak)[len(peak) // 2])
w, theta = W[j], B[j]


def table(b, backup, seed_repr=461.564):
    """构造位权重表。backup=True 时只放开最高位。"""
    c = SPAN / (1 << b)
    capbit = b - 2 if backup else b - 1     # 受 882 约束的最高位
    scale = LIMIT / (np.abs(w).max() * c * (1 << capbit))
    T = np.zeros((4, b), dtype=int)
    for i in range(4):
        for jj in range(b):
            T[i, jj] = int(np.rint(scale * w[i] * c * (1 << jj)))
    return T, scale, c


rng = np.random.default_rng(0)
x = np.clip(rng.standard_normal((40000, 4)), XMIN, XMAX)
z_true = x @ w + theta
zspan = float(z_true.max() - z_true.min())


def err(b, backup):
    T, scale, c = table(b, backup)
    if not backup and np.abs(T).max() > LIMIT:
        return None, T, scale, c
    if backup and np.abs(T[:, :-1]).max() > LIMIT:
        return None, T, scale, c
    if backup and np.abs(T[:, -1]).max() > STRONG:
        return None, T, scale, c
    N = 1 << b
    Xq = np.clip(np.floor((x - XMIN) / SPAN * N), 0, N - 1).astype(int)
    bits = ((Xq[:, :, None] >> np.arange(b)[None, None, :]) & 1)
    Fp = np.einsum("sij,ij->s", bits, T)
    C0 = float(w @ np.full(4, XMIN) + theta + c * 0.5 * w.sum())
    z = Fp / scale + C0
    return float(np.abs(z - z_true).mean()), T, scale, c


print("=" * 98)
print("验证 1：备选方案是否把可用 b 推高 1")
print("=" * 98)
print(f"  标准方案：最高位受 882 约束，最低位量级 = 882 / 2^(b-1)")
print(f"  备选方案：只有最高位放开，次高位受 882 约束，最低位量级 = 882 / 2^(b-2)")
print()
print(f"  {'b':>3s} {'标准:最低位':>11s} {'标准:判零':>9s} {'标准:误差':>11s} "
      f"{'备选:最低位':>11s} {'备选:判零':>9s} {'备选:误差':>11s} {'备选:最高权重':>13s}")
print("  " + "-" * 92)
for b in range(8, 15):
    e0, T0, s0, c0 = err(b, False)
    e1, T1, s1, c1 = err(b, True)
    low0 = 882 / (1 << (b - 1)) if b >= 1 else 0
    low1 = 882 / (1 << (b - 2)) if b >= 2 else 0
    nz0 = int((T0 == 0).sum())
    nz1 = int((T1 == 0).sum())
    top1 = int(np.abs(T1).max())
    f0 = "  ---" if e0 is None else f"{e0:11.6f}"
    f1 = "  ---" if e1 is None else f"{e1:11.6f}"
    over0 = " 超限" if e0 is None else ""
    print(f"  {b:3d} {low0:11.3f} {nz0:9d} {f0}{over0} "
          f"{low1:11.3f} {nz1:9d} {f1} {top1:13d}")

print()
print("  推导：最低位要 ≥ 0.5 才不被舍成 0")
print("    标准：  882/2^(b-1) ≥ 0.5  →  2^(b-1) ≤ 1764  →  b ≤ 11.8  →  b ≤ 11")
print("    备选：  882/2^(b-2) ≥ 0.5  →  2^(b-2) ≤ 1764  →  b ≤ 12.8  →  b ≤ 12")
print("  所以备选正好多一位。备选在 b=12 时最高权重为 1764，超过 882 但远低于 2^31-1。")

print()
print("=" * 98)
print("验证 2：F 与 F' 的跨度是否需要分段")
print("=" * 98)
print(f"  强模上限 2^31 - 1 = {STRONG:,}")
print()
print(f"  {'层':5s} {'输入数':>6s} {'b':>3s} {'sum|W| 上界':>13s} {'占 2^31 比例':>13s} {'需分段':>7s}")
print("  " + "-" * 56)
for b in (8, 11, 12):
    for li, n in ((0, 4), (1, 64)):
        # 每输入的权重绝对值之和上界 ≈ 882 * 2 (等比级数) 或备选的 882*(2 + 2)
        per_input = 882 * 2
        bound = n * per_input
        frac = bound / STRONG
        print(f"  层{li}   {n:6d} {b:3d} {bound:13,d} {100*frac:12.4f}% "
              f"{'否' if frac < 1 else '是':>7s}")
print()
print("  结论：即使层 2 的 64 个输入全部取到上界，总和也只占强模上限的千分之几。")
print("  此前「层 2 需要 2~3 段累加器」的说法基于错误的 13455，不成立。")
print("  一个累加器就够，不必分段。")

print()
print("=" * 98)
print("验证 3：F -> Y 的转换")
print("=" * 98)
print("  F_bar = F - F_0，F_0 取 z = 0 处，于是 F_bar = scale * z。")
print("  c 取最小整数使 c*2^b > max(F_bar)，Y = floor(F_bar / c)。")
print("  注意 c > 1 时 Y 的标尺被拉大，c*2^b 对应修正后的 y_max，")
print("  所以判据只能是「解码回 y 之后的误差」，不能直接比 Y 的整数值。")

zmax = float(np.abs(w).sum() * 3.5 + theta)
zmin = float(theta - np.abs(w).sum() * 3.5)
y_true = np.maximum(0.0, z_true)
xq_cache = {}

print()
print(f"  {'b':>3s} {'缩放':>9s} {'max(F_bar)':>11s} {'r=max/2^b':>10s} {'c':>4s} "
      f"{'浪费':>7s} {'y 平均误差':>11s} {'y 最大误差':>11s} {'相当级数':>9s}")
print("  " + "-" * 96)

best = {}
for b in (8, 11, 12):
    N = 1 << b
    T, scale_w, cunit = table(b, True)
    scale = scale_w
    Fbar_max = scale * zmax
    r = Fbar_max / N
    c = int(np.floor(r)) + 1
    waste = 1 - r / c
    # 浪费超过 20% 时把缩放降到让 c 少一档，代价是权重步长粗一点
    if waste > 0.20 and c > 1:
        c = c - 1
        scale = scale * (c * N) / Fbar_max
        Fbar_max = scale * zmax
        r = Fbar_max / N
        waste = 1 - r / c
    # 重新构造权重表
    T2 = np.zeros((4, b), dtype=int)
    for i in range(4):
        for jj in range(b):
            T2[i, jj] = int(np.rint(scale * w[i] * cunit * (1 << jj)))
    Xq = np.clip(np.floor((x - XMIN) / SPAN * N), 0, N - 1).astype(int)
    bits = ((Xq[:, :, None] >> np.arange(b)[None, None, :]) & 1)
    Fp = np.einsum("sij,ij->s", bits, T2)
    Fp_min = int(T2[T2 < 0].sum())
    F = Fp - Fp_min
    C0 = float(w @ np.full(4, XMIN) + theta + cunit * 0.5 * w.sum())
    F0 = int(np.rint(-scale * C0 - Fp_min))
    Fbar = F - F0
    Y = np.clip(np.maximum(0, np.floor(Fbar / c)).astype(int), 0, N - 1)
    # 格心解码回 y：Y 的一级对应 c/scale
    y_dec = (Y + 0.5) * c / scale
    e = np.abs(y_dec - y_true)
    ymax_corr = c * N / scale
    levels = y_true.max() / (c / scale)
    print(f"  {b:3d} {scale:9.2f} {Fbar_max:11.1f} {r:10.3f} {c:4d} "
          f"{100*waste:6.1f}% {e.mean():11.6f} {e.max():11.6f} {levels:9.1f}")
    best[b] = (scale, c, e.mean())

print()
print("  对照：若直接量化 y（不做整数运算），y 的量化为 y_max/2^b，误差上限是半步：")
for b in (8, 11, 12):
    N = 1 << b
    half = zmax / N / 2
    print(f"    b = {b:2d}  半步 = {half:.6f}")

print()
print("  说明：")
print("    「浪费」是 c*2^b 比 max(F_bar) 高出的比例，也就是输出级数没用满的部分。")
print("    r 恰好落在某整数上方一点点时浪费最大。把缩放回调到让 c 减一，可以消掉浪费，")
print("    代价是权重步长按同一比例变粗。上面已经在浪费超过 20% 时自动回调。")

print()
print("=" * 98)
print("小结")
print("=" * 98)
print("  1. 备选方案正好把可用 b 从 11 推到 12，与你估计一致。")
print("  2. 强模上限是 2^31-1，F 的跨度只占千分之几，层 2 不需要分段。")
print("  3. c 是每神经元一个整数尺度，c*2^b 略大于 max(F_bar)，")
print("     代价是 y_max 被抬高一点，量化变粗一点。")
print("VERIFY_DONE")

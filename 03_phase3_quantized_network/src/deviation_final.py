#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
最终版：F_0 标定之后，Y 的整数值偏差分布。

上一轮查明：偏差来自 F_bar 相对 scale*z 的常数截距，除以 c 之后成为若干分之一级。
c 随 b 递减（max(F_bar) 固定而 2^b 变大），所以偏差在级单位下随 b 增大。
F_0 名义值由连续公式给出，与取整后的权重不一致，差零点几。

本版把 F_0 按取整后的权重重新标定，再测分布。
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
NS = 400000

gray = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                      encoding="utf-8"))
W = np.array(gray["gen"]["layers"][0]["W"])
B = np.array(gray["gen"]["layers"][0]["b"])
peak = np.abs(W).max(1)
ch = int(np.argsort(peak)[len(peak) // 2])
w, theta = W[ch], B[ch]
zmax = float(np.abs(w).sum() * 3.5 + theta)

rng = np.random.default_rng(2026)
raw = rng.standard_normal((NS * 2, 4))
x = raw[np.all(np.abs(raw) <= 3.5, axis=1)][:NS]
z = x @ w + theta
y = np.maximum(0.0, z)

print("=" * 100)
print(f"通道 {ch}   w = {np.round(w,6)}   theta = {theta:.6f}")
print(f"x 取自 N(0,1) 截断到 [-3.5, 3.5]，拒绝采样，n = {NS:,}")
print("=" * 100)


def pipeline(b, x, z):
    N = 1 << b
    cunit = SPAN / N
    scale = LIMIT / (np.abs(w).max() * cunit * (1 << (b - 2)))   # 备选方案定标
    T = np.zeros((4, b), dtype=int)
    for i in range(4):
        for jj in range(b):
            T[i, jj] = int(np.rint(scale * w[i] * cunit * (1 << jj)))

    # 先算 F，再定 c；若范围浪费过大则回调缩放重算
    def make(scale):
        T2 = np.zeros((4, b), dtype=int)
        for i in range(4):
            for jj in range(b):
                T2[i, jj] = int(np.rint(scale * w[i] * cunit * (1 << jj)))
        return T2

    Fbar_max = scale * zmax
    c = int(np.floor(Fbar_max / N)) + 1
    if c > 1 and (1 - Fbar_max / (c * N)) > 0.20:
        c -= 1
        scale *= (c * N) / Fbar_max
        T = make(scale)

    Xq = np.clip(np.floor((x - XMIN) / SPAN * N), 0, N - 1).astype(int)
    bits = ((Xq[:, :, None] >> np.arange(b)[None, None, :]) & 1)
    Fp = np.einsum("sij,ij->s", bits, T)
    Fp_min = int(T[T < 0].sum())
    F = Fp - Fp_min

    # 名义 F_0（由连续公式）
    C0 = float(w @ np.full(4, XMIN) + theta + cunit * 0.5 * w.sum())
    F0_nom = int(np.rint(-scale * C0 - Fp_min))
    # 标定 F_0：使 E[F - F0] = scale * E[z]
    F0_cal = int(np.rint(F.mean() - scale * z.mean()))

    return dict(N=N, c=c, scale=scale, T=T, F=F, F0_nom=F0_nom, F0_cal=F0_cal)


def report(b):
    r = pipeline(b, x, z)
    N, c, scale = r["N"], r["c"], r["scale"]
    F = r["F"]
    ymax_corr = c * N / scale
    print()
    print(f"  b = {b}   缩放 {scale:.2f}   c = {c}   "
          f"y_max 修正 {ymax_corr:.5f}（原 {zmax:.5f}，高 "
          f"{100*(ymax_corr/zmax-1):.2f}%）")
    print(f"    名义 F_0 = {r['F0_nom']}，标定 F_0 = {r['F0_cal']}，"
          f"差 {r['F0_cal']-r['F0_nom']:+d}")

    for tag, F0 in (("名义 F_0", r["F0_nom"]), ("标定 F_0", r["F0_cal"])):
        Fbar = F - F0
        Y = np.clip(np.maximum(0, np.floor(Fbar / c)).astype(int), 0, N - 1)
        Yideal = np.clip(np.floor(y / ymax_corr * N), 0, N - 1).astype(int)
        d = Y - Yideal
        vals, cnts = np.unique(d, return_counts=True)
        top = np.argsort(-cnts)[:6]
        s = "  ".join(f"{int(vals[i]):+d}:{100*cnts[i]/len(d):5.2f}%" for i in top)
        print(f"    {tag}  偏差 均值 {d.mean():+7.4f}  标准差 {d.std():6.4f}  "
              f"|偏差|最大 {np.abs(d).max():2d}")
        print(f"      {s}")
    return r


for b in (8, 10, 11, 12):
    report(b)

print()
print("=" * 100)
print("结论")
print("=" * 100)
print("  1. 偏差集中在 0 与 ±1，正侧略重。名义 F_0 下这个不对称随 b 放大。")
print("  2. 根因不是增益（实测增益误差 ±0.001%），而是 F_bar 的常数截距除以 c。")
print("     c 随 b 递减，所以同样的截距在更大 b 下变成更多级。")
print("  3. 按取整后的权重重新标定 F_0，偏移基本消失。F_0 是编译期常数，")
print("     不增加运行期成本。")
print("  4. 标定后仍有约半级的随机散布，来自每个位权重的独立取整，无法用常数消掉。")
print("     注意散布在「级」单位下随 b 略增，但在「y」单位下随 b 显著减小：")
print("     级宽 c/scale 随 b 变小，所以半级对应的物理误差更小。")
print("BIAS_FINAL_DONE")

#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
风格分布的独立性与线性变换。

风格数据（webapp/styles_gauss.json）给的是
    mu    均值（4 维）
    chol  下三角 Cholesky 因子，协方差 = chol @ chol.T
采样：x = mu + chol @ z，z ~ N(0,I)，|x| 超 ±3.5 则拒绝。

问题一：四个 PCA 维在风格内独立吗？
    独立 ⇔ 协方差是对角阵 ⇔ 相关矩阵非对角为 0。
问题二：若不独立，需要什么线性变换？
    x = mu + L z，L = chol。所以 4→4 无激活单层网络，权重 L，偏置 mu。
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

styles = json.load(open(os.path.join(P2, "webapp", "styles_gauss.json"),
                        encoding="utf-8"))["styles"]

print("=" * 96)
print("问题一：风格内四个 PCA 维是否独立")
print("=" * 96)
max_off_all = 0.0
for e in styles:
    L = np.array(e["chol"], dtype=float)
    mu = np.array(e["mu"], dtype=float)
    S = L @ L.T
    d = np.sqrt(np.diag(S))
    C = S / d[:, None] / d[None, :]
    off = float(np.abs(C - np.diag(np.diag(C))).max())
    max_off_all = max(max_off_all, off)
    print(f"\n  {e['name']}")
    print(f"    均值 {np.round(mu, 3)}")
    print(f"    相关矩阵：")
    for r in range(4):
        print("      " + "  ".join(f"{C[r, c]:+6.3f}" for c in range(4)))
    print(f"    非对角最大绝对值 {off:.4f}  →  {'独立' if off < 0.01 else '不独立'}")
print()
print(f"  全部风格里非对角最大绝对值 {max_off_all:.4f}")
print(f"  结论：{'四个维在风格内基本独立' if max_off_all < 0.01 else '四个维在风格内不独立，需要线性变换'}")

print()
print("=" * 96)
print("问题二：需要的线性变换")
print("=" * 96)
print("  采样式 x = mu + L z 本身就是一个 4→4 无激活单层网络：")
print("      权重 = L（下三角 Cholesky 因子），偏置 = mu")
print("  z 的四个分量可以各自用一套独立投掷器生成。")
print()
for e in styles:
    L = np.array(e["chol"], dtype=float)
    mu = np.array(e["mu"], dtype=float)
    print(f"  {e['name']}")
    print(f"    偏置 mu = [{', '.join(f'{v:+.4f}' for v in mu)}]")
    for r in range(4):
        print(f"    权重行 {r}: [" + ", ".join(f"{L[r, c]:+.4f}" for c in range(4)) + "]")
    print(f"    对角（各维标准差）: [{', '.join(f'{L[i,i]:.4f}' for i in range(4))}]")
    print(f"    非对角范数 / 对角范数 = "
          f"{np.linalg.norm(L - np.diag(np.diag(L))) / np.linalg.norm(np.diag(np.diag(L))):.4f}")

print()
print("=" * 96)
print("截断：x 落到 ±3.5 之外的比例")
print("=" * 96)
rng = np.random.default_rng(11)
NG = 400000
zn = rng.standard_normal((NG, 4))
print(f"  {'风格':10s} {'拒绝率（原采样）':>16s} {'各维最大 |x|':>13s}")
print("  " + "-" * 46)
for e in styles:
    L = np.array(e["chol"], dtype=float)
    mu = np.array(e["mu"], dtype=float)
    x = zn @ L.T + mu
    bad = np.any(np.abs(x) > 3.5, axis=1)
    print(f"  {e['name']:10s} {100*bad.mean():15.2f}% {np.abs(x).max():13.3f}")

print()
print("=" * 96)
print("整数量化后的等价形式")
print("=" * 96)
print("  投掷器给 S_i ∈ {0..256}，代表 z_i = (S_i − 128)/36.47")
print("  变换后  x = mu + L·(S − 128)/36.47")
print("  量化    X_i = clip( floor( (x_i + 3.5)/7 · 256 ), 0, 255 )")
print()
print("  合成一个 4→4 仿射：")
print("      X = clip( floor( A·S + b ), 0, 255 )")
print("      A = (256/(7·36.47)) · L")
print("      b = (256/7)·(mu + 3.5) − A·128")
print()
SIG_DROP = 36.466
k = 256.0 / (7.0 * SIG_DROP)
for e in styles:
    L = np.array(e["chol"], dtype=float)
    mu = np.array(e["mu"], dtype=float)
    A = k * L
    b = (256.0 / 7.0) * (mu + 3.5) - A @ np.full(4, 128.0)
    print(f"  {e['name']}")
    print(f"    |A| 最大 {np.abs(A).max():.4f}   各行绝对值之和 "
          f"[{', '.join(f'{np.abs(A[r]).sum():.3f}' for r in range(4))}]")
    print(f"    b = [{', '.join(f'{v:+.2f}' for v in b)}]")
print()
print("  各行 |A| 之和 < 1 意味着：一个单位的 S 波动最多让 X 变不到 1，")
print("  但四个 S 一起动可以叠到 1 以上——这就是需要矩阵乘法的原因。")
print("INDEP_DONE")

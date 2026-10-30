#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
风格高斯里有多少质量落在 ±3.5 立方体之外，以及投掷器路该怎么改。

问题
    原采样是「抽 z，算 x = mu + Lz，若 |x| 有分量超 3.5 就整个丢弃」。
    投掷器路做不到丢弃（要重抽循环），只能把超界的 x 截断到边界。
    若拒绝率是 47%，那就是 47% 的样本被压在立方体表面上——分布完全变了。

根因
    好几个风格的均值本身就在立方体之外（|mu| > 3.5）。
    这些风格实际是「被立方体裁出来的形状」，截断不是可有可无的装饰。

方案
    直接用「接受后的 x」重新拟合高斯 (mu', L')，把参数搬进立方体内部。
    再用这套参数喂投掷器，截断就只影响极小的一部分。
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

rng = np.random.default_rng(23)
NG = 400000

print("=" * 100)
print("一、均值在立方体内还是外")
print("=" * 100)
print(f"  {'风格':10s} {'mu 的范数':>10s} {'最大分量':>9s} {'在立方体内':>10s} {'拒绝率':>8s}")
print("  " + "-" * 54)
for e in styles:
    mu = np.array(e["mu"], dtype=float)
    L = np.array(e["chol"], dtype=float)
    z = rng.standard_normal((NG, 4))
    x = z @ L.T + mu
    bad = np.any(np.abs(x) > 3.5, axis=1)
    print(f"  {e['name']:10s} {np.linalg.norm(mu):10.3f} {np.abs(mu).max():9.3f} "
          f"{'是' if np.abs(mu).max() <= 3.5 else '否':>10s} {100*bad.mean():7.1f}%")

print()
print("=" * 100)
print("二、用「接受后的 x」重新拟合（投掷器友好的参数）")
print("=" * 100)
refit = {}
for e in styles:
    mu = np.array(e["mu"], dtype=float)
    L = np.array(e["chol"], dtype=float)
    z = rng.standard_normal((NG, 4))
    x = z @ L.T + mu
    keep = np.all(np.abs(x) <= 3.5, axis=1)
    xa = x[keep]
    mu2 = xa.mean(0)
    cov2 = np.cov(xa.T, bias=True)
    L2 = np.linalg.cholesky(cov2)
    refit[e["name"]] = (mu2, L2)
    # 检查新参数下再拒绝的比例
    z2 = rng.standard_normal((NG, 4))
    x2 = z2 @ L2.T + mu2
    bad2 = np.any(np.abs(x2) > 3.5, axis=1)
    print(f"\n  {e['name']}")
    print(f"    原 mu   = [{', '.join(f'{v:+.3f}' for v in mu)}]   拒绝率 {100*(1-keep.mean()):5.1f}%")
    print(f"    新 mu'  = [{', '.join(f'{v:+.3f}' for v in mu2)}]   拒绝率 {100*bad2.mean():5.1f}%")
    print(f"    原对角  [{', '.join(f'{L[i,i]:.3f}' for i in range(4))}]")
    print(f"    新对角  [{', '.join(f'{L2[i,i]:.3f}' for i in range(4))}]")
    # 分布距离：接受后的 x 与 新参数生成并截断的 x，比较各维直方图
    x3 = np.clip(x2, -3.5, 3.5)
    tv = 0.0
    for d in range(4):
        h1, _ = np.histogram(xa[:, d], bins=64, range=(-3.6, 3.6), density=False)
        h2, _ = np.histogram(x3[:, d], bins=64, range=(-3.6, 3.6), density=False)
        h1 = h1 / h1.sum()
        h2 = h2 / h2.sum()
        tv = max(tv, 0.5 * np.abs(h1 - h2).sum())
    print(f"    各维边缘分布的最大偏差（新参数+截断 vs 原始接受后）: {tv:.4f}")

outp = os.path.join(ROOT, "results", "styles_refit.json")
json.dump({k: {"mu": [float(v) for v in m], "chol": [[float(v) for v in row] for row in L]}
           for k, (m, L) in refit.items()}, open(outp, "w", encoding="utf-8"), indent=1)
print(f"\n  已写入 {outp}")
print("REFIT_DONE")

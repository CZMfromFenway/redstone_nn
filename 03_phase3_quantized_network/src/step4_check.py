#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
第 4 步：层 2 的位权重。为什么比层 1 复杂。

层 1 的输入步长是均匀的：delta_i = Delta / 2^b，四个输入一样。
层 2 的输入步长是逐输入的：delta_j = c_j / alpha_j，64 个输入各不相同。

层 1 的缩放可以逐神经元：alpha_j。
层 2 的缩放必须全局：beta。因为 32 个输出要互相比。

全局缩放的代价：|v| * delta 小的神经元，它的低位权重会被舍成 0。
这里量化这个代价。
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
mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
U1W = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
TH1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
V2W = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
PHI = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)

B = 8
NB = 1 << B
XLO, XHI = -3.5, 3.5
DELTA = XHI - XLO
LIMIT = 882

# ---- 第 2~3 步，得到层 1 的 alpha、c
alpha = LIMIT / (np.abs(U1W).max(1) * DELTA / 4)
U1 = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB)
             * (1 << np.arange(B))[None, None, :]).astype(np.int64)
Hmin = np.where(U1 < 0, U1, 0).sum(axis=(1, 2))
Fmax = np.where(U1 > 0, U1, 0).sum(axis=(1, 2)) - Hmin
A1 = np.rint(-alpha * (TH1 + XLO * U1W.sum(1)
                       + DELTA / (2 ** (B + 1)) * U1W.sum(1)) - Hmin).astype(np.int64)
c1 = np.maximum(np.ceil((Fmax - A1) / NB), 1).astype(np.int64)

delta2 = c1 / alpha

print("=" * 100)
print("层 2 的输入步长 δ_j = c_j / α_j")
print("=" * 100)
print(f"  范围 [{delta2.min():.6f}, {delta2.max():.6f}]，中位 {np.median(delta2):.6f}")
print(f"  最大 / 最小 = {delta2.max()/delta2.min():.1f} 倍   ← 层 1 的四个输入是 1.0 倍")

# ---- 层 2 的缩放 beta
g = np.abs(V2W) * delta2[None, :]              # (32, 64) 每个 (p,j) 的增益
gmax = g.max()
beta = LIMIT / (gmax * (1 << (B - 2)))
V2 = np.rint(beta * V2W[:, :, None] * delta2[None, :, None]
             * (1 << np.arange(B))[None, None, :]).astype(np.int64)

print()
print("=" * 100)
print("层 2 的缩放 β（全局）")
print("=" * 100)
print(f"  max|v·δ| = {gmax:.6f}")
print(f"  β = 882 / (max|v·δ| · 2^(b-2)) = {beta:.4f}")
print(f"  校验：k = b-2 的最大 |V| = {np.abs(V2[:, :, B-2]).max()}")
print(f"        k = b-1 的最大 |V| = {np.abs(V2[:, :, B-1]).max()}")

print()
print("=" * 100)
print("全局缩放的代价：逐神经元的增益比 r_p = max_j|v·δ| / 全局最大")
print("=" * 100)
rp = g.max(1) / gmax
print(f"  r_p 范围 [{rp.min():.5f}, {rp.max():.5f}]")
print(f"  r_p 的分位：p5 {np.percentile(rp,5):.4f}  中位 {np.median(rp):.4f}  "
      f"p95 {np.percentile(rp,95):.4f}")
print()
print("  有效位数 = 满足 |V^(k)| >= 1 的 k 的个数")
eff = (np.abs(V2) >= 1).sum(2)                 # (32, 64)
eff_p = eff.max(1)                             # 该神经元最好的那个输入的位数
print(f"  每神经元有效位数（取该神经元 64 个输入里最多的）：范围 [{eff_p.min()}, {eff_p.max()}]，"
      f"中位 {int(np.median(eff_p))}（满 {B}）")
print()
print(f"  {'r_p 分档':16s} {'神经元数':>8s} {'有效位数中位':>12s}")
print("  " + "-" * 44)
for lo, hi in ((0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 1.01)):
    m = (rp >= lo) & (rp < hi)
    if m.sum() == 0:
        continue
    print(f"  [{lo:.2f}, {hi:.2f})      {int(m.sum()):8d} {int(np.median(eff_p[m])):12d}")

print()
print("  为什么：k 位的权重 = 882·r_p·2^k / 2^(b-2)。")
print("  k=0 时要 >= 0.5，需要 r_p >= 0.5·2^(b-2)/882 = %.4f" % (0.5 * 2 ** (B - 2) / 882))
print("  k=1 时要 >= 0.5，需要 r_p >= %.4f" % (0.5 * 2 ** (B - 1) / 882))
print("  r_p 越小，能被表示的位数越少。")

print()
print("=" * 100)
print("对照：如果层 2 也用逐神经元缩放（不需要可比）")
print("=" * 100)
beta_pn = LIMIT / (g.max(1) * (1 << (B - 2)))
V2pn = np.rint(beta_pn[:, None, None] * V2W[:, :, None] * delta2[None, :, None]
               * (1 << np.arange(B))[None, None, :]).astype(np.int64)
eff_pn = (np.abs(V2pn) >= 1).sum(2).max(1)
print(f"  逐神经元缩放时：每神经元有效位数范围 [{eff_pn.min()}, {eff_pn.max()}]，"
      f"中位 {int(np.median(eff_pn))}")
print(f"  全局缩放时：    每神经元有效位数范围 [{eff_p.min()}, {eff_p.max()}]，"
      f"中位 {int(np.median(eff_p))}")
print(f"  丢失位数中位 {int(np.median(eff_pn - eff_p))}")
print("  但逐神经元缩放会让 32 个输出不可比，min-max 失效。这是必须付的代价。")
print("STEP4_DONE")

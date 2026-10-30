#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
每种方案下 Z 的量程与位宽，以及 min-max 的成本代理。

Z ∈ [0, 15β]，β = w_ref / (max|v_pj·δ_j| · 2^(b-2))
    w_ref 翻倍 → β 翻倍 → Z 量程翻倍 → 多一位
    b 变化 → δ_j 随之变（δ_j = c_j/α_j，c_j ~ 2^-b），所以乘积 2^(b-2)·δ 大致不变
    → Z 的位宽主要由 w_ref 决定，b 的影响次要（本脚本实测确认）
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
RESULTS = os.path.join(ROOT, "results")
P2 = os.path.join(os.path.dirname(ROOT), "02_phase2_avatar")

mj = json.load(open(os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json"),
                    encoding="utf-8"))
U1W = np.array(mj["gen"]["layers"][0]["W"], dtype=np.float64)
TH1 = np.array(mj["gen"]["layers"][0]["b"], dtype=np.float64)
V2W = np.array(mj["gen"]["layers"][1]["W"], dtype=np.float64)
PHI = np.array(mj["gen"]["layers"][1]["b"], dtype=np.float64)

DELTA = 7.0
XLO = -3.5
CAP = 882


def stats(b, w_ref):
    NB = 1 << b
    alpha = CAP / (np.abs(U1W).max(1) * DELTA / 4) * (w_ref / CAP)
    U = np.rint(alpha[:, None, None] * U1W[:, :, None] * (DELTA / NB)
                * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    Hmin = np.where(U < 0, U, 0).sum(axis=(1, 2))
    Hmax = np.where(U > 0, U, 0).sum(axis=(1, 2))
    A = np.rint(-alpha * (TH1 + XLO * U1W.sum(1)
                          + DELTA / (2 ** (b + 1)) * U1W.sum(1)) - Hmin).astype(np.int64)
    c = np.maximum(np.ceil((Hmax - Hmin - A) / (NB - 1)), 1).astype(np.int64)
    delta = c / alpha
    mx = float((np.abs(V2W) * delta[None, :]).max())
    beta = w_ref / (mx * (1 << (b - 2)))
    T = int(round(15 * beta))
    bits = int(np.ceil(np.log2(T + 1)))
    # 层 2 位权重
    V = np.rint(beta * V2W[:, :, None] * delta[None, :, None]
                * (1 << np.arange(b))[None, None, :]).astype(np.int64)
    Gmin = np.where(V < 0, V, 0).sum(axis=(1, 2))
    Gmax = np.where(V > 0, V, 0).sum(axis=(1, 2))
    Nmax = int((Gmax - Gmin).max())
    return dict(b=b, w_ref=w_ref, T=T, bits=bits,
                delta_min=float(delta.min()), delta_max=float(delta.max()),
                Nmax=Nmax, Nbits=int(np.ceil(np.log2(Nmax + 1))),
                Vtop=int(np.abs(V).max()))


print("=" * 104)
print("Z 的量程与位宽")
print("=" * 104)
print(f"  {'情形':12s} {'w_ref':>6s} {'b':>3s} {'15β = Z上界':>12s} {'Z 位数':>7s} "
      f"{'δ 范围':>22s} {'非负累加 N 上界':>14s} {'N 位数':>7s}")
print("  " + "-" * 96)
rows = []
for nm, wr in (("B 无环", 441), ("A 1 环", 882), ("C 2 环", 1764)):
    for b in (8, 9, 10, 11, 12):
        r = stats(b, wr)
        rows.append((nm, r))
        print(f"  {nm if b == 8 else '':12s} {wr:6d} {b:3d} {r['T']:12d} "
              f"{r['bits']:7d} "
              f"[{r['delta_min']:.6f}, {r['delta_max']:.6f}] "
              f"{r['Nmax']:14d} {r['Nbits']:7d}")

print()
print("=" * 104)
print("每种情形的推荐 b 下的 Z 位宽")
print("=" * 104)
best = {"B 无环": (9, 441, 5.69), "A 1 环": (10, 882, 3.00), "C 2 环": (11, 1764, 1.64)}
print(f"  {'情形':10s} {'b*':>3s} {'w_ref':>6s} {'Z 上界':>9s} {'Z 位数':>7s} "
      f"{'精度':>8s} {'min-max 成本（位）':>18s}")
print("  " + "-" * 66)
for nm, (b, wr, acc) in best.items():
    r = stats(b, wr)
    print(f"  {nm:10s} {b:3d} {wr:6d} {r['T']:9d} {r['bits']:7d} {acc:7.2f}% "
          f"{r['bits']:18d}")
print()
print("  旧方案：b=8, w_ref=882 → 4.65%")
r0 = stats(8, 882)
print(f"    Z 上界 {r0['T']}，{r0['bits']} 位")
print()
print("=" * 104)
print("关键观察")
print("=" * 104)
print("  1. w_ref 翻倍 → Z 上界翻倍 → 多 1 位。这与预期一致。")
print("  2. b 从 8 变到 12，Z 的上界几乎不动（δ 的 2^-b 被 2^(b-2) 抵消）")
print("     → Z 位宽主要由 w_ref 决定，b 只影响精度不影响量程。")
print("  3. 所以换方案时：")
print("       精度由 (w_ref, b) 共同决定；")
print("       min-max 体积只由 w_ref 决定。")
print()
print("  ⇒ 「B 方案省 min-max」是它的额外好处：11 位 vs A 的 12 位 vs C 的 13 位。")
print()
for nm, (b, wr, _) in best.items():
    r = stats(b, wr)
    print(f"    {nm}: Z 上界 {r['T']:6d} → {r['bits']} 位，"
          f"min-max 要处理 {r['bits']} 位二进制")
print()
print("  另外：非负累加量 N 的上界一直在 3 万左右，需要 15 位——")
print("  但 N 是中间量，不参与 min-max。")
print("SUMMARY_DONE")

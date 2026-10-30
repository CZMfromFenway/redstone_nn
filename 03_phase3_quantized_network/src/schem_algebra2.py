#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
修正：强模信号 S = 15 + |W| ∈ [15, 897]，所以 |W| ≤ 882 = W_ref。

单元
    W>0: 位=1 → 15 ；位=0 → S
    W<0: 位=1 → S  ；位=0 → 15

定义「激活指示」 x̄ = x (W>0) / 1-x (W<0)
    出 = 15 + |W|(1 - x̄)
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

B, NIN = 8, 4
rng = np.random.default_rng(11)

print("=" * 94)
print("一、信号量程")
print("=" * 94)
print("  S = 15 + |W|")
print("  S ∈ [15, 897]  ⇒  |W| ∈ [0, 882]")
print()
print("  ⚠ 882 正是我们的 W_ref！容器上限 897 与 15+|W| 的构造完全自洽。")
print("  ⚠ 一个木桶就能装下**全精度**的位权重，我上一条『量程只有 15』是错的。")

print()
print("=" * 94)
print("二、代数（这次应该完全对上）")
print("=" * 94)
print("  出 = 15 + |W|(1 - x̄)")
print("  ⇒ Σ出 = 15b + Σ|W| - Σ|W|x̄")
print()
print("  而 Σ|W|x̄ = Σ_{W>0} W·x + Σ_{W<0} |W|(1-x)")
print("            = Σ_k W_k x_k + Σ_{W<0}|W_k|")
print("            = Σ_k W_k x_k - Σ_k min(0, W_k)")
print("            = N   ← 就是非负累加量！")
print()
print("  ⇒  Σ出 = 15b + Σ|W| - N")
print()
print("  ⇒")
print("        N = K - Σ出 ,   K = 15b + Σ|W|")
print()
print("  ★ 两个环存的就是这个 K（配凑项）★")

print()
print("=" * 94)
print("三、数值验证")
print("=" * 94)
ok = True
for tr in range(4):
    W = rng.integers(-882, 883, size=(NIN, B))
    X = rng.integers(0, 2, size=(NIN, B))
    S = 15 + np.abs(W)
    out = np.zeros_like(W)
    for i in range(NIN):
        for k in range(B):
            active = (X[i, k] == 1) == (W[i, k] > 0)
            out[i, k] = 15 if active else S[i, k]
    got_sum = out.sum(axis=1)
    N = (W * X).sum(axis=1) - np.minimum(0, W).sum(axis=1)
    K = 15 * B + np.abs(W).sum(axis=1)
    back = K - got_sum
    d = np.abs(back - N).max()
    ok &= (d == 0)
    print(f"  试验 {tr+1}:  Σ出 = {got_sum.tolist()}")
    print(f"            K   = {K.tolist()}")
    print(f"            K-Σ出 = {back.tolist()}   N = {N.tolist()}   差 {d}")
print()
print(f"  → N = K - Σ出 严格成立：{'是' if ok else '否'}")

print()
print("=" * 94)
print("四、K 有多大")
print("=" * 94)
for wr, tag in ((0, "全 0 权重"), (882, "|W| 全取 882")):
    print(f"  {tag:>14s}:  K = 15×{B} + Σ|W| = {15*B} + {wr*B} = {15*B+wr*B}")
print()
print(f"  K 最大 = {15*B+882*B} ，远超单个 897 信号 → **所以必须用环存** ✓")
print(f"  这也解释了为什么 K 要拆成两个环：单环装不下 {15*B+882*B}。")

print()
print("=" * 94)
print("五、这个设计的三个漂亮之处")
print("=" * 94)
print("  ① 折叠免费：N 直接从「关态基值」里减出来了，不需要 Σmin(0,W) 的电路")
print("  ② 符号免费：W<0 只是把「15/S」两个出口对调，不需要负号")
print("  ③ 常数用环：K 是一个固定的、只依赖权重的大整数，正好适合烧录")

json.dump({"S_range": [15, 897], "W_max": 882,
           "algebra": "N = K - sum(out), K = 15b + sum|W|",
           "K_max": 15 * B + 882 * B},
          open(os.path.join(RESULTS, "schem_algebra2.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  已写入 results/schem_algebra2.json")
print("ALG2_DONE")

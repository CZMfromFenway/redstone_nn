#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
rsnn.schem 单层的代数分析。

按你给的单元行为
    比较器 rear = 木桶信号 S，side = 彩色方块（激活=15，否则=0）
    输出 ∈ { 高电平 h, 容器强度 S }

「彩色位为 1 → 输出 15；为 0 → 输出容器强度」意味着
    高位 h = 15，低位 S = 15 − |W|   （只有这样 S 才落在 0..15）
    |W| = h − S = 15 − S

于是单个（输入 i, 位 k）的贡献
    出 = 15 − |W|·(1 − x)   （W>0）
    出 = 15 − |W|·x         （W<0）

对 8 位求和
    Σ 出 = Σ W_k x_k + ( 15·b − Σ|W_k| )
                          └──────┬──────┘
                             常数 = Σ S_k = C_min
"""
import itertools
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

B = 8
NIN = 4
rng = np.random.default_rng(3)

print("=" * 92)
print("一、单元模型：两态输出")
print("=" * 92)
print("  设 W>0 时：x=1 → 15， x=0 → S")
print("     W<0 时：x=1 → S，  x=0 → 15   （符号反转）")
print()
print("  要让它代表 W·x，需要有  出(x=1) − 出(x=0) = ±|W|")
print("  →  |W| = 15 − S ，即 S = 15 − |W|")
print()
print("  可行范围：S ∈ [0,15] ⇒ |W| ∈ [0,15]")
print("  ⚠ 木桶信号只有 15 档 ⇒ **单个木桶最多表示 |W| ≤ 15**")

print()
print("=" * 92)
print("二、对 8 位求和的代数验证")
print("=" * 92)
ok = True
for trial in range(3):
    W = rng.integers(-14, 15, size=(NIN, B))      # |W| ≤ 14
    X = rng.integers(0, 2, size=(NIN, B))
    S = 15 - np.abs(W)
    out = np.zeros_like(W)
    for i in range(NIN):
        for k in range(B):
            hi = (X[i, k] == 1) == (W[i, k] > 0)
            out[i, k] = 15 if hi else S[i, k]
    # 结构实际得到的累加值
    got = out.sum(axis=1)
    # 理论值 = Σ W·x + K，K = Σ(15-|W| 或 15)
    # 常数项：W>0 时基值 S=15-|W|；W<0 时基值 15
    konst = np.where(W > 0, 15 - np.abs(W), 15).sum(axis=1)
    want = (W * X).sum(axis=1) + konst
    d = np.abs(got - want).max()
    ok &= (d == 0)
    print(f"  试验 {trial+1}: Σ出 = {got.tolist()}")
    print(f"           ΣWx+Cmin = {want.tolist()}   最大差 {d}")
print()
print(f"  → 代数完全成立：{'是' if ok else '否'}")
print()
print("  即结构自动给出 **N = C − C_min** 这一「折叠后」的量，不用额外电路。")

print()
print("=" * 92)
print("三、但位权重量程不够")
print("=" * 92)
print("  本项目 b=8、W_ref=882 时，层1 的位权重范围：")
print("     W^(0) = 1  …  W^(7) = 2×882 = 1764")
print(f"  而单个木桶只能到 |W| ≤ 15")
print(f"  ⇒ 缺口 {1764/15:.0f} 倍")
print()
print("  可能的补救：")
print("    (a) 每个权重用多个木桶串联（1764/15 ≈ 118 个，太多）")
print("    (b) 用比较器环存大权重（你之前说环存「固定常数」，不是权重）")
print("    (c) 强模信号分两级：先在小信号域算，再用环放大")
print("    (d) 这个样机**本来就是小权重的逻辑演示**，不是全精度版")

print()
print("=" * 92)
print("四、两条总线的功能推断")
print("=" * 92)
print("  x=11 列：9 个朝南比较器 + 11 个白色混凝土交替")
print("    → 白色方块被前一个比较器「充能」，再给后一个比较器提供信号")
print("    → 这就是你描述的传播机制 ✓")
print("    → 功能：**纵向汇流总线**（把 4 个位对周期的结果串起来）")
print()
print("  x=2 列：10 个朝北比较器 + 灰色基座交替")
print("    → 方向相反，功能：**回程/反馈总线**")
print()
print("  两个环：")
print("    灰环 (0-2, 0-2)  四角是浅灰混凝土 → 存常数 K1")
print("    白环 (11-13,23-25) 四角是白混凝土 → 存常数 K2")
print("    → 环体是「自持循环」，信号绕一圈不掉，因此能长期存一个 >15 的整数")
print("      （用绕圈次数或环内多级串联编码大数）")

json.dump({"unit_model": "out in {15, 15-|W|}, |W|<=15",
           "algebra": "sum_out = sum(W*x) + sum(S) = N",
           "weight_range_limit": 15},
          open(os.path.join(RESULTS, "schem_algebra.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  已写入 results/schem_algebra.json")
print("ALG_DONE")

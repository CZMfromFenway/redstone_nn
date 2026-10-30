#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
体积对比定稿（LUT 宽度用 4.5·n·(b/4)）。

修正后
    V_LUT   = 4.5 · n · (b/4) · m · 32 = 36 n m b
    V_bits  = 3b · 2n · 2m             = 12 n m b
    比值 = 3

交叉检查
    容器数比 = 4
    每容器体积：LUT 36/4 = 9 方块³；位权重 12/1 = 12 方块³
    ⇒ 体积比 = 4 × (9/12) = 3   ✓ 与上面一致
    而且 LUT 的 9 方块³/容器 与一期文档 V≈4n·h·L（L=36）反推的一致
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

N1, M1 = 4, 64
N2, M2 = 64, 32


def v_bits(n, m, b, rings):
    return (3 * b) * (2 * n) * (2 * m) * (b + rings) / b


def v_lut(n, m, b):
    return 4.5 * n * (b / 4) * m * 32


print("=" * 90)
print("一、单位体积系数")
print("=" * 90)
print("  V_bits = 3b · 2n · 2m      = 12 n m b")
print("  V_LUT  = 4.5·n·(b/4)·m·32  = 36 n m b")
print(f"  比值 = 36/12 = {36/12:.0f}")
print()
print("  交叉检查（容器口径）：")
print(f"    容器/权重   LUT 4b   vs   位权重 b       → 4 : 1")
print(f"    方块³/容器  LUT 9    vs   位权重 12      → 0.75 : 1")
print(f"    体积比 = 4 × 0.75 = {4*0.75:.0f}  ✓")

print()
print("=" * 90)
print("二、本项目（4→64→32）")
print("=" * 90)
print(f"  {'配置':>24s} {'层1':>10s} {'层2':>11s} {'合计':>11s} {'LUT 同位宽':>12s} {'比值':>6s}")
print("  " + "-" * 78)
rows = []
for bx, by, r1, r2, tag in (
        (8, 10, 1, 1, "b=8/10 单环（定稿）"),
        (8, 10, 0, 0, "b=8/10 无环（假想）"),
        (9, 10, 1, 1, "b=9/10 单环"),
        (8, 9, 1, 1, "b=8/9 单环"),
        (10, 10, 1, 1, "b=10/10 单环")):
    v1 = v_bits(N1, M1, bx, r1)
    v2 = v_bits(N2, M2, by, r2)
    vl = v_lut(N1, M1, bx) + v_lut(N2, M2, by)
    rows.append((tag, v1, v2, v1 + v2, vl, vl / (v1 + v2)))
    print(f"  {tag:>24s} {v1:10,.0f} {v2:11,.0f} {v1+v2:11,.0f} "
          f"{vl:12,.0f} {vl/(v1+v2):6.2f}")

print()
best = rows[0]
print(f"  定稿配置：{best[0]}")
print(f"    数模结合核心体积 ≈ {best[3]:,.0f} 方块³")
print(f"    强模 LUT 同位宽   ≈ {best[4]:,.0f} 方块³")
print(f"    比值 {best[5]:.2f}")

print()
print("=" * 90)
print("三、容器与环（定稿）")
print("=" * 90)
c1 = N1 * M1 * 8
c2 = N2 * M2 * 10
print(f"  层1 容器 4×64×8   = {c1:,}")
print(f"  层2 容器 64×32×10 = {c2:,}")
print(f"  合计              = {c1+c2:,}")
print(f"  环：层1 最高位 64 个 + 层2 最高位 32 个 = 96 个")
print()
print(f"  若改两层都用双环（W_ref 层1 也到 1764）：+64 个环，")
print(f"    体积 ×(8+2)/8 于层1 = {v_bits(N1,M1,8,2):,.0f}（原 {v_bits(N1,M1,8,1):,.0f}）")
print(f"    即层1 体积从 27,648 涨到 30,720，占比很小")

json.dump({"v_per_weight_bits": "12·n·m·b", "v_per_weight_lut": "36·n·m·b",
           "ratio": 3.0,
           "final": {"bx": 8, "by": 10, "rings": 96,
                     "containers": c1 + c2,
                     "volume": best[3], "lut_equiv": best[4]}},
          open(os.path.join(RESULTS, "volume_final.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  已写入 results/volume_final.json")
print("VOLF_DONE")

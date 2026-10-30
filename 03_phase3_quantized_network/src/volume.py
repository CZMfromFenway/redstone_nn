#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
体积对比：数模结合（位权重） vs 强模 LUT，只比全连接核心。

两个口径
    甲 容器数（可精确计算，无争议）
    乙 用户的体积模型（含走线/排版系数，需你确认含义）

用户给的模型
    数模结合: V ≈ 3b × 2n × 2m
    LUT     : V ≈ 4.5n × 1m × b × 32
    环      : 单侧 ×(b+1)/b，双侧 ×(b+2)/b
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

# ---------------- 目标网络（二期灰度头像）
N1, M1 = 4, 64        # 层1  输入, 输出
N2, M2 = 64, 32       # 层2
BX, BY = 8, 10        # 编译器选的最优位宽

print("=" * 92)
print("一、容器数（可精确计算，无排版假设）")
print("=" * 92)
print("  位权重：每（输入, 输出, 位）一个容器")
print("  LUT  ：每（输入, 输出, 半字节）16 个容器 = 每位的 4 倍")
print()
c_ours = N1 * M1 * BX + N2 * M2 * BY
c_lut = N1 * M1 * 4 * BX + N2 * M2 * 4 * BY
print(f"  数模结合  b_x={BX}, b_y={BY}:  {N1}x{M1}x{BX} + {N2}x{M2}x{BY} "
      f"= {c_ours:,} 个容器")
print(f"  强模 LUT  同位宽            :  ×4 = {c_lut:,} 个容器")
print(f"  比值 LUT/位权重 = {c_lut/c_ours:.2f}")
print()
print("  注意：这 4 倍与 b 无关，也与网络无关 —— 是结构常数。")

# ---------------- 用户的体积模型
print()
print("=" * 92)
print("二、按你的体积模型算")
print("=" * 92)


def v_ours(n, m, b, rings=0):
    return (3 * b) * (2 * n) * (2 * m) * (b + rings) / b


def v_lut(n, m, b):
    return 4.5 * n * m * b * 32


print(f"  {'配置':>26s} {'数模结合':>12s} {'强模 LUT':>12s} {'比值':>7s}")
print("  " + "-" * 62)
for bx, by, ring, tag in ((4, 4, 0, "b=4, 无环"),
                          (8, 8, 1, "b=8, 单环"),
                          (8, 10, 1, "b=8/10, 单环（本项目）"),
                          (10, 10, 1, "b=10, 单环"),
                          (11, 11, 2, "b=11, 双环"),
                          (12, 12, 2, "b=12, 双环")):
    vo = (3 * bx) * (2 * N1) * (2 * M1) + (3 * by) * (2 * N2) * (2 * M2)
    vo *= (by + ring) / by
    vl = 4.5 * N2 * M2 * by * 32 + 4.5 * N1 * M1 * bx * 32
    print(f"  {tag:>26s} {vo:12,.0f} {vl:12,.0f} {vl/vo:7.2f}")

print()
print("  单项系数拆解（层 2，64→32）：")
print(f"    数模结合 3b × 2n × 2m = 3×{BY} × {2*N2} × {2*M2} = "
      f"{3*BY*2*N2*2*M2:,}")
print(f"    LUT     4.5n × m × b×32 = 4.5×{N2} × {M2} × {BY*32} = "
      f"{4.5*N2*M2*BY*32:,.0f}")

# ---------------- 一期文档自己的估值
print()
print("=" * 92)
print("三、一期文档（05_强模LUT数学性质 §7）的体积估计")
print("=" * 92)
print("  原文：宽度 4 格/输入线，高度 1 格/神经元，长度 32~40（16 容器×2）")
print("        V ≈ 4n · h · L,  L ≈ 32~40")
print()
for L in (32, 36, 40):
    v1 = 4 * N2 * M2 * L
    print(f"    L={L}:  层2 体积 ≈ {v1:,.0f} 方块³   每容器 "
          f"{v1/(16*N2*M2):.2f} 方块³")
print()
print("  折算：LUT 的体积 ≈ 每容器 9 方块³（L=36）")
print(f"        位权重的容器数 {c_ours:,}，若同密度则体积 ≈ "
      f"{c_ours*9:,.0f}")
print(f"        实际 LUT 若 {c_lut:,} 容器 × 9 = {c_lut*9:,.0f}")
print()
print("  这个口径下比值 = 4.00（与容器比一致）")

json.dump({"containers_ours": c_ours, "containers_lut": c_lut,
           "ratio_containers": c_lut / c_ours},
          open(os.path.join(RESULTS, "volume_compare.json"), "w", encoding="utf-8"),
          indent=1)
print("\n  已写入 results/volume_compare.json")
print("VOL_DONE")

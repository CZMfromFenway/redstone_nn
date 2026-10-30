#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
容器信号可达性复核（对齐 Minecraft Wiki「红石比较器 § 检测容器」的权威口径）

Wiki 给出（以其计算器字段为准）：
    equiv64  = unstackable*64 + stack16*4 + stack64      # 折算成"64 堆叠当量"的计数
    signal   = 1 + floor(14*floor(equiv64) / (container*64))   (equiv64>0)，否则 0
    usedSlots= unstackable + ceil(stack16/16) + ceil(stack64/64)
    usedSlots > container  → 需要命令（超叠/超格）

注意 http 的"槽位占用"是按**正常堆叠**算的，所以超叠（一格塞 64 个不可堆叠）
在 Wiki 口径里属于"需命令"，不是生存可得的常规配置。

本脚本回答三件事：
  1. 修正 03/src/container_reach.py 的一个建模缺陷：它的 DP 要求"每格都非空"，
     而 Wiki 显然允许空格（空容器 signal=0）。
  2. 分别给出「计入 / 不计入 16 堆叠」的可达信号数，回答"雪球是否偏保守"。
  3. 细调带：k 个槽做细调时，可达信号的上界在哪。
"""
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

NONSTACK = set(range(64, 64 * 64 + 1, 64))   # 1..64 个不可堆叠 → 64..4096（步 64）
S16      = set(range(4, 4 * 64 + 1, 4))      # 1..64 个 16 堆叠  → 4..256（步 4）
STACK    = set(range(1, 65))                 # 1..64 个可堆叠   → 1..64（步 1）
EMPTY    = {0}

def reach(n, kinds, allow_empty):
    per = sorted(set().union(*[kinds[k] for k in kinds]) | (EMPTY if allow_empty else set()))
    cap = 64 * 64 * n
    cur = np.zeros(cap + 1, dtype=bool)
    cur[0] = True
    for _ in range(n):
        nxt = np.zeros(cap + 1, dtype=bool)
        for d in per:
            if d > cap:
                break
            nxt[d:] |= cur[:cap + 1 - d]
        cur = nxt
    return cur

def sig_of(U, n):
    return 0 if U == 0 else 1 + (14 * U) // (64 * n)

def coverage(n, kinds, allow_empty=True):
    R = reach(n, kinds, allow_empty)
    Us = np.flatnonzero(R)
    S = np.array([sig_of(u, n) for u in Us])
    have = np.zeros(898, dtype=bool)
    have[S] = True
    return have

def first_gap(have):
    g = np.flatnonzero(~have)
    return int(g[0]) if len(g) else None

def line(label, have):
    print(f"    {label:34s} 可达 {have.sum():3d}/898 ({100*have.sum()/898:5.1f}%)"
          f"  首个缺口 {first_gap(have)}")

print("=" * 84)
print("一、修正建模缺陷：是否允许空格")
print("=" * 84)
for nm, n in (("木桶", 27), ("投掷器", 9), ("漏斗", 5)):
    print(f"\n{nm}（n={n}）")
    a = coverage(n, {"A": NONSTACK, "B": S16, "C": STACK}, allow_empty=True)
    b = coverage(n, {"A": NONSTACK, "B": S16, "C": STACK}, allow_empty=False)
    line("允许空格（正确）", a)
    line("强制每格非空（原脚本口径）", b)
    print(f"    → 原脚本因强制非空，少算了 {int(a.sum()-b.sum())} 个信号")

print()
print("=" * 84)
print("二、雪球（16 堆叠）到底有没有被算进去")
print("=" * 84)
for nm, n in (("木桶", 27), ("投掷器", 9), ("漏斗", 5)):
    print(f"\n{nm}（n={n}）")
    w = coverage(n, {"A": NONSTACK, "B": S16, "C": STACK})
    wo = coverage(n, {"A": NONSTACK, "C": STACK})
    line("计 A+B+C（含 16 堆叠）", w)
    line("只计 A+C（无 16 堆叠）", wo)
    only = np.flatnonzero(w & ~wo)
    print(f"    → 16 堆叠 净带来 {int(w.sum()-wo.sum())} 个信号"
          + (f"，例如 {only[:12].tolist()}" if len(only) else ""))

print()
print("=" * 84)
print("三、细调带：k 个槽做细调时，信号能推到多高")
print("=" * 84)
print("  E = U/64（格当量）。1 个槽的贡献上限：不可堆叠 64 E、16 堆叠 4 E、可堆叠 1 E")
for nm, n in (("木桶", 27), ("投掷器", 9), ("漏斗", 5)):
    print(f"\n{nm}（n={n}）：E 上限 {64*n}")
    for k in range(1, min(n, 5) + 1):
        E = 64 * (n - k) + 4 * k
        U = 64 * E
        print(f"    k={k} 个细调槽（全塞 16 堆叠）→ 细调带顶 E={E:4d}, U={U:6d}, 信号 {sig_of(U, n)}")
    print(f"    全槽满 16 堆叠只能到 E={4*n}，所以细调槽越多，能到的信号越低")

print()
print("=" * 84)
print("四、你的例子：漏斗 前 4 格超叠斧头 + 第 5 格换物品")
print("=" * 84)
n = 5
A = 4 * 4096
print(f"  前 4 格 = 4×(64 斧) = {A} 单位（E=256）")
for label, vals in (("第 5 格 1..64 个斧头", range(64, 4097, 64)),
                    ("第 5 格 1..64 个雪球", range(4, 257, 4)),
                    ("第 5 格 1..64 个可堆叠", range(1, 65))):
    sigs = sorted({sig_of(A + d, n) for d in vals})
    print(f"    {label:22s} 覆盖信号 {sigs[0]}..{sigs[-1]}：{len(sigs)} 个"
          f"  （{sigs[:5]} … {sigs[-1]}）")

print()
print("=" * 84)
print("五、结论")
print("=" * 84)
print("  1. 原脚本强制每格非空 → 数字偏小；修正后见上表，应以修正值为准")
print("  2. 16 堆叠早就被计入（PER_SLOT 含 S16），730/804 不是漏算雪球造成的")
print("  3. 雪球的唯一作用：把「细调槽」的上限从 1 E 抬到 4 E")
print("  4. 细调带上界 = 64(n−k) + 4k；要上 897 必须全槽超叠斧头，此时只剩整数 E")

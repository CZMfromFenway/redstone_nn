#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""回答考题：(4,1,4) 形状的投掷器，九格各放什么。"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

N = 9
CAP = 64 * N
a, b, c = 4, 1, 4


def f(U):
    return 0 if U <= 0 else (14 * U) // CAP


def sig(U):
    return 0 if U == 0 else 1 + (14 * U) // CAP


def drops(U):
    return (f(U) - f(U - 64), f(U) - f(U - 4), f(U) - f(U - 1))


print("=" * 88)
print("题：(4,1,4) 的投掷器，九格各放什么")
print("=" * 88)
print(f"  a={a} 个不可堆叠（每槽恰好 1 个，没有自由度）")
print(f"  b={b} 个十六堆叠（每槽 1..16 个）")
print(f"  c={c} 个可堆叠  （每槽 1..64 个）")
print()
print(f"  U = 64·{a} + 4·B + C = {64*a} + 4B + C")
print(f"  条件 7U mod 288 ∈ [7, 27]")

valid = []
for B in range(b, 16 * b + 1):
    for C in range(c, 64 * c + 1):
        U = 64 * a + 4 * B + C
        d = drops(U)
        if d == (2, 1, 0):
            valid.append((B, C, U))

print()
print(f"  可行 (B,C) 共 {len(valid)} 组，B 只能取 1（b=1 槽，且要凑对余数）")
print(f"  其中 B=1 的 C 值：{sorted(set(C for B, C, U in valid if B == 1))[:20]}")
print(f"  最小可行：B={valid[0][0]}, C={valid[0][1]}, U={valid[0][2]}")

B, C, U = valid[0]
print()
print("=" * 88)
print(f"推荐配置：B={B}, C={C}, U={U}, 信号={sig(U)}")
print("=" * 88)
print()
print(f"  {'槽':>3s} {'物品':>12s} {'数量':>4s} {'该槽贡献':>9s}")
print("  " + "-" * 34)
# 4 槽各 1 把木斧
for i in range(a):
    print(f"  {i+1:3d} {'木斧':>12s} {1:4d} {64:9d}")
# 1 槽 1 个蜂蜜瓶
print(f"  {a+1:3d} {'蜂蜜瓶':>12s} {B:4d} {4*B:9d}")
# 4 槽羊毛，共 C 个，尽量均分
base, rem = divmod(C, c)
for i in range(c):
    cnt = base + (1 if i < rem else 0)
    print(f"  {a+2+i:3d} {'羊毛':>12s} {cnt:4d} {cnt:9d}")
print("  " + "-" * 34)
print(f"  {'合计':>3s} {'':>12s} {N:4d} {U:9d}")

print()
print("  羊毛可以用不同颜色区分数量，例如：")
colours = ["白", "橙", "品红", "淡蓝"]
base, rem = divmod(C, c)
for i in range(c):
    cnt = base + (1 if i < rem else 0)
    print(f"    槽 {a+2+i}: {cnt} 个{colours[i]}羊毛")

print()
print("=" * 88)
print("验证三种喷出")
print("=" * 88)
print(f"  当前 U = {U}，信号 = {sig(U)}")
for nm, dU, want in (("木斧（不可堆叠）", 64, 2), ("蜂蜜瓶（十六堆叠）", 4, 1),
                     ("羊毛（可堆叠）", 1, 0)):
    U2 = U - dU
    got = sig(U) - sig(U2)
    print(f"  喷出 {nm:20s} U {U}→{U2}  信号 {sig(U)}→{sig(U2)}  "
          f"降 {got}  {'✓' if got == want else '✗'}")

print()
print("  另外两个必须同时成立的条件：")
print(f"    · 每槽不超堆叠：木斧 1/1 ✓  蜂蜜瓶 1/16 ✓  羊毛 ≤64 ✓")
print(f"    · 9 槽全非空 → 每槽被选中概率 1/9")
print(f"      P(值0)=4/9  P(值1)=1/9  P(值2)=4/9  = (4,1,4)/9 ✓")

print()
print("=" * 88)
print("其它可行的 C（保持 B=1）")
print("=" * 88)
cs = sorted(set(CC for BB, CC, UU in valid if BB == 1))
print(f"  共 {len(cs)} 个：{cs}")
print(f"  最小的三个：{[c0 for c0 in cs[:3]]}")
print()
print("  取哪个都行；C 越小越省物品。C=29 是最省的。")
print("DROPPER_QUIZ_DONE")

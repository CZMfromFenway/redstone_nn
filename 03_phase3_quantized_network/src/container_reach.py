#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
木桶 / 漏斗 / 投掷器 的 0..897 信号强度可达性与所需物品。

一、通用信号公式
    容器 n 格。定义「单位」U：一格装满 64 个可堆叠物品 = 64 单位。
        可堆叠物品（最大堆叠 64）×1 个 = 1 单位
        十六堆叠物品（最大堆叠 16）×1 个 = 4 单位
        不可堆叠物品（最大堆叠 1）×1 个 = 64 单位
    于是
        signal = 0                          (U = 0)
        signal = 1 + floor(14·U / (64n))    (U > 0)
    满格 U = 64n  →  signal = 15
    单格超叠不可堆叠到 64 个 → 每格贡献 64×64 = 4096 单位 = 64 格当量
    全格如此 → U = 64·64n  →  signal = 1 + 14·64 = 897

二、三种容器的差别只在 n
    木桶 n=27  漏斗 n=5  投掷器 n=9
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
RESULTS = os.path.join(HERE, "results")

# 单格可贡献的单位数
NONSTACK = set(range(64, 64 * 64 + 1, 64))     # 不可堆叠：1..64 个 → 64..4096
S16 = set(range(4, 4 * 64 + 1, 4))             # 十六堆叠：1..64 个 → 4..256
STACK = set(range(1, 65))                      # 可堆叠：1..64 个 → 1..64
#
# 2026-10-31 修（kb 待提案登记的那一条）：**空槽必须计入**。
#   原写法 PER_SLOT = NONSTACK | S16 | STACK 里没有 0，等于强制「每格都非空」，
#   于是空容器（U = 0 → signal = 0）在 DP 里不可达：
#       木桶被算成 897/898、漏斗 789/898（各少一个信号，正是 0），
#       且各信号段的「可选 U 数」也系统性偏小。
#   空槽是合法状态（信号 0 只能由它给出），必须并进来。
#   修后应与 kb/10_游戏机制/容器信号.md 的定案数字一致：
#       木桶 898/898（100%）、漏斗 790/898（88.0%）、首缺口 730。
#   注：此前文档里写的 898/898 是对的（另有算法给出），错的是本脚本的 DP，
#   所以本修复不改变任何已发表结论，只是让脚本与结论对齐。
EMPTY = {0}                                    # 空槽 → 0 单位
PER_SLOT = sorted(EMPTY | NONSTACK | S16 | STACK)


def reach(n):
    """n 格容器的可达 U 集合（布尔数组）。"""
    cap = 64 * 64 * n
    cur = np.zeros(cap + 1, dtype=bool)
    cur[0] = True
    for _ in range(n):
        nxt = np.zeros(cap + 1, dtype=bool)
        for d in PER_SLOT:
            if d > cap:
                break
            nxt[d:] |= cur[:cap + 1 - d]
        cur = nxt
    return cur


def signal_of(U, n):
    return 0 if U == 0 else 1 + (14 * U) // (64 * n)


print("=" * 100)
print("一、通用信号公式")
print("=" * 100)
print("  signal = 1 + floor(14·U/(64n))，U 单位为「可堆叠物品当量」")
print(f"  {'容器':8s} {'格数':>4s} {'满格 U':>8s} {'满格信号':>8s} "
      f"{'极限 U':>10s} {'极限信号':>8s}")

for nm, n in (("木桶", 27), ("漏斗", 5), ("投掷器", 9)):
    print(f"  {nm:8s} {n:4d} {64*n:8d} {signal_of(64*n, n):8d} "
          f"{64*64*n:10d} {signal_of(64*64*n, n):8d}")

print()
print("=" * 100)
print("二、木桶（27 格）与漏斗（5 格）的可达性")
print("=" * 100)
for nm, n in (("木桶", 27), ("漏斗", 5)):
    R = reach(n)
    Us = np.flatnonzero(R)
    sigs = np.array([signal_of(u, n) for u in Us])
    uniq = np.unique(sigs)
    print(f"\n  {nm}（n={n}）")
    print(f"    可达 U 个数 {len(Us)}/{64*64*n+1}"
          f"（{100*len(Us)/(64*64*n+1):.1f}%）")
    print(f"    可达信号 {len(uniq)}/898（{100*len(uniq)/898:.1f}%）")
    # 每个信号对应的 U 个数（分辨率）
    mult = np.zeros(898, dtype=int)
    for s in sigs:
        mult[s] += 1
    ok = mult > 0
    print(f"    最高可达信号 {uniq.max()}")
    print()
    print(f"    {'信号段':>12s} {'可达':>6s} {'平均每个信号的可选 U 数':>22s}")
    for lo, hi in ((0, 99), (100, 299), (300, 599), (600, 799), (800, 850),
                   (851, 890), (891, 897)):
        seg = mult[lo:hi + 1]
        got = int((seg > 0).sum())
        avg = float(seg[seg > 0].mean()) if got else 0.0
        print(f"    {f'{lo}..{hi}':>12s} {got:3d}/{hi-lo+1:3d} {avg:22.2f}")

print()
print("=" * 100)
print("三、投掷器（9 格）：三种喷出物要分别让信号 −2 / −1 / 0")
print("=" * 100)
print("  每个槽放 1 个物品，共 9 个。形状 (a,b,c)：a 个不可堆叠、b 个十六堆叠、c 个可堆叠")
print("  U = 64a + 4b + c")
print()
print("  要求（记 f(U) = floor(14U/576)）：")
print("    f(U) − f(U−64) = 2     ← 喷出不可堆叠，信号 −2")
print("    f(U) − f(U−4)  = 1     ← 喷出十六堆叠，信号 −1")
print("    f(U) − f(U−1)  = 0     ← 喷出可堆叠，信号不变")
print()


def drops(U, n):
    f = lambda x: (14 * x) // (64 * n) if x > 0 else 0
    return f(U) - f(max(0, U - 64)), f(U) - f(max(0, U - 4)), f(U) - f(max(0, U - 1))


print(f"  {'(a,b,c)':>10s} {'U':>5s} {'信号':>5s} {'喷出后 −2/−1/0 ?':>18s}")
print("  " + "-" * 46)
good = []
for a in range(10):
    for b in range(10 - a):
        c = 9 - a - b
        U = 64 * a + 4 * b + c
        d1, d2, d3 = drops(U, 9)
        okk = (d1 == 2 and d2 == 1 and d3 == 0)
        if okk:
            good.append((a, b, c, U))
        if a + b + c == 9 and (okk or (a, b, c) in ((9, 0, 0), (0, 0, 9), (0, 9, 0),
                                                   (4, 1, 4), (3, 3, 3))):
            print(f"  {str((a,b,c)):>10s} {U:5d} {signal_of(U,9):5d} "
                  f"{f'{d1}/{d2}/{d3}':>10s} {'✓' if okk else '✗'}")

print()
print(f"  满足 −2/−1/0 的形状共 {len(good)}/55 个：")
for a, b, c, U in good:
    print(f"    (a,b,c)=({a},{b},{c})  U={U}  signal={signal_of(U,9)}  "
          f"7U mod 288 = {(7*U) % 288}")

print()
print("  备注：每喷出一个，U 就下降，下一次的落点会变。")
print("        所以「喷出后读出信号差」只能可靠用一次；连续喷需要每次先复位。")

# 出口：木桶/漏斗在特定信号下的推荐物品配置
print()
print("=" * 100)
print("四、给定信号求最小物品配置（木桶 / 漏斗）")
print("=" * 100)


def config_for(sig, n, prefer_fine=True):
    """返回 (U, [(kind, count_per_slot), ...]) 或 None。"""
    cap = 64 * 64 * n
    lo = max(1, -(-(sig - 1) * 64 * n // 14))
    hi = -(-sig * 64 * n // 14) - 1
    hi = min(hi, cap)
    if lo > hi:
        return None
    # 在 [lo, hi] 里找一个能用 n 格表达的 U：贪心用大块 + 余数
    for U in range(lo, hi + 1):
        rest = U
        slots = []
        # 先用不可堆叠凑大块（每格最多 4096）
        while rest >= 64 and len(slots) < n:
            take = min(4096, (rest // 64) * 64)
            slots.append(("不可堆叠", take // 64))
            rest -= take
        while rest >= 4 and len(slots) < n:
            take = min(256, (rest // 4) * 4)
            slots.append(("十六堆叠", take // 4))
            rest -= take
        if rest > 0 and len(slots) < n:
            slots.append(("可堆叠", rest))
            rest = 0
        if rest == 0 and len(slots) <= n:
            return U, slots
    return None


for nm, n in (("木桶", 27), ("漏斗", 5)):
    print(f"\n  {nm}（{n} 格）")
    print(f"    {'信号':>5s} {'U':>7s} {'塞法':>10s} {'详列'}")
    for sig in (1, 15, 100, 255, 511, 700, 864, 882, 897):
        r = config_for(sig, n)
        if r is None:
            print(f"    {sig:5d} 不可达")
            continue
        U, slots = r
        txt = "  ".join(f"{k}×{c}" for k, c in slots)
        print(f"    {sig:5d} {U:7d} {len(slots):10d} {txt}")

json.dump({"formula": "signal = 1 + 14*U//(64*n); U: 1 per stackable item, "
                      "4 per 16-stack item, 64 per non-stackable item",
           "containers": {"barrel": 27, "hopper": 5, "dropper": 9},
           "dropper_valid_shapes": [{"abc": [a, b, c], "U": U}
                                    for a, b, c, U in good]},
          open(os.path.join(RESULTS, "container_reach.json"), "w",
               encoding="utf-8"), indent=1)
print()
print("  已写入 results/container_reach.json")
print("CONTAINER_DONE")

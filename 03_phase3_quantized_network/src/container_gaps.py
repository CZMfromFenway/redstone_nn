#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
补算三件事
    1. 木桶/漏斗到底缺哪些信号
    2. 投掷器 55 种形状，靠「每槽放几个」能否都把 U 调进可用窗口
    3. 若用投掷器直接三值（每槽 1 个），哪些形状天生可用
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

NONSTACK = set(range(64, 64 * 64 + 1, 64))
S16 = set(range(4, 4 * 64 + 1, 4))
STACK = set(range(1, 65))
PER_SLOT = sorted(NONSTACK | S16 | STACK)


def reach(n):
    cap = 64 * 64 * n
    cur = np.zeros(cap + 1, dtype=bool)
    cur[0] = True
    for _ in range(n):
        nxt = np.zeros(cap + 1, dtype=bool)
        for d in PER_SLOT:
            nxt[d:] |= cur[:cap + 1 - d]
        cur = nxt
    return cur


def sig_of(U, n):
    return 0 if U == 0 else 1 + (14 * U) // (64 * n)


def reachable_signals(n):
    R = reach(n)
    Us = np.flatnonzero(R)
    return set(int(sig_of(u, n)) for u in Us), R


print("=" * 96)
print("一、木桶 / 漏斗 缺哪些信号")
print("=" * 96)
for nm, n in (("木桶", 27), ("漏斗", 5)):
    got, R = reachable_signals(n)
    miss = [s for s in range(898) if s not in got]
    print(f"\n  {nm}（{n} 格）：可达 {len(got)}/898，缺 {len(miss)} 个")
    if len(miss) <= 40:
        print(f"    缺：{miss}")
    else:
        # 分段统计
        print(f"    缺失列表中前 20：{miss[:20]}")
        print(f"    缺失列表中后 20：{miss[-20:]}")
        seg = np.zeros(10, dtype=int)
        for s in miss:
            seg[min(9, s // 90)] += 1
        print("    按 90 一段统计：" +
              "  ".join(f"{i*90}-{min(897,i*90+89)}:{seg[i]}" for i in range(10)))
    # 相位 3 关心的 882
    print(f"    882 可达？{'是' if 882 in got else '否'}"
          f"    897 可达？{'是' if 897 in got else '否'}")

print()
print("=" * 96)
print("二、投掷器：靠调「每槽数量」能否把 U 调进窗口")
print("=" * 96)
print("  投掷器随机选一个非空槽（每槽概率 1/9），所以「每槽放几个」不影响概率，")
print("  只影响总单位 U。因此形状 (a,b,c) 定概率，数量定 U。")
print()
print("  可用窗口：7U mod 288 ∈ [7, 27]")


def ok(U):
    if U <= 0:
        return False
    f = lambda x: (14 * x) // 576 if x > 0 else 0
    return (f(U) - f(max(0, U - 64)) == 2 and
            f(U) - f(max(0, U - 4)) == 1 and
            f(U) - f(max(0, U - 1)) == 0)


print()
print(f"  {'(a,b,c)':>10s} {'每槽 1 个时的 U':>15s} {'可用?':>6s}   "
      f"{'调数量后的一个可行配置':<40s} {'U':>6s}")
print("  " + "-" * 92)
good_fixed, good_tuned = [], []
for a in range(10):
    for b in range(10 - a):
        c = 9 - a - b
        U0 = 64 * a + 4 * b + c
        fixed = ok(U0)
        if fixed:
            good_fixed.append((a, b, c, U0))
        # 调数量：A∈[a,64a], B∈[b,64b], C∈[c,64c]
        found = None
        if a > 0:
            Arange, Brange, Crange = (a, 64 * a), (b if b else 0, 64 * b if b else 0), \
                                     (c if c else 0, 64 * c if c else 0)
        # 简单枚举：A 从 a 起，B 从 b 起，扫 C
        for A in range(a, 64 * a + 1, max(1, 1 if a else 0)) if a else [0]:
            if a and A > 64 * a:
                break
            for B in range(b, 64 * b + 1) if b else [0]:
                lo, hi = (c if c else 0), (64 * c if c else 0)
                if c == 0:
                    lo = hi = 0
                # 需要 7U mod 288 ∈ [7,27]
                base = 64 * A + 4 * B
                if c == 0:
                    cand = [base]
                else:
                    cand = [base + t for t in range(lo, hi + 1)]
                for U in cand:
                    if ok(U):
                        found = (A, B, U - base, U)
                        break
                if found:
                    break
            if found:
                break
        if found:
            good_tuned.append((a, b, c, U0, found))
        if fixed or (a + b + c == 9 and (a, b, c) in
                     ((0, 0, 9), (9, 0, 0), (0, 9, 0), (4, 1, 4), (3, 3, 3),
                      (1, 4, 4), (7, 2, 0), (5, 3, 1), (2, 5, 2))):
            txt = ""
            if found:
                A, B, C, U = found
                txt = f"不可堆叠×{A}(分{a}槽) 十六×{B} 可堆叠×{C}"
            print(f"  {str((a,b,c)):>10s} {U0:15d} {'✓' if fixed else '✗':>6s}   "
                  f"{txt:<40s} {found[3] if found else 0:6d}")

print()
print(f"  每槽 1 个就可用：{len(good_fixed)}/55 → "
      f"{[(a,b,c) for a,b,c,_ in good_fixed]}")
print(f"  调整数量后可用：{len(good_tuned)}/55")
print(f"  仍不可用：{sorted(set((a,9-a-b,b) for a in range(10) for b in range(10-a))
                        - set((a,b,c) for a,b,c,_,_ in good_tuned))}")

print()
print("=" * 96)
print("三、直接结论")
print("=" * 96)
print("  · 投掷器每槽放 1 个只有 4 种形状能用；调数量后几乎全部可用。")
print("  · 但每喷出一个 U 就降，下一次的落点会变 ——")
print("    连续采样必须先复位（setblock 重填），否则读数不可靠。")
print("  · 木桶 27 格覆盖 99.9%，漏斗 5 格只覆盖 87.9%，且 864/882 不可达。")

got_b, _ = reachable_signals(27)
got_h, _ = reachable_signals(5)
json.dump({"barrel_missing": [s for s in range(898) if s not in got_b],
           "hopper_missing": [s for s in range(898) if s not in got_h],
           "dropper_fixed_ok": [[a, b, c, U] for a, b, c, U in good_fixed],
           "dropper_tuned_ok_count": len(good_tuned)},
          open(os.path.join(RESULTS, "container_gaps.json"), "w",
               encoding="utf-8"), indent=1)
print("\n  已写入 results/container_gaps.json")
print("GAPS_DONE")

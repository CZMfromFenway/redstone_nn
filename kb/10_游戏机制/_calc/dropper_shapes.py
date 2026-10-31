#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""投掷器三值随机源：验证"只能 0/1/2 档"与"7U mod 288 ∈ [7,27]"，并给出各形状最简装填。

模型（n=9）：
    U = 64a + 4β + γ          a/b/c 为三类槽数，β∈[b,16b]、γ∈[c,64c]
    s = 1 + floor(14U / 576) = 1 + floor(7U / 288)
    开火丢一件 → ΔU = 64 / 4 / 1
    要求三个下降量分别恰为 2 / 1 / 0
"""
from itertools import product

N = 9
STEP = 64 * N / 14.0          # 信号每涨 1 档所需的 U 增量 ≈ 41.14


def s(U):
    return 1 + (14 * U) // (64 * N)


def drops(U):
    """返回 (丢不可堆叠, 丢16堆叠, 丢可堆叠) 各自的信号下降量。"""
    return s(U) - s(U - 64), s(U) - s(U - 4), s(U) - s(U - 1)


def ok(U):
    return drops(U) == (2, 1, 0)


def best_for(a, b, c):
    """该形状下物品总数最少的一组 (总数, U, β, γ)。"""
    best = None
    for B in range(b, 16 * b + 1):
        for C in range(c, 64 * c + 1):
            U = 64 * a + 4 * B + C
            if ok(U):
                tot = a + B + C
                if best is None or tot < best[0]:
                    best = (tot, U, B, C)
    return best


def main():
    print(f"信号步长 = 576/14 = {STEP:.3f}；64/{STEP:.3f} = {64/STEP:.3f}")
    print("→ 丢一件最多跨 2 档（64 < 3×41.14 = 123.4），且只有不可堆叠能到 2 档")
    print()
    print(f"{'形状':>10} {'P(2,1,0)':>22} {'装填(斧/瓶/毛)':>14} {'物品':>4} {'U':>5} {'7U%288':>6}")
    shapes = [(3, 3, 3), (5, 1, 3), (7, 1, 1), (4, 1, 4), (2, 2, 5), (0, 1, 8)]
    for a, b, c in shapes:
        r = best_for(a, b, c)
        if r is None:
            print(f"({a},{b},{c})  无解")
            continue
        tot, U, B, C = r
        p = f"{a/9:.3f},{b/9:.3f},{c/9:.3f}"
        print(f"{str((a,b,c)):>10} {p:>22} {f'{a}/{B}/{C}':>14} {tot:>4} {U:>5} {(7*U)%288:>6}")

    print()
    print("窗口检验：7U mod 288 的可行区间")
    good = [U for U in range(1, 64 * 64 * N, 1) if ok(U)]
    mods = sorted({(7 * U) % 288 for U in good})
    print(f"  100 万个 U 里满足三档的，7U%288 只取 {len(mods)} 个值：{mods[:5]} … {mods[-5:]}")
    print(f"  即 [7, 27]，宽度 21/288 = {21/288*100:.1f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""投掷器三值随机源：核对"三档"的成立条件，并给出各形状最简装填。

模型（容器格数 n=9 固定）：
    U = 64a + 4β + γ          a/b/c 为三类**占用槽数**，β∈[b,16b]、γ∈[c,64c]
    s = 1 + floor(14U / 576) = 1 + floor(7U / 288)      ← 分母是容器格数 9
    开火丢一件 → ΔU = 64 / 4 / 1

概率：
    投掷器只从**非空槽**里等概率挑一个，所以
        k = a + b + c        （占用槽数，1 ≤ k ≤ 9）
        P(Δs=2,1,0) = (a/k, b/k, c/k)
    k=1 时没有随机性；k≥2 才有意义。
"""
from itertools import product

N = 9
STEP = 64 * N / 14.0          # 信号每涨 1 档所需的 U 增量 ≈ 41.14


def s(U, n=N):
    return 1 + (14 * U) // (64 * n)


def drops(U):
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


def row(a, b, c, note=''):
    r = best_for(a, b, c)
    k = a + b + c
    if r is None:
        return f'({a},{b},{c}) 无解'
    tot, U, B, C = r
    p = f'{a}/{k}, {b}/{k}, {c}/{k}'
    return (f'({a},{b},{c})  k={k}  P=({p})  装填 斧{a}/雪{B}/毛{C}  '
            f'总数{tot}  U={U}  7U%288={(7*U)%288}  {note}')


def main():
    print(f'信号步长 = 576/14 = {STEP:.3f}；64/{STEP:.3f} = {64/STEP:.3f}')
    print('→ 丢一件最多跨 2 档（64 < 3×41.14 = 123.4），且只有不可堆叠能到 2 档')
    print()
    print('【甲】分母 = 9 的形状')
    for shp in [(3, 3, 3), (5, 1, 3), (7, 1, 1), (4, 1, 4), (2, 2, 5), (0, 1, 8)]:
        print('  ' + row(*shp))
    print()
    print('【乙】不占满 9 格的形状（分母更小，可用分布更多）')
    for shp in [(1, 1, 0), (1, 0, 1), (1, 3, 0), (2, 1, 0), (3, 2, 0), (2, 3, 0)]:
        print('  ' + row(*shp))
    print()
    print('【丙】窗口检验：7U mod 288 的可行区间')
    good = [U for U in range(1, 64 * 64 * N) if ok(U)]
    mods = sorted({(7 * U) % 288 for U in good})
    print(f'  满足三档的 U，7U%288 只取 {len(mods)} 个值：[{mods[0]} .. {mods[-1]}]，'
          f'宽度 {len(mods)}/288 = {len(mods)/288*100:.1f}%')


if __name__ == '__main__':
    main()

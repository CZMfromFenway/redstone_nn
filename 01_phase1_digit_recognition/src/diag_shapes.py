#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
从截图重建图案并诊断
======================
重建用户画的"环形 0"（第 1 张截图）与"粗 0"（第 5 张）等，
计算各层 F，并分析为什么判成 5 / 3。
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
TARGET = 128
GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]

# 从截图重建（# = 点亮）。行 0 在顶部
RING = [
    "...............",
    "...............",
    "......##.......",
    ".....####......",
    "....##..##.....",
    "....#....##....",
    "...#.....##....",
    "...#.....##....",
    "...#.....##....",
    "...#.....##....",
    "....#...##.....",
    "....##..##.....",
    ".....####......",
    "......##.......",
    "...............",
]


def parse(rows):
    """rows: 字符串列表 -> (15,15) bool。用 '#' 表示点亮。
    兼容用户画的形状：这里直接用字符画。"""
    m = np.zeros((15, 15), dtype=bool)
    for r, line in enumerate(rows[:15]):
        for c, ch in enumerate(line[:15]):
            if ch in '#1':
                m[r, c] = True
    return m


def analyze(name, m):
    W = np.load(os.path.join(RES, 'Wint_model.npz'))['W'].astype(int)
    bias = np.load(os.path.join(RES, 'Wint_model.npz'))['b'].astype(int)
    x = m.reshape(-1).astype(np.int64)
    Z = x @ W + bias
    F = Z + TARGET
    order = np.argsort(-F)
    print(f"\n{'='*78}")
    print(f"{name}   点亮 {int(m.sum())} 格")
    print('=' * 78)
    for r in range(15):
        print('  ' + ''.join('#' if m[r, c] else '.' for c in range(15)))
    print(f"\n  {'层(数字)':>9} | {'z':>5} | {'F':>5} | 排序")
    print("  " + "-" * 34)
    for rank, k in enumerate(order, 1):
        mark = '  <<< 最大' if rank == 1 else ''
        print(f"  {k+1:>4}({k})   | {Z[k]:>5} | {F[k]:>5} | #{rank}{mark}")
    print(f"\n  -> 识别为 {order[0]}，F={F[order[0]]}，"
          f"次大 {order[1]} F={F[order[1]]}，margin={F[order[0]]-F[order[1]]}")

    # 逐支线（针对识别结果层）
    k = int(order[0]); ln = k + 1
    par = '偶数' if ln % 2 == 0 else '奇数'
    c = (15 - W[:, k]) if par == '偶数' else (15 + W[:, k])
    print(f"\n  第 {ln} 层（数字 {k}，{par}层）逐支线: G_i0 - Σ输出 = G_i")
    tot = 0
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = np.array([r * 15 + cc for r in range(r0, r0 + nr) for cc in range(15)])
        cg = c[pxs]
        g0 = int(np.maximum(cg, 15).sum())
        L = m.reshape(-1)[pxs]
        out = np.where(L, 15, cg) if par == '偶数' else np.where(L, cg, 15)
        s = int(out.sum()); g = g0 - s; tot += g
        print(f"    G_{gi+1}: G_i0={g0:>4}  Σ输出={s:>4}  G={g:>4}  "
              f"(该支线点亮 {int(L.sum())}/{len(pxs)})")
    print(f"    ΣG_i = {tot}   F = F_0 - ΣG_i = {F[k]}")
    return int(order[0])


def main():
    print("=" * 78)
    print("图案诊断")
    print("=" * 78)
    m = parse(RING)
    analyze("重建：环形 0（细笔画，类似截图1）", m)

    # 把它加粗（模拟用户在 MC 里用粗笔画）
    DISK = [(dy, dx) for dy in range(-1, 2) for dx in range(-1, 2)]
    mm = m.copy()
    for it in (1, 2):
        acc = mm.copy()
        for dy, dx in DISK:
            acc |= np.roll(np.roll(mm, dy, 0), dx, 1)
        mm = acc
        analyze(f"重建：环形 0 加粗 {it} 次", mm)


if __name__ == '__main__':
    main()

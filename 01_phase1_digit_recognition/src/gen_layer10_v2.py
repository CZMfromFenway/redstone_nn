#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
第十层（数字 9）整层 225 个容器  —— x 奇偶修正版
====================================================
坐标规则（最终）:
  * y 决定层 -> 第 10 层整层 y = -1
  * x 沿列: 奇数行（13,11,9,...）最右列 x=0  ->  最左列 x=-28
            偶数行（14,12,10,...）最右列 x=-1 ->  最左列 x=-29
            同一行内每往左一个像素 x -= 2
  * z 沿行: 13->0, 14->-2, 11->-4, 12->-6, 9->-8, 10->-10,
            7->-12, 8->-14, 5->-16, 6->-18, 3->-20, 4->-22,
            1->-24, 2->-26, 0->-30      （相邻差 2，唯 2->0 差 4）

容器值: 第 10 层（数字 9）为偶数层 -> c = 15 - W
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DP = os.path.join(os.path.dirname(ROOT), 'datapack', 'tiny_cnn')
FUNC_DIR = os.path.join(DP, 'data', 'deploy_weights', 'function')

COLS, ROWS = 15, 15
ORDER = [13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0]
ZSTEP = [2] * 13 + [4]
Y_LAYER = -1


def build_Z():
    Z, z = {}, 0
    for i, r in enumerate(ORDER):
        Z[r] = z
        if i < len(ZSTEP):
            z -= ZSTEP[i]
    return Z


def x_start(row):
    """最右列（col=14）的 x：奇数行 0，偶数行 -1"""
    return 0 if row % 2 == 1 else -1


def main():
    W = np.load(os.path.join(ROOT, 'results', 'Wint_model.npz'))['W'].astype(int)
    Z = build_Z()
    k = 9
    cvals = 15 - W[:, k]
    os.makedirs(FUNC_DIR, exist_ok=True)

    print("=" * 88)
    print("第十层（数字 9）整层 225 个容器  —— x 已修正")
    print("=" * 88)
    print(f"\n  {'像素行':>6} | {'奇偶':>4} | {'z':>5} | {'最右列 x':>8} | {'最左列 x':>8}")
    print("  " + "-" * 50)
    for r in ORDER:
        xs = x_start(r)
        print(f"  {r:>6} | {'奇' if r%2 else '偶':>4} | {Z[r]:>5} | {xs:>8} | "
              f"{xs-28:>8}")

    lines = ["# 第十层（数字 9）整层放置 -- 由 gen_deploy.py 生成",
             "# 执行位置: ~ ~ ~",
             f"# 整层 y = {Y_LAYER}，15x15 = 225 个容器",
             "# x: 奇数行 0..-28 ; 偶数行 -1..-29 （每列 -2）",
             "# z: 行 13=0, 14=-2, 11=-4, 12=-6, ..., 2=-26, 0=-30",
             ""]
    meta = []
    for r in ORDER:
        z = Z[r]; xs = x_start(r)
        lines.append(f"# --- 像素行 {r} (z={z}, 最右列 x={xs}) ---")
        for col in range(COLS - 1, -1, -1):
            x = xs - 2 * (COLS - 1 - col)
            sig = int(cvals[r * 15 + col])
            lines.append(f"execute at @s positioned {x} {Y_LAYER} {z} run "
                         f"function vessel:barrel_{sig}")
            meta.append({'row': r, 'col': col, 'x': x, 'y': Y_LAYER,
                         'z': z, 'signal': sig})
        lines.append("")

    path = os.path.join(FUNC_DIR, 'place_layer10_digit9.mcfunction')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    sigs = [m['signal'] for m in meta]
    print(f"\n[已生成] {path}")
    print(f"  指令数 {len(meta)}   信号范围 [{min(sigs)}, {max(sigs)}]")
    print(f"  x 取值: " +
          ", ".join(str(v) for v in sorted(set(m['x'] for m in meta))))
    print(f"  z 取值: " +
          ", ".join(str(v) for v in sorted(set(m['z'] for m in meta), reverse=True)))

    print(f"\n[文件前 20 条]")
    with open(path, encoding='utf-8') as f:
        for i, ln in enumerate(f):
            if i >= 21:
                break
            print('  ' + ln.rstrip())

    json.dump({'layer': 10, 'digit': 9, 'y': Y_LAYER, 'order': ORDER,
               'Z': {str(a): b for a, b in Z.items()}, 'cells': meta},
              open(os.path.join(ROOT, 'results', 'layer10_deploy.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n明细已存: results/layer10_deploy.json")


if __name__ == '__main__':
    main()

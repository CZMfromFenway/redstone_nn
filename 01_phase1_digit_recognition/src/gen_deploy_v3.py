#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
生成 deploy_weights 放置函数（最终坐标规则 + 真实容器值）
==========================================================
坐标规则（czm 澄清）:
  * z 从大到小对应的像素行: 13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0
  * 最右列 x 在 -1 / 0 之间交错（奇数行 x=-1，偶数行 x=0）
  * 同一行内，每往左一个像素 x-2
  * y 恒为 -1
  * z 序列: 除"2 与 0 之间差 4"，其余相邻差 2
      13 -> z0
      14 -> z-2
      11 -> z-4
      12 -> z-6
      ...
      1  -> z-24
      2  -> z-26
      0  -> z-30

容器值:
  偶数层（第 2,4,6,8,10 层）: c = 15 - W
  奇数层（第 1,3,5,7,9 层） : c = 15 + W
  最顶层容器层关联数字 9 -> 用第 10 层（偶数层）的权重
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DP = os.path.join(os.path.dirname(ROOT), 'datapack', 'tiny_cnn')
FUNC_DIR = os.path.join(DP, 'data', 'deploy_weights', 'function')

ROWS, COLS = 15, 15
Y_BASE = -1
# z 从大到小对应的像素行（czm 给定）
ORDER = [13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0]
# z 值：相邻差 2，唯 2 与 0 之间差 4
Z = {}
z = 0
for i, r in enumerate(ORDER):
    Z[r] = z
    if i + 1 < len(ORDER):
        z -= 2 if not (r == 2 and ORDER[i + 1] == 0) else 4


def x_right(row):
    """最右列像素的 x：奇数行 -1，偶数行 0"""
    return -1 if row % 2 == 1 else 0


def x_of(row, col):
    return x_right(row) - 2 * (14 - col)


def main():
    W = np.load(os.path.join(ROOT, 'results', 'Wint_model.npz'))['W'].astype(int)
    os.makedirs(FUNC_DIR, exist_ok=True)

    print("=" * 90)
    print("行 → 坐标映射（最终版）")
    print("=" * 90)
    print(f"  {'序':>3} | {'像素行':>6} | {'奇偶':>4} | {'z':>5} | {'最右 x':>7} | "
          f"{'最左 x':>7}")
    print("  " + "-" * 52)
    for i, r in enumerate(ORDER):
        print(f"  {i:>3} | {r:>6} | {'奇' if r%2 else '偶':>4} | {Z[r]:>5} | "
              f"{x_right(r):>7} | {x_of(r,0):>7}")

    # 检查 z 差
    print(f"\n  z 相邻差: ", end='')
    diffs = [Z[ORDER[i]] - Z[ORDER[i+1]] for i in range(len(ORDER)-1)]
    print(diffs)

    # ---- 容器层归属（每层两行）----
    print(f"\n[容器层归属]")
    layers = []
    for i in range(0, len(ORDER), 2):
        pair = ORDER[i:i+2]
        layers.append(pair)
        digit = 9 - len(layers) + 1
        print(f"  层{len(layers)-1} (数字 {digit}): 像素行 {pair}, "
              f"z = {[Z[r] for r in pair]}")

    # ---- 生成最顶层（像素行 13，z=0）----
    k = 9                       # 数字 9 -> 第 10 层（偶数）
    parity = '偶数'
    row = 13
    cvals = 15 - W[:, k] if parity == '偶数' else 15 + W[:, k]
    print(f"\n[最顶层] 像素行 {row}, z={Z[row]}, 关联数字 {k}（第 {k+1} 层, {parity}层）")

    lines = ["# 由 gen_deploy.py 自动生成",
             "# 执行位置: ~ ~ ~",
             f"# 最顶层容器层: 像素行 {row}, z={Z[row]}, 关联数字 {k}（{parity}层）",
             ""]
    rows_out = []
    for col in range(COLS - 1, -1, -1):
        p = row * 15 + col
        sig = int(cvals[p])
        lines.append(f"execute at @s positioned {x_of(row,col)} {Y_BASE} {Z[row]} "
                     f"run function vessel:barrel_{sig}")
        rows_out.append({'row': row, 'col': col, 'x': x_of(row, col),
                         'y': Y_BASE, 'z': Z[row], 'signal': sig})
    path = os.path.join(FUNC_DIR, 'place_top_layer.mcfunction')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[已生成] {path}   指令数 {len(rows_out)}")

    print(f"\n  信号值序列（col 14 -> 0）: {[d['signal'] for d in rows_out]}")
    print(f"  信号范围 [{min(d['signal'] for d in rows_out)}, "
          f"{max(d['signal'] for d in rows_out)}]")

    # 全层坐标存档
    coords = {}
    for r in range(ROWS):
        for c in range(COLS):
            coords[f"{r},{c}"] = [x_of(r, c), Y_BASE, Z[r]]
    json.dump({'order': ORDER, 'Z': {str(k): v for k, v in Z.items()},
               'layers': [[int(r) for r in p] for p in layers],
               'coords': coords,
               'top_layer_signals': [d['signal'] for d in rows_out]},
              open(os.path.join(ROOT, 'results', 'vessel_coords.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"坐标映射已存: results/vessel_coords.json")


if __name__ == '__main__':
    main()

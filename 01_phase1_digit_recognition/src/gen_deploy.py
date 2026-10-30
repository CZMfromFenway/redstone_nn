#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
生成 deploy_weights 数据包函数：把容器放到正确位置
=====================================================
坐标规则（czm 给定）:
  执行位置在 ~ ~ ~
  * 最顶层的容器在 z-1（最高的一层）
  * 每向下一层，z 减 2
  * 同一行像素内，最右像素在 x=0；每往左一个像素，x 减 2
  * 最右列像素的基准坐标，按像素行给出:
      像素行 14（最底行）  : y-1, z-1
      像素行 13           : y-1, z-2
      像素行 12           : y-1, z-5
      像素行 11           : y-1, z-6
      ... 每向上一个像素行, z 减 2

本脚本先生成**最顶层**（即像素行 14，z-1 那一层）的放置函数。

用法:
  python gen_deploy.py            # 生成最顶层
  python gen_deploy.py --all      # 生成全部 15 层
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# 数据包路径
DP = os.path.join(os.path.dirname(ROOT), 'datapack', 'tiny_cnn')
FUNC_DIR = os.path.join(DP, 'data', 'deploy_weights', 'function')

# 空间规则
Z_TOP = -1          # 最顶层 z
Z_STEP = -2         # 每向下一层的 z 变化
X_STEP = -2         # 每往左一个像素，x 变化
Y_BASE = -1         # 最右列像素的 y（最底一条基准线上）
ROWS = 15
COLS = 15


def row_z_offset(row):
    """
    返回像素行 row 相对"最顶层 z"的偏移。
    从最底层（row 14）向上:
        row 14 -> 0
        row 13 -> -1
        row 12 -> -4
        row 11 -> -5
        ...
        即: 先 -1，此后每两行 -4
    """
    if row == ROWS - 1:
        return 0
    k = (ROWS - 1) - row            # 向上数第 k 行, k>=1
    if k == 1:
        return -1
    return -1 - 4 * ((k - 1) // 2) - 2 * ((k - 1) % 2)


def main():
    args = sys.argv[1:]
    do_all = '--all' in args
    os.makedirs(FUNC_DIR, exist_ok=True)

    print("=" * 84)
    print("生成 deploy_weights 放置函数")
    print("=" * 84)
    print(f"数据包: {DP}")
    print(f"输出到: {FUNC_DIR}")

    # 列出每行基准坐标（最右列像素）
    print(f"\n[各像素行的基准坐标（最右列 col=14）]")
    print(f"  {'像素行':>6} | {'y':>4} | {'z':>5} | 说明")
    print("  " + "-" * 50)
    for r in range(ROWS):
        z = Z_TOP + row_z_offset(r)
        tag = ''
        if r == ROWS - 1:
            tag = '最底层（= 最顶层容器层）'
        elif r == ROWS - 2:
            tag = '次底层'
        elif r in (ROWS - 3, ROWS - 4):
            tag = ''
        print(f"  {r:>6} | {Y_BASE:>4} | {z:>5} | {tag}")

    # 生成函数
    rows_to_do = [ROWS - 1] if not do_all else list(range(ROWS - 1, -1, -1))
    empty = all(True for _ in rows_to_do)
    lines = []
    lines.append("# 由 gen_deploy.py 自动生成")
    lines.append("# 执行位置: ~ ~ ~")
    lines.append("# 坐标规则: 每往左一个像素 x-2；每向上一层 z-2；"
                 "像素行13额外 -1")
    lines.append("")
    n_place = 0
    for r in rows_to_do:
        z = Z_TOP + row_z_offset(r)
        lines.append(f"# --- 像素行 {r} (y={Y_BASE}, z={z}) ---")
        for c in range(COLS - 1, -1, -1):
            x = 0 + X_STEP * (COLS - 1 - c)
            lines.append(
                f"execute at @s positioned {x} {Y_BASE} {z} run "
                f"function vessel:barrel_0")
            n_place += 1
        lines.append("")

    name = 'place_top_layer.mcfunction' if not do_all else 'place_all_layers.mcfunction'
    path = os.path.join(FUNC_DIR, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\n[已生成] {path}")
    print(f"  放置指令数: {n_place}")
    print(f"  覆盖像素行: {rows_to_do}")

    # 打印最顶层的详细坐标表，便于核对
    r = ROWS - 1
    z = Z_TOP + row_z_offset(r)
    print(f"\n[最顶层坐标明细] 像素行 {r}, y={Y_BASE}, z={z}")
    print(f"  {'像素列':>6} | {'x':>5} | {'y':>4} | {'z':>5}")
    print("  " + "-" * 32)
    for c in range(COLS):
        x = 0 + X_STEP * (COLS - 1 - c)
        print(f"  {c:>6} | {x:>5} | {Y_BASE:>4} | {z:>5}")

    # 存一份坐标映射供后续用
    coord = {}
    for rr in range(ROWS):
        zz = Z_TOP + row_z_offset(rr)
        for cc in range(COLS):
            coord[f"{rr},{cc}"] = [0 + X_STEP * (COLS - 1 - cc), Y_BASE, zz]
    json.dump({'rule': {'Z_TOP': Z_TOP, 'Z_STEP': Z_STEP, 'X_STEP': X_STEP,
                        'Y_BASE': Y_BASE},
               'coords': coord},
              open(os.path.join(ROOT, 'results', 'vessel_coords.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n坐标映射已存: results/vessel_coords.json")


if __name__ == '__main__':
    main()

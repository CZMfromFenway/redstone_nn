#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
生成 deploy_weights 放置函数（按 czm 澄清后的规则）
=====================================================
坐标规则:
  * 执行位置 ~ ~ ~
  * 最顶层容器层（z-1）对应**像素行 14（偶）与行 13（奇）**
    - 行 14（最右列 col14）: ~-1 ~-1 ~-2
    - 行 13（最右列 col14）: ~   ~-1 ~
  * 偶数行: z = -2 - 2*(14-n)
  * 奇数行: z = -4 * floor((13-n)/2)
  * 同一行内: 最右 col14 在 x0，每往左一个像素 x-2

  即"两行一组构成一个容器层"，层间 z 差 4

输出:
  datapack/tiny_cnn/data/deploy_weights/function/place_top_layer.mcfunction
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DP = os.path.join(os.path.dirname(ROOT), 'datapack', 'tiny_cnn')
FUNC_DIR = os.path.join(DP, 'data', 'deploy_weights', 'function')

ROWS, COLS = 15, 15
Y_BASE = -1
X_STEP = -2


def z_of(row):
    if row % 2 == 0:
        return -2 - 2 * (14 - row)
    return -4 * ((13 - row) // 2)


def x_of(col):
    return X_STEP * (COLS - 1 - col)


def main():
    os.makedirs(FUNC_DIR, exist_ok=True)
    print("=" * 84)
    print("行 → 坐标映射（澄清后）")
    print("=" * 84)
    print(f"  {'像素行':>6} | {'奇偶':>4} | {'最右列 z':>8} | {'最右列 x':>8} | "
          f"{'容器层':>6}")
    print("  " + "-" * 48)
    lay = {}
    for r in range(ROWS):
        z = z_of(r)
        # 层号: z 排序
        print(f"  {r:>6} | {'偶' if r%2==0 else '奇':>4} | {z:>8} | {x_of(COLS-1):>8} | "
              f"{'':>6}")
    print()
    print(f"  {'层':>3} | {'z':>5} | 像素行")
    print("  " + "-" * 30)
    zs = sorted(set(z_of(r) for r in range(ROWS)), reverse=True)
    for i, z in enumerate(zs):
        rs = [r for r in range(ROWS) if z_of(r) == z]
        print(f"  {i:>3} | {z:>5} | {rs}")

    # ---- 生成最顶层 ----
    top_z = max(z_of(r) for r in range(ROWS))
    top_rows = sorted([r for r in range(ROWS) if z_of(r) == top_z], reverse=True)
    print(f"\n[最顶层] z = {top_z}, 像素行 = {top_rows}")

    lines = ["# 由 gen_deploy.py 自动生成（澄清后的坐标规则）",
             "# 执行位置: ~ ~ ~",
             f"# 最顶层: z={top_z}, 像素行 {top_rows}",
             ""]
    n = 0
    for r in top_rows:
        z = z_of(r)
        lines.append(f"# --- 像素行 {r} (z={z}) ---")
        for c in range(COLS - 1, -1, -1):
            lines.append(f"execute at @s positioned {x_of(c)} {Y_BASE} {z} run "
                         f"function vessel:barrel_0")
            n += 1
        lines.append("")

    path = os.path.join(FUNC_DIR, 'place_top_layer.mcfunction')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[已生成] {path}   指令数 {n}")

    # 明细
    print(f"\n[最顶层坐标明细]")
    print(f"  {'像素行':>6} | {'像素列':>6} | {'x':>5} | {'y':>4} | {'z':>5}")
    print("  " + "-" * 38)
    for r in top_rows:
        z = z_of(r)
        for c in range(COLS):
            print(f"  {r:>6} | {c:>6} | {x_of(c):>5} | {Y_BASE:>4} | {z:>5}")
        print("  " + "-" * 38)

    # 全层坐标映射存档
    coords = {}
    for r in range(ROWS):
        for c in range(COLS):
            coords[f"{r},{c}"] = [x_of(c), Y_BASE, z_of(r)]
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    json.dump({'rule': {'even_z': 'z = -2 - 2*(14-n)', 'odd_z': 'z = -4*floor((13-n)/2)',
                        'x': 'x = -2*(14-c)', 'y': Y_BASE},
               'layers': {str(z): [r for r in range(ROWS) if z_of(r) == z] for z in zs},
               'coords': coords},
              open(os.path.join(ROOT, 'results', 'vessel_coords.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n坐标映射已存: results/vessel_coords.json")


if __name__ == '__main__':
    main()

#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
第十层（数字 9）整层 225 个容器放置函数
==========================================
坐标理解（修正后）:
  * y 决定层 -> 第 10 层整层在同一个 y 上
  * x 与 z 是平面内的两个方向（行与列）

推导出的平面坐标:
  基准: 像素行 14、列 14（最右列）-> x=-1, y=-1, z=-2
  奇数行最右列 x=-1 ; 偶数行最右列 x=0   （x 在 -1/0 之间交错）
  同一行内每往左一个像素 x-2  -> x ∈ [-1,-29] 或 [0,-28]
  z 按行: 13->0, 14->-2, 11->-4, 12->-6, ..., 1->-24, 2->-26, 0->-30
      相邻差 2，唯 2->0 差 4

  => 15 个 x 位置 × 15 个 z 位置 = 225

容器值（数字 9 = 第 10 层）:
  第 10 层为偶数层 -> c = 15 - W
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DP = os.path.join(os.path.dirname(ROOT), 'datapack', 'tiny_cnn')
FUNC_DIR = os.path.join(DP, 'data', 'deploy_weights', 'function')

COLS, ROWS = 15, 15
Y_LAYER = -1
ORDER = [13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0]
ZSTEP = [2]*13 + [4]


def build_Z():
    Z = {}
    z = 0
    for i, r in enumerate(ORDER):
        Z[r] = z
        if i < len(ZSTEP):
            z -= ZSTEP[i]
    return Z


def main():
    W = np.load(os.path.join(ROOT, 'results', 'Wint_model.npz'))['W'].astype(int)
    Z = build_Z()
    k = 9                                  # 数字 9
    cvals = 15 - W[:, k]                   # 第 10 层 = 偶数层
    os.makedirs(FUNC_DIR, exist_ok=True)

    print("=" * 92)
    print("第十层（数字 9）整层 225 个容器（y = -1）")
    print("=" * 92)

    print(f"\n[平面坐标轴]")
    print(f"  x: 奇数行最右列 = -1 ; 偶数行最右列 = 0 ; 每往左一个像素 x -= 2")
    print(f"     奇数行 x ∈ {{-1,-3,...,-29}}   偶数行 x ∈ {{0,-2,...,-28}}")
    print(f"  z: 按行给定（13->0, 14->-2, 11->-4, ..., 2->-26, 0->-30）")
    print(f"  y: 恒为 {Y_LAYER}（整层同高度）")

    print(f"\n[行 -> z]")
    print(f"  {'像素行':>6} | {'奇偶':>4} | {'z':>5} | {'该行 x 序列起点(最右)':>22}")
    print("  " + "-" * 48)
    for r in ORDER:
        xr = -1 if r % 2 else 0
        print(f"  {r:>6} | {'奇' if r%2 else '偶':>4} | {Z[r]:>5} | {xr:>22}")

    # 生成
    lines = ["# 第十层（数字 9）整层放置  -- 由 gen_deploy.py 生成",
             "# 执行位置: ~ ~ ~",
             f"# 整层 y = {Y_LAYER}，共 15x15 = 225 个容器",
             "# 坐标: x 沿列（每格 -2），z 沿行（见下），y 恒定",
             ""]
    rows_meta = []
    n = 0
    for r in ORDER:
        z = Z[r]
        xr = -1 if r % 2 else 0
        lines.append(f"# --- 像素行 {r} (z={z}, 最右列 x={xr}) ---")
        for col in range(COLS - 1, -1, -1):
            x = xr - 2 * (COLS - 1 - col)
            p = r * 15 + col
            sig = int(cvals[p])
            lines.append(f"execute at @s positioned {x} {Y_LAYER} {z} run "
                         f"function vessel:barrel_{sig}")
            rows_meta.append({'row': r, 'col': col, 'x': x, 'y': Y_LAYER,
                              'z': z, 'signal': sig})
            n += 1
        lines.append("")

    path = os.path.join(FUNC_DIR, 'place_layer10_digit9.mcfunction')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\n[已生成] {path}")
    print(f"  指令数: {n}")

    # 统计
    sigs = [d['signal'] for d in rows_meta]
    print(f"  容器信号范围 [{min(sigs)}, {max(sigs)}]   不同取值 "
          f"{sorted(set(sigs))}")
    xs = sorted(set(d['x'] for d in rows_meta))
    zs = sorted(set(d['z'] for d in rows_meta), reverse=True)
    print(f"  x 取值 {len(xs)} 个: {xs[0]} .. {xs[-1]}")
    print(f"  z 取值 {len(zs)} 个: {zs[0]} .. {zs[-1]}")

    # 前 20 条预览
    print(f"\n[文件前 20 条]")
    with open(path, encoding='utf-8') as f:
        for i, ln in enumerate(f):
            if i >= 22:
                break
            print('  ' + ln.rstrip())

    json.dump({'layer': 10, 'digit': 9, 'y': Y_LAYER, 'order': ORDER,
               'Z': {str(k2): v for k2, v in Z.items()},
               'cells': rows_meta},
              open(os.path.join(ROOT, 'results', 'layer10_deploy.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n明细已存: results/layer10_deploy.json")


if __name__ == '__main__':
    main()

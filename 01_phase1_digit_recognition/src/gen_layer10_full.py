#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
第十层（数字 9）完整放置文件：225 权重容器 + 8 个 G_{i,0} + F_0
==================================================================
指令格式（相对坐标，0 分量写作 ~ 且不补空格）:
  execute at @s positioned ~<x> ~-1 ~<z> run function vessel:barrel_<信号>

坐标:
  权重容器: x 沿列（奇数行 0..-28，偶数行 -1..-29），z 沿行
            z: 行13=0, 14=-2, 11=-4, 12=-6, 9=-8, 10=-10, 7=-12, 8=-14,
               5=-16, 6=-18, 3=-20, 4=-22, 1=-24, 2=-26, 0=-30
  G_{1,0}: ~-30 ~-1 ~1    此后 i+1 -> z 减 4
  F_0    : ~1 ~-1 ~-28
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DP = os.path.join(os.path.dirname(ROOT), 'datapack', 'tiny_cnn')
FUNC_DIR = os.path.join(DP, 'data', 'deploy_weights', 'function')
RES = os.path.join(ROOT, 'results')
ORDER = [13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0]
Y_LAYER = -1
K = 9


def fmt(x, y, z):
    def p(v):
        return '~' if v == 0 else f'~{v}'
    return f"{p(x)} {p(y)} {p(z)}"


def build_Z():
    Z, z = {}, 0
    for i, r in enumerate(ORDER):
        Z[r] = z
        if i < len(ORDER) - 1:
            z -= 4 if r == 2 else 2
    return Z


def x_start(row):
    return 0 if row % 2 == 1 else -1


def main():
    W = np.load(os.path.join(RES, 'Wint_model.npz'))['W'].astype(int)
    Z = build_Z()
    cvals = 15 - W[:, K]                     # 第 10 层 = 偶数层
    GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]
    G0, Gmax = [], []
    for (r0, nr) in GROUPS:
        pxs = np.array([r * 15 + c for r in range(r0, r0 + nr) for c in range(15)])
        cg = cvals[pxs]
        G0.append(int(np.maximum(cg, 15).sum()))
        Gmax.append(G0[-1] - int(np.minimum(cg, 15).sum()))
    F0 = sum(Gmax)

    lines = [f"# 第十层（数字 {K}）完整放置：225 权重容器 + 8 个 G_{{i,0}} + F_0",
             "# 执行位置: ~ ~ ~",
             f"# y 恒为 -1（层由 y 决定）",
             f"# 容器信号范围 [{int(cvals.min())}, {int(cvals.max())}]",
             ""]
    n = 0
    for r in ORDER:
        z = Z[r]; xs = x_start(r)
        lines.append(f"# --- 权重：像素行 {r} (z={z}) ---")
        for col in range(14, -1, -1):
            x = xs - 2 * (14 - col)
            sig = int(cvals[r * 15 + col])
            lines.append(f"execute at @s positioned {fmt(x,Y_LAYER,z)} run "
                         f"function vessel:barrel_{sig}")
            n += 1
        lines.append("")
    lines.append("# --- 支线偏置 G_{i,0} ---")
    for i in range(8):
        z = 1 - 4 * i
        lines.append(f"execute at @s positioned {fmt(-30,Y_LAYER,z)} run "
                     f"function vessel:barrel_{G0[i]}")
        n += 1
    lines.append("")
    lines.append("# --- 总线偏置 F_0 ---")
    lines.append(f"execute at @s positioned {fmt(1,Y_LAYER,-28)} run "
                 f"function vessel:barrel_{F0}")
    n += 1
    lines.append("")

    path = os.path.join(FUNC_DIR, 'layer10_digit9_full.mcfunction')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print("=" * 88)
    print(f"已生成: {path}")
    print(f"  指令总数 {n}  = 225 权重 + 8 个 G_{{i,0}} + 1 个 F_0")
    print(f"\n  G_{{i,0}} 信号: {G0}   F_0 = {F0}")
    print(f"  权重信号范围 [{int(cvals.min())}, {int(cvals.max())}]")

    print(f"\n[文件头 12 行]")
    with open(path, encoding='utf-8') as f:
        for i, ln in enumerate(f):
            if i >= 12:
                break
            print('  ' + ln.rstrip())
    print(f"\n[偏置部分]")
    with open(path, encoding='utf-8') as f:
        alll = f.read().splitlines()
    for ln in alll:
        if 'G_{i,0}' in ln or 'F_0' in ln or (ln.startswith('execute') and
                                              ('~-30' in ln or '~1 ~-1 ~-28' in ln)):
            print('  ' + ln)

    # 坐标清单
    allc = []
    for r in ORDER:
        z = Z[r]; xs = x_start(r)
        for col in range(14, -1, -1):
            allc.append((xs - 2 * (14 - col), Y_LAYER, z))
    for i in range(8):
        allc.append((-30, Y_LAYER, 1 - 4 * i))
    allc.append((1, Y_LAYER, -28))
    print(f"\n[重复坐标检查] 共 {len(allc)} 个位置，"
          f"去重后 {len(set(allc))} 个 -> "
          f"{'无重复 OK' if len(allc) == len(set(allc)) else '有重复！'}")

    json.dump({'layer': 10, 'digit': K, 'y': Y_LAYER, 'n_commands': n,
               'G_i0': G0, 'G_max': Gmax, 'F_0': F0,
               'signal_range': [int(cvals.min()), int(cvals.max())]},
              open(os.path.join(RES, 'layer10_full.json'), 'w'),
              ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

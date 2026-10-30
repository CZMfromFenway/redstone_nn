#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
第十层（数字 9）：G_{i,0} 与 F_0 的坐标与信号值
==================================================
坐标（czm 给定）:
  G_{1,0}: ~-30 ~-1 ~1
  此后 i+1 -> z 减 4
  F_0    : ~1 ~-1 ~-28

指令格式（相对坐标）:
  execute at @s positioned ~<x> ~-1 ~<z> run function vessel:barrel_<信号>
  某个分量为 0 时省略（如 ~-30 ~-1 ~1）

信号值: G_{i,0} = sum_j max(c_j, 15)   （偶数层 c = 15 - W）
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]


def fmt(x, y, z):
    """相对坐标：0 分量写 ~，非 0 写 ~<值>"""
    def part(v):
        return '~' if v == 0 else f'~{v}'
    return f"{part(x)} {part(y)} {part(z)}"


def main():
    W = np.load(os.path.join(RES, 'Wint_model.npz'))['W'].astype(int)
    k = 9
    parity = '偶数'                      # 第 10 层
    c_all = (15 - W[:, k]) if parity == '偶数' else (15 + W[:, k])

    print("=" * 92)
    print(f"数字 {k}（第 {k+1} 层，{parity}层）: G_{{i,0}} 与 F_0")
    print("=" * 92)

    G0, Gmax = [], []
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = np.array([r * 15 + c for r in range(r0, r0 + nr) for c in range(15)])
        cg = c_all[pxs]
        g0 = int(np.maximum(cg, 15).sum())
        gmax = g0 - int(np.minimum(cg, 15).sum())
        G0.append(g0); Gmax.append(gmax)
    F0 = sum(Gmax)

    print(f"\n[G_{{i,0}} 坐标与信号]")
    print(f"  {'名':>6} | {'坐标':>16} | {'信号值':>6} | {'G_i^max':>8} | "
          f"{'<=511?':>7}")
    print("  " + "-" * 58)
    for i in range(8):
        x, y, z = -30, -1, 1 - 4 * i
        print(f"  G_{i+1},0 | {fmt(x,y,z):>16} | {G0[i]:>6} | {Gmax[i]:>8} | "
              f"{'OK' if G0[i] <= 511 else '超 511':>7}")
    print(f"\n  F_0 坐标 = {fmt(1,-1,-28)}   信号值 = {F0}   "
          f"{'OK' if F0 <= 511 else '超 511'}")

    print(f"\n[对照] 225 个权重容器的信号范围 "
          f"[{int(c_all.min())}, {int(c_all.max())}]（都在 0~511 内）")

    # 生成追加片段
    lines = ["", "# --- G_{i,0} 支线偏置 ---"]
    for i in range(8):
        x, y, z = -30, -1, 1 - 4 * i
        lines.append(f"execute at @s positioned {fmt(x,y,z)} run "
                     f"function vessel:barrel_{G0[i]}")
    lines.append("")
    lines.append("# --- F_0 总线偏置 ---")
    lines.append(f"execute at @s positioned {fmt(1,-1,-28)} run "
                 f"function vessel:barrel_{F0}")
    lines.append("")
    out = '\n'.join(lines)
    print(f"\n[追加片段]")
    print(out)

    p = os.path.join(RES, 'layer10_bias_snippet.mcfunction')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(out.lstrip('\n'))
    print(f"已存: {p}")

    # z 冲突检查
    print(f"\n[坐标冲突检查]")
    xw = set()
    for r in range(15):
        pass
    zc = set()
    for r in range(15):
        zc.add(0 - 2 * r)          # 近似：行 r 的 z
    zs_w = set()
    ORDER = [13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0]
    zz, z = {}, 0
    for i, r in enumerate(ORDER):
        zz[r] = z
        if i < 14:
            z -= 2 if not (r == 2) else 4
    zs_w = set(zz.values())
    zs_g = set(1 - 4 * i for i in range(8))
    print(f"  权重容器的 z 集合: {sorted(zs_w, reverse=True)}")
    print(f"  G_{{i,0}} 的 z 集合: {sorted(zs_g, reverse=True)}")
    print(f"  F_0 的 z: -28")
    inter = zs_w & zs_g
    print(f"  交集: {sorted(inter, reverse=True) if inter else '无'}"
          f"  {'<- 有重叠，可能碰撞' if inter else ''}")
    print(f"  F_0 的 z=-28 是否在权重集合内: {'是' if -28 in zs_w else '否'}")

    json.dump({'digit': k, 'G_i0': G0, 'G_max': Gmax, 'F_0': F0,
               'G_coords': [[-30, -1, 1 - 4 * i] for i in range(8)],
               'F0_coord': [1, -1, -28]},
              open(os.path.join(RES, 'layer10_bias.json'), 'w'),
              ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

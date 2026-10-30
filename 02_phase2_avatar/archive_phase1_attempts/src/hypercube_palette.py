#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 16 色搜寻"超立方体排列"（格雷码兼容）
==========================================
目标（czm 的提议）:
  把 16 色分配到 4 位二值码的 16 个顶点上，使得:
    * 汉明距离 1 的两色（翻 1 位）  -> 颜色**相近**
    * 汉明距离 4 的两色（4 位全翻）-> 颜色**差距最大**

为什么有用: 红石里 4 个二值变量 = 4 条线（而不是 16 路 one-hot）；
  且"位翻转数"天然对应"颜色变化量"，高位/低位的分工可以映射到
  亮度/细节。

优化目标（两个，都要小）:
  J1 = 所有 32 条立方体边的平均色距        （相邻要近）
  J2 = 所有 8 对"对顶点"的平均色距的**负值**（对顶点要远）

搜索方法: 模拟退火 + 2-opt 交换
"""
import os, json, numpy as np, itertools, colorsys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")

# 4 位超立方体的 16 个顶点
VERTS = np.array([[int(b) for b in format(i, '04b')] for i in range(16)], dtype=int)
# 边: 汉明距离 1 的对（32 条）
EDGES = [(i, j) for i in range(16) for j in range(i + 1, 16)
         if (VERTS[i] != VERTS[j]).sum() == 1]
# 对顶点: 汉明距离 4（8 对）
OPP = [(i, j) for i in range(16) for j in range(i + 1, 16)
       if (VERTS[i] != VERTS[j]).sum() == 4]


def to_lab(rgb):
    """sRGB -> 近似 CIELAB，用于感知距离"""
    r, g, b = [x / 255.0 for x in rgb]
    def f(u):
        return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4
    r, g, b = f(r), f(g), f(b)
    X = r * 0.4124 + g * 0.3576 + b * 0.1805
    Y = r * 0.2126 + g * 0.7152 + b * 0.0722
    Z = r * 0.0193 + g * 0.1192 + b * 0.9505
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    def h(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = h(X / Xn), h(Y / Yn), h(Z / Zn)
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


def main():
    print("=" * 96)
    print("为 16 色搜寻超立方体排列（4 位格雷码兼容）")
    print("=" * 96)
    j = json.load(open(os.path.join(DATA, 'palette_avatar16.json')))
    P = np.array(j['palette_ordered'], dtype=float)
    names = j['names']
    K = len(P)
    print(f"调色板: {K} 色")
    LAB = np.array([to_lab(c) for c in P])
    D = np.linalg.norm(LAB[:, None] - LAB[None], axis=2)      # 感知色距矩阵
    print(f"色距范围: 最近 {D[D>0].min():.1f}  最远 {D.max():.1f}")

    def cost(perm):
        """perm[v] = 顶点 v 上的颜色索引"""
        P_ = D[np.ix_(perm, perm)]
        j1 = np.mean([P_[a, b] for a, b in EDGES])              # 边要短
        opp = np.mean([P_[a, b] for a, b in OPP])               # 对顶点要长
        return j1, -opp

    best = None
    rng = np.random.default_rng(0)
    for restart in range(60):
        perm = rng.permutation(K)
        j1, j2 = cost(perm)
        T = 50.0
        for it in range(4000):
            a, b = rng.choice(K, 2, replace=False)
            perm2 = perm.copy(); perm2[a], perm2[b] = perm2[b], perm2[a]
            n1, n2 = cost(perm2)
            d = (n1 + n2) - (j1 + j2)
            if d < 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
                perm, j1, j2 = perm2, n1, n2
            T *= 0.999
        if best is None or (j1 + j2) < (best[0] + best[1]):
            best = (j1, j2, perm.copy())
    j1, j2, perm = best
    print(f"\n最优: 边平均色距 = {j1:.2f}   对顶点平均色距 = {-j2:.2f}")
    print(f"      (随机排列的期望: 边 {np.mean(D):.2f}, 对顶点 {np.mean(D):.2f})")

    # 对照: 理想超立方体的比例
    print(f"      比值 对顶点/边 = {-j2/j1:.2f}   (理想越大越好)")

    # 位语义分析: 每一位翻转平均改变多少色距
    print(f"\n[位语义] 每一位翻转引起的平均色距")
    bit_effect = []
    for b in range(4):
        ds = []
        for i in range(16):
            v2 = VERTS[i].copy(); v2[b] = 1 - v2[b]
            k = int(''.join(map(str, v2)), 2)
            ds.append(D[perm[i], perm[k]])
        bit_effect.append(np.mean(ds))
    for b, e in enumerate(bit_effect):
        print(f"  位 {b} (值 {2**(3-b)}): 平均色距 {e:6.2f}")

    print(f"\n[最终分配] 顶点(二进制) -> 颜色")
    print(f"  {'顶点':>6} | {'HEX':>8} | {'RGB':<18} | MC 纹理 | 亮度")
    print("  " + "-" * 74)
    out = []
    for v in range(16):
        c = P[perm[v]]
        hx = '#%02X%02X%02X' % tuple(int(x) for x in c)
        lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        bits = ''.join(map(str, VERTS[v]))
        print(f"  {bits:>6} | {hx:>8} | ({c[0]:>3.0f},{c[1]:>3.0f},{c[2]:>3.0f})      | "
              f"{names[perm[v]]:<22} | {lum:>3.0f}")
        out.append({'bits': bits, 'rgb': [int(x) for x in c], 'name': names[perm[v]]})

    json.dump({'assignment': out, 'edge_dist': j1, 'opp_dist': -j2,
               'bit_effect': bit_effect,
               'random_edge': float(np.mean(D))},
              open(os.path.join(DATA, 'palette_hypercube.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: data/palette_hypercube.json")


if __name__ == '__main__':
    main()

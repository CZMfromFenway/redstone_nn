#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超立方体排列搜索（可指定调色板，对比混凝土 vs 定制）
========================================================
用法:
  python hypercube_compare.py            # 对比两套调色板
  python hypercube_compare.py concrete   # 只跑混凝土
"""
import os, json, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

VERTS = np.array([[int(b) for b in format(i, '04b')] for i in range(16)], dtype=int)
EDGES = [(i, j) for i in range(16) for j in range(i + 1, 16)
         if (VERTS[i] != VERTS[j]).sum() == 1]
OPP = [(i, j) for i in range(16) for j in range(i + 1, 16)
       if (VERTS[i] != VERTS[j]).sum() == 4]


def to_lab(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    def f(u): return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4
    r, g, b = f(r), f(g), f(b)
    X = r * 0.4124 + g * 0.3576 + b * 0.1805
    Y = r * 0.2126 + g * 0.7152 + b * 0.0722
    Z = r * 0.0193 + g * 0.1192 + b * 0.9505
    def h(t): return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = h(X / 0.95047), h(Y), h(Z / 1.08883)
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


def search(P, names, restarts=80, iters=5000, seed=0):
    LAB = np.array([to_lab(c) for c in P])
    D = np.linalg.norm(LAB[:, None] - LAB[None], axis=2)
    K = len(P)

    def cost(perm):
        P_ = D[np.ix_(perm, perm)]
        j1 = np.mean([P_[a, b] for a, b in EDGES])
        opp = np.mean([P_[a, b] for a, b in OPP])
        return j1, -opp

    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        perm = rng.permutation(K)
        j1, j2 = cost(perm)
        T = 60.0
        for it in range(iters):
            a, b = rng.choice(K, 2, replace=False)
            p2 = perm.copy(); p2[a], p2[b] = p2[b], p2[a]
            n1, n2 = cost(p2)
            d = (n1 + n2) - (j1 + j2)
            if d < 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
                perm, j1, j2 = p2, n1, n2
            T *= 0.999
        if best is None or (j1 + j2) < (best[0] + best[1]):
            best = (j1, j2, perm.copy())
    j1, j2, perm = best
    # 理论下界（贪心估计）：16 点 4 维立方体的最短边平均
    return j1, -j2, perm, D


def report(tag, P, names):
    j1, opp, perm, D = search(P, names)
    rnd = float(np.mean(D))
    print(f"\n{'='*92}")
    print(f"### {tag}   边平均 {j1:.2f}   对顶点平均 {opp:.2f}   比值 {opp/j1:.2f}")
    print(f"###   随机排列基准: 都是 {rnd:.2f}  ->  边改善 {rnd/j1:.2f}x, 对顶点改善 {opp/rnd:.2f}x")
    print(f"{'='*92}")
    bit = []
    for b in range(4):
        ds = []
        for i in range(16):
            v2 = VERTS[i].copy(); v2[b] = 1 - v2[b]
            k = int(''.join(map(str, v2)), 2)
            ds.append(D[perm[i], perm[k]])
        bit.append(float(np.mean(ds)))
    w = [8, 4, 2, 1]
    print(f"  位权重:      {w}")
    print(f"  平均色距:    {[round(x,1) for x in bit]}")
    print(f"  权重序与色距序一致? {'是' if all(bit[i]>=bit[i+1] for i in range(3)) else '否'}")
    print(f"\n  {'顶点':>6} | {'HEX':>8} | {'RGB':<18} | MC 纹理")
    print("  " + "-" * 66)
    out = []
    for v in range(16):
        c = P[perm[v]]
        bits = ''.join(map(str, VERTS[v]))
        hx = '#%02X%02X%02X' % tuple(int(x) for x in c)
        print(f"  {bits:>6} | {hx:>8} | ({c[0]:>3.0f},{c[1]:>3.0f},{c[2]:>3.0f})      | {names[perm[v]]}")
        out.append({'bits': bits, 'hex': hx, 'rgb': [int(x) for x in c], 'name': names[perm[v]]})
    return {'tag': tag, 'edge': j1, 'opp': opp, 'ratio': opp / j1,
            'random': rnd, 'bit_effect': bit, 'assignment': out}


def main():
    pal = json.load(open(os.path.join(DATA, 'palettes.json')))
    j16 = json.load(open(os.path.join(DATA, 'palette_avatar16.json')))
    order16 = pal['order16']
    concrete = np.array(pal['pal16'], dtype=float)
    custom = np.array(j16['palette_ordered'], dtype=float)
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    res = []
    if which in ('both', 'concrete'):
        res.append(report('MC 混凝土 16 色（设计色环）', concrete, order16))
    if which in ('both', 'custom'):
        res.append(report('定制 16 色（数据驱动）', custom, j16['names']))

    if len(res) == 2:
        print(f"\n{'='*92}")
        print("### 结论")
        print(f"{'='*92}")
        for r in res:
            print(f"  {r['tag']:<28} 边 {r['edge']:6.2f}  对顶点 {r['opp']:6.2f}  "
                  f"比值 {r['ratio']:.2f}  位语义单调 {all(r['bit_effect'][i]>=r['bit_effect'][i+1] for i in range(3))}")
        better = max(res, key=lambda r: r['ratio'])
        print(f"\n  -> 超立方体结构更好的是: **{better['tag']}** (比值 {better['ratio']:.2f})")
    json.dump(res, open(os.path.join(DATA, 'palette_hypercube_cmp.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: data/palette_hypercube_cmp.json")


if __name__ == '__main__':
    main()

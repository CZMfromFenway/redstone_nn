#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
数字 9（第 10 层，偶数层）: 最终正确解
========================================
偶数层规则:
  像素亮起 -> 截断 -> 输出 15
  像素不亮 -> 输出容器值 c_j = 15 - W_j
  G_i = G_i0 - sum_j(输出_j)
  G_i0 = sum_j max(c_j, 15)    <- 保证 G_i >= 0（这是所有子集下 sum 输出的上界）
  G_i 范围 = [0, G_i0 - sum_j min(c_j,15)]
  F = F_0 - sum_i G_i
  F_0 = sum_i G_i^max          <- 保证 F >= 0

验证: F=0 时应为"负权重全亮、正权重全不亮"
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]


def load_idx(p):
    import struct
    with open(p, 'rb') as f:
        mg = struct.unpack('>I', f.read(4))[0]; nd = mg & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(nd)]
        d = np.frombuffer(f.read(), dtype=np.uint8)
    return d.reshape(dims)


def to15(im):
    n = len(im)
    a = im.reshape(n, 28, 28).astype(np.float32).reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    o = np.zeros((n, 15, 15), dtype=np.float32)
    o[:, :14, :14] = a; o[:, 14, :14] = a[:, 13, :]
    o[:, :14, 14] = a[:, :, 13]; o[:, 14, 14] = a[:, 13, 13]
    return o


def main():
    z = np.load(os.path.join(RES, 'Wint_model.npz'))
    W = z['W'].astype(int); bias = z['b'].astype(int)
    k = 9; layer = k + 1
    print("=" * 104)
    print(f"数字 {k}（第 {layer} 层，偶数层）: 最终解")
    print("=" * 104)

    rows = []
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = [r * 15 + c for r in range(r0, r0 + nr) for c in range(15)]
        wg = W[pxs, k]; cg = 15 - wg
        g0 = int(np.maximum(cg, 15).sum())
        gmax = g0 - int(np.minimum(cg, 15).sum())
        rows.append({'G': gi + 1, 'row0': r0, 'rows': nr, 'n': len(pxs),
                     'G_i0': g0, 'G_min': 0, 'G_max': gmax,
                     'w_sum': int(wg.sum()), 'w_pos': int(wg[wg > 0].sum()),
                     'w_neg': int(wg[wg < 0].sum()), 'cg': cg, 'wg': wg})

    print(f"  {'支线':>4} | {'行':>7} | {'n':>3} | {'W和':>5} | {'W正':>4} | {'W负':>5} | "
          f"{'G_i0':>5} | {'G_max':>6} | {'G 范围':>9}")
    print("  " + "-" * 82)
    for r in rows:
        print(f"  G_{r['G']:<2} | {r['row0']:>3}-{r['row0']+r['rows']-1:<3} | {r['n']:>3} | "
              f"{r['w_sum']:>5} | {r['w_pos']:>4} | {r['w_neg']:>5} | {r['G_i0']:>5} | "
              f"{r['G_max']:>6} | [0, {r['G_max']:>2}]")

    F0 = sum(r['G_max'] for r in rows)
    print(f"\n  F_0 = sum(G_i^max) = {' + '.join(str(r['G_max']) for r in rows)} = {F0}")
    print(f"  F 范围 = [0, {F0}]")

    # 验证 F=0 配置
    print(f"\n[验证1] F=0 时的配置")
    lit_neg = np.zeros(225, dtype=int)
    for r in rows:
        pxs = [rr * 15 + c for rr in range(r['row0'], r['row0'] + r['rows'])
               for c in range(15)]
        for i, p in enumerate(pxs):
            if r['wg'][i] < 0:
                lit_neg[p] = 1
    w9 = W[:, k]; neg = w9 < 0; pos = w9 > 0
    S = 0
    for r in rows:
        pxs = [rr * 15 + c for rr in range(r['row0'], r['row0'] + r['rows'])
               for c in range(15)]
        lg = lit_neg[pxs]
        S += int(np.where(lg == 1, 15, r['cg']).sum())
    print(f"  负权重点亮 {int(lit_neg[neg].sum())}/{int(neg.sum())}  "
          f"正权重点亮 {int(lit_neg[pos].sum())}/{int(pos.sum())}")
    print(f"  sum(G_i) = {S}   F = {F0} - {S} = {F0-S}  "
          f"{'✓ =0' if F0-S == 0 else '✗'}")

    # 逐样本核对
    print(f"\n[验证2] 逐样本核对（前 2000 测试样本）")
    RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    thr = to15(load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))).mean()
    Fte = (to15(Xte) > thr).astype(np.float64).reshape(len(Xte), -1)
    Zte = Fte @ W + bias
    Xb = Fte[:2000]
    Gs = np.zeros((len(Xb), 8), dtype=np.int64)
    for gi, r in enumerate(rows):
        pxs = [rr * 15 + c for rr in range(r['row0'], r['row0'] + r['rows'])
               for c in range(15)]
        lit = Xb[:, pxs]
        Gs[:, gi] = r['G_i0'] - np.where(lit == 1, 15, r['cg'][None, :]).sum(1)
    Fb = F0 - Gs.sum(1)
    zb = Zte[:2000, k]
    print(f"  G 非负: 最小 {Gs.min()}  {'✓' if Gs.min() >= 0 else '✗'}")
    print(f"  G_i 范围（实测）: " + " ".join(
        f"G{i+1}[{Gs[:,i].min()},{Gs[:,i].max()}]" for i in range(8)))
    print(f"  arch F 范围 [{Fb.min()}, {Fb.max()}]")
    print(f"  net  z 范围 [{zb.min():.0f}, {zb.max():.0f}]")
    corr = float(np.corrcoef(Fb, zb)[0, 1])
    print(f"  corr(F, z) = {corr:+.6f}")
    d = Fb + zb
    d2 = Fb - zb
    if abs(corr + 1) < 1e-9:
        print(f"  F 与 z 反向 -> 核对 F + z: 范围 [{d.min()}, {d.max()}] "
              f"标准差 {d.std():.4f}")
        if d.max() - d.min() == 0:
            print(f"  => **F = -z + {d[0]}**  （架构与网络等价，输出端需取反）")
    else:
        print(f"  F - z 范围 [{d2.min()}, {d2.max()}] 标准差 {d2.std():.4f}")
        if d2.max() - d2.min() == 0:
            print(f"  => F = z + {d2[0]}")

    print(f"\n{'='*104}")
    print("### 最终结果")
    print(f"  {'支线':>4} | {'G_i0':>5} | {'G 范围':>10}")
    for r in rows:
        print(f"  G_{r['G']:<2} | {r['G_i0']:>5} | [0, {r['G_max']:>2}]")
    print(f"\n  F_0 = {F0}")
    print(f"  F 范围 = [0, {F0}]")

    json.dump({'digit': k, 'layer': layer, 'parity': '偶数', 'F_0': F0,
               'branches': [{'G': r['G'], 'G_i0': r['G_i0'], 'G_max': r['G_max'],
                             'w_sum': r['w_sum'], 'w_pos': r['w_pos'],
                             'w_neg': r['w_neg']} for r in rows]},
              open(os.path.join(RES, 'mc_digit9_final.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\n已写出: results/mc_digit9_final.json")


if __name__ == '__main__':
    main()

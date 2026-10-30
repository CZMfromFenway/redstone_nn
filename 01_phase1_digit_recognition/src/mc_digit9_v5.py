#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
数字 9（第 10 层，偶数层）: 正确取 G_i0 = sum(c_j)
====================================================
前几版的错误根源: 我取 G_i0 = sum_j max(c_j,15)（"使 G 最小值为 0"）
  但偶数层 c_j = 15-W_j，正权重时 c_j<15，于是
     G_i^max = G_i0 - sum_j min(c_j,15) = -W_和  < 0
  物理上 G 不可能为负 -> 定义错了。

正确取法: G_i0 = sum_j c_j（把所有容器值加起来）
  全部点亮   -> sum=c_j  -> G_i = 0            （最小）
  全部不亮   -> sum=15n  -> G_i = sum(c)-15n = -W_和
  仅负权重点亮 -> G_i = W_负和                   （最大，因为 W_和 = W_正和 + W_负和）

  于是 G_i^max = W_负和,  F_0 = sum_i W_负和,i
  F=0 时恰好"负权重全亮、正权重全不亮"  <- czm 的要求
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
    k = 9; layer = k + 1; parity = '偶数'
    print("=" * 100)
    print(f"数字 {k}（第 {layer} 层，{parity}层）: G_i0 = sum(c_j)")
    print("=" * 100)
    print(f"偶数层: c_j = 15 - W_j ; 不亮输出 15, 点亮输出 c_j")
    print(f"不变量: 正权重点亮 -> F 增大 ; 负权重点亮 -> F 减小\n")

    rows = []
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = [r * 15 + c for r in range(r0, r0 + nr) for c in range(15)]
        wg = W[pxs, k]
        cg = 15 - wg
        n = len(pxs)
        g0 = int(cg.sum())
        w_pos = int(wg[wg > 0].sum()); w_neg = int(wg[wg < 0].sum())
        gmax = w_neg                                  # 仅负权重点亮
        gmin = 0                                      # 全部点亮
        rows.append({'G': gi + 1, 'row0': r0, 'rows': nr, 'n': n,
                     'c_sum': g0, 'G_i0': g0, 'G_min': gmin, 'G_max': gmax,
                     'w_pos': w_pos, 'w_neg': w_neg, 'w_sum': int(wg.sum()),
                     'c_min': int(cg.min()), 'c_max': int(cg.max())})

    print(f"  {'支线':>4} | {'行':>7} | {'n':>3} | {'W正和':>5} | {'W负和':>5} | "
          f"{'sum(c)':>7} | {'G_i0':>5} | {'G_min':>5} | {'G_max':>6} | {'G 范围':>9}")
    print("  " + "-" * 88)
    for r in rows:
        print(f"  G_{r['G']:<2} | {r['row0']:>3}-{r['row0']+r['rows']-1:<3} | {r['n']:>3} | "
              f"{r['w_pos']:>5} | {r['w_neg']:>5} | {r['c_sum']:>7} | {r['G_i0']:>5} | "
              f"{r['G_min']:>5} | {r['G_max']:>6} | [{r['G_min']},{r['G_max']:>3}]")

    F0 = sum(r['G_max'] for r in rows)
    print(f"\n  F_0 = sum(G_i^max) = sum(W负和) = "
          f"{' + '.join(str(r['G_max']) for r in rows)} = {F0}")
    print(f"  F 范围 = [0, {F0}]")

    # ---------- 验证 F=0 配置 ----------
    print(f"\n[验证1] F=0 时的输入配置")
    lit_neg = np.zeros(225, dtype=int)
    for r in rows:
        pxs = [rr * 15 + c for rr in range(r['row0'], r['row0'] + r['rows'])
               for c in range(15)]
        wg = W[pxs, k]
        for i, p in enumerate(pxs):
            if wg[i] < 0:
                lit_neg[p] = 1
    w9 = W[:, k]; neg = w9 < 0; pos = w9 > 0
    tot = 0
    for r in rows:
        pxs = [rr * 15 + c for rr in range(r['row0'], r['row0'] + r['rows'])
               for c in range(15)]
        wg = W[pxs, k]; cg = 15 - wg
        tot += int(np.where(lit_neg[pxs] == 1, cg, 15).sum())
    print(f"  负权重点亮 {int(lit_neg[neg].sum())}/{int(neg.sum())}  "
          f"正权重点亮 {int(lit_neg[pos].sum())}/{int(pos.sum())}")
    print(f"  sum(G_i) = {tot}   F = {F0} - {tot} = {F0 - tot}  "
          f"{'✓ 等于 0' if F0 - tot == 0 else '✗'}")

    print(f"\n[验证2] 逐样本核对（前 2000 个测试样本）")
    RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    thr = to15(load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))).mean()
    Fte = (to15(Xte) > thr).astype(np.float64).reshape(len(Xte), -1)
    Zte = Fte @ W + bias

    def arch_F(Xb):
        B = len(Xb)
        Gs = np.zeros((B, 8), dtype=np.int64)
        for gi, r in enumerate(rows):
            pxs = [rr * 15 + c for rr in range(r['row0'], r['row0'] + r['rows'])
                   for c in range(15)]
            wg = W[pxs, k]; cg = 15 - wg
            lit = Xb[:, pxs]
            s = np.where(lit == 1, cg[None, :], 15).sum(1)
            Gs[:, gi] = r['G_i0'] - s
        return F0 - Gs.sum(1), Gs

    Fb, Gs = arch_F(Fte[:2000])
    zb = Zte[:2000, k]
    print(f"  arch F 范围 [{Fb.min()}, {Fb.max()}]   net z 范围 [{zb.min():.0f}, {zb.max():.0f}]")
    print(f"  G 非负检查: 最小 G = {Gs.min()}  {'✓' if Gs.min() >= 0 else '✗ 有负值'}")
    corr = float(np.corrcoef(Fb, zb)[0, 1])
    print(f"  corr(F, z) = {corr:+.6f}")
    d = Fb - zb
    print(f"  F - z 范围 [{d.min()}, {d.max()}]  "
          f"{'常数 ✓' if d.max()-d.min()==0 else '非常数'}")
    if d.max() - d.min() == 0:
        print(f"  => F = z + {d[0]}")
    else:
        # 检查是否差一个与样本有关的量
        print(f"  差值的标准差 {d.std():.3f}（若接近 0 说明近似常数）")

    json.dump({'digit': k, 'layer': layer, 'parity': parity, 'F_0': F0,
               'branches': [{'G': r['G'], 'G_i0': r['G_i0'], 'G_min': r['G_min'],
                             'G_max': r['G_max'], 'w_pos': r['w_pos'],
                             'w_neg': r['w_neg']} for r in rows],
               'F_range': [0, F0]},
              open(os.path.join(RES, 'mc_digit9_v5.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\n已写出: results/mc_digit9_v5.json")


if __name__ == '__main__':
    main()

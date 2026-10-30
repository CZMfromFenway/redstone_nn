#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
数字 9：最终精确表（修正 G_i^min 与 offset 来源）
====================================================
偶数层:
  输出_j = (点亮 ? 15 : c_j),  c_j = 15 - W_j
  G_i  = G_i0 - sum(输出_j)
  G_i0 = sum_j max(c_j, 15)     -> G_i >= 0 恒成立，且 G_i^min = 0
  G_i^max = G_i0 - sum_j min(c_j,15)
  F = F_0 - sum(G_i),  F_0 = sum_i G_i^max

恒等式（用于核对）:
  输出_j = 15 - W_j*(1-L_j)     其中 L_j=1 表示点亮
  => sum(输出) = 15n - sum(W_j) + sum(W_j L_j)
  => G_i = [G_i0 - 15n + sum(W_j)] - sum(W_j L_j)
  => G_i = A_i - sum_j W_j L_j,   A_i = G_i0 - 15n + W_和

  F = F_0 - sum_i G_i = F_0 - sum_i A_i + sum_all W_j L_j
    = 常数 + z（z = 网络 logit，不含 bias）

  => offset = F_0 - sum_i A_i + bias... 逐项核对
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
    k = 9
    print("=" * 100)
    print(f"数字 {k}（第 {k+1} 层，偶数层）精确表")
    print("=" * 100)

    brs = []
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = np.array([r * 15 + c for r in range(r0, r0 + nr) for c in range(15)])
        wg = W[pxs, k]; cg = 15 - wg
        g0 = int(np.maximum(cg, 15).sum())
        gmax = g0 - int(np.minimum(cg, 15).sum())
        A = g0 - 15 * len(pxs) + int(wg.sum())
        brs.append({'G': gi + 1, 'r0': r0, 'nr': nr, 'n': len(pxs), 'pxs': pxs,
                    'wg': wg, 'cg': cg, 'G_i0': g0, 'G_min': 0, 'G_max': gmax,
                    'A': A, 'w_sum': int(wg.sum()),
                    'w_pos': int(wg[wg > 0].sum()), 'w_neg': int(wg[wg < 0].sum()),
                    'sum_max': int(np.maximum(cg, 15).sum()),
                    'sum_min': int(np.minimum(cg, 15).sum())})

    print(f"\n[1] 支线表")
    print(f"  {'支线':>4} | {'行':>7} | {'n':>3} | {'W和':>5} | {'sum(c)':>7} | "
          f"{'sum_max':>8} | {'G_i0':>5} | {'G_max':>6} | {'G 范围':>10} | {'A_i':>5}")
    print("  " + "-" * 96)
    for b in brs:
        print(f"  G_{b['G']:<2} | {b['r0']:>3}-{b['r0']+b['nr']-1:<3} | {b['n']:>3} | "
              f"{b['w_sum']:>5} | {b['cg'].sum():>7} | {b['sum_max']:>8} | "
              f"{b['G_i0']:>5} | {b['G_max']:>6} | [0, {b['G_max']:>2}]  | {b['A']:>5}")

    F0 = sum(b['G_max'] for b in brs)
    sumA = sum(b['A'] for b in brs)
    print(f"\n[2] F_0 = sum(G_i^max) = {' + '.join(str(b['G_max']) for b in brs)} = {F0}")
    print(f"    sum(A_i) = {' + '.join(str(b['A']) for b in brs)} = {sumA}")
    print(f"\n[3] 恒等式: F = F_0 - sum_i A_i + sum_all(W_j L_j)")
    print(f"    F_0 - sum(A_i) = {F0 - sumA}")
    print(f"    而 z(不含bias) = sum_all(W_j L_j),  bias = {bias[k]}")
    print(f"    => F = z + {F0 - sumA}")
    print(f"    实测 offset 应为 {F0-sumA}")

    # 逐样本核对
    RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    thr = to15(load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))).mean()
    Xin = (to15(Xte) > thr).astype(np.int64).reshape(len(Xte), -1)
    Zte = Xin.astype(np.float64) @ W + bias
    lit_all = Xin.astype(bool)
    Gs = np.zeros((len(Xin), 8), dtype=np.int64)
    for gi, b in enumerate(brs):
        lit = lit_all[:, b['pxs']]
        Gs[:, gi] = b['G_i0'] - np.where(lit, 15, b['cg'][None, :]).sum(1)
    F = F0 - Gs.sum(1)
    z9 = Zte[:, k]
    print(f"\n[4] 逐样本核对（10000 样本）")
    print(f"    G 非负: 最小 {Gs.min()}  {'OK' if Gs.min()>=0 else 'FAIL'}")
    print(f"    G_i 实测范围: " + " ".join(
        f"G{i+1}[{Gs[:,i].min()},{Gs[:,i].max()}]" for i in range(8)))
    print(f"    F 范围 [{F.min()}, {F.max()}]   z 范围 [{z9.min():.0f}, {z9.max():.0f}]")
    d = F - z9
    print(f"    F - z 范围 [{d.min()}, {d.max()}]  标准差 {d.std():.6f}  "
          f"{'常数 OK' if d.max()-d.min()==0 else 'FAIL'}")
    print(f"    F = z + {d[0]}")

    # F=0 配置
    print(f"\n[5] F=0 的配置")
    lit_neg = np.zeros(225, dtype=int)
    for b in brs:
        for i, p in enumerate(b['pxs']):
            if b['wg'][i] < 0:
                lit_neg[p] = 1
    w9 = W[:, k]; neg = w9 < 0; pos = w9 > 0
    S = 0
    for b in brs:
        lg = lit_neg[b['pxs']]
        S += int(np.where(lg == 1, 15, b['cg']).sum())
    print(f"    负权重点亮 {int(lit_neg[neg].sum())}/{int(neg.sum())}   "
          f"正权重点亮 {int(lit_neg[pos].sum())}/{int(pos.sum())}")
    print(f"    sum(G) = {S}   F = {F0} - {S} = {F0-S}")
    print(f"    此时网络 z = {w9[lit_neg==1].sum() + bias[k]}")

    # F_min / z_min 的关系
    print(f"\n[6] 极值关系")
    print(f"    F_0 = {F0}")
    print(f"    sum(G_i^max) = {F0}")
    print(f"    F=0 时 sum(G) 取最大 = {F0}")
    print(f"    此时 z = {w9[lit_neg==1].sum() + bias[k]}")
    print(f"    而 F - z = {d[0]}  =>  {0} - ({w9[lit_neg==1].sum()+bias[k]}) = "
          f"{-w9[lit_neg==1].sum()-bias[k]}")
    print(f"    -> offset = {d[0]} 与上式 {-w9[lit_neg==1].sum()-bias[k]} "
          f"{'一致 OK' if d[0] == -w9[lit_neg==1].sum()-bias[k] else '不一致'}")

    print(f"\n{'='*100}")
    print("### 修正后的表")
    print(f"  {'支线':>4} | {'G_i0':>5} | {'G_i 范围':>12}")
    for b in brs:
        print(f"  G_{b['G']:<2} | {b['G_i0']:>5} | [0, {b['G_max']:>2}]")
    print(f"\n  F_0 = {F0}")
    print(f"  offset (F = z + offset) = {d[0]}")

    json.dump({'digit': k, 'F_0': F0, 'offset': int(d[0]),
               'sum_A': sumA,
               'branches': [{'G': b['G'], 'G_i0': b['G_i0'], 'G_min': 0,
                             'G_max': b['G_max'], 'A': b['A'],
                             'w_sum': b['w_sum'], 'sum_c': int(b['cg'].sum()),
                             'sum_max': b['sum_max'], 'sum_min': b['sum_min']}
                            for b in brs]},
              open(os.path.join(RES, 'mc_digit9_precise.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: results/mc_digit9_precise.json")


if __name__ == '__main__':
    main()

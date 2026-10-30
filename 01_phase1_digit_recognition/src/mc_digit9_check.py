#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
数字 9：最终核对（所有量在同一个脚本里算，避免残留旧表）
==========================================================
偶数层:
  像素亮起 -> 输出 15
  像素不亮 -> 输出 c_j = 15 - W_j
  G_i0 = sum_j max(c_j, 15)          <- 保证 G>=0 的最小值
  G_i  = G_i0 - sum_j(输出_j)
  F_0  = sum_i G_i^max
  F    = F_0 - sum_i G_i
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
    print(f"数字 {k}（第 {k+1} 层，偶数层）最终核对")
    print("=" * 100)

    # 全部在本地构造，不用任何旧表
    branches = []
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = np.array([r * 15 + c for r in range(r0, r0 + nr) for c in range(15)])
        wg = W[pxs, k]; cg = 15 - wg
        g0 = int(np.maximum(cg, 15).sum())
        # 完全点亮 -> 输出全 15 -> G 最小
        gmin = g0 - int((15 * len(pxs)))
        # 仅负权重点亮（输出15）、正权重与零不亮（输出 c）
        lit = (wg < 0)
        s_at_gmax = int(np.where(lit, 15, cg).sum())
        gmax = g0 - s_at_gmax
        branches.append({'G': gi + 1, 'r0': r0, 'nr': nr, 'pxs': pxs,
                         'wg': wg, 'cg': cg, 'G_i0': g0,
                         'G_min': gmin, 'G_max': gmax, 's_at_gmax': s_at_gmax})

    print(f"  {'支线':>4} | {'n':>3} | {'G_i0':>5} | {'sum@Gmax':>8} | {'G_min':>6} | "
          f"{'G_max':>6}")
    print("  " + "-" * 56)
    for b in branches:
        print(f"  G_{b['G']:<2} | {len(b['pxs']):>3} | {b['G_i0']:>5} | "
              f"{b['s_at_gmax']:>8} | {b['G_min']:>6} | {b['G_max']:>6}")

    F0 = sum(b['G_max'] for b in branches)
    Gmin_tot = sum(b['G_min'] for b in branches)
    print(f"\n  F_0 = sum(G_i^max) = {' + '.join(str(b['G_max']) for b in branches)} = {F0}")
    print(f"  sum(G_i) 范围 = [{Gmin_tot}, {F0}]")
    print(f"  F 范围 = [0, {F0 - Gmin_tot}]")

    # -------- 逐样本核对 --------
    RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    thr = to15(load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))).mean()
    Xin = (to15(Xte) > thr).astype(np.int64).reshape(len(Xte), -1)   # (10000, 225)
    Zte = Xin.astype(np.float64) @ W + bias

    lit_all = Xin.astype(bool)
    Gs = np.zeros((len(Xin), 8), dtype=np.int64)
    for gi, b in enumerate(branches):
        lit = lit_all[:, b['pxs']]                  # (B, n)
        s = np.where(lit, 15, b['cg'][None, :]).sum(1)
        Gs[:, gi] = b['G_i0'] - s
    F = F0 - Gs.sum(1)

    print(f"\n[逐样本核对 10000 个测试样本]")
    print(f"  G 非负: 最小 {Gs.min()}  {'✓' if Gs.min() >= 0 else '✗'}")
    print(f"  G_i 实测范围: " + " ".join(
        f"G{i+1}[{Gs[:,i].min()},{Gs[:,i].max()}]" for i in range(8)))
    print(f"  F 实测范围 [{F.min()}, {F.max()}]")

    z9 = Zte[:, k]
    print(f"  z (网络) 范围 [{z9.min():.0f}, {z9.max():.0f}]")
    c_pos = float(np.corrcoef(F, z9)[0, 1])
    c_neg = float(np.corrcoef(F, -z9)[0, 1])
    print(f"  corr(F, z) = {c_pos:+.6f}   corr(F, -z) = {c_neg:+.6f}")
    for sgn, nm in ((+1, 'F - z'), (-1, 'F + z')):
        d = F - sgn * z9
        print(f"  {nm}: 范围 [{d.min()}, {d.max()}]  标准差 {d.std():.6f}  "
              f"{'常数 ✓' if d.max()-d.min() == 0 else ''}")
        if d.max() - d.min() == 0:
            print(f"    => F = {'+z' if sgn>0 else '-z'} + {d[0]}")

    # -------- 直接验算 G：用 G = sum(输出) 的形式应该等价于权重和 --------
    print(f"\n[独立验算] 用 G_i = G_i0 - sum(输出)，检查是否等于 -sum(W*lit) + const")
    b0 = branches[0]
    print(f"  G_1: G_10={b0['G_i0']}, sum(c)={int(b0['cg'].sum())}")
    print(f"       G_1 = {b0['G_i0']} - [sum(c) - sum(c_j*lit_j) + 15*sum(lit_j)]")
    print(f"           = {b0['G_i0']} - {int(b0['cg'].sum())} + sum(c_j*lit) - 15*sum(lit)")
    print(f"       而 c_j - 15 = -W_j, 所以 G_1 = {b0['G_i0']-int(b0['cg'].sum())}"
          f" - sum(W_j*lit_j)")
    print(f"       即 G_1 = const - sum(W_j * lit_j)")
    print(f"       const = G_10 - sum(c) = {b0['G_i0']-int(b0['cg'].sum())}")
    # 验证
    lit = lit_all[:100, b0['pxs']]
    g_direct = b0['G_i0'] - np.where(lit, 15, b0['cg'][None, :]).sum(1)
    g_formula = (b0['G_i0'] - int(b0['cg'].sum())) - (lit * b0['wg'][None, :]).sum(1)
    print(f"       两种算法前 100 样本一致: "
          f"{'✓' if np.array_equal(g_direct, g_formula) else '✗'}")

    print(f"\n{'='*100}")
    print("### 最终参数")
    print(f"  {'支线':>4} | {'G_i0':>5} | {'G 范围':>10}")
    for b in branches:
        print(f"  G_{b['G']:<2} | {b['G_i0']:>5} | [{b['G_min']}, {b['G_max']:>2}]")
    print(f"\n  F_0 = {F0}   F 范围 = [0, {F0 - Gmin_tot}]")

    json.dump({'digit': k, 'F_0': F0,
               'branches': [{'G': b['G'], 'G_i0': b['G_i0'], 'G_min': b['G_min'],
                             'G_max': b['G_max']} for b in branches]},
              open(os.path.join(RES, 'mc_digit9_check.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\n已写出: results/mc_digit9_check.json")


if __name__ == '__main__':
    main()

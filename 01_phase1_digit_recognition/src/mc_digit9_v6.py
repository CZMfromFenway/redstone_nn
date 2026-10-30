#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
数字 9（第 10 层，偶数层）: 正确表达式 G_i = sum(点亮?c_j:15)
==============================================================
历次错误的根源: 我在脚本里写成 G_i = G_i0 - sum(点亮?c_j:15)，
  那个负号把"点亮/不亮"的贡献整个反过来了，导致
    - G 出现负值
    - F 与 z 反号
    - F=0 的配置与 czm 的要求相反

正确（与 czm 的不变量一致）:
    偶数层: c_j = 15 - W_j
            G_i = sum_j (点亮 ? c_j : 15)
    点亮 W>0 -> c<15 -> G 减小 -> F=F0-sum(G) 增大 ✓
    点亮 W<0 -> c>15 -> G 增大 -> F 减小 ✓

  区间:  G_i^min = sum_j min(c_j,15) = sum_j c_j        (全部点亮)
         G_i^max = sum_j max(c_j,15) = 15*(#负) + sum_{W>0} c_j
                                       = 15*(#负) + sum_{W>0}(15-W)
                                       = 15n - W_正和
  G_i0 由 F 的基准决定；F = F_0 - sum(G_i)
  要求: F 反号于 z（因为截断使 G 变小 -> F 变大），
        且 F=0 时"负权重全亮、正权重全不亮"
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
    print(f"数字 {k}（第 {layer} 层，{parity}层）: G_i = sum(点亮?c:15)")
    print("=" * 100)

    rows = []
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = [r * 15 + c for r in range(r0, r0 + nr) for c in range(15)]
        wg = W[pxs, k]; cg = 15 - wg
        n = len(pxs)
        gmin = int(np.minimum(cg, 15).sum())        # 全部点亮
        gmax = int(np.maximum(cg, 15).sum())        # 负权重点亮、正权重不亮
        rows.append({'G': gi + 1, 'row0': r0, 'rows': nr, 'n': n, 'wg': wg, 'cg': cg,
                     'G_min': gmin, 'G_max': gmax,
                     'w_pos': int(wg[wg > 0].sum()), 'w_neg': int(wg[wg < 0].sum()),
                     'c_min': int(cg.min()), 'c_max': int(cg.max())})

    print(f"  {'支线':>4} | {'行':>7} | {'n':>3} | {'W正和':>5} | {'W负和':>5} | "
          f"{'c范围':>9} | {'G_min':>6} | {'G_max':>6} | {'G 范围':>12}")
    print("  " + "-" * 88)
    for r in rows:
        print(f"  G_{r['G']:<2} | {r['row0']:>3}-{r['row0']+r['rows']-1:<3} | {r['n']:>3} | "
              f"{r['w_pos']:>5} | {r['w_neg']:>5} | [{r['c_min']:>2},{r['c_max']:>2}]  | "
              f"{r['G_min']:>6} | {r['G_max']:>6} | [{r['G_min']},{r['G_max']}]")

    Gmax_tot = sum(r['G_max'] for r in rows)
    Gmin_tot = sum(r['G_min'] for r in rows)
    print(f"\n  sum(G_i) 最大 = {Gmax_tot}   （负权重全亮、正权重不亮）")
    print(f"  sum(G_i) 最小 = {Gmin_tot}   （全部点亮）")

    # F 的基准: 要求 F=0 时 sum(G) 最大 -> F_0 = Gmax_tot
    F0 = Gmax_tot
    print(f"\n  取 F_0 = sum(G_i^max) = {F0}")
    print(f"  则 F 范围 = [{F0-Gmin_tot}, 0]，F=0 对应'负权重全亮、正权重不亮' ✓")

    # ---- 验证 ----
    print(f"\n[验证1] F=0 的配置")
    lit_neg = np.zeros(225, dtype=int)
    for r in rows:
        pxs = [rr * 15 + c for rr in range(r['row0'], r['row0'] + r['rows'])
               for c in range(15)]
        for i, p in enumerate(pxs):
            if r['wg'][i] < 0:
                lit_neg[p] = 1
    w9 = W[:, k]; neg = w9 < 0; pos = w9 > 0
    print(f"  负权重点亮 {int(lit_neg[neg].sum())}/{int(neg.sum())}  "
          f"正权重点亮 {int(lit_neg[pos].sum())}/{int(pos.sum())}")
    S = 0
    for r in rows:
        pxs = [rr * 15 + c for rr in range(r['row0'], r['row0'] + r['rows'])
               for c in range(15)]
        lg = lit_neg[pxs]
        S += int(np.where(lg == 1, r['cg'], 15).sum())
    print(f"  sum(G_i) = {S}   F = {F0} - {S} = {F0-S}  "
          f"{'✓' if F0-S == 0 else '✗'}")

    # ---- 逐样本核对 ----
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
        Gs[:, gi] = np.where(lit == 1, r['cg'][None, :], 15).sum(1)
    Fb = F0 - Gs.sum(1)
    zb = Zte[:2000, k]
    print(f"  G 非负: 最小 {Gs.min()}  {'✓' if Gs.min() >= 0 else '✗'}")
    print(f"  arch F 范围 [{Fb.min()}, {Fb.max()}]")
    print(f"  net  z 范围 [{zb.min():.0f}, {zb.max():.0f}]")
    corr = float(np.corrcoef(Fb, zb)[0, 1])
    print(f"  corr(F, z) = {corr:+.6f}  -> {'同向' if corr>0.999 else '反向' if corr<-0.999 else '不相关/非线性'}")
    d = Fb - zb
    print(f"  F - z: 范围 [{d.min()}, {d.max()}]  标准差 {d.std():.4f}")
    if d.max() - d.min() == 0:
        print(f"  => F = z + {d[0]}  **完全复现网络** ✓")
    elif np.corrcoef(Fb, -zb)[0,1] > 0.999:
        dr = Fb + zb
        print(f"  反向核对: F + z 范围 [{dr.min()}, {dr.max()}] 标准差 {dr.std():.4f}")
        if dr.max() - dr.min() == 0:
            print(f"  => F = -z + {dr[0]}   （需在输出端取反）")
        else:
            print(f"  => 与 z 反向但差一个非恒定偏移")

    json.dump({'digit': k, 'layer': layer, 'parity': parity, 'F_0': F0,
               'branches': [{'G': r['G'], 'G_min': r['G_min'], 'G_max': r['G_max'],
                             'w_pos': r['w_pos'], 'w_neg': r['w_neg']} for r in rows]},
              open(os.path.join(RES, 'mc_digit9_v6.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\n已写出: results/mc_digit9_v6.json")


if __name__ == '__main__':
    main()

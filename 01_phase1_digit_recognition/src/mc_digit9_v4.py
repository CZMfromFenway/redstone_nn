#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
数字 9（第 10 层，偶数层）: 正确建模
======================================
不变量（两种层都成立）:
    正权重像素点亮 -> F 增大
    负权重像素点亮 -> F 减小

偶数层:
    不亮 -> 输出 15
    点亮 -> 输出 c_j         （比 15 少，所以 G 被"多减"）
    ΔF = 15 - c_j = W_j   ==>  c_j = 15 - W_j

    G_i = G_i0 - sum_j(不亮?15:c_j)
    点亮 W_j>0 -> c<15 -> G 减小
    点亮 W_j<0 -> c>15 -> G 增大
    => G_i 最大 <=> 负权重点亮、正权重不亮
    => G_i 最小 <=> 正权重点亮、负权重不亮

    G_i0 = 使 G_i 最小值 = 0 = sum_j(正权重点亮、负权重不亮的输出)
    G_i^max = G_i0 - sum_j(负权重点亮、正权重不亮的输出)
    F_0 = sum_i G_i^max   （使 F 恒非负，F=0 时恰为"负权重全亮"配置）
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]


def main():
    z = np.load(os.path.join(RES, 'Wint_model.npz'))
    W = z['W'].astype(int); bias = z['b'].astype(int)
    k = 9; layer = k + 1; parity = '偶数'
    print("=" * 100)
    print(f"数字 {k}（第 {layer} 层，{parity}层）: 重新核算")
    print("=" * 100)
    print(f"不变量: 正权重点亮 -> F 增大 ; 负权重点亮 -> F 减小")
    print(f"偶数层: c = 15 - W ; 不亮输出 15, 点亮输出 c ; ΔF = 15-c = W ✓\n")

    rows = []
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = [r * 15 + c for r in range(r0, r0 + nr) for c in range(15)]
        wg = W[pxs, k]
        cg = 15 - wg
        n = len(pxs)
        # G 最小的配置: 正权重点亮(c)，负权重不亮(15)
        out_at_gmin = np.where(wg > 0, cg, 15)
        # G 最大的配置: 负权重点亮(c)，正权重不亮(15)
        out_at_gmax = np.where(wg < 0, cg, 15)
        g0 = int(out_at_gmin.sum())               # 使 G_min = 0
        gmax = int(g0 - out_at_gmax.sum())
        rows.append({'G': gi + 1, 'row0': r0, 'rows': nr, 'n': n,
                     'w_sum': int(wg.sum()), 'c_min': int(cg.min()),
                     'c_max': int(cg.max()), 'G_i0': g0, 'G_max': gmax,
                     'n_pos': int((wg > 0).sum()), 'n_neg': int((wg < 0).sum())})

    print(f"  {'支线':>4} | {'行':>7} | {'n':>3} | {'正/负':>7} | {'c范围':>9} | "
          f"{'G_i0':>5} | {'G_max':>6} | {'G 范围':>9}")
    print("  " + "-" * 76)
    for r in rows:
        print(f"  G_{r['G']:<2} | {r['row0']:>3}-{r['row0']+r['rows']-1:<3} | {r['n']:>3} | "
              f"{r['n_pos']:>3}/{r['n_neg']:<3} | [{r['c_min']:>2},{r['c_max']:>2}]  | "
              f"{r['G_i0']:>5} | {r['G_max']:>6} | [0, {r['G_max']:>2}]")

    F0 = sum(r['G_max'] for r in rows)
    print(f"\n  F_0 = sum(G_i^max) = {' + '.join(str(r['G_max']) for r in rows)} = {F0}")
    print(f"  F 范围 = [0, {F0}]")

    # ---- 验证 F=0 的配置 ----
    print(f"\n[验证] F=0 时的配置")
    lit = np.zeros(225, dtype=int)
    for r in rows:
        pxs = [rr * 15 + c for rr in range(r['row0'], r['row0'] + r['rows'])
               for c in range(15)]
        wg = W[pxs, k]
        for i, p in enumerate(pxs):
            lit[p] = 1 if wg[i] < 0 else 0       # 负权重点亮
    w9 = W[:, k]
    neg = w9 < 0; pos = w9 > 0
    print(f"  点亮总数 {lit.sum()}")
    print(f"    负权重: {int(lit[neg].sum())} / {int(neg.sum())}  "
          f"{'全部点亮 OK' if int(lit[neg].sum())==int(neg.sum()) else '不符'}")
    print(f"    正权重: {int(lit[pos].sum())} / {int(pos.sum())}  "
          f"{'全部不亮 OK' if int(lit[pos].sum())==0 else '不符'}")

    # 直接算 F
    tot = 0
    for r in rows:
        pxs = [rr * 15 + c for rr in range(r['row0'], r['row0'] + r['rows'])
               for c in range(15)]
        wg = W[pxs, k]; cg = 15 - wg
        lg = lit[pxs]
        tot += int(np.where(lg == 1, cg, 15).sum())
    print(f"\n  该配置下 sum(G_i) = {tot}")
    print(f"  F = F_0 - sum(G_i) = {F0} - {tot} = {F0 - tot}  "
          f"{'✓ 等于 0' if F0 - tot == 0 else '✗'}")

    # ---- 与网络核对 ----
    print(f"\n[与网络核对]")
    print(f"  网络数字 {k}: bias = {bias[k]}, 权重和 = {W[:,k].sum()}")
    # 计算网络 z 与该配置的一一对应
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
    RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    thr = to15(load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))).mean()
    Fte = (to15(Xte) > thr).astype(np.float64).reshape(len(Xte), -1)
    Zte = Fte @ W + bias
    print(f"  网络 z 在测试集范围 [{Zte[:,k].min():.0f}, {Zte[:,k].max():.0f}]，"
          f"跨度 {Zte[:,k].max()-Zte[:,k].min():.0f}")

    # 架构 F 是否 = 常数 - z  (或 + z)
    # 逐样本算架构 F 并与 z 比对
    print(f"\n  逐样本核对（取前 2000 个测试样本）")
    def arch_F(Xb):
        """给定二值输入 (B,225)，返回每条支线的 G 与总 F"""
        B = len(Xb)
        Gs = np.zeros((B, 8), dtype=np.int64)
        for gi, r in enumerate(rows):
            pxs = [rr * 15 + c for rr in range(r['row0'], r['row0'] + r['rows'])
                   for c in range(15)]
            wg = W[pxs, k]; cg = 15 - wg
            lit = Xb[:, pxs]                       # (B, n)
            s = np.where(lit == 1, cg[None, :], 15).sum(1)
            Gs[:, gi] = r['G_i0'] - s
        F = F0 - Gs.sum(1)
        return F
    Fb = arch_F(Fte[:2000])
    zb = Zte[:2000, k]
    print(f"    arch F 范围 [{Fb.min()}, {Fb.max()}]")
    print(f"    net  z 范围 [{zb.min():.0f}, {zb.max():.0f}]")
    # 相关性
    corr = float(np.corrcoef(Fb, zb)[0, 1])
    print(f"    corr(F, z) = {corr:+.6f}   -> "
          f"{'同向' if corr > 0 else '反向' if corr < 0 else '无相关'}")
    d = Fb - zb
    print(f"    F - z 范围 [{d.min()}, {d.max()}]   "
          f"{'常数 ✓' if d.max() - d.min() == 0 else '非常数 ✗'}")
    if d.max() - d.min() == 0:
        print(f"    => F = z + {d[0]}")

    json.dump({'digit': k, 'layer': layer, 'parity': parity, 'F_0': F0,
               'G_i0': [r['G_i0'] for r in rows],
               'G_max': [r['G_max'] for r in rows],
               'const_offset': int(d[0]) if d.max() - d.min() == 0 else None},
              open(os.path.join(RES, 'mc_digit9_v4.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\n已写出: results/mc_digit9_v4.json")


if __name__ == '__main__':
    main()

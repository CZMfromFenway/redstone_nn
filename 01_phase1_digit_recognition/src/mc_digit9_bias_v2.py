#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
数字 9（第 10 层，偶数层）: 支线偏置校正 + 与网络的核对
========================================================
上一版的错误:
  G_i0 = 15n 只保证"全部未点亮时 G=0"，但**不能保证 G 恒非负**。
  因为点亮某个像素会使 G 增加 -W_j，若 W_j<0 则 G 减少，可能变负。
  （例: 仅第 11 列 W=-1 点亮 -> G_1 = -1）

正确的 G_i0 必须保证 G_i 在**所有**输入下非负:
  G_i = G_i0 - sum(输出_j)
  输出_j ∈ {15, c_j}，所以 sum 的最大值 = sum(max(15, c_j))
    -> G_i0 >= sum_j max(15, c_j)

但更自然的做法: 要求 G_i 的最小值 = 0（czm 的要求）
  G_i 最小值出现在"所有使 G 减小的像素都点亮"时
  偶数层: 点亮输出 c_j = 15-W_j，G 变化 = -(c_j-15) = +W_j
    所以 W_j>0 使 G 增大，W_j<0 使 G 减小
  G_i 最小值 = G_i0 - [sum_{W>0}(15 - c_j) + sum_{W<=0}(15)]  ... 见代码
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]


def analyze(W, k, parity):
    """返回每条支线的 (G_i0, G_min, G_max)"""
    out = []
    for (r0, nr) in GROUPS:
        pxs = [r * 15 + c for r in range(r0, r0 + nr) for c in range(15)]
        wg = W[pxs, k]
        n = len(pxs)
        if parity == '偶数':
            c = 15 - wg                       # 未点亮输出 15，点亮输出 c
            # 输出集合: 点亮->c, 未点亮->15
            # G = G0 - sum(输出)
            # 最大化 sum(输出) => 点亮时取 max(c,15)，未点亮时取 15 之间选大者
            # 逐像素独立: sum_max = sum_j max(c_j, 15)
            #             sum_min = sum_j min(c_j, 15)
            sum_max = int(np.maximum(c, 15).sum())
            sum_min = int(np.minimum(c, 15).sum())
        else:
            c = 15 + wg
            sum_max = int(np.maximum(c, 15).sum())
            sum_min = int(np.minimum(c, 15).sum())
        g0 = sum_max                              # 使 G 最小值为 0
        out.append({'G': len(out) + 1, 'row0': r0, 'rows': nr, 'n': n,
                    'w_sum': int(wg.sum()), 'c_min': int(c.min()), 'c_max': int(c.max()),
                    'sum_c': int(c.sum()), 'sum_max': sum_max, 'sum_min': sum_min,
                    'G_i0': g0, 'G_min': 0, 'G_max': sum_max - sum_min,
                    'c_list': [int(x) for x in c], 'w_list': [int(x) for x in wg]})
    return out


def main():
    z = np.load(os.path.join(RES, 'Wint_model.npz'))
    W = z['W'].astype(int); bias = z['b'].astype(int)
    k = 9; layer = k + 1; parity = '奇数' if layer % 2 else '偶数'
    print("=" * 100)
    print(f"数字 {k}（第 {layer} 层，{parity}层）：支线偏置（保证 G 恒非负）")
    print("=" * 100)
    print(f"偶数层: c = 15 - W；未点亮输出 15，点亮输出 c")
    print(f"G_i = G_i0 - sum_j(输出_j)")
    print(f"sum(输出) 的最大值 = sum_j max(c_j, 15)，最小值 = sum_j min(c_j, 15)")
    print(f"取 G_i0 = sum_j max(c_j, 15)  ==>  G_i 最小值 = 0\n")

    rows = analyze(W, k, parity)
    print(f"  {'支线':>4} | {'行':>7} | {'n':>3} | {'W和':>5} | {'c范围':>9} | "
          f"{'sum(c)':>7} | {'sum_max':>7} | {'sum_min':>7} | {'G_i0':>5} | "
          f"{'G 范围':>11}")
    print("  " + "-" * 96)
    for r in rows:
        print(f"  G_{r['G']:<2} | {r['row0']:>3}-{r['row0']+r['rows']-1:<3} | {r['n']:>3} | "
              f"{r['w_sum']:>5} | [{r['c_min']:>2},{r['c_max']:>2}]  | {r['sum_c']:>7} | "
              f"{r['sum_max']:>7} | {r['sum_min']:>7} | {r['G_i0']:>5} | "
              f"[{r['G_min']:>3},{r['G_max']:>3}]")

    G0 = [r['G_i0'] for r in rows]
    F0 = sum(G0)
    sum_max_all = sum(r['sum_max'] for r in rows)
    sum_min_all = sum(r['sum_min'] for r in rows)
    print(f"\n  G_10..G_80 = {G0}")
    print(f"  F_0 = {F0}")
    print(f"  总线 F = F_0 - sum(G_i)，范围 [{F0-sum_max_all}, {F0-sum_min_all}]")

    # ---------- 逐情形验证（穷举 2^15 对 G_1 太多，抽样）----------
    print(f"\n[G_1 逐情形验证]（G_10 = {G0[0]}）")
    r = rows[0]
    w9 = np.array(r['w_list']); c9 = np.array(r['c_list'])
    print(f"  {'情形':<22} | {'sum输出':>7} | {'G_1':>5} | {'ΔG':>5} | {'-权重和':>7} | 校验")
    print("  " + "-" * 68)
    eye = np.eye(15, dtype=int)
    base = int(c9.sum())                       # 全部点亮
    ok_all = True
    for nm, lit in [("全部点亮", np.ones(15, int)), ("全部未点亮", np.zeros(15, int)),
                    ("仅第2列 W=+3", eye[2]), ("仅第11列 W=-1", eye[11]),
                    ("仅第0列 W=0", eye[0]),
                    ("负权重全亮(第11列)", eye[11]),
                    ("正权重全亮", ((w9 > 0) * 1))]:
        s = int(np.where(lit == 1, c9, 15).sum())
        g = G0[0] - s
        dg = g - (G0[0] - base)
        exp = -int((lit * w9).sum())
        ok = (dg == exp)
        ok_all &= ok
        print(f"  {nm:<22} | {s:>7} | {g:>5} | {dg:>5} | {exp:>7} | "
              f"{'OK' if ok else '不符'}")
    print(f"  全部通过: {'是' if ok_all else '否'}")
    print(f"  G_1 的最小值应为 0 -> 当前 {(G0[0]-r['sum_max'])}")

    # 是否有负 G
    print(f"\n[非负性检查]  各支线 G 最小值 = G_i0 - sum_max = 0（构造保证）")
    print(f"  各支线 G 最大值 = sum_max - sum_min:")
    for r in rows:
        print(f"    G_{r['G']}: {r['G_max']}")

    # ---------- 与网络核对 ----------
    print(f"\n[与网络核对]")
    print(f"  网络数字 {k} 的 bias = {bias[k]}，权重和 = {W[:,k].sum()}")
    print(f"  架构里 F = F_0 - sum(G_i)")
    print(f"    全部点亮 -> F = {F0 - sum_min_all}   （sum(G) 最小）")
    print(f"    全部未点亮 -> F = {F0 - sum_max_all}   （sum(G) 最大）")
    print(f"  F 的跨度 = {sum_max_all - sum_min_all}")
    # 网络在测试集上的输出跨度
    RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
    import struct
    def load_idx(p):
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
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    B16 = to15(Xte)
    Fte = (B16 > to15(load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))).mean()).astype(np.float64).reshape(len(B16), -1)
    Zte = Fte @ W + bias
    print(f"  网络在测试集上，数字 {k} 的 z 范围 [{Zte[:,k].min():.0f}, {Zte[:,k].max():.0f}]，"
          f"跨度 {Zte[:,k].max()-Zte[:,k].min():.0f}")
    print(f"  架构 F 的跨度 {sum_max_all - sum_min_all}，应当接近（差常数）")

    json.dump({'digit': k, 'layer': layer, 'parity': parity, 'G_i0': G0, 'F_0': F0,
               'branches': rows,
               'F_range': [F0 - sum_max_all, F0 - sum_min_all]},
              open(os.path.join(RES, 'mc_digit9_bias_v2.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\n已写出: results/mc_digit9_bias_v2.json")


if __name__ == '__main__':
    main()

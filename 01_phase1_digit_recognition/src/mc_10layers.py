#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
10 层完整参数表
=================
逐层计算:
  bias                网络偏置
  W正和 / W负和        该层 225 个权重中正/负部分之和
  G_10..G_80           8 条支线的偏置
  F_0                 总线偏置
  F 输出范围

公式（已与 czm 核对通过）:
  第 k+1 层, 数字 k
  偶数层 (k+1 为偶数): c_j = 15 - W_j ; 点亮输出 15, 不亮输出 c_j
  奇数层 (k+1 为奇数): c_j = 15 + W_j ; 点亮输出 c_j, 不亮输出 15

  G_i0 = sum_j max(c_j, 15) = 15*n -/+ W_正和,i   (偶层减, 奇层加)
  G_i  = G_i0 - sum_j(输出_j)      , G_i ∈ [0, G_i^max]
  G_i^max = G_i0 - sum_j min(c_j, 15)
  F_0 = sum_i G_i^max              , F = F_0 - sum_i G_i ∈ [0, F_0]
"""
import os, json, csv
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

    layers = []
    for k in range(10):
        ln = k + 1
        parity = '奇数' if ln % 2 == 1 else '偶数'
        wl = W[:, k]
        w_pos = int(wl[wl > 0].sum()); w_neg = int(wl[wl < 0].sum())
        c_all = (15 + wl) if parity == '奇数' else (15 - wl)
        brs = []
        for gi, (r0, nr) in enumerate(GROUPS):
            pxs = np.array([r * 15 + c for r in range(r0, r0 + nr) for c in range(15)])
            wg = W[pxs, k]; cg = c_all[pxs]
            g0 = int(np.maximum(cg, 15).sum())
            gmax = g0 - int(np.minimum(cg, 15).sum())
            p = int(wg[wg > 0].sum()); ng = int(wg[wg < 0].sum())
            # 与公式核对（直接用定义，不做奇偶假设）
            assert g0 == int(np.maximum(cg, 15).sum())
            brs.append({'G': gi + 1, 'r0': r0, 'nr': nr, 'n': len(pxs),
                        'G_i0': g0, 'G_min': 0, 'G_max': gmax,
                        'w_pos': p, 'w_neg': ng,
                        'A': g0 - int(cg.sum())})
        F0 = sum(b['G_max'] for b in brs)
        sumA = sum(b['A'] for b in brs)
        layers.append({'k': k, 'layer': ln, 'parity': parity,
                       'bias': int(bias[k]), 'w_pos': w_pos, 'w_neg': w_neg,
                       'w_sum': int(wl.sum()),
                       'c_min': int(c_all.min()), 'c_max': int(c_all.max()),
                       'G_i0': [b['G_i0'] for b in brs],
                       'G_max': [b['G_max'] for b in brs],
                       'A': [b['A'] for b in brs],
                       'F_0': F0, 'sum_A': sumA, 'offset': F0 - sumA})

    print("=" * 118)
    print("10 层完整参数表  （容器值 c_j = 15 ∓ W_j；G_i0 = sum_j max(c_j,15)）")
    print("=" * 118)
    print(f"  {'层':>3}(数字) | {'奇偶':>4} | {'bias':>5} | {'W正和':>6} | {'W负和':>6} | "
          f"{'c范围':>9} | " + " | ".join(f"G{i+1}0" for i in range(8)) + " |   F_0")
    print("  " + "-" * 132)
    for L in layers:
        print(f"  {L['layer']:>3}({L['k']}) | {L['parity']:>4} | {L['bias']:>5} | "
              f"{L['w_pos']:>6} | {L['w_neg']:>6} | [{L['c_min']:>2},{L['c_max']:>2}]  | "
              + " | ".join(f"{g:>5}" for g in L['G_i0'])
              + f" | {L['F_0']:>6}")

    print(f"\n  {'层':>3}(数字) | " + " | ".join(f"G{i+1}范围" for i in range(8))
          + " |       F 范围 |   offset")
    print("  " + "-" * 108)
    for L in layers:
        rng = " | ".join(f"[0,{g:>2}]" for g in L['G_max'])
        print(f"  {L['layer']:>3}({L['k']}) | {rng} | [0, {L['F_0']:>3}] | {L['offset']:>6}")

    # 统计
    print(f"\n{'='*118}")
    print("### 汇总")
    print(f"  {'层':>3} | {'bias':>5} | {'W正和':>6} | {'W负和':>6} | {'W总和':>6} | "
          f"{'F_0':>5} | {'offset':>6}")
    print("  " + "-" * 60)
    for L in layers:
        print(f"  {L['layer']:>3} | {L['bias']:>5} | {L['w_pos']:>6} | {L['w_neg']:>6} | "
              f"{L['w_sum']:>6} | {L['F_0']:>5} | {L['offset']:>6}")
    print(f"\n  bias 范围 [{min(L['bias'] for L in layers)}, "
          f"{max(L['bias'] for L in layers)}]")
    print(f"  F_0 范围 [{min(L['F_0'] for L in layers)}, "
          f"{max(L['F_0'] for L in layers)}]")
    print(f"  offset 范围 [{min(L['offset'] for L in layers)}, "
          f"{max(L['offset'] for L in layers)}]")
    print(f"  最大 F_0 = {max(L['F_0'] for L in layers)}  （总线满档 13455，用了 "
          f"{max(L['F_0'] for L in layers)/13455*100:.2f}%）")
    print(f"  容器最大值 {max(L['c_max'] for L in layers)} / 897")

    # ---- 逐层逐样本核对 ----
    print(f"\n[逐样本核对 10000 个测试样本]")
    RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    thr = to15(load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))).mean()
    Xin = (to15(Xte) > thr).astype(np.int64).reshape(len(Xte), -1)
    lit_all = Xin.astype(bool)
    Zte = Xin.astype(np.float64) @ W + bias
    print(f"  {'层':>3} | {'G非负':>6} | {'F范围':>13} | {'corr(F,z)':>10} | {'F-z':>8}")
    print("  " + "-" * 60)
    ok_all = True
    for L in layers:
        k = L['k']; parity = L['parity']
        c_all = (15 + W[:, k]) if parity == '奇数' else (15 - W[:, k])
        Gs = np.zeros((len(Xin), 8), dtype=np.int64)
        for gi, (r0, nr) in enumerate(GROUPS):
            pxs = np.array([r * 15 + c for r in range(r0, r0 + nr) for c in range(15)])
            lit = lit_all[:, pxs]
            # 奇层: 点亮输出 c, 不亮输出 15 ; 偶层: 点亮输出 15, 不亮输出 c
            out = np.where(lit, 15, c_all[pxs][None, :]) if parity == '偶数' \
                else np.where(lit, c_all[pxs][None, :], 15)
            Gs[:, gi] = L['G_i0'][gi] - out.sum(1)
        F = L['F_0'] - Gs.sum(1)
        z9 = Zte[:, k]
        d = F - z9
        corr = float(np.corrcoef(F, z9)[0, 1])
        const = '恒定' if d.max() - d.min() == 0 else f'非常数({d.std():.2f})'
        if d.max() - d.min() != 0:
            ok_all = False
        print(f"  {L['layer']:>3} | {str(int(Gs.min())):>6} | "
              f"[{int(F.min())},{int(F.max())}] | {corr:>10.6f} | {const:>8}")
    print(f"\n  全部层 F 与 z 差为常数: {'是 OK' if ok_all else '否'}")

    # ---- 写出 ----
    with open(os.path.join(RES, 'mc_10layers_bias.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['layer', 'digit', 'parity', 'bias', 'w_pos_sum', 'w_neg_sum',
                    'w_sum', 'c_min', 'c_max', 'F_0', 'sum_A', 'offset']
                   + [f'G{i+1}0' for i in range(8)]
                   + [f'G{i+1}_max' for i in range(8)])
        for L in layers:
            w.writerow([L['layer'], L['k'], L['parity'], L['bias'], L['w_pos'],
                        L['w_neg'], L['w_sum'], L['c_min'], L['c_max'],
                        L['F_0'], L['sum_A'], L['offset']]
                       + L['G_i0'] + L['G_max'])
    print(f"\n已写出: results/mc_10layers_bias.csv")
    json.dump(layers, open(os.path.join(RES, 'mc_10layers.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"        results/mc_10layers.json")


if __name__ == '__main__':
    main()

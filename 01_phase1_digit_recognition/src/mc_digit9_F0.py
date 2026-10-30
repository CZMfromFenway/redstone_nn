#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
数字 9（第 10 层，偶数层）: F_0 = sum(G_i^max)  —— 正确取法
=============================================================
czm 的定义:
  F_0 = sum_i G_i^max       （每条支线各自最大值之和）
  这样所有 G_i 同时取最大时 F = 0，且 F 恒非负。

验证目标:
  F = 0 时的输入配置，应当恰好是"负权重像素全亮、正权重像素全不亮"

偶数层: c_j = 15 - W_j，未点亮输出 15，点亮输出 c_j
  G_i = G_i0 - sum(输出)
  点亮 W_j>0 -> 输出 c_j < 15 -> sum 变小 -> G 变大  ... 等等，重新推
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
    print(f"数字 {k}（第 {layer} 层，{parity}层）: F_0 = sum(G_i^max)")
    print("=" * 100)

    # ---- 先确定 G_i0：使 G_i 最小值 = 0 ----
    # G_i = G_i0 - sum(输出);  输出 ∈ {15, c_j}
    # sum 最大 => 每像素取 max(15, c_j) => G 最小
    # G_i0 = sum_j max(15, c_j)  -> G_i 最小 = 0
    print(f"\n[步骤1] G_i0 = sum_j max(15, c_j)   （使 G_i 最小值为 0）")
    rows = []
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = [r * 15 + c for r in range(r0, r0 + nr) for c in range(15)]
        wg = W[pxs, k]
        cg = 15 - wg
        g0 = int(np.maximum(cg, 15).sum())
        rows.append({'G': gi + 1, 'row0': r0, 'rows': nr, 'n': len(pxs),
                     'wg': wg, 'cg': cg, 'G_i0': g0,
                     'sum_max': int(np.maximum(cg, 15).sum()),
                     'sum_min': int(np.minimum(cg, 15).sum()),
                     'w_sum': int(wg.sum())})

    # ---- 步骤2: G_i 的最大值 ----
    # G_i 最大 <=> sum(输出) 最小 <=> 每像素取 min(15, c_j)
    # 点亮 W_j>0 -> c_j<15 -> 取 c_j (点亮)
    # 不亮 W_j<0 -> c_j>15 -> 取 15 (不亮)
    # 所以: 正权重点亮、负权重不亮 => 最大化 G
    print(f"\n[步骤2] G_i^max = G_i0 - sum_j min(15, c_j)")
    print(f"  最小化 sum(输出) 的配置: W_j>0 点亮(取 c_j<15)，W_j<=0 不亮(取 15)")
    print(f"  即: **正权重点亮、负权重不亮** 时 G_i 最大")
    print(f"  （注意这与 czm 说的相反，见后面验证）\n")
    for r in rows:
        wg, cg = r['wg'], r['cg']
        # 最大化 G: 正权重点亮(c), 负权重不亮(15)
        best = np.where(wg > 0, cg, 15)
        r['G_max'] = int(r['G_i0'] - best.sum())
        r['sum_at_gmax'] = int(best.sum())
        # 最小化 G: 正权重不亮(15), 负权重点亮(c)
        worst = np.where(wg < 0, cg, 15)
        r['G_min_check'] = int(r['G_i0'] - worst.sum())
        r['sum_at_gmin'] = int(worst.sum())
    print(f"  {'支线':>4} | {'n':>3} | {'W和':>5} | {'G_i0':>5} | {'sum@Gmax':>8} | "
          f"{'G_max':>6} | {'sum@Gmin':>8} | {'G_min':>6}")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  G_{r['G']:<2} | {r['n']:>3} | {r['w_sum']:>5} | {r['G_i0']:>5} | "
              f"{r['sum_at_gmax']:>8} | {r['G_max']:>6} | {r['sum_at_gmin']:>8} | "
              f"{r['G_min_check']:>6}")

    # ---- 步骤3: F_0 ----
    F0 = sum(r['G_max'] for r in rows)
    print(f"\n[步骤3] F_0 = sum(G_i^max) = {' + '.join(str(r['G_max']) for r in rows)}")
    print(f"       = {F0}")
    print(f"  F = F_0 - sum(G_i)")
    print(f"  sum(G_i) 最大时 = sum(G_i^max) = {F0}  ->  F = 0  ✓（恒非负）")

    # ---- 步骤4: F=0 的配置验证 ----
    print(f"\n[步骤4] 验证 F=0 时的输入配置")
    # F=0 <=> sum(G_i) = F0 <=> 每条 G_i 都取最大
    # G_i 最大 <=> 正权重点亮、负权重不亮
    lit_for_F0 = np.zeros(225, dtype=int)
    for r in rows:
        pxs = [rr * 15 + c for rr in range(r['row0'], r['row0'] + r['rows'])
               for c in range(15)]
        for i, p in enumerate(pxs):
            lit_for_F0[p] = 1 if r['wg'][i] > 0 else 0
    w9_all = W[:, k]
    neg_mask = (w9_all < 0)
    pos_mask = (w9_all > 0)
    print(f"  F=0 时点亮的像素数: {lit_for_F0.sum()}")
    print(f"    其中负权重像素: {int((lit_for_F0[neg_mask] == 1).sum())} / {int(neg_mask.sum())}")
    print(f"    其中正权重像素: {int((lit_for_F0[pos_mask] == 1).sum())} / {int(pos_mask.sum())}")
    print(f"  => F=0 时: 正权重全部点亮, 负权重全部不亮")

    # 反过来验证: 负权重全亮、正权重不亮 时 F = ?
    lit_alt = np.zeros(225, dtype=int)
    lit_alt[neg_mask] = 1
    Ssum = 0
    for r in rows:
        pxs = [rr * 15 + c for rr in range(r['row0'], r['row0'] + r['rows'])
               for c in range(15)]
        wg = r['wg']; cg = r['cg']
        litg = lit_alt[pxs]
        Ssum += int(np.where(litg == 1, cg, 15).sum())
    print(f"\n  反过来: 负权重全亮、正权重不亮")
    print(f"    sum(G_i) = {Ssum}   ->  F = {F0} - {Ssum} = {F0-Ssum}")
    print(f"    即 G_i 全部取最小 (=0)，F 取最大值 {F0-Ssum}")

    # ---- 步骤5: 与网络核对 ----
    print(f"\n[步骤5] 与网络核对")
    print(f"  网络数字 {k}: bias = {bias[k]}, 权重和 = {W[:,k].sum()}")
    print(f"  网络 z = w·x + b,  权重非零个数 {( W[:,k]!=0).sum()}")
    print(f"  架构 F = F_0 - sum(G_i)")
    print(f"    F 的跨度（全不亮 -> F=0 配置）= {F0}")
    print(f"    G_i^max 之和 = {F0}")
    print(f"  网络在测试集上 z 的跨度需要单独算")

    # ---- 输出最终表 ----
    print(f"\n{'='*100}")
    print(f"### 数字 9（第 {layer} 层，{parity}层）最终参数")
    print(f"{'='*100}")
    print(f"  {'支线':>4} | {'行':>7} | {'n':>3} | {'G_i0':>5} | {'G_i^max':>7} | "
          f"{'G 范围':>10}")
    print("  " + "-" * 60)
    for r in rows:
        print(f"  G_{r['G']:<2} | {r['row0']:>3}-{r['row0']+r['rows']-1:<3} | {r['n']:>3} | "
              f"{r['G_i0']:>5} | {r['G_max']:>7} | [0, {r['G_max']:>2}]")
    print(f"\n  F_0 = sum(G_i^max) = {F0}")
    print(f"  F 范围 = [0, {F0}]   （F=0 时正权重全亮、负权重全不亮）")

    json.dump({'digit': k, 'layer': layer, 'parity': parity,
               'F_0': F0,
               'branches': [{'G': r['G'], 'row0': r['row0'], 'rows': r['rows'],
                             'n': r['n'], 'G_i0': r['G_i0'], 'G_max': r['G_max'],
                             'w_sum': r['w_sum']} for r in rows]},
              open(os.path.join(RES, 'mc_digit9_F0.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\n已写出: results/mc_digit9_F0.json")


if __name__ == '__main__':
    main()

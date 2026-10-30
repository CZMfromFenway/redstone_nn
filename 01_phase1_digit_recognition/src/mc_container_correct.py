#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
MC 容器值：正确公式（权重直接加减，无倍数）
=============================================
czm 给定的关系:
    奇数层: c = 15 + W        （截断使 G 减小）
    偶数层: c = 15 - W        （截断使 G 增大）

  验证: W=0 -> c=15 ; W=3 -> c=18 ; W=1 -> c=16
        偶数层反转: W=3 -> c=12 ; W=1 -> c=14

  c=15 是"权重为 0 时的基准值"，15 不是倍数。

容器约束 [0,897]:
  奇数层: 15+W ∈ [0,897]  -> W ∈ [-15, 882]
  偶数层: 15-W ∈ [0,897]  -> W ∈ [-882, 15]
  实际权重范围 [-12, 7]，两式都合法 -> 容器值落在 [3, 27]

支线: G_i = G_i0 - sum_j c_ij
      G_i0 = max(0, -min_j c_ij)   （保证 G_i >= 0）
总线: F = F_0 - sum_i G_i ,  F_0 = sum_i G_i0
"""
import os, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]      # G_1..G_8


def main():
    print("=" * 94)
    print("MC 容器值（正确公式: c = 15 ± W）")
    print("=" * 94)
    z = np.load(os.path.join(RES, 'Wint_model.npz'))
    W = z['W'].astype(int); bias = z['b'].astype(int)
    print(f"权重范围 [{W.min()}, {W.max()}]   形状 {W.shape}")

    print(f"\n[公式验证]")
    print(f"  {'W':>4} | {'奇数层 c=15+W':>13} | {'偶数层 c=15-W':>13}")
    print("  " + "-" * 38)
    for w in (-12, -3, -1, 0, 1, 3, 7):
        print(f"  {w:>4} | {15+w:>13} | {15-w:>13}")

    k = 9; row = 14
    idx = [row * 15 + c for c in range(15)]
    w9 = W[idx, k]
    print(f"\n[数字 9 的 G_1: 最底行 15 个像素]")
    print(f"  权重: {w9.tolist()}")
    c_odd = 15 + w9
    c_even = 15 - w9
    print(f"  奇数层容器: {c_odd.tolist()}")
    print(f"  偶数层容器: {c_even.tolist()}")

    # ---------- 逐像素表 ----------
    print(f"\n  {'列':>3} | {'像素':>5} | {'W':>4} | {'c(奇)':>6} | {'c(偶)':>6} | "
          f"{'ΔG 奇':>6} | {'ΔF 奇':>6}")
    print("  " + "-" * 58)
    for i, p in enumerate(idx):
        w = int(w9[i]); co = int(c_odd[i]); ce = int(c_even[i])
        print(f"  {i:>3} | {p:>5} | {w:>4} | {co:>6} | {ce:>6} | "
              f"{15-co:>6} | {co-15:>6}")
    print(f"\n  校验: ΔG(奇) = 15-c = -W ; ΔF(奇) = W")
    assert np.all(15 - c_odd == -w9)
    assert np.all(15 - c_even == w9)
    print(f"  ✓ 两种层都满足" + ("（偶数层符号相反）" if True else ""))

    # ---------- G_1 情形模拟 ----------
    print(f"\n[G_1 输出模拟]")
    eye = np.eye(15, dtype=int)
    cases = [("全部未点亮", np.zeros(15, int)), ("全部点亮", np.ones(15, int)),
             ("仅第2列 W=+3", eye[2]), ("仅第3列 W=+1", eye[3]),
             ("仅第11列 W=-1", eye[11]), ("仅第0列 W=0", eye[0])]
    g_off_o = int(c_odd.sum()); g_off_e = int(c_even.sum())
    print(f"  {'情形':<18} | {'G(奇)':>6} | {'ΔG奇':>5} | {'ΔF奇':>5} | 校验")
    print("  " + "-" * 56)
    for nm, lit in cases:
        g = int(np.where(lit == 1, 15, c_odd).sum())
        dg = g - g_off_o
        exp = int((lit * w9).sum())
        ok = (dg == -exp)
        print(f"  {nm:<18} | {g:>6} | {dg:>5} | {-dg:>5} | {'OK' if ok else '不符'}")
    print(f"  （ΔF(奇) = -ΔG = 点亮像素的权重和）")

    # ---------- 8 条支线 ----------
    print(f"\n[数字 9 的 8 条支线]")
    print(f"  {'支线':>4} | {'行':>7} | {'n':>3} | {'权重和':>6} | {'c(奇)范围':>11} |"
          f" {'G_i0(奇)':>9} | {'c(偶)范围':>11} | {'G_i0(偶)':>9}")
    print("  " + "-" * 84)
    g0_odd, g0_even = [], []
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = [r * 15 + c for r in range(r0, r0 + nr) for c in range(15)]
        wg = W[pxs, k]
        co = 15 + wg; ce = 15 - wg
        go = int(max(0, -co.min())); ge = int(max(0, -ce.min()))
        g0_odd.append(go); g0_even.append(ge)
        print(f"  G_{gi+1:<2} | {r0:>3}-{r0+nr-1:<3} | {nr*15:>3} | {wg.sum():>6} | "
              f"[{co.min():>4},{co.max():>4}] | {go:>9} | "
              f"[{ce.min():>4},{ce.max():>4}] | {ge:>9}")
    print(f"\n  奇数层: G_i0 = {g0_odd}   F_0 = {sum(g0_odd)}")
    print(f"  偶数层: G_i0 = {g0_even}   F_0 = {sum(g0_even)}")

    # ---------- 全局 ----------
    co_all = 15 + W; ce_all = 15 - W
    print(f"\n[全局: 10 层 x 225 容器]")
    print(f"  奇数层容器范围 [{co_all.min()}, {co_all.max()}]   "
          f"负值 {(co_all<0).sum()}  超897 {(co_all>897).sum()}")
    print(f"  偶数层容器范围 [{ce_all.min()}, {ce_all.max()}]   "
          f"负值 {(ce_all<0).sum()}  超897 {(ce_all>897).sum()}")

    # ---------- 10 层偏置表 ----------
    print(f"\n[10 层 x 8 支线 的偏置]")
    print(f"  {'数字':>4} | {'奇偶':>4} | " +
          " | ".join(f"G{i+1}0" for i in range(8)) + " |   F_0")
    print("  " + "-" * 84)
    rows_bias = []
    for kk in range(10):
        parity = '奇数' if (kk + 1) % 2 == 1 else '偶数'
        gs = []
        for (r0, nr) in GROUPS:
            pxs = [r * 15 + c for r in range(r0, r0 + nr) for c in range(15)]
            wg = W[pxs, kk]
            cg = (15 + wg) if parity == '奇数' else (15 - wg)
            gs.append(int(max(0, -cg.min())))
        print(f"  {kk:>4} | {parity:>4} | " + " | ".join(f"{g:>3}" for g in gs) +
              f" | {sum(gs):>5}")
        rows_bias.append({'digit': kk, 'parity': parity, 'F0': sum(gs),
                          **{f'G{i+1}0': gs[i] for i in range(8)}})

    # ---------- 写出 CSV ----------
    out = []
    for kk in range(10):
        parity = '奇数' if (kk + 1) % 2 == 1 else '偶数'
        for gi, (r0, nr) in enumerate(GROUPS):
            for rr in range(r0, r0 + nr):
                for cc in range(15):
                    p = rr * 15 + cc
                    w = int(W[p, kk])
                    c = (15 + w) if parity == '奇数' else (15 - w)
                    out.append({'layer': kk + 1, 'digit': kk, 'parity': parity,
                                'G': gi + 1, 'row': rr, 'col': cc, 'pixel': p,
                                'weight': w, 'container': c})
    with open(os.path.join(RES, 'mc_container_TABLE.csv'), 'w', newline='',
              encoding='utf-8') as f:
        wtr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wtr.writeheader(); wtr.writerows(out)
    with open(os.path.join(RES, 'mc_bias_TABLE.csv'), 'w', newline='',
              encoding='utf-8') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows_bias[0].keys()))
        wtr.writeheader(); wtr.writerows(rows_bias)
    print(f"\n[已写出]")
    print(f"  results/mc_container_TABLE.csv  （10 层 x 225 容器 = 2250 行）")
    print(f"  results/mc_bias_TABLE.csv       （10 层的 F_0 与 G_i0）")

    json.dump({'formula': 'c = 15 +/- W (odd: +, even: -)',
               'w_range': [int(W.min()), int(W.max())],
               'container_odd_range': [int(co_all.min()), int(co_all.max())],
               'container_even_range': [int(ce_all.min()), int(ce_all.max())],
               'digit9_G1_weights': [int(x) for x in w9],
               'digit9_G1_containers_odd': [int(x) for x in c_odd],
               'digit9_G1_containers_even': [int(x) for x in c_even],
               'digit9_G1_G10_odd': int(g0_odd[0]),
               'digit9_G1_G10_even': int(g0_even[0])},
              open(os.path.join(RES, 'mc_container_summary.json'), 'w'), indent=2)
    print(f"  results/mc_container_summary.json")


if __name__ == '__main__':
    main()

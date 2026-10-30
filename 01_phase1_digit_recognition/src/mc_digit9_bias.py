#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
数字 9（第 10 层，偶数层）: G_10..G_80 的取值与 G_1..G_8 的输出范围
====================================================================
偶数层: c_j = 15 - W_j
  像素未点亮 -> 截断 -> 输出 15
  像素点亮   -> 输出 c_j

支线 G_i (n 个像素):
  最小值 = 15n            （全部点亮）
  最大值 = sum_j c_j      （全部未点亮）

要求 G_i 在最小值时输出 0:
  G_i0 = 15n
  G_i 的范围 = [0, sum_j c_j - 15n]

总线 F = F_0 - sum(G_i),  F_0 = sum(G_i0) = 15 * 225 = 3375
  F 的范围（当全部未点亮时最大）:
     F_max = F_0 - sum(G_i_min) = 3375 - 3375 = 0   -- 全部点亮时
     F_min = F_0 - sum(G_i_max) = 3375 - sum(所有 c)  -- 全部未点亮时
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
    k = 9
    layer = k + 1
    parity = '奇数' if layer % 2 == 1 else '偶数'
    print("=" * 96)
    print(f"数字 {k}（第 {layer} 层，{parity}层）: 支线偏置与输出范围")
    print("=" * 96)
    print(f"公式: 偶数层 c = 15 - W，未点亮输出 15，点亮输出 c\n")

    print(f"  {'支线':>4} | {'行范围':>8} | {'n':>3} | {'权重和':>6} | "
          f"{'c 最小':>6} | {'c 最大':>6} | {'sum(c)':>7} | {'G_i0':>5} | "
          f"{'G 范围':>12} | {'跨度':>5}")
    print("  " + "-" * 96)
    G0 = []
    rows = []
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = [r * 15 + c for r in range(r0, r0 + nr) for c in range(15)]
        wg = W[pxs, k]
        cg = 15 - wg                                # 偶数层
        n = len(pxs)
        gmin_raw = 15 * n
        gmax_raw = int(cg.sum())
        g0 = gmin_raw                               # 使最小值输出 0
        G0.append(g0)
        lo, hi = 0, gmax_raw - g0
        print(f"  G_{gi+1:<2} | {r0:>3}-{r0+nr-1:<3} | {n:>3} | {wg.sum():>6} | "
              f"{cg.min():>6} | {cg.max():>6} | {gmax_raw:>7} | {g0:>5} | "
              f"[{lo:>4},{hi:>5}] | {hi-lo:>5}")
        rows.append({'G': gi + 1, 'row0': r0, 'rows': nr, 'n': n,
                     'w_sum': int(wg.sum()), 'c_min': int(cg.min()),
                     'c_max': int(cg.max()), 'sum_c': gmax_raw,
                     'G_i0': g0, 'G_min': lo, 'G_max': hi, 'span': hi - lo})
    F0 = sum(G0)
    print(f"\n  各支线偏置: G_10..G_80 = {G0}")
    print(f"  F_0 = sum(G_i0) = {F0}   (= 15 x 225)")

    # 总线范围
    sum_c_all = sum(r['sum_c'] for r in rows)
    print(f"\n[总线 F]")
    print(f"  F = F_0 - sum(G_i) = {F0} - sum(G_i)")
    print(f"  全部点亮时:  sum(G_i) = sum(15n) = {F0}  -> F = {F0 - F0}")
    print(f"  全部未点亮时: sum(G_i) = sum(c) = {sum_c_all}  -> F = {F0 - sum_c_all}")
    print(f"  -> F 范围 [{F0 - sum_c_all}, 0]   跨度 {sum_c_all - F0}")

    # 与网络要的 logit 对照
    print(f"\n[与网络的对应]")
    print(f"  网络的数字 {k} 偏置 (bias) = {bias[k]}")
    print(f"  网络的权重和 = {W[:, k].sum()}")
    print(f"  本架构里 F 的变化量 = 点亮像素的权重和（符号相反见下）")

    # 实际 G 的输出范围（奇偶对比）
    print(f"\n[奇偶两层对比]")
    print(f"  {'项':<20} | {'奇数层':>10} | {'偶数层':>10}")
    print("  " + "-" * 46)
    c_odd = 15 + W[:, k]; c_even = 15 - W[:, k]
    go0 = sum(15 * nr * 15 for _, nr in GROUPS)
    print(f"  {'容器范围':<20} | [{c_odd.min():>3},{c_odd.max():>3}]    | "
          f"[{c_even.min():>3},{c_even.max():>3}]")
    print(f"  {'F_0':<20} | {go0:>10} | {go0:>10}")
    print(f"  {'F 范围':<20} | [{go0-int(c_odd.sum())},{0}]    | "
          f"[{go0-int(c_even.sum())},{0}]")

    # 逐个情形验证 G_1
    print(f"\n[G_1 逐情形验证]（G_10 = {G0[0]}）")
    pxs = [14 * 15 + c for c in range(15)]
    w9 = W[pxs, k]; c9 = 15 - w9
    eye = np.eye(15, dtype=int)
    print(f"  {'情形':<18} | {'sum输出':>7} | {'G_1=G_10-sum':>12} | {'ΔG':>5} | "
          f"{'ΔF=W和':>7} | 校验")
    print("  " + "-" * 70)
    base = int(c9.sum())
    for nm, lit in [("全部未点亮", np.zeros(15, int)), ("全部点亮", np.ones(15, int)),
                    ("仅第2列 W=+3", eye[2]), ("仅第11列 W=-1", eye[11]),
                    ("仅第0列 W=0", eye[0])]:
        s = int(np.where(lit == 1, c9, 15).sum())
        g = G0[0] - s
        dg = g - (G0[0] - base)
        exp = int((lit * w9).sum())
        ok = (dg == exp)
        print(f"  {nm:<18} | {s:>7} | {g:>12} | {dg:>5} | {exp:>7} | "
              f"{'OK' if ok else '不符'}")
    print(f"  （偶数层: 点亮使 sum 减小 -> G 增大 -> F 减小，ΔG = +权重和）")

    json.dump({'digit': k, 'layer': layer, 'parity': parity,
               'G_i0': G0, 'F_0': F0,
               'branches': rows,
               'F_range': [F0 - sum_c_all, 0]},
              open(os.path.join(RES, 'mc_digit9_bias.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"\n已写出: results/mc_digit9_bias.json")


if __name__ == '__main__':
    main()

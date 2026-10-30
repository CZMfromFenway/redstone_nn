#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
数字 9 的 G_1：求解满足两个要求的 G_10 与 F 结构
==================================================
czm 给定的两个要求:
  (1) 比较器不能输出负值  -> G_i >= 0
  (2) 仅负权重像素点亮时, G_i 最大 (F 最小)
      仅正权重像素点亮时, G_i 最小 (F 最大)
      并要求这两个值分别是 11 和 0

偶数层: c_j = 15 - W_j
        像素亮起 -> 截断 -> 输出 15
        像素不亮 -> 输出容器值 c_j
        G_i = G_i0 - sum_j(输出_j)
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")


def main():
    z = np.load(os.path.join(RES, 'Wint_model.npz'))
    W = z['W'].astype(int); bias = z['b'].astype(int)
    k = 9
    row = 14
    pxs = [row * 15 + c for c in range(15)]
    w = W[pxs, k]
    c = 15 - w
    print("=" * 92)
    print("数字 9 的 G_1 详解（偶数层）")
    print("=" * 92)
    print(f"\n  权重 W : {w.tolist()}")
    print(f"  容器 c : {c.tolist()}")
    print(f"\n  正权重个数 = {(w>0).sum()}  负权重个数 = {(w<0).sum()}  "
          f"零 = {(w==0).sum()}")
    print(f"  正权重和 = {w[w>0].sum()}   负权重和 = {w[w<0].sum()}")
    print(f"  sum(c) = {c.sum()}   15n = {15*len(w)}")

    # 四个关键情形
    lit_pos = (w > 0).astype(int)
    lit_neg = (w < 0).astype(int)
    lit_all = np.ones(len(w), dtype=int)
    lit_non = np.zeros(len(w), dtype=int)

    def out_sum(lit):
        """sum_j(输出) = 亮起? 15 : c_j"""
        return int(np.where(lit == 1, 15, c).sum())

    print(f"\n  各情形的 sum(输出):")
    for nm, lit in [("仅正权重点亮", lit_pos), ("仅负权重点亮", lit_neg),
                    ("全部亮起", lit_all), ("全部不亮", lit_non)]:
        print(f"    {nm:<16} = {out_sum(lit)}")

    # 反解 G_i0: 要求 仅负权重 -> G=11, 仅正权重 -> G=0
    s_neg = out_sum(lit_neg)
    s_pos = out_sum(lit_pos)
    print(f"\n  由要求反解 G_10:")
    print(f"    仅负权重点亮 -> G=11  ==>  G_10 = 11 + {s_neg} = {11+s_neg}")
    print(f"    仅正权重点亮 -> G=0   ==>  G_10 =  0 + {s_pos} = {s_pos}")
    if 11 + s_neg != s_pos:
        print(f"    **两个要求矛盾！差 {abs((11+s_neg)-s_pos)} **")
        print(f"    -> 说明我对'哪些像素点亮'的理解与 czm 不同")
    else:
        print(f"    两者一致: G_10 = {s_pos}")

    # 检查 G 非负（遍历所有子集太大，用极值判断）
    print(f"\n  [G 非负检查] G = G_10 - sum(输出)")
    print(f"    sum(输出) 最大 = ?  逐像素: 亮起固定输出 15，不亮输出 c_j")
    print(f"      c_j 最小值 = {c.min()} (权重最大处 W={w.max()})")
    print(f"      c_j 最大值 = {c.max()} (权重最小处 W={w.min()})")
    print(f"    若要 G >= 0，需 G_10 >= max over all subsets of sum(输出)")
    print(f"      = 每个像素取 max(15, c_j) 之和 = {int(np.maximum(c,15).sum())}")
    print(f"    而 sum(输出) 最小 = 每个像素取 min(15, c_j) = "
          f"{int(np.minimum(c,15).sum())}")

    G10_min_for_nonneg = int(np.maximum(c, 15).sum())
    print(f"\n  ==> G 非负要求 G_10 >= {G10_min_for_nonneg}")
    print(f"      该值下 G 范围 = [0, {G10_min_for_nonneg - int(np.minimum(c,15).sum())}]")
    G10 = G10_min_for_nonneg
    print(f"\n  取 G_10 = {G10}，各情形:")
    print(f"    {'情形':<16} | {'sum输出':>7} | {'G_1':>5}")
    print("    " + "-" * 34)
    for nm, lit in [("仅正权重点亮", lit_pos), ("仅负权重点亮", lit_neg),
                    ("全部亮起", lit_all), ("全部不亮", lit_non)]:
        print(f"    {nm:<16} | {out_sum(lit):>7} | {G10-out_sum(lit):>5}")
    print(f"\n    G_1 范围 = [0, {G10 - int(np.minimum(c,15).sum())}]")
    print(f"    G_1 最大值 = sum(W_负) 的绝对值 = {abs(w[w<0].sum())}  "
          f"(实际 {G10 - int(np.minimum(c,15).sum())})")

    # 结论
    print(f"\n{'='*92}")
    print("### 结论")
    print(f"  G_10 = {G10}")
    print(f"  G_1 范围 = [0, {G10 - int(np.minimum(c,15).sum())}]")
    print(f"  仅负权重点亮时 G_1 = {G10 - out_sum(lit_neg)}")
    print(f"  仅正权重点亮时 G_1 = {G10 - out_sum(lit_pos)}")
    print(f"  F 的非负: F_0 = sum_i(G_i^max)")

    json.dump({'digit': k, 'G_10': G10,
               'G_1_range': [0, G10 - int(np.minimum(c, 15).sum())],
               'weights': [int(x) for x in w],
               'containers': [int(x) for x in c]},
              open(os.path.join(RES, 'mc_digit9_G1_solve.json'), 'w'), indent=2)
    print(f"\n已写出: results/mc_digit9_G1_solve.json")


if __name__ == '__main__':
    main()

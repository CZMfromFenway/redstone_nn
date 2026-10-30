#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
数字 9 的 G_1 模拟（v3：符号与偏移都修正）
=============================================
前两版的错误:
  v1: c = 15W + 15        -> 符号反了
  v2: c = offset+15(1-W)  -> 偏移量加进容器值，破坏了 W=0 的情形

正确形式:
  容器值: c = 15(1 - W)          （不含偏移）
  W=0 -> c=15（与 czm 的例子一致：截断 15 时 ΔG=0）
  W=1 -> c=0    (ΔG=+15, ΔF=-15)
  W=-1 -> c=30  (ΔG=-15, ΔF=+15)

  若 c 越界（W>1 时为负），**把偏移加在支线偏置 G_i0 上**，不加进容器:
      G_i = G_i0 - sum(c)
    这样 W 的作用仍由 Δc = -15W 决定，与 G_i0 无关。

容器合法范围 [0,897] -> 15(1-W) ∈ [0,897] -> W ∈ [-58.8, +1]
  对 W ∈ [-12, +7]: 负权重没问题；正权重需靠 G_i0 承接
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")


def main():
    print("=" * 90)
    print("数字 9 的 G_1 模拟（v3）")
    print("=" * 90)
    z = np.load(os.path.join(RES, 'Wint_model.npz'))
    W = z['W'].astype(int); bias = z['b'].astype(int)
    k = 9; row = 14
    idx = [row * 15 + c for c in range(15)]
    w9 = W[idx, k]
    print(f"权重全体范围 [{W.min()}, {W.max()}]   数字 9 偏置 = {bias[k]}")
    print(f"第 {row} 行对数字 9 的权重: {w9.tolist()}")

    c9 = 15 * (1 - w9)                     # 容器值（不含偏移）
    print(f"\n[容器值] c = 15(1-W)   范围 [{c9.min()}, {c9.max()}]")
    print(f"  需要容器在 [0,897]。当前 c 为负的个数: {(c9<0).sum()}")
    print(f"  -> 支线偏置 G_10 = {-c9.min()} 可保证 G_1 = G_10 - sum(c) >= 0")

    # ---------- 1. 逐像素表 ----------
    print(f"\n[1] 第 {row} 行 15 个像素逐项")
    print(f"  {'列':>3} | {'像素':>5} | {'W':>4} | {'容器 c':>7} | {'未点亮输出':>10} | "
          f"{'点亮输出':>9} | {'ΔG':>5} | {'ΔF':>5}")
    print("  " + "-" * 72)
    for i, p in enumerate(idx):
        w = int(w9[i]); c = int(c9[i])
        dg = 15 - c
        print(f"  {i:>3} | {p:>5} | {w:>4} | {c:>7} | {c:>10} | {15:>9} | "
              f"{dg:>5} | {-dg:>5}")
    print(f"\n  校验: ΔG 应等于 -15W")
    for i in range(15):
        dg = 15 - int(c9[i]); exp = -15 * int(w9[i])
        assert dg == -exp, (i, dg, exp)
    print(f"  全部 15 个像素: ΔG = -15W  ✓")

    # ---------- 2. 情形模拟 ----------
    print(f"\n[2] G_1 输出模拟（未含支线偏置）")
    print(f"  {'情形':<20} | {'G_1':>6} | {'ΔG':>6} | {'ΔG/15':>7} | {'-权重和':>8} | 校验")
    print("  " + "-" * 70)
    eye = np.eye(15, dtype=int)
    cases = [("全部未点亮", np.zeros(15, int)), ("全部点亮", np.ones(15, int)),
             ("仅第0列 W=0", eye[0]), ("仅第2列 W=+3", eye[2]),
             ("仅第11列 W=-1", eye[11]), ("仅第3列 W=+1", eye[3]),
             ("奇数列点亮", np.array([1, 0] * 7 + [1])),
             ("偶数列点亮", np.array([0, 1] * 7 + [0]))]
    g_off = int(c9.sum())
    allok = True
    for nm, lit in cases:
        g = int(np.where(lit == 1, 15, c9).sum())
        dg = g - g_off
        exp = -int((lit * w9).sum())
        ok = (dg == 15 * exp)
        allok &= ok
        print(f"  {nm:<20} | {g:>6} | {dg:>6} | {dg/15:>7.2f} | {exp:>8} | "
              f"{'OK' if ok else '不符'}")
    print(f"\n  全部通过: {'是' if allok else '否'}")
    print(f"  （校验收敛: ΔG/15 = -权重和）")
    print(f"  例: 仅第2列(W=+3)点亮 -> ΔG = {15 - int(c9[2])}，ΔF = +45 = 15*3 ✓")

    # ---------- 3. 数字 9 全部 225 容器 ----------
    print(f"\n[3] 数字 9 的 8 条支线（全部 225 个容器）")
    print(f"  {'支线':>4} | {'行':>7} | {'n':>3} | {'权重和':>6} | {'容器范围':>12} | "
          f"{'越界':>4} | {'需要的 G_i0':>10}")
    print("  " + "-" * 76)
    GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]
    tot_bad = 0
    g0s = []
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = [r * 15 + c for r in range(r0, r0 + nr) for c in range(15)]
        wg = W[pxs, k]
        cg = 15 * (1 - wg)
        bad = int((cg < 0).sum() + (cg > 897).sum())
        tot_bad += bad
        g0 = int(max(0, -cg.min()))        # 使 G_i = G_i0 - sum(c) >= 0
        g0s.append(g0)
        print(f"  G_{gi+1:<2} | {r0:>3}-{r0+nr-1:<3} | {nr*15:>3} | {wg.sum():>6} | "
              f"[{cg.min():>4},{cg.max():>4}] | {bad:>4} | {g0:>10}")
    print(f"\n  越界总数: {tot_bad}")
    print(f"  各支线偏置 G_10..G_80 = {g0s}")
    print(f"  F_0 = sum(G_i0) = {sum(g0s)}")

    # ---------- 4. 全局 ----------
    print(f"\n[4] 全局（10 层都算）")
    allc = 15 * (1 - W)
    print(f"  全部 2250 个容器的值范围 [{allc.min()}, {allc.max()}]")
    print(f"  负值个数 {(allc<0).sum()}  (>897 的个数 {(allc>897).sum()})")
    print(f"  需要支线偏置: 每层各 8 个，由该支线内最负的容器决定")
    print(f"  全局最大权重 {W.max()} -> 全局最负容器 {allc.min()}")

    # 完整 10 层的支线偏置
    print(f"\n[5] 10 层 x 8 支线 的 G_i0（= 使该支线非负的最小值）")
    print(f"  {'数字':>4} | " + " | ".join(f"G{i+1}0" for i in range(8)) + " |  F_0")
    print("  " + "-" * 78)
    for kk in range(10):
        gs = []
        for (r0, nr) in GROUPS:
            pxs = [r * 15 + c for r in range(r0, r0 + nr) for c in range(15)]
            cg = 15 * (1 - W[pxs, kk])
            gs.append(int(max(0, -cg.min())))
        print(f"  {kk:>4} | " + " | ".join(f"{g:>5}" for g in gs) + f" | {sum(gs):>6}")

    json.dump({'digit': k, 'weights': [int(x) for x in w9],
               'containers': [int(x) for x in c9],
               'G10_needed': int(max(0, -c9.min())),
               'global_container_range': [int(allc.min()), int(allc.max())]},
              open(os.path.join(RES, 'mc_g1_digit9_v3.json'), 'w'), indent=2)
    print(f"\n已写出: results/mc_g1_digit9_v3.json")


if __name__ == '__main__':
    main()

#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
数字 9 的 G_1 模拟：最底层 15 个像素
======================================
澄清后的算法:
  * 容器值直接由权重算出，不引入"基础值 b"
      奇数层: c_j = 15 * W_j          （像素未点亮 -> 输出 c_j；点亮 -> 输出 15）
      偶数层: c_j = 15 * W_j          （镜像截断方向）
    权重与容器值的换算: W = (c_unlit - 15) / 15
  * 容器值必须落在 [0, 897]
  * 支队偏置 G_10 = sum(15*W_j_negative)，使 G_1 在整个区间内非负

本次只用 G_1（最后一行 15 个像素），数字 9
先给出该行 15 个像素对数字 9 的权重，再给容器值、逐个模拟
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")


def main():
    print("=" * 92)
    print("数字 9 的 G_1 模拟（最底层 15 个像素，第 14 行）")
    print("=" * 92)
    z = np.load(os.path.join(RES, 'Wint_model.npz'))
    W = z['W'].astype(int); b = z['b'].astype(int)
    print(f"载入整数权重: 范围 [{W.min()},{W.max()}]  形状 {W.shape}")

    k = 9                       # 数字 9
    row = 14                    # 最底一行
    idx = [row * 15 + c for c in range(15)]
    w9 = W[idx, k]
    print(f"\n数字 {k} 的偏置 (bias) = {b[k]}")

    # ---------- 1. 权重 ----------
    print(f"\n[1] 第 {row} 行 15 个像素对数字 9 的权重")
    print(f"  {'列':>3} | {'像素索引':>8} | {'权重 W':>7}")
    print("  " + "-" * 28)
    for c, p in enumerate(idx):
        print(f"  {c:>3} | {p:>8} | {w9[c]:>7}")
    print(f"\n  权重和 = {w9.sum()}   最小 {w9.min()}   最大 {w9.max()}")
    print(f"  非零个数 = {(w9 != 0).sum()} / 15")

    # ---------- 2. 容器值 ----------
    print(f"\n[2] 容器值（两种截断方向）")
    print(f"  换算: 未点亮时输出 c，点亮时输出 15；ΔG = 15*W  ==>  W = (c_lit15 - 15)/15")
    print(f"  取 c = 15*W + 15，则: 奇数层未点亮输出 c，点亮输出 15 (Δ = 15-15W)")
    print()
    print(f"  {'列':>3} | {'W':>4} | {'c 奇数层':>9} | {'c 偶数层':>9} | {'容器合法?':>9}")
    print("  " + "-" * 52)
    c_odd = 15 * w9 + 15          # 奇数层: 未点亮输出 c
    c_even = 15 - 15 * w9         # 偶数层: 未点亮输出 15, 点亮输出 c
    for c in range(15):
        ok = 0 <= c_odd[c] <= 897 and 0 <= c_even[c] <= 897
        print(f"  {c:>3} | {w9[c]:>4} | {c_odd[c]:>9} | {c_even[c]:>9} | "
              f"{'OK' if ok else '**超限**':>9}")
    print(f"\n  奇数层容器范围 [{c_odd.min()}, {c_odd.max()}]")
    print(f"  偶数层容器范围 [{c_even.min()}, {c_even.max()}]")
    print(f"  越界个数: 奇 {(c_odd<0).sum()+(c_odd>897).sum()}  "
          f"偶 {(c_even<0).sum()+(c_even>897).sum()}")

    # ---------- 3. 逐个模拟 ----------
    print(f"\n[3] G_1 输出模拟（奇数层）")
    print(f"  规则: 像素点亮 -> 该容器输出 15；未点亮 -> 输出容器值 c")
    print(f"  G_1 = sum(每个容器的输出)   （支队偏置 G_10 = 0，因为 c 已保证非负）")
    base = 15 * 15                                  # 全部点亮时的输出
    prints = [
        ("全部未点亮", np.zeros(15, dtype=int)),
        ("全部点亮", np.ones(15, dtype=int)),
        ("仅第 0 列", np.eye(15, dtype=int)[0]),
        ("仅第 7 列(中心)", np.eye(15, dtype=int)[7]),
        ("奇数列点亮", np.array([1, 0] * 7 + [1])),
        ("偶数列点亮", np.array([0, 1] * 7 + [0])),
    ]
    print(f"\n  {'情形':<16} | {'G_1 输出':>9} | {'ΔG(相对全未亮)':>13} | "
          f"{'等效权重和':>10}")
    print("  " + "-" * 60)
    g_alloff = int(c_odd.sum())
    for nm, lit in prints:
        g = int(np.where(lit == 1, 15, c_odd).sum())
        dg = g - g_alloff
        print(f"  {nm:<16} | {g:>9} | {dg:>13} | {dg/15:>10.2f}")

    print(f"\n  校验: ΔG / 15 应该等于'点亮像素的权重之和'")
    for nm, lit in prints:
        g = int(np.where(lit == 1, 15, c_odd).sum())
        dg = g - g_alloff
        exp = int((lit * w9).sum())
        flag = "OK" if abs(dg / 15 - exp) < 1e-9 else "**不符**"
        print(f"    {nm:<16} ΔG/15 = {dg/15:>6.2f}   权重和 = {exp:>4}   {flag}")

    # ---------- 4. 全 225 个像素的容器值总表（数字 9） ----------
    print(f"\n[4] 数字 9 全部 225 个容器的信号值")
    print(f"  {'支线':>4} | {'行范围':>7} | {'n':>3} | {'权重和':>7} | "
          f"{'奇数层范围':>13} | {'偶数层范围':>13} | 越界")
    print("  " + "-" * 76)
    GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]
    allok = True
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = [r * 15 + c for r in range(r0, r0 + nr) for c in range(15)]
        wg = W[pxs, k]
        co = 15 * wg + 15
        ce = 15 - 15 * wg
        bad = int((co < 0).sum() + (co > 897).sum() + (ce < 0).sum() + (ce > 897).sum())
        allok &= (bad == 0)
        print(f"  G_{gi+1:<2} | {r0:>3}-{r0+nr-1:<3} | {nr*15:>3} | {wg.sum():>7} | "
              f"[{co.min():>4},{co.max():>4}]   | [{ce.min():>4},{ce.max():>4}]   | {bad}")
    print(f"\n  全部 225 个容器都合法: {'是' if allok else '否'}")

    # ---------- 5. G_1 的偏置需求 ----------
    print(f"\n[5] G_1 支队偏置 G_10 的需求")
    print(f"  奇数层: G_1 = sum(未点亮? c : 15)")
    print(f"    全部未点亮时 G_1 = {int(c_odd.sum())}  （这是最大值）")
    print(f"    全部点亮时   G_1 = {base}")
    print(f"  -> 若希望 G_1 在'全部点亮'时为 0，需要支队偏置 G_10 = {base}")
    print(f"     （因为 G_1 = G_10 - sum(c)，全亮时 sum(c)=0）")
    print(f"  若不做偏置（直接让 G_1 = sum(输出)），G_1 始终 >= 0，也能工作")

    json.dump({'digit': k, 'row': row,
               'weights': [int(x) for x in w9],
               'c_odd': [int(x) for x in c_odd],
               'c_even': [int(x) for x in c_even],
               'bias': int(b[k])},
              open(os.path.join(RES, 'mc_g1_digit9.json'), 'w'), indent=2)
    print(f"\n已写出: results/mc_g1_digit9.json")


if __name__ == '__main__':
    main()

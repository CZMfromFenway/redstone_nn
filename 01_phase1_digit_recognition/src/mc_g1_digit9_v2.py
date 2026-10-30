#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
数字 9 的 G_1 模拟（修正符号）
================================
上一版错误: 写成 c = 15W + 15，符号反了，导致校验失败。

正确关系（由"截断使 G 变小 -> 使 F 变大"推出）:
  像素点亮时输出 15，所以 ΔG = 15 - c
  又 ΔF = -15W 且 ΔF = -ΔG
  ==> 15 - c = 15W  ==>  c = 15(1 - W)

  验证: c=0 -> W=+1 ; c=30 -> W=-1   （与 czm 的例子一致）

容器必须落在 [0,897]:
  W = +1  -> c = 0     （最小）
  W = -11 -> c = 180   （最大）
  所以 W 越大，容器值越小；W 每 +1，容器值 -15。
  当 W 超过 1 时容器值会变负 -> 需要整体抬高 (偏移量 = 15*(Wmax-1))
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")


def main():
    print("=" * 92)
    print("数字 9 的 G_1 模拟（修正符号）")
    print("=" * 92)
    z = np.load(os.path.join(RES, 'Wint_model.npz'))
    W = z['W'].astype(int); bias = z['b'].astype(int)
    k = 9; row = 14
    idx = [row * 15 + c for c in range(15)]
    w9 = W[idx, k]
    Wmax_all = int(W.max()); Wmin_all = int(W.min())
    print(f"权重: 全体范围 [{Wmin_all}, {Wmax_all}]   数字 9 偏置 = {bias[k]}")
    print(f"第 {row} 行对数字 9 的权重范围 [{w9.min()}, {w9.max()}]")

    # 全局需要的偏移（保证所有容器的 c 非负）
    offset = 15 * max(0, Wmax_all - 1)
    print(f"\n[偏移量] 因为全局最大权重 = {Wmax_all}，需要整体抬高")
    print(f"  offset = 15 * ({Wmax_all} - 1) = {offset}")
    print(f"  容器值公式: c = offset + 15*(1 - W)")

    def c_of(w):
        return offset + 15 * (1 - w)

    # ---------- 1. 权重 ----------
    print(f"\n[1] 第 {row} 行 15 个像素对数字 9 的权重")
    print(f"  {'列':>3} | {'像素':>5} | {'W':>4} | {'容器值 c':>9} | {'点亮时输出':>9} | "
          f"{'ΔG=15-c':>9} | {'ΔF=-ΔG':>8}")
    print("  " + "-" * 74)
    tot_unlit, tot_lit = 0, 0
    for c_i, p in enumerate(idx):
        w = int(w9[c_i]); cv = int(c_of(w))
        dg = 15 - cv; df = -dg
        tot_unlit += cv; tot_lit += 15
        print(f"  {c_i:>3} | {p:>5} | {w:>4} | {cv:>9} | {15:>9} | {dg:>9} | {df:>8}")
    print(f"\n  全部未点亮: G_1 = {tot_unlit}")
    print(f"  全部点亮:   G_1 = {tot_lit}")
    print(f"  差值 = {tot_lit - tot_unlit}，除以 15 = {(tot_lit-tot_unlit)/15:.2f}")
    print(f"  权重和 = {w9.sum()}  （应该相等）")

    # ---------- 2. 逐情形模拟 ----------
    print(f"\n[2] G_1 输出模拟（奇数层）")
    print(f"  {'情形':<18} | {'G_1':>7} | {'ΔG':>7} | {'ΔG/15':>7} | {'权重和':>7} | 校验")
    print("  " + "-" * 68)
    cases = [
        ("全部未点亮", np.zeros(15, int)),
        ("全部点亮", np.ones(15, int)),
        ("仅第0列", np.eye(15, int)[0]),
        ("仅第7列(中心)", np.eye(15, int)[7]),
        ("仅第11列(W=-1)", np.eye(15, int)[11]),
        ("仅第2列(W=+3)", np.eye(15, int)[2]),
        ("奇数列点亮", np.array([1, 0] * 7 + [1])),
        ("偶数列点亮", np.array([0, 1] * 7 + [0])),
    ]
    allok = True
    for nm, lit in cases:
        g = int(np.where(lit == 1, 15, [int(c_of(w)) for w in w9]).sum())
        dg = g - tot_unlit
        exp = int((lit * w9).sum())
        ok = abs(dg / 15 - (-exp)) < 1e-9     # ΔG/15 = -权重和
        allok &= ok
        print(f"  {nm:<18} | {g:>7} | {dg:>7} | {dg/15:>7.2f} | {exp:>7} | "
              f"{'OK' if ok else '不符'}")
    print(f"\n  注: ΔG/15 = -(点亮像素权重和)，因为 ΔF = -ΔG = 15*权重和")
    print(f"  全部情形校验通过: {'是' if allok else '否'}")

    # ---------- 3. 全部 225 容器 ----------
    print(f"\n[3] 数字 9 全部 225 个容器的信号值（offset = {offset}）")
    print(f"  {'支线':>4} | {'行':>7} | {'n':>3} | 权重和 | {'容器范围':>13} | "
          f"{'全未亮 G':>9} | {'全亮 G':>7} | 越界")
    print("  " + "-" * 80)
    GROUPS = [(14, 1)] + [(r, 2) for r in range(12, -1, -2)]
    tot_bad = 0
    for gi, (r0, nr) in enumerate(GROUPS):
        pxs = [r * 15 + c for r in range(r0, r0 + nr) for c in range(15)]
        wg = W[pxs, k]
        cg = c_of(wg)
        bad = int((cg < 0).sum() + (cg > 897).sum())
        tot_bad += bad
        print(f"  G_{gi+1:<2} | {r0:>3}-{r0+nr-1:<3} | {nr*15:>3} | {wg.sum():>6} | "
              f"[{cg.min():>4},{cg.max():>4}] | {int(cg.sum()):>9} | {nr*15*15:>7} | {bad}")
    print(f"\n  全部 225 个容器越界总数: {tot_bad}  "
          f"({'全部合法' if tot_bad == 0 else '**有越界**'})")

    # ---------- 4. 总线偏置 ----------
    print(f"\n[4] 偏置需求")
    print(f"  若要求 F 在'全部点亮'时等于数字 9 的 logit 尺度，需要:")
    print(f"    F_0 = sum(G_i0),  G_i0 使 G_i 在全亮时为 0")
    g0 = []
    for gi, (r0, nr) in enumerate(GROUPS):
        n = nr * 15
        g0.append(n * 15)          # 全亮时容器都输出 15
    print(f"    各支线 G_i0 = n*15 = {g0}")
    print(f"    F_0 = sum = {sum(g0)}")
    print(f"  当前容器余量: 最大容器值 {int(c_of(Wmin_all))} / 897")

    json.dump({'digit': k, 'row': row, 'offset': offset,
               'weights': [int(x) for x in w9],
               'containers': [int(c_of(w)) for w in w9],
               'G1_all_unlit': int(tot_unlit), 'G1_all_lit': int(tot_lit),
               'bias_model': int(bias[k]),
               'max_container_global': int(c_of(Wmin_all))},
              open(os.path.join(RES, 'mc_g1_digit9_v2.json'), 'w'), indent=2)
    print(f"\n已写出: results/mc_g1_digit9_v2.json")


if __name__ == '__main__':
    main()

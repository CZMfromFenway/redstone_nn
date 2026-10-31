# -*- coding: utf-8 -*-
"""对照：知识库《强模 LUT：从输入量化到非负累加量 N》 vs 一期 LUT 机制脚本。

一期脚本（archive/src/strong_lut_math_final.py、strong_lut_two_truncations.py）
是"容器 b → 步长"的实现级验证；本脚本按知识库正文的公式独立重算，
逐项比较两者的数值是否一致，并统计差的最大值。
"""
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), '_crosscheck_01.txt')
lines = []


# ---------- 一期实现：16 容器链 ----------
def a_comp(k, x):
    return 14 + k - x


def a_dust(k, x):
    return min(15, 14 + k - x)


def S(x, b, mode):
    f = a_comp if mode == 'comp' else a_dust
    tot = 0
    for k in range(16):
        a = f(k, x)
        tot += a if a >= 15 else int(b[k])
    return tot


# ---------- 知识库公式 ----------
def step_kb(mode, x, V):
    """KB §三：V = 15 − ΔΛ（红石粉） / (29−x) − ΔΛ（比较器）。解出 ΔΛ。"""
    return (15 - V) if mode == 'dust' else ((29 - x) - V)


def main():
    rng = np.random.default_rng(20261031)

    # ---- 1. 步长公式：实现 vs KB 公式 ----
    lines.append('### 1. 步长公式  ΔΛ = 15 − V（红石粉）/ (29−x) − V（比较器）')
    for mode in ('dust', 'comp'):
        bad = 0
        tot = 0
        seen = set()
        for _ in range(4000):
            b = rng.integers(0, 898, size=16)
            for x in range(15):
                d = S(x, b, mode) - S(x + 1, b, mode)      # 顺式 ΔΛ
                pred = step_kb(mode, x, int(b[x + 1]))
                seen.add(int(d))
                tot += 1
                if d != pred:
                    bad += 1
        lines.append(f'  {mode:5s}: 不匹配 {bad}/{tot}   '
                     f'实测 ΔΛ 范围 [{min(seen)}, {max(seen)}]')

    # ---- 2. 窗口边界 ----
    lines.append('')
    lines.append('### 2. 由 V∈[0,897] 推出的窗口')
    for mode in ('dust', 'comp'):
        lo = min(step_kb(mode, x, 897) for x in range(15))
        hi = max(step_kb(mode, x, 0) for x in range(15))
        # 注意：comp 的负端随 x 变，取"最紧"的那个（x=0）
        lo_tight = max(step_kb(mode, x, 897) for x in range(15))
        lines.append(f'  {mode:5s}: ΔΛ ∈ [{lo}, {hi}]   最紧负端 [{lo_tight}]')
    lines.append('  交集（两种都能实现）= [−868, +15]；'
                 '故 |ΔΛ| ≤ 868，与 KB 的 868 一致')

    # ---- 3. 868 是真正的交集边界 ----
    lines.append('')
    lines.append('### 3. 交集边界检验：−868 两侧')
    for target in (-867, -868, -869, -870):
        ok_dust = all(0 <= 15 - target <= 897 for _ in [0])
        ok_comp = all(0 <= 29 - x - target <= 897 for x in range(15))
        lines.append(f'  ΔΛ={target:5d}: 红石粉可实现={ok_dust}  比较器可实现={ok_comp}')

    # ---- 4. ρ 与非负累加上界 ----
    lines.append('')
    lines.append('### 4. 斜率 ρ 与 N 的上界')
    Lmax = 868
    for w in ([0.8, -0.2], [0.6, -0.3], [0.5, 0.5, -0.1]):
        w = np.array(w)
        rho = Lmax * np.abs(w) / np.abs(w).max()
        nmax_round = int(sum(np.rint(15 * rho)))
        nmax_naive = 15 * rho.sum()
        lines.append(f'  w={w.tolist()}  ρ={np.round(rho,3).tolist()}  '
                     f'N_max(取整后)={nmax_round}  Σ15ρ={nmax_naive:.3f}')

    # ---- 5. 半字节分解恒等式 ----
    lines.append('')
    lines.append('### 5. 半字节分解  ρ^(l)=16^(l-(L-1))ρ^(L-1)，Σ ρ^(l)ξ^(l) = ρ^(0)·X')
    rho_top = 868.0
    for L in (1, 2, 3):
        rho = np.array([rho_top * 16 ** (l - (L - 1)) for l in range(L)])
        worst = 0.0
        for _ in range(2000):
            xi = rng.integers(0, 16, size=L)
            X = int(sum(xi[l] * 16 ** l for l in range(L)))
            lhs = float(sum(rho[l] * xi[l] for l in range(L)))
            rhs = rho[0] * X
            worst = max(worst, abs(lhs - rhs))
        lines.append(f'  L={L}  ρ={np.round(rho,4).tolist()}  '
                     f'max|Σρξ − ρ⁰X|={worst:.3g}')

    open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
    print('done')


if __name__ == '__main__':
    main()

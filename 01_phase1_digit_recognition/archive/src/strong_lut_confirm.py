#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
确认 czm 提出的三点
====================
1) 红石粉截断: 非自由区每个容器把 c 锁成 15;  比较器截断: 非自由区的 c 是等差数列,
   其和随自由区扩大而非线性(平方)变化; 红石粉截断则是线性的。
2) 通过顺式/反式的常数 B、B~ 可以把"死去和"的影响从 lambda 中消掉。
3) max x 在本项目里是固定的(16 个取值, 最大 15), 不是输入条数;
   单个 lambda 的输出极限 = 15 * 897;  Lambda 的上限 = 该值 * 输入条数。
"""
import numpy as np, json, os

def a_comp(k, x):
    return 14 + k - x

def a_dust(k, x):
    return min(15, 14 + k - x)

def c_of(b, a):
    return a if a >= 15 else int(b)

def parts(x, b, mode):
    """返回 (自由区之和, 非自由区之和)"""
    f = a_comp if mode == "comp" else a_dust
    free = sum(int(b[k]) for k in range(16) if f(k, x) < 15)
    dead = sum(c_of(b[k], f(k, x)) for k in range(16) if f(k, x) >= 15)
    return free, dead

def main():
    out = {}
    print("=" * 94)
    print("确认 czm 提出的三点")
    print("=" * 94)

    # ---------- 1. 非自由区之和的形状 ----------
    print("\n[1] 非自由区之和  Dead(x) 随 x 的变化 (取 b=0, 即最小情形)")
    b0 = np.zeros(16, dtype=np.int64)
    print(f"   {'x':>3} | {'comp Dead(x)':>14} | {'dust Dead(x)':>14} | 一阶差分(comp) | 一阶差分(dust)")
    prev = {"comp": None, "dust": None}
    rows = []
    for x in range(16):
        dc = parts(x, b0, "comp")[1]
        dd = parts(x, b0, "dust")[1]
        d1c = "" if prev["comp"] is None else dc - prev["comp"]
        d1d = "" if prev["dust"] is None else dd - prev["dust"]
        print(f"   {x:3d} | {dc:14d} | {dd:14d} | {str(d1c):>14} | {str(d1d):>14}")
        rows.append((x, dc, dd))
        prev = {"comp": dc, "dust": dd}
        out[f"dead_{x}"] = {"comp": dc, "dust": dd}

    dc = np.array([r[1] for r in rows]); dd = np.array([r[2] for r in rows])
    print(f"\n   comp Dead 是二次的? 二阶差分 = {np.diff(dc, 2).tolist()}")
    print(f"   dust Dead 是线性的? 二阶差分 = {np.diff(dd, 2).tolist()}")
    print(f"   comp Dead(0) = {dc[0]}  (=14+13+...+0 = 105, 再加 b 的贡献)")
    print(f"   dust Dead(0) = {dd[0]}  (=15*15 = 225)")

    # ---------- 2. 常数 B 能否消掉死去和 ----------
    print("\n[2] lambda = B - (自由和 + 死去和);  死去和只依赖 x, 与 b 无关时能否消掉")
    rng = np.random.default_rng(7)
    print("   验证: 对固定 x, Dead(x) 是否只依赖 x (即 b 的变化是否只影响自由区)")
    for mode in ("comp", "dust"):
        f = a_comp if mode == "comp" else a_dust
        ok = True; ex = None
        for _ in range(2000):
            b = rng.integers(0, 898, size=16)
            for x in range(16):
                # 死去和按定义只统计 a>=15 的项, 而这些项 c=a, 与 b 无关
                d = sum(c_of(b[k], f(k, x)) for k in range(16) if f(k, x) >= 15)
                d_ref = sum(f(k, x) for k in range(16) if f(k, x) >= 15)
                if d != d_ref:
                    ok = False; ex = (x, d, d_ref)
        print(f"   {mode:5s}: 死去和与 b 无关 = {ok}   例: {ex}")
        out[f"dead_indep_of_b_{mode}"] = bool(ok)

    print("\n   => 死去和 Dead(x) 是 x 的确定函数, 可以并入 B(x) 或写成 B - Dead(x)。")
    print("      注意 B 在原文中是常数(可超 897), 而 Dead(x) 随 x 变化;")
    print("      两者相加的净效果是给 lambda 叠加一个只依赖 x 的已知函数,")
    print("      它被下游激活 f_j 吸收(对所有 i 相同)。")

    # ---------- 3. 极限值 ----------
    print("\n[3] 极限值")
    print(f"   max x 固定 = 15 (0..15 共 16 个取值)")
    # 单列 lambda 的最大变化幅度
    for mode in ("comp", "dust"):
        f = a_comp if mode == "comp" else a_dust
        # lambda = B - S, 变化幅度 = max S - min S
        Smax = sum(f(k, 0) for k in range(16))          # x=0 时 S 最大
        Smin = sum(f(k, 15) for k in range(16))         # x=15 时 S 最小
        print(f"   {mode:5s}: S(0)={Smax}, S(15)={Smin}, 幅度={Smax-Smin}")
    print(f"   理论: 单列 15 步, 每步幅度最大 15*897 = {15*897}")
    print(f"         => 单列 lambda 变化幅度上限 = {15*15*897} = 15*15*897")
    for nin in (1, 4, 15):
        print(f"   n={nin:2d} 条输入线: Lambda 变化幅度上限 = {nin}*{15*15*897} = {nin*15*15*897}")
        out[f"lambda_max_n{nin}"] = nin * 15 * 15 * 897

    # 数值验证: 单列幅度
    print("\n   数值验证单列幅度 (随机 b, 取极端组合)")
    for mode in ("comp", "dust"):
        f = a_comp if mode == "comp" else a_dust
        bmax = np.full(16, 897, dtype=np.int64)
        bmin = np.zeros(16, dtype=np.int64)
        S_of = lambda x, b: sum(c_of(b[k], f(k, x)) for k in range(16))
        print(f"   {mode:5s}: b=0 -> S(0)={S_of(0,bmin)}, S(15)={S_of(15,bmin)}")
        print(f"   {mode:5s}: b=897 -> S(0)={S_of(0,bmax)}, S(15)={S_of(15,bmax)}")

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    p = os.path.join(res, "confirm_three_points.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已写出:", p)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强模 LUT 单元数学解析 —— 最终定稿
==================================
判据 (原文, czm 更正):
    c_k(x) = a_k(x)             若 a_k(x) >= 15
    c_k(x) = min(b_k, a_k(x))   若 a_k(x) <  15
选择器: a_k(x) = 14 + k - x
顺式:   lambda(x) = B - sum_k c_k(x)

核心结果 (待验证): lambda(x+1) - lambda(x) = 15 - b_{x+1}
   即: 容器 b_{x+1} 直接控制输出阶梯在 x 处的步长, 一一对应。
"""
import numpy as np, json, os

def a_of(k, x):
    return 14 + k - x

def c_of(b, a):
    return a if a >= 15 else min(int(b), a)

def S(x, b):
    return sum(c_of(b[k], a_of(k, x)) for k in range(16))

def lam(x, b, B=0):
    return B - S(x, b)

def main():
    out = {}
    rng = np.random.default_rng(20261003)
    print("=" * 90)
    print("强模 LUT 单元数学解析 (最终)")
    print("=" * 90)

    # ---------- 主命题 ----------
    print("\n[主命题]  lambda(x+1) - lambda(x) = 15 - b_{x+1}   (0 <= b <= 14)")
    bad = 0; got = set()
    for _ in range(30000):
        b = rng.integers(0, 15, size=16)
        for x in range(15):
            d = lam(x + 1, b) - lam(x, b)
            pred = 15 - int(b[x + 1])
            got.add(d)
            if d != pred:
                bad += 1
    print(f"   不匹配 = {bad}/450000")
    print(f"   实测步长集合 = {sorted(got)}")
    out["step_bad"] = bad
    out["step_values"] = sorted(got)

    # ---------- b_{x+1} 与步长一一对应 ----------
    print("\n[对应表]  容器值 -> 步长")
    tab = {}
    for v in range(0, 15):
        b = np.zeros(16, dtype=int); b[1] = v
        d = lam(1, b) - lam(0, b)
        tab[int(v)] = int(d)
        print(f"   b= {v:2d}  ->  lambda 步长 = {d:3d}")
    out["value_to_step"] = tab

    # ---------- 构造性: 任意目标阶梯 ----------
    print("\n[构造性]  任意目标阶梯 lambda, 步长 u_1..u_15 ∈ [1,15] 是否可实现")
    tests = {
        "恒定步长 1":   [0] + list(range(1, 16)),
        "恒定步长 15":  [0] + [15 * i for i in range(1, 16)],
        "随机步长":     None,
        "前陡后缓":     None,
    }
    for nm, t in tests.items():
        if nm == "随机步长":
            u = rng.integers(1, 16, size=15); t = np.concatenate([[0], np.cumsum(u)]).tolist()
        if nm == "前陡后缓":
            u = np.concatenate([np.full(7, 15), np.full(8, 1)]); t = np.concatenate([[0], np.cumsum(u)]).tolist()
        t = np.asarray(t, dtype=np.int64)
        u = np.diff(t)
        b = np.zeros(16, dtype=np.int64)
        b[1:] = 15 - u                      # 由 lambda(x+1)-lambda(x)=15-b_{x+1}
        v = np.array([lam(x, b) for x in range(16)], dtype=np.int64)
        v = v - v[0]
        err = int(np.max(np.abs(v - (t - t[0]))))
        print(f"   {nm:10s} 目标步长={u.tolist()}")
        print(f"   {'':10s} 构造 b={b.tolist()}")
        print(f"   {'':10s} 实现 lambda={v.tolist()}  误差={err}")
        out[f"construct_{nm}"] = {"err": err, "b": b.tolist()}

    # ---------- 网络层 ----------
    print("\n[网络层]")
    print("   lambda_ij(x) = lambda_ij(0) + sum_{t=1..x} w_{ij,t},  w_{ij,t} = 15 - b_{ij,t+1} ∈ [1,15]")
    print("   Lambda_j = sum_i lambda_ij(x_i)")
    print("            = [sum_i lambda_ij(0)] + sum_i sum_{t<=x_i} w_{ij,t}")
    print("   由于 sum_{t<=x_i} w 是 x_i 的单调函数, 可写成  W_{ij}*x_i + e_{ij}(x_i),")
    print("   其中 W_{ij} = w 的某种平均, e 为有界单调误差。")
    print("   => 网络 = 整数权重(4bit)线性阈值单元 + 单调不减激活")

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    p = os.path.join(res, "mechanism_final_corrected.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已写出:", p)

if __name__ == "__main__":
    main()

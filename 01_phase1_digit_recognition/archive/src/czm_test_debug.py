#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐项明细: 看每个 k 在 x -> x+1 时到底怎么变"""
import numpy as np

B = 330
b = np.array([0, 21, 21, 34, 26, 26, 24, 15, 30, 28, 5, 19, 33, 17, 8, 8], dtype=np.int64)

def a(k, x):
    return 14 + k - x

def c(k, x):
    ak = a(k, x)
    return ak if ak >= 15 else int(b[k])

def S(x):
    return sum(c(k, x) for k in range(16))

print("判据: c_k(x) = a_k(x) if a_k(x)>=15 else b_k ;  a_k(x)=14+k-x")
print(f"b = {b.tolist()}\n")
for x in range(4):
    row = [c(k, x) for k in range(16)]
    print(f"x={x}: S={S(x):4d}  c_k = {row}")
    print(f"      a_k = {[a(k,x) for k in range(16)]}")

print("\n[x -> x+1 的逐项变化]")
for x in range(3):
    ch = [(k, c(k, x), c(k, x + 1), c(k, x + 1) - c(k, x)) for k in range(16)
          if c(k, x) != c(k, x + 1)]
    print(f"x={x}->{x+1}: ΔS={S(x+1)-S(x)}  变化的项 (k, 旧, 新, 差) = {ch}")

print("\n[候选公式检验]")
print(f"{'x':>3} {'S(x)':>7} {'ΔS':>6} | {'b_{x+1}-29+x':>12} {'b_{x+1}-15':>11} {'b_{x+1}-a_x(x)':>15}")
for x in range(16):
    dS = S(x + 1) - S(x) if x < 15 else None
    p1 = int(b[x + 1]) - 29 + x if x < 15 else None
    p2 = int(b[x + 1]) - 15 if x < 15 else None
    p3 = int(b[x + 1]) - a(x, x) if x < 15 else None
    print(f"{x:3d} {S(x):7d} {str(dS):>6} | {str(p1):>12} {str(p2):>11} {str(p3):>15}")

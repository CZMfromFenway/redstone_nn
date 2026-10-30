#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把两种截断的真值、候选闭式、候选步长公式放在一起，找出到底哪一个是对的"""
import numpy as np

def c_of(b, k, x, mode):
    a = (14 + k - x) if mode == "comp" else min(15, 14 + k - x)
    return a if a >= 15 else int(b[k])

def S(b, x, mode):
    return sum(c_of(b, k, x, mode) for k in range(16))

b = np.array([0, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
print("b =", b.tolist(), "\n")

for mode in ("comp", "dust"):
    print("=" * 70)
    print(f"mode = {mode}")
    print(f"{'x':>3} {'S(x)':>8} {'ΔS':>7}  c_k 明细")
    prev = None
    for x in range(16):
        s = S(b, x, mode)
        d = "" if prev is None else s - prev
        print(f"{x:3d} {s:8d} {str(d):>7}  {[c_of(b,k,x,mode) for k in range(16)]}")
        prev = s
    print()
    for x in range(15):
        dS = S(b, x + 1, mode) - S(b, x, mode)
        cands = {
            "b_{x+1}-15": int(b[x+1]) - 15,
            "b_{x+1}-29+x": int(b[x+1]) - 29 + x,
        }
        print(f"  x={x}: 实测ΔS={dS:6d}   " +
              "   ".join(f"{k}={v}" for k, v in cands.items()) +
              ("   <== " + ", ".join(k for k, v in cands.items() if v == dS) if any(v == dS for v in cands.values()) else "   <== 无匹配"))
    print()

# 闭式候选
def T(m): return m * (m + 1) // 2
print("=" * 70)
print("闭式候选核对 (对随机 b)")
rng = np.random.default_rng(3)
for _ in range(2):
    bb = rng.integers(0, 898, 16)
    print("\nb =", bb.tolist())
    for mode in ("comp", "dust"):
        print(f"  {mode}:")
        for x in (0, 1, 5, 15):
            s = S(bb, x, mode)
            p1 = int(np.sum(bb[:x+1])) + 15 * (15 - x)
            p2 = int(np.sum(bb[:x+1])) + 15 * (15 - x) + T(14 - x)
            p3 = int(np.sum(bb[:x+1])) + sum(14 + k - x for k in range(x+1, 16))
            print(f"    x={x:2d}: 实测={s:6d}  [b前缀+15*(15-x)]={p1:6d}  "
                  f"[+T(14-x)]={p2:6d}  [b前缀+a_k尾]={p3:6d}")

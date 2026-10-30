#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""定稿公式大规模验证"""
import numpy as np, json, os

def c_of(b, k, x, mode):
    a = (14 + k - x) if mode == "comp" else min(15, 14 + k - x)
    return a if a >= 15 else int(b[k])

def S(b, x, mode):
    return sum(c_of(b, k, x, mode) for k in range(16))

def T(m): return m * (m + 1) // 2

def S_comp_f(b, x):
    return int(np.sum(b[:x + 1])) + 15 * (15 - x) + T(14 - x)

def S_dust_f(b, x):
    return int(np.sum(b[:x + 1])) + 15 * (15 - x)

def main():
    rng = np.random.default_rng(20261003)
    rep = {}
    print("=" * 88); print("定稿公式大规模验证"); print("=" * 88)

    N = 20000
    # 1. 闭式
    for mode, f in (("comp", S_comp_f), ("dust", S_dust_f)):
        bad = 0
        for _ in range(N):
            b = rng.integers(0, 898, size=16)
            for x in range(16):
                if S(b, x, mode) != f(b, x):
                    bad += 1
        print(f"[1] {mode:5s} 闭式不匹配 = {bad}/{N*16}")
        rep[f"closed_{mode}"] = bad

    # 2. 步长
    for mode, g in (("comp", lambda b, x: int(b[x+1]) - 29 + x),
                    ("dust", lambda b, x: int(b[x+1]) - 15)):
        bad = 0
        for _ in range(N):
            b = rng.integers(0, 898, size=16)
            for x in range(15):
                if S(b, x+1, mode) - S(b, x, mode) != g(b, x):
                    bad += 1
        print(f"[2] {mode:5s} 步长公式不匹配 = {bad}/{N*15}")
        rep[f"step_{mode}"] = bad

    # 3. 步长窗口
    print("\n[3] Δλ 可行窗口 (b∈[0,897])")
    for mode in ("comp", "dust"):
        for form in ("forward", "reverse"):
            got = set()
            for _ in range(3000):
                b = rng.integers(0, 898, size=16)
                for x in range(15):
                    dS = S(b, x+1, mode) - S(b, x, mode)
                    got.add(-dS if form == "forward" else dS)
            print(f"    {mode:5s} {form:8s}: [{min(got):6d}, {max(got):5d}]  ({len(got)} 个取值)")
            rep[f"window_{mode}_{form}"] = [min(got), max(got), len(got)]

    # 4. span
    print("\n[4] span")
    for mode in ("comp", "dust"):
        bad = 0; mx = 0
        for _ in range(5000):
            b = rng.integers(0, 898, size=16)
            v = np.array([S(b, x, mode) for x in range(16)])
            span = int(v.max() - v.min())
            if mode == "dust":
                step = np.array([15 - int(b[t]) for t in range(1, 16)], dtype=np.int64)
            else:
                step = np.array([(29 - (t - 1)) - int(b[t]) for t in range(1, 16)], dtype=np.int64)
            pref = np.concatenate([[0], np.cumsum(step)]).astype(np.int64)
            if span != int(pref.max() - pref.min()):
                bad += 1
            mx = max(mx, span)
        print(f"    {mode:5s}: span = max(前缀和)-min(前缀和),  不匹配={bad}/5000,  实测最大={mx}")
        rep[f"span_{mode}"] = {"bad": bad, "max": mx}

    # 5. b_0 无关性
    print("\n[5] b_0 不进入 span")
    for mode in ("comp", "dust"):
        spans = set()
        base = rng.integers(0, 898, size=16)
        for b0 in range(0, 898, 61):
            bb = base.copy(); bb[0] = b0
            v = np.array([S(bb, x, mode) for x in range(16)])
            spans.add(int(v.max() - v.min()))
        print(f"    {mode:5s}: span 集合 = {spans}")
        rep[f"b0_{mode}"] = sorted(spans)

    # 6. 反解
    print("\n[6] 反解 b (b_0=0)")
    for mode in ("comp", "dust"):
        bad = 0
        for _ in range(3000):
            b = rng.integers(0, 898, size=16); b[0] = 0
            lam = -np.array([S(b, x, mode) for x in range(16)], dtype=np.int64)
            d = np.diff(lam)
            bb = np.zeros(16, dtype=np.int64)
            for x in range(15):
                bb[x+1] = (15 - int(d[x])) if mode == "dust" else ((29 - x) - int(d[x]))
            if not np.array_equal(bb, b):
                bad += 1
        print(f"    {mode:5s}: 反解失败 = {bad}/3000")
        rep[f"invert_{mode}"] = bad

    # 7. 边界
    print("\n[7] 上下界 (b 拉满 897)")
    bm = np.full(16, 897, dtype=np.int64)
    for mode in ("comp", "dust"):
        v = np.array([S(bm, x, mode) for x in range(16)])
        print(f"    {mode:5s}: S∈[{v.min()}, {v.max()}]")
        rep[f"bounds_{mode}"] = [int(v.min()), int(v.max())]
    print(f"    Δλ 窗口宽度恒为 897;  span 上界 15*897 = {15*897}")

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    json.dump(rep, open(os.path.join(res, "final_audit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/final_audit.json")

if __name__ == "__main__":
    main()

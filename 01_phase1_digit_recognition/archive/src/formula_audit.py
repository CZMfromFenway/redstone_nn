#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
定稿公式全量验证
=================
对文档中将要写出的每一条公式做数值核对。
"""
import numpy as np, json, os

def a_comp(k, x): return 14 + k - x
def a_dust(k, x): return min(15, 14 + k - x)

def c_of(b, k, x, mode):
    a = a_comp(k, x) if mode == "comp" else a_dust(k, x)
    return a if a >= 15 else int(b[k])

def S_brute(b, x, mode):
    return sum(c_of(b, k, x, mode) for k in range(16))

def S_comp_closed(b, x):
    """S(x) = sum_{i<=x} b_i + T(14-x),  T(m)=m(m+1)/2"""
    m = 14 - x
    return int(np.sum(b[:x + 1])) + m * (m + 1) // 2
def S_dust_closed(b, x):
    """S(x) = sum_{i<=x} b_i + 15*(15-x)"""
    return int(np.sum(b[:x + 1])) + 15 * (15 - x)

def main():
    rng = np.random.default_rng(20261003)
    rep = {}
    print("=" * 90); print("定稿公式全量验证"); print("=" * 90)

    M = 4000
    # ---- 1. 闭式解 ----
    print("\n[1] 闭式解")
    for mode, f in (("comp", S_comp_closed), ("dust", S_dust_closed)):
        bad = 0
        for _ in range(M):
            b = rng.integers(0, 898, size=16)
            for x in range(16):
                if S_brute(b, x, mode) != f(b, x):
                    bad += 1
        print(f"   {mode:5s}  S(x) 闭式与逐项模拟不匹配 = {bad}/{M*16}")
        rep[f"closed_{mode}"] = bad

    # ---- 2. 步长公式 ----
    print("\n[2] 步长公式")
    for mode, g in (("comp", lambda b, x: int(b[x+1]) - 29 + x),
                    ("dust", lambda b, x: int(b[x+1]) - 15)):
        bad = 0
        for _ in range(M):
            b = rng.integers(0, 898, size=16)
            for x in range(15):
                if S_brute(b, x + 1, mode) - S_brute(b, x, mode) != g(b, x):
                    bad += 1
        tag = "b_{x+1}-29+x" if mode == "comp" else "b_{x+1}-15"
        print(f"   {mode:5s}  ΔS(x) = {tag:12s} 不匹配 = {bad}/{M*15}")
        rep[f"step_{mode}"] = bad

    # ---- 3. 步长可行窗口 ----
    print("\n[3] 步长 Δλ 的可行取值 (b∈[0,897])")
    for mode in ("comp", "dust"):
        for form in ("forward", "reverse"):
            got = set()
            for _ in range(4000):
                b = rng.integers(0, 898, size=16)
                for x in range(15):
                    dS = S_brute(b, x + 1, mode) - S_brute(b, x, mode)
                    got.add(-dS if form == "forward" else dS)   # Δλ
            lo, hi = min(got), max(got)
            print(f"   {mode:5s} {form:8s}: Δλ ∈ [{lo:6d}, {hi:5d}]  (实测 {len(got)} 个取值)")
            rep[f"window_{mode}_{form}"] = {"min": lo, "max": hi, "n": len(got)}

    # ---- 4. span 公式 ----
    print("\n[4] span 公式")
    # dust: λ(x) = λ(0) + Σ_{t=1..x}(15-b_t)  -> span = max/min of 前缀和
    bad = 0; mx = 0
    for _ in range(4000):
        b = rng.integers(0, 898, size=16)
        v = np.array([S_brute(b, x, "dust") for x in range(16)])
        span = int(v.max() - v.min())
        pref = np.concatenate([[0], np.cumsum(15 - b[1:])]).astype(np.int64)
        pred = int(pref.max() - pref.min())
        mx = max(mx, span)
        if span != pred:
            bad += 1
    print("   dust: span = max(前缀和)-min(前缀和),  前缀和_x = sum_{t<=x}(15-b_t)")
    print(f"         不匹配 = {bad}/4000,  实测最大 span = {mx}")
    rep["span_dust"] = {"bad": bad, "max_span": mx}

    # comp: λ(x) = λ(0) + Σ_{t=1..x}[(29-(t-1)) - b_t]
    bad = 0; mx = 0
    for _ in range(4000):
        b = rng.integers(0, 898, size=16)
        v = np.array([S_brute(b, x, "comp") for x in range(16)])
        span = int(v.max() - v.min())
        steps = np.array([(29 - (t - 1)) - int(b[t]) for t in range(1, 16)], dtype=np.int64)
        pref = np.concatenate([[0], np.cumsum(steps)]).astype(np.int64)
        pred = int(pref.max() - pref.min())
        mx = max(mx, span)
        if span != pred:
            bad += 1
    print("   comp: span = max(前缀和)-min(前缀和),  步长 = (29-(t-1))-b_t")
    print(f"         不匹配 = {bad}/4000,  实测最大 span = {mx}")
    rep["span_comp"] = {"bad": bad, "max_span": mx}

    # ---- 5. b_0 无关性 ----
    print("\n[5] b_0 不影响 span")
    for mode in ("comp", "dust"):
        spans = set()
        base = rng.integers(0, 898, size=16)
        for b0 in range(0, 898, 97):
            bb = base.copy(); bb[0] = b0
            v = np.array([S_brute(bb, x, mode) for x in range(16)])
            spans.add(int(v.max() - v.min()))
        print(f"   {mode:5s}: b_0 变化时 span 集合 = {spans}")
        rep[f"b0_{mode}"] = sorted(spans)

    # ---- 6. 上下界 ----
    print("\n[6] 绝对上下界")
    bmax = np.full(16, 897, dtype=np.int64)
    for mode in ("comp", "dust"):
        v = np.array([S_brute(bmax, x, mode) for x in range(16)])
        print(f"   {mode:5s}  b=897: S ∈ [{v.min()}, {v.max()}]  => λ = B - S ∈ [B-{v.max()}, B-{v.min()}]")
        rep[f"bounds_{mode}"] = {"Smin": int(v.min()), "Smax": int(v.max())}
    print(f"   Δλ 的窗口宽度恒为 897;  span 上界 15*897 = {15*897}")

    # ---- 7. 反解与重建 (dust, comp) ----
    print("\n[7] 反解 b 并重建")
    for mode in ("comp", "dust"):
        bad = 0
        for _ in range(2000):
            b = rng.integers(0, 898, size=16); b[0] = 0
            S = np.array([S_brute(b, x, mode) for x in range(16)], dtype=np.int64)
            lam = -S
            d = np.diff(lam)
            bb = np.zeros(16, dtype=np.int64)
            for x in range(15):
                bb[x + 1] = (15 - int(d[x])) if mode == "dust" else ((29 - x) - int(d[x]))
            if not np.array_equal(bb, b):
                bad += 1
        print(f"   {mode:5s}: 反解返回原 b 的失败次数 = {bad}/2000")
        rep[f"invert_{mode}"] = bad

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    json.dump(rep, open(os.path.join(res, "final_formula_audit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/final_formula_audit.json")

if __name__ == "__main__":
    main()

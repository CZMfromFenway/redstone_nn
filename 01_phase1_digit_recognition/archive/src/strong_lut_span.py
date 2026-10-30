#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1) lambda 的可观测变程(span) 由 b_1..b_15 的极差决定, 与绝对高度无关; 绝对高度由 B 调。
2) b_0 只贡献常数, 不参与变程。
3) 体积不应含输出幅度。
"""
import numpy as np, json, os

def a_comp(k, x):
    return 14 + k - x

def a_dust(k, x):
    return min(15, 14 + k - x)

def c_of(b, a):
    return a if a >= 15 else int(b)

def S(x, b, mode):
    f = a_comp if mode == "comp" else a_dust
    return sum(c_of(b[k], f(k, x)) for k in range(16))

def main():
    out = {}
    rng = np.random.default_rng(11)
    print("=" * 92)
    print("span 关系与 b_0 的角色")
    print("=" * 92)

    # ---------- 1. b_0 是否影响 span ----------
    print("\n[1] 固定 b_1..b_15, 改变 b_0, 看 lambda 的变程是否变化")
    for mode in ("comp", "dust"):
        base = np.array([0, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
                        dtype=np.int64)
        spans = set()
        for b0 in (0, 1, 100, 500, 897):
            bb = base.copy(); bb[0] = b0
            v = np.array([S(x, bb, mode) for x in range(16)], dtype=np.int64)
            spans.add(int(v.max() - v.min()))
        print(f"   {mode:5s}: 不同 b_0 下的 span 集合 = {spans}  (应只有一个值)")
        out[f"b0_span_{mode}"] = sorted(spans)

    # ---------- 2. span 的上界是否 = max b - min b (在 b_1..b_15 上) ----------
    print("\n[2] 验证  span(lambda) = max_{t=1..15} b_t - min_{t=1..15} b_t")
    for mode in ("comp", "dust"):
        bad = 0; mx = 0
        for _ in range(5000):
            b = rng.integers(0, 898, size=16)
            v = np.array([S(x, b, mode) for x in range(16)], dtype=np.int64)
            span = int(v.max() - v.min())
            pred = int(b[1:].max() - b[1:].min())
            mx = max(mx, span)
            if span != pred:
                bad += 1
        print(f"   {mode:5s}: 不匹配 = {bad}/5000,  实测最大 span = {mx}")
        out[f"span_rule_{mode}"] = {"bad": bad, "max_span": mx}

    # ---------- 3. 极端构造 ----------
    print("\n[3] 极端: b_1..b_15 取遍 {0,897} 两端")
    for mode in ("comp", "dust"):
        b = np.zeros(16, dtype=np.int64); b[1:] = 897
        v = np.array([S(x, b, mode) for x in range(16)], dtype=np.int64)
        b2 = np.zeros(16, dtype=np.int64)
        v2 = np.array([S(x, b2, mode) for x in range(16)], dtype=np.int64)
        print(f"   {mode:5s}: b=0 -> span={v2.max()-v2.min()},  S={v2.tolist()}")
        print(f"   {mode:5s}: b=897 -> span={v.max()-v.min()},  S={v.tolist()}")
        out[f"extreme_{mode}"] = {"b0_span": int(v2.max() - v2.min()),
                                  "b897_span": int(v.max() - v.min())}

    # ---------- 4. 绝对值与 B ----------
    print("\n[4] 绝对高度由 B 平移: lambda = B - S")
    b = np.zeros(16, dtype=np.int64); b[1:] = 897
    S0 = np.array([S(x, b, "dust") for x in range(16)], dtype=np.int64)
    for B in (0, 5000, 14352, 30000):
        lam = B - S0
        print(f"   B={B:6d}: lambda = [{lam.min()}, {lam.max()}]  span={lam.max()-lam.min()}")
        out[f"B_shift_{B}"] = {"min": int(lam.min()), "max": int(lam.max())}

    # ---------- 5. 体积: 只由 n, h, L 决定 ----------
    print("\n[5] 体积模型 (不含输出幅度):  V = 4n * h * L")
    for n, h in ((4, 15), (4, 48), (4, 64), (15, 15)):
        for L in (32, 40):
            lut = 4 * n * h * L
            dt = 8 * n * h * 3
            print(f"   n={n:2d} h={h:3d} L={L}: LUT={lut:8,}  DT(d=3)={dt:8,}  比={lut/dt:.1f}x")
            out[f"vol_n{n}_h{h}_L{L}"] = {"lut": lut, "dt": dt, "ratio": lut / dt}

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    p = os.path.join(res, "span_and_volume.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已写出:", p)

if __name__ == "__main__":
    main()

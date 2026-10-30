#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强模 LUT 单元机制核查 (v11, 定稿)
==================================
精确关系 (已在 v9 验证 0/64000 不匹配):

    S(x) = sum_{j=0}^{15} min( b_j, 14 + j - x )
    lambda(x) = B - S(x)                       (顺式, B 为常数)
    lambda(x) = S(x) + const                   (反式)

本脚本回答两个定量问题, 用模拟退火而非弱贪心:
  Q1  lambda 的步长集合是什么?
  Q2  哪些目标阶梯 lambda 可以被精确实现?
"""
import numpy as np, json, os

def S_all(b):
    """返回 x=0..15 的 S 值"""
    x = np.arange(16)
    # min(b_j, 14+j-x) 对 j=0..15, x=0..15
    A = 14 + np.arange(16)[None, :] - x[:, None]        # (16,16) A[x,j]
    return np.minimum(np.asarray(b)[None, :], A).sum(axis=1)

def lam_all(b, B=0):
    return B - S_all(b)

def anneal(target, iters=200000, seed=0, bmax=897):
    """在 b ∈ [0,bmax]^16 上最小化 max_x |lam(x) - target(x)|"""
    rng = np.random.default_rng(seed)
    t = np.asarray(target, dtype=np.int64)
    t0 = t - t[0]                      # 只关心形状(常数偏移可吸收)
    b = rng.integers(0, 15, size=16).astype(np.int64)

    def loss(bb):
        v = -S_all(bb)                 # 顺式取下 B=0 得到 -S
        return int(np.max(np.abs((v - v[0]) - t0)))
    cur = loss(b); best = (cur, b.copy())
    for it in range(iters):
        T = max(1.0, 12.0 * (1 - it / iters))
        j = int(rng.integers(16))
        old = int(b[j])
        b[j] = int(np.clip(old + rng.integers(-15, 16), 0, bmax))
        e = loss(b)
        if e <= cur or rng.random() < np.exp((cur - e) / T):
            cur = e
        else:
            b[j] = old
        if cur < best[0]:
            best = (cur, b.copy())
        if best[0] == 0:
            break
    return best

def main():
    rng = np.random.default_rng(3)
    out = {}
    print("=" * 88); print("强模 LUT 机制核查 v11 (定稿)"); print("=" * 88)

    # Q1 步长集合
    print("\n[Q1] lambda 的步长集合")
    for label, lo, hi in (("b∈[0,14]", 0, 14), ("b∈[0,15]", 0, 15), ("b∈[0,897]", 0, 897)):
        incs = set()
        for _ in range(4000):
            b = rng.integers(lo, hi + 1, size=16)
            v = lam_all(b, 0)
            incs.update(int(t) for t in np.unique(np.diff(v)))
        print(f"   {label:12s} 步长集合 = {sorted(incs)}")
        out[f"steps_{label}"] = sorted(incs)

    # Q2 可达阶梯
    print("\n[Q2] 目标阶梯的精确可实现性 (模拟退火, b∈[0,897])")
    tests = {
        "恒定":       [0] * 16,
        "每步+1":     list(range(16)),
        "每步+14":    [14 * i for i in range(16)],
        "每步+13":    [13 * i for i in range(16)],
        "凹阶梯":     [0, 0, 14, 14, 14, 28, 28, 28, 28, 42, 42, 42, 42, 42, 56, 56],
        "凸阶梯":     [0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 6, 6, 6, 10, 10, 15],
        "二值阶跃":   [0] * 8 + [15] * 8,
        "随机非减":   None,
    }
    for nm, t in tests.items():
        if t is None:
            st = np.minimum(rng.integers(0, 15, size=15), 14)
            t = np.concatenate([[0], np.cumsum(st)]).tolist()
        e, b = anneal(t, iters=60000, seed=1)
        v = lam_all(b, 0); v = v - v[0]
        print(f"   {nm:10s} 目标={t[:8]}... 误差={e}")
        if e == 0:
            print(f"   {'':10s} 精确实现, b={b.tolist()}")
        else:
            print(f"   {'':10s} 最佳逼近 = {v.tolist()}")
        out[f"anneal_{nm}"] = {"err": e, "b": b.tolist()}

    # 结论
    print("\n[结论]")
    print("   - lambda 是 x 的单调不减阶梯 (顺式取负号后单调不减)")
    print("   - 单张表格的自由度 = 16 个 b_j, 但它们通过 min(.,14+j-x) 耦合,")
    print("     因此可达阶梯并非任意非减阶梯, 需要一个精确的可达性刻画。")

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    p = os.path.join(res, "mechanism_v11.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已写出:", p)

if __name__ == "__main__":
    main()

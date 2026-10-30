#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：加宽 + 专项训练（定稿版：只用已验证有效的组件）
======================================================
【实验配置】
  输入空间   (x1..x4) ∈ {0..15}^4，全部 65536 状态穷举
  目标标签   N4_MLP_T1 官方标签（4->8->1 tanh MLP，seed 71000，正类 18436）
  评价       全部 65536 状态上的最终 0/1 错误数
  网络       4->h->1；隐藏层 round((tanh+1)/2*15) 量化到 16 级（STE）；
             输出层 sign(sum_j v_j h_j - theta)，h_j 为 0~15 整数，v/theta 高精度

  两阶段
    阶段1  训练隐藏层连续的实数 MLP（BCE，1500 步）
    阶段2  以阶段1 全部权重初始化，把隐藏层量化到 16 级，
           用**精确 hinge**（margin=1）微调，固定 lr 小步长
  另附     阶段2b：hinge 之后再用 BCE 小步长抛光
"""
import numpy as np, json, os, time

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def evaluate(Xn, W1, b1, v, th, y, L=16):
    Hq = np.clip(np.round((np.tanh(Xn @ W1.T + b1) + 1) * (L - 1) / 2), 0, L - 1)
    return int(np.sum(((Hq @ v - th) >= 0).astype(np.int64) != y))

def run(Xn, y, h, loss, seed, init=None, steps=2500, lr=0.03, bs=8192, L=16):
    rng = np.random.default_rng(seed); N = len(Xn)
    if init is None:
        W1 = rng.normal(0, 0.6, (h, 4)); b1 = np.zeros(h)
        v = rng.normal(0, 0.6, h); th = 0.0
    else:
        W1, b1, v, th = init[0].copy(), init[1].copy(), init[2].copy(), float(init[3])
    ps = [W1, b1, v, np.array(th)]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    s = (L - 1) / 2; ys = 2 * y - 1
    best = None
    for t in range(1, steps + 1):
        idx = rng.choice(N, bs, replace=False)
        xb = Xn[idx]; yb = y[idx]; yy = ys[idx]
        Hc = np.tanh(xb @ W1.T + b1)
        Hq = np.clip(np.round((Hc + 1) * s), 0, L - 1)
        z = Hq @ v - th
        if loss == "bce":
            p = 1 / (1 + np.exp(-z)); dz = (p - yb) / bs
        else:                                   # 精确 hinge
            m = 1 - yy * z
            dz = (-yy * (m > 0)) / bs
        gv = Hq.T @ dz; gth = -dz.sum()
        dH = np.outer(dz, v) * (s * (1 - Hc ** 2))
        g = [dH.T @ xb, dH.sum(0), gv, np.array(gth)]
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 250 == 0 or t == steps:
            e = evaluate(Xn, W1, b1, v, th, y, L)
            if best is None or e < best[0]:
                best = (e, W1.copy(), b1.copy(), v.copy(), th)
    return best

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1t = np.array(m["W1"]); b1t = np.array(m["b1"]); W2t = np.array(m["W2"]); b2t = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)
    errA = int(np.sum(((np.tanh(Xn @ W1t.T + b1t) @ W2t + b2t) >= 0).astype(np.int8) != lab))

    print("=" * 104)
    print("实验：加宽 + 专项训练（定稿版）")
    print("=" * 104)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举")
    print(f"目标标签 : N4_MLP_T1（正类 {int(y.sum())}）")
    print(f"参照     : A 原 Teacher（隐藏层连续）= {errA}/{N}")
    print(f"\n{'h':>4} | {'阶段1 连续':>11} | {'阶段2 hinge':>12} | {'阶段2b +BCE抛光':>15} | {'用时(s)':>8}")
    print("-" * 104)
    rows = []
    for h in (16, 32, 64):
        t0 = time.time()
        r1 = None
        for seed in (0, 1):
            r = run(Xn, y, h, "bce", seed)
            if r1 is None or r[0] < r1[0]: r1 = r
        r2 = None
        for seed in (0, 1):
            r = run(Xn, y, h, "hinge", seed, init=r1[1:], lr=0.02)
            if r2 is None or r[0] < r2[0]: r2 = r
        r3 = None
        for seed in (0, 1):
            r = run(Xn, y, h, "bce", seed, init=r2[1:], lr=0.005, steps=1000)
            if r3 is None or r[0] < r3[0]: r3 = r
        el = time.time() - t0
        print(f"{h:4d} | {r1[0]:11d} | {r2[0]:12d} | {r3[0]:15d} | {el:8.0f}")
        rows.append({"h": h, "s1": r1[0], "s2": r2[0], "s2b": r3[0], "time_s": el})

    print("\n[错误率]")
    print(f"   {'h':>4} | {'阶段1':>10} | {'阶段2':>10} | {'阶段2b':>10}")
    for r in rows:
        print(f"   {r['h']:4d} | {r['s1']/N*100:9.4f}% | {r['s2']/N*100:9.4f}% | {r['s2b']/N*100:9.4f}%")

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    json.dump({"A": errA, "rows": rows},
              open(os.path.join(res, "width_scan_final.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: results/width_scan_final.json")

if __name__ == "__main__":
    main()

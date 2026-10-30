#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：加宽隐层 + 专项训练，能否补回"隐藏层只能传 4 bit"造成的容量损失
======================================================================
【实验配置】
  输入空间   (x1..x4) ∈ {0..15}^4，全部 65536 状态穷举
  目标标签   N4_MLP_T1 官方标签（4->8->1 tanh MLP，seed 71000，正类 18436）
  评价       全部 65536 状态上的最终 0/1 错误数

  训练方法（"离散中间神经元 + 反编译"）
    网络 4 -> h -> 1，隐藏层输出 = round((tanh(pre)+1)/2*(L-1))，L=16，STE 反传；
    隐藏层的 16 级输出直接就是强模 LUT 的阶梯取值，训练完可反编译。
    输出层：隐藏值已是整数，取 sign(sum_j v_j*h_j - theta)，v,theta 为高精度整数。

  扫描
    h ∈ {8, 16, 32, 64}  ×  两种隐藏层目标 {量化到 16 级, 连续 tanh}
"""
import numpy as np, json, os, time

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def train(X, y, h, quantized=True, L=16, seed=0, steps=1500, lr=0.05, bs=8192, verbose=False):
    """
    4->h->1，隐藏层可选量化到 L 级。输出层为线性判别。
    返回 (最终错误, 隐藏编码 Hq, v, theta)
    """
    rng = np.random.default_rng(seed)
    N = len(X)
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    W1 = rng.normal(0, 0.6, (h, 4)); b1 = np.zeros(h)
    v = rng.normal(0, 0.6, h); th = 0.0
    params = [W1, b1, v, np.array(th)]
    ms = [np.zeros_like(p) for p in params]; vs = [np.zeros_like(p) for p in params]
    best = None
    for t in range(1, steps + 1):
        idx = rng.choice(N, bs, replace=False)
        xb = Xn[idx]; yb = y[idx]
        pre = xb @ W1.T + b1
        Hc = np.tanh(pre)
        if quantized:
            s = (L - 1) / 2
            Hq = np.clip(np.round((Hc + 1) * s), 0, L - 1)
            Huse = Hq                       # 前向用量化值
            # STE: d(round(a))/da = 1；量化器整体是 a = (L-1)/2*(tanh+1)，
            #      链式法则给出 dHq/dpre = (L-1)/2 * (1 - tanh^2)
            dHdHc = s * (1 - Hc ** 2)
        else:
            Huse = Hc
            dHdHc = 1 - Hc ** 2
        z = Huse @ v - th
        p = 1 / (1 + np.exp(-z)); dz = (p - yb) / bs
        gv = Huse.T @ dz; gth = -dz.sum()
        dH = np.outer(dz, v) * dHdHc
        gW1 = dH.T @ xb; gb1 = dH.sum(axis=0)
        for i, (pp, g) in enumerate(zip(params, [gW1, gb1, gv, np.array(gth)])):
            ms[i] = 0.9 * ms[i] + 0.1 * g; vs[i] = 0.999 * vs[i] + 0.001 * g ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 250 == 0 or t == steps:
            e, Hq_all = evaluate(Xn, W1, b1, v, th, quantized, L, y)
            if best is None or e < best[0]:
                best = (e, Hq_all.copy(), W1.copy(), b1.copy(), v.copy(), th)
            if verbose:
                print(f"      step {t:5d}: 错误 = {e}")
    return best

def evaluate(Xn, W1, b1, v, th, quantized, L, y):
    pre = Xn @ W1.T + b1
    Hc = np.tanh(pre)
    if quantized:
        Hq = np.clip(np.round((Hc + 1) * (L - 1) / 2), 0, L - 1)
    else:
        Hq = Hc
    pred = ((Hq @ v - th) >= 0).astype(np.int64)
    return int(np.sum(pred != y)), Hq

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1t = np.array(m["W1"]); b1t = np.array(m["b1"])
    W2t = np.array(m["W2"]); b2t = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)

    print("=" * 108)
    print("实验：加宽隐层 + 专项训练，补回隐藏层 4 bit 传输造成的容量损失")
    print("=" * 108)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举")
    print(f"目标标签 : N4_MLP_T1（4->8->1 tanh，seed 71000，正类 {int(y.sum())}）")
    print(f"训练     : 4->h->1，隐藏层 round((tanh+1)/2*15) 量化到 16 级，STE 反传")
    print(f"           输出层用高精度阈值 sign(sum v_j h_j - theta)，h_j 为 0~15 整数")

    # 原 Teacher 参照
    predA = ((np.tanh(Xn @ W1t.T + b1t) @ W2t + b2t) >= 0).astype(np.int8)
    errA = int(np.sum(predA != lab))
    print(f"\n[参照] A 原 Teacher（隐藏层连续）: 错误 = {errA}/{N}")

    print(f"\n{'h':>4} | {'隐藏层':>10} | {'最佳错误':>9} | {'错误率':>9} | {'隐藏值个数':>10} | {'用时(s)':>8}")
    print("-" * 108)
    rows = []
    for h in (8, 16, 32, 64):
        for q in (True, False):
            t0 = time.time()
            best = None
            for seed in (0, 1):
                r = train(X, y, h, quantized=q, seed=seed, steps=1500)
                if best is None or r[0] < best[0]:
                    best = r
            el = time.time() - t0
            nlev = len(np.unique(np.round(best[1] * 1 if q else best[1] * 0 + best[1])))
            tag = "量化 0~15" if q else "连续 tanh"
            print(f"{h:4d} | {tag:>10} | {best[0]:9d} | {best[0]/N*100:8.4f}% | "
                  f"{len(np.unique(best[1])):10d} | {el:8.0f}")
            rows.append({"h": h, "quantized": q, "err": best[0], "time_s": el})

    print("\n[说明] '隐藏值个数' 是隐藏层实际用到的不同取值数（量化时上限 16）。")

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    json.dump({"A": errA, "rows": rows,
               "config": {"label": "N4_MLP_T1", "states": 65536,
                          "train": "4->h->1, hidden quantized to 16 levels, STE, Adam, 1500 steps, 2 seeds"}},
              open(os.path.join(res, "width_scan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: results/width_scan.json")

if __name__ == "__main__":
    main()

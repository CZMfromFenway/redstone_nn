#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：加宽隐层 + 专项训练（第二轮：换损失函数）
================================================
【实验配置】
  输入空间   (x1..x4) ∈ {0..15}^4，全部 65536 状态穷举
  目标标签   N4_MLP_T1 官方标签（4->8->1 tanh MLP，seed 71000，正类 18436）
  评价       全部 65536 状态上的最终 0/1 错误数

  网络 4->h->1；隐藏层 round((tanh+1)/2*15) 量化到 16 级，STE 反传；
  输出层 sign(sum_j v_j h_j - theta)，v/theta 高精度整数（h_j 是 0~15 整数）。

  对比两种损失 + 一种初始化：
    L1  BCE（逻辑损失）
    L2  hinge  max(0, 1 - y'*z)，y' = ±1   —— 强行推大间隔，逼近"精确可分"
    L3  hinge + 用实数 MLP 的隐藏层做初始化
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
    return int(np.sum(((Hq @ v - th) >= 0).astype(np.int64) != y)), Hq

def train(X, y, h, loss="bce", seed=0, steps=2500, lr=0.03, bs=8192, L=16,
          init=None, margin=1.0, verbose=False):
    rng = np.random.default_rng(seed)
    N = len(X); Xn = (X.astype(np.float64) - 7.5) / 7.5
    if init is not None:
        W1, b1 = init[0].copy(), init[1].copy()
        # 若是实数 MLP 的权重，按 tanh 尺度不变性直接可用
    else:
        W1 = rng.normal(0, 0.6, (h, 4)); b1 = np.zeros(h)
    v = rng.normal(0, 0.6, h); th = 0.0
    params = [W1, b1, v, np.array(th)]
    ms = [np.zeros_like(p) for p in params]; vs = [np.zeros_like(p) for p in params]
    s = (L - 1) / 2
    ysign = 2 * y - 1
    best = None
    for t in range(1, steps + 1):
        idx = rng.choice(N, bs, replace=False)
        xb = Xn[idx]; yb = y[idx]; ys = ysign[idx]
        Hc = np.tanh(xb @ W1.T + b1)
        Hq = np.clip(np.round((Hc + 1) * s), 0, L - 1)
        z = Hq @ v - th
        if loss == "bce":
            p = 1 / (1 + np.exp(-z)); dz = (p - yb) / bs
        else:                                  # hinge
            m = 1 - ys * z
            act = (m > 0).astype(np.float64)
            dz = (-ys * act * margin) / bs
        gv = Hq.T @ dz; gth = -dz.sum()
        dH = np.outer(dz, v) * (s * (1 - Hc ** 2))
        gW1 = dH.T @ xb; gb1 = dH.sum(axis=0)
        for i, (pp, g) in enumerate(zip(params, [gW1, gb1, gv, np.array(gth)])):
            ms[i] = 0.9 * ms[i] + 0.1 * g; vs[i] = 0.999 * vs[i] + 0.001 * g ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 250 == 0 or t == steps:
            e, _ = evaluate(Xn, W1, b1, v, th, y, L)
            if best is None or e < best[0]:
                best = (e, W1.copy(), b1.copy(), v.copy(), th)
            if verbose:
                print(f"      step {t:5d}: 错误 = {e}")
    return best

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1t = np.array(m["W1"]); b1t = np.array(m["b1"])
    W2t = np.array(m["W2"]); b2t = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)
    errA = int(np.sum(((np.tanh(Xn @ W1t.T + b1t) @ W2t + b2t) >= 0).astype(np.int8) != lab))

    print("=" * 106)
    print("实验：加宽隐层 + 专项训练（换损失函数）")
    print("=" * 106)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举")
    print(f"目标标签 : N4_MLP_T1（正类 {int(y.sum())}）")
    print(f"网络     : 4->h->1，隐藏层量化 16 级（STE），输出层高精度阈值")
    print(f"参照     : A 原 Teacher（隐藏层连续）= {errA}/{N}")
    print(f"\n{'h':>4} | {'损失/初始化':>22} | {'最佳错误':>9} | {'错误率':>9} | {'用时(s)':>8}")
    print("-" * 106)
    rows = []
    for h in (16, 32, 64):
        for tag, loss, use_init in (("BCE", "bce", False),
                                    ("hinge", "hinge", False),
                                    ("hinge+教师初始化", "hinge", True)):
            t0 = time.time(); best = None
            for seed in (0, 1):
                init = None
                if use_init:
                    # 用真正的实数 MLP（隐藏层连续、无量化）先训一个，取其隐藏层权重
                    from numpy.random import default_rng
                    rng = default_rng(seed)
                    Wi = rng.normal(0, 0.6, (h, 4)); bi = np.zeros(h)
                    vi = rng.normal(0, 0.6, h); thi = 0.0
                    ps = [Wi, bi, vi, np.array(thi)]
                    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
                    for t in range(1, 1501):
                        idx = rng.choice(N, 8192, replace=False)
                        xb = Xn[idx]; yb = y[idx]
                        Hc = np.tanh(xb @ Wi.T + bi); z = Hc @ vi - thi
                        p = 1/(1+np.exp(-z)); dz = (p - yb)/8192
                        g = [np.outer(dz, vi)*(1-Hc**2).T @ xb if False else
                             (np.outer(dz, vi)*(1-Hc**2)).T @ xb,
                             (np.outer(dz, vi)*(1-Hc**2)).sum(0), Hc.T @ dz, np.array(-dz.sum())]
                        for i,(pp,gg) in enumerate(zip(ps,g)):
                            ms[i]=0.9*ms[i]+0.1*gg; vs[i]=0.999*vs[i]+0.001*gg**2
                            pp -= 0.03*(ms[i]/(1-0.9**t))/(np.sqrt(vs[i]/(1-0.999**t))+1e-8)
                    init = (Wi, bi)
                r = train(X, y, h, loss=loss, seed=seed, init=init)
                if best is None or r[0] < best[0]:
                    best = r
            el = time.time() - t0
            print(f"{h:4d} | {tag:>22} | {best[0]:9d} | {best[0]/N*100:8.4f}% | {el:8.0f}")
            rows.append({"h": h, "loss": tag, "err": best[0], "time_s": el})

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    json.dump({"A": errA, "rows": rows},
              open(os.path.join(res, "width_scan_loss.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: results/width_scan_loss.json")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：加宽 + 专项训练（方案：三阶段，阶段3改为"可微软量化退火"）
================================================================
【实验配置】
  输入空间   (x1..x4) ∈ {0..15}^4，全部 65536 状态穷举
  目标标签   N4_MLP_T1 官方标签（4->8->1 tanh MLP，seed 71000，正类 18436）
  评价       全部 65536 状态上的最终 0/1 错误数

  网络 4->h->1。隐藏层必须输出 0~15（纵向弱模），输出层取高精度阈值。

  三阶段训练
    阶段1  训练隐藏层连续的实数 MLP（4->h->1）
    阶段2  以阶段1的隐藏层权重初始化，隐藏层量化到 16 级，STE + 平滑 hinge 微调
    阶段3  软量化退火：把"硬量化"换成温度可调的软量化
              Hq_soft = sum_{t=1}^{15} sigmoid((Hc - c_t)/T)
            c_t 为可学习的 15 个断点，T 从大到小退火。
            T→0 时 sigmoid 变成阶跃，Hq_soft 收敛到整数档号。
            这样阶段3 直接在"断点位置"上做梯度优化，精度对准量化损失。
"""
import numpy as np, json, os, time

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def hard_q(Hc, thresh):
    return np.sum(Hc[:, :, None] >= thresh[None, :, :], axis=2).astype(np.int64)

def err_of(Hq, v, th, y):
    return int(np.sum(((Hq @ v - th) >= 0).astype(np.int64) != y))

def stage1(Xn, y, h, seed, steps=1500, lr=0.03, bs=8192):
    rng = np.random.default_rng(seed); N = len(Xn)
    W1 = rng.normal(0, 0.6, (h, 4)); b1 = np.zeros(h); v = rng.normal(0, 0.6, h); th = 0.0
    ps = [W1, b1, v, np.array(th)]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    best = None
    for t in range(1, steps + 1):
        idx = rng.choice(N, bs, replace=False)
        xb = Xn[idx]; yb = y[idx]
        Hc = np.tanh(xb @ W1.T + b1); z = Hc @ v - th
        p = 1 / (1 + np.exp(-z)); dz = (p - yb) / bs
        dH = np.outer(dz, v) * (1 - Hc ** 2)
        g = [dH.T @ xb, dH.sum(0), Hc.T @ dz, np.array(-dz.sum())]
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 500 == 0 or t == steps:
            e = int(np.sum(((np.tanh(Xn @ W1.T + b1) @ v - th) >= 0).astype(np.int64) != y))
            if best is None or e < best[0]:
                best = (e, W1.copy(), b1.copy(), v.copy(), th)
    return best

def stage2(Xn, y, h, seed, init, steps=2000, lr=0.03, bs=8192, L=16):
    """init = (W1, b1, v, th)：完全沿用阶段1 的权重（含输出权重）"""
    rng = np.random.default_rng(seed); N = len(Xn)
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
        # 平滑 hinge: 3*tanh'(z)*(-y) 近似，仅在边界附近起作用
        dz = (-yy * (1 - np.tanh(3 * z) ** 2) * 3.0) / bs
        gv = Hq.T @ dz; gth = -dz.sum()
        dH = np.outer(dz, v) * (s * (1 - Hc ** 2))
        g = [dH.T @ xb, dH.sum(0), gv, np.array(gth)]
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 500 == 0 or t == steps:
            e = err_of(np.clip(np.round((np.tanh(Xn @ W1.T + b1) + 1) * s), 0, L - 1), v, th, y)
            if best is None or e < best[0]:
                best = (e, W1.copy(), b1.copy(), v.copy(), th)
    return best

def stage3(Xn, y, h, seed, init, L=16, epochs=8, lr=0.02, bs=8192):
    """
    软量化退火：15 个断点 c_t 可学，温度 T 逐步降。
      Hq_soft = sum_t sigmoid((Hc - c_t)/T)
    T 小时它与硬量化一致；用硬量化评估早停。
    """
    rng = np.random.default_rng(seed); N = len(Xn)
    W1, b1, v, th = init[0].copy(), init[1].copy(), init[2].copy(), init[3]
    Hc_all = np.tanh(Xn @ W1.T + b1)
    # 断点初始化：按每列的 tanh 分布等分位
    C = np.stack([np.quantile(Hc_all[:, j], np.linspace(0, 1, L + 1)[1:-1]) for j in range(h)])
    C = C.astype(np.float64)                     # (h, L-1)
    ps = [v, np.array(th), C]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    ys = 2 * y - 1
    best = (err_of(hard_q(Hc_all, C), v, th, y), C.copy(), v.copy(), th)
    step = 0
    for ep in range(epochs):
        T = 0.5 * (0.35 ** (ep / max(1, epochs - 1)))       # 0.5 -> 0.0025
        for it in range(200):
            step += 1
            idx = rng.choice(N, bs, replace=False)
            Hc = Hc_all[idx]; yy = ys[idx]
            # 软量化
            Sg = 1 / (1 + np.exp(-(Hc[:, :, None] - C[None, :, :]) / T))   # (bs,h,L-1)
            Hq = Sg.sum(axis=2) + (np.round((Hc + 1) * (L - 1) / 2) - Sg.sum(axis=2)) * 0.0
            Hq = Sg.sum(axis=2)                        # 软档号
            z = Hq @ v - th
            dz = (-yy * (1 - np.tanh(3 * z) ** 2) * 3.0) / bs
            gv = Hq.T @ dz; gth = -dz.sum()
            dHq = np.outer(dz, v)
            # dHq/dC
            dS = Sg * (1 - Sg) / T                     # (bs,h,L-1)
            gC = -np.einsum('nk,nkl->kl', dHq, dS)
            g = [gv, np.array(gth), gC]
            for i, (pp, gg) in enumerate(zip(ps, g)):
                ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
                pp -= lr * (ms[i] / (1 - 0.9 ** step)) / (np.sqrt(vs[i] / (1 - 0.999 ** step)) + 1e-8)
            # 保持断点有序
            C[:] = np.sort(C, axis=1)
        e = err_of(hard_q(Hc_all, C), v, th, y)
        if e < best[0]:
            best = (e, C.copy(), v.copy(), th)
        print(f"      ep{ep} T={T:.4f}: 硬量化错误 = {e}")
    return best

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1t = np.array(m["W1"]); b1t = np.array(m["b1"]); W2t = np.array(m["W2"]); b2t = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)
    errA = int(np.sum(((np.tanh(Xn @ W1t.T + b1t) @ W2t + b2t) >= 0).astype(np.int8) != lab))

    print("=" * 100)
    print("实验：加宽 + 专项训练（阶段3 = 软量化退火）")
    print("=" * 100)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举")
    print(f"目标标签 : N4_MLP_T1（正类 {int(y.sum())}）")
    print(f"参照     : A 原 Teacher（隐藏层连续）= {errA}/{N}")
    print(f"\n{'h':>4} | {'阶段1':>7} | {'阶段2':>7} | {'阶段3':>7} | {'最终错误率':>10} | {'用时(s)':>8}")
    print("-" * 100)
    rows = []
    for h in (16, 32, 64):
        t0 = time.time()
        r1 = None
        for seed in (0, 1):
            r = stage1(Xn, y, h, seed)
            if r1 is None or r[0] < r1[0]: r1 = r
        r2 = None
        for seed in (0, 1):
            r = stage2(Xn, y, h, seed, r1[1:])
            if r2 is None or r[0] < r2[0]: r2 = r
        e3, C3, v3, th3 = stage3(Xn, y, h, 0, r2[1:])
        el = time.time() - t0
        print(f"{h:4d} | {r1[0]:7d} | {r2[0]:7d} | {e3:7d} | {e3/N*100:9.4f}% | {el:8.0f}")
        rows.append({"h": h, "s1": r1[0], "s2": r2[0], "s3": int(e3), "time_s": el})

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    json.dump({"A": errA, "rows": rows},
              open(os.path.join(res, "width_scan_softquant.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: results/width_scan_softquant.json")

if __name__ == "__main__":
    main()

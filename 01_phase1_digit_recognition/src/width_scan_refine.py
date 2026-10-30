#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：加宽 + 专项训练（第三轮：教师初始化 + 逐列量化阈值精修）
================================================================
【实验配置】
  输入空间   (x1..x4) ∈ {0..15}^4，全部 65536 状态穷举
  目标标签   N4_MLP_T1 官方标签（4->8->1 tanh，seed 71000，正类 18436）
  评价       全部 65536 状态上的最终 0/1 错误数

  流程
    阶段1  训练一个隐藏层连续的实数 MLP（4->h->1），作为"教师"
    阶段2  以其隐藏层权重初始化，把隐藏层量化到 16 级（STE + hinge）微调
    阶段3  精细校正：对每个隐藏神经元，独立搜索它的 15 个量化阈值（16 档 → 15 个断点），
           使分类错误最小。这是一个廉价的坐标下降，直接对准"量化造成的信息损失"。
"""
import numpy as np, json, os, time

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def quantize(Hc, thresh, L=16):
    """thresh: (h, L-1) 的 15 个断点；档号 = #{t: Hc >= thresh[:,t]}"""
    return np.sum(Hc[:, :, None] >= thresh[None, :, :], axis=2).astype(np.int64)

def err_of(Hq, v, th, y):
    return int(np.sum(((Hq @ v - th) >= 0).astype(np.int64) != y))

def train_continuous(Xn, y, h, seed, steps=1500, lr=0.03, bs=8192):
    rng = np.random.default_rng(seed); N = len(Xn)
    W1 = rng.normal(0, 0.6, (h, 4)); b1 = np.zeros(h)
    v = rng.normal(0, 0.6, h); th = 0.0
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

def train_quantized(Xn, y, h, seed, init, steps=2500, lr=0.03, bs=8192, L=16):
    rng = np.random.default_rng(seed); N = len(Xn)
    W1, b1 = init[0].copy(), init[1].copy()
    v = rng.normal(0, 0.6, h); th = 0.0
    ps = [W1, b1, v, np.array(th)]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    s = (L - 1) / 2; ysign = 2 * y - 1
    best = None
    for t in range(1, steps + 1):
        idx = rng.choice(N, bs, replace=False)
        xb = Xn[idx]; yb = y[idx]; ys = ysign[idx]
        Hc = np.tanh(xb @ W1.T + b1)
        Hq = np.clip(np.round((Hc + 1) * s), 0, L - 1)
        term = (1 + np.exp(-(Hq @ v - th)))
        w = np.abs(v)
        dz = (-ys * (1 - np.tanh(Hq @ v - th) ** 2) * 3.0) / bs   # 平滑 hinge
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

def refine_thresholds(Xn, y, W1, b1, v, th, L=16, iters=4, seed=0):
    """逐列独立搜索 15 个量化阈值（坐标下降）"""
    rng = np.random.default_rng(seed)
    N = len(Xn); h = W1.shape[0]
    Hc = np.tanh(Xn @ W1.T + b1)                     # (N,h)
    # 初始阈值：把 [-1,1] 等分, 形状 (h, L-1)
    thresh = np.tile(np.linspace(-1, 1, L + 1)[1:-1], (h, 1))
    Hq = quantize(Hc, thresh, L)
    cur = err_of(Hq, v, th, y)
    for it in range(iters):
        improved = False
        for j in range(h):
            col = Hc[:, j]
            vals = np.unique(col)
            cand = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else np.array([])
            if len(cand) > 200:
                cand = np.quantile(cand, np.linspace(0, 1, 200))
            base = thresh[j].copy()
            best_local = (cur, base.copy())
            for c in cand:
                for k in range(L - 1):
                    trial = base.copy()
                    trial[k] = c
                    trial.sort()
                    th2 = thresh.copy(); th2[j] = trial
                    e = err_of(quantize(Hc, th2, L), v, th, y)
                    if e < best_local[0]:
                        best_local = (e, trial)
            if best_local[0] < cur:
                cur = best_local[0]; thresh[j] = best_local[1]; improved = True
        if not improved:
            break
    return cur, thresh

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1t = np.array(m["W1"]); b1t = np.array(m["b1"]); W2t = np.array(m["W2"]); b2t = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)
    errA = int(np.sum(((np.tanh(Xn @ W1t.T + b1t) @ W2t + b2t) >= 0).astype(np.int8) != lab))

    print("=" * 108)
    print("实验：加宽 + 专项训练（教师初始化 + 逐列量化阈值精修）")
    print("=" * 108)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举")
    print(f"目标标签 : N4_MLP_T1（正类 {int(y.sum())}）")
    print(f"参照     : A 原 Teacher（隐藏层连续）= {errA}/{N} = 0.0000%")
    print(f"\n{'h':>4} | {'阶段1 连续MLP':>14} | {'阶段2 量化微调':>14} | {'阶段3 阈值精修':>14} | {'用时(s)':>8}")
    print("-" * 108)
    rows = []
    for h in (16, 32, 64):
        t0 = time.time()
        b1r = None
        for seed in (0, 1):
            r = train_continuous(Xn, y, h, seed)
            if b1r is None or r[0] < b1r[0]:
                b1r = r
        b2r = None
        for seed in (0, 1):
            r = train_quantized(Xn, y, h, seed, (b1r[1], b1r[2]))
            if b2r is None or r[0] < b2r[0]:
                b2r = r
        e3, th3 = refine_thresholds(Xn, y, b2r[1], b2r[2], b2r[3], b2r[4], seed=0)
        el = time.time() - t0
        print(f"{h:4d} | {b1r[0]:14d} | {b2r[0]:14d} | {e3:14d} | {el:8.0f}")
        rows.append({"h": h, "stage1_continuous": b1r[0], "stage2_quantized": b2r[0],
                     "stage3_refined": int(e3), "time_s": el})
        print(f"     | {'':>14} | {'':>14} | 错误率 {e3/N*100:.4f}% |")

    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(os.path.dirname(here), "results"); os.makedirs(res, exist_ok=True)
    json.dump({"A": errA, "rows": rows},
              open(os.path.join(res, "width_scan_refine.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n已写出: results/width_scan_refine.json")

if __name__ == "__main__":
    main()

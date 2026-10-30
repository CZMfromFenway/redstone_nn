#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：b=8 拆分结构 —— 定稿版
=============================
【编码】  v = round((tanh+1)/2*255) ∈ [0,255]  =>  tanh = 2v/255 - 1
          v = 16*hi + lo,  hi,lo ∈ [0,15]

【读出】  z = sum_j W_j*tanh_j + b2
            = sum_{j,k} [ K * 2 * 16^k * W_j / 255 ] * nib_{j,k} + K*(b2 - sum_j W_j)
          其中 K 是整体尺度（把 z 放大到强模域）。

【最小二乘验证】 实测最优权重比 hi/lo = 16.0，且权重 = K*2*16^k*W_j/255，与上式一致。

【两个容器约束】
  高位满档: 15*A_hi_max <= 897      => K <= 897*255/(15*2*16*|W|max)
  低位满档: 15*A_lo_max <= 897      => K <= 897*255/(15*2*1*|W|max)   （更紧）
  两者相差 16 倍 —— 所以**低位先把容器限制住**。
"""
import numpy as np, json, os

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def encode(Hc, b=8):
    return np.clip(np.round((Hc + 1) * (2 ** b - 1) / 2), 0, 2 ** b - 1).astype(np.int64)

def split8(v):
    return np.concatenate([((v >> 4) & 0xF).astype(np.float64),
                           (v & 0xF).astype(np.float64)], axis=1)

def err(Z, w, th, y):
    return int(np.sum(((Z @ w - th) >= 0).astype(np.int64) != y))

def compile_K(W2, K, b=8, b2=0.0):
    denom = 2 ** b - 1
    a_hi = np.round(K * 2 * 16 * W2 / denom)
    a_lo = np.round(K * 2 * 1 * W2 / denom)
    return np.concatenate([a_hi, a_lo]), -K * (b2 - W2.sum()), a_hi, a_lo

def train_readout(Z, y, w0, th0, steps=2500, lr=0.05, bs=8192, seed=0, tag="r"):
    rng = np.random.default_rng(seed); N = len(Z)
    w = np.asarray(w0, float).copy(); th = float(th0)
    ms = np.zeros_like(w); vs = np.zeros_like(w); mb = vb = 0.0
    ys = 2 * y - 1
    best = (err(Z, w, th, y), w.copy(), th)
    for t in range(1, steps + 1):
        idx = rng.choice(N, bs, replace=False)
        Zb = Z[idx]; yb = y[idx]; yy = ys[idx]
        z = Zb @ w - th
        if t <= steps * 3 // 4:
            m = 1 - yy * z; dz = (-yy * (m > 0)) / bs
        else:
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30))); dz = (p - yb) / bs
        gw = Zb.T @ dz; gb = -dz.sum()
        ms = 0.9 * ms + 0.1 * gw; vs = 0.999 * vs + 0.001 * gw ** 2
        w -= lr * (ms / (1 - 0.9 ** t)) / (np.sqrt(vs / (1 - 0.999 ** t)) + 1e-8)
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb ** 2
        th -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == steps:
            e = err(Z, w, th, y)
            if e < best[0]:
                best = (e, w.copy(), th)
    return best

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1 = np.array(m["W1"]); b1 = np.array(m["b1"])
    W2 = np.array(m["W2"]); b2 = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)
    Hc = np.tanh(Xn @ W1.T + b1)
    errA = int(np.sum(((Hc @ W2 + b2) >= 0).astype(np.int8) != lab))
    v = encode(Hc, 8); Z = split8(v)
    wmax = np.abs(W2).max(); jm = int(np.argmax(np.abs(W2)))

    print("=" * 100, flush=True)
    print("实验：b=8 拆分结构（定稿）", flush=True)
    print("=" * 100, flush=True)
    print(f"参照 A 原 Teacher = {errA}/{N}", flush=True)
    print(f"|W2|max = {wmax:.6f} (j={jm})", flush=True)

    K_hi = 897 * 255 / (15 * 2 * 16 * wmax)
    K_lo = 897 * 255 / (15 * 2 * 1 * wmax)
    print(f"\n[两个容器约束给出的 K 上限]", flush=True)
    print(f"   高位满档 15*A_hi<=897 => K <= {K_hi:.0f}", flush=True)
    print(f"   低位满档 15*A_lo<=897 => K <= {K_lo:.0f}   <-- 更紧", flush=True)
    print(f"   => K_max = {K_lo:.0f}", flush=True)
    print(f"\n   K={K_hi:.0f} 时: A_hi_max={K_hi*2*16*wmax/255:.1f}(满档 {15*K_hi*2*16*wmax/255:.0f}), "
          f"A_lo_max={K_hi*2*wmax/255:.1f}(满档 {15*K_hi*2*wmax/255:.0f})", flush=True)
    print(f"   K={K_lo:.0f} 时: A_hi_max={K_lo*2*16*wmax/255:.1f}(满档 {15*K_lo*2*16*wmax/255:.0f}), "
          f"A_lo_max={K_lo*2*wmax/255:.1f}(满档 {15*K_lo*2*wmax/255:.0f})", flush=True)

    print(f"\n[K 扫描]", flush=True)
    print(f"   {'K':>8} | {'A_hi_max':>8} | {'A_lo_max':>8} | {'hi满档':>7} | {'lo满档':>7} | "
          f"{'越界':>4} | {'编译错误':>8} | {'训练后':>7} | {'错误率':>9}", flush=True)
    rows = []
    for K in (58, 117, 235, 470, int(K_hi), int(K_lo)):
        w0, th0, ah, al = compile_K(W2, K, 8, b2)
        ahm, alm = np.abs(ah).max(), np.abs(al).max()
        over = "是" if (15 * ahm > 897 or 15 * alm > 897) else "否"
        e0 = err(Z, w0, th0, y)
        e1, _, _ = train_readout(Z, y, w0, th0, steps=2500, seed=0, tag=f"K{K}")
        print(f"   {K:8d} | {ahm:8.1f} | {alm:8.1f} | {15*ahm:7.0f} | {15*alm:7.0f} | {over:>4} | "
              f"{e0:8d} | {e1:7d} | {e1/N*100:8.4f}%", flush=True)
        rows.append({"K": K, "a_hi_max": float(ahm), "a_lo_max": float(alm),
                     "over": over == "是", "compile_err": int(e0), "train_err": int(e1)})

    Kb = min(rows, key=lambda r: r["train_err"])["K"]
    w0, th0, ah, al = compile_K(W2, Kb, 8, b2)
    print(f"\n[最优 K={Kb} 的权重表]", flush=True)
    print(f"   {'j':>3} | {'W2_j':>10} | {'A_hi':>8} | {'A_lo':>8} | {'比值':>6} | "
          f"{'hi满档':>8} | {'lo满档':>8}", flush=True)
    for j in range(8):
        r = (ah[j] / al[j]) if al[j] else None
        print(f"   {j:3d} | {W2[j]:+10.6f} | {ah[j]:8.1f} | {al[j]:8.1f} | "
              f"{(f'{r:6.1f}' if r else '   -- ')} | {15*ah[j]:8.0f} | {15*al[j]:8.0f}", flush=True)
    print(f"   阈值 th = {th0:.1f}", flush=True)

    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump({"A": errA, "N": N, "K_hi_limit": float(K_hi), "K_lo_limit": float(K_lo), "rows": rows},
              open(os.path.join(resdir, "split_b8_final.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/split_b8_final.json", flush=True)

if __name__ == "__main__":
    main()

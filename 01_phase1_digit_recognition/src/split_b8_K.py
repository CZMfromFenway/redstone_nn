#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：b=8 拆分结构，按 czm 的 0xEF 例子确定权重分配
=====================================================
【czm 的算式】  W*x = hi*840 + lo*52   （0xEF: hi=14, lo=15）
  唯一自洽的解释：整体尺度 S 使
      lambda_hi 的步长（最大 |W| 那一路）= 840/15 = 56     => 满档 840
      lambda_lo 的步长                  = 56/16 = 3.5     => 满档 52
  即 lambda_hi 承担 W*16 的刻度，lambda_lo 承担 W*1 的刻度，步长比 16。

【实现】
  设整体尺度为 K（相对 W2 的倍数）。
    A_hi_j = K * 16 * W2_j / 255      （高位每档贡献）
    A_lo_j = K *  1 * W2_j / 255      （低位每档贡献）
  取 K 使 max|A_hi| * 15 = 868（总跨度用满容器）
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
    """返回 (N,16)，列序 [hi 的 8 路, lo 的 8 路]"""
    return np.concatenate([((v >> 4) & 0xF).astype(np.float64),
                           (v & 0xF).astype(np.float64)], axis=1)

def err(Z, w, th, y):
    return int(np.sum(((Z @ w - th) >= 0).astype(np.int64) != y))

def compile_K(W2, K, b=8, b2=0.0):
    """
    A_hi_j = K*16*W2_j/255,  A_lo_j = K*1*W2_j/255
    阈值 th = -K*(b2 - sum W2)
    返回 (w, th, A_hi, A_lo)
    """
    denom = 2 ** b - 1
    a_hi = np.round(K * 16 * W2 / denom)
    a_lo = np.round(K * 1 * W2 / denom)
    w = np.concatenate([a_hi, a_lo])
    th = -K * (b2 - W2.sum())
    return w, th, a_hi, a_lo

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
            print(f"      [{tag}] step {t:5d}: 错误={e:6d} 最好={best[0]}", flush=True)
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
    wmax = np.abs(W2).max(); jmax = int(np.argmax(np.abs(W2)))

    print("=" * 100, flush=True)
    print("实验：b=8 拆分，按「高位步长：满档 840 / 低位步长：满档 52」分配", flush=True)
    print("=" * 100, flush=True)
    print(f"参照 A 原 Teacher = {errA}/{N}", flush=True)
    print(f"|W2|max = {wmax:.6f} (j={jmax}),  W2[{jmax}] = {W2[jmax]:+.6f}", flush=True)

    # K 由 max|A_hi|*15 = 868 定
    K = 868 * (2 ** 8 - 1) / (15 * 16 * wmax)
    print(f"\n[整体尺度 K]", flush=True)
    print(f"   A_hi_max*15 = 868  =>  K = 868*255/(15*16*{wmax:.4f}) = {K:.1f}", flush=True)
    a_hi_max = K * 16 * wmax / 255
    print(f"   => A_hi_max = {a_hi_max:.2f} (步长), 满档 = {15*a_hi_max:.0f}", flush=True)
    print(f"   => A_lo_max = {a_hi_max/16:.2f} (步长), 满档 = {15*a_hi_max/16:.0f}", flush=True)
    print(f"   与 czm 例子对照: 高位满档应 ≈840, 低位满档应 ≈52  ✓", flush=True)

    print(f"\n[K 扫描]", flush=True)
    print(f"   {'K':>8} | {'A_hi_max':>8} | {'A_lo_max':>8} | {'hi满档':>7} | {'lo满档':>7} | "
          f"{'越界':>4} | {'编译错误':>8} | {'训练后':>7} | {'错误率':>9}", flush=True)
    rows = []
    for K in (0.25 * K, 0.5 * K, K, 1.5 * K, 2 * K):
        w0, th0, a_hi, a_lo = compile_K(W2, K, 8, b2)
        ah, al = np.abs(a_hi).max(), np.abs(a_lo).max()
        over = "是" if (ah > 868 or al > 868 or 15 * ah > 897) else "否"
        e0 = err(Z, w0, th0, y)
        e1, _, _ = train_readout(Z, y, w0, th0, steps=2500, seed=0, tag=f"K{K:.0f}")
        print(f"   {K:8.0f} | {ah:8.1f} | {al:8.1f} | {15*ah:7.0f} | {15*al:7.0f} | {over:>4} | "
              f"{e0:8d} | {e1:7d} | {e1/N*100:8.4f}%", flush=True)
        rows.append({"K": float(K), "a_hi_max": float(ah), "a_lo_max": float(al),
                     "over": over == "是", "compile_err": int(e0), "train_err": int(e1)})

    # 展示 K 处的权重表
    print(f"\n[K={K:.0f} 时的权重表]", flush=True)
    w0, th0, a_hi, a_lo = compile_K(W2, K, 8, b2)
    print(f"   {'j':>3} | {'W2_j':>10} | {'A_hi(每档)':>11} | {'A_lo(每档)':>11} | "
          f"{'hi满档':>8} | {'lo满档':>8}", flush=True)
    for j in range(8):
        print(f"   {j:3d} | {W2[j]:+10.6f} | {a_hi[j]:11.1f} | {a_lo[j]:11.1f} | "
              f"{15*a_hi[j]:8.1f} | {15*a_lo[j]:8.1f}", flush=True)
    print(f"   比值 A_hi/A_lo = {[round(a/b,1) if b else None for a,b in zip(a_hi,a_lo)]}", flush=True)
    print(f"   阈值 th = {th0:.1f}", flush=True)

    best = min(rows, key=lambda r: r["train_err"])
    print(f"\n[最优] K={best['K']:.0f}, 错误={best['train_err']}/{N} "
          f"({best['train_err']/N*100:.4f}%)", flush=True)

    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump({"A": errA, "N": N, "K_opt": float(K), "rows": rows},
              open(os.path.join(resdir, "split_b8_K.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/split_b8_K.json", flush=True)

if __name__ == "__main__":
    main()

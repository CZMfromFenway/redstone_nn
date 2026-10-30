#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReLU 离散化（按 czm 的规则，最终版）
=====================================
【真实坐标系 -> 整数坐标系】
  某神经元: u_j = x.w + b_j,  真实范围 [u_min, u_max]
  按整体尺度 K 线性映射到整数坐标系 Lambda: Lambda = K*u  (取 K 使整数坐标非负)
  转折点在 u=0, 即 Lambda_kink = -K*b_j  ... 等价地 u=-b_j 处

  czm 的例子: u in [-3,6] -> Lambda in [0,24000], K=4000/1 => 实际上 K=8000/3? 
  校验: u=-3 -> Lambda=0, u=0 -> Lambda=8000, u=6 -> Lambda=24000
        线性: Lambda = (8000/3)*(u+3) = 2666.7u + 8000
        即 K=2666.7, 偏移 +8000
        u=-b_j = 0 时 Lambda=8000  ✓

【编码规则】
  死区 [Lambda=0, Lambda_kink] 整段压成 v=0, 不占编码空间
  活跃区间 [Lambda_kink, Lambda_max] 映射到 [0, 255]:
      v = clamp(round(lv*(Lambda - Lambda_kink)/(Lambda_max - Lambda_kink)), 0, lv)

【第二层】
  逐通道缩放因子不同 -> 所有 W 都被重新缩放, 由最小二乘吸收。

【物理约束】
  读出权重即 lambda 的步长系数, 要求 |w| <= 897 (容器上限)。
"""
import numpy as np, json, os

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def train_relu(Xn, y, h=8, seed=0, steps=8000, lr=0.02, bs=8192, log_every=4000):
    rng = np.random.default_rng(seed); N = len(Xn)
    W1 = rng.normal(0, 0.6, (h, 4)); b1 = np.zeros(h)
    W2 = rng.normal(0, 0.6, h); b2 = 0.0
    ps = [W1, b1, W2, np.array(b2)]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    best = None
    for t in range(1, steps + 1):
        idx = rng.choice(N, bs, replace=False)
        xb = Xn[idx]; yb = y[idx]
        U = xb @ W1.T + b1; H = np.maximum(U, 0); z = H @ W2 + b2
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30))); dz = (p - yb) / bs
        dU = np.outer(dz, W2) * (U > 0)
        g = [dU.T @ xb, dU.sum(0), H.T @ dz, np.array(dz.sum())]
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % log_every == 0 or t == steps:
            Hf = np.maximum(Xn @ W1.T + b1, 0)
            e = int(np.sum(((Hf @ W2 + b2) >= 0).astype(np.int64) != y))
            if best is None or e < best[0]:
                best = (e, W1.copy(), b1.copy(), W2.copy(), float(b2))
            print(f"      [relu s{seed}] step {t}: 错误={e} 最好={best[0]}", flush=True)
    return best

def build_K(U, W1, target_lo=0.0, target_hi=24000.0):
    """
    按 czm 的口径选整体尺度 K 与偏移:
      U=-3 -> 0, U=0 -> 8000, U=6 -> 24000  =>  K = 24000/9, 偏移 = 8000
    这里对每个真实神经元用其自身范围: U in [u_min,u_max] ->
      Lambda = K*(U - u_min),  取 K = 24000/(u_max-u_min)  (使最大映射到 24000)
      转折点 Lambda_kink = K*(0 - u_min) = -K*u_min
    """
    h = U.shape[1]
    us = U.min(axis=0); ux = U.max(axis=0)
    K = np.where(ux - us > 1e-12, (target_hi - target_lo) / (ux - us), 1.0)
    kink = K * (0.0 - us)                     # 转折点在整数坐标中的位置
    return K, us, ux, kink

def encode_active(U, K, us, ux, bits):
    """死区压成 0; 活跃区间 [转折点, 上限] -> [0, lv]"""
    N, h = U.shape
    lv = (1 << bits) - 1
    Lam = K[None, :] * (U - us[None, :])
    kink = K * (0.0 - us)
    hi = K * (ux - us)
    span = np.where(hi - kink > 1e-12, hi - kink, 1.0)
    v = np.round(lv * (Lam - kink[None, :]) / span[None, :])
    v = np.clip(v, 0, lv).astype(np.int64)          # 死区 -> 0
    if bits == 4:
        return v.astype(float), 1
    nch = bits // 4
    Z = np.concatenate([((v >> (4 * k)) & 0xF) for k in range(nch)], axis=1).astype(float)
    return Z, nch

def encode_full(U, K, us, ux, bits):
    """对照: 整个区间 [min,max] -> [0,lv] (死区占空间)"""
    N, h = U.shape
    lv = (1 << bits) - 1
    Lam = K[None, :] * (U - us[None, :])
    hi = K * (ux - us)
    v = np.clip(np.round(lv * Lam / np.where(hi > 1e-12, hi, 1.0)[None, :]), 0, lv).astype(np.int64)
    if bits == 4:
        return v.astype(float), 1
    nch = bits // 4
    Z = np.concatenate([((v >> (4 * k)) & 0xF) for k in range(nch)], axis=1).astype(float)
    return Z, nch

def fit_int(Z, target, S):
    A = np.hstack([Z, np.ones((len(Z), 1))])
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    return np.round(coef[:-1] * S).astype(np.int64), -round(coef[-1] * S)

def best_err(Z, target, yR, cap=897):
    """在 |w|<=cap 的约束下找最优 S"""
    b = None
    for S in (100, 200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000):
        w, th = fit_int(Z, target, S)
        wm = int(np.abs(w).max())
        e = int(np.sum(((Z @ w - th) >= 0).astype(np.int64) != yR))
        if wm <= cap and (b is None or e < b[0]):
            b = (e, S, wm)
    if b is None:                       # 全部越界, 报最小错误作参考
        for S in (100, 200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000):
            w, th = fit_int(Z, target, S)
            e = int(np.sum(((Z @ w - th) >= 0).astype(np.int64) != yR))
            if b is None or e < b[0]:
                b = (e, S, int(np.abs(w).max()))
        b = (b[0], b[1], b[2])
    return b

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)

    print("=" * 100, flush=True)
    print("ReLU 离散化（按 czm 规则）：死区 [0,kink] 压成 0，活跃段 -> 0~255", flush=True)
    print("=" * 100, flush=True)

    best = None
    for s in (0, 1, 2, 3, 4):
        r = train_relu(Xn, y, 8, seed=s, steps=8000)
        if best is None or r[0] < best[0]:
            best = r
    _, W1r, b1r, W2r, b2r = best
    W1r = np.asarray(W1r, float); b1r = np.asarray(b1r, float).reshape(-1)
    W2r = np.asarray(W2r, float).reshape(-1); b2r = float(b2r)
    U = Xn @ W1r.T + b1r
    H = np.maximum(U, 0)
    yR = ((H @ W2r + b2r) >= 0).astype(np.int64)
    print(f"\nReLU 老师拟合官方标签 = {best[0]}/{N} ({best[0]/N*100:.4f}%)", flush=True)
    print(f"真实预激活 u 的范围 = [{U.min():.3f}, {U.max():.3f}]", flush=True)

    K, us, ux, kink = build_K(U, W1r)
    print(f"\n[按 czm 口径的整数坐标]  整体尺度 K 使最大映射到 24000", flush=True)
    print(f"   {'j':>3} | {'u_min':>8} | {'u_max':>8} | {'K':>9} | {'kink(=K*(0-u_min))':>19} | "
          f"{'hi':>8} | {'活跃区间宽':>10} | {'全区间宽':>9}", flush=True)
    for j in range(8):
        print(f"   {j:3d} | {us[j]:8.3f} | {ux[j]:8.3f} | {K[j]:9.1f} | {kink[j]:19.0f} | "
              f"{K[j]*(ux[j]-us[j]):8.0f} | {K[j]*ux[j]:10.0f} | {K[j]*(ux[j]-us[j]):9.0f}",
              flush=True)
    print(f"   例: j=0 若 u_min=-3,u_max=6 -> K=2666.7, kink=8000, hi=24000", flush=True)

    print(f"\n[实测] 第二层整数读出（|w| <= 897 容器约束下选最优 S）", flush=True)
    print(f"   {'编码':>28} | {'弱模线':>6} | {'S':>7} | {'权重max':>8} | {'错误':>7} | {'错误率':>9}",
          flush=True)
    print("-" * 92, flush=True)
    out = {"relu_teacher": int(best[0]), "N": N}
    for nm, bits, enc in (
            ("4bit 全区间(死区占空间)", 4, encode_full),
            ("4bit 活跃区间(死区不占)", 4, encode_active),
            ("8bit 全区间(拆分)", 8, encode_full),
            ("8bit 活跃区间(死区不占)", 8, encode_active)):
        Z, nch = enc(U, K, us, ux, bits)
        e, S, wm = best_err(Z, H @ W2r + b2r, yR)
        print(f"   {nm:>28} | {nch*8:6d} | {S:7d} | {wm:8d} | {e:7d} | {e/N*100:8.4f}%", flush=True)
        out[nm] = {"err": int(e), "S": int(S), "wmax": int(wm), "channels": int(nch * 8)}

    print(f"\n   对照: tanh 老师(官方) 4bit ≈1005 (1.534%), 8bit拆分 75 (0.1144%)", flush=True)
    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump(out, open(os.path.join(resdir, "relu_active.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/relu_active.json", flush=True)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReLU 编码（按 czm 规则，用真正的 Lambda 最值）
===============================================
【真正的整数坐标】
  Lambda_j = sum_i lambda_{j,i}(x_i)     <- 4 张 LUT 之和（不含常数/尺度）
  它的真实最值由 LUT 步长决定，各神经元不同（实测 25886~42809），不是固定的 24000。

【转折点】
  u_j = Lambda_j*sc_j - sum_i W1_ji + b1_j
  u_j = 0  <=>  Lambda_kink_j = (sum_i W1_ji - b1_j)/sc_j
  实测 kink 落在各区间的 21%~64% 处。

【编码规则（czm）】
  死区 [Lambda_min, kink] 整段压成 v=0，不占编码空间
  活跃段 [kink, Lambda_max] 映射到 [0, lv]:
      v = clamp(round(lv*(Lambda-kink)/(Lambda_max-kink)), 0, lv)

【物理约束】读出权重即 lambda 步长系数，|w| <= 897。
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

def build_Q(X, W1):
    """
    第一层 LUT 和: Q_j = sum_i |W1_ji| * lambda_{j,i}(x_i)
    **lambda 全部为非负幅值上升阶梯**（lambda>=0 是硬约束），
    符号不进入 Q，完全交给第二层的整数权重承担。
    故 Q_j >= 0，且 Q_min = 0（所有 x_i=0 时）。
    返回 Q (N,h), rho, up
    """
    W1 = np.asarray(W1, float)
    h = W1.shape[0]
    rho = np.zeros((h, 4))
    for j in range(h):
        mx = np.abs(W1[j]).max()
        if mx > 1e-12:
            rho[j] = 868.0 * np.abs(W1[j]) / mx
    x = np.arange(16)
    up = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
    N = len(X); Xi = X.astype(np.int64)
    Q = np.zeros((N, h), dtype=np.int64)
    for j in range(h):
        if np.abs(W1[j]).max() <= 1e-12:
            continue
        for i in range(4):
            if abs(W1[j, i]) <= 1e-12:
                continue
            Q[:, j] += up[j, i, Xi[:, i]]        # 幅值, 非负
    return Q, rho, up

def sc_of(W1):
    W1 = np.asarray(W1, float)
    return np.array([(np.abs(W1[j]).max() / (868 * 7.5)) if np.abs(W1[j]).max() > 1e-12 else 0.0
                     for j in range(W1.shape[0])])

def encode(Q, kink, hi, bits, dead_to_zero=True):
    """
    死区压成 0; 活跃段 [kink, hi] -> [0, lv]
      dead_to_zero=True : v = clamp(round(lv*(Q-kink)/(hi-kink)), 0, lv)
      dead_to_zero=False: 全区间对照 v = clamp(round(lv*(Q-Qmin)/(hi-Qmin)),0,lv)
    """
    N, h = Q.shape
    lv = (1 << bits) - 1
    if dead_to_zero:
        span = np.where(hi - kink > 1e-12, hi - kink, 1.0)
        v = np.round(lv * (Q - kink[None, :]) / span[None, :])
    else:
        qmin = Q.min(axis=0)
        span = np.where(hi - qmin > 1e-12, hi - qmin, 1.0)
        v = np.round(lv * (Q - qmin[None, :]) / span[None, :])
    v = np.clip(v, 0, lv).astype(np.int64)
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
    b = None
    for S in (50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000, 32000):
        w, th = fit_int(Z, target, S)
        wm = int(np.abs(w).max())
        e = int(np.sum(((Z @ w - th) >= 0).astype(np.int64) != yR))
        if wm <= cap and (b is None or e < b[0]):
            b = (e, S, wm)
    return b

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)

    print("=" * 104, flush=True)
    print("ReLU 编码（真正 Lambda 最值 + 死区压成 0）", flush=True)
    print("=" * 104, flush=True)

    best = None
    for s in (0, 1, 2, 3, 4):
        r = train_relu(Xn, y, 8, seed=s, steps=8000)
        if best is None or r[0] < best[0]:
            best = r
    _, W1r, b1r, W2r, b2r = best
    W1r = np.asarray(W1r, float); b1r = np.asarray(b1r, float).reshape(-1)
    W2r = np.asarray(W2r, float).reshape(-1); b2r = float(b2r)
    print(f"\nReLU 老师拟合官方标签 = {best[0]}/{N} ({best[0]/N*100:.4f}%)", flush=True)

    Q, rho, up = build_Q(X, W1r)
    sc = sc_of(W1r)
    Qmin = Q.min(axis=0); Qmax = Q.max(axis=0)
    kink = (W1r.sum(axis=1) - b1r) / sc              # u=0 处的 Q
    print(f"\n[真正的 Lambda 最值]  Q_j = sum_i lambda_ji(x_i)", flush=True)
    print(f"   {'j':>3} | {'Q_min':>9} | {'Q_max':>9} | {'真宽':>8} | {'kink':>10} | "
          f"{'kink位置':>9} | {'活跃宽':>8} | {'死区占比':>9}", flush=True)
    for j in range(8):
        act = Qmax[j] - kink[j]
        dead = (kink[j] - Qmin[j]) / (Qmax[j] - Qmin[j]) * 100
        print(f"   {j:3d} | {Qmin[j]:9d} | {Qmax[j]:9d} | {Qmax[j]-Qmin[j]:8d} | "
              f"{kink[j]:10.1f} | {(kink[j]-Qmin[j])/(Qmax[j]-Qmin[j])*100:8.1f}% | "
              f"{act:8.1f} | {dead:8.1f}%", flush=True)
    print(f"   => 真实宽度 {int((Qmax-Qmin).min())}~{int((Qmax-Qmin).max())}，"
          f"远不是 24000", flush=True)

    # 目标: 用真实 u 算 z (自洽)
    U = Q * sc[None, :] - W1r.sum(axis=1)[None, :] + b1r[None, :]
    H = np.maximum(U, 0)
    yR = ((H @ W2r + b2r) >= 0).astype(np.int64)
    tgt = H @ W2r + b2r
    e1 = int(np.sum(((tgt >= 0).astype(np.int64) != yR)))
    print(f"\n第一层 LUT 后仅预激活错误 = {e1}/{N}", flush=True)

    print(f"\n[实测] 第二层整数读出（|w| <= 897）", flush=True)
    print(f"   {'编码':>34} | {'弱模线':>6} | {'S':>6} | {'权重max':>8} | {'错误':>7} | {'错误率':>9}",
          flush=True)
    print("-" * 100, flush=True)
    out = {"relu_teacher": int(best[0]), "N": N,
           "Q_ranges": [[int(Qmin[j]), int(Qmax[j])] for j in range(8)],
           "kink": [float(kink[j]) for j in range(8)]}
    for nm, bits, dz in (
            ("4bit 全区间(死区占空间)", 4, False),
            ("4bit 死区压成0(活跃段->255)", 4, True),
            ("8bit 全区间(死区占空间,拆分)", 8, False),
            ("8bit 死区压成0(活跃段->255,拆分)", 8, True)):
        Z, nch = encode(Q, kink, Qmax, bits, dead_to_zero=dz)
        r = best_err(Z, tgt, yR)
        if r is None:
            print(f"   {nm:>34} | {nch*8:6d} | {'--':>6} | {'--':>8} | {'--':>7} | 全部越界", flush=True)
            continue
        e, S, wm = r
        print(f"   {nm:>34} | {nch*8:6d} | {S:6d} | {wm:8d} | {e:7d} | {e/N*100:8.4f}%", flush=True)
        out[nm] = {"err": int(e), "S": int(S), "wmax": int(wm), "channels": int(nch * 8)}

    print(f"\n   对照 tanh: 4bit ≈1005 (1.534%), 8bit拆分 75 (0.1144%)", flush=True)
    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump(out, open(os.path.join(resdir, "relu_trueQ.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/relu_trueQ.json", flush=True)

if __name__ == "__main__":
    main()

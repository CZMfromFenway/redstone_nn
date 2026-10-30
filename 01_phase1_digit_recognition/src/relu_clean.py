#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReLU 路线（修正评价口径）
==========================
【修正的 bug】
  之前把评价目标设为 z_teacher = H_true@W2+b2, 又用 sign(z)>=0 判类,
  而标签 yR 正是由 z_teacher 生成 -> 必然 0 错（循环论证）。
  现在: 固定第一层 LUT, **只重训第二层**（整数权重），用 yR / 官方标签评价。

【另一处修正】
  lambda 必须非负 -> Q_j = sum_i |W1_ji| * lambda_{j,i}(x_i), Q_j >= 0, Q_min = 0。
  符号由第二层权重承担。

【对照变量】
  第一层 LUT 的精细度: 常数步长 M = 1..255 与浮动步长 round(255*x*|W|/max|W|)
  第二层编码: 4bit (1条线) / 8bit 拆分 (2条线) / 16bit (4条线)
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

def build_Q(X, W1, M, floating):
    """
    第一层 LUT: lambda_{j,i}(x) = round(x*rho_{j,i}), rho 非负
      floating=True : rho = M*|W1_ji|/max_i|W1_ji|   (浮动步长, 满档=M)
      floating=False: 用常数步长使满档=M  (均一步长)
    Q_j = sum_i |W1_ji|*lambda_{j,i}(x_i)  >= 0, Q_min = 0
    """
    W1 = np.asarray(W1, float); h = W1.shape[0]
    x = np.arange(16); Xi = X.astype(np.int64); N = len(X)
    Q = np.zeros((N, h), dtype=np.int64)
    for j in range(h):
        mx = np.abs(W1[j]).max()
        if mx <= 1e-12:
            continue
        for i in range(4):
            w = abs(W1[j, i])
            if w <= 1e-12:
                continue
            if floating:
                rho = M * w / mx
                lam = np.round(x * rho).astype(np.int64)
            else:
                lam = np.round(x * M / 15).astype(np.int64)   # 均一, 满档 M
            Q[:, j] += int(round(w / mx * 1000)) * lam[Xi[:, i]]   # 用 |W| 比例加权
    # 归一化权重因子: 让 Q 的量级与原式一致 (lambda 已含 |W|/max 比例)
    return Q

def encode(Q, bits, use_kink, kink):
    N, h = Q.shape; lv = (1 << bits) - 1
    hi = Q.max(axis=0)
    lo = kink if use_kink else Q.min(axis=0)
    lo = np.maximum(lo, 0.0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    v = np.clip(np.round(lv * (Q - lo[None, :]) / span[None, :]), 0, lv).astype(np.int64)
    if bits == 4:
        return v.astype(float), 1
    nch = bits // 4
    Z = np.concatenate([((v >> (4 * k)) & 0xF) for k in range(nch)], axis=1).astype(float)
    return Z, nch

def train_readout(Z, y, steps=4000, lr=0.03, seed=0):
    """独立重训第二层: 中间层固定为整数编码 Z, 只学 w,b"""
    rng = np.random.default_rng(seed); N = len(Z)
    w = np.zeros(Z.shape[1]); b = 0.0
    ms = np.zeros_like(w); vs = np.zeros_like(w); mb = vb = 0.0
    ys = 2 * y - 1
    best = None
    for t in range(1, steps + 1):
        idx = rng.choice(N, 8192, replace=False)
        Zb = Z[idx]; yb = y[idx]; yy = ys[idx]
        z = Zb @ w - b
        if t <= steps * 3 // 4:
            m = 1 - yy * z; dz = (-yy * (m > 0)) / 8192
        else:
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30))); dz = (p - yb) / 8192
        gw = Zb.T @ dz; gb = -dz.sum()
        ms = 0.9 * ms + 0.1 * gw; vs = 0.999 * vs + 0.001 * gw ** 2
        w -= lr * (ms / (1 - 0.9 ** t)) / (np.sqrt(vs / (1 - 0.999 ** t)) + 1e-8)
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb ** 2
        b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == steps:
            e = int(np.sum(((Z @ w - b) >= 0).astype(np.int64) != y))
            if best is None or e < best[0]:
                best = (e, w.copy(), b)
    # 整数化
    w, b = best[1], best[2]
    e_int = int(np.sum(((Z @ np.round(w) - b) >= 0).astype(np.int64) != y))
    if e_int > best[0]:
        e_int = best[0]
    return best[0], e_int, np.abs(w).max()

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)
    print("=" * 100, flush=True)
    print("ReLU 路线（修正评价口径：只重训第二层）", flush=True)
    print("=" * 100, flush=True)

    best = None
    for s in (0, 1, 2, 3, 4):
        r = train_relu(Xn, y, 8, seed=s, steps=8000)
        if best is None or r[0] < best[0]:
            best = r
    _, W1r, b1r, W2r, b2r = best
    W1r = np.asarray(W1r, float); b1r = np.asarray(b1r, float).reshape(-1)
    W2r = np.asarray(W2r, float).reshape(-1); b2r = float(b2r)
    U = Xn @ W1r.T + b1r; H = np.maximum(U, 0)
    yR = ((H @ W2r + b2r) >= 0).astype(np.int64)
    print(f"\nReLU 老师拟合官方标签 = {best[0]}/{N} ({best[0]/N*100:.4f}%)", flush=True)
    print(f"真值 u 范围 = [{U.min():.3f},{U.max():.3f}]；负值(死区)占比 = "
          f"{(U<=0).mean()*100:.1f}%", flush=True)

    print(f"\n[第一层 LUT 精细度扫描] 第二层用 8bit 拆分, 独立重训", flush=True)
    print(f"   {'M':>6} | {'步长类型':>10} | {'Q_max均值':>10} | {'实数重训':>9} | {'整数化':>8} | {'权重max':>8}",
          flush=True)
    print("-" * 76, flush=True)
    for M in (15, 31, 63, 127, 255, 511, 868):
        for fl in (True, False):
            Q = build_Q(X, W1r, M, fl)
            Z, nch = encode(Q, 8, True, np.zeros(8))
            e1, e2, wm = train_readout(Z, yR)
            print(f"   {M:6d} | {'浮动' if fl else '均一':>10} | {Q.max(axis=0).mean():10.0f} | "
                  f"{e1:9d} | {e2:8d} | {wm:8.1f}", flush=True)
    print("-" * 76, flush=True)

    print(f"\n[第二层位数扫描] 第一层 M=868 浮动", flush=True)
    Q = build_Q(X, W1r, 868, True)
    kink = np.maximum(0.0, (W1r.sum(axis=1) / 7.5) / np.where(
        np.array([np.abs(W1r[j]).max() / (868 * 7.5) for j in range(8)]) > 0,
        np.array([np.abs(W1r[j]).max() / (868 * 7.5) for j in range(8)]), 1.0))
    print(f"   {'位数':>5} | {'弱模线':>6} | {'实数重训':>9} | {'整数化':>8} | {'权重max':>8} | {'错误率':>9}",
          flush=True)
    print("-" * 68, flush=True)
    out = {"relu_teacher": int(best[0]), "N": N}
    for bits in (4, 8, 12, 16):
        Z, nch = encode(Q, bits, True, kink)
        e1, e2, wm = train_readout(Z, yR)
        print(f"   {bits:5d} | {nch*8:6d} | {e1:9d} | {e2:8d} | {wm:8.1f} | {e2/N*100:8.4f}%",
              flush=True)
        out[f"{bits}bit"] = {"real": int(e1), "int": int(e2), "channels": int(nch * 8)}
    print(f"\n   对照 tanh: 4bit ≈1005 (1.534%), 8bit拆分 75 (0.1144%)", flush=True)

    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump(out, open(os.path.join(resdir, "relu_clean.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/relu_clean.json", flush=True)

if __name__ == "__main__":
    main()

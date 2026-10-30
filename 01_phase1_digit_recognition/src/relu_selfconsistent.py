#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReLU 自洽实验：ReLU 老师定义真值表，强模 LUT 复现它
=====================================================
【自洽设定】ReLU 老师 = 4->8->1, 隐藏层连续, 它自己生成真值表。
            评价标准 = 强模 LUT 复现"该 ReLU 老师输出"的错误数。

【两个编译版本】
  V1  中间层 4 bit/神经元  -> 8 条弱模线
  V2  中间层 8 bit/神经元  -> 16 条弱模线 (拆分高/低半字节)

【编译办法】
  第一层 4->8: 强模 LUT。坐标 i 的步长 rho=868*|W1_ji|/max_i|W1_ji|
               lambda 用幅值上升阶梯, 符号交给输出层
  第二层: tanh 用 v=round((tanh+1)/2*255) 编码;
          ReLU 用 v=round(relu*scale) 编码 —— 线性, 不需要查表
  读出: 最小二乘后按尺度取整 (整数权重)
"""
import numpy as np, json, os

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def train(Xn, y, h, act, seed=0, steps=6000, lr=0.02, bs=8192, log_every=2000, tag="t",
          y_true=None):
    rng = np.random.default_rng(seed); N = len(Xn)
    ev = y if y_true is None else y_true
    W1 = rng.normal(0, 0.6, (h, 4)); b1 = np.zeros(h)
    W2 = rng.normal(0, 0.6, h); b2 = 0.0
    ps = [W1, b1, W2, np.array(b2)]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    def hid(x):
        U = x @ W1.T + b1
        return (np.tanh(U) if act == "tanh" else np.maximum(U, 0))
    best = None
    for t in range(1, steps + 1):
        idx = rng.choice(N, bs, replace=False)
        xb = Xn[idx]; yb = y[idx]
        U = xb @ W1.T + b1
        H = np.tanh(U) if act == "tanh" else np.maximum(U, 0)
        z = H @ W2 + b2
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30))); dz = (p - yb) / bs
        dU = np.outer(dz, W2) * ((1 - H ** 2) if act == "tanh" else (U > 0))
        g = [dU.T @ xb, dU.sum(0), H.T @ dz, np.array(dz.sum())]
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % log_every == 0 or t == steps:
            Hf = hid(Xn)
            e = int(np.sum(((Hf @ W2 + b2) >= 0).astype(np.int8) != ev))
            if best is None or e < best[0]:
                best = (e, W1.copy(), b1.copy(), W2.copy(), float(b2))
            print(f"      [{tag}] step {t}: 错误={e} 最好={best[0]}", flush=True)
    return best

def lut_lambda(W1, cap=868.0):
    h = W1.shape[0]
    rho = np.zeros((h, 4))
    for j in range(h):
        mx = np.abs(W1[j]).max()
        if mx > 1e-12:
            rho[j] = cap * np.abs(W1[j]) / mx
    x = np.arange(16)
    up = np.round(x[None, None, :] * rho[:, :, None]).astype(np.int64)
    return up, rho

def layer1(X, W1, b1):
    """返回第一层预激活 U1（含 -sum W1 + b1 的常数补偿）"""
    up, rho = lut_lambda(W1)
    h = W1.shape[0]; N = len(X); Xi = X.astype(np.int64)
    S = np.zeros((N, h), dtype=np.int64)
    for j in range(h):
        if np.abs(W1[j]).max() <= 1e-12:
            continue
        for i in range(4):
            if abs(W1[j, i]) <= 1e-12:
                continue
            S[:, j] += (1 if W1[j, i] > 0 else -1) * up[j, i, Xi[:, i]]
    sc = np.array([(np.abs(W1[j]).max() / (868 * 7.5)) if np.abs(W1[j]).max() > 1e-12 else 0.0
                   for j in range(h)])
    return S * sc[None, :] - W1.sum(axis=1)[None, :] + b1[None, :]

def encode_channels(H, act, bits):
    """
    act='tanh': v = round((H+1)/2*(2^bits-1))
    act='relu': 线性转弱模。逐神经元按最大值定标到 2^bits-1
    bits=4 -> 1 通道; bits=8 -> 2 通道 (高/低半字节)
    """
    N, h = H.shape
    lv = (1 << bits) - 1
    if act == "tanh":
        v = np.clip(np.round((H + 1) * lv / 2), 0, lv).astype(np.int64)
    else:
        cm = H.max(axis=0); cm = np.where(cm > 1e-12, cm, 1.0)
        v = np.clip(np.round(H * lv / cm[None, :]), 0, lv).astype(np.int64)
    if bits == 4:
        return v.astype(float), 1
    nch = bits // 4
    Z = np.concatenate([((v >> (4 * k)) & 0xF) for k in range(nch)], axis=1).astype(float)
    return Z, nch

def fit_int(Z, target, S):
    A = np.hstack([Z, np.ones((len(Z), 1))])
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    return np.round(coef[:-1] * S).astype(np.int64), -round(coef[-1] * S)

def err_of(Z, w, th, labR):
    return int(np.sum(((Z @ w - th) >= 0).astype(np.int8) != labR))

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1o = np.array(m["W1"]); b1o = np.array(m["b1"])
    W2o = np.array(m["W2"]); b2o = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)
    errT = int(np.sum(((np.tanh(Xn @ W1o.T + b1o) @ W2o + b2o) >= 0).astype(np.int8) != lab))

    print("=" * 100, flush=True)
    print("ReLU 自洽实验：ReLU 老师定义真值表，强模 LUT 复现它", flush=True)
    print("=" * 100, flush=True)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举", flush=True)
    print(f"参照     : 官方 tanh 老师 = {errT}/{N}", flush=True)

    # ---- 训练 ReLU 老师：先拟合官方标签，再用它自己的输出当真值表 ----
    print(f"\n[1] 训练 ReLU 老师 4->8->1（连续隐藏层）", flush=True)
    bestR = None
    for seed in (0, 1, 2, 3, 4):
        r = train(Xn, y, 8, "relu", seed=seed, steps=6000, tag=f"relu s{seed}")
        if bestR is None or r[0] < bestR[0]:
            bestR = r
    print(f"   ReLU 老师对官方标签的最好错误 = {bestR[0]}/{N} ({bestR[0]/N*100:.4f}%)", flush=True)

    _, W1r, b1r, W2r, b2r = bestR
    Hr = np.maximum(Xn @ W1r.T + b1r, 0)
    yR = ((Hr @ W2r + b2r) >= 0).astype(np.int64)
    labR = (2 * yR - 1).astype(np.int8)
    print(f"   => 以此 ReLU 老师定义真值表（正类 {int(yR.sum())}）", flush=True)

    # ---- 用该真值表重训（验证可复现性）----
    print(f"\n[2] 用 ReLU 自洽真值表重训 4->8->1，看能否回到 0 错", flush=True)
    for act in ("relu", "tanh"):
        ee = []
        for seed in (0, 1, 2):
            r = train(Xn, yR, 8, act, seed=seed, steps=6000, tag=f"TT {act} s{seed}")
            ee.append(r[0])
        print(f"   {act:>5}: 重训错误 = {ee}  (最好 {min(ee)}, {min(ee)/N*100:.4f}%)", flush=True)

    # ---- 编译 ----
    U1 = layer1(X, W1r, b1r)
    H1 = np.maximum(U1, 0)
    dev = np.max(np.abs(U1 - (Xn @ W1r.T + b1r)))
    target = H1 @ W2r + b2r
    e1only = int(np.sum(((H1 @ W2r + b2r) >= 0).astype(np.int8) != labR))
    print(f"\n[3] 编译 ReLU 老师到强模 LUT", flush=True)
    print(f"   第一层 LUT 后预激活最大偏差 = {dev:.3e}", flush=True)
    print(f"   仅第一层 LUT（第二层实数）错误 = {e1only}/{N}", flush=True)

    print(f"\n   {'版本':>26} | {'弱模线':>6} | {'S':>8} | {'权重max':>8} | {'错误':>7} | {'错误率':>9}",
          flush=True)
    print("-" * 92, flush=True)
    out = {"teacher_official": errT, "relu_teacher_err": int(bestR[0]), "N": N}
    for bits, name in ((4, "V1 4bit/神经元"), (8, "V2 8bit/神经元(拆分)")):
        Z, nch = encode_channels(H1, "relu", bits)
        for S in (2000, 8000, 32000, 128000):
            w, th = fit_int(Z, target, S)
            e = err_of(Z, w, th, labR)
            nm = name.ljust(26)
            print(f"   {nm} | {nch*h:6d} | {S:8d} | {np.abs(w).max():8d} | {e:7d} | "
                  f"{e/N*100:8.4f}%", flush=True)
            out[f"relu_{bits}bit_S{S}"] = e
        print("-" * 92, flush=True)

    # 同样对 tanh 官方老师做，作对照
    print(f"\n[4] 对照：官方 tanh 老师做同样的编译", flush=True)
    U1t = layer1(X, W1o, b1o)
    Ht = np.tanh(U1t)
    tgtT = Ht @ W2o + b2o
    print(f"   仅第一层 LUT 错误 = {int(np.sum(((Ht @ W2o + b2o) >= 0).astype(np.int8) != lab))}",
          flush=True)
    print(f"   {'版本':>26} | {'弱模线':>6} | {'S':>8} | {'权重max':>8} | {'错误':>7} | {'错误率':>9}",
          flush=True)
    for bits, name in ((4, "V1 4bit/神经元"), (8, "V2 8bit/神经元(拆分)")):
        Z, nch = encode_channels(Ht, "tanh", bits)
        for S in (2000, 8000, 32000):
            w, th = fit_int(Z, tgtT, S)
            e = err_of(Z, w, th, lab)
            nm = name.ljust(26)
            print(f"   {nm} | {nch*h:6d} | {S:8d} | {np.abs(w).max():8d} | {e:7d} | "
                  f"{e/N*100:8.4f}%", flush=True)
            out[f"tanh_{bits}bit_S{S}"] = e

    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump(out, open(os.path.join(resdir, "relu_selfconsistent.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/relu_selfconsistent.json", flush=True)

if __name__ == "__main__":
    main()

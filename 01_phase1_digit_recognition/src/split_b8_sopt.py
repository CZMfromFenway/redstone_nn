#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：b=8 拆分结构，按"高位容器用满"确定 s（czm 口径）
========================================================
量纲对齐（关键）：
  lambda 的"步长" = x 变 1 时 lambda 变多少。
  对最后读出:  lambda_hi * w_hi + lambda_lo * w_lo
  而 v = 16*hi + lo,  读出 z = sum_j W_j * v_j / 127.5
  故 w_hi 必须 = 16 * w_lo；"高位步长 840 / 低位步长 52" 正是这个 16 倍关系。

  A_hi_j = s * 2*16*W_j / 255      (高位步长)
  A_lo_j = s * 2*1 *W_j / 255 = A_hi_j / 16
  s 由"最大 |W_j| 的高位跨度 15*A_hi 用满容器"确定：
      15 * A_hi_max = 868  =>  s*32*|W|max/255 * 15 = 868
"""
import numpy as np, json, os, time

CAP_STEP, CAP_CONT = 868, 897

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

def compile_w(W2, s, b=8, b2=0.0):
    denom = 2 ** b - 1
    a_hi = np.round(s * 2 * 16 * W2 / denom)
    a_lo = np.round(s * 2 * 1 * W2 / denom)
    return np.concatenate([a_hi, a_lo]), -s * (b2 - W2.sum())

def train_readout(Z, y, w0, th0, steps=2500, lr=0.05, bs=8192, seed=0, log_every=2500, tag="r"):
    rng = np.random.default_rng(seed); N = len(Z)
    w = np.asarray(w0, dtype=np.float64).copy(); th = float(th0)
    ms = np.zeros_like(w); vs = np.zeros_like(w); mb = vb = 0.0
    ys = 2 * y - 1
    best = (err(Z, w, th, y), w.copy(), th)
    t0 = time.time()
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
        if t % log_every == 0 or t == steps:
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
    wmax = np.abs(W2).max()

    print("=" * 100, flush=True)
    print("实验：b=8 拆分，按「最大 |W| 的高位跨度用满容器」确定 s", flush=True)
    print("=" * 100, flush=True)
    print(f"参照 A 原 Teacher = {errA}/{N}", flush=True)
    print(f"|W2|max = {wmax:.6f}", flush=True)

    # 由 15*A_hi_max = CAP 反解 s
    s_hi = CAP_STEP * 255 / (15 * 32 * wmax)
    s_cont = (CAP_CONT - 15) * 255 / (15 * 32 * wmax)
    print(f"\n[按容器用满反解 s]", flush=True)
    print(f"   15*A_hi_max = {CAP_STEP}  =>  s = {s_hi:.1f}", flush=True)
    print(f"   15*A_hi_max = {CAP_CONT} =>  s = {s_cont:.1f}", flush=True)

    print(f"\n[s 扫描：从 64 到超过上限]", flush=True)
    print(f"   {'s':>7} | {'A_hi_max':>8} | {'A_lo_max':>8} | {'15*A_hi_max':>11} | "
          f"{'越界':>4} | {'编译错误':>8} | {'训练后':>7} | {'错误率':>9}", flush=True)
    rows = []
    for s in (64, 128, 256, 512, int(s_hi * 0.5), int(s_hi * 0.75), int(s_hi),
              int(s_hi * 1.25), int(s_hi * 2)):
        w0, th0 = compile_w(W2, s, 8, b2)
        a_hi, a_lo = w0[8:], w0[:8]
        span = 15 * np.abs(a_hi).max()
        over = "是" if (np.abs(a_hi).max() > CAP_STEP or np.abs(a_lo).max() > CAP_CONT) else "否"
        e0 = err(Z, w0, th0, y)
        e1, _, _ = train_readout(Z, y, w0, th0, steps=2500, seed=0, tag=f"s{s}")
        print(f"   {s:7d} | {np.abs(a_hi).max():8.0f} | {np.abs(a_lo).max():8.0f} | "
              f"{span:11.0f} | {over:>4} | {e0:8d} | {e1:7d} | {e1/N*100:8.4f}%", flush=True)
        rows.append({"s": s, "a_hi_max": float(np.abs(a_hi).max()),
                     "span": float(span), "over": over == "是",
                     "compile_err": int(e0), "train_err": int(e1)})

    best = min(rows, key=lambda r: r["train_err"])
    print(f"\n[最优] s = {best['s']}，错误 = {best['train_err']}/{N} "
          f"({best['train_err']/N*100:.4f}%)，此时高位跨度 = {best['span']:.0f}", flush=True)
    w0, th0 = compile_w(W2, best["s"], 8, b2)
    print(f"   A_hi = {w0[8:].astype(int).tolist()}", flush=True)
    print(f"   A_lo = {w0[:8].astype(int).tolist()}", flush=True)
    print(f"   比值 A_hi/A_lo = {[round(a/b,1) if b else None for a,b in zip(w0[8:], w0[:8])]}",
          flush=True)

    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    json.dump({"A": errA, "N": N, "s_from_container": float(s_hi), "rows": rows},
              open(os.path.join(resdir, "split_b8_sopt.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n已写出: results/split_b8_sopt.json", flush=True)

if __name__ == "__main__":
    main()

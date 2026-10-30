#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验：多层强模 LUT 网络（逐层贪心 Fisher + 联合训练对照）
==========================================================
【实验配置】
  输入空间   (x1..x4) ∈ {0..15}^4，全部 65536 状态穷举
  目标标签   N4_MLP_T1 官方标签（4->8->1 tanh MLP，seed 71000，正类 18436）
  评价       全部 65536 状态上的最终 0/1 错误数

  网络 4 -> [h1] -> [h2] -> 1
    * 中间层输出量化到 16 级（0~15 弱模，纵向布线要求）
    * 最后一层输入仍是 0~15 整数，但输出**不量化**：直接取高精度阈值
      sign(sum_j v_j y_j - theta)

  贪心目标（czm 提议）：Fisher 比，在量化编码上计算
      J = (mu+ - mu-)^2 / (sig+^2 + sig-^2)
      对编码的梯度（w 为当前 batch 的 LDA 方向）:
        dJ/dy_j = 2 w_j (mu+ - mu-) / S  -  2 w_j (z - mu_s) * D / S^2
      其中 z = w.y, S = sig+^2+sig-^2, D = (mu+-mu-)^2/(n+ * n-) 量级因子。
      实现上直接用"投影后 1 维"的 Fisher 梯度，再经 w 反传到各单元。
"""
import numpy as np, json, os, time

L = 16
BS = 8192

def load_experiment():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rec = os.path.join(base, "DT_Redstone_Research_Archive_20261001", "DT_baseline_recovery")
    m = json.load(open(os.path.join(rec, "teachers", "N4_MLP_T1.json"), encoding="utf-8"))
    lab = np.load(os.path.join(rec, "teachers", "N4_MLP_T1.npy")).astype(np.int8)
    X = np.indices((16,) * 4, dtype=np.int8).reshape(4, -1).T
    return m, lab, X

def quant(Hc):
    return np.clip(np.round((Hc + 1) * (L - 1) / 2), 0, L - 1)

def fisher_ratio(Z, y):
    Z = np.asarray(Z, dtype=np.float64)
    m0 = Z[y == 0].mean(0); m1 = Z[y == 1].mean(0); d = m1 - m0
    if Z.ndim == 1:
        Z = Z[:, None]
    S0 = np.atleast_2d(np.cov(Z[y == 0], rowvar=False))
    S1 = np.atleast_2d(np.cov(Z[y == 1], rowvar=False))
    Sw = S0 + S1 + 1e-8 * np.eye(Z.shape[1])
    w = np.linalg.solve(Sw, d)
    return float((w @ d) ** 2 / max(w @ Sw @ w, 1e-12))

def lda_dir(Z, y):
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    m0 = Z[y == 0].mean(0); m1 = Z[y == 1].mean(0); d = m1 - m0
    S0 = np.atleast_2d(np.cov(Z[y == 0], rowvar=False))
    S1 = np.atleast_2d(np.cov(Z[y == 1], rowvar=False))
    Sw = S0 + S1 + 1e-8 * np.eye(Z.shape[1])
    v = np.linalg.solve(Sw, d)
    return v, 0.5 * (v @ (m0 + m1))

def err_of(Z, v, th, y):
    return int(np.sum(((Z @ v - th) >= 0).astype(np.int64) != y))

def layer_eval(Hq, y):
    v, th = lda_dir(Hq, y)
    return err_of(Hq, v, th, y), float(fisher_ratio(Hq, y))

def fisher_grad_proj(proj, y):
    """
    1 维投影 proj 上的 Fisher 比 J = (mu+-mu-)^2/(var+ + var-)
    返回 dJ/dproj （逐样本）
    """
    p1 = proj[y == 1]; p0 = proj[y == 0]
    n1 = max(len(p1), 1); n0 = max(len(p0), 1)
    m1 = p1.mean() if len(p1) else 0.0; m0 = p0.mean() if len(p0) else 0.0
    v1 = p1.var() if len(p1) else 0.0; v0 = p0.var() if len(p0) else 0.0
    S = v1 + v0 + 1e-9
    D = (m1 - m0)
    g = np.zeros_like(proj)
    # dD^2/dp_i
    g[y == 1] += 2 * D / n1
    g[y == 0] -= 2 * D / n0
    # -D^2/S^2 * dS/dp_i ;  dS/dp_i = 2(p_i - m_s)/n_s
    coef = -(D ** 2) / (S ** 2)
    g[y == 1] += coef * 2 * (p1 - m1) / n1
    g[y == 0] += coef * 2 * (p0 - m0) / n0
    return g / S

def train_layer(Xin, y, h, seed=0, steps=1500, lr=0.05, bs=BS, tag="L", log_every=250):
    """训练一层：y_out = quant(tanh(Xin @ W.T + b))，贪心目标 = 量化编码上的 Fisher 比"""
    rng = np.random.default_rng(seed)
    Nin, din = Xin.shape
    W = rng.normal(0, 0.6, (h, din)); b = np.zeros(h)
    ps = [W, b]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    s = (L - 1) / 2
    best = None; t0 = time.time()
    for t in range(1, steps + 1):
        idx = rng.choice(Nin, bs, replace=False)
        xb = Xin[idx]; yb = y[idx]
        Hc = np.tanh(xb @ W.T + b)
        Hq = quant(Hc).astype(np.float64)
        w, _ = lda_dir(Hq, yb)
        w = w / max(np.linalg.norm(w), 1e-12)
        proj = Hq @ w
        gproj = fisher_grad_proj(proj, yb)           # (bs,)
        # Fisher 比是尺度不变的，原始梯度没有稳定尺度 -> 归一化到单位量级
        gn = np.linalg.norm(gproj) / max(len(gproj), 1)
        if gn > 1e-12:
            gproj = gproj / gn
        # 反传到编码： dJ/dHq = gproj[:,None] * w[None,:]；再经量化 STE 与 tanh
        dH = np.outer(gproj, w) * (s * (1 - Hc ** 2))
        gW = dH.T @ xb; gb = dH.sum(0)
        for i, (pp, gg) in enumerate(zip(ps, [gW, gb])):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp += lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % log_every == 0 or t == steps:
            Hq_all = quant(np.tanh(Xin @ W.T + b))
            e, fr = layer_eval(Hq_all, y)
            if best is None or e < best[0]:
                best = (e, W.copy(), b.copy(), Hq_all.copy().astype(np.int64), fr)
            print(f"      [{tag}] step {t:5d} ({time.time()-t0:4.0f}s): "
                  f"LDA错误={e:6d}  Fisher比={fr:8.4f}  最好={best[0]}", flush=True)
    return best

def joint_train(Xn, y, arch, seed=0, steps=2000, lr=0.03, bs=BS, tag="joint", log_every=250):
    """整网联合训练；中间层量化，前 3/4 步用 hinge，后 1/4 用 BCE"""
    rng = np.random.default_rng(seed)
    dims = [Xn.shape[1]] + list(arch)
    Ws = [rng.normal(0, 0.6, (dims[i+1], dims[i])) for i in range(len(arch))]
    bs_ = [np.zeros(dims[i+1]) for i in range(len(arch))]
    v = rng.normal(0, 0.6, arch[-1]); th = 0.0
    # 参数顺序：按层交错 (W1, b1, W2, b2, ...)，最后是输出层 (v, th)
    params_order = []
    for li in range(len(arch)):
        params_order.append(("W", li))
        params_order.append(("b", li))
    params_order.append(("v", None))
    params_order.append(("th", None))
    def build_params():
        out = []
        for kind, li in params_order:
            if kind == "W": out.append(Ws[li])
            elif kind == "b": out.append(bs_[li])
            elif kind == "v": out.append(v)
            else: out.append(np.array(th))
        return out
    ps = build_params()
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    s = (L - 1) / 2; ys = 2 * y - 1
    N = len(Xn); best = None; t0 = time.time()
    for t in range(1, steps + 1):
        idx = rng.choice(N, bs, replace=False)
        xb = Xn[idx]; yb = y[idx]; yy = ys[idx]
        ins = [xb] + [None] * len(arch)
        Hcs = []; outs = []
        cur = xb
        for li in range(len(arch)):
            Hc = np.tanh(cur @ Ws[li].T + bs_[li]); Hcs.append(Hc)
            out = quant(Hc) if li < len(arch) - 1 else Hc
            outs.append(out)
            if li + 1 < len(arch):
                ins[li + 1] = out
            cur = out
        z = cur @ v - th
        if t <= steps * 3 // 4:
            m = 1 - yy * z; dz = (-yy * (m > 0)) / bs
        else:
            p = 1 / (1 + np.exp(-z)); dz = (p - yb) / bs
        # 反向传播（显式逐层，不用闭包）
        gradsW = [None] * len(arch)      # 每层的 dW
        gradsB = [None] * len(arch)      # 每层的 db
        dcur = np.outer(dz, v)           # dL/d(out of last layer)
        for li in range(len(arch) - 1, -1, -1):
            Hc = Hcs[li]
            if li < len(arch) - 1:
                scale = s * (1 - Hc ** 2)        # 量化层的 STE 系数
            else:
                scale = 1 - Hc ** 2              # 最后一层不量化
            dH = dcur * scale                    # (bs, dims[li+1])
            inp = ins[li]                        # (bs, dims[li])
            gradsW[li] = dH.T @ inp
            gradsB[li] = dH.sum(0)
            if li > 0:
                dcur = dH @ Ws[li]
        g = []
        for kind, li in params_order:
            if kind == "W": g.append(gradsW[li])
            elif kind == "b": g.append(gradsB[li])
            elif kind == "v": g.append(cur.T @ dz)
            else: g.append(np.array(-dz.sum()))
        assert len(g) == len(ps) and all(
            np.shape(a) == np.shape(b) for a, b in zip(g, ps)), \
            ("grad/param 形状不符", [np.shape(a) for a in g], [np.shape(b) for b in ps])
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % log_every == 0 or t == steps:
            cur2 = Xn
            for li in range(len(arch)):
                Hc = np.tanh(cur2 @ Ws[li].T + bs_[li])
                cur2 = quant(Hc) if li < len(arch) - 1 else Hc
            e = int(np.sum(((cur2 @ v - th) >= 0).astype(np.int64) != y))
            if best is None or e < best[0]:
                best = (e,)
            print(f"      [{tag}] step {t:5d} ({time.time()-t0:4.0f}s): 错误={e:6d} 最好={best[0]}",
                  flush=True)
    return best

def main():
    m, lab, X = load_experiment()
    y = (lab > 0).astype(np.int64)
    W1t = np.array(m["W1"]); b1t = np.array(m["b1"]); W2t = np.array(m["W2"]); b2t = float(m["b2"])
    Xn = (X.astype(np.float64) - 7.5) / 7.5
    N = len(lab)
    errA = int(np.sum(((np.tanh(Xn @ W1t.T + b1t) @ W2t + b2t) >= 0).astype(np.int8) != lab))

    print("=" * 100, flush=True)
    print("实验：多层强模 LUT 网络（逐层贪心 Fisher + 联合对照）", flush=True)
    print("=" * 100, flush=True)
    print(f"输入空间 : (x1..x4) ∈ {{0..15}}^4，全部 {N} 状态穷举", flush=True)
    print(f"目标标签 : N4_MLP_T1（正类 {int(y.sum())}）", flush=True)
    print(f"参照     : A 原 Teacher（隐藏层连续）= {errA}/{N}", flush=True)
    print(f"中间层量化到 {L} 级；最后一层输入为 0~15 整数、输出不量化", flush=True)

    archs = [(8,), (16,), (8, 8), (16, 16), (32, 32)]
    results = {}
    resdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(resdir, exist_ok=True)
    for arch in archs:
        name = "->".join(["4"] + [str(a) for a in arch] + ["1"])
        print(f"\n{'#'*100}\n### 结构 {name}   (中间层量化到 {L} 级)\n{'#'*100}", flush=True)
        cur = Xn.copy(); layers = []
        for li, h in enumerate(arch):
            print(f"   [贪心 第{li+1}层] 输入维度 {cur.shape[1]} -> {h} 单元", flush=True)
            best = None
            for seed in (0, 1):
                r = train_layer(cur, y, h, seed=seed, steps=1000, tag=f"{name} L{li+1} s{seed}")
                if best is None or r[0] < best[0]:
                    best = r
            layers.append(best)
            print(f"   => 第{li+1}层：LDA读出错误 = {best[0]}/{N} ({best[0]/N*100:.4f}%)  "
                  f"Fisher比 = {best[4]:.4f}", flush=True)
            cur = best[3].astype(np.float64)
        greedy_err = layers[-1][0]
        print(f"   [贪心 完成] 最终错误 = {greedy_err}/{N} ({greedy_err/N*100:.4f}%)", flush=True)
        print(f"   [联合训练] 整网 hinge(3/4)+BCE(1/4)", flush=True)
        jb = None
        for seed in (0, 1):
            r = joint_train(Xn, y, arch, seed=seed, steps=1200, tag=f"{name} joint s{seed}")
            if jb is None or r[0] < jb[0]:
                jb = r
        print(f"   => 联合训练：错误 = {jb[0]}/{N} ({jb[0]/N*100:.4f}%)", flush=True)
        results[name] = {"greedy": int(greedy_err), "joint": int(jb[0]),
                         "layer_fisher": [float(x[4]) for x in layers],
                         "layer_err": [int(x[0]) for x in layers]}
        json.dump({"A": errA, "N": N, "results": results},
                  open(os.path.join(resdir, "multilayer.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"   (已写入 results/multilayer.json)", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("汇总", flush=True)
    print("=" * 100, flush=True)
    print(f"   {'结构':>12} | {'贪心错误':>9} | {'错误率':>9} | {'联合错误':>9} | {'错误率':>9}")
    print(f"   {'A 原Teacher':>12} | {errA:9d} | {errA/N*100:8.4f}% | {'-':>9} | {'-':>9}")
    for name, r in results.items():
        print(f"   {name:>12} | {r['greedy']:9d} | {r['greedy']/N*100:8.4f}% | "
              f"{r['joint']:9d} | {r['joint']/N*100:8.4f}%")
    print("\n已写出: results/multilayer.json", flush=True)

if __name__ == "__main__":
    main()

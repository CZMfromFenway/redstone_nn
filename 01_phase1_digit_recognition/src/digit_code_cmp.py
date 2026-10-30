#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
四方对比: 编码方式 vs 输出形式
================================
问题: 225->h 的编码方案能否追平 225->10 argmax？

四种配置（同一数据、同一协议）:
  A. 225 -> 10 argmax            （基准）
  B. 225 -> h 独立 sigmoid        （每比特独立决定）
  C. 225 -> 2^h 路 softmax        （码作为整体类别，多正类目标）
  D. 225 -> h 独立 sigmoid，但码集**单码**（每数字恰好1码，最省）

评价: 最近码解码（预测码 -> 最近已分配码 -> 其数字）
"""
import os, json, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
RES = os.path.join(ROOT, "results")
SIM = {(3, 8): 1.0, (6, 9): 0.9, (4, 9): 0.8, (1, 7): 0.7, (0, 8): 0.7,
       (5, 8): 0.6, (0, 6): 0.6, (3, 5): 0.6, (2, 7): 0.5, (4, 7): 0.5}


def load_idx(p):
    import struct
    with open(p, 'rb') as f:
        mg = struct.unpack('>I', f.read(4))[0]; nd = mg & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(nd)]
        d = np.frombuffer(f.read(), dtype=np.uint8)
    return d.reshape(dims)


def to15(im):
    n = len(im)
    a = im.reshape(n, 28, 28).astype(np.float32).reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    o = np.zeros((n, 15, 15), dtype=np.float32)
    o[:, :14, :14] = a; o[:, 14, :14] = a[:, 13, :]
    o[:, :14, 14] = a[:, :, 13]; o[:, 14, 14] = a[:, 13, 13]
    return o


def sim(d, e):
    return 1.0 if d == e else SIM.get((d, e), SIM.get((e, d), 0.0))


def search(h, restarts=60, iters=4000, seed=0):
    NC = 1 << h
    C = np.array([[int(b) for b in format(i, f'0{h}b')] for i in range(NC)], dtype=int)
    D = (C[:, None, :] != C[None, :, :]).sum(2)
    base = NC // 10; rem = NC - base * 10
    tgt = np.array([base + (1 if i < rem else 0) for i in range(10)], float)

    def cost(asg):
        cnt = np.bincount(asg, minlength=10).astype(float)
        if (cnt == 0).any():
            return 1e9
        c = 3.0 * np.abs(cnt - tgt).sum()
        for d in range(10):
            ix = np.flatnonzero(asg == d)
            if len(ix) > 1:
                c += D[np.ix_(ix, ix)].sum() / (len(ix) * (len(ix) - 1))
        for a in range(10):
            for b in range(a + 1, 10):
                s = sim(a, b)
                if s <= 0:
                    continue
                ia = np.flatnonzero(asg == a); ib = np.flatnonzero(asg == b)
                c += 0.3 * (1 - s) * D[np.ix_(ia, ib)].min()
        return c

    rng = np.random.default_rng(seed); best = None
    for r in range(restarts):
        asg = np.repeat(np.arange(10), tgt.astype(int))
        extra = NC - len(asg)
        if extra > 0:
            asg = np.concatenate([asg, rng.integers(0, 10, extra)])
        asg = asg[:NC]; rng.shuffle(asg)
        c = cost(asg); T = 2.0
        for it in range(iters):
            k = rng.integers(0, NC); old = asg[k]; new = rng.integers(0, 10)
            if new == old:
                continue
            asg[k] = new; c2 = cost(asg)
            if c2 < c or rng.random() < np.exp(-(c2 - c) / max(T, 1e-9)):
                c = c2
            else:
                asg[k] = old
            T *= 0.999
        if best is None or c < best[0]:
            best = (c, asg.copy())
    return best[1], C, D


def adams(params, grads, state, t, lr):
    M, V = state
    for i, (p, g) in enumerate(zip(params, grads)):
        M[i] = 0.9 * M[i] + 0.1 * g
        V[i] = 0.999 * V[i] + 0.001 * g ** 2
        p -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)


def main():
    h = 4
    print("=" * 96)
    print("四方对比：编码方式 × 输出形式")
    print("=" * 96)
    Xtr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    A = to15(Xtr); B = to15(Xte); thr = A.mean()
    Ftr = (A > thr).astype(np.float64).reshape(len(A), -1)
    Fte = (B > thr).astype(np.float64).reshape(len(B), -1)
    d_in = Ftr.shape[1]; n = len(Ftr)
    print(f"数据 train {Ftr.shape}  test {Fte.shape}")

    asg, C, D = search(h)
    NC = 1 << h
    print(f"\n[h={h} 码集]")
    for d in range(10):
        print(f"  {d}: {', '.join(''.join(map(str,C[i])) for i in np.flatnonzero(asg==d))}")
    BM = (C * 2 - 1).astype(np.float64)
    rep = np.zeros(10, int)
    for d in range(10):
        ix = np.flatnonzero(asg == d)
        rep[d] = ix[D[np.ix_(ix, ix)].sum(1).argmin()]

    steps, lr, bs = 4000, 0.05, 128
    rng = np.random.default_rng(0)

    def run_adam(d_out, target_fn, pred_fn, name, nparam):
        W = np.zeros((d_in, d_out)); b = np.zeros(d_out)
        P = [W, b]; M = [np.zeros_like(W), np.zeros_like(b)]
        V = [np.zeros_like(W), np.zeros_like(b)]
        best = 0.0
        for t in range(1, steps + 1):
            idx = rng.integers(0, n, bs)
            Fb, yb = Ftr[idx], ytr[idx]
            gW, gb = target_fn(Fb, yb, W, b)
            adams(P, [gW, gb], (M, V), t, lr)
            if t % 1000 == 0 or t == steps:
                acc = pred_fn(Fte, yte, W, b)
                best = max(best, acc)
        print(f"  {name:<34} 参数 {nparam:>5}  最好 {best:.4f}", flush=True)
        return best

    print(f"\n[对比]")
    res = {}

    # A. 10 路 argmax
    def tgtA(Fb, yb, W, b):
        Z = Fb @ W + b; mx = Z.max(1, keepdims=True)
        e = np.exp(Z - mx); Pm = e / e.sum(1, keepdims=True)
        dZ = Pm.copy(); dZ[np.arange(len(yb)), yb] -= 1; dZ /= len(yb)
        return Fb.T @ dZ, dZ.sum(0)

    def preA(Fte, yte, W, b):
        return float(((Fte @ W + b).argmax(1) == yte).mean())
    res['A_10argmax'] = run_adam(10, tgtA, preA, 'A. 225->10 argmax', d_in * 10 + 10)

    # B. h 独立 sigmoid（代表码）
    Ytr_b = C[rep[ytr]].astype(np.float64) * 2 - 1

    def tgtB(Fb, yb, W, b):
        Z = Fb @ W + b
        P = 1 / (1 + np.exp(-np.clip(Z, -30, 30)))
        dZ = (P - (Ytr_b_slice(yb) > 0)) / len(yb)
        return Fb.T @ dZ, dZ.sum(0)
    # 为速度，预先建好 (n,h)
    YtrB = C[rep[ytr]].astype(np.float64) * 2 - 1

    def tgtB2(Fb, yb, W, b):
        Z = Fb @ W + b
        P = 1 / (1 + np.exp(-np.clip(Z, -30, 30)))
        # 需要对应样本的 Y，这里 yb 是索引
        pass

    # 直接内联实现 B/C 以便控制
    def runB():
        W = np.zeros((d_in, h)); b = np.zeros(h)
        P = [W, b]; M = [np.zeros_like(W), np.zeros_like(b)]
        V = [np.zeros_like(W), np.zeros_like(b)]
        best = 0.0
        for t in range(1, steps + 1):
            idx = rng.integers(0, n, bs)
            Fb = Ftr[idx]; Yb = YtrB[idx]
            Z = Fb @ W + b
            Pb = 1 / (1 + np.exp(-np.clip(Z, -30, 30)))
            dZ = (Pb - (Yb > 0)) / len(idx)
            adams(P, [Fb.T @ dZ, dZ.sum(0)], (M, V), t, lr)
            if t % 1000 == 0 or t == steps:
                Zt = Fte @ W + b
                hard = np.where(Zt > 0, 1.0, -1.0)
                d2 = ((hard[:, None, :] - BM[None, :, :]) ** 2).sum(2)
                best = max(best, float((asg[d2.argmin(1)] == yte).mean()))
        return best
    res['B_h_sigmoid'] = runB()
    print(f"  {'B. 225->h 独立 sigmoid':<34} 参数 {d_in*h+h:>5}  最好 {res['B_h_sigmoid']:.4f}")

    # C. 2^h 路 softmax（多正类）
    valid = np.zeros((10, NC)); 
    for d in range(10):
        valid[d, asg == d] = 1.0
    valid /= valid.sum(1, keepdims=True)
    Vtr = valid[ytr]

    def runC():
        W = np.zeros((d_in, NC)); b = np.zeros(NC)
        P = [W, b]; M = [np.zeros_like(W), np.zeros_like(b)]
        V = [np.zeros_like(W), np.zeros_like(b)]
        best = 0.0
        for t in range(1, steps + 1):
            idx = rng.integers(0, n, bs)
            Fb = Ftr[idx]; Tb = Vtr[idx]
            Z = Fb @ W + b; mx = Z.max(1, keepdims=True)
            e = np.exp(Z - mx); Pm = e / e.sum(1, keepdims=True)
            dZ = (Pm - Tb) / len(idx)
            adams(P, [Fb.T @ dZ, dZ.sum(0)], (M, V), t, lr)
            if t % 1000 == 0 or t == steps:
                # 预测码 = argmax，解码 = 该码所属数字
                Zt = Fte @ W + b
                best = max(best, float((asg[Zt.argmax(1)] == yte).mean()))
        return best
    res['C_softmax_codes'] = runC()
    print(f"  {'C. 225->2^h softmax(码)':<34} 参数 {d_in*NC+NC:>5}  最好 {res['C_softmax_codes']:.4f}")

    # D. h 独立 sigmoid，单码（每数字恰好1码，最省）
    NCd = 16
    asg1 = np.zeros(NCd, dtype=int); rep1 = np.zeros(10, dtype=int)
    # 用搜索结果的代表码，其余码归给最近的数字
    for d in range(10):
        rep1[d] = rep[d]
    dr = ((C[:, None, :] - C[rep1][None, :, :]) ** 2).sum(2)
    nearest = dr.argmin(1)
    asg1 = nearest
    YtrD = C[rep1[ytr]].astype(np.float64) * 2 - 1

    def runD():
        W = np.zeros((d_in, h)); b = np.zeros(h)
        P = [W, b]; M = [np.zeros_like(W), np.zeros_like(b)]
        V = [np.zeros_like(W), np.zeros_like(b)]
        best = 0.0
        for t in range(1, steps + 1):
            idx = rng.integers(0, n, bs)
            Fb = Ftr[idx]; Yb = YtrD[idx]
            Z = Fb @ W + b
            Pb = 1 / (1 + np.exp(-np.clip(Z, -30, 30)))
            dZ = (Pb - (Yb > 0)) / len(idx)
            adams(P, [Fb.T @ dZ, dZ.sum(0)], (M, V), t, lr)
            if t % 1000 == 0 or t == steps:
                Zt = Fte @ W + b
                hard = np.where(Zt > 0, 1.0, -1.0)
                d2 = ((hard[:, None, :] - BM[None, :, :]) ** 2).sum(2)
                best = max(best, float((asg1[d2.argmin(1)] == yte).mean()))
        return best
    res['D_h_sigmoid_single'] = runD()
    print(f"  {'D. 225->h 单码 sigmoid':<34} 参数 {d_in*h+h:>5}  最好 {res['D_h_sigmoid_single']:.4f}")

    print(f"\n{'='*96}")
    print("### 结论")
    for k, v in res.items():
        print(f"  {k:<24} {v:.4f}")
    print(f"{'='*96}")
    json.dump(res, open(os.path.join(RES, 'digit_code_compare.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()

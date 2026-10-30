#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
方案 C 随 h 的变化: h = 4, 5, 6
==================================
问题: 增大 h（更多码）会提升准确率吗？

注意方向: 方案 C 的输出是 2^h 条线，所以
  h=4 -> 16 条     h=5 -> 32 条     h=6 -> 64 条
这是**增加**输出线，不是减少。

测什么: 每数字分到的码变多（1.6 -> 3.2 -> 6.4）是否带来容错红利
对照: 10 路 argmax = 0.8915
"""
import os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
RES = os.path.join(ROOT, "results")
SIM = {(3, 8): 1.0, (6, 9): 0.9, (4, 9): 0.8, (1, 7): 0.7, (0, 8): 0.7,
       (5, 8): 0.6, (0, 6): 0.6, (3, 5): 0.6, (2, 7): 0.5, (4, 7): 0.5,
       (0, 9): 0.5, (1, 4): 0.4, (2, 3): 0.4, (5, 6): 0.3}


def sim(a, b):
    return 1.0 if a == b else SIM.get((a, b), SIM.get((b, a), 0.0))


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


def search(h, restarts=25, iters=3500, seed=0):
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


def main():
    print("=" * 96)
    print("方案 C 随 h 的变化（h = 4, 5, 6）")
    print("=" * 96)
    Xtr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    A = to15(Xtr); B = to15(Xte); thr = A.mean()
    Ftr = (A > thr).astype(np.float64).reshape(len(A), -1)
    Fte = (B > thr).astype(np.float64).reshape(len(B), -1)
    d_in = Ftr.shape[1]; n = len(Ftr)
    print(f"数据 train {Ftr.shape}  test {Fte.shape}\n")

    steps, lr, bs = 4000, 0.05, 128
    summary = []
    for h in (4, 5, 6):
        NC = 1 << h
        t0 = time.time()
        asg, C, D = search(h)
        cnt = np.bincount(asg, minlength=10)
        # 每个数字的代表码
        rep = np.zeros(10, int)
        for d in range(10):
            ix = np.flatnonzero(asg == d)
            rep[d] = ix[D[np.ix_(ix, ix)].sum(1).argmin()]
        # 组内平均距离
        intra = np.mean([D[np.ix_(np.flatnonzero(asg == d), np.flatnonzero(asg == d))].mean()
                         if (asg == d).sum() > 1 else 0.0 for d in range(10)])
        # 相似数字加权最小码距
        v = w = 0.0
        for (a, b), s in SIM.items():
            ia = np.flatnonzero(asg == a); ib = np.flatnonzero(asg == b)
            v += s * D[np.ix_(ia, ib)].min(); w += s
        print(f"[h={h}]  码数 {NC}  每数字码数 {cnt.tolist()}")
        print(f"       组内平均距离 {intra:.3f}   相似对加权最小码距 {v/w:.2f}  "
              f"[搜索 {time.time()-t0:.0f}s]")

        # 目标: 代表码 0.95 + 备选 0.05
        T = np.zeros((10, NC))
        for d in range(10):
            ix = np.flatnonzero(asg == d)
            others = [c for c in ix if c != rep[d]]
            T[d, rep[d]] = 0.95 if others else 1.0
            for c in others:
                T[d, c] = 0.05 / len(others)
        Tg = T[ytr]

        rng = np.random.default_rng(0)
        W = np.zeros((d_in, NC)); b = np.zeros(NC)
        P = [W, b]; M = [np.zeros_like(W), np.zeros_like(b)]
        V = [np.zeros_like(W), np.zeros_like(b)]
        best = 0.0
        for t in range(1, steps + 1):
            idx = rng.integers(0, n, bs)
            Fb = Ftr[idx]; Tb = Tg[idx]
            Z = Fb @ W + b; mx = Z.max(1, keepdims=True)
            e = np.exp(Z - mx); Pm = e / e.sum(1, keepdims=True)
            dZ = (Pm - Tb) / len(idx)
            g = [Fb.T @ dZ, dZ.sum(0)]
            for i, (pp, gg) in enumerate(zip(P, g)):
                M[i] = 0.9 * M[i] + 0.1 * gg; V[i] = 0.999 * V[i] + 0.001 * gg ** 2
                pp -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)
            if t % 1000 == 0 or t == steps:
                best = max(best, float((asg[(Fte @ W + b).argmax(1)] == yte).mean()))
        nparam = d_in * NC + NC
        print(f"       准确率 {best:.4f}   参数 {nparam}   输出线 {NC}\n", flush=True)
        summary.append({'h': h, 'NC': NC, 'acc': best, 'param': nparam,
                        'codes_per_digit': cnt.tolist(), 'intra': float(intra),
                        'sim_min': float(v / w)})

    print("=" * 96)
    print("### 汇总")
    print(f"  {'h':>3} | {'码数/输出线':>10} | {'每数字码数':>12} | {'参数':>7} | {'准确率':>7}")
    print("  " + "-" * 62)
    for s in summary:
        avg = np.mean(s['codes_per_digit'])
        print(f"  {s['h']:>3} | {s['NC']:>10} | {avg:>12.1f} | {s['param']:>7} | {s['acc']:>7.4f}")
    print(f"\n  对照: 225->10 argmax   输出线 10   参数 2260   准确率 0.8915")
    print("=" * 96)
    json.dump(summary, open(os.path.join(RES, 'digit_code_h_sweep.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()

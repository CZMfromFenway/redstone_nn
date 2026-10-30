#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
方案 B 随 h 的变化: h 个独立 sigmoid, h = 4, 5, 6
====================================================
方案 B 的输出 = h 个独立二值 -> 只能产生 2^h 个码

两个相反的效应:
  (+) 好处: 码空间更大 -> 10 个数字可以排得更开（Hamming 距离更大）
  (-) 坏处: 浪费的码更多（只有 10 个是"命中"的）
        h=4 -> 16 码, 浪费 6  (37.5%)
        h=5 -> 32 码, 浪费 22 (68.8%)
        h=6 -> 64 码, 浪费 54 (84.4%)

另外测: 加隐藏层是否能解决"独立输出无法协同"的问题
  (a) 线性  225 -> h
  (b) 隐藏  225 -> h1 -> h

对照: 225->10 argmax = 0.8915, 方案 C(h=4) = 0.8936
"""
import os, json, itertools
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


def pick_codes(h, restarts=200, seed=0):
    """
    为 10 个数字各选一个码，使:
      - 相似数字的码距小
      - 整体码尽量分散（不同数字要分开）
    """
    NC = 1 << h
    C = np.array([[int(b) for b in format(i, f'0{h}b')] for i in range(NC)], dtype=int)
    D = (C[:, None, :] != C[None, :, :]).sum(2)

    def cost(sel):
        c = 0.0
        # 相似数字 -> 码距小
        for (a, b), s in SIM.items():
            c += s * D[sel[a], sel[b]]
        # 不相似的数字 -> 码距大（促进分散）
        for a in range(10):
            for b in range(a + 1, 10):
                s = sim(a, b)
                c += (1.0 - s) * (h - D[sel[a], sel[b]]) * 0.8
        return c

    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        sel = rng.choice(NC, 10, replace=False)
        c = cost(sel)
        # 局部搜索
        for it in range(300):
            i = rng.integers(0, 10)
            old = sel[i]
            new = rng.integers(0, NC)
            if new in sel:
                continue
            sel[i] = new
            c2 = cost(sel)
            if c2 < c:
                c = c2
            else:
                sel[i] = old
        if best is None or c < best[0]:
            best = (c, sel.copy())
    return best[1], C, D


def main():
    print("=" * 96)
    print("方案 B 随 h 的变化（h 个独立 sigmoid）")
    print("=" * 96)
    Xtr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    A = to15(Xtr); B16 = to15(Xte); thr = A.mean()
    Ftr = (A > thr).astype(np.float64).reshape(len(A), -1)
    Fte = (B16 > thr).astype(np.float64).reshape(len(B16), -1)
    d_in = Ftr.shape[1]; n = len(Ftr)
    print(f"数据 train {Ftr.shape}  test {Fte.shape}\n")

    steps, lr, bs = 4000, 0.05, 128
    out = []
    for h in (4, 5, 6):
        NC = 1 << h
        sel, C, D = pick_codes(h)
        # 代表 (-1/+1)
        REPB = (C[sel] * 2 - 1).astype(np.float64)          # (10,h)
        dmin = min(D[sel[a], sel[b]] for a in range(10) for b in range(a + 1, 10))
        # 相似对的加权码距
        v = w = 0.0
        for (a, b), s in SIM.items():
            v += s * D[sel[a], sel[b]]; w += s
        print(f"[h={h}]  码数 {NC}  浪费 {NC-10} 个 ({(NC-10)/NC*100:.1f}%)")
        print(f"       代表码: " + "  ".join(
            f"{d}:{format(sel[d], f'0{h}b')}" for d in range(10)))
        print(f"       任意两数字的最小码距 {dmin}   相似对加权码距 {v/w:.2f}")

        Ytr = REPB[ytr]
        Yte = REPB[yte]
        rng = np.random.default_rng(0)

        def decode(Zt):
            hard = np.where(Zt > 0, 1.0, -1.0)
            d2 = ((hard[:, None, :] - REPB[None, :, :]) ** 2).sum(2)
            return sel[d2.argmin(1)]

        # (a) 线性 225 -> h
        W = np.zeros((d_in, h)); b = np.zeros(h)
        P = [W, b]; M = [np.zeros_like(W), np.zeros_like(b)]
        V = [np.zeros_like(W), np.zeros_like(b)]
        bestA = 0.0
        for t in range(1, steps + 1):
            idx = rng.integers(0, n, bs)
            Fb, Yb = Ftr[idx], Ytr[idx]
            Z = Fb @ W + b
            Pb = 1 / (1 + np.exp(-np.clip(Z, -30, 30)))
            dZ = (Pb - (Yb > 0)) / len(idx)
            g = [Fb.T @ dZ, dZ.sum(0)]
            for i, (pp, gg) in enumerate(zip(P, g)):
                M[i] = 0.9 * M[i] + 0.1 * gg; V[i] = 0.999 * V[i] + 0.001 * gg ** 2
                pp -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)
            if t % 1000 == 0 or t == steps:
                bestA = max(bestA, float((decode(Fte @ W + b) == yte).mean()))
        print(f"       (a) 线性  225->{h}        准确率 {bestA:.4f}  参数 {d_in*h+h}")

        # (b) 隐藏层 225 -> H1 -> h
        H1 = 128
        W1 = np.zeros((d_in, H1)); b1 = np.zeros(H1)
        W2 = np.zeros((H1, h)); b2 = np.zeros(h)
        P = [W1, b1, W2, b2]
        M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
        bestB = 0.0
        rng2 = np.random.default_rng(1)
        for t in range(1, steps + 1):
            idx = rng2.integers(0, n, bs)
            Fb, Yb = Ftr[idx], Ytr[idx]
            h1 = np.maximum(Fb @ W1 + b1, 0)
            Z = h1 @ W2 + b2
            Pb = 1 / (1 + np.exp(-np.clip(Z, -30, 30)))
            dZ = (Pb - (Yb > 0)) / len(idx)
            gW2 = h1.T @ dZ; gb2 = dZ.sum(0)
            dh = (dZ @ W2.T) * (h1 > 0)
            gW1 = Fb.T @ dh; gb1 = dh.sum(0)
            g = [gW1, gb1, gW2, gb2]
            for i, (pp, gg) in enumerate(zip(P, g)):
                M[i] = 0.9 * M[i] + 0.1 * gg; V[i] = 0.999 * V[i] + 0.001 * gg ** 2
                pp -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)
            if t % 1000 == 0 or t == steps:
                h1t = np.maximum(Fte @ W1 + b1, 0)
                bestB = max(bestB, float((decode(h1t @ W2 + b2) == yte).mean()))
        nparam = d_in*H1 + H1 + H1*h + h
        print(f"       (b) 隐藏  225->{H1}->{h}   准确率 {bestB:.4f}  参数 {nparam}\n", flush=True)
        out.append({'h': h, 'NC': NC, 'waste': NC - 10, 'dmin': int(dmin),
                    'sim_dist': float(v/w), 'acc_linear': bestA, 'acc_hidden': bestB,
                    'param_linear': d_in*h+h, 'param_hidden': nparam})

    print("=" * 96)
    print("### 汇总（方案 B: h 个独立 sigmoid）")
    print(f"  {'h':>3} | {'码数':>5} | {'浪费':>6} | {'最小码距':>7} | "
          f"{'线性准确率':>9} | {'隐藏层准确率':>11}")
    print("  " + "-" * 68)
    for s in out:
        print(f"  {s['h']:>3} | {s['NC']:>5} | {s['waste']:>6} | {s['dmin']:>7} | "
              f"{s['acc_linear']:>9.4f} | {s['acc_hidden']:>11.4f}")
    print(f"\n  对照: 225->10 argmax 0.8915   方案 C(h=4) 0.8936")
    print("=" * 96)
    json.dump(out, open(os.path.join(RES, 'digit_codeB_sweep.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()

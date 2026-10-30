#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
方案 B 的正确版本: h 位独立输出 + **每数字占据一组紧密的码**
==============================================================
之前的两处错误:
  1. 只给每数字 1 个码（应该给 2^h/10 个，h=6 时约 6.4 个）
  2. 用"最近码解码"（应该用"落在码集内即算对"，这才是容错）

正确设计（czm 的原始要求）:
  * 2^h 个码**全部**分配给 10 个数字（浪费 0）
  * 同一数字的码**彼此接近**（组内汉明距离小）
  * 相似数字的码集**相距近**（近类次相似）
  * 判据: 输出码落在正确数字的码集内

输出: h 个独立 sigmoid（红石只需 h 条线）
对照: 10 路 argmax 0.8915 / 方案 C(h=4, 2^h softmax) 0.8936
"""
import os, json
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


def solve_assignment(h, restarts=40, iters=6000, seed=0, w_sim=0.5, w_intra=1.5):
    """
    把 2^h 个码**全部**分给 10 个数字:
      目标 = 组内紧凑 (w_intra) + 相似数字的组间距离小 (w_sim) + 组大小均匀
    """
    NC = 1 << h
    C = np.array([[int(b) for b in format(i, f'0{h}b')] for i in range(NC)], dtype=int)
    D = (C[:, None, :] != C[None, :, :]).sum(2)
    base = NC // 10; rem = NC - base * 10
    tgt = np.array([base + (1 if i < rem else 0) for i in range(10)], float)

    PAIRS = [(a, b) for a in range(10) for b in range(a + 1, 10)]

    def cost(asg):
        cnt = np.bincount(asg, minlength=10).astype(float)
        if (cnt == 0).any():
            return 1e9
        c = 2.0 * np.abs(cnt - tgt).sum()
        # 组内紧凑
        for d in range(10):
            ix = np.flatnonzero(asg == d)
            if len(ix) > 1:
                c += w_intra * D[np.ix_(ix, ix)].sum() / (len(ix) * (len(ix) - 1))
        # 相似数字的组间距离（用最小距离）
        for a, b in PAIRS:
            s = sim(a, b)
            if s <= 0:
                continue
            ia = np.flatnonzero(asg == a); ib = np.flatnonzero(asg == b)
            c += w_sim * s * D[np.ix_(ia, ib)].min()
        # 不相似数字的组要分开（用组间平均距离的负值）
        for a, b in PAIRS:
            s = sim(a, b)
            if s > 0.2:
                continue
            ia = np.flatnonzero(asg == a); ib = np.flatnonzero(asg == b)
            c -= 0.4 * D[np.ix_(ia, ib)].mean()
        return c

    rng = np.random.default_rng(seed); best = None
    for r in range(restarts):
        asg = np.repeat(np.arange(10), tgt.astype(int))
        extra = NC - len(asg)
        if extra > 0:
            asg = np.concatenate([asg, rng.integers(0, 10, extra)])
        asg = asg[:NC]; rng.shuffle(asg)
        c = cost(asg); T = 2.5
        for it in range(iters):
            k = rng.integers(0, NC); old = asg[k]; new = rng.integers(0, 10)
            if new == old:
                continue
            asg[k] = new; c2 = cost(asg)
            if c2 < c or rng.random() < np.exp(-(c2 - c) / max(T, 1e-9)):
                c = c2
            else:
                asg[k] = old
            T *= 0.9995
        if best is None or c < best[0]:
            best = (c, asg.copy())
    return best[1], C, D


def main():
    print("=" * 96)
    print("方案 B 正确版: h 位独立输出 + 每数字占一组紧密的码")
    print("=" * 96)
    Xtr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte')) if False else \
        load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
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
        asg, C, D = solve_assignment(h)
        cnt = np.bincount(asg, minlength=10)
        # 组内平均距离 与 相似对的组间最小距离
        intra = np.mean([D[np.ix_(np.flatnonzero(asg == d), np.flatnonzero(asg == d))].mean()
                         for d in range(10)])
        v = w = 0.0
        for (a, b), s in SIM.items():
            ia = np.flatnonzero(asg == a); ib = np.flatnonzero(asg == b)
            v += s * D[np.ix_(ia, ib)].min(); w += s
        # 代表码（组内到其它码距离和最小）
        rep = np.zeros(10, int)
        for d in range(10):
            ix = np.flatnonzero(asg == d)
            rep[d] = ix[D[np.ix_(ix, ix)].sum(1).argmin()]
        print(f"[h={h}]  码数 {NC}  每数字码数 {cnt.tolist()}  （平均 {cnt.mean():.1f}）")
        print(f"       组内平均距离 {intra:.2f}   相似对组间最小码距 {v/w:.2f}")

        # ---- 训练: h 个独立 sigmoid, 目标 = 代表码 ----
        Ytr = (C[rep[ytr]] * 2 - 1).astype(np.float64)
        rng = np.random.default_rng(0)
        W = np.zeros((d_in, h)); b = np.zeros(h)
        P = [W, b]; M = [np.zeros_like(W), np.zeros_like(b)]
        V = [np.zeros_like(W), np.zeros_like(b)]
        bestIn, bestNear = 0.0, 0.0
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
                Zt = Fte @ W + b
                code = (Zt > 0).astype(int)
                codeidx = code.dot(1 << np.arange(h)[::-1])
                acc_in = float((asg[codeidx] == yte).mean())        # 落在码集内
                # 对照: 最近码解码
                hard = code * 2.0 - 1.0
                d2 = ((hard[:, None, :] - (C * 2 - 1)[None, :, :]) ** 2).sum(2)
                acc_near = float((asg[d2.argmin(1)] == yte).mean())
                bestIn = max(bestIn, acc_in); bestNear = max(bestNear, acc_near)
        nparam = d_in * h + h
        print(f"       准确率: 落在码集内 {bestIn:.4f}   最近码解码 {bestNear:.4f}   "
              f"参数 {nparam}   输出线 {h}\n", flush=True)
        out.append({'h': h, 'NC': NC, 'cnt': cnt.tolist(), 'intra': float(intra),
                    'sim_min': float(v / w), 'acc_in': bestIn, 'acc_near': bestNear,
                    'param': nparam})

    print("=" * 96)
    print("### 汇总")
    print(f"  {'h':>3} | {'码数':>5} | {'每数字码数':>10} | {'输出线':>6} | {'参数':>6} | "
          f"{'落在码集内':>10} | {'最近码':>8}")
    print("  " + "-" * 76)
    for s in out:
        print(f"  {s['h']:>3} | {s['NC']:>5} | {np.mean(s['cnt']):>10.1f} | {s['h']:>6} | "
              f"{s['param']:>6} | {s['acc_in']:>10.4f} | {s['acc_near']:>8.4f}")
    print(f"\n  对照: 10路argmax 0.8915   方案C(2^h softmax, h=4) 0.8936")
    print("=" * 96)
    json.dump(out, open(os.path.join(RES, 'digit_codeB_correct.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()

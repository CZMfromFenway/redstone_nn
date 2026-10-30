#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
方案 C 的目标函数变体对比
===========================
C1 (我之前用的): 多正类，每个码 1/|codes| —— 网络要在多个码间分摊概率
C2 (更自然):     只学代表码（概率1），其余码当"备选"容错
C3:              混合 —— 代表码权重 0.8，备选码共 0.2

假设: C2 应该优于 C1，因为训练信号不冲突，而多码的容错红利仍在。

另外检验: 码集是否真的保住了"相似数字码相近"
"""
import os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
RES = os.path.join(ROOT, "results")

# 上次搜索得到的 h=4 码集
ASG = {}
for d, cs in {0: [0b0101, 0b0111], 1: [0b0001, 0b0011], 2: [0b1000, 0b1001],
              3: [0b0100, 0b0110], 4: [0b0000, 0b1010], 5: [0b1100, 0b1101],
              6: [0b0010], 7: [0b1011], 8: [0b1111], 9: [0b1110]}.items():
    for c in cs:
        ASG[c] = d
asg = np.array([ASG[c] for c in range(16)], dtype=int)

SIM = {(3, 8): 1.0, (6, 9): 0.9, (4, 9): 0.8, (1, 7): 0.7, (0, 8): 0.7,
       (5, 8): 0.6, (0, 6): 0.6, (3, 5): 0.6, (2, 7): 0.5, (4, 7): 0.5,
       (0, 9): 0.5, (1, 4): 0.4, (2, 3): 0.4, (5, 6): 0.3}


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


def main():
    print("=" * 96)
    print("方案 C 的目标函数变体 + 码集相似性检验")
    print("=" * 96)
    C16 = np.array([[int(b) for b in format(i, '04b')] for i in range(16)], dtype=int)
    D = (C16[:, None, :] != C16[None, :, :]).sum(2)

    # ---------- 检验 1: 码集是否保住相似性 ----------
    print("\n[检验 1] 码集是否保住'相似数字的码相近'？")
    print(f"  {'数字对':>8} | {'相似度':>6} | {'最小码距':>8} | {'平均码距':>8} | 评价")
    print("  " + "-" * 62)
    tot_s, tot_d = 0.0, 0.0
    for (a, b), s in sorted(SIM.items(), key=lambda kv: -kv[1]):
        ia = np.flatnonzero(asg == a); ib = np.flatnonzero(asg == b)
        dd = D[np.ix_(ia, ib)]
        ok = "✓ 相近" if dd.min() <= 2 else ("△ 中等" if dd.min() <= 3 else "✗ 偏远")
        print(f"  {a}-{b:>6} | {s:>6.1f} | {dd.min():>8} | {dd.mean():>8.2f} | {ok}")
        tot_s += s; tot_d += s * dd.min()
    print(f"\n  相似度加权平均最小码距 = {tot_d/tot_s:.2f}  （越小越好）")
    # 对照: 随机分配
    rng = np.random.default_rng(0)
    rnd = []
    for _ in range(2000):
        a2 = rng.integers(0, 10, 16)
        if len(np.unique(a2)) < 10:
            continue
        v = 0.0; w = 0.0
        for (a, b), s in SIM.items():
            ia = np.flatnonzero(a2 == a); ib = np.flatnonzero(a2 == b)
            if len(ia) == 0 or len(ib) == 0:
                continue
            v += s * D[np.ix_(ia, ib)].min(); w += s
        if w > 0:
            rnd.append(v / w)
    print(f"  随机分配的同样量 = {np.mean(rnd):.2f} ± {np.std(rnd):.2f}")
    print(f"  -> 优化后 {'更好' if tot_d/tot_s < np.mean(rnd) else '没有改善'}")

    # ---------- 检验 2: 目标函数变体 ----------
    Xtr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    A = to15(Xtr); B = to15(Xte); thr = A.mean()
    Ftr = (A > thr).astype(np.float64).reshape(len(A), -1)
    Fte = (B > thr).astype(np.float64).reshape(len(B), -1)
    d_in = Ftr.shape[1]; n = len(Ftr)

    # 每个数字的码列表 与 代表码（组内到其它码距离和最小）
    codes_of = {d: np.flatnonzero(asg == d) for d in range(10)}
    rep = {d: codes_of[d][D[np.ix_(codes_of[d], codes_of[d])].sum(1).argmin()] for d in range(10)}
    print(f"\n[码集]")
    for d in range(10):
        cs = codes_of[d]
        rp = rep[d]
        mark = ' '.join(('*'+format(c,'04b') if c==rp else ' '+format(c,'04b')) for c in cs)
        print(f"  {d}: {mark}   (* = 代表码)")

    def build_T(alpha):
        """alpha = 代表码的权重；其余分给备选码"""
        T = np.zeros((10, 16))
        for d in range(10):
            cs = codes_of[d]; rp = rep[d]
            others = [c for c in cs if c != rp]
            if not others:
                T[d, rp] = 1.0
            else:
                T[d, rp] = alpha
                for c in others:
                    T[d, c] = (1 - alpha) / len(others)
        return T

    steps, lr, bs = 4000, 0.05, 128

    def run(T, name):
        Tg = T[ytr]
        rng = np.random.default_rng(0)
        W = np.zeros((d_in, 16)); b = np.zeros(16)
        M = [np.zeros_like(W), np.zeros_like(b)]; V = [np.zeros_like(W), np.zeros_like(b)]
        P = [W, b]; best = 0.0
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
        return best

    print(f"\n[检验 2] 目标函数变体（alpha = 代表码的目标权重）")
    print(f"  {'变体':<34} | {'准确率':>7}")
    print("  " + "-" * 46)
    res = {}
    for alpha, nm in ((0.5, 'C1 均分（每个码 1/|codes|）'),
                      (0.8, 'C3a 代表 0.8 + 备选 0.2'),
                      (0.95, 'C3b 代表 0.95 + 备选 0.05'),
                      (1.0, 'C2 只学代表码（备选当容错）')):
        T = build_T(alpha)
        if alpha == 1.0:      # 单码目标
            T = np.zeros((10, 16))
            for d in range(10):
                T[d, rep[d]] = 1.0
        acc = run(T, nm)
        res[nm] = acc
        print(f"  {nm:<34} | {acc:>7.4f}", flush=True)

    print(f"\n  对照: 225->10 argmax = 0.8915（同协议）")
    json.dump({'similarity_check': {'weighted_min_dist': tot_d / tot_s,
                                    'random_baseline': float(np.mean(rnd))},
               'variants': res}, open(os.path.join(RES, 'digit_codeC.json'), 'w'),
              ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

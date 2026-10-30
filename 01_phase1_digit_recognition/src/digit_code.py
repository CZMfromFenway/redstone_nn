#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
225 -> h 二进制编码的数字识别
================================
思路（czm）: 不用 10 路 argmax，改成 h 个二值输出（h=4~6），
  每个数字对应 2^h 个码中的**一组**（占满全部码）。

约束:
  1. 占满: 所有 2^h 个码都分配给某个数字
  2. 同类相似: 同一数字的码之间汉明距离尽量小
  3. 近类次相似: 视觉相似数字（3/8, 6/9, 4/9, 1/7）的码距离次小

评价: 预测码落在正确数字的码集内即算对
对照: 225->10 线性 argmax = 0.8915（此前实测）

用法: python digit_code.py [h]      h 默认 4
"""
import os, json, sys, itertools, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
RES = os.path.join(ROOT, "results")

# 数字相似度矩阵（先验，0=最不相似, 1=最相似）
SIM_PAIRS = {
    (3, 8): 1.0, (6, 9): 0.9, (4, 9): 0.8, (1, 7): 0.7,
    (5, 8): 0.6, (0, 8): 0.7, (0, 6): 0.6, (2, 7): 0.5,
    (3, 5): 0.6, (4, 7): 0.5, (0, 9): 0.5, (1, 4): 0.4,
}


def load_idx(path):
    import struct
    with open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        ndim = magic & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(ndim)]
        d = np.frombuffer(f.read(), dtype=np.uint8)
    return d.reshape(dims)


def to15(imgs):
    n = len(imgs)
    a = imgs.reshape(n, 28, 28).astype(np.float32).reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    o = np.zeros((n, 15, 15), dtype=np.float32)
    o[:, :14, :14] = a; o[:, 14, :14] = a[:, 13, :]
    o[:, :14, 14] = a[:, :, 13]; o[:, 14, 14] = a[:, 13, 13]
    return o


def sim(d, e):
    """两个数字的相似度（对称）"""
    if d == e:
        return 1.0
    return SIM_PAIRS.get((d, e), SIM_PAIRS.get((e, d), 0.0))


def search_codeset(h, restarts=40, iters=3000, seed=0):
    """
    搜索码集分配: 把 2^h 个码划成 10 组
    目标: 组内距离小 + 相似数字的组间距离小
    """
    NC = 1 << h
    CODES = np.array([[int(b) for b in format(i, f'0{h}b')] for i in range(NC)], dtype=int)
    DIST = (CODES[:, None, :] != CODES[None, :, :]).sum(2)      # (NC,NC) 汉明距离

    def cost(asg):
        # asg: (NC,) 每个码属于哪个数字
        tot = 0.0
        for dd in range(10):
            idx = np.flatnonzero(asg == dd)
            if len(idx) == 0:
                return 1e9                        # 每个数字至少要一个码
            if len(idx) > 1:
                tot += DIST[np.ix_(idx, idx)].sum() / (len(idx) * (len(idx) - 1)) * 1.0
        # 组间: 相似数字的码应该更近
        for d1 in range(10):
            for d2 in range(d1 + 1, 10):
                s = sim(d1, d2)
                if s <= 0:
                    continue
                i1 = np.flatnonzero(asg == d1); i2 = np.flatnonzero(asg == d2)
                # 最小跨组距离
                m = DIST[np.ix_(i1, i2)].min()
                tot += (1.0 - s) * m * 2.0
        return tot

    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        # 初始: 每个码随机分配，保证每组非空
        asg = rng.integers(0, 10, NC)
        asg[rng.choice(NC, 10, replace=False)] = np.arange(10)
        c = cost(asg)
        T = 3.0
        for it in range(iters):
            k = rng.integers(0, NC)
            old = asg[k]
            new = rng.integers(0, 10)
            if new == old:
                continue
            if (asg == old).sum() <= 1:
                continue
            asg[k] = new
            c2 = cost(asg)
            if c2 < c or rng.random() < np.exp(-(c2 - c) / max(T, 1e-9)):
                c = c2
            else:
                asg[k] = old
            T *= 0.998
        if best is None or c < best[0]:
            best = (c, asg.copy())
    return best[1], best[0], CODES, DIST


def main():
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print("=" * 96)
    print(f"225 -> {h} 位编码的数字识别")
    print("=" * 96)

    # ---------- 搜索码集 ----------
    t0 = time.time()
    asg, c, CODES, DIST = search_codeset(h, restarts=30, iters=2500, seed=0)
    NC = 1 << h
    print(f"\n[1] 码集分配（{NC} 个码 -> 10 个数字）  [{time.time()-t0:.0f}s]")
    print(f"  {'数字':>4} | {'码数':>4} | 码（二进制）")
    print("  " + "-" * 60)
    for d in range(10):
        idx = np.flatnonzero(asg == d)
        b = [' '.join(map(str, CODES[i])) for i in idx]
        print(f"  {d:>4} | {len(idx):>4} | {', '.join(b)}")

    # 组内平均距离
    intra = []
    for d in range(10):
        idx = np.flatnonzero(asg == d)
        intra.append(DIST[np.ix_(idx, idx)].mean() if len(idx) > 1 else 0.0)
    print(f"\n  组内平均汉明距离: {[round(x,2) for x in intra]}")
    print(f"  总体组内平均: {np.mean([x for x in intra]):.3f}  （理想 = 1.0）")

    # 相似数字的跨组最小距离
    print(f"\n  相似数字的码距离（应较小）:")
    for (d1, d2) in sorted(SIM_PAIRS, key=lambda p: -SIM_PAIRS[p])[:6]:
        i1 = np.flatnonzero(asg == d1); i2 = np.flatnonzero(asg == d2)
        print(f"    {d1}-{d2} (相似度 {SIM_PAIRS[(d1,d2)]:.1f}): 最小距离 "
              f"{DIST[np.ix_(i1,i2)].min()}  平均 {DIST[np.ix_(i1,i2)].mean():.2f}")

    # ---------- 数据 ----------
    Xtr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    A = to15(Xtr); B = to15(Xte)
    thr = A.mean()
    Ftr = (A > thr).astype(np.float64).reshape(len(A), -1)
    Fte = (B > thr).astype(np.float64).reshape(len(B), -1)
    print(f"\n[2] 数据: 15x15 二值  train {Ftr.shape}  test {Fte.shape}")

    # 目标: 每个样本一个"码"，用该数字的码中**最近的那个**（训练时用全部码做多标签）
    # 做法: 把 2^h 个码当作类别，损失 = 该数字全部码上的 softmax（多正类交叉熵）
    valid = np.zeros((10, NC), dtype=np.float64)
    for d in range(10):
        valid[d, asg == d] = 1.0
    valid /= valid.sum(1, keepdims=True)
    print(f"  每个数字可用码数: {valid.sum(1).astype(int).tolist()}")

    # ---------- 训练 ----------
    W = np.zeros((Ftr.shape[1], NC)); b = np.zeros(NC)
    M = [np.zeros_like(W), np.zeros_like(b)]; V = [np.zeros_like(W), np.zeros_like(b)]
    ps = [W, b]
    rng = np.random.default_rng(0)
    steps, lr, bs = 4000, 0.05, 128
    n = len(Ftr)
    best = 0.0
    t0 = time.time()
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        Fb, yb = Ftr[idx], ytr[idx]
        Z = Fb @ W + b
        mx = Z.max(1, keepdims=True)
        e = np.exp(Z - mx); P = e / e.sum(1, keepdims=True)
        # 多正类: 目标分布 = 该数字的码上均匀
        Tt = valid[yb]                                  # (bs,NC)
        dZ = (P - Tt) / bs
        gW = Fb.T @ dZ; gb = dZ.sum(0)
        for i, (pp, gg) in enumerate(zip(ps, [gW, gb])):
            M[i] = 0.9 * M[i] + 0.1 * gg; V[i] = 0.999 * V[i] + 0.001 * gg ** 2
            pp -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == steps:
            Zt = Fte @ W + b
            pred = Zt.argmax(1)
            acc = float((asg[pred] == yte).mean())
            best = max(best, acc)
            print(f"    step {t:5d}: 测试准确率 {acc:.4f}  （最好 {best:.4f}）", flush=True)

    # ---------- 对照: 225->10 argmax ----------
    print(f"\n[3] 对照: 225 -> 10 线性 argmax")
    W2 = np.zeros((Ftr.shape[1], 10)); b2 = np.zeros(10)
    M2 = [np.zeros_like(W2), np.zeros_like(b2)]; V2 = [np.zeros_like(W2), np.zeros_like(b2)]
    ps2 = [W2, b2]; best2 = 0.0
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        Fb, yb = Ftr[idx], ytr[idx]
        Z = Fb @ W2 + b2
        mx = Z.max(1, keepdims=True)
        e = np.exp(Z - mx); P = e / e.sum(1, keepdims=True)
        dZ = P.copy(); dZ[np.arange(bs), yb] -= 1; dZ /= bs
        gW = Fb.T @ dZ; gb = dZ.sum(0)
        for i, (pp, gg) in enumerate(zip(ps2, [gW, gb])):
            M2[i] = 0.9 * M2[i] + 0.1 * gg; V2[i] = 0.999 * V2[i] + 0.001 * gg ** 2
            pp -= lr * (M2[i] / (1 - 0.9 ** t)) / (np.sqrt(V2[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 2000 == 0 or t == steps:
            acc = float(((Fte @ W2 + b2).argmax(1) == yte).mean())
            best2 = max(best2, acc)
    print(f"    225->10 最好 {best2:.4f}")

    print(f"\n{'='*96}")
    print(f"### 结果  h={h}")
    print(f"  225->{h} 编码（{NC} 个码，{10} 数字）  最好 {best:.4f}")
    print(f"  225->10 argmax                       最好 {best2:.4f}")
    print(f"  参数: 编码 {Ftr.shape[1]*NC}  vs  argmax {Ftr.shape[1]*10}")
    print(f"{'='*96}")
    json.dump({'h': h, 'acc_code': best, 'acc_argmax': best2,
               'assignment': asg.tolist(), 'intra': [float(x) for x in intra]},
              open(os.path.join(RES, f'digit_code_h{h}.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()

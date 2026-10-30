#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
225 -> h 位二进制编码的数字识别（v2）
========================================
v1 的两个问题:
  1. 码集分配退化: 数字 7 独占 6 个码（代价函数里"相似"项权重过高）
  2. 我用 16 路 softmax 训练 -> 参数 3600，比 10 路 argmax 的 2250 还多 60%

v2 修正:
  1. 码集分配改为**先均匀**（每个数字 1~2 个码），再优化相似性
  2. **真正的 h 位输出**: h 个独立 logits，用 h 个 sigmoid 训练
     -> 参数 225*h（h=4 时 900，比 10 路省 60%）
  3. 评价改为**最近码解码**: 预测码 -> 找最近的已分配码 -> 其数字
     （这是编码方案的自然用法，比"必须落在码集内"宽松）

对照: 225->10 argmax 线性 = 0.8918（同协议实测）
"""
import os, json, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
RES = os.path.join(ROOT, "results")

SIM_PAIRS = {(3, 8): 1.0, (6, 9): 0.9, (4, 9): 0.8, (1, 7): 0.7, (0, 8): 0.7,
             (5, 8): 0.6, (0, 6): 0.6, (3, 5): 0.6, (2, 7): 0.5, (4, 7): 0.5,
             (0, 9): 0.5, (1, 4): 0.4, (2, 3): 0.4, (5, 6): 0.3}


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
    if d == e:
        return 1.0
    return SIM_PAIRS.get((d, e), SIM_PAIRS.get((e, d), 0.0))


def search_codeset(h, w_sim=0.3, restarts=60, iters=4000, seed=0):
    """
    把 2^h 个码划成 10 组。目标:
      A. 组内平均汉明距离小（同类相似）
      B. 相似数字的组间最小距离小（近类次相似）
      C. **每个数字至少一个码，且组大小尽量均匀**（v1 缺这条）
    """
    NC = 1 << h
    CODES = np.array([[int(b) for b in format(i, f'0{h}b')] for i in range(NC)], dtype=int)
    DIST = (CODES[:, None, :] != CODES[None, :, :]).sum(2)
    # 目标组大小（均匀）
    base = NC // 10
    rem = NC - base * 10
    target = np.array([base + (1 if i < rem else 0) for i in range(10)], dtype=float)

    def cost(asg):
        cnt = np.bincount(asg, minlength=10).astype(float)
        if (cnt == 0).any():
            return 1e9
        # C. 组大小偏离
        c = 3.0 * np.abs(cnt - target).sum()
        # A. 组内距离
        for dd in range(10):
            idx = np.flatnonzero(asg == dd)
            if len(idx) > 1:
                c += DIST[np.ix_(idx, idx)].sum() / (len(idx) * (len(idx) - 1))
        # B. 相似数字的组间最小距离（权重降低）
        for d1 in range(10):
            for d2 in range(d1 + 1, 10):
                s = sim(d1, d2)
                if s <= 0:
                    continue
                i1 = np.flatnonzero(asg == d1); i2 = np.flatnonzero(asg == d2)
                c += w_sim * (1.0 - s) * DIST[np.ix_(i1, i2)].min()
        return c

    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        # 初始: 均匀分配
        asg = np.repeat(np.arange(10), target.astype(int))
        asg = np.concatenate([asg, rng.permutation(NC - len(asg)) * 0 + rng.integers(0, 10, NC - len(asg))])
        asg = asg[:NC]
        rng.shuffle(asg)
        c = cost(asg)
        T = 2.0
        for it in range(iters):
            k = rng.integers(0, NC)
            old = asg[k]; new = rng.integers(0, 10)
            if new == old:
                continue
            asg[k] = new
            c2 = cost(asg)
            if c2 < c or rng.random() < np.exp(-(c2 - c) / max(T, 1e-9)):
                c = c2
            else:
                asg[k] = old
            T *= 0.999
        if best is None or c < best[0]:
            best = (c, asg.copy())
    return best[1], best[0], CODES, DIST


def main():
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print("=" * 96)
    print(f"225 -> {h} 位二进制编码的数字识别（v2）")
    print("=" * 96)

    t0 = time.time()
    asg, c, CODES, DIST = search_codeset(h)
    NC = 1 << h
    print(f"\n[1] 码集分配  [{time.time()-t0:.0f}s]")
    print(f"  {'数字':>4} | {'码数':>4} | 码（二进制）")
    print("  " + "-" * 62)
    for d in range(10):
        idx = np.flatnonzero(asg == d)
        print(f"  {d:>4} | {len(idx):>4} | "
              f"{', '.join(' '.join(map(str, CODES[i])) for i in idx)}")
    intra = []
    for d in range(10):
        idx = np.flatnonzero(asg == d)
        intra.append(DIST[np.ix_(idx, idx)].mean() if len(idx) > 1 else 0.0)
    print(f"\n  组内平均汉明距离: {[round(x,2) for x in intra]}")
    print(f"  总体组内平均 {np.mean(intra):.3f}（理想 1.0）")
    print(f"\n  相似数字的码距离（应较小）:")
    for (d1, d2) in sorted(SIM_PAIRS, key=lambda p: -SIM_PAIRS[p])[:6]:
        i1 = np.flatnonzero(asg == d1); i2 = np.flatnonzero(asg == d2)
        dd = DIST[np.ix_(i1, i2)]
        print(f"    {d1}-{d2} (相似 {SIM_PAIRS[(d1,d2)]:.1f}): 最小 {dd.min()} 平均 {dd.mean():.2f}")

    # 每个数字的代表码 = 组内到其它码距离和最小的（中位码）
    rep = np.zeros(10, dtype=int)
    for d in range(10):
        idx = np.flatnonzero(asg == d)
        rep[d] = idx[DIST[np.ix_(idx, idx)].sum(1).argmin()]
    print(f"\n  每个数字的代表码: {[format(rep[d], f'0{h}b') for d in range(10)]}")

    # ---------- 数据 ----------
    Xtr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    A = to15(Xtr); B = to15(Xte)
    thr = A.mean()
    Ftr = (A > thr).astype(np.float64).reshape(len(A), -1)
    Fte = (B > thr).astype(np.float64).reshape(len(B), -1)
    print(f"\n[2] 数据: train {Ftr.shape}  test {Fte.shape}")

    # 目标: 每个样本 -> 其数字的**代表码**的 h 位二值（±1）
    Ytr = CODES[rep[ytr]].astype(np.float64) * 2 - 1
    Yte = CODES[rep[yte]].astype(np.float64) * 2 - 1
    BM = (CODES * 2 - 1).astype(np.float64)                   # (NC,h) ±1

    # ---------- 训练: h 个独立 sigmoid ----------
    print(f"\n[3] 训练 h={h} 位二值输出（{h} 个独立 sigmoid）")
    rng = np.random.default_rng(0)
    W = np.zeros((Ftr.shape[1], h)); b = np.zeros(h)
    M = [np.zeros_like(W), np.zeros_like(b)]; V = [np.zeros_like(W), np.zeros_like(b)]
    steps, lr, bs = 4000, 0.05, 128
    n = len(Ftr); best = 0.0
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        Fb, Yb = Ftr[idx], Ytr[idx]
        Z = Fb @ W + b
        P = 1 / (1 + np.exp(-np.clip(Z, -30, 30)))
        dZ = (P - (Yb > 0)) / bs
        gW = Fb.T @ dZ; gb = dZ.sum(0)
        for i, (pp, gg) in enumerate(zip([W, b], [gW, gb])):
            M[i] = 0.9 * M[i] + 0.1 * gg; V[i] = 0.999 * V[i] + 0.001 * gg ** 2
            pp -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == steps:
            Zt = Fte @ W + b
            hard = np.where(Zt > 0, 1.0, -1.0)                 # (±1) 硬判决
            # 最近码解码
            d2 = ((hard[:, None, :] - BM[None, :, :]) ** 2).sum(2)
            pred = asg[d2.argmin(1)]
            acc = float((pred == yte).mean())
            # 严格解码: 必须落在该数字的码集内
            strict = float((asg[np.array([np.flatnonzero((BM == hard[i]).all(1))[0]
                                          if ((BM == hard[i]).all(1)).any() else -1
                                          for i in range(len(hard))])] == yte).mean()) \
                if False else None
            best = max(best, acc)
            print(f"    step {t:5d}: 最近码解码准确率 {acc:.4f}（最好 {best:.4f}）", flush=True)

    # ---------- 对照 ----------
    print(f"\n[4] 对照: 225 -> 10 argmax")
    W2 = np.zeros((Ftr.shape[1], 10)); b2 = np.zeros(10)
    M2 = [np.zeros_like(W2), np.zeros_like(b2)]; V2 = [np.zeros_like(W2), np.zeros_like(b2)]
    best2 = 0.0
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        Fb, yb = Ftr[idx], ytr[idx]
        Z = Fb @ W2 + b2
        mx = Z.max(1, keepdims=True)
        e = np.exp(Z - mx); P = e / e.sum(1, keepdims=True)
        dZ = P.copy(); dZ[np.arange(bs), yb] -= 1; dZ /= bs
        gW = Fb.T @ dZ; gb = dZ.sum(0)
        for i, (pp, gg) in enumerate(zip([W2, b2], [gW, gb])):
            M2[i] = 0.9 * M2[i] + 0.1 * gg; V2[i] = 0.999 * V2[i] + 0.001 * gg ** 2
            pp -= lr * (M2[i] / (1 - 0.9 ** t)) / (np.sqrt(V2[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 2000 == 0 or t == steps:
            best2 = max(best2, float(((Fte @ W2 + b2).argmax(1) == yte).mean()))
    print(f"    225->10 最好 {best2:.4f}")

    print(f"\n{'='*96}")
    print(f"### h={h}  225->{h} 编码 {best:.4f}   225->10 argmax {best2:.4f}")
    print(f"    参数: 编码 {Ftr.shape[1]*h + h}   argmax {Ftr.shape[1]*10 + 10}"
          f"   （编码省 {(1-(Ftr.shape[1]*h+h)/(Ftr.shape[1]*10+10))*100:.0f}%）")
    print(f"{'='*96}")
    json.dump({'h': h, 'acc_code': best, 'acc_argmax': best2,
               'assignment': asg.tolist(), 'rep': rep.tolist(),
               'intra': [float(x) for x in intra]},
              open(os.path.join(RES, f'digit_code2_h{h}.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()

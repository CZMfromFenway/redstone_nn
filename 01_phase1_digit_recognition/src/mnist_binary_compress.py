#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
继续压缩二值特征数：两档阈值 + 邻域 OR 抽稀
=============================================
已知: [1,4,9] 三档 = 675 特征, 0.9069 (全量 MNIST)
目标: 把特征数压到 450 以下, 看准确率还剩多少
"""
import numpy as np, os, struct, json, time

RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'


def load_idx(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        ndim = magic & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(dims)


Xtr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))


def to15(imgs):
    n = len(imgs)
    a = imgs.reshape(n, 28, 28).astype(np.float32).reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    out = np.zeros((n, 15, 15), dtype=np.float32)
    out[:, :14, :14] = a
    out[:, 14, :14] = a[:, 13, :]
    out[:, :14, 14] = a[:, :, 13]
    out[:, 14, 14] = a[:, 13, 13]
    return out


Atr, Ate = to15(Xtr), to15(Xte)
thr = Atr.mean()
Btr = (Atr > thr).astype(np.float32)
Bte = (Ate > thr).astype(np.float32)


def conv_all1(B):
    n = len(B)
    P = np.zeros((n, 17, 17), dtype=np.float32); P[:, 1:16, 1:16] = B
    out = np.zeros((n, 15, 15), dtype=np.float32)
    for dr in range(3):
        for dc in range(3):
            out += P[:, dr:dr + 15, dc:dc + 15]
    return out


def or_pool_fixed(Bt, out_size):
    """把 m x m 通过 max 池化到 out_size x out_size（保持二值）"""
    n, m, _ = Bt.shape
    if out_size >= m:
        return Bt
    # 用 np.maximum.reduceat 做不等分块
    edges = np.linspace(0, m, out_size + 1).astype(int)
    rows = np.maximum.reduceat(Bt, edges[:-1], axis=1)
    cols = np.maximum.reduceat(rows, edges[:-1], axis=2)
    return cols


def make_feats(R, taus, out_size):
    outs = []
    for ta in taus:
        Bt = (R >= ta).astype(np.float32)
        if out_size < 15:
            Bt = or_pool_fixed(Bt, out_size)
        outs.append(Bt.reshape(len(R), -1))
    return np.concatenate(outs, axis=1)


def train_linear(Ftr, ytr, Fte, yte, steps=3000, lr=0.05, bs=128, seed=0):
    n, nf = Ftr.shape
    rng = np.random.default_rng(seed)
    W = np.zeros((nf, 10)); b = np.zeros(10)
    ps = [W, b]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    best = 0.0
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        Fb, yb = Ftr[idx], ytr[idx]
        Z = Fb @ W + b
        Pm = np.exp(Z - Z.max(1, keepdims=True)); Pm /= Pm.sum(1, keepdims=True)
        dZ = Pm.copy(); dZ[np.arange(bs), yb] -= 1; dZ /= bs
        g = [Fb.T @ dZ, dZ.sum(0)]
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == steps:
            best = max(best, float(((Fte @ W + b).argmax(1) == yte).mean()))
    return best


Rtr, Rte = conv_all1(Btr), conv_all1(Bte)

print("=" * 100)
print("继续压缩：两档阈值 / 邻域 OR 抽稀（全量 MNIST, 10000 测试）")
print("=" * 100)
print(f"{'阈值集':>14} | {'档':>2} | {'抽稀到':>7} | {'特征数':>7} | {'权重数':>8} | {'准确率':>8}")
print("-" * 100)
rows = []
configs = [
    ([1, 4, 9], 15, "基准(三档/满分辨率)"),
    ([1, 9], 15, "两档 1,9"),
    ([1, 5], 15, "两档 1,5"),
    ([4, 9], 15, "两档 4,9"),
    ([1, 4, 9], 11, "三档 OR->11"),
    ([1, 4, 9], 8, "三档 OR->8"),
    ([1, 4, 9], 5, "三档 OR->5"),
    ([1, 9], 8, "两档 OR->8"),
    ([1, 4, 7, 9], 8, "四档 OR->8"),
]
for taus, osz, name in configs:
    Ftr = make_feats(Rtr, taus, osz); Fte = make_feats(Rte, taus, osz)
    nf = Ftr.shape[1]
    t0 = time.time()
    acc = train_linear(Ftr, ytr, Fte, yte, steps=3000)
    print(f"{name:>14} | {len(taus):>2} | {osz:>7} | {nf:>7} | {nf*10:>8} | {acc:>8.4f}   "
          f"({time.time()-t0:.0f}s)", flush=True)
    rows.append({"taus": taus, "out": osz, "name": name, "nfeat": int(nf),
                 "weights": int(nf * 10), "acc": float(acc)})

json.dump(rows, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "results", "mnist_binary_compress.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已写出: results/mnist_binary_compress.json")

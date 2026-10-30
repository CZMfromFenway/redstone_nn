#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
900 -> 更少二值特征：三档阈值能否保住 0.90
=============================================
【约束】所有特征必须二值（否则火把截断结构用不了）
【关键修正】全 1 卷积的输出范围是 0..9，所以原来的阈值 12 是**废档**（整条通道恒 0）

扫描:
  阈值集 (3 档 / 4 档 / 5 档)，全部落在 1..9 的有效范围内
  + 邻域 OR 抽稀（保持二值）
数据: 全量 MNIST, 10000 测试
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
print(f"全量 MNIST 15x15 二值: train {len(Btr)} test {len(Bte)}  前景 {Btr.mean():.3f}")


def conv_all1(B):
    """全 1 卷积, stride 1, padding 1 -> (n,15,15) 整数 0..9"""
    n = len(B)
    P = np.zeros((n, 17, 17), dtype=np.float32); P[:, 1:16, 1:16] = B
    out = np.zeros((n, 15, 15), dtype=np.float32)
    for dr in range(3):
        for dc in range(3):
            out += P[:, dr:dr + 15, dc:dc + 15]
    return out


def or_pool(Bt, p):
    """对二值图做 pxp 的 OR 池化（保持二值）"""
    n, m, _ = Bt.shape
    g = m // p
    if g < 1:
        return Bt
    Bc = Bt[:, :g * p, :g * p].reshape(n, g, p, g, p)
    return Bc.max(axis=(2, 4))


def make_feats(R, taus, pool):
    outs = []
    for ta in taus:
        Bt = (R >= ta).astype(np.float32)
        if pool > 1:
            Bt = or_pool(Bt, pool)
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


Rtr = conv_all1(Btr)
Rte = conv_all1(Bte)
print(f"全 1 卷积输出范围: train [{Rtr.min():.0f}, {Rtr.max():.0f}]  "
      f"test [{Rte.min():.0f}, {Rte.max():.0f}]")
print(f"  -> 阈值必须落在 1..9;  原来的 tau=12 恒不触发（废档）")

print("\n" + "=" * 100)
print("扫描（全量 MNIST, 10000 测试）")
print("=" * 100)
print(f"{'阈值集':>22} | {'档':>2} | {'OR池':>4} | {'每档格数':>8} | {'特征数':>7} | {'权重数':>8} | {'准确率':>8}")
print("-" * 100)
rows = []
sets = [
    ([1, 4, 8, 12], "1,4,8,12 (旧,含废档)"),
    ([1, 4, 7], "1,4,7"),
    ([1, 4, 9], "1,4,9"),
    ([1, 3, 5, 7, 9], "1,3,5,7,9 (5档)"),
    ([1, 2, 3], "1,2,3 (低端3档)"),
    ([3, 5, 7], "3,5,7 (中端3档)"),
    ([5, 7, 9], "5,7,9 (高端3档)"),
    ([1, 5, 9], "1,5,9 (均匀3档)"),
    ([2, 4, 6, 8], "2,4,6,8 (4档有效)"),
]
for taus, name in sets:
    for pool in (1,):
        Ftr = make_feats(Rtr, taus, pool); Fte = make_feats(Rte, taus, pool)
        nf = Ftr.shape[1]
        t0 = time.time()
        acc = train_linear(Ftr, ytr, Fte, yte, steps=3000)
        print(f"{name:>22} | {len(taus):>2} | {pool:>4} | {nf//len(taus):>8} | {nf:>7} | "
              f"{nf*10:>8} | {acc:>8.4f}   ({time.time()-t0:.0f}s)", flush=True)
        rows.append({"taus": taus, "name": name, "pool": pool, "nfeat": int(nf),
                     "weights": int(nf * 10), "acc": float(acc)})

json.dump(rows, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "results", "mnist_binary_sweep.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已写出: results/mnist_binary_sweep.json")

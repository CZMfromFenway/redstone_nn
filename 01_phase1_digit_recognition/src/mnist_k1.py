#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MNIST 15x15 二值 —— K=1 结构的两条路线
========================================
路线 A: 3x3 卷积 -> ReLU(0..15) -> 3x3 最大池化 -> 5x5=25 特征 -> 致密层
路线 B: 3x3 卷积 -> ReLU(0..15) -> T 个阈值二值化 -> 每通道 3x3 平均池化
        -> T*25 个特征 -> 致密层          (只用 1 个核, 但得到 T 个"通道")

对比项: 测试准确率 / 特征数 / 扇出数 / LUT 单元数
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
    a = imgs.reshape(n, 28, 28).astype(np.float32)
    a = a.reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    out = np.zeros((n, 15, 15), dtype=np.float32)
    out[:, :14, :14] = a
    out[:, 14, :14] = a[:, 13, :]
    out[:, :14, 14] = a[:, :, 13]
    out[:, 14, 14] = a[:, 13, 13]
    return out


Xtr15, Xte15 = to15(Xtr), to15(Xte)
thr = Xtr15.mean()
Btr = (Xtr15 > thr).astype(np.float32)
Bte = (Xte15 > thr).astype(np.float32)
rng = np.random.default_rng(0)
i_tr = rng.choice(len(Btr), 8000, replace=False)
i_te = rng.choice(len(Bte), 2000, replace=False)
Btr, ytr_s = Btr[i_tr], ytr[i_tr]
Bte, yte_s = Bte[i_te], yte[i_te]
print(f"数据: 15x15 二值, train {len(Btr)} test {len(Bte)}  前景占比 {Btr.mean():.3f}")


def conv3x3(B, W):
    n = B.shape[0]
    P = np.zeros((n, 17, 17), dtype=np.float32); P[:, 1:16, 1:16] = B
    out = np.zeros((n, 15, 15), dtype=np.float32)
    for dr in range(3):
        for dc in range(3):
            out += W[dr, dc] * P[:, dr:dr + 15, dc:dc + 15]
    return out


def padB(B):
    n = B.shape[0]
    P = np.zeros((n, 17, 17), dtype=np.float32); P[:, 1:16, 1:16] = B
    return P


def train_A(B, y, Bte, yte, steps=4000, lr=0.02, bs=64, seed=0, verbose=True):
    """路线 A: 3x3 最大池化"""
    n = len(B); rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0, (3, 3)).astype(np.float64)
    Wd = rng.normal(0, 0.1, (25, 10)); bd = np.zeros(10)
    ps = [W, Wd, bd]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    best = 0.0

    def pool(R):
        Rs = R.reshape(-1, 5, 3, 5, 3)
        return Rs.max(axis=(2, 4)).reshape(-1, 25), Rs

    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        Bb, yb = B[idx], y[idx]
        Rc = conv3x3(Bb, W); R = np.clip(Rc, 0, 15)
        P, Rs = pool(R)
        Z = P @ Wd + bd
        Zs = Z - Z.max(1, keepdims=True)
        Pm = np.exp(Zs); Pm /= Pm.sum(1, keepdims=True)
        dZ = Pm.copy(); dZ[np.arange(bs), yb] -= 1; dZ /= bs
        gWd = P.T @ dZ; gbd = dZ.sum(0)
        dP = dZ @ Wd.T
        # 回传到池化前的 argmax 位置
        flat = Rs.reshape(bs, 25, 9)
        am = flat.argmax(axis=2)
        dR = np.zeros((bs, 225))
        rows = np.arange(bs)[:, None]
        dR[rows, np.arange(25)[None, :] * 9 + am] = dP
        dR = dR.reshape(bs, 15, 15)
        dRc = dR * ((Rc > 0) & (Rc < 15))
        Pb = padB(Bb)
        gW = np.zeros((3, 3))
        for dr in range(3):
            for dc in range(3):
                gW[dr, dc] = float((dRc * Pb[:, dr:dr + 15, dc:dc + 15]).sum())
        g = [gW, gWd, gbd]
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == steps:
            Wc = np.clip(np.round(W), -6, 6)
            Rc2 = conv3x3(Bte, Wc); R2 = np.clip(Rc2, 0, 15)
            P2, _ = pool(R2)
            a = float(((P2 @ Wd + bd).argmax(1) == yte).mean())
            best = max(best, a)
            if verbose:
                print(f"    step {t:5d}: 测试 {a:.4f}  核={Wc.astype(int).tolist()}", flush=True)
    return best, W, Wd, bd


def train_B(B, y, Bte, yte, T=4, steps=4000, lr=0.02, bs=64, seed=0, verbose=True):
    """路线 B: T 个阈值二值化 + 3x3 平均池化"""
    n = len(B); rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0, (3, 3)).astype(np.float64)
    taus = np.array([1.0, 4.0, 8.0, 12.0])[:T]
    nf = T * 25
    Wd = rng.normal(0, 0.1, (nf, 10)); bd = np.zeros(10)
    ps = [W, Wd, bd]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    best = 0.0

    def feats(R):
        """R:(b,15,15) 0..15 -> (b, T*25)"""
        Bt = np.stack([(R >= ta).astype(np.float32) for ta in taus], axis=1)   # (b,T,15,15)
        Bp = Bt.reshape(-1, T, 5, 3, 5, 3).mean(axis=(3, 5))                   # (b,T,5,5)
        return Bp.reshape(-1, nf), Bt

    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        Bb, yb = B[idx], y[idx]
        Rc = conv3x3(Bb, W); R = np.clip(Rc, 0, 15)
        F, Bt = feats(R)
        Z = F @ Wd + bd
        Pm = np.exp(Z - Z.max(1, keepdims=True)); Pm /= Pm.sum(1, keepdims=True)
        dZ = Pm.copy(); dZ[np.arange(bs), yb] -= 1; dZ /= bs
        gWd = F.T @ dZ; gbd = dZ.sum(0)
        dF = (dZ @ Wd.T).reshape(bs, T, 5, 5)
        # 平均池化反传: 均匀分到 9 个位置
        dBt = np.repeat(np.repeat(dF / 9.0, 3, axis=2), 3, axis=3)             # (b,T,15,15)
        # 阈值 STE: R >= tau 的导数视作 1(仅在该阈值"活跃"区)
        dR = np.zeros_like(R)
        for ti, ta in enumerate(taus):
            dR += dBt[:, ti]                                   # STE
        dRc = dR * ((Rc > 0) & (Rc < 15))
        Pb = padB(Bb)
        gW = np.zeros((3, 3))
        for dr in range(3):
            for dc in range(3):
                gW[dr, dc] = float((dRc * Pb[:, dr:dr + 15, dc:dc + 15]).sum())
        g = [gW, gWd, gbd]
        for i, (pp, gg) in enumerate(zip(ps, g)):
            ms[i] = 0.9 * ms[i] + 0.1 * gg; vs[i] = 0.999 * vs[i] + 0.001 * gg ** 2
            pp -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == steps:
            Wc = np.clip(np.round(W), -6, 6)
            Rc2 = conv3x3(Bte, Wc); R2 = np.clip(Rc2, 0, 15)
            F2, _ = feats(R2)
            a = float(((F2 @ Wd + bd).argmax(1) == yte).mean())
            best = max(best, a)
            if verbose:
                print(f"    step {t:5d}: 测试 {a:.4f}  核={Wc.astype(int).tolist()}", flush=True)
    return best, W, Wd, bd, taus


# ---------------- 成本估算 ----------------
def cost_A(K=1):
    return {"卷积单元": 225 * K * 9, "池化单元": 25 * K * 9,
            "致密单元": 25 * K * 10, "扇出": 225 * 9 * K}


def cost_B(T=4):
    return {"卷积单元": 225 * 9, "阈值比较器": T * 225, "池化单元": T * 25 * 9,
            "致密单元": T * 25 * 10, "扇出": 225 * 9}


print("\n" + "=" * 92)
print("路线 A: K=1, 3x3 卷积 + ReLU(0..15) + 3x3 最大池化(25 特征) + 致密层")
print("=" * 92)
t0 = time.time()
aA, WA, WdA, bdA = train_A(Btr, ytr_s, Bte, yte_s, steps=4000)
print(f"  => 路线 A 最好 {aA:.4f}   用时 {time.time()-t0:.0f}s   核={np.clip(np.round(WA),-6,6).astype(int).tolist()}")

print("\n" + "=" * 92)
print("路线 B: K=1, 3x3 卷积 + ReLU(0..15) + 4 阈值二值化 + 平均池化(100 特征)")
print("=" * 92)
t0 = time.time()
aB, WB, WdB, bdB, taus = train_B(Btr, ytr_s, Bte, yte_s, T=4, steps=4000)
print(f"  => 路线 B 最好 {aB:.4f}   用时 {time.time()-t0:.0f}s   核={np.clip(np.round(WB),-6,6).astype(int).tolist()}")
print(f"     阈值 = {taus.tolist()}")

print("\n" + "=" * 92)
print("成本对比（单元数 / 扇出）")
print("=" * 92)
cA, cB = cost_A(1), cost_B(4)
print(f"  {'项目':>12} | {'路线A':>12} | {'路线B':>12}")
print("  " + "-" * 44)
for k in cA:
    print(f"  {k:>12} | {cA[k]:12d} | {cB[k]:12d}")
print(f"  {'合计':>12} | {sum(cA.values()):12d} | {sum(cB.values()):12d}")
print(f"\n  准确率: 路线A {aA:.4f}   路线B {aB:.4f}")

json.dump({"A": {"acc": aA, "kernel": np.clip(np.round(WA), -6, 6).tolist(), "cost": cA},
           "B": {"acc": aB, "kernel": np.clip(np.round(WB), -6, 6).tolist(),
                 "taus": taus.tolist(), "cost": cB}},
          open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "mnist_k1.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n已写出: results/mnist_k1.json")

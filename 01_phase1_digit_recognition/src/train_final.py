#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
方案 D：归一化 + 增广 组合
=============================
在 28x28 上做 平移/缩放/阈值 增广 -> 下采样 -> 二值化
-> 再做 质心对中 + 尺度归一 -> 训练

对照 A(原始) / C(归一化无增广) / D(归一化+增广)
"""
import os, json, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
SEED = 1234
YY, XX = np.mgrid[0:15, 0:15].astype(np.float32)
DISK1 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DISK28 = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)
          if dy*dy + dx*dx <= 4]


def load_idx(p):
    with open(p, 'rb') as f:
        mg = struct.unpack('>I', f.read(4))[0]; nd = mg & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(nd)]
        d = np.frombuffer(f.read(), dtype=np.uint8)
    return d.reshape(dims)


def shift28(img, dy, dx):
    out = np.zeros_like(img)
    y0, y1 = max(0, dy), min(28, 28 + dy); x0, x1 = max(0, dx), min(28, 28 + dx)
    out[:, y0:y1, x0:x1] = img[:, y0-dy:y1-dy, x0-dx:x1-dx]
    return out


def zoom28(img, s):
    yy, xx = np.mgrid[0:28, 0:28].astype(np.float32)
    sy = np.clip(np.round((yy-13.5)/s+13.5).astype(int), 0, 27)
    sx = np.clip(np.round((xx-13.5)/s+13.5).astype(int), 0, 27)
    return img[:, sy.ravel(), sx.ravel()].reshape(img.shape[0], 28, 28)


def down15(img28):
    n = img28.shape[0]
    a = img28.reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    o = np.zeros((n, 15, 15), dtype=np.float32)
    o[:, :14, :14] = a; o[:, 14, :14] = a[:, 13, :]
    o[:, :14, 14] = a[:, :, 13]; o[:, 14, 14] = a[:, 13, 13]
    return o


def dil15(b):
    o = b.copy()
    for dy, dx in DISK1:
        o |= np.roll(np.roll(b, dy, 1), dx, 2)
    return o


def ero15(b):
    o = b.copy()
    for dy, dx in DISK1:
        o &= np.roll(np.roll(b, dy, 1), dx, 2)
    return o


def stats(B):
    n = B.shape[0]
    flat = B.reshape(n, 225).astype(np.float32)
    tot = np.maximum(flat.sum(1), 1)
    cx = (flat * XX.ravel()[None, :]).sum(1) / tot
    cy = (flat * YY.ravel()[None, :]).sum(1) / tot
    dx = XX.ravel()[None, :] - cx[:, None]
    dy = YY.ravel()[None, :] - cy[:, None]
    sd = np.sqrt(np.maximum((flat*(dx*dx+dy*dy)).sum(1)/tot, 1e-6))
    return cx, cy, sd


def resample15(B, cx, cy, s):
    n = B.shape[0]
    oy = (YY[None]-7.0)/s[:, None, None] + cy[:, None, None]
    ox = (XX[None]-7.0)/s[:, None, None] + cx[:, None, None]
    iy = np.clip(np.round(oy).astype(int), 0, 14)
    ix = np.clip(np.round(ox).astype(int), 0, 14)
    idx = (iy.reshape(n, -1)*15 + ix.reshape(n, -1))
    return np.take_along_axis(B.reshape(n, 225), idx, axis=1).reshape(n, 15, 15)


def norm_full(B, S0=4.0):
    cx, cy, sd = stats(B)
    s = np.clip(np.where(sd > 0.5, S0/sd, 1.0), 0.5, 2.0)
    return resample15(B, cx, cy, s)


def augment28(imgs28, rng):
    z = imgs28.astype(np.float32)
    n = len(z)
    z = shift28(z, int(rng.integers(-2, 3)), int(rng.integers(-2, 3)))
    z = zoom28(z, float(rng.uniform(0.80, 1.20)))
    d = down15(z)
    lo = d.reshape(n, -1).min(1); hi = d.reshape(n, -1).max(1)
    thr = lo + (hi - lo) * rng.uniform(0.20, 0.80)
    b = d > thr[:, None, None]
    m = rng.integers(0, 3)
    if m == 1: b = ero15(b)
    elif m == 2: b = dil15(b)
    return b


def train(X, y, ep=25, lr=0.05, bs=512, seed=SEED):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((225, 10))*0.01
    b = np.zeros(10)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    N = len(X); step = 0
    for e in range(ep):
        perm = rng.permutation(N); l = lr*(0.85**(e//5))
        for i in range(0, N, bs):
            idx = perm[i:i+bs]; xb = X[idx]; yb = y[idx]
            lo = xb @ W + b; lo -= lo.max(1, keepdims=True)
            ex = np.exp(lo); P = ex/ex.sum(1, keepdims=True)
            Bn = len(idx)
            dl = P.copy(); dl[np.arange(Bn), yb] -= 1; dl /= Bn
            gW = xb.T @ dl; gb = dl.sum(0); step += 1
            for Pp, Gg, mm, vv in ((W, gW, mW, vW), (b, gb, mb, vb)):
                mm *= b1; mm += (1-b1)*Gg
                vv *= b2; vv += (1-b2)*(Gg*Gg)
                Pp -= l*(mm/(1-b1**step))/(np.sqrt(vv/(1-b2**step))+eps)
    return W, b


def main():
    tr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    te = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    thr = down15(tr.astype(np.float32)).mean()
    Btr0 = (down15(tr.astype(np.float32)) > thr)
    Bte0 = (down15(te.astype(np.float32)) > thr)

    print("=" * 92)
    print("方案 D：归一化 + 增广")
    print("=" * 92)

    # 构造 D 的训练集：增广 + 归一化
    rng = np.random.default_rng(SEED)
    REPS = 5
    print(f"\n生成增广+归一化训练集（{REPS} 份/张）...")
    Xs = []
    for r in range(REPS):
        Ba = augment28(tr, rng)
        Xs.append(norm_full(Ba).reshape(len(Ba), 225).astype(np.float64))
        print(f"  rep {r+1}: 平均点亮 {Xs[-1].sum(1).mean():.1f}")
    XtrD = np.concatenate(Xs, 0)
    YtrD = np.tile(ytr, REPS)
    # 也加原样+归一化
    XtrD = np.concatenate([XtrD, norm_full(Btr0).reshape(len(Btr0), 225).astype(np.float64)], 0)
    YtrD = np.concatenate([YtrD, ytr], 0)
    print(f"  D 训练集 {XtrD.shape}")

    print(f"\n训练 D ...")
    WD, bD = train(XtrD, YtrD, ep=18)
    XteD = norm_full(Bte0).reshape(len(Bte0), 225).astype(np.float64)
    aD = float(((XteD @ WD + bD).argmax(1) == yte).mean())
    print(f"  D 原始测试准确率 {aD:.4f}")

    # 载入 A 与 C
    zC = np.load(os.path.join(RES, 'Wnorm_model.npz'))
    WC, bC = zC['W'], zC['b']
    XtrA = Btr0.reshape(len(Btr0), 225).astype(np.float64)
    XteA = Bte0.reshape(len(Bte0), 225).astype(np.float64)
    print(f"\n重训 A（同种子，便于对照）...")
    WA, bA = train(XtrA, ytr, ep=25)
    aA = float(((XteA @ WA + bA).argmax(1) == yte).mean())
    aC = float(((XteD @ WC + bC).argmax(1) == yte).mean())
    print(f"  A {aA:.4f}   C {aC:.4f}   D {aD:.4f}")

    # 鲁棒性
    DISK1l = DISK1
    def sh(B, dy, dx):
        o = np.zeros_like(B)
        y0,y1 = max(0,dy), min(15,15+dy); x0,x1 = max(0,dx), min(15,15+dx)
        o[:, y0:y1, x0:x1] = B[:, y0-dy:y1-dy, x0-dx:x1-dx]
        return o
    def zm(B, s):
        sy = np.clip(np.round((YY-7)/s+7).astype(int),0,14)
        sx = np.clip(np.round((XX-7)/s+7).astype(int),0,14)
        return B[:, sy.ravel(), sx.ravel()].reshape(len(B),15,15)

    sub = slice(0, 3000)
    tests = [('原样', lambda B: B),
             ('下移1', lambda B: sh(B,1,0)), ('上移1', lambda B: sh(B,-1,0)),
             ('右移1', lambda B: sh(B,0,1)), ('左移1', lambda B: sh(B,0,-1)),
             ('下移2', lambda B: sh(B,2,0)), ('右下1', lambda B: sh(B,1,1)),
             ('放大1.15', lambda B: zm(B,1.15)), ('缩小0.85', lambda B: zm(B,0.85)),
             ('膨胀1', lambda B: dil15(B)), ('膨胀2', lambda B: dil15(dil15(B))),
             ('腐蚀1', lambda B: ero15(B))]

    print(f"\n{'='*92}\n鲁棒性（3000 张）\n{'='*92}")
    print(f"  {'扰动':<12} | {'A 原始':>9} | {'C 归一':>9} | {'D 归一+增广':>11}")
    print("  " + "-" * 54)
    out = {}
    for tn, tf in tests:
        Bp = tf(Bte0[sub])
        # A: 直接
        a_ = float(((Bp.reshape(len(Bp),225).astype(np.float64) @ WA + bA
                     ).argmax(1) == yte[sub]).mean())
        # C, D: 归一化后
        Bn = norm_full(Bp).reshape(len(Bp),225).astype(np.float64)
        c_ = float(((Bn @ WC + bC).argmax(1) == yte[sub]).mean())
        d_ = float(((Bn @ WD + bD).argmax(1) == yte[sub]).mean())
        out[tn] = [a_, c_, d_]
        print(f"  {tn:<12} | {a_:>9.4f} | {c_:>9.4f} | {d_:>11.4f}")

    # 整数化 D
    print(f"\n{'='*92}\nD 的整数化\n{'='*92}")
    best = None
    for qm in [16, 24, 32, 48, 64, 96, 128]:
        Wq = np.round(WD*qm)
        a = float(((XteD @ (Wq/qm) + bD).argmax(1) == yte).mean())
        print(f"  qmax={qm:>4}: {a:.4f}  W范围[{int(Wq.min())},{int(Wq.max())}]")
        if best is None or a >= best[1]-1e-9:
            best = (qm, a, Wq)
    qm, a, Wq = best
    Wi = Wq.astype(int); bi = np.round(bD*qm).astype(int)
    print(f"  选用 qmax={qm}, 准确率 {a:.4f}, W范围[{Wi.min()},{Wi.max()}], "
          f"bias {bi.tolist()}")
    np.savez(os.path.join(RES, 'Wfinal_model.npz'), W=WD, b=bD, S0=4.0)
    np.savez(os.path.join(RES, 'Wfinal_int.npz'), W=Wi, b=bi, S0=4.0)
    json.dump({'acc': {'A': aA, 'C': aC, 'D': aD},
               'robust': out, 'qmax': int(qm),
               'W_int_range': [int(Wi.min()), int(Wi.max())],
               'bias_int': bi.tolist()},
              open(os.path.join(RES, 'final_report.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已存: results/Wfinal_model.npz / Wfinal_int.npz / final_report.json")


if __name__ == '__main__':
    main()

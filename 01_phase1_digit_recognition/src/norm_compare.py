#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
对照：只做质心对中（不做缩放）—— 红石实现简单得多
======================================================
A. 原始像素                （基线 0.8738）
B. 质心对中（不缩放）      （只需 求和/除法/平移）
C. 质心对中 + 尺度归一      （需要 除法 + 缩放）

同时验证: 归一化是幂等的（对已归一化图再归一化不应改变）
"""
import os, json, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
SEED = 11


def load_idx(p):
    with open(p, 'rb') as f:
        mg = struct.unpack('>I', f.read(4))[0]; nd = mg & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(nd)]
        d = np.frombuffer(f.read(), dtype=np.uint8)
    return d.reshape(dims)


def down15(img28):
    n = img28.shape[0]
    a = img28.reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    o = np.zeros((n, 15, 15), dtype=np.float32)
    o[:, :14, :14] = a; o[:, 14, :14] = a[:, 13, :]
    o[:, :14, 14] = a[:, :, 13]; o[:, 14, 14] = a[:, 13, 13]
    return o


YY, XX = np.mgrid[0:15, 0:15].astype(np.float32)


def stats(B):
    n = B.shape[0]
    flat = B.reshape(n, 225).astype(np.float32)
    tot = np.maximum(flat.sum(1), 1)
    cx = (flat * XX.ravel()[None, :]).sum(1) / tot
    cy = (flat * YY.ravel()[None, :]).sum(1) / tot
    dx = XX.ravel()[None, :] - cx[:, None]
    dy = YY.ravel()[None, :] - cy[:, None]
    sd = np.sqrt(np.maximum((flat * (dx*dx + dy*dy)).sum(1) / tot, 1e-6))
    return cx, cy, sd


def resample(B, cx, cy, s):
    n = B.shape[0]
    oy = (YY[None] - 7.0) / s[:, None, None] + cy[:, None, None]
    ox = (XX[None] - 7.0) / s[:, None, None] + cx[:, None, None]
    iy = np.clip(np.round(oy).astype(int), 0, 14)
    ix = np.clip(np.round(ox).astype(int), 0, 14)
    idx = (iy.reshape(n, -1) * 15 + ix.reshape(n, -1))
    return np.take_along_axis(B.reshape(n, 225), idx, axis=1).reshape(n, 15, 15)


def norm_shift_only(B):
    cx, cy, sd = stats(B)
    return resample(B, cx, cy, np.ones_like(sd))


def norm_full(B, S0=4.0):
    cx, cy, sd = stats(B)
    s = np.clip(np.where(sd > 0.5, S0 / sd, 1.0), 0.5, 2.0)
    return resample(B, cx, cy, s)


def train(X, y, ep=25, lr=0.05, bs=512, seed=SEED):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((225, 10)) * 0.01
    b = np.zeros(10)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    N = len(X); step = 0
    for e in range(ep):
        perm = rng.permutation(N); l = lr * (0.85 ** (e // 5))
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            xb = X[idx]; yb = y[idx]
            lo = xb @ W + b; lo -= lo.max(1, keepdims=True)
            ex = np.exp(lo); P = ex / ex.sum(1, keepdims=True)
            Bn = len(idx)
            dl = P.copy(); dl[np.arange(Bn), yb] -= 1; dl /= Bn
            gW = xb.T @ dl; gb = dl.sum(0); step += 1
            for Pp, Gg, mm, vv in ((W, gW, mW, vW), (b, gb, mb, vb)):
                mm *= b1; mm += (1-b1)*Gg
                vv *= b2; vv += (1-b2)*(Gg*Gg)
                Pp -= l * (mm/(1-b1**step)) / (np.sqrt(vv/(1-b2**step)) + eps)
    return W, b


def main():
    tr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    te = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    thr = down15(tr.astype(np.float32)).mean()
    Btr = (down15(tr.astype(np.float32)) > thr)
    Bte = (down15(te.astype(np.float32)) > thr)

    print("=" * 90)
    print("三种输入表示对照")
    print("=" * 90)

    # 幂等性检查
    N1 = norm_full(Bte[:1000])
    N2 = norm_full(N1)
    print(f"\n[幂等性] 归一化后再归一化，改动像素数 "
          f"{int((N1 != N2).sum())} / {N1.size} "
          f"({(N1!=N2).mean()*100:.2f}%)")

    schemes = {}
    for nm, f in [('A 原始像素', None),
                  ('B 质心对中', norm_shift_only),
                  ('C 对中+尺度归一', norm_full)]:
        Xtr = (Btr.reshape(len(Btr), 225).astype(np.float64) if f is None
               else f(Btr).reshape(len(Btr), 225).astype(np.float64))
        Xte = (Bte.reshape(len(Bte), 225).astype(np.float64) if f is None
               else f(Bte).reshape(len(Bte), 225).astype(np.float64))
        print(f"\n训练 {nm} ...")
        W, b = train(Xtr, ytr)
        a = float(((Xte @ W + b).argmax(1) == yte).mean())
        print(f"  测试准确率 {a:.4f}")
        schemes[nm] = {'W': W, 'b': b, 'acc': a, 'f': f}

    # 鲁棒性对照
    DISK1 = [(0,1),(0,-1),(1,0),(-1,0)]
    def sh(B, dy, dx):
        o = np.zeros_like(B)
        y0,y1 = max(0,dy), min(15,15+dy); x0,x1 = max(0,dx), min(15,15+dx)
        o[:, y0:y1, x0:x1] = B[:, y0-dy:y1-dy, x0-dx:x1-dx]
        return o
    def dil(B):
        o = B.copy()
        for dy,dx in DISK1: o |= np.roll(np.roll(B,dy,1),dx,2)
        return o
    def ero(B):
        o = B.copy()
        for dy,dx in DISK1: o &= np.roll(np.roll(B,dy,1),dx,2)
        return o
    def zm(B, s):
        sy = np.clip(np.round((YY-7)/s+7).astype(int),0,14)
        sx = np.clip(np.round((XX-7)/s+7).astype(int),0,14)
        return B[:, sy.ravel(), sx.ravel()].reshape(len(B),15,15)

    sub = slice(0, 3000)
    tests = [('原样', lambda B: B),
             ('下移1', lambda B: sh(B,1,0)),
             ('上移1', lambda B: sh(B,-1,0)),
             ('右移1', lambda B: sh(B,0,1)),
             ('左移1', lambda B: sh(B,0,-1)),
             ('下移2', lambda B: sh(B,2,0)),
             ('右下1', lambda B: sh(B,1,1)),
             ('放大1.15', lambda B: zm(B,1.15)),
             ('缩小0.85', lambda B: zm(B,0.85)),
             ('膨胀1', lambda B: dil(B)),
             ('膨胀2', lambda B: dil(dil(B))),
             ('腐蚀1', lambda B: ero(B))]

    print(f"\n{'='*90}\n鲁棒性对照（3000 张测试样本）\n{'='*90}")
    print(f"  {'扰动':<12} | " + " | ".join(f"{nm:>14}" for nm in schemes))
    print("  " + "-" * 74)
    out = {}
    for tn, tf in tests:
        Bp = tf(Bte[sub])
        row = []
        for nm, S in schemes.items():
            Xa = (Bp.reshape(len(Bp), 225).astype(np.float64) if S['f'] is None
                  else S['f'](Bp).reshape(len(Bp), 225).astype(np.float64))
            a = float(((Xa @ S['W'] + S['b']).argmax(1) == yte[sub]).mean())
            row.append(a)
        out[tn] = row
        print(f"  {tn:<12} | " + " | ".join(f"{v:>14.4f}" for v in row))

    # 保存最好的（C）
    S = schemes['C 对中+尺度归一']
    for qm in [32, 64, 128]:
        Wq = np.round(S['W'] * qm).astype(int)
        bq = np.round(S['b'] * qm).astype(int)
        Xte = norm_full(Bte).reshape(len(Bte), 225).astype(np.float64)
        a = float(((Xte @ (Wq/qm) + S['b']).argmax(1) == yte).mean())
        print(f"\n  C 方案 qmax={qm}: 准确率 {a:.4f}  "
              f"W范围[{Wq.min()},{Wq.max()}]")
        if qm == 32:
            np.savez(os.path.join(RES, 'Wnorm32_int.npz'), W=Wq, b=bq, S0=4.0)

    json.dump({'schemes': {k: v['acc'] for k, v in schemes.items()},
               'robust': out},
              open(os.path.join(RES, 'norm_compare.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已存: results/norm_compare.json")


if __name__ == '__main__':
    main()

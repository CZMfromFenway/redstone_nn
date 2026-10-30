#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
方案：输入归一化（质心对中 + 尺度归一）+ 重训
================================================
流程（每张 15x15 二值图）:
  1. 算质心 (cx, cy) = Σp·x / Σp , Σp·y / Σp
  2. 算展布 s = sqrt( Σp·((x-cx)^2+(y-cy)^2) / Σp )
  3. 把图按 (cx,cy) 对中、按 s/s0 缩放，重采样回 15x15，再二值化
     -> 得到"规范化图"

这消除了平移与缩放差异。红石里同样可实现（求和/比较器减法/LUT 除法）。
"""
import os, json, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
SEED = 7
np.random.seed(SEED)


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


def affine15(B, cx, cy, s):
    """把二值图 (n,15,15) 按 每张的 中心(cx,cy) 和 尺度 s 重采样到 15x15 标准网格"""
    n = B.shape[0]
    yy, xx = np.mgrid[0:15, 0:15].astype(np.float32)
    oy = (yy[None, :, :] - 7.0) / s[:, None, None] + cy[:, None, None]
    ox = (xx[None, :, :] - 7.0) / s[:, None, None] + cx[:, None, None]
    iy = np.clip(np.round(oy).astype(int), 0, 14)
    ix = np.clip(np.round(ox).astype(int), 0, 14)
    idx = (iy.reshape(n, -1) * 15 + ix.reshape(n, -1))     # (n,225)
    flat = B.reshape(n, 225)
    return np.take_along_axis(flat, idx, axis=1).reshape(n, 15, 15)


def normalize(B):
    """B: (n,15,15) bool -> 规范化 (n,15,15) bool"""
    n = B.shape[0]
    flat = B.reshape(n, 225).astype(np.float32)
    tot = flat.sum(1)
    tot_s = np.where(tot > 0, tot, 1)
    yy, xx = np.mgrid[0:15, 0:15].astype(np.float32)
    cx = (flat * xx.ravel()[None, :]).sum(1) / tot_s
    cy = (flat * yy.ravel()[None, :]).sum(1) / tot_s
    # 展布
    dx = xx.ravel()[None, :] - cx[:, None]
    dy = yy.ravel()[None, :] - cy[:, None]
    var = (flat * (dx * dx + dy * dy)).sum(1) / tot_s
    sd = np.sqrt(np.maximum(var, 1e-6))
    # 目标展布
    S0 = 4.0
    scale = np.where(sd > 0.5, S0 / sd, 1.0)
    scale = np.clip(scale, 0.5, 2.0)
    out = affine15(B, cx, cy, scale)
    return out


def main():
    tr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    te = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    thr = down15(tr.astype(np.float32)).mean()
    Btr = (down15(tr.astype(np.float32)) > thr)
    Bte = (down15(te.astype(np.float32)) > thr)

    print("=" * 92)
    print("输入归一化（质心对中 + 尺度归一）")
    print("=" * 92)
    t0 = normalize(Btr[:5000])
    print(f"  归一化前平均点亮 {Btr[:5000].reshape(5000,-1).sum(1).mean():.1f}")
    print(f"  归一化后平均点亮 {t0.reshape(5000,-1).sum(1).mean():.1f}")

    # 用规范化输入训线性模型
    print(f"\n构造规范化训练集 ...")
    Ntr = normalize(Btr)
    Nte = normalize(Bte)
    X = Ntr.reshape(len(Ntr), 225).astype(np.float64)
    Xte = Nte.reshape(len(Nte), 225).astype(np.float64)

    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((225, 10)) * 0.01
    b = np.zeros(10)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    LR, BS, EP = 0.05, 512, 25
    N = len(X); step = 0
    print(f"训练 {EP} epoch ...")
    for ep in range(EP):
        perm = rng.permutation(N); lr = LR * (0.85 ** (ep // 5))
        for i in range(0, N, BS):
            idx = perm[i:i+BS]
            xb = X[idx]; yb = ytr[idx]
            lo = xb @ W + b
            lo -= lo.max(1, keepdims=True)
            ex = np.exp(lo); P = ex / ex.sum(1, keepdims=True)
            Bn = len(idx)
            dl = P.copy(); dl[np.arange(Bn), yb] -= 1; dl /= Bn
            gW = xb.T @ dl; gb = dl.sum(0)
            step += 1
            for Pp, Gg, mm, vv in ((W, gW, mW, vW), (b, gb, mb, vb)):
                mm *= b1; mm += (1-b1)*Gg
                vv *= b2; vv += (1-b2)*(Gg*Gg)
                Pp -= lr * (mm/(1-b1**step)) / (np.sqrt(vv/(1-b2**step)) + eps)
        if ep % 5 == 0 or ep == EP-1:
            lo = X @ W + b
            print(f"  epoch {ep+1:>2}: train_acc {(lo.argmax(1)==ytr).mean():.4f}")

    def acc(Xa, ya):
        return float(((Xa @ W + b).argmax(1) == ya).mean())

    print(f"\n{'='*92}\n评估（都用规范化输入）\n{'='*92}")
    print(f"  原始测试集          : {acc(Xte, yte):.4f}")
    # 各种扰动
    DISK1 = [(0,1),(0,-1),(1,0),(-1,0)]
    def sh(B, dy, dx):
        out = np.zeros_like(B)
        ys0, ys1 = max(0,dy), min(15,15+dy); xs0, xs1 = max(0,dx), min(15,15+dx)
        out[:, ys0:ys1, xs0:xs1] = B[:, ys0-dy:ys1-dy, xs0-dx:xs1-dx]
        return out
    def dil(B):
        o = B.copy()
        for dy,dx in DISK1: o |= np.roll(np.roll(B,dy,1),dx,2)
        return o
    def ero(B):
        o = B.copy()
        for dy,dx in DISK1: o &= np.roll(np.roll(B,dy,1),dx,2)
        return o
    def zoom(B, s):
        yy, xx = np.mgrid[0:15,0:15].astype(np.float32)
        sy = np.clip(np.round((yy-7)/s+7).astype(int),0,14)
        sx = np.clip(np.round((xx-7)/s+7).astype(int),0,14)
        return B[:, sy.ravel(), sx.ravel()].reshape(len(B),15,15)

    res = {}
    print(f"\n  {'扰动':<16} | {'规范化后点亮':>10} | {'准确率':>8}")
    print("  " + "-" * 44)
    for nm, Bp in [('原样', Bte[:4000]),
                   ('下移1', sh(Bte[:4000],1,0)),
                   ('上移1', sh(Bte[:4000],-1,0)),
                   ('右移1', sh(Bte[:4000],0,1)),
                   ('左移1', sh(Bte[:4000],0,-1)),
                   ('下移2', sh(Bte[:4000],2,0)),
                   ('右下1', sh(Bte[:4000],1,1)),
                   ('放大1.15', zoom(Bte[:4000],1.15)),
                   ('缩小0.85', zoom(Bte[:4000],0.85)),
                   ('膨胀1', dil(Bte[:4000])),
                   ('膨胀2', dil(dil(Bte[:4000]))),
                   ('腐蚀1', ero(Bte[:4000]))]:
        Np_ = normalize(Bp).reshape(len(Bp), 225).astype(np.float64)
        a = acc(Np_, yte[:4000])
        res[nm] = a
        print(f"  {nm:<16} | {Np_.sum(1).mean():>10.1f} | {a:>8.4f}")

    # 整数化
    print(f"\n{'='*92}\n整数化\n{'='*92}")
    best = None
    for qm in [16, 24, 32, 48, 64, 96, 128, 192]:
        Wq = np.round(W * qm)
        z = Xte @ (Wq / qm) + b
        a = float((z.argmax(1) == yte).mean())
        print(f"  qmax={qm:>4}: 原始测试准确率 {a:.4f}  "
              f"W范围[{int(Wq.min())},{int(Wq.max())}]")
        if best is None or a >= best[1] - 1e-9:
            best = (qm, a, Wq)
    qm, a, Wq = best
    print(f"  选用 qmax={qm}, 准确率 {a:.4f}")
    Wi = Wq.astype(int)
    bi = np.round(b * qm).astype(int)
    print(f"  W 整数范围 [{Wi.min()}, {Wi.max()}]")
    print(f"  bias 整数 {bi.tolist()}")

    np.savez(os.path.join(RES, 'Wnorm_model.npz'), W=W, b=b, S0=4.0)
    np.savez(os.path.join(RES, 'Wnorm_int.npz'), W=Wi, b=bi, S0=4.0)
    json.dump({'acc_original': acc(Xte, yte), 'robust': res,
               'qmax': int(qm), 'W_int_range': [int(Wi.min()), int(Wi.max())],
               'bias_int': bi.tolist()},
              open(os.path.join(RES, 'norm_report.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已存: results/Wnorm_model.npz / Wnorm_int.npz / norm_report.json")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
普通 GAN（对照组，用于和对称分解版比较）
==========================================
结构: z(d) -> Linear(d,128) -> ReLU -> Linear(128,64) -> tanh -> 8x8
超参数与 gan_sym.py 完全一致。
"""
import os, json, numpy as np, time
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")

D_Z = 32; H_G = 128; H_D = 64; C = 16
EPOCHS = int(os.environ.get('EPOCHS', '3000'))
BS = 64; LR = 1e-4
TAG = os.environ.get('TAG', 'plain')


def init(dims, seed=0, scale=None):
    rng = np.random.default_rng(seed); P = []
    for i in range(len(dims) - 1):
        s = scale if scale else np.sqrt(2.0 / dims[i])
        P.append(rng.normal(0, s, (dims[i], dims[i + 1])).astype(np.float32))
        P.append(np.zeros(dims[i + 1], dtype=np.float32))
    return P


def g_forward(P, Z):
    acts = [Z]; h = Z; L = len(P) // 2
    for i in range(L):
        h = h @ P[2 * i] + P[2 * i + 1]
        h = np.maximum(h, 0) if i < L - 1 else np.tanh(h)
        acts.append(h)
    return acts


def d_forward(P, X):
    acts = [X]; h = X; L = len(P) // 2
    for i in range(L):
        h = h @ P[2 * i] + P[2 * i + 1]
        if i < L - 1:
            h = np.where(h > 0, h, 0.2 * h)
        acts.append(h)
    return acts


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def adam(P, G, M, V, t, lr):
    for i in range(len(P)):
        M[i] = 0.9 * M[i] + 0.1 * G[i]
        V[i] = 0.999 * V[i] + 0.001 * G[i] ** 2
        P[i] -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)


def main():
    print("=" * 96)
    print(f"普通 GAN  TAG={TAG}（对照组）")
    print("=" * 96)
    F = np.load(os.path.join(DATA, 'faces8_rgb.npy')).astype(np.float64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    flat = F.reshape(-1, 3)
    idx = ((flat[:, None] - pal[None].astype(np.float64)) ** 2).sum(2).argmin(1)
    X = (idx.reshape(len(F), 64) / (C - 1) * 2 - 1).astype(np.float32)
    print(f"数据: {X.shape}  均值 {X.mean():.3f}")

    n = len(X)
    G = init([D_Z, H_G, 64], seed=1)
    D = init([64, H_D, 1], seed=2, scale=0.1)
    MG = [np.zeros_like(p) for p in G]; VG = [np.zeros_like(p) for p in G]
    MD = [np.zeros_like(p) for p in D]; VD = [np.zeros_like(p) for p in D]
    rng = np.random.default_rng(0)
    t_g = t_d = 0; t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        for _ in range(max(1, n // BS)):
            Xr = X[rng.integers(0, n, BS)]
            Z = rng.normal(0, 1, (BS, D_Z)).astype(np.float32)
            Xf = g_forward(G, Z)[-1]
            ar = d_forward(D, Xr); af = d_forward(D, Xf)
            d_ar = (sig(ar[-1]) - 0.9) / BS; d_af = (sig(af[-1]) - 0.0) / BS
            t_d += 1
            gW2 = ar[-2].T @ d_ar + af[-2].T @ d_af
            gb2 = d_ar.sum(0) + d_af.sum(0)
            dHr = (d_ar @ D[2].T) * np.where(ar[-2] > 0, 1.0, 0.2)
            dHf = (d_af @ D[2].T) * np.where(af[-2] > 0, 1.0, 0.2)
            adam(D, [Xr.T @ dHr + Xf.T @ dHf, dHr.sum(0) + dHf.sum(0), gW2, gb2],
                 MD, VD, t_d, LR)

            Z = rng.normal(0, 1, (BS, D_Z)).astype(np.float32)
            ag = g_forward(G, Z); Xf = ag[-1]
            ad = d_forward(D, Xf)
            d_ad = (sig(ad[-1]) - 0.9) / BS
            dHd = (d_ad @ D[2].T) * np.where(ad[-2] > 0, 1.0, 0.2)
            dPre = (dHd @ D[0].T) * (1 - Xf ** 2)
            t_g += 1
            gW2 = ag[-2].T @ dPre; gb2 = dPre.sum(0)
            dH = (dPre @ G[2].T) * (ag[-2] > 0)
            adam(G, [Z.T @ dH, dH.sum(0), gW2, gb2], MG, VG, t_g, LR)

        if ep % 250 == 0 or ep == 1:
            Z = rng.normal(0, 1, (256, D_Z)).astype(np.float32)
            Xf = g_forward(G, Z)[-1]
            pr = sig(d_forward(D, X[:256])[-1]).mean()
            pf = sig(d_forward(D, Xf)[-1]).mean()
            print(f"  ep {ep:5d}  D(真)={pr:.3f} D(假)={pf:.3f}  生成std={Xf.std():.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    json.dump({'W1': G[0].tolist(), 'b1': G[1].tolist(), 'W2': G[2].tolist(),
               'b2': G[3].tolist(), 'd_z': D_Z, 'h': H_G, 'palette': pal.tolist(),
               'mode': 'plain'},
              open(os.path.join(RES, f'gan_generator_{TAG}.json'), 'w'))

    def to_img(vec):
        ii = np.clip(np.round((vec + 1) / 2 * (C - 1)), 0, C - 1).astype(int)
        return pal[ii].reshape(8, 8, 3)

    SC = 12
    def grid(imgs, cols=12):
        rows = (len(imgs) + cols - 1) // cols
        cv = np.full((rows * 8, cols * 8, 3), 40, dtype=np.uint8)
        for i, im in enumerate(imgs):
            r, c = divmod(i, cols)
            cv[r*8:(r+1)*8, c*8:(c+1)*8] = im
        big = np.kron(cv, np.ones((SC, SC, 1), dtype=np.uint8))
        for r in range(rows + 1):
            big[r*8*SC:r*8*SC+1, :] = 90
        for c in range(cols + 1):
            big[:, c*8*SC:c*8*SC+1] = 90
        return big

    r2 = np.random.default_rng(123)
    Z = r2.normal(0, 1, (48, D_Z)).astype(np.float32)
    Image.fromarray(grid([to_img(v) for v in g_forward(G, Z)[-1]])).save(
        os.path.join(RES, f'gan_samples_{TAG}.png'))
    Image.fromarray(grid([to_img(v) for v in X[r2.choice(n, 48, replace=False)]])).save(
        os.path.join(RES, f'gan_real_{TAG}.png'))
    print(f"已保存: {RES}/gan_samples_{TAG}.png")


if __name__ == '__main__':
    main()

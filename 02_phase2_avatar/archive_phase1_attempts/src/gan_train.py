#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无条件 GAN 生成 8x8 玩家头像（16 色定制调色板）
==================================================
架构（与红石可移植性对齐，只用最后导出的 G 网络）:
  G: z(64) -> Linear(64,128) -> ReLU -> Linear(128,64) -> sigmoid -> 8x8 调色板索引
  D: x(64) -> Linear(64,64) -> ReLU -> Linear(64,1)

训练: 标准非饱和 GAN 损失，交替更新
输出: 生成样本网格 PNG + 真实样本对照 PNG
"""
import os, json, numpy as np, time
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

D_Z = 64
H_G = 128
H_D = 64
C = 16
EPOCHS = 400
BS = 64
LR = 2e-4


def init(dims, seed=0, scale=None):
    rng = np.random.default_rng(seed); P = []
    for i in range(len(dims) - 1):
        s = scale if scale else np.sqrt(2.0 / dims[i])
        P.append(rng.normal(0, s, (dims[i], dims[i + 1])).astype(np.float32))
        P.append(np.zeros(dims[i + 1], dtype=np.float32))
    return P


def fwd_g(P, Z, relu_last=True):
    acts = [Z]; h = Z; L = len(P) // 2
    for i in range(L):
        h = h @ P[2 * i] + P[2 * i + 1]
        if i < L - 1 or relu_last:
            if i < L - 1:
                h = np.maximum(h, 0)
        acts.append(h)
    acts[-1] = 1.0 / (1.0 + np.exp(-np.clip(acts[-1], -30, 30)))
    return acts


def fwd_d(P, X):
    acts = [X]; h = X; L = len(P) // 2
    for i in range(L):
        h = h @ P[2 * i] + P[2 * i + 1]
        if i < L - 1:
            h = np.maximum(h, 0)
        acts.append(h)
    return acts


def adam(P, G, M, V, t, lr):
    for i in range(len(P)):
        M[i] = 0.9 * M[i] + 0.1 * G[i]
        V[i] = 0.999 * V[i] + 0.001 * G[i] ** 2
        P[i] -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)


def main():
    print("=" * 96)
    print("无条件 GAN 生成 8x8 头像（16 色定制调色板）")
    print("=" * 96)
    # 用**定制调色板**重新量化头像（faces8.npy 是旧 MC16 的标注，不能用）
    F = np.load(os.path.join(DATA, 'faces8_rgb.npy')).astype(np.float64)
    pal_full = json.load(open(os.path.join(DATA, 'palette_avatar16.json')))
    pal = np.array(pal_full['palette_ordered'], dtype=np.uint8)
    flat = F.reshape(-1, 3)
    lab_idx = ((flat[:, None] - pal[None].astype(np.float64)) ** 2).sum(2).argmin(1)
    lab = (lab_idx.reshape(len(F), 64) / (C - 1)).astype(np.float32)
    qerr = float(((pal[lab_idx].reshape(F.shape) - F) ** 2).mean())
    print(f"数据: {lab.shape}  调色板: {pal.shape}")
    print(f"定制调色板量化 MSE = {qerr:.1f}  (RMS {np.sqrt(qerr/3):.1f})")

    X = lab
    n = len(X)
    G = init([D_Z, H_G, 64], seed=1)
    D = init([64, H_D, 1], seed=2, scale=0.1)
    MG = [np.zeros_like(p) for p in G]; VG = [np.zeros_like(p) for p in G]
    MD = [np.zeros_like(p) for p in D]; VD = [np.zeros_like(p) for p in D]
    rng = np.random.default_rng(0)
    t_g = t_d = 0
    hist = []
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        for _ in range(n // BS):
            # ---- D ----
            idx = rng.integers(0, n, BS)
            Xr = X[idx]
            Z = rng.normal(0, 1, (BS, D_Z)).astype(np.float32)
            Xf = fwd_g(G, Z)[-1]
            ar = fwd_d(D, Xr); af = fwd_d(D, Xf)
            pr = 1 / (1 + np.exp(-np.clip(ar[-1], -30, 30)))
            pf = 1 / (1 + np.exp(-np.clip(af[-1], -30, 30)))
            # logits 直接算梯度（BCE with logits）
            d_ar = (pr - 1.0) / BS
            d_af = (pf - 0.0) / BS
            t_d += 1
            gW2 = ar[-2].T @ d_ar + af[-2].T @ d_af
            gb2 = d_ar.sum(0) + d_af.sum(0)
            dHr = (d_ar @ D[2].T) * (ar[-2] > 0)
            dHf = (d_af @ D[2].T) * (af[-2] > 0)
            gW1 = Xr.T @ dHr + Xf.T @ dHf
            gb1 = dHr.sum(0) + dHf.sum(0)
            adam(D, [gW1, gb1, gW2, gb2], MD, VD, t_d, LR)

            # ---- G ----
            Z = rng.normal(0, 1, (BS, D_Z)).astype(np.float32)
            ag = fwd_g(G, Z); Xf = ag[-1]
            ad = fwd_d(D, Xf)
            pd = 1 / (1 + np.exp(-np.clip(ad[-1], -30, 30)))
            d_ad = (pd - 1.0) / BS                      # 非饱和: 希望 D 判为真
            dHd = (d_ad @ D[2].T) * (ad[-2] > 0)
            dXf = dHd @ D[0].T
            # sigmoid 反传
            dPre = dXf * Xf * (1 - Xf)
            t_g += 1
            gW2 = ag[-2].T @ dPre; gb2 = dPre.sum(0)
            dH = (dPre @ G[2].T) * (ag[-2] > 0)
            gW1 = Z.T @ dH; gb1 = dH.sum(0)
            adam(G, [gW1, gb1, gW2, gb2], MG, VG, t_g, LR)

        if ep % 25 == 0 or ep == 1:
            Z = rng.normal(0, 1, (256, D_Z)).astype(np.float32)
            Xf = fwd_g(G, Z)[-1]
            # 模式多样性: 生成样本的像素值标准差
            div = float(Xf.std(0).mean())
            # D 的判别
            pr = 1 / (1 + np.exp(-np.clip(fwd_d(D, X[:256])[-1], -30, 30))).mean()
            pf = 1 / (1 + np.exp(-np.clip(fwd_d(D, Xf)[-1], -30, 30))).mean()
            hist.append((ep, float(pr), float(pf), div))
            print(f"  ep {ep:4d}  D(真)={pr:.3f}  D(假)={pf:.3f}  生成多样性={div:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 存权重 + 出图 ----
    W1, b1, W2, b2 = G
    json.dump({'W1': W1.tolist(), 'b1': b1.tolist(), 'W2': W2.tolist(), 'b2': b2.tolist(),
               'd_z': D_Z, 'h': H_G, 'palette': pal.tolist()},
              open(os.path.join(RES, 'gan_generator.json'), 'w'))

    def to_img(vec):
        idx = np.clip(np.round(vec * (C - 1)), 0, C - 1).astype(int)
        return pal[idx].reshape(8, 8, 3)

    SCALE = 12
    def grid(imgs, cols=12):
        rows = (len(imgs) + cols - 1) // cols
        canvas = np.full((rows * 8, cols * 8, 3), 40, dtype=np.uint8)
        for i, im in enumerate(imgs):
            r, c = divmod(i, cols)
            canvas[r*8:(r+1)*8, c*8:(c+1)*8] = im
        big = np.kron(canvas, np.ones((SCALE, SCALE, 1), dtype=np.uint8))
        # 画网格线
        for r in range(rows + 1):
            big[r*8*SCALE:r*8*SCALE+1, :] = 90
        for c in range(cols + 1):
            big[:, c*8*SCALE:c*8*SCALE+1] = 90
        return big

    rng2 = np.random.default_rng(123)
    Z = rng2.normal(0, 1, (48, D_Z)).astype(np.float32)
    gen_imgs = [to_img(v) for v in fwd_g(G, Z)[-1]]
    Image.fromarray(grid(gen_imgs, 12)).save(os.path.join(RES, 'gan_samples.png'))

    real_imgs = [to_img(v) for v in X[rng2.choice(len(X), 48, replace=False)]]
    Image.fromarray(grid(real_imgs, 12)).save(os.path.join(RES, 'gan_real.png'))
    print(f"\n已保存:")
    print(f"  生成样本: {RES}/gan_samples.png")
    print(f"  真实对照: {RES}/gan_real.png")
    print(f"  G 权重  : {RES}/gan_generator.json")


if __name__ == '__main__':
    main()

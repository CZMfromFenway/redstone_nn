#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B 方案：条件生成（可移植到红石的前馈结构）
=============================================
结构:
  Encoder  E: x(64)          -> Linear(64,32)  -> ReLU
  Decoder  G: [E(x) | z(32)] -> Linear(64,128) -> ReLU -> Linear(128,64) -> tanh
  Discriminator D: [x_in(64) | x_out(64)] -> Linear(128,64) -> LeakyReLU
                                          -> Linear(64,1)

判别器判断的是**配对是否真实**:
  真: (x, x)      —— 输入与输出相同
  假: (x, G(x,z)) —— 输出是重建
这比无条件 GAN 的"这像不像头像"容易得多 -> 训练稳定。

总损失: L = L_adv + lambda_rec * L1(x, x_hat)
  L1 项保证结构不跑偏，L_adv 项提供"生成新样本"的能力。

采样方式:
  (a) 重建: z=0, x_hat = G(E(x), 0)
  (b) 生成: z ~ N(0,I), x_hat = G(E(x), z)   <- 同一张脸的不同变体
  (c) 无输入生成: z ~ N(0,I), x_hat = G(0, z)
"""
import os, json, numpy as np, time
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")

D_Z = 32; H_E = 32; H_G = 128; H_D = 64; C = 16
EPOCHS = int(os.environ.get('EPOCHS', '4000'))
BS = 64; LR = 1e-4
LAM_REC = float(os.environ.get('LAM_REC', '20.0'))
TAG = os.environ.get('TAG', 'condB')


def init(dims, seed=0, scale=None):
    rng = np.random.default_rng(seed); P = []
    for i in range(len(dims) - 1):
        s = scale if scale else np.sqrt(2.0 / dims[i])
        P.append(rng.normal(0, s, (dims[i], dims[i + 1])).astype(np.float32))
        P.append(np.zeros(dims[i + 1], dtype=np.float32))
    return P


def enc(P, X):
    return np.maximum(X @ P[0] + P[1], 0)


def dec(P, EZ):
    """P: [W1(dz+de, h), b1, W2(h,64), b2]"""
    h = np.maximum(EZ @ P[0] + P[1], 0)
    return np.tanh(h @ P[2] + P[3]), h


def dfwd(P, X):
    h = np.where((a := X @ P[0] + P[1]) > 0, a, 0.2 * a)
    return h @ P[2] + P[3], h, a


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def adam(P, G, M, V, t, lr):
    for i in range(len(P)):
        M[i] = 0.9 * M[i] + 0.1 * G[i]
        V[i] = 0.999 * V[i] + 0.001 * G[i] ** 2
        P[i] -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)


def main():
    print("=" * 96)
    print(f"B 方案：条件生成  TAG={TAG}")
    print("=" * 96)
    F = np.load(os.path.join(DATA, 'faces8_rgb.npy')).astype(np.float64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    flat = F.reshape(-1, 3)
    idx = ((flat[:, None] - pal[None].astype(np.float64)) ** 2).sum(2).argmin(1)
    X = (idx.reshape(len(F), 64) / (C - 1) * 2 - 1).astype(np.float32)
    n = len(X)
    print(f"数据: {X.shape}  均值 {X.mean():.3f}")

    E = init([64, H_E], seed=1)
    G = init([H_E + D_Z, H_G, 64], seed=2)
    D = init([128, H_D, 1], seed=3, scale=0.1)
    ME = [np.zeros_like(p) for p in E]; VE = [np.zeros_like(p) for p in E]
    MG = [np.zeros_like(p) for p in G]; VG = [np.zeros_like(p) for p in G]
    MD = [np.zeros_like(p) for p in D]; VD = [np.zeros_like(p) for p in D]
    rng = np.random.default_rng(0)
    t_e = t_g = t_d = 0
    t0 = time.time()

    def make_ez(x, z):
        return np.concatenate([enc(E, x), z], axis=1)

    for ep in range(1, EPOCHS + 1):
        for _ in range(max(1, n // BS)):
            xb = X[rng.integers(0, n, BS)]
            z = rng.normal(0, 1, (BS, D_Z)).astype(np.float32)
            ez = make_ez(xb, z)
            xh, hg = dec(G, ez)
            # ---------- D ----------
            pair_r = np.concatenate([xb, xb], 1)
            pair_f = np.concatenate([xb, xh], 1)
            lr_, hr, ar = dfwd(D, pair_r)
            lf_, hf, af = dfwd(D, pair_f)
            d_ar = (sig(lr_) - 0.9) / BS
            d_af = (sig(lf_) - 0.0) / BS
            t_d += 1
            gW2 = hr.T @ d_ar + hf.T @ d_af; gb2 = d_ar.sum(0) + d_af.sum(0)
            dhr = (d_ar @ D[2].T) * np.where(ar > 0, 1.0, 0.2)
            dhf = (d_af @ D[2].T) * np.where(af > 0, 1.0, 0.2)
            gW1 = pair_r.T @ dhr + pair_f.T @ dhf; gb1 = dhr.sum(0) + dhf.sum(0)
            adam(D, [gW1, gb1, gW2, gb2], MD, VD, t_d, LR)

            # ---------- G + E ----------
            z = rng.normal(0, 1, (BS, D_Z)).astype(np.float32)
            ez = make_ez(xb, z)
            xh, hg = dec(G, ez)
            pair_f = np.concatenate([xb, xh], 1)
            lf_, hf, af = dfwd(D, pair_f)
            d_af = (sig(lf_) - 0.9) / BS
            dhf = (d_af @ D[2].T) * np.where(af > 0, 1.0, 0.2)
            dPair = dhf @ D[0].T                      # (BS,128)
            d_xh_adv = dPair[:, 64:]
            # L1 重构项
            d_xh_rec = np.sign(xh - xb) * (2.0 * LAM_REC / (BS * 64))
            d_xh = d_xh_adv + d_xh_rec
            dPre = d_xh * (1 - xh ** 2)
            t_g += 1
            gW2 = hg.T @ dPre; gb2 = dPre.sum(0)
            dH = (dPre @ G[2].T) * (hg > 0)
            gW1 = ez.T @ dH; gb1 = dH.sum(0)
            adam(G, [gW1, gb1, gW2, gb2], MG, VG, t_g, LR)
            # 梯度回传到 encoder 部分（ez 的前 H_E 维来自 E(xb)）
            d_ez = dH @ G[0].T
            d_E = d_ez[:, :H_E] * (enc(E, xb) > 0)
            t_e += 1
            adam(E, [xb.T @ d_E, d_E.sum(0)], ME, VE, t_e, LR)

        if ep % 250 == 0 or ep == 1:
            zz = np.zeros((256, D_Z), dtype=np.float32)
            xh0, _ = dec(G, make_ez(X[:256], zz))
            zr = rng.normal(0, 1, (256, D_Z)).astype(np.float32)
            xhr, _ = dec(G, make_ez(X[:256], zr))
            pr = sig(dfwd(D, np.concatenate([X[:256], X[:256]], 1))[0]).mean()
            pf = sig(dfwd(D, np.concatenate([X[:256], xhr], 1))[0]).mean()
            rec = float(np.abs(xh0 - X[:256]).mean())
            print(f"  ep {ep:5d}  D(真)={pr:.3f} D(假)={pf:.3f}  重建L1={rec:.4f}  "
                  f"生成std={xhr.std():.3f}  ({time.time()-t0:.0f}s)", flush=True)

    json.dump({'E': [p.tolist() for p in E], 'G': [p.tolist() for p in G],
               'd_z': D_Z, 'h_e': H_E, 'h_g': H_G, 'palette': pal.tolist(),
               'mode': 'conditional_B'},
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
    sel = r2.choice(n, 48, replace=False)
    # (a) 纯重建 z=0 —— 用于诊断自编码器本身
    z0r = np.zeros((48, D_Z), dtype=np.float32)
    xrec, _ = dec(G, make_ez(X[sel], z0r))
    Image.fromarray(grid([to_img(v) for v in xrec])).save(
        os.path.join(RES, f'gan_recon_{TAG}.png'))
    print(f"  z=0 重建 L1 = {np.abs(xrec - X[sel]).mean():.4f}")
    # (b) 同输入 + 不同 z —— 真正的"生成"
    zr = r2.normal(0, 1, (48, D_Z)).astype(np.float32)
    xg, _ = dec(G, make_ez(X[sel], zr))
    Image.fromarray(grid([to_img(v) for v in xg])).save(
        os.path.join(RES, f'gan_samples_{TAG}.png'))
    print(f"  z~N 生成 与输入 L1 = {np.abs(xg - X[sel]).mean():.4f}")
    Image.fromarray(grid([to_img(v) for v in X[sel]])).save(
        os.path.join(RES, f'gan_input_{TAG}.png'))
    # (c) 无输入生成（纯 z）
    z0 = r2.normal(0, 1, (48, D_Z)).astype(np.float32)
    ez0 = np.concatenate([np.zeros((48, H_E), dtype=np.float32), z0], 1)
    x0, _ = dec(G, ez0)
    Image.fromarray(grid([to_img(v) for v in x0])).save(
        os.path.join(RES, f'gan_free_{TAG}.png'))
    print(f"\n已保存: recon / samples / input / free  ({TAG})")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B 方案 v2：one-hot 颜色表示的条件生成
========================================
v1 的问题: 把调色板索引当**标量**喂进网络。
  索引是有序序号，但网络做加权求和 -> "红(5)+橙(6)"这种平均在颜色上无意义。
  仿真: z=0 的重建 L1 仍有 0.10。

v2 修正: 颜色用 **one-hot (16 通道)**，即输入是 8x8x16 = 1024 维。
  Encoder: 1024 -> 64 -> 32
  Decoder: [32|32] -> 128 -> 1024  ->  取 argmax 得颜色
  这也更接近红石实现（每个颜色一个通道 = 一条线）

比 v1 大，但先求"能不能生成好头像"，再谈压缩。
"""
import os, json, numpy as np, time
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")

C = 16; D_Z = 32; H_E = 64; H_G = 128
EPOCHS = int(os.environ.get('EPOCHS', '3000'))
BS = 64; LR = 2e-4; LAM_REC = 1.0
TAG = os.environ.get('TAG', 'condB_oh')


def init(dims, seed=0, scale=None):
    rng = np.random.default_rng(seed); P = []
    for i in range(len(dims) - 1):
        s = scale if scale else np.sqrt(2.0 / dims[i])
        P.append(rng.normal(0, s, (dims[i], dims[i + 1])).astype(np.float32))
        P.append(np.zeros(dims[i + 1], dtype=np.float32))
    return P


def adam(P, G, M, V, t, lr):
    for i in range(len(P)):
        M[i] = 0.9 * M[i] + 0.1 * G[i]
        V[i] = 0.999 * V[i] + 0.001 * G[i] ** 2
        P[i] -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def main():
    print("=" * 96)
    print(f"B v2: one-hot 颜色表示  TAG={TAG}")
    print("=" * 96)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)     # (n,64) 0..15
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    n = len(lab)
    # one-hot: (n, 64, 16) -> (n, 1024)
    OH = np.zeros((n, 64, C), dtype=np.float32)
    OH[np.arange(n)[:, None], np.arange(64)[None, :], lab] = 1.0
    OH = OH.reshape(n, 64 * C) * 2 - 1              # 映射到 [-1,1]
    print(f"数据: one-hot {OH.shape}  (原始索引 {lab.shape})")

    # Encoder: 1024 -> 64 -> 32
    E = init([1024, H_E, D_Z], seed=1)
    # Decoder: [32|32]=64 -> 128 -> 1024
    G = init([D_Z * 2, H_G, 1024], seed=2)
    D = init([2048, 128, 1], seed=3, scale=0.05)
    ME = [np.zeros_like(p) for p in E]; VE = [np.zeros_like(p) for p in E]
    MG = [np.zeros_like(p) for p in G]; VG = [np.zeros_like(p) for p in G]
    MD = [np.zeros_like(p) for p in D]; VD = [np.zeros_like(p) for p in D]
    rng = np.random.default_rng(0)
    t_e = t_g = t_d = 0; t0 = time.time()

    def e_fwd(X):
        a1 = np.maximum(X @ E[0] + E[1], 0)
        return np.maximum(a1 @ E[2] + E[3], 0), a1

    def g_fwd(EZ):
        h = np.maximum(EZ @ G[0] + G[1], 0)
        return np.tanh(h @ G[2] + G[3]), h

    def d_fwd(X):
        a = X @ D[0] + D[1]
        h = np.where(a > 0, a, 0.2 * a)
        return h @ D[2] + D[3], h, a

    for ep in range(1, EPOCHS + 1):
        for _ in range(max(1, n // BS)):
            ib = rng.integers(0, n, BS)
            xb = OH[ib]
            z = rng.normal(0, 1, (BS, D_Z)).astype(np.float32)
            eb, a1 = e_fwd(xb)
            ez = np.concatenate([eb, z], 1)
            xh, hg = g_fwd(ez)
            # ---- D ----
            pr_, hr, ar = d_fwd(np.concatenate([xb, xb], 1))
            pf_, hf, af = d_fwd(np.concatenate([xb, xh], 1))
            d_ar = (sig(pr_) - 0.9) / BS; d_af = (sig(pf_) - 0.0) / BS
            t_d += 1
            gW2 = hr.T @ d_ar + hf.T @ d_af; gb2 = d_ar.sum(0) + d_af.sum(0)
            dhr = (d_ar @ D[2].T) * np.where(ar > 0, 1.0, 0.2)
            dhf = (d_af @ D[2].T) * np.where(af > 0, 1.0, 0.2)
            pairr = np.concatenate([xb, xb], 1); pairf = np.concatenate([xb, xh], 1)
            adam(D, [pairr.T @ dhr + pairf.T @ dhf, dhr.sum(0) + dhf.sum(0), gW2, gb2],
                 MD, VD, t_d, LR)
            # ---- G (对抗 + L1 重构) ----
            z = rng.normal(0, 1, (BS, D_Z)).astype(np.float32)
            eb, a1 = e_fwd(xb)
            ez = np.concatenate([eb, z], 1)
            xh, hg = g_fwd(ez)
            pf_, hf, af = d_fwd(np.concatenate([xb, xh], 1))
            d_af = (sig(pf_) - 0.9) / BS
            dhf = (d_af @ D[2].T) * np.where(af > 0, 1.0, 0.2)
            dPair = dhf @ D[0].T
            d_xh = dPair[:, 1024:] + np.sign(xh - xb) * (2.0 * LAM_REC / (BS * 1024))
            dPre = d_xh * (1 - xh ** 2)
            t_g += 1
            gW2 = hg.T @ dPre; gb2 = dPre.sum(0)
            dH = (dPre @ G[2].T) * (hg > 0)
            adam(G, [ez.T @ dH, dH.sum(0), gW2, gb2], MG, VG, t_g, LR)
            # ---- E ----
            d_ez = dH @ G[0].T
            d_eb = d_ez[:, :D_Z]
            d_a1 = (d_eb @ E[2].T) * (a1 > 0)
            t_e += 1
            adam(E, [xb.T @ d_a1, d_a1.sum(0), a1.T @ d_eb, d_eb.sum(0)],
                 ME, VE, t_e, LR)

        if ep % 250 == 0 or ep == 1:
            zz = np.zeros((200, D_Z), dtype=np.float32)
            eb, _ = e_fwd(OH[:200])
            xr, _ = g_fwd(np.concatenate([eb, zz], 1))
            lab_r = xr.reshape(-1, 64, C).argmax(2)
            acc = float((lab_r == lab[:200]).mean())
            zr = rng.normal(0, 1, (200, D_Z)).astype(np.float32)
            xg, _ = g_fwd(np.concatenate([eb, zr], 1))
            lab_g = xg.reshape(-1, 64, C).argmax(2)
            acc_g = float((lab_g == lab[:200]).mean())
            pr = sig(d_fwd(np.concatenate([OH[:200], OH[:200]], 1))[0]).mean()
            pf = sig(d_fwd(np.concatenate([OH[:200], xg], 1))[0]).mean()
            print(f"  ep {ep:5d}  D(真)={pr:.3f} D(假)={pf:.3f}  "
                  f"重建像素准确率={acc:.3f}  生成像素准确率={acc_g:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    json.dump({'E': [p.tolist() for p in E], 'G': [p.tolist() for p in G],
               'd_z': D_Z, 'h_e': H_E, 'h_g': H_G, 'palette': pal.tolist(),
               'mode': 'condB_onehot'},
              open(os.path.join(RES, f'gan_generator_{TAG}.json'), 'w'))

    SC = 12
    def to_img(l):
        return pal[l].reshape(8, 8, 3)

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
    eb, _ = e_fwd(OH[sel])
    z0 = np.zeros((48, D_Z), dtype=np.float32)
    xr, _ = g_fwd(np.concatenate([eb, z0], 1))
    Image.fromarray(grid([to_img(v) for v in xr.reshape(-1, 64, C).argmax(2)])).save(
        os.path.join(RES, f'gan_recon_{TAG}.png'))
    zr = r2.normal(0, 1, (48, D_Z)).astype(np.float32)
    xg, _ = g_fwd(np.concatenate([eb, zr], 1))
    Image.fromarray(grid([to_img(v) for v in xg.reshape(-1, 64, C).argmax(2)])).save(
        os.path.join(RES, f'gan_samples_{TAG}.png'))
    Image.fromarray(grid([to_img(v) for v in lab[sel]])).save(
        os.path.join(RES, f'gan_input_{TAG}.png'))
    zf = r2.normal(0, 1, (48, D_Z)).astype(np.float32)
    xf, _ = g_fwd(np.concatenate([np.zeros((48, D_Z), dtype=np.float32), zf], 1))
    Image.fromarray(grid([to_img(v) for v in xf.reshape(-1, 64, C).argmax(2)])).save(
        os.path.join(RES, f'gan_free_{TAG}.png'))
    print(f"\n已保存 recon / samples / input / free  ({TAG})")


if __name__ == '__main__':
    main()

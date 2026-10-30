#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
卷积式学生 + 对称分解 + HSL 颜色编码
======================================
三个设计决定:
  1. 卷积式局部感受野: 每个输出位置只读自身 3x3 邻域（9 个像素 x 16 色 one-hot = 144 维）
     -> 相邻输出共享输入邻域 -> 空间一致，不产生马赛克
     -> 红石: 扇入从 64 降到 9
  2. 输出 HSL 三通道 (H, S, L) 而非 16 路 one-hot
     -> 输出只需 3 条线；就近锁定到 16 色调色板
     -> 网络可在颜色间插值
  3. 对称/反对称分解
     -> 输入: 左半 s(32) 与反对称差 d(32) 分开
     -> 输出: 只预测左半 32 个位置的 HSL(96 维)，右半用 clamp(s+d) 合成

训练数据: 真实头像的 (3x3 邻域, 中心像素的真实 HSL)
  这是一个**局部去噪算子** —— 反复应用可从噪声迭代出结构

输出文件:
  conv_<档位>_iter<N>.png     第 N 轮迭代
  conv_<档位>_real.png        真实对照
  conv_<档位>_net.json        网络权重
"""
import os, json, sys, time, colorsys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
C = 16
LIDX = np.array([r * 8 + c for r in range(8) for c in range(4)], dtype=int)
RIDX = np.array([r * 8 + (7 - c) for r in range(8) for c in range(4)], dtype=int)

H_HID = int(os.environ.get('H_HID', '64'))
EPOCHS = int(os.environ.get('EPOCHS', '30'))
BS = 256
LR = 1e-3


def rgb_to_hsl_arr(pal):
    out = []
    for c in pal:
        h, l, s = colorsys.rgb_to_hls(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
        out.append([h, s, l])
    return np.array(out, dtype=np.float64)     # (16,3) -> H,S,L


def hsl_to_rgb_arr(hsl):
    out = []
    for h, s, l in hsl:
        r, g, b = colorsys.hls_to_rgb(float(h) % 1.0, float(np.clip(l, 0, 1)), float(np.clip(s, 0, 1)))
        out.append([r * 255, g * 255, b * 255])
    return np.array(out)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'd15'
    dclip = 3 if tag == 'd3' else 15
    print("=" * 96)
    print(f"卷积式学生 + 对称分解 + HSL 编码   {tag}")
    print("=" * 96)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    HSL = rgb_to_hsl_arr(pal)                  # (16,3)
    print(f"数据 {lab.shape}   调色板 {C} 色")
    print(f"HSL 范围: H[{HSL[:,0].min():.2f},{HSL[:,0].max():.2f}] "
          f"S[{HSL[:,1].min():.2f},{HSL[:,1].max():.2f}] "
          f"L[{HSL[:,2].min():.2f},{HSL[:,2].max():.2f}]")

    # ---- 输入编码: 3x3 邻域 padding 后的 one-hot ----
    def neighbors_onehot(X):
        """X:(B,64) 索引 -> (B,64,9*16) 每个位置的3x3邻域 one-hot"""
        B = len(X)
        img = X.reshape(B, 8, 8)
        P = np.zeros((B, 10, 10), dtype=np.int64)       # padding 1
        P[:, 1:9, 1:9] = img
        cols = []
        for dr in range(3):
            for dc in range(3):
                cols.append(P[:, dr:dr+8, dc:dc+8].reshape(B, 64))
        nb = np.stack(cols, 1)                          # (B,64,9)
        nbf = nb.reshape(B, 64 * 9)                     # (B,576)
        oh = np.zeros((B, 64 * 9, C), dtype=np.float32)
        oh[np.arange(B)[:, None], np.arange(64 * 9)[None, :], nbf] = 1.0
        return (oh.reshape(B, 64, 9 * C) * 2 - 1)

    # ---- 目标: 中心像素的 HSL（左半 32 个位置）----
    def target_hsl(X):
        """X:(B,64) -> (B,32,3) 左半位置的 HSL"""
        idx = X[:, LIDX]
        return HSL[idx]                                  # (B,32,3)

    # ---- 训练 ----
    print(f"\n构造训练数据（每位置 3x3 邻域 -> 中心 HSL）...")
    t0 = time.time()
    N = min(len(lab), 20000)
    rng = np.random.default_rng(0)
    sub = lab[rng.choice(len(lab), N, replace=False)]
    # 输入和目标一起构造（内存: N*64*144 float32）
    Nin = 9 * C
    W1 = rng.normal(0, np.sqrt(2.0 / Nin), (Nin, H_HID)).astype(np.float32)
    b1 = np.zeros(H_HID, dtype=np.float32)
    W2 = rng.normal(0, np.sqrt(2.0 / H_HID), (H_HID, 3)).astype(np.float32)
    b2 = np.zeros(3, dtype=np.float32)
    print(f"网络: {Nin} -> {H_HID} -> 3   （每个输出位置独立应用，权重共享）")
    print(f"参数量: {W1.size + H_HID + W2.size + 3}")

    M = [np.zeros_like(p) for p in (W1, b1, W2, b2)]
    V = [np.zeros_like(p) for p in (W1, b1, W2, b2)]
    t = 0
    for ep in range(1, EPOCHS + 1):
        idx = rng.permutation(N)
        tl, tn = 0.0, 0
        for s in range(0, N - BS + 1, BS):
            ib = idx[s:s + BS]
            Xb = sub[ib]
            Xin = neighbors_onehot(Xb)                  # (B,64,144)
            Tb = target_hsl(Xb)                          # (B,32,3)
            XL = Xin[:, LIDX, :]                         # (B,32,144) 只用左半位置
            B_, L_, D_ = XL.shape
            flat = XL.reshape(B_ * L_, D_)
            h = np.maximum(flat @ W1 + b1, 0)
            y = h @ W2 + b2                              # (B*32,3)
            yv = y.reshape(B_, L_, 3)
            d = 2 * (yv - Tb) / (B_ * L_)
            tl += float(((yv - Tb) ** 2).mean()) * B_
            tn += B_
            gW2 = h.T @ d.reshape(-1, 3); gb2 = d.reshape(-1, 3).sum(0)
            dh = (d.reshape(-1, 3) @ W2.T) * (h > 0)
            gW1 = flat.T @ dh; gb1 = dh.sum(0)
            t += 1
            for k, g in enumerate((gW1, gb1, gW2, gb2)):
                M[k] = 0.9 * M[k] + 0.1 * g
                V[k] = 0.999 * V[k] + 0.001 * g ** 2
                P_ = (W1, b1, W2, b2)[k]
                P_ -= LR * (M[k] / (1 - 0.9 ** t)) / (np.sqrt(V[k] / (1 - 0.999 ** t)) + 1e-8)
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:3d}  MSE {tl/max(tn,1):.6f}  ({time.time()-t0:.0f}s)", flush=True)

    json.dump({'W1': W1.tolist(), 'b1': b1.tolist(), 'W2': W2.tolist(), 'b2': b2.tolist(),
               'palette': pal.tolist(), 'hsl': HSL.tolist(),
               'mode': 'conv_sym_hsl', 'kernel': 3, 'hid': H_HID},
              open(os.path.join(RES, f'conv_{tag}_net.json'), 'w'))
    print(f"\n已保存网络 conv_{tag}_net.json")

    # ---- 迭代精修 ----
    def apply_net(X):
        """对整张图应用网络（任意位置都算，但只用于迭代）"""
        B = len(X)
        img = X.reshape(B, 8, 8)
        P = np.zeros((B, 10, 10), dtype=np.int64); P[:, 1:9, 1:9] = img
        outs = np.zeros((B, 64, 3))
        # 分块避免内存
        for pos in range(64):
            r, c = divmod(pos, 8)
            nb = np.stack([P[:, r+dr, c+dc] for dr in range(3) for dc in range(3)], 1)  # (B,9)
            oh = np.zeros((B, 9, C), dtype=np.float32)
            oh[np.arange(B)[:, None], np.arange(9)[None, :], nb] = 1.0
            flat = (oh.reshape(B, 9 * C) * 2 - 1)
            h = np.maximum(flat @ W1 + b1, 0)
            outs[:, pos] = h @ W2 + b2
        return outs                                   # (B,64,3) HSL

    def lock_hsl(hsl):
        """把 HSL 锁定到最近的调色板颜色"""
        d = ((hsl[:, :, None, :] - HSL[None, None, :, :]) ** 2).sum(3)
        return d.argmin(2)                            # (B,64)

    def to_img(l): return pal[l].reshape(8, 8, 3)

    SC = 8
    def grid(imgs, cols=16):
        rows_ = (len(imgs) + cols - 1) // cols
        cv = np.full((rows_ * 8, cols * 8, 3), 40, dtype=np.uint8)
        for i, im in enumerate(imgs):
            r_, c_ = divmod(i, cols)
            cv[r_*8:(r_+1)*8, c_*8:(c_+1)*8] = im
        big = np.kron(cv, np.ones((SC, SC, 1), dtype=np.uint8))
        for r_ in range(rows_ + 1):
            big[r_*8*SC:r_*8*SC+2, :] = 200
        for c_ in range(cols + 1):
            big[:, c_*8*SC:c_*8*SC+2] = 200
        return big

    NIMG = 32
    r2 = np.random.default_rng(123)
    X = r2.integers(0, C, (NIMG, 64))
    for it in range(1, 9):
        hsl = apply_net(X)
        Xn = lock_hsl(hsl)
        # 对称分解: 右半 = clamp(左半 + d)，d 由网络对右半的预测给出
        hs = apply_net(Xn)
        d_pred = np.clip(lock_hsl(hs)[:, RIDX] - lock_hsl(hs)[:, LIDX], -dclip, dclip)
        Xn[:, RIDX] = np.clip(Xn[:, LIDX] + d_pred, 0, C - 1)
        X = Xn
        uc = np.mean([len(np.unique(v)) for v in X])
        sym = float((X[:, LIDX] == X[:, RIDX]).mean())
        Image.fromarray(grid([to_img(v) for v in X])).save(
            os.path.join(RES, f'conv_{tag}_iter{it}.png'))
        print(f"  迭代 {it}: 颜色数 {uc:.2f}  对称率 {sym:.2f}  -> conv_{tag}_iter{it}.png",
              flush=True)

    Image.fromarray(grid([to_img(v) for v in lab[r2.choice(len(lab), NIMG, replace=False)]])).save(
        os.path.join(RES, f'conv_{tag}_real.png'))
    print(f"  真实对照 -> conv_{tag}_real.png")


if __name__ == '__main__':
    main()

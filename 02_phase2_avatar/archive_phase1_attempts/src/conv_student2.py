#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
卷积式 + 掩码预测 + 对称分解 + HSL 编码（修正版）
====================================================
上一版失败原因: MSE 目标 -> 网络学的是**条件均值** -> 反复应用 = 不断模糊
  -> 收敛到 2 色常值图。（预测均值 != 采样分布）

本版修正:
  1. 训练目标改为**分类**（16 色交叉熵），而非回归 MSE
  2. 训练时随机挖掉位置，用**可见位置**预测被挖位置
  3. 生成时**采样**（按 softmax），而非取均值
  4. 输入仍是 3x3 局部邻域（卷积式，红石友好）
  5. 对称分解: 只预测左半 32 个位置，右半用 s+d 合成
  6. HSL 三通道作为**辅助监督**（帮助泛化），但主输出是 16 色分布

输出: conv2_<档位>_iter<N>.png  conv2_<档位>_real.png  conv2_<档位>_net.json
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
H_HID = int(os.environ.get('H_HID', '128'))
EPOCHS = int(os.environ.get('EPOCHS', '40'))
BS = 256
LR = 1e-3
K_R = 3                                    # 感受野半径 1 -> 3x3


def rgb_to_hsl_arr(pal):
    out = []
    for c in pal:
        h, l, s = colorsys.rgb_to_hls(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
        out.append([h, s, l])
    return np.array(out, dtype=np.float64)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'd15'
    dclip = 3 if tag == 'd3' else 15
    print("=" * 96)
    print(f"卷积式 + 掩码预测 + 对称分解 + HSL   {tag}")
    print("=" * 96)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    HSL = rgb_to_hsl_arr(pal)
    N = min(len(lab), 30000)
    rng = np.random.default_rng(0)
    sub = lab[rng.choice(len(lab), N, replace=False)]
    print(f"数据 {lab.shape} -> 用 {N}   调色板 {C} 色")

    # 输入维度: 3x3 邻域(9) x 16 色 one-hot + 位置 one-hot(32) = 144 + 32 = 176
    NIN = 9 * C + 9 + 32      # 9 邻域x16色 + 9 未知标志 + 32 位置
    W1 = rng.normal(0, np.sqrt(2.0 / NIN), (NIN, H_HID)).astype(np.float32)
    b1 = np.zeros(H_HID, dtype=np.float32)
    W2 = rng.normal(0, np.sqrt(2.0 / H_HID), (H_HID, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)
    W3 = rng.normal(0, np.sqrt(2.0 / H_HID), (H_HID, 3)).astype(np.float32)
    b3 = np.zeros(3, dtype=np.float32)
    P = [W1, b1, W2, b2, W3, b3]
    M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
    nparam = sum(p.size for p in P)
    print(f"网络: {NIN} -> {H_HID} -> (16 色 logits | 3 HSL)   参数量 {nparam}")

    POSOH = np.eye(32, dtype=np.float32)       # 左半位置 one-hot

    def make_batch(Xb, seed):
        """
        随机挖掉一部分位置（只挖左半），用可见的 3x3 邻域预测被挖位置
        返回 Xin, 目标位置, 目标颜色
        """
        B = len(Xb)
        r = np.random.default_rng(seed)
        n_unk = r.integers(8, 33, B)                     # 每样本挖 8~32 个
        unk = np.zeros((B, 32), dtype=bool)
        for i in range(B):
            unk[i, r.choice(32, n_unk[i], replace=False)] = True
        Xv = Xb.copy()
        # 把被挖位置的左半值置为一个"未知"色（用 0），并记录
        Xv[:, LIDX] = np.where(unk, 0, Xb[:, LIDX])
        img = Xv.reshape(B, 8, 8)
        Pd = np.zeros((B, 10, 10), dtype=np.int64); Pd[:, 1:9, 1:9] = img
        # 每个左半位置的 3x3 邻域
        outs = []
        for i in range(B):
            cols = []
            for k, pos in enumerate(LIDX):
                rr, cc = divmod(pos, 8)
                cols.append(Pd[i, rr:rr+3, cc:cc+3].reshape(9))
            outs.append(np.stack(cols))                   # (32,9)
        nb = np.stack(outs)                               # (B,32,9)
        oh = np.zeros((B, 32, 9, C), dtype=np.float32)
        oh[np.arange(B)[:, None, None], np.arange(32)[None, :, None],
           np.arange(9)[None, None, :], nb] = 1.0
        oh = (oh.reshape(B, 32, 9 * C) * 2 - 1)
        Xin = np.concatenate([oh, np.tile(POSOH[None], (B, 1, 1))], 2)   # (B,32,176)
        return Xin, unk

    t0 = time.time()
    t = 0
    for ep in range(1, EPOCHS + 1):
        idx = rng.permutation(N)
        tl, tc, tn = 0.0, 0, 0
        for s in range(0, N - BS + 1, BS):
            Xb = sub[idx[s:s + BS]]
            Xin, unk = make_batch(Xb, seed=ep * 100003 + s)
            B_, L_, D_ = Xin.shape
            flat = Xin.reshape(B_ * L_, D_)
            h = np.maximum(flat @ W1 + b1, 0)
            lg = (h @ W2 + b2).reshape(B_, L_, C)
            hsl = (h @ W3 + b3).reshape(B_, L_, 3)
            tgt = Xb[:, LIDX]                              # (B,32)
            # 只对被挖位置算损失
            m = unk.reshape(B_, L_)
            lgf = lg.reshape(-1, C)[m.reshape(-1)]
            tgf = tgt.reshape(-1)[m.reshape(-1)]
            if len(tgf) == 0:
                continue
            mx = lgf.max(1, keepdims=True)
            e = np.exp(lgf - mx); sm = e / e.sum(1, keepdims=True)
            tl += float((-np.log(sm[np.arange(len(tgf)), tgf] + 1e-12)).mean()) * len(tgf)
            tc += int((lgf.argmax(1) == tgf).sum()); tn += len(tgf)
            # 梯度
            dl = np.zeros((B_ * L_, C), dtype=np.float32)
            idxm = np.flatnonzero(m.reshape(-1))
            dl[idxm] = sm.copy(); dl[idxm, tgf] -= 1; dl[idxm] /= len(tgf)
            # HSL 辅助损失
            hslf = hsl.reshape(-1, 3)
            hsl_t = HSL[tgt.reshape(-1)]
            dhsl = np.zeros_like(hslf)
            w_hsl = 0.05
            dhsl[idxm] = 2 * (hslf[idxm] - hsl_t[idxm]) * w_hsl / len(tgf)
            gW2 = h.T @ dl; gb2 = dl.sum(0)
            gW3 = h.T @ dhsl; gb3 = dhsl.sum(0)
            dh = (dl @ W2.T + dhsl @ W3.T) * (h > 0)
            gW1 = flat.T @ dh; gb1 = dh.sum(0)
            t += 1
            for k, g in enumerate((gW1, gb1, gW2, gb2, gW3, gb3)):
                M[k] = 0.9 * M[k] + 0.1 * g
                V[k] = 0.999 * V[k] + 0.001 * g ** 2
                P[k] -= LR * (M[k] / (1 - 0.9 ** t)) / (np.sqrt(V[k] / (1 - 0.999 ** t)) + 1e-8)
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d}  CE {tl/max(tn,1):.4f}  准确率 {tc/max(tn,1):.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    json.dump({'W1': W1.tolist(), 'b1': b1.tolist(), 'W2': W2.tolist(), 'b2': b2.tolist(),
               'W3': W3.tolist(), 'b3': b3.tolist(),
               'palette': pal.tolist(), 'hsl': HSL.tolist(),
               'mode': 'conv_mask_sym_hsl', 'hid': H_HID},
              open(os.path.join(RES, f'conv2_{tag}_net.json'), 'w'))
    print(f"已保存 conv2_{tag}_net.json")

    # ---------- 生成 ----------
    def gen(nn, seed=0, temp=1.0, rounds=24):
        r = np.random.default_rng(seed)
        X = np.zeros((nn, 64), dtype=np.int64)
        known = np.zeros((nn, 64), dtype=bool)
        for rd in range(rounds):
            # 对每个样本，选出仍未知的左半位置
            todo = [np.flatnonzero(~known[i, LIDX]) for i in range(nn)]
            if all(len(t) == 0 for t in todo):
                break
            # 一次性算所有样本（用当前 X）
            Xin, _ = make_batch_full(X, known)
            B_, L_, D_ = Xin.shape
            h = np.maximum(Xin.reshape(-1, D_) @ W1 + b1, 0)
            lg = (h @ W2 + b2).reshape(B_, L_, C) / max(temp, 1e-6)
            mx = lg.max(2, keepdims=True)
            pr = np.exp(lg - mx); pr /= pr.sum(2, keepdims=True)
            # 对每个样本，随机挑一个未知位置填入（采样）
            for i in range(nn):
                if len(todo[i]) == 0:
                    continue
                k = r.choice(todo[i])
                col = r.choice(C, p=pr[i, k])
                X[i, LIDX[k]] = col
                known[i, LIDX[k]] = True
            # 右半用 s+d 合成（d 先用 0，后续可加）
            X[:, RIDX] = X[:, LIDX]
        return X

    def make_batch_full(Xb, known):
        """全位置输入（不挖洞），用于生成"""
        B = len(Xb)
        img = Xb.reshape(B, 8, 8)
        Pd = np.zeros((B, 10, 10), dtype=np.int64); Pd[:, 1:9, 1:9] = img
        outs = []
        for pos in LIDX:
            rr, cc = divmod(pos, 8)
            outs.append(Pd[:, rr:rr+3, cc:cc+3].reshape(B, 9))
        nb = np.stack(outs, 1)                            # (B,32,9)
        oh = np.zeros((B, 32, 9, C), dtype=np.float32)
        oh[np.arange(B)[:, None, None], np.arange(32)[None, :, None],
           np.arange(9)[None, None, :], nb] = 1.0
        oh = (oh.reshape(B, 32, 9 * C) * 2 - 1)
        return np.concatenate([oh, np.tile(POSOH[None], (B, 1, 1))], 2), None

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
    for temp, nm in ((0.7, 'T07'), (1.0, 'T10')):
        G = gen(NIMG, seed=77, temp=temp, rounds=40)
        uc = np.mean([len(np.unique(v)) for v in G])
        sym = float((G[:, LIDX] == G[:, RIDX]).mean())
        Image.fromarray(grid([to_img(v) for v in G])).save(
            os.path.join(RES, f'conv2_{tag}_{nm}.png'))
        print(f"  {nm}: 颜色数 {uc:.2f}  对称率 {sym:.2f}  -> conv2_{tag}_{nm}.png", flush=True)
    r2 = np.random.default_rng(7)
    Image.fromarray(grid([to_img(v) for v in lab[r2.choice(len(lab), NIMG, replace=False)]])).save(
        os.path.join(RES, f'conv2_{tag}_real.png'))
    print(f"  真实对照 -> conv2_{tag}_real.png")


if __name__ == '__main__':
    main()

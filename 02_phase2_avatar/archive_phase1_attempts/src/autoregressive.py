#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
备选方案：自回归模型生成 8x8 头像（离散数据上比 GAN 稳定）
=============================================================
原理: 链式法则把联合分布拆成 64 个条件分布的乘积
   p(x) = prod_{i=1..64} p(x_i | x_1..x_{i-1})
   逐像素生成: 先定像素0, 再依像素0定像素1, ...

实现: 一个共享的前馈网络，第 i 步读 [已生成像素的 one-hot | 位置编码 i]
   输出版本 16 个 logits -> 采样/argmax
   （类似 PixelCNN 的"无 mask 的逐位置版本"，用位置 i 作为输入区分位置）

为什么选它:
  - 训练就是最大似然，**没有对抗** -> 不会崩
  - 离散输出用交叉熵 -> 梯度干净
  - 推理时 64 步前向，**每步都是同一个网络** -> 红石上可复用同一套结构

代价:
  - 生成需要 64 次前向（延迟高），但红石上可以流水线
  - 需要把"已生成像素"作为输入回灌
"""
import os, json, numpy as np, time
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")

C = 16; D = 64; EP = int(os.environ.get('EPOCHS', '400'))
BS = 64; LR = 1e-3; H = 256
TAG = os.environ.get('TAG', 'ar')
# 输入 = [已生成像素的标量索引(未知填 -1 → 映射到 0) | 未知掩码 | 位置索引] = 64+64+64
IN_DIM = D * 3


def init(dims, seed=0):
    rng = np.random.default_rng(seed); P = []
    for i in range(len(dims) - 1):
        P.append(rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1])).astype(np.float32))
        P.append(np.zeros(dims[i + 1], dtype=np.float32))
    return P


def adam(P, G, M, V, t, lr):
    for i in range(len(P)):
        M[i] = 0.9 * M[i] + 0.1 * G[i]
        V[i] = 0.999 * V[i] + 0.001 * G[i] ** 2
        P[i] -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)


def build_input(lab, mask):
    """
    lab: (B,64) 索引 0..C-1; mask: (B,64) bool, True 表示该位置**已知**
    输入 = [ 归一化颜色索引(未知处 0) | 未知掩码(1=未知) | 位置索引 ] = 192 维
    紧凑表示，避免 5184 维的大瓶颈
    """
    B = len(lab)
    col = (lab.astype(np.float32) - (C - 1) / 2) / ((C - 1) / 2)     # -> [-1,1]
    col = col * mask                                                  # 未知处置 0
    unk = (~mask).astype(np.float32)
    pos = np.tile(np.arange(D, dtype=np.float32)[None] / D * 2 - 1, (B, 1))
    return np.concatenate([col, unk, pos], 1)


def main():
    print("=" * 96)
    print(f"自回归模型生成 8x8 头像  TAG={TAG}")
    print("=" * 96)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    n = len(lab)
    print(f"数据: {lab.shape}  位置数 {D}  颜色 {C}")
    print(f"网络输入维度 = 64*16 + 64(掩码) + 64(位置) = {64*16+64+64}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(n); ntr = int(n * 0.9)
    tr, te = lab[perm[:ntr]], lab[perm[ntr:]]
    print(f"train {len(tr)}  test {len(te)}")

    P = init([IN_DIM, H, C], seed=1)
    M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
    t = 0; t0 = time.time()

    for ep in range(1, EP + 1):
        idx = rng.permutation(len(tr))
        tot, corr = 0.0, 0
        for s in range(0, len(tr) - BS + 1, BS):
            xb = tr[idx[s:s + BS]]
            # 随机掩码: 每个样本随机决定"已知前 k 个像素"
            k = rng.integers(0, D, len(xb))              # 已知像素个数
            mask = (np.arange(D)[None, :] < k[:, None])
            # 目标: 第 k 个像素（第一个未知的）
            tgt_pos = k
            Xin = build_input(xb, mask)
            h = np.maximum(Xin @ P[0] + P[1], 0)
            logits = h @ P[2] + P[3]                     # (B,C)
            # softmax 交叉熵，只对目标位置那个像素
            tgt = xb[np.arange(len(xb)), tgt_pos]
            mx = logits.max(1, keepdims=True)
            e = np.exp(logits - mx); sm = e / e.sum(1, keepdims=True)
            ll = float((mx[:, 0] + np.log(e.sum(1))).mean())
            tot += -ll * len(xb)
            corr += int((logits.argmax(1) == tgt).sum())
            dlog = sm.copy(); dlog[np.arange(len(xb)), tgt] -= 1; dlog /= len(xb)
            gW2 = h.T @ dlog; gb2 = dlog.sum(0)
            dh = (dlog @ P[2].T) * (h > 0)
            gW1 = Xin.T @ dh; gb1 = dh.sum(0)
            t += 1
            adam(P, [gW1, gb1, gW2, gb2], M, V, t, LR)

        if ep % 20 == 0 or ep == 1:
            # 验证: 逐像素条件准确率
            vc, vt = 0, 0
            for s in range(0, min(len(te), 512), BS):
                xb = te[s:s + BS]
                k = rng.integers(0, D, len(xb))
                mask = (np.arange(D)[None, :] < k[:, None])
                Xin = build_input(xb, mask)
                h = np.maximum(Xin @ P[0] + P[1], 0)
                lg = h @ P[2] + P[3]
                vt += len(xb)
                vc += int((lg.argmax(1) == xb[np.arange(len(xb)), k]).sum())
            print(f"  ep {ep:4d}  训练损失 {tot/max(len(tr)//BS*BS,1):.4f}  "
                  f"训练逐像素准确率 {corr/max(len(tr)//BS*BS,1):.3f}  "
                  f"**验证逐像素准确率 {vc/vt:.3f}**  ({time.time()-t0:.0f}s)", flush=True)

    json.dump({'W1': P[0].tolist(), 'b1': P[1].tolist(), 'W2': P[2].tolist(),
               'b2': P[3].tolist(), 'palette': pal.tolist(), 'mode': 'autoregressive'},
              open(os.path.join(RES, f'gan_generator_{TAG}.json'), 'w'))

    # ---- 生成: 从位置 0 开始逐个采样 ----
    def sample(nn, seed=0, temp=1.0):
        r = np.random.default_rng(seed)
        out = np.zeros((nn, D), dtype=np.int64)
        for i in range(D):
            mask = np.zeros((nn, D), dtype=bool); mask[:, :i] = True
            Xin = build_input(out, mask)
            h = np.maximum(Xin @ P[0] + P[1], 0)
            lg = (h @ P[2] + P[3]) / max(temp, 1e-6)
            mx = lg.max(1, keepdims=True)
            e = np.exp(lg - mx); pr = e / e.sum(1, keepdims=True)
            if temp <= 1e-3:
                out[:, i] = lg.argmax(1)
            else:
                out[:, i] = [r.choice(C, p=pr[j]) for j in range(nn)]
        return out

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

    for temp, nm in ((1.0, 'samples'), (0.7, 'samples_t07')):
        S = sample(48, seed=123, temp=temp)
        Image.fromarray(grid([to_img(v) for v in S])).save(
            os.path.join(RES, f'ar_{nm}_{TAG}.png'))
    r2 = np.random.default_rng(7)
    Image.fromarray(grid([to_img(v) for v in te[r2.choice(len(te), 48, replace=False)]])).save(
        os.path.join(RES, f'ar_real_{TAG}.png'))
    print(f"\n已保存: ar_samples_{TAG}.png / ar_samples_t07_{TAG}.png / ar_real_{TAG}.png")


if __name__ == '__main__':
    main()

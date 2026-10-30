#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自回归 v2：任意顺序掩码（每个位置看到"除自己外的全部像素"）
==============================================================
v1 的问题: 用光栅顺序（0->1->...->63），预测第 8 个像素（第二行行首）时
  它右边和下面的像素还全未知 -> 空间上下文被割裂 -> 生成噪点。

v2 修正: 训练时对每个位置 i，输入 = **除 i 之外的全部像素**（+位置编码）
  推理时循环 64 轮，每轮选一个"最确定"的未知位置填入（置信度最高优先）。
  这样每个位置的预测都能利用完整空间上下文。

网络: [颜色索引(64) | 未知掩码(64) | 位置索引(64)] = 192 -> 256 -> 16
"""
import os, json, numpy as np, time
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")

C = 16; D = 64; EP = int(os.environ.get('EPOCHS', '600'))
BS = 64; LR = 1e-3; H = 256
TAG = os.environ.get('TAG', 'ar2')
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
    col = (lab.astype(np.float32) - (C - 1) / 2) / ((C - 1) / 2)
    col = col * mask
    unk = (~mask).astype(np.float32)
    pos = np.tile(np.arange(D, dtype=np.float32)[None] / D * 2 - 1, (len(lab), 1))
    return np.concatenate([col, unk, pos], 1), pos


def fwd(P, Xin):
    h = np.maximum(Xin @ P[0] + P[1], 0)
    return h @ P[2] + P[3], h


def main():
    print("=" * 96)
    print(f"自回归 v2：任意顺序掩码  TAG={TAG}")
    print("=" * 96)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    n = len(lab)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n); ntr = int(n * 0.9)
    tr, te = lab[perm[:ntr]], lab[perm[ntr:]]
    print(f"数据 {lab.shape}  train {len(tr)}  test {len(te)}  输入维 {IN_DIM}")

    P = init([IN_DIM, H, C], seed=1)
    M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
    t = 0; t0 = time.time()

    for ep in range(1, EP + 1):
        idx = rng.permutation(len(tr))
        tl, tc, tn = 0.0, 0, 0
        for s in range(0, len(tr) - BS + 1, BS):
            xb = tr[idx[s:s + BS]]
            k = rng.integers(0, D, len(xb))                 # 未知的位置
            mask = np.ones((len(xb), D), dtype=bool)
            mask[np.arange(len(xb)), k] = False
            Xin, _ = build_input(xb, mask)
            lg, h = fwd(P, Xin)
            tgt = xb[np.arange(len(xb)), k]
            mx = lg.max(1, keepdims=True)
            e = np.exp(lg - mx); sm = e / e.sum(1, keepdims=True)
            tl += float((-np.log(sm[np.arange(len(xb)), tgt] + 1e-12)).mean()) * len(xb)
            tc += int((lg.argmax(1) == tgt).sum()); tn += len(xb)
            dlog = sm.copy(); dlog[np.arange(len(xb)), tgt] -= 1; dlog /= len(xb)
            gW2 = h.T @ dlog; gb2 = dlog.sum(0)
            dh = (dlog @ P[2].T) * (h > 0)
            gW1 = Xin.T @ dh; gb1 = dh.sum(0)
            t += 1
            adam(P, [gW1, gb1, gW2, gb2], M, V, t, LR)

        if ep % 40 == 0 or ep == 1:
            vc, vn = 0, 0
            for s in range(0, len(te), BS):
                xb = te[s:s + BS]
                k = rng.integers(0, D, len(xb))
                mask = np.ones((len(xb), D), dtype=bool); mask[np.arange(len(xb)), k] = False
                Xin, _ = build_input(xb, mask)
                lg, _ = fwd(P, Xin)
                vc += int((lg.argmax(1) == xb[np.arange(len(xb)), k]).sum()); vn += len(xb)
            print(f"  ep {ep:4d}  损失 {tl/max(tn,1):.4f}  训练准确率 {tc/max(tn,1):.3f}  "
                  f"**验证准确率 {vc/max(vn,1):.3f}**  ({time.time()-t0:.0f}s)", flush=True)

    json.dump({'W1': P[0].tolist(), 'b1': P[1].tolist(), 'W2': P[2].tolist(),
               'b2': P[3].tolist(), 'palette': pal.tolist(), 'mode': 'ar_anyorder'},
              open(os.path.join(RES, f'gan_generator_{TAG}.json'), 'w'))

    # ---- 采样: 逐轮填"置信度最高"的未知位置 ----
    def sample(nn, seed=0, temp=1.0, greedy=False):
        r = np.random.default_rng(seed)
        out = np.zeros((nn, D), dtype=np.int64)
        known = np.zeros((nn, D), dtype=bool)
        for step in range(D):
            Xin, _ = build_input(out, known)
            lg, _ = fwd(P, Xin)
            # 只考虑仍未知的位置
            lg = lg / max(temp, 1e-6)
            mx = lg.max(1, keepdims=True)
            pr = np.exp(lg - mx); pr /= pr.sum(1, keepdims=True)
            conf = pr.max(1)[:, None] * (~known) + (-1.0) * known   # (nn,D)
            pick = conf.argmax(1)                   # (nn,) 最确定的未知位置
            rows = np.arange(nn)
            if greedy:
                val = lg[rows, pick].argmax(1)
            else:
                val = np.array([r.choice(C, p=pr[j]) for j in range(nn)])
            out[rows, pick] = val
            known[rows, pick] = True
        return out

    SC = 12
    def to_img(l): return pal[l].reshape(8, 8, 3)

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

    for temp, nm, gr in ((1.0, 'samples', False), (0.7, 'samples_t07', False),
                         (1.0, 'samples_greedy', True)):
        S = sample(48, seed=123, temp=temp, greedy=gr)
        Image.fromarray(grid([to_img(v) for v in S])).save(
            os.path.join(RES, f'ar2_{nm}_{TAG}.png'))
    r2 = np.random.default_rng(7)
    Image.fromarray(grid([to_img(v) for v in te[r2.choice(len(te), 48, replace=False)]])).save(
        os.path.join(RES, f'ar2_real_{TAG}.png'))
    print(f"已保存: ar2_samples_{TAG}.png / _t07 / _greedy / ar2_real_{TAG}.png")


if __name__ == '__main__':
    main()

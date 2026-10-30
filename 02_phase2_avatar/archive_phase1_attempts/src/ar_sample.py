#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用已训练的任意顺序自回归模型出图（不重训）
==============================================
采样策略: 逐轮挑"置信度最高"的未知位置填入（共 64 轮）
  贪心 / 温度 1.0 / 温度 0.7 三种
"""
import os, json, numpy as np, sys
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results"); DATA = os.path.join(ROOT, "data")
C = 16; D = 64
TAG = sys.argv[1] if len(sys.argv) > 1 else 'ar2a'


def build_input(lab, mask):
    col = (lab.astype(np.float32) - (C - 1) / 2) / ((C - 1) / 2)
    col = col * mask
    unk = (~mask).astype(np.float32)
    pos = np.tile(np.arange(D, dtype=np.float32)[None] / D * 2 - 1, (len(lab), 1))
    return np.concatenate([col, unk, pos], 1)


j = json.load(open(os.path.join(RES, f'gan_generator_{TAG}.json')))
W1 = np.array(j['W1'], dtype=np.float32); b1 = np.array(j['b1'], dtype=np.float32)
W2 = np.array(j['W2'], dtype=np.float32); b2 = np.array(j['b2'], dtype=np.float32)
pal = np.array(j['palette'], dtype=np.uint8)
print(f"载入 {TAG}: W1{W1.shape} W2{W2.shape}")


def fwd(X):
    h = np.maximum(X @ W1 + b1, 0)
    return h @ W2 + b2


def sample(nn, seed=0, temp=1.0, greedy=False):
    r = np.random.default_rng(seed)
    out = np.zeros((nn, D), dtype=np.int64)
    known = np.zeros((nn, D), dtype=bool)
    rows = np.arange(nn)
    for step in range(D):
        lg = fwd(build_input(out, known)) / max(temp, 1e-6)
        mx = lg.max(1, keepdims=True)
        pr = np.exp(lg - mx); pr /= pr.sum(1, keepdims=True)
        conf = pr.max(1)[:, None] * (~known) + (-1.0) * known   # (nn,D)
        pick = conf.argmax(1)
        if greedy:
            val = pr[rows, pick].argmax(1) if False else np.array(
                [r.choice(C, p=pr[i]) for i in range(nn)])  # 贪心=取该位置概率最大的颜色
        else:
            val = np.array([r.choice(C, p=pr[i]) for i in range(nn)])
        out[rows, pick] = val
        known[rows, pick] = True
    return out


def to_img(l): return pal[l].reshape(8, 8, 3)


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


lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
for temp, nm, gr in ((1.0, 'samples', False), (0.8, 'samples_t08', False),
                     (1.0, 'samples_greedy', True)):
    S = sample(60, seed=123, temp=temp, greedy=gr)
    uc = np.mean([len(np.unique(s)) for s in S])
    # 与真实样本的最近汉明距离（新颖性）
    dd = np.mean([(lab != s).sum(1).min() for s in S])
    Image.fromarray(grid([to_img(v) for v in S], 12)).save(
        os.path.join(RES, f'ar2_{nm}_{TAG}.png'))
    print(f"  {nm:>16}: 平均颜色数 {uc:.2f}  到最近真实 {dd:.1f} 像素")

r2 = np.random.default_rng(7)
Image.fromarray(grid([to_img(v) for v in lab[r2.choice(len(lab), 60, replace=False)]], 12)).save(
    os.path.join(RES, f'ar2_real_{TAG}.png'))
lab_u = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
print(f"  真实参考: 平均颜色数 {np.mean([len(np.unique(s)) for s in lab[:2000]]):.2f}")
print(f"已保存 ar2_*_{TAG}.png")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从已训练模型采样（修正模式崩塌）
==================================
上一版的错误: 每轮都填"置信度最高"的位置 -> **填充顺序确定性** ->
  所有样本走同一条路径 -> 60 张图几乎相同（模式崩塌）。

修正:
  A. 随机顺序:      每轮随机挑一个未知位置
  B. 置信度顺序+温度: 按置信度挑位置，但颜色按温度采样（而非argmax）
  C. 固定顺序(光栅):  已知识必为前缀
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


def sample(nn, seed=0, temp=1.0, mode='random'):
    r = np.random.default_rng(seed)
    out = np.zeros((nn, D), dtype=np.int64)
    known = np.zeros((nn, D), dtype=bool)
    rows = np.arange(nn)
    for step in range(D):
        lg = fwd(build_input(out, known)) / max(temp, 1e-6)
        mx = lg.max(1, keepdims=True)
        pr = np.exp(lg - mx); pr /= pr.sum(1, keepdims=True)
        if mode == 'confidence':
            conf = pr.max(1)[:, None] * (~known) + (-1.0) * known
            pick = conf.argmax(1)
        else:                                   # 随机挑一个未知位置
            pick = np.array([r.choice(np.flatnonzero(~known[i])) for i in range(nn)])
        # 颜色按温度采样（不是 argmax），保证随机性
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

def diversity(S):
    """两两之间的平均汉明距离 —— 衡量样本是否互相不同"""
    A = S[:, None, :] != S[None, :, :]
    n = len(S)
    return float(A.sum(2).sum() / (n * (n - 1)))

real_div = diversity(lab[:120])
print(f"真实头像的两两平均汉明距离 = {real_div:.1f}  (这是多样性的上限参考)")

for mode, temp, nm in (('random', 1.0, 'samples_rand'),
                       ('random', 0.8, 'samples_rand_t08'),
                       ('confidence', 1.5, 'samples_conf_t15')):
    S = sample(60, seed=123, temp=temp, mode=mode)
    dv = diversity(S)
    uc = np.mean([len(np.unique(s)) for s in S])
    dd = np.mean([(lab != s).sum(1).min() for s in S])
    Image.fromarray(grid([to_img(v) for v in S], 12)).save(
        os.path.join(RES, f'ar3_{nm}_{TAG}.png'))
    print(f"  {nm:>18}: 颜色数 {uc:.2f}  **样本间多样性 {dv:.1f}**  到最近真实 {dd:.1f}  "
          f"(真实多样性 {real_div:.1f})")

r2 = np.random.default_rng(7)
Image.fromarray(grid([to_img(v) for v in lab[r2.choice(len(lab), 60, replace=False)]], 12)).save(
    os.path.join(RES, f'ar3_real_{TAG}.png'))
print(f"已保存 ar3_*_{TAG}.png")

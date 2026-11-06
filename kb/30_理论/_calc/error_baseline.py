# -*- coding: utf-8 -*-
"""三类误差的随机基线：不依赖任何具体网络。

基线设定（作者 2026-10-31 指定）：
    输入 16 通道、输出 16 通道的**随机**全连接层；
    输入位宽 = 输出位宽 = b（固定相等）；
    权重 iid 正态，无偏置；
    输入 x ~ U[-3.5, 3.5]。

在基线上扫描 输入个数 n 与 位宽 b，分别量三类误差：
    eps_x  输入量化：rms(xhat - x) / Delta
    eps_w  权重取整：rms( (N-P)/sigma - eta(xhat) ) / Delta_eta   （输入用逆回值，排除 eps_x）
    eps_y  输出量化：delta_Y / (2 Delta_eta)，delta_Y = Gamma / sigma

输出：控制台表 + kb/30_理论/_calc/_error_baseline.txt + kb/figures/error_scaling.png
用法： python error_baseline.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_TXT = os.path.join(HERE, '_error_baseline.txt')
OUT_PNG = r'E:\redstone_nn\kb\figures\error_scaling.png'

X_MIN, X_MAX = -3.5, 3.5
DELTA = X_MAX - X_MIN
NSAMP = 4096
TRIALS = 40


def one_layer(n, m, b, wref, rng, bias=False):
    """返回 eps_x, eps_w, eps_y 三个数（该次随机实例）。"""
    delta = DELTA / 2 ** b
    w = rng.normal(0.0, 1.0 / np.sqrt(n), size=(m, n))
    th = rng.normal(0.0, 0.3, size=m) if bias else np.zeros(m)
    x = rng.uniform(X_MIN, X_MAX, size=(NSAMP, n))

    # --- 输入量化
    X = np.clip(np.floor((x - X_MIN) / DELTA * 2 ** b), 0, 2 ** b - 1).astype(np.int64)
    xhat = X_MIN + (X + 0.5) * delta
    eps_x = np.sqrt(np.mean((xhat - x) ** 2)) / DELTA

    # --- 缩放（逐通道 sigma_j，隐层口径）
    amax = np.abs(w).max(axis=1)                     # 每行最大 |w|
    sigma = wref / (amax * delta * 2 ** (b - 2))     # (m,)

    # --- 位权重 W^(k)_{j,i}
    k = np.arange(b)
    W = np.rint(sigma[:, None, None] * w[:, :, None] * delta * 2.0 ** k[None, None, :])  # (m,n,b)
    bits = ((X[:, :, None] >> k[None, None, :]) & 1).astype(np.int64)                     # (S,n,b)

    # --- 累加与非负化
    C = np.einsum('snb,mnb->sm', bits, W)
    Cmin = np.minimum(W, 0).sum(axis=(1, 2))
    N = C - Cmin[None, :]

    # --- 零点：x=0 处的 C
    _x0 = int(np.clip(np.floor((0.0 - X_MIN) / DELTA * 2 ** b), 0, 2 ** b - 1))
    X0 = np.full(n, _x0, dtype=np.int64)
    b0 = ((X0[:, None] >> k[None, :]) & 1).astype(np.int64)
    P = np.einsum('nb,mnb->m', b0, W) - Cmin

    # --- 真实预激活（用逆回输入，排除 eps_x；并以 X0 处的格点为共同原点）
    xhat0 = X_MIN + (X0 + 0.5) * delta
    eta = (xhat - xhat0[None, :]) @ w.T
    eta_hat = (N - P[None, :]) / sigma[None, :]
    d_eta = eta.max(axis=0) - eta.min(axis=0)
    d_eta = np.maximum(d_eta, 1e-12)
    eps_w = np.sqrt(np.mean((eta_hat - eta) ** 2, axis=0)) / d_eta
    eps_w = float(np.mean(eps_w))

    # --- 输出量化
    Nmax = np.abs(W).sum(axis=(1, 2))
    Gamma = np.maximum(np.ceil((Nmax - P) / (2 ** b - 1)), 1)
    delta_Y = Gamma / sigma
    eps_y = float(np.mean(delta_Y / (2 * d_eta)))

    return eps_x, eps_w, eps_y


def run(ns, bs, wref, m=16, bias=False):
    rows = []
    for n in ns:
        for b in bs:
            acc = np.zeros(3)
            for t in range(TRIALS):
                rng = np.random.default_rng(1000 * b + 17 * n + t)
                acc += np.array(one_layer(n, m, b, wref, rng, bias))
            acc /= TRIALS
            rows.append((n, b, *acc))
    return rows


def main():
    ns = [4, 8, 16, 32, 64]
    bs = [6, 8, 10, 12, 14]
    WREF = 882

    rows = run(ns, bs, WREF)
    L = ['# 三类误差：随机基线（16 输出通道，无偏置，输入 U[-3.5,3.5]）',
         f'# 每格点 {TRIALS} 次随机实例 x {NSAMP} 样本；W_ref = {WREF}', '']
    L.append('=== 扫位宽 b（n = 16 固定）===')
    L.append('   b |      eps_x |      eps_w |      eps_y |   eps_x/2^-(b+1)')
    for (n, b, ex, ew, ey) in rows:
        if n != 16:
            continue
        L.append(f'  {b:2d} | {ex:9.3e} | {ew:9.3e} | {ey:9.3e} | {ex/2.0**-(b+1):.4f}')

    L.append('')
    L.append('=== 扫输入个数 n（b = 10 固定）===')
    L.append('   n |      eps_x |      eps_w |      eps_y | eps_w/sqrt(b)')
    for (n, b, ex, ew, ey) in rows:
        if b != 10:
            continue
        L.append(f'  {n:2d} | {ex:9.3e} | {ew:9.3e} | {ey:9.3e} | {ew/np.sqrt(b):9.3e}')

    L.append('')
    L.append('=== 交叉点（n = 16，看 eps_x 与 eps_w 谁大）===')
    for (n, b, ex, ew, ey) in rows:
        if n != 16:
            continue
        L.append(f'  b={b:2d}  eps_x={ex:9.3e}  eps_w={ew:9.3e}  比值 x/w = {ex/ew:6.2f}')

    txt = chr(10).join(L)
    open(OUT_TXT, 'w', encoding='utf-8').write(txt)
    print(txt)

    W, H = 900, 560
    im = Image.new('RGB', (W, H), (20, 20, 24))
    dr = ImageDraw.Draw(im)
    dr.text((14, 10), 'random baseline: 16 outputs, no bias   log10(eps) vs b', fill=(235, 235, 245))
    bx0, by0, bw, bh = 70, 40, W - 130, H - 110
    dr.rectangle([bx0, by0, bx0 + bw, by0 + bh], outline=(70, 70, 80))
    allv = [r[2] for r in rows] + [r[3] for r in rows] + [r[4] for r in rows]
    lo = np.floor(np.log10(min(allv)))
    hi = np.ceil(np.log10(max(allv)))

    def px(b):
        return bx0 + (b - 6) / (14 - 6) * bw

    def py(v):
        return by0 + bh - (np.log10(v) - lo) / (hi - lo) * bh

    for b in bs:
        x = px(b)
        dr.line([x, by0, x, by0 + bh], fill=(40, 40, 48))
        dr.text((x - 6, by0 + bh + 6), str(b), fill=(160, 160, 170))
    for e in range(int(lo), int(hi) + 1):
        y = by0 + bh - (e - lo) / (hi - lo) * bh
        dr.line([bx0, y, bx0 + bw, y], fill=(40, 40, 48))
        dr.text((bx0 - 34, y - 6), '1e%d' % e, fill=(160, 160, 170))

    COL = {4: (120, 200, 255), 8: (120, 255, 180), 16: (255, 220, 120),
           32: (255, 150, 150), 64: (220, 150, 255)}
    for n in ns:
        pts = [(r[1], r[3]) for r in rows if r[0] == n]
        for (a, va), (b2, vb) in zip(pts, pts[1:]):
            dr.line([px(a), py(va), px(b2), py(vb)], fill=COL[n], width=2)
        dr.text((px(pts[-1][0]) + 6, py(pts[-1][1]) - 6), 'w n=%d' % n, fill=COL[n])
    for key, col, lbl in ((2, (200, 200, 210), 'eps_x'), (4, (255, 110, 110), 'eps_y')):
        pts = [(r[1], r[key]) for r in rows if r[0] == 16]
        for (a, va), (b2, vb) in zip(pts, pts[1:]):
            dr.line([px(a), py(va), px(b2), py(vb)], fill=col, width=2)
        dr.text((px(pts[-1][0]) + 4, py(pts[-1][1]) + 6), lbl, fill=col)
    dr.text((bx0, H - 26), 'coloured = eps_w (various n)   grey = eps_x   red = eps_y   (n=16)',
            fill=(170, 170, 180))
    im.save(OUT_PNG)
    print('wrote', OUT_PNG)


if __name__ == '__main__':
    main()

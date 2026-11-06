# -*- coding: utf-8 -*-
"""三层实验：

一、随机基线上三类误差随 (n, b) 的标度（原有用）
二、以**单层输出逆回误差**为判据的最优位宽
三、两层串联时，上一层 eps_y 与下一层 eps_x 的耦合（加 vs 取大 vs 只用其一）

基线：随机全连接单层；权重 iid 正态(1/sqrt n)；无偏置；输入 U[-3.5,3.5]；
输入位宽 = 输出位宽 = b；逐通道 sigma_j；W_ref = 882。

用法： python error_baseline.py
输出： _error_baseline.txt  +  kb/figures/error_scaling.png
"""
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_TXT = os.path.join(HERE, '_error_baseline.txt')
OUT_PNG = r'E:\redstone_nn\kb\figures\error_scaling.png'

X_MIN, X_MAX = -3.5, 3.5
DELTA = X_MAX - X_MIN
NSAMP = 4096
TRIALS = 40
WREF = 882


# ---------------------------------------------------------------- 单层
def layer_int(w, th, x, wref, b, delta, delta_y=None):
    """把一层跑成整数管线。x 是实数；返回 (eta, relu(eta), Y, 逆回值, 统计量)。"""
    n = x.shape[1]
    X = np.clip(np.floor((x - X_MIN) / delta), 0, 2 ** b - 1).astype(np.int64)
    xhat = X_MIN + (X + 0.5) * delta
    amax = np.abs(w).max(axis=1)
    sigma = wref / (amax * delta * 2 ** (b - 2))
    k = np.arange(b)
    W = np.rint(sigma[:, None, None] * w[:, :, None] * delta * 2.0 ** k[None, None, :])
    bits = ((X[:, :, None] >> k[None, None, :]) & 1).astype(np.int64)
    C = np.einsum('snb,mnb->sm', bits, W)
    Cmin = np.minimum(W, 0).sum(axis=(1, 2))
    N = C - Cmin[None, :]
    _x0 = int(np.clip(np.floor((0.0 - X_MIN) / delta), 0, 2 ** b - 1))
    X0 = np.full(n, _x0, dtype=np.int64)
    b0 = ((X0[:, None] >> k[None, :]) & 1).astype(np.int64)
    P = np.einsum('nb,mnb->m', b0, W) - Cmin
    xhat0 = X_MIN + (X0 + 0.5) * delta
    eta = (xhat - xhat0[None, :]) @ w.T + th[None, :]
    Nmax = np.abs(W).sum(axis=(1, 2))
    Gamma = np.maximum(np.ceil((Nmax - P) / (2 ** b - 1)), 1)
    Y = np.clip(np.floor((N - P[None, :]) / Gamma[None, :]), 0, 2 ** b - 1)
    # 逆回：Y 是"相对零点 P 的整数"，所以 eta_hat = Y*Gamma/sigma（再加回 th 那一支的常数）
    y_hat = np.maximum(Y * Gamma[None, :] / sigma[None, :], 0)
    return dict(eta=eta, y=np.maximum(eta, 0), y_hat=y_hat, Y=Y, sigma=sigma,
                Gamma=Gamma, P=P, xhat=xhat, xhat0=xhat0, X=X, W=W, N=N, Cmin=Cmin)


def one_layer(n, m, b, rng, wref=WREF):
    w = rng.normal(0.0, 1.0 / np.sqrt(n), size=(m, n))
    th = np.zeros(m)
    x = rng.uniform(X_MIN, X_MAX, size=(NSAMP, n))
    delta = DELTA / 2 ** b
    r = layer_int(w, th, x, wref, b, delta)
    params = r
    eta, y, y_hat = r['eta'], r['y'], r['y_hat']

    eps_x = np.sqrt(np.mean((r['xhat'] - x) ** 2)) / DELTA

    # 权重取整：输入用"量化后输入的逆回值" xhat，对比精确路径 —— 排除 eps_x
    eta_w = (r['xhat'] - r['xhat0'][None, :]) @ w.T
    eta_w_hat = (r['N'] - r['P'][None, :]) / r['sigma'][None, :]
    d_eta = params['eta'].max(0) - params['eta'].min(0)
    eps_w = float(np.mean(np.sqrt(np.mean((eta_w_hat - eta_w) ** 2, axis=0)) / np.maximum(d_eta, 1e-12)))

    # 输出量化：参考值是 "N-P 的逆回"，即 eta_hat；不含 eps_x / eps_w
    eta_int = (r['N'] - r['P'][None, :]) / r['sigma'][None, :]
    y_ref = np.maximum(eta_int, 0)
    dY = y_ref.max(0) - y_ref.min(0)
    eps_y = float(np.mean(np.sqrt(np.mean((y_hat - y_ref) ** 2, axis=0)) / np.maximum(dY, 1e-12)))

    # 单层输出逆回误差（判据）：逆回值 vs 真实输出，此时三类误差都在里面
    eps_out = float(np.mean(np.sqrt(np.mean((y_hat - y) ** 2, axis=0)) / np.maximum(dY, 1e-12)))
    return eps_x, eps_w, eps_y, eps_out


def run(ns, bs, wref=WREF, m=16):
    rows = []
    for n in ns:
        for b in bs:
            acc = np.zeros(4)
            for t in range(TRIALS):
                acc += np.array(one_layer(n, m, b, np.random.default_rng(1000 * b + 17 * n + t), wref))
            rows.append((n, b, *(acc / TRIALS)))
    return rows


# ---------------------------------------------------------------- 两层耦合
def two_layer(b1, b2, n=16, m=16, rng=None, mode='full'):
    """A: n->m，B: m->m。返回值：端到端相对误差。"""
    rng = rng or np.random.default_rng(7)
    w1 = rng.normal(0, 1 / np.sqrt(n), size=(m, n))
    w2 = rng.normal(0, 1 / np.sqrt(m), size=(m, m))
    th1, th2 = np.zeros(m), np.zeros(m)
    x = rng.uniform(X_MIN, X_MAX, size=(NSAMP, n))
    d1, d2 = DELTA / 2 ** b1, DELTA / 2 ** b2

    A = layer_int(w1, th1, x, WREF, b1, d1)
    y1_true = A['y']                       # relu 之后的真实值
    # A 的输出取整（真实管线里 B 拿到的就是 A 的整数输出）
    y1_int = A['Y']                        # 0..2^b1-1

    # B 的输入：把 A 的整数输出当作"量化后的输入整数"，跨度是 y1 的取值范围
    span = y1_true.max() - y1_true.min()
    span = max(span, 1e-9)
    Xb = y1_int.astype(np.int64)
    # B 的输入还原：用 A 的逆回（这正是 B 实际看到的实数）
    xb_hat = A['y_hat']

    # B 的真值输入应当是 y1_true；误差就是 A 的输出量化
    eta2_true = y1_true @ w2.T + th2
    if mode == 'full':
        r = layer_int(w2, th2, xb_hat, WREF, b2, d2)
        eta2 = (r['xhat'] - r['xhat0'][None, :]) @ w2.T + th2[None, :]
        err = r['y_hat'] - np.maximum(eta2_true, 0)
    elif mode == 'b_only':           # 只有 B 的误差（输入用真值）
        r = layer_int(w2, th2, y1_true, WREF, b2, d2)
        eta2 = (r['xhat'] - r['xhat0'][None, :]) @ w2.T + th2[None, :]
        err = r['y_hat'] - np.maximum(eta2_true, 0)
    else:                             # only A：B 完全精确
        eta2 = xb_hat @ w2.T + th2
        err = np.maximum(eta2, 0) - np.maximum(eta2_true, 0)
    dY = np.maximum(np.maximum(eta2_true, 0).max(0) - np.maximum(eta2_true, 0).min(0), 1e-12)
    return float(np.mean(np.sqrt(np.mean(err ** 2, axis=0)) / dY))


def main():
    ns = [4, 8, 16, 32, 64]
    bs = [6, 8, 10, 12, 14]

    rows = run(ns, bs)
    L = ['# 三类误差：随机基线（16 输出通道，无偏置，输入 U[-3.5,3.5]）',
         '# 每格点 %d 次随机实例 x %d 样本；W_ref = %d' % (TRIALS, NSAMP, WREF), '']
    L.append('=== 扫位宽 b（n = 16 固定）===')
    L.append('   b |      eps_x |      eps_w |      eps_y |   eps_out')
    for (n, b, ex, ew, ey, eo) in rows:
        if n == 16:
            L.append('  %2d | %9.3e | %9.3e | %9.3e | %9.3e' % (b, ex, ew, ey, eo))

    L.append('')
    L.append('=== 扫输入个数 n（b = 10 固定）===')
    L.append('   n |      eps_x |      eps_w |      eps_y |   eps_out')
    for (n, b, ex, ew, ey, eo) in rows:
        if b == 10:
            L.append('  %2d | %9.3e | %9.3e | %9.3e | %9.3e' % (n, ex, ew, ey, eo))

    L.append('')
    L.append('=== 以「单层输出逆回误差」为判据的最优位宽 ===')
    for n in (4, 16, 64):
        pts = [(b, eo) for (nn, b, ex, ew, ey, eo) in rows if nn == n]
        if not pts:
            continue
        bmin, emin = min(pts, key=lambda z: z[1])
        L.append('  n=%2d : b* = %2d  (eps_out = %.3e)  明细 %s' % (
            n, bmin, emin, ', '.join('b%d:%.2e' % (b, e) for b, e in pts)))

    L.append('')
    L.append('=== 两层串联：上一层 eps_y 与这一层 eps_x 的耦合（b1=b2=10）===')
    full, onlyb, onlya = [], [], []
    for t in range(TRIALS):
        r = np.random.default_rng(100 + t)
        full.append(two_layer(10, 10, rng=r, mode='full'))
        onlyb.append(two_layer(10, 10, rng=np.random.default_rng(100 + t), mode='b_only'))
        onlya.append(two_layer(10, 10, rng=np.random.default_rng(100 + t), mode='a_only'))
    F, B, A_ = np.mean(full), np.mean(onlyb), np.mean(onlya)
    L.append('  端到端实测            %.4e' % F)
    L.append('  只留 A 的贡献(e_A)    %.4e' % A_)
    L.append('  只留 B 的贡献(e_B)    %.4e' % B)
    L.append('  平方和相加 sqrt(A^2+B^2) %.4e   比值 实测/相加 = %.2f' % (np.hypot(A_, B), F / np.hypot(A_, B)))
    L.append('  取大 max(A,B)          %.4e   比值 实测/取大 = %.2f' % (max(A_, B), F / max(A_, B)))
    L.append('  直接相加 A+B           %.4e   比值 实测/直接加 = %.2f' % (A_ + B, F / (A_ + B)))

    txt = '\n'.join(L)
    open(OUT_TXT, 'w', encoding='utf-8').write(txt)
    print(txt)

    # ---- 图：eps vs b
    W, H = 900, 560
    im = Image.new('RGB', (W, H), (20, 20, 24))
    dr = ImageDraw.Draw(im)
    dr.text((14, 10), 'random baseline: log10(eps) vs b   (m=16, W_ref=882)', fill=(235, 235, 245))
    bx0, by0, bw, bh = 70, 40, W - 150, H - 110
    dr.rectangle([bx0, by0, bx0 + bw, by0 + bh], outline=(70, 70, 80))
    allv = [max(v, 1e-12) for v in ([r[2] for r in rows] + [r[3] for r in rows] + [r[4] for r in rows] + [r[5] for r in rows])]
    lo, hi = np.floor(np.log10(min(allv))), np.ceil(np.log10(max(allv)))

    def px(b):
        return bx0 + (b - 6) / (14 - 6) * bw

    def py(v):
        return by0 + bh - (np.log10(v) - lo) / (hi - lo) * bh

    for b in bs:
        dr.line([px(b), by0, px(b), by0 + bh], fill=(40, 40, 48))
        dr.text((px(b) - 6, by0 + bh + 6), str(b), fill=(160, 160, 170))
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
    for key, col, lbl in ((2, (200, 200, 210), 'eps_x'), (4, (255, 110, 110), 'eps_y'),
                          (5, (140, 255, 140), 'eps_out')):
        pts = [(r[1], r[key]) for r in rows if r[0] == 16]
        for (a, va), (b2, vb) in zip(pts, pts[1:]):
            dr.line([px(a), py(va), px(b2), py(vb)], fill=col, width=2)
        dr.text((px(pts[-1][0]) + 4, py(pts[-1][1]) + 6), lbl, fill=col)
    dr.text((bx0, H - 26), 'coloured = eps_w (various n)   white eps_x   red eps_y   green eps_out',
            fill=(170, 170, 180))
    im.save(OUT_PNG)
    print('wrote', OUT_PNG)


if __name__ == '__main__':
    main()

#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
重训：加入平移 / 粗细 / 大小 三类增广
==========================================
增广（在 28x28 灰度上做）:
  1. 平移: 随机 dx,dy 各 -2..2 像素（零填充）
  2. 大小: 以中心为原点随机缩放 0.78 ~ 1.22
  3. 粗细: 随机选择二值化阈值；并随机做 1 次腐蚀 / 保持 / 1 次膨胀
处理链: 28x28 灰度 -> 平移 -> 缩放 -> 下采样 15x15(4x4均值) -> 二值化

模型: 线性 softmax, 输入 225 二值, 输出 10
输出:
  results/Waug_model.npz         (浮点权重)
  results/Waug_int.npz           (整数化权重 + 偏置)
  results/aug_report.json
"""
import os, json, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
SEED = 20240517
np.random.seed(SEED)

DISK1 = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def load_idx(p):
    with open(p, 'rb') as f:
        mg = struct.unpack('>I', f.read(4))[0]; nd = mg & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(nd)]
        d = np.frombuffer(f.read(), dtype=np.uint8)
    return d.reshape(dims)


def shift(img, dy, dx):
    """img: (n,28,28) float -> 平移, 零填充"""
    out = np.zeros_like(img)
    ys0, ys1 = max(0, dy), min(28, 28 + dy)
    xs0, xs1 = max(0, dx), min(28, 28 + dx)
    out[:, ys0:ys1, xs0:xs1] = img[:, ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def zoom(img, s):
    """以 13.5,13.5 为中心缩放 s 倍（最近邻）。img: (n,28,28)"""
    yy, xx = np.mgrid[0:28, 0:28].astype(np.float32)
    sy = np.clip(np.round((yy - 13.5) / s + 13.5).astype(int), 0, 27)
    sx = np.clip(np.round((xx - 13.5) / s + 13.5).astype(int), 0, 27)
    return img[:, sy.ravel(), sx.ravel()].reshape(img.shape[0], 28, 28)


def dilate1(b):
    out = b.copy()
    for dy, dx in DISK1:
        out |= np.roll(np.roll(b, dy, 1), dx, 2)
    return out


def erode1(b):
    out = b.copy()
    for dy, dx in DISK1:
        out &= np.roll(np.roll(b, dy, 1), dx, 2)
    return out


def down15(img28):
    n = img28.shape[0]
    a = img28.reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    o = np.zeros((n, 15, 15), dtype=np.float32)
    o[:, :14, :14] = a; o[:, 14, :14] = a[:, 13, :]
    o[:, :14, 14] = a[:, :, 13]; o[:, 14, 14] = a[:, 13, 13]
    return o


def augment_once(imgs28, rng):
    """一批 28x28 uint8 -> 15x15 bool"""
    z = imgs28.astype(np.float32)
    n = len(z)
    # 1) 平移
    dy = rng.integers(-2, 3); dx = rng.integers(-2, 3)
    z = shift(z, int(dy), int(dx))
    # 2) 缩放
    s = float(rng.uniform(0.78, 1.22))
    z = zoom(z, s)
    # 3) 下采样
    d = down15(z)
    # 4) 二值化阈值（相对该图自身范围）
    lo = d.reshape(n, -1).min(1); hi = d.reshape(n, -1).max(1)
    frac = rng.uniform(0.20, 0.80)
    thr = lo + (hi - lo) * frac
    b = d > thr[:, None, None]
    # 5) 粗细（形态学）
    mode = rng.integers(0, 3)
    if mode == 1:
        b = erode1(b)
    elif mode == 2:
        b = dilate1(b)
    return b


def make_dataset(imgs28, y, reps, rng):
    Xs, ys = [], []
    for _ in range(reps):
        Xs.append(augment_once(imgs28, rng))
        ys.append(y)
    X = np.concatenate(Xs, 0).reshape(-1, 225).astype(np.float32)
    Y = np.concatenate(ys, 0)
    return X, Y


def main():
    rng = np.random.default_rng(SEED)
    tr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    te = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    print(f"训练 {tr.shape}  测试 {te.shape}")

    # 基线的原始表示
    base_thr = down15(tr.astype(np.float32)).mean()
    B0te = (down15(te.astype(np.float32)) > base_thr).reshape(len(te), -1)

    # ---- 构造增广训练集 ----
    REPS = 6
    print(f"\n生成增广训练集（{REPS} 份/张）...")
    Xtr, Ytr = make_dataset(tr, ytr, REPS, rng)
    print(f"  增广后 {Xtr.shape}  平均点亮 {Xtr.sum(1).mean():.1f}")
    # 也加入原样数据（保底）
    Xori = (down15(tr.astype(np.float32)) > base_thr).reshape(len(tr), -1).astype(np.float32)
    Xtr = np.concatenate([Xtr, Xori], 0)
    Ytr = np.concatenate([Ytr, ytr], 0)
    print(f"  加上原样后 {Xtr.shape}")

    # ---- 训练线性 softmax（numpy 手写 Adam）----
    rng2 = np.random.default_rng(SEED)
    Wf = (rng2.standard_normal((225, 10)) * 0.01)
    bf = np.zeros(10)
    mW = np.zeros_like(Wf); vW = np.zeros_like(Wf)
    mb = np.zeros_like(bf); vb = np.zeros_like(bf)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    LR = 0.05
    N = len(Xtr); BS = 512; EPOCHS = 20
    print(f"\n训练 {EPOCHS} epoch, batch {BS}, lr {LR}")
    step = 0
    for ep in range(EPOCHS):
        perm = rng2.permutation(N)
        tot = 0.0; corr = 0
        lr = LR * (0.85 ** (ep // 4))
        for i in range(0, N, BS):
            idx = perm[i:i+BS]
            xb = Xtr[idx]                       # (B,225)
            yb = Ytr[idx]
            logits = xb @ Wf + bf
            logits -= logits.max(1, keepdims=True)
            ex = np.exp(logits)
            P = ex / ex.sum(1, keepdims=True)
            B = len(idx)
            loss = -np.log(np.maximum(P[np.arange(B), yb], 1e-12)).mean()
            dlog = P.copy(); dlog[np.arange(B), yb] -= 1.0; dlog /= B
            gW = xb.T @ dlog
            gb = dlog.sum(0)
            step += 1
            for P_, G_, m_, v_ in ((Wf, gW, mW, vW), (bf, gb, mb, vb)):
                m_ *= beta1; m_ += (1 - beta1) * G_
                v_ *= beta2; v_ += (1 - beta2) * (G_ * G_)
                mh = m_ / (1 - beta1 ** step)
                vh = v_ / (1 - beta2 ** step)
                P_ -= lr * mh / (np.sqrt(vh) + eps)
            tot += loss * B
            corr += int((P.argmax(1) == yb).sum())
        if ep % 2 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep+1:>2}: loss {tot/N:.4f}  train_acc {corr/N:.4f}")

    # ---- 评估 ----
    def acc_on(bl, lab):
        z = bl.astype(np.float64) @ Wf + bf
        return float((z.argmax(1) == lab).mean())

    print(f"\n{'='*88}\n评估\n{'='*88}")
    print(f"  原始测试集（未增广）准确率: {acc_on(B0te, yte):.4f}")

    print(f"\n  增广鲁棒性测试（每种 4000 张测试样本，固定增广）:")
    n_sub = 4000
    print(f"  {'条件':<34} | {'平均点亮':>8} | {'准确率':>8}")
    print("  " + "-" * 60)
    robust = {}
    rng2 = np.random.default_rng(999)
    # 平移
    for dy, dx, nm in [(2, 0, '下移2'), (-2, 0, '上移2'),
                       (0, 2, '右移2'), (0, -2, '左移2'),
                       (2, -2, '右下2')]:
        z = shift(te[:n_sub].astype(np.float32), dy, dx)
        b = (down15(z) > base_thr).reshape(n_sub, -1)
        a = acc_on(b, yte[:n_sub])
        robust[f'shift_{nm}'] = a
        print(f"  {'平移 '+nm:<34} | {b.sum(1).mean():>8.1f} | {a:>8.4f}")
    # 大小
    for s in (0.8, 0.9, 1.1, 1.2):
        z = zoom(te[:n_sub].astype(np.float32), s)
        b = (down15(z) > base_thr).reshape(n_sub, -1)
        a = acc_on(b, yte[:n_sub])
        robust[f'zoom_{s}'] = a
        print(f"  {'缩放 x'+str(s):<34} | {b.sum(1).mean():>8.1f} | {a:>8.4f}")
    # 粗细
    b0 = (down15(te[:n_sub].astype(np.float32)) > base_thr)
    for nm, bb in [('腐蚀1(更细)', erode1(b0)), ('原样', b0),
                   ('膨胀1(更粗)', dilate1(b0)),
                   ('膨胀2', dilate1(dilate1(b0))),
                   ('膨胀3', dilate1(dilate1(dilate1(b0))))]:
        a = acc_on(bb.reshape(n_sub, -1), yte[:n_sub])
        robust[f'thick_{nm}'] = a
        print(f"  {'粗细 '+nm:<34} | {bb.reshape(n_sub,-1).sum(1).mean():>8.1f} | "
              f"{a:>8.4f}")
    # 阈值粗细
    d15 = down15(te[:n_sub].astype(np.float32))
    for f in (0.3, 0.5, 0.7):
        lo = d15.reshape(n_sub, -1).min(1); hi = d15.reshape(n_sub, -1).max(1)
        b = d15 > (lo + (hi - lo) * f)[:, None, None]
        a = acc_on(b.reshape(n_sub, -1), yte[:n_sub])
        robust[f'thr_{f}'] = a
        print(f"  {'阈值比例 '+str(f):<34} | {b.reshape(n_sub,-1).sum(1).mean():>8.1f} | "
              f"{a:>8.4f}")

    # ---- 整数化 ----
    print(f"\n{'='*88}\n整数化\n{'='*88}")
    qmax = 256
    Wi = np.clip(np.round(Wf * qmax), -32768, 32767).astype(int)
    nz = Wi[Wi != 0]
    print(f"  qmax={qmax}: W 范围 [{Wi.min()}, {Wi.max()}], "
          f"最大绝对值 {np.abs(Wi).max()}, 非零 {len(nz)}/2250")

    # 选择最小 qmax 使准确率不明显下降
    best = None
    for qm in [16, 24, 32, 48, 64, 96, 128, 192, 256]:
        Wq = np.round(Wf * qm)
        Wq[Wq == 0] = 0
        # 用浮点 bias，只看权重影响
        z = B0te.astype(np.float64) @ (Wq / qm) + bf
        a = float((z.argmax(1) == yte).mean())
        print(f"    qmax={qm:>4}: 原始测试准确率 {a:.4f}  "
              f"W范围[{int(Wq.min())},{int(Wq.max())}]")
        if best is None or a >= best[1] - 1e-6:
            best = (qm, a, Wq)
    print(f"  选用 qmax={best[0]}, 准确率 {best[1]:.4f}")

    Wi = best[2].astype(int)
    # 偏置也整数化（同尺度）
    bi = np.round(bf * best[0]).astype(int)
    print(f"  bias 整数化: {bi.tolist()}")
    print(f"  W 整数范围 [{Wi.min()}, {Wi.max()}]")
    np.savez(os.path.join(RES, 'Waug_model.npz'), W=Wf, b=bf)
    np.savez(os.path.join(RES, 'Waug_int.npz'), W=Wi, b=bi)

    json.dump({'qmax': int(best[0]),
               'acc_original': acc_on(B0te, yte),
               'robust': robust,
               'W_int_range': [int(Wi.min()), int(Wi.max())],
               'bias_int': bi.tolist()},
              open(os.path.join(RES, 'aug_report.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已存: results/Waug_model.npz / Waug_int.npz / aug_report.json")


if __name__ == '__main__':
    main()

#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
诚实评估：用训练集切出验证集，测试集完全不碰
================================================
协议:
  train   50000  (从 MNIST train 切)
  valid   10000  (从 MNIST train 切，用于选 checkpoint)
  test    10000  (MNIST t10k，最后只用一次)

对照: 之前 int_weight_model.py 的协议（train 60000, 每 1000 步在 test 上存档）
"""
import os, json, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
LO, HI = -12, 7


def load_idx(p):
    with open(p, 'rb') as f:
        mg = struct.unpack('>I', f.read(4))[0]; nd = mg & 0xFF
        dims = [struct.unpack('>I', f.read(4))[0] for _ in range(nd)]
        d = np.frombuffer(f.read(), dtype=np.uint8)
    return d.reshape(dims)


def to15(im):
    n = len(im)
    a = im.reshape(n, 28, 28).astype(np.float32).reshape(n, 14, 2, 14, 2).mean(axis=(2, 4))
    o = np.zeros((n, 15, 15), dtype=np.float32)
    o[:, :14, :14] = a; o[:, 14, :14] = a[:, 13, :]
    o[:, :14, 14] = a[:, :, 13]; o[:, 14, 14] = a[:, 13, 13]
    return o


def train_int(Ftr, ytr, Fsel, ysel, steps=6000, lr=0.05, bs=128, seed=0):
    d_in = Ftr.shape[1]; n = len(Ftr)
    rng = np.random.default_rng(seed)
    W = np.zeros((d_in, 10)); b = np.zeros(10)
    M = [np.zeros_like(W), np.zeros_like(b)]
    V = [np.zeros_like(W), np.zeros_like(b)]
    best = (0.0, None, None)
    for t in range(1, steps + 1):
        Wq = np.clip(np.round(W), LO, HI); bq = np.clip(np.round(b), LO, HI)
        idx = rng.integers(0, n, bs)
        Fb, yb = Ftr[idx], ytr[idx]
        Z = Fb @ Wq + bq
        Z -= Z.max(1, keepdims=True)
        e = np.exp(Z); P = e / e.sum(1, keepdims=True)
        dZ = P.copy(); dZ[np.arange(bs), yb] -= 1; dZ /= bs
        g = [Fb.T @ dZ, dZ.sum(0)]
        for i, (pp, gg) in enumerate(zip([W, b], g)):
            M[i] = 0.9 * M[i] + 0.1 * gg
            V[i] = 0.999 * V[i] + 0.001 * gg ** 2
            pp -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == steps:
            acc = float(((Fsel @ Wq + bq).argmax(1) == ysel).mean())
            if acc > best[0]:
                best = (acc, Wq.copy(), bq.copy())
    return best


def main():
    Xtr_all = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr_all = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    A = to15(Xtr_all); B16 = to15(Xte); thr = A.mean()
    Ftr_all = (A > thr).astype(np.float64).reshape(len(A), -1)
    Fte = (B16 > thr).astype(np.float64).reshape(len(B16), -1)

    print("=" * 90)
    print("诚实评估：验证集选自训练集，测试集只用一次")
    print("=" * 90)

    rng = np.random.default_rng(2024)
    perm = rng.permutation(len(Ftr_all))
    tr_i, va_i = perm[:50000], perm[50000:]
    Ftr, ytr = Ftr_all[tr_i], ytr_all[tr_i]
    Fva, yva = Ftr_all[va_i], ytr_all[va_i]
    print(f"\n  train {Ftr.shape}  valid {Fva.shape}  test {Fte.shape}")

    print(f"\n[A] 用验证集选 checkpoint（正确协议）")
    acc_sel, WqA, bqA = train_int(Ftr, ytr, Fva, yva)
    print(f"  验证集(用于选择)准确率  {acc_sel:.4f}")
    testA = float(((Fte @ WqA + bqA).argmax(1) == yte).mean())
    print(f"  测试集准确率（只评一次）{testA:.4f}")
    # 训练集表现
    trA = float(((Ftr @ WqA + bqA).argmax(1) == ytr).mean())
    print(f"  训练集准确率           {trA:.4f}")
    print(f"  -> 泛化间隙 (train - test) = {trA - testA:+.4f}")

    print(f"\n[B] 用测试集选 checkpoint（之前 int_weight_model.py 的协议）")
    acc_sel2, WqB, bqB = train_int(Ftr, ytr, Fte, yte)
    print(f"  测试集(被用于选择)准确率 {acc_sel2:.4f}")
    print(f"  -> 两个协议的差别 = {acc_sel2 - testA:+.4f}  "
          f"(这就是'偷看测试集'带来的乐观偏差)")

    print(f"\n[C] 交叉验证：5 折，每折在训练集上训、在留出折上评")
    folds = np.array_split(rng.permutation(len(Ftr_all)), 5)
    cvs = []
    for i in range(5):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(5) if j != i])
        _, Wq, bq = train_int(Ftr_all[tr], ytr_all[tr], Ftr_all[va], ytr_all[va],
                              steps=4000)
        a = float(((Ftr_all[va] @ Wq + bq).argmax(1) == ytr_all[va]).mean())
        cvs.append(a)
        print(f"  折 {i+1}: {a:.4f}")
    print(f"  5 折平均 {np.mean(cvs):.4f}  (标准差 {np.std(cvs):.4f})")

    out = {'protocol_A': {'valid_sel': acc_sel, 'test': testA, 'train': trA,
                          'gap': trA - testA},
           'protocol_B': {'test_sel': acc_sel2,
                          'optimism': acc_sel2 - testA},
           'cv5': {'mean': float(np.mean(cvs)), 'std': float(np.std(cvs)),
                   'folds': [float(x) for x in cvs]},
           'full_model_acc': 0.8738}
    json.dump(out, open(os.path.join(RES, 'honest_eval.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已存: results/honest_eval.json")


if __name__ == '__main__':
    main()

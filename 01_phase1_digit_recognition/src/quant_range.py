#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
225 -> 10 线性分类器：权重整数化到 [-882, 882] 后准确率还剩多少
=================================================================
约束来源（docs/05、07）:
  单容器 b ∈ [0,897]；单步长 |Δλ| ≤ 868；输入 x ∈ {0..15}
  强模 LUT 里"权重 = 阶梯步长"，所以可达的整数权重约在 [-868,+15]
  双侧各自满档 15×897 = 13455，所以 ±882 的量级完全可以实现
  真正的限制是: **必须是整数**

测三件事:
  A. 训练浮点 -> 直接四舍五入取整（后处理）
  B. 边训练边量化（QAT，直通估计）
  C. 不同整数范围（±8, ±16, ±64, ±882）的对比
另外: 偏置也用同样范围约束测一次
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
RES = os.path.join(ROOT, "results")


def load_idx(p):
    import struct
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


def train(Ftr, ytr, Fte, yte, steps=4000, lr=0.05, bs=128, seed=0,
          quant=None, reward_scale=None, qat=False):
    """
    quant: None 或 (lo, hi) —— 若给出，权重被量化到该整数范围
    qat:   True 时用直通估计（前向量化，反向用原始梯度）
    """
    d_in = Ftr.shape[1]; n = len(Ftr)
    rng = np.random.default_rng(seed)
    W = np.zeros((d_in, 10)); b = np.zeros(10)
    M = [np.zeros_like(W), np.zeros_like(b)]; V = [np.zeros_like(W), np.zeros_like(b)]

    def quantize(P):
        if quant is None:
            return P
        lo, hi = quant
        return np.clip(np.round(P), lo, hi)

    best_float, best_q = 0.0, 0.0
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        Fb, yb = Ftr[idx], ytr[idx]
        Wq, bq = quantize(W), quantize(b)
        Z = Fb @ Wq + bq
        mx = Z.max(1, keepdims=True)
        e = np.exp(Z - mx); P = e / e.sum(1, keepdims=True)
        dZ = P.copy(); dZ[np.arange(bs), yb] -= 1; dZ /= bs
        g = [Fb.T @ dZ, dZ.sum(0)]          # STE: 量化处梯度直通
        for i, (pp, gg) in enumerate(zip([W, b], g)):
            M[i] = 0.9 * M[i] + 0.1 * gg; V[i] = 0.999 * V[i] + 0.001 * gg ** 2
            pp -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == steps:
            accf = float(((Fte @ W + b).argmax(1) == yte).mean())
            Wq2, bq2 = quantize(W), quantize(b)
            accq = float(((Fte @ Wq2 + bq2).argmax(1) == yte).mean())
            best_float = max(best_float, accf); best_q = max(best_q, accq)
    Wq, bq = quantize(W), quantize(b)
    return best_float, best_q, W, b, Wq, bq


def main():
    print("=" * 92)
    print("权重整数化到 [-882, 882] 后的准确率")
    print("=" * 92)
    Xtr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    A = to15(Xtr); B16 = to15(Xte); thr = A.mean()
    Ftr = (A > thr).astype(np.float64).reshape(len(A), -1)
    Fte = (B16 > thr).astype(np.float64).reshape(len(B16), -1)
    print(f"数据 train {Ftr.shape}  test {Fte.shape}\n")

    # ---------- 基准: 浮点 ----------
    print("[基准] 浮点权重")
    accf, _, W, b, _, _ = train(Ftr, ytr, Fte, yte, quant=None, seed=0)
    print(f"  浮点准确率 {accf:.4f}")
    print(f"  权重范围 [{W.min():.3f}, {W.max():.3f}]   绝对最大值 {np.abs(W).max():.3f}")

    # ---------- A. 事后取整，不同范围 ----------
    print(f"\n[A] 训练浮点 -> 事后四舍五入取整（不重训）")
    print(f"  {'范围':>12} | {'截断比例':>8} | {'取整后准确率':>12} | {'掉的点':>8}")
    print("  " + "-" * 56)
    resA = {}
    for lo, hi in [(-882, 882), (-64, 64), (-16, 16), (-8, 8), (-4, 4), (-2, 2)]:
        Wq = np.clip(np.round(W), lo, hi); bq = np.clip(np.round(b), lo, hi)
        acc = float(((Fte @ Wq + bq).argmax(1) == yte).mean())
        clipped = float(((W < lo) | (W > hi)).mean())
        resA[(lo, hi)] = acc
        print(f"  [{lo:>5},{hi:>5}] | {clipped:>7.2%} | {acc:>12.4f} | "
              f"{(acc-accf)*100:>+7.2f}")
    print(f"  注: ±882 时截断比例 {(np.abs(W)>882).mean():.2%}（几乎不截断）")

    # ---------- B. QAT（边训练边量化） ----------
    print(f"\n[B] QAT: 边训练边量化到整数（直通估计）")
    print(f"  {'范围':>12} | {'QAT准确率':>10} | {'事后取整':>9} | {'QAT提升':>8}")
    print("  " + "-" * 50)
    resB = {}
    for lo, hi in [(-882, 882), (-16, 16), (-8, 8), (-4, 4), (-2, 2)]:
        _, accq, _, _, Wq, bq = train(Ftr, ytr, Fte, yte, quant=(lo, hi), seed=0)
        a_post = resA.get((lo, hi), float('nan'))
        resB[(lo, hi)] = accq
        print(f"  [{lo:>5},{hi:>5}] | {accq:>10.4f} | {a_post:>9.4f} | "
              f"{(accq-a_post)*100:>+7.2f}")
        if (lo, hi) == (-882, 882):
            nz = (Wq != 0).sum()
            print(f"       整数权重统计: 非零 {nz}/{Wq.size}  范围 "
                  f"[{Wq.min():.0f},{Wq.max():.0f}]  偏置范围 [{bq.min():.0f},{bq.max():.0f}]")

    # ---------- C. 结论 ----------
    print(f"\n{'='*92}")
    print("### 结论")
    print(f"  浮点                          {accf:.4f}")
    print(f"  取整到 [-882,882]（事后）       {resA[(-882,882)]:.4f}")
    print(f"  取整到 [-882,882]（QAT）        {resB[(-882,882)]:.4f}")
    print(f"\n  作为对比，±882 相对浮点权的最大绝对值 {np.abs(W).max():.1f}：")
    print(f"    权重绝对值分布: "
          + "  ".join(f"P{q}={np.percentile(np.abs(W),q):.2f}" for q in (50, 90, 99, 100)))
    print(f"    需要用满 ±882 的权重比例: {(np.abs(W)>100).mean():.4%}")
    print(f"{'='*92}")
    json.dump({'float': accf, 'post': {f'{k[0]}_{k[1]}': v for k, v in resA.items()},
               'qat': {f'{k[0]}_{k[1]}': v for k, v in resB.items()}},
              open(os.path.join(RES, 'quant_range_analysis.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()

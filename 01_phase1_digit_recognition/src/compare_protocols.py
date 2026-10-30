#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
两种划分下的权重对比
======================
旧协议: train 60000, test 10000（test 兼任验证 -> 现已写进数据包）
新协议: train 50000, valid 10000(从 train 切), test 10000

比较:
  1. 准确率
  2. 权重差异（逐元素）: 相同个数、平均绝对差、最大绝对差、相关系数
  3. 若把新权重写进数据包，容器值会变多少
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

    Wold = np.load(os.path.join(RES, 'Wint_model.npz'))['W'].astype(int)
    bold = np.load(os.path.join(RES, 'Wint_model.npz'))['b'].astype(int)
    acc_old = float(((Fte @ Wold + bold).argmax(1) == yte).mean())

    print("=" * 96)
    print("两种数据划分下的权重对比")
    print("=" * 96)
    print(f"\n  旧协议: train 60000, test 10000（test 兼任验证）")
    print(f"    -> 数据包里的权重，测试集准确率 {acc_old:.4f}")

    # 新协议
    rng = np.random.default_rng(2024)
    perm = rng.permutation(len(Ftr_all))
    tr_i, va_i = perm[:50000], perm[50000:]
    print(f"\n  新协议: train 50000, valid 10000, test 10000")
    accs, Ws, bs_ = [], [], []
    for seed in (0, 1, 2):
        a_sel, Wq, bq = train_int(Ftr_all[tr_i], ytr_all[tr_i],
                                  Ftr_all[va_i], ytr_all[va_i], seed=seed)
        at = float(((Fte @ Wq + bq).argmax(1) == yte).mean())
        av = float(((Ftr_all[va_i] @ Wq + bq).argmax(1) == ytr_all[va_i]).mean())
        atr = float(((Ftr_all[tr_i] @ Wq + bq).argmax(1) == ytr_all[tr_i]).mean())
        accs.append(at); Ws.append(Wq); bs_.append(bq)
        print(f"    seed {seed}: valid {av:.4f}  train {atr:.4f}  test {at:.4f}")

    Wnew = Ws[0]; bnew = bs_[0]

    print(f"\n{'='*96}\n权重逐元素对比（旧 vs 新 seed0）\n{'='*96}")
    diff = Wnew - Wold
    same = int((diff == 0).sum())
    total = diff.size
    print(f"  完全相同元素     {same} / {total}  ({same/total*100:.1f}%)")
    print(f"  平均绝对差       {np.abs(diff).mean():.4f}")
    print(f"  最大绝对差       {np.abs(diff).max()}")
    print(f"  相关系数         {np.corrcoef(Wold.ravel(), Wnew.ravel())[0,1]:.6f}")
    print(f"  差异分布:")
    vals, cnt = np.unique(diff, return_counts=True)
    for v, c in zip(vals, cnt):
        if abs(v) <= 3 or c > 20:
            print(f"    Δ={v:>+3}: {c:>5} ({c/total*100:>5.1f}%)")
    nz_old = int((Wold != 0).sum()); nz_new = int((Wnew != 0).sum())
    print(f"  非零个数         旧 {nz_old}  新 {nz_new}")
    print(f"  权重范围         旧 [{Wold.min()},{Wold.max()}]  "
          f"新 [{Wnew.min()},{Wnew.max()}]")
    print(f"  旧非零但新为0    {int(((Wold!=0)&(Wnew==0)).sum())}")
    print(f"  旧0但新非零      {int(((Wold==0)&(Wnew!=0)).sum())}")

    # 容器值差异
    print(f"\n  容器值差异（偶数层 c=15-W）:")
    idx = [k for k in range(10) if (k+1) % 2 == 0]
    for k in idx[:3]:
        co = 15 - Wold[:, k]; cn = 15 - Wnew[:, k]
        print(f"    数字{k}: 旧范围[{co.min()},{co.max()}] "
              f"新范围[{cn.min()},{cn.max()}]  平均差 {np.abs(cn-co).mean():.2f}")

    # 新旧权重的预测一致性
    Po = (Fte @ Wold + bold).argmax(1)
    Pn = (Fte @ Wnew + bnew).argmax(1)
    print(f"\n  预测一致性: {(Po==Pn).mean()*100:.2f}% 的测试样本给出相同答案")
    print(f"  两者都对的样本: {int(((Po==yte)&(Pn==yte)).sum())}")
    print(f"  只有旧的判对: {int(((Po==yte)&(Pn!=yte)).sum())}")
    print(f"  只有新的判对: {int(((Pn==yte)&(Po!=yte)).sum())}")

    # 多种子稳定性（新协议）
    print(f"\n  新协议的种子间稳定性:")
    for i in range(3):
        for j in range(i+1, 3):
            d = np.abs(Ws[i]-Ws[j])
            print(f"    seed{i} vs seed{j}: 相同 {int((d==0).sum())}/{total} "
                  f"({(d==0).mean()*100:.1f}%)  平均差 {d.mean():.3f}")

    json.dump({'acc_old': acc_old, 'acc_new_seeds': accs,
               'same_frac': same/total, 'mad': float(np.abs(diff).mean()),
               'max_abs_diff': int(np.abs(diff).max()),
               'corr': float(np.corrcoef(Wold.ravel(), Wnew.ravel())[0,1]),
               'pred_agreement': float((Po==Pn).mean())},
              open(os.path.join(RES, 'weight_protocol_compare.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n已存: results/weight_protocol_compare.json")


if __name__ == '__main__':
    main()

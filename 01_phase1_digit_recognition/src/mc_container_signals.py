#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
MC 容器信号分配表：10 层 x 225 容器
=====================================
架构（用户给定）:
  10 层，第 k 层对应数字 k（第1层=0, 第2层=1, ...）
  每层: 总线 F = F_0 - G_1 - ... - G_8
       支线 G_i = G_{i0} - c_{i1} - ... - c_{in}
       G_1: n=15（最后一行, 15 像素）
       G_2..G_8: n=30（各两行）
  奇数层: 像素点亮 -> 取消截断（比较器输出15）
  偶数层: 像素点亮 -> 截断（比较器输出15）

推导出的信号代数:
  有效权重 = 15 x 容器里存的权重
  G_{i0} = n * b
  F_0 = 225 * b
  约束: F_0 <= 13455 (总线满档) -> b <= 59.8 -> max|W| <= 3.99

两套方案:
  甲: 权重 / 3（保留 [-12,7] 的信息，容器值可能是小数 -> 取整）
  乙: 重训权重到 [-3,3]（完全整数）

输出:
  results/mc_signal_<方案>.csv     每层 225 个容器的信号值
  results/mc_bias_<方案>.csv       每层的 F_0 / G_{10}..G_{80}
"""
import os, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = r'D:\czm我的世界\文件\Comparator Network\tiny_cnn\data\MNIST\raw'
RES = os.path.join(ROOT, "results")

# 像素分组: G_1 = 第14行(0-based, 15像素); G_2..G_8 = 两行各30像素
GP = []                                   # [(起始行, 行数)]
GP.append((14, 1))
for r in range(12, -1, -2):               # 12-13, 10-11, ..., 0-1
    GP.append((r, 2))
assert len(GP) == 8 and sum(rn * 15 for _, rn in GP) == 225, GP


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


def train_int(Ftr, ytr, Fte, yte, lo, hi, steps=6000, lr=0.05, bs=128, seed=0):
    d_in = Ftr.shape[1]; n = len(Ftr)
    rng = np.random.default_rng(seed)
    W = np.zeros((d_in, 10)); b = np.zeros(10)
    M = [np.zeros_like(W), np.zeros_like(b)]; V = [np.zeros_like(W), np.zeros_like(b)]
    best = (0.0, None, None)
    for t in range(1, steps + 1):
        Wq = np.clip(np.round(W), lo, hi); bq = np.clip(np.round(b), lo, hi)
        idx = rng.integers(0, n, bs)
        Fb, yb = Ftr[idx], ytr[idx]
        Z = Fb @ Wq + bq
        mx = Z.max(1, keepdims=True)
        e = np.exp(Z - mx); P = e / e.sum(1, keepdims=True)
        dZ = P.copy(); dZ[np.arange(bs), yb] -= 1; dZ /= bs
        for i, (pp, gg) in enumerate(zip([W, b], [Fb.T @ dZ, dZ.sum(0)])):
            M[i] = 0.9 * M[i] + 0.1 * gg; V[i] = 0.999 * V[i] + 0.001 * gg ** 2
            pp -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)
        if t % 1000 == 0 or t == steps:
            acc = float(((Fte @ Wq + bq).argmax(1) == yte).mean())
            if acc > best[0]:
                best = (acc, Wq.copy(), bq.copy())
    return best


def build_tables(W, bias, scale, tag, outdir=RES):
    """
    由整数权重 W (225,10) 与偏置 bias (10,)
    生成每层的容器信号 c_ij 与偏置 F_0 / G_i0
    scale: 权重除以几（用于满足 b <= 59.8 的约束）
    """
    d_in = W.shape[0]
    Ws = W / scale
    bmax = np.abs(Ws).max()
    # 取 b = ceil(15 * bmax)，并保证 225*b <= 13455
    b = int(np.ceil(15 * bmax))
    b = min(b, int(13455 // 225))
    rows = []
    biases = []
    for k in range(10):
        parity = '奇数' if (k + 1) % 2 == 1 else '偶数'   # 第 k+1 层
        F0 = 225 * b
        g0s = []
        for gi, (r0, nr) in enumerate(GP):
            n = nr * 15
            g0s.append(n * b)
            for rr in range(r0, r0 + nr):
                for c in range(15):
                    p = rr * 15 + c
                    w = Ws[p, k]
                    # 奇数层: c = b + 15w ; 偶数层: c = b - 15w
                    cv = b + 15 * w if parity == '奇数' else b - 15 * w
                    rows.append({'layer': k + 1, 'digit': k, 'parity': parity,
                                 'G': gi + 1, 'row': rr, 'col': c,
                                 'pixel_index': p, 'weight_scaled': round(float(w), 4),
                                 'container_signal': round(float(cv), 4)})
        biases.append({'layer': k + 1, 'digit': k, 'parity': parity,
                       'b_base': b, 'F0_bus': F0,
                       **{f'G{i+1}0': g0s[i] for i in range(8)},
                       'F0_check': sum(g0s)})
    with open(os.path.join(outdir, f'mc_signal_{tag}.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(outdir, f'mc_bias_{tag}.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(biases[0].keys()))
        w.writeheader(); w.writerows(biases)
    cv = np.array([r['container_signal'] for r in rows])
    return b, cv, biases


def main():
    print("=" * 94)
    print("MC 容器信号分配表")
    print("=" * 94)
    Xtr = load_idx(os.path.join(RAW, 'train-images-idx3-ubyte'))
    ytr = load_idx(os.path.join(RAW, 'train-labels-idx1-ubyte'))
    Xte = load_idx(os.path.join(RAW, 't10k-images-idx3-ubyte'))
    yte = load_idx(os.path.join(RAW, 't10k-labels-idx1-ubyte'))
    A = to15(Xtr); B16 = to15(Xte); thr = A.mean()
    Ftr = (A > thr).astype(np.float64).reshape(len(A), -1)
    Fte = (B16 > thr).astype(np.float64).reshape(len(B16), -1)

    print("\n[像素分组]")
    for gi, (r0, nr) in enumerate(GP):
        print(f"  G_{gi+1}: 第 {r0}~{r0+nr-1} 行, {nr*15} 像素")

    # 载入已训练的整数权重
    z = np.load(os.path.join(RES, 'Wint_model.npz'))
    W0, b0 = z['W'], z['b']
    print(f"\n[载入已训练整数权重] 范围 [{W0.min()},{W0.max()}]  准确率 {float(z['acc']):.4f}")
    print(f"  max|W| = {np.abs(W0).max()}")

    # ---------- 方案甲: 缩放 ----------
    print(f"\n{'#'*94}\n### 方案甲: 权重 / 3（保留 [-12,7] 的信息）\n{'#'*94}")
    print(f"  约束检查: max|W|/3 = {np.abs(W0).max()/3:.2f}  ->  需要 b >= "
          f"{np.ceil(15*np.abs(W0).max()/3):.0f}，而 b <= 59.8")
    if np.abs(W0).max() / 3 > 3.99:
        print(f"  ** 仍超限！需除以 {np.ceil(np.abs(W0).max()/3.99):.0f} **")
    sc = int(np.ceil(np.abs(W0).max() / 3.99))
    b_a, cv_a, bias_a = build_tables(W0, b0, sc, 'A')
    print(f"  缩放 {sc}: b = {b_a}   F_0 = {225*b_a}  (上限 13455)")
    print(f"  容器信号范围 [{cv_a.min():.1f}, {cv_a.max():.1f}]  (须落在 [0,897])")
    print(f"  超限容器数: {(cv_a<0).sum() + (cv_a>897).sum()}")

    # ---------- 方案乙: 重训到 [-3,3] ----------
    print(f"\n{'#'*94}\n### 方案乙: 重训权重到 [-3,3]（完全整数，无精度损失）\n{'#'*94}")
    acc_w, W1, b1 = train_int(Ftr, ytr, Fte, yte, -3, 3)
    print(f"  准确率 {acc_w:.4f}  权重范围 [{W1.min()},{W1.max()}]  "
          f"非零 {(W1!=0).sum()}/{W1.size}")
    b_b, cv_b, bias_b = build_tables(W1, b1, 1, 'B')
    print(f"  b = {b_b}   F_0 = {225*b_b}  (上限 13455)")
    print(f"  容器信号范围 [{cv_b.min():.1f}, {cv_b.max():.1f}]  (须落在 [0,897])")
    print(f"  超限容器数: {(cv_b<0).sum() + (cv_b>897).sum()}")

    # ---------- 汇总 ----------
    print(f"\n{'='*94}")
    print("### 汇总")
    print(f"  {'方案':<26} | {'准确率':>7} | {'b':>4} | {'F_0':>6} | "
          f"{'容器范围':>16} | {'超限':>5}")
    print("  " + "-" * 82)
    print(f"  {'甲: 权重/'+str(sc)+'（'+str(int(np.abs(W0).max()))+'→'+str(round(np.abs(W0).max()/sc,1))+'）':<26} | "
          f"{float(z['acc']):>7.4f} | {b_a:>4} | {225*b_a:>6} | "
          f"[{cv_a.min():>6.1f},{cv_a.max():>6.1f}] | {(cv_a<0).sum()+(cv_a>897).sum():>5}")
    print(f"  {'乙: 重训到 [-3,3]':<26} | {acc_w:>7.4f} | {b_b:>4} | {225*b_b:>6} | "
          f"[{cv_b.min():>6.1f},{cv_b.max():>6.1f}] | {(cv_b<0).sum()+(cv_b>897).sum():>5}")
    print(f"  物理上限: 容器 [0,897]   总线 F_0 <= 13455")

    # 各支线偏置表
    for tag, bl in (('A', bias_a), ('B', bias_b)):
        print(f"\n[方案{tag} 各层偏置]")
        print(f"  {'层':>3}(数字) | {'奇偶':>4} | {'b':>4} | {'F_0':>6} | "
              + " | ".join(f'G{i+1}0' for i in range(8)))
        print("  " + "-" * 92)
        for x in bl[:3] + bl[-1:]:
            print(f"  {x['layer']:>3}({x['digit']}) | {x['parity']:>4} | {x['b_base']:>4} | "
                  f"{x['F0_bus']:>6} | " + " | ".join(
                      f"{x[f'G{i+1}0']:>5}" for i in range(8)))
        print(f"  （共 10 层，中间层偏置相同，完整表见 CSV）")

    json.dump({'schemeA': {'scale': sc, 'b': b_a, 'F0': 225*b_a, 'acc': float(z['acc']),
                           'cv_range': [float(cv_a.min()), float(cv_a.max())]},
               'schemeB': {'scale': 1, 'b': b_b, 'F0': 225*b_b, 'acc': acc_w,
                           'cv_range': [float(cv_b.min()), float(cv_b.max())]},
               'groups': [{'G': i+1, 'row0': r0, 'nrows': nr} for i, (r0, nr) in enumerate(GP)]},
              open(os.path.join(RES, 'mc_signal_summary.json'), 'w'), indent=2)
    print(f"\n已写出: results/mc_signal_A.csv  mc_bias_A.csv")
    print(f"        results/mc_signal_B.csv  mc_bias_B.csv")
    print(f"        results/mc_signal_summary.json")


if __name__ == '__main__':
    main()

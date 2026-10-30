#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C 方案：离散混合模型直接采样生成 8x8 头像
============================================
模型: p(x) = sum_k pi_k * prod_i Cat(x_i | theta_ki)
  每个成分 = 一个"头像原型"
  theta_ki : 第 k 个成分下第 i 个像素的颜色分布 (16 维类别分布)

训练: EM（极稳，无对抗）
采样: 抽成分 -> 逐像素独立抽颜色

评估:
  - 训练集对数似然
  - 留出集对数似然（看是否过拟合）
  - 生成样本的独立颜色数分布（与真实分布对比）
"""
import os, json, numpy as np, time
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
C = 16


def em(X, K, iters=80, seed=0, verbose=True):
    """
    X: (n,64) 整数 0..C-1
    返回 pi, theta(未归一化的 log 概率的 norm 形式), ll
    """
    n, D = X.shape
    rng = np.random.default_rng(seed)
    # 初始化: 随机挑 K 个样本作为原型中心 + 拉普拉斯平滑
    pi = np.full(K, 1.0 / K)
    theta = np.zeros((K, D, C))
    idx = rng.choice(n, K, replace=False) if n >= K else rng.choice(n, K)
    for k in range(K):
        for i in range(D):
            theta[k, i, X[idx[k], i]] += 0.6
        theta[k] += 0.4 / C
    theta /= theta.sum(2, keepdims=True)

    ll_hist = []
    for it in range(iters):
        # ---- E ---- (向量化)
        # theta: (K,D,C); X: (n,D) -> 取每个样本每个像素在每成分下的概率
        # logp[n,k] = log pi_k + sum_i log theta[k,i,X[n,i]]
        # 用 theta[:, arange(D), X] : (K, n, D)
        tt = theta[:, np.arange(D)[None, :], X]          # (K, n, D)
        logp = np.log(tt + 1e-300).sum(2).T + np.log(pi + 1e-300)[None, :]   # (n,K)
        mx = logp.max(1, keepdims=True)
        e = np.exp(logp - mx)
        ssum = e.sum(1, keepdims=True)
        w = e / ssum
        ll = float((mx[:, 0] + np.log(ssum[:, 0])).mean())
        ll_hist.append(ll)
        # ---- M ---- (向量化)
        Nk = w.sum(0) + 1e-10
        pi = Nk / n
        theta = np.zeros((K, D, C))
        for c in range(C):
            # (n,D) 指示 X==c, 加权求和对 n
            theta[:, :, c] = np.einsum('nk,nd->kd', w, (X == c).astype(np.float64))
        theta += 1e-3
        theta /= theta.sum(2, keepdims=True)
        if verbose and (it % 10 == 0 or it == iters - 1):
            print(f"     EM 迭代 {it:3d}: 平均对数似然 = {ll:.4f}", flush=True)
    return pi, theta, ll_hist


def sample(pi, theta, n, seed=0):
    K, D, _ = theta.shape
    rng = np.random.default_rng(seed)
    ks = rng.choice(K, n, p=pi)
    out = np.zeros((n, D), dtype=np.int64)
    for j, k in enumerate(ks):
        for i in range(D):
            out[j, i] = rng.choice(C, p=theta[k, i])
    return out


def loglik(X, pi, theta):
    n, D = X.shape
    logp = np.zeros((n, len(pi)))
    for k in range(len(pi)):
        logp[:, k] = np.log(pi[k] + 1e-300) + \
            np.log(theta[k][np.arange(D)[None, :], X]).sum(1)
    mx = logp.max(1, keepdims=True)
    return float((mx[:, 0] + np.log(np.exp(logp - mx).sum(1))).mean())


def main():
    print("=" * 92)
    print("C 方案：离散混合模型生成 8x8 头像")
    print("=" * 92)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    print(f"数据: {lab.shape}  (调色板 {C} 色)")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(lab))
    ntr = int(len(lab) * 0.85)
    Xtr, Xte = lab[perm[:ntr]], lab[perm[ntr:]]
    print(f"train {Xtr.shape}  test {Xte.shape}")

    # 均匀分布的基准对数似然
    ll_unif = -np.log(C) * 64
    print(f"\n基准: 均匀分布对数似然 = {ll_unif:.1f}")

    results = {}
    for K in (50, 150, 400, 1000):
        print(f"\n{'#'*92}\n### K = {K} 个成分\n{'#'*92}")
        t0 = time.time()
        pi, theta, hist = em(Xtr, K, iters=60, seed=1)
        ll_tr = loglik(Xtr, pi, theta)
        ll_te = loglik(Xte, pi, theta)
        # 采样并统计
        S = sample(pi, theta, 2000, seed=5)
        uc_s = np.array([len(np.unique(s)) for s in S])
        uc_r = np.array([len(np.unique(s)) for s in Xtr[:2000]])
        print(f"  K={K:<5} 训练LL={ll_tr:8.3f}  测试LL={ll_te:8.3f}  "
              f"参数量={K*64*C}  用时 {time.time()-t0:.0f}s")
        print(f"         生成图颜色数 {uc_s.mean():.2f}  真实 {uc_r.mean():.2f}")
        results[K] = {'ll_train': ll_tr, 'll_test': ll_te, 'params': K * 64 * C,
                      'gen_colors': float(uc_s.mean()), 'real_colors': float(uc_r.mean()),
                      'pi': pi.tolist()}

    # 用最好或 K=1000 的模型出图
    K = 1000
    pi, theta, _ = em(Xtr, K, iters=60, seed=1)
    S = sample(pi, theta, 48, seed=123)

    def to_img(row):
        return pal[row].reshape(8, 8, 3)

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

    Image.fromarray(grid([to_img(r) for r in S])).save(
        os.path.join(RES, 'mixture_samples.png'))
    R = Xtr[rng.choice(len(Xtr), 48, replace=False)]
    Image.fromarray(grid([to_img(r) for r in R])).save(
        os.path.join(RES, 'mixture_real.png'))
    print(f"\n已保存: {RES}/mixture_samples.png  mixture_real.png")
    json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != 'pi'}
               for k, v in results.items()},
              open(os.path.join(RES, 'mixture_results.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()

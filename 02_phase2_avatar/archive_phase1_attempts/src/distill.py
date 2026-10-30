#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蒸馏混合近邻教师 -> 前馈网络（可移植红石）
=============================================
教师（混合近邻，T=5.0）: 给定输入头像 x，
  在联合空间 (s,d) 找 k=30 近邻，逐位置按距离加权采样颜色。

学生（前馈网络）:
  输入 = [x 的 64 个颜色索引(归一化) | 位置 i 的 one-hot... 用标量位置]
  隐藏 = 128 -> 256 (ReLU)
  输出 = 16 路 logits（该位置的颜色分布）
  损失 = 交叉熵（标签 = 教师在该位置实际抽到的颜色）
  这在梯度上等价于学教师的分布 p_teacher(c | x, i)

为什么不会崩塌: 监督学习，目标确定，无对抗。

用法: python distill.py d3   /   python distill.py d15
输出: results/distill_<档位>_samples.png  等
"""
import os, json, sys, time
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
C = 16
LIDX = np.array([r * 8 + c for r in range(8) for c in range(4)], dtype=int)
RIDX = np.array([r * 8 + (7 - c) for r in range(8) for c in range(4)], dtype=int)

K_NB = 30
TEMP = 5.0
H = 256
EPOCHS = int(os.environ.get('EPOCHS', '60'))
BS = 128
LR = 1e-3


def build_teacher(lab, pool_n=20000, seed=0):
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(lab), pool_n, replace=False)
    S, D = lab[sel][:, LIDX], (lab[sel][:, RIDX] - lab[sel][:, LIDX])
    return S, D, sel


def teacher_select(S, D, pi, K=K_NB, seed=0, temp=TEMP):
    """教师: 返回 (每位置的邻居索引 pick(32,), 距离 dist(K,))"""
    POOL = len(S)
    ds = np.empty(POOL, dtype=np.int32); dd = np.empty(POOL, dtype=np.int32)
    for t in range(0, POOL, 5000):
        ds[t:t+5000] = (S[t:t+5000] != S[pi]).sum(1)
        dd[t:t+5000] = (D[t:t+5000] != D[pi]).sum(1)
    dist = ds + dd
    nb = np.argpartition(dist, K)[:K]
    w = np.exp(-dist[nb] / max(temp, 1e-6)); w /= w.sum()
    cum = np.cumsum(w)
    r = np.random.default_rng(seed)
    pick = np.clip((r.random(32)[:, None] > cum[None, :]).sum(1), 0, K - 1)
    return nb, pick, dist[nb]


def compose(S, D, nb, pick, dclip):
    sv = S[nb][pick, np.arange(32)]
    dv = np.clip(D[nb][pick, np.arange(32)], -dclip, dclip)
    X = np.zeros(64, dtype=np.int64)
    X[LIDX] = sv
    X[RIDX] = np.clip(sv + dv, 0, C - 1)
    return X


def net_input(x_flat, pos):
    """x_flat:(B,64) 索引; pos:(B,) 位置 -> 输入 (B,128) = 颜色64 + 位置one-hot64"""
    xn = (x_flat.astype(np.float32) - (C - 1) / 2) / ((C - 1) / 2)
    ph = np.zeros((len(pos), 64), dtype=np.float32)
    ph[np.arange(len(pos)), pos] = 1.0
    return np.concatenate([xn, ph], 1)


def init(dims, seed=0):
    rng = np.random.default_rng(seed); P = []
    for i in range(len(dims) - 1):
        P.append(rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1])).astype(np.float32))
        P.append(np.zeros(dims[i + 1], dtype=np.float32))
    return P


def adam(P, G, M, V, t, lr):
    for i in range(len(P)):
        M[i] = 0.9 * M[i] + 0.1 * G[i]
        V[i] = 0.999 * V[i] + 0.001 * G[i] ** 2
        P[i] -= lr * (M[i] / (1 - 0.9 ** t)) / (np.sqrt(V[i] / (1 - 0.999 ** t)) + 1e-8)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'd15'
    dclip = 3 if tag == 'd3' else 15
    print("=" * 96)
    print(f"蒸馏混合近邻教师 -> 前馈网络   d∈[±{dclip}]  T={TEMP}  k={K_NB}")
    print("=" * 96)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    S, D, sel = build_teacher(lab, 20000, seed=0)
    print(f"教师原型池 {len(S)}   数据总 {len(lab)}")

    # ---- 生成/加载蒸馏数据集 ----
    cache = os.path.join(DATA, f'distill_{tag}_data.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        Xin, ypos, ycol = z['Xin'], z['ypos'], z['ycol']
        print(f"载入缓存 {cache}")
    else:
        print("生成蒸馏数据集（教师采样）...")
        t0 = time.time()
        N_IN = 1500           # 输入头像数
        N_S = 8               # 每个输入采样几次
        rng = np.random.default_rng(1)
        proto_ids = rng.choice(len(S), N_IN, replace=False)
        Xs, Ps, Cs = [], [], []
        for k, pi in enumerate(proto_ids):
            xin = compose(S, D, np.array([pi]*1), np.array([0]), dclip)  # 输入用原型本身
            xin = lab[sel[pi]].copy()
            for s in range(N_S):
                nb, pick, _ = teacher_select(S, D, pi, seed=k * 100 + s)
                X = compose(S, D, nb, pick, dclip)
                Xs.append(np.tile(xin, (64, 1)))          # (64,64)
                Ps.append(np.arange(64))
                Cs.append(X)                              # (64,) 教师实际颜色
            if (k + 1) % 300 == 0:
                print(f"   {k+1}/{N_IN}  {time.time()-t0:.0f}s", flush=True)
        Xin = np.concatenate(Xs).astype(np.int64)
        ypos = np.concatenate(Ps).astype(np.int64)
        ycol = np.concatenate(Cs).astype(np.int64)
        np.savez_compressed(cache, Xin=Xin, ypos=ypos, ycol=ycol)
        print(f"数据集: {Xin.shape}  [{time.time()-t0:.0f}s] -> {cache}")

    N = len(ycol)
    print(f"蒸馏样本数: {N}")
    # 划分
    rng = np.random.default_rng(7)
    perm = rng.permutation(N); ntr = int(N * 0.9)
    tr, te = perm[:ntr], perm[ntr:]

    P = init([128, H, C], seed=1)
    M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
    t = 0; t0 = time.time()
    Xall = net_input(Xin, ypos)            # 预计算输入

    for ep in range(1, EPOCHS + 1):
        idx = tr[rng.permutation(len(tr))]
        tl, tc, tn = 0.0, 0, 0
        for s in range(0, len(idx) - BS + 1, BS):
            ii = idx[s:s + BS]
            Xb = Xall[ii]; yb = ycol[ii]
            h = np.maximum(Xb @ P[0] + P[1], 0)
            lg = h @ P[2] + P[3]
            mx = lg.max(1, keepdims=True)
            e = np.exp(lg - mx); sm = e / e.sum(1, keepdims=True)
            tl += float((-np.log(sm[np.arange(len(ii)), yb] + 1e-12)).mean()) * len(ii)
            tc += int((lg.argmax(1) == yb).sum()); tn += len(ii)
            dl = sm.copy(); dl[np.arange(len(ii)), yb] -= 1; dl /= len(ii)
            gW2 = h.T @ dl; gb2 = dl.sum(0)
            dh = (dl @ P[2].T) * (h > 0)
            gW1 = Xb.T @ dh; gb1 = dh.sum(0)
            t += 1
            adam(P, [gW1, gb1, gW2, gb2], M, V, t, LR)

        if ep % 10 == 0 or ep == 1:
            # 验证 = 与教师分布的交叉熵 + 逐像素准确率
            vce, vc, vn = 0.0, 0, 0
            for s in range(0, len(te), 2048):
                ii = te[s:s + 2048]
                h = np.maximum(Xall[ii] @ P[0] + P[1], 0)
                lg = h @ P[2] + P[3]
                mx = lg.max(1, keepdims=True)
                e = np.exp(lg - mx); sm = e / e.sum(1, keepdims=True)
                vce += float((-np.log(sm[np.arange(len(ii)), ycol[ii]] + 1e-12)).sum())
                vc += int((lg.argmax(1) == ycol[ii]).sum()); vn += len(ii)
            # 均匀基线
            print(f"  ep {ep:3d}  训练CE {tl/max(tn,1):.4f} 准确 {tc/max(tn,1):.3f} | "
                  f"**验证CE {vce/vn:.4f} 准确 {vc/vn:.3f}**  (均匀基线 CE={np.log(C):.4f})  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    json.dump({'W1': P[0].tolist(), 'b1': P[1].tolist(), 'W2': P[2].tolist(),
               'b2': P[3].tolist(), 'palette': pal.tolist(),
               'mode': f'distill_{tag}', 'temp': TEMP, 'k': K_NB},
              open(os.path.join(RES, f'distill_{tag}_net.json'), 'w'))

    # ---- 生成: 反复前向卷积式生成 ----
    def gen(nn, seed=0, temp=1.0, gsteps=3):
        """迭代式生成: 从随机初始图开始，反复用网络刷新每个位置"""
        r = np.random.default_rng(seed)
        X = r.integers(0, C, (nn, 64))
        for step in range(gsteps):
            cur = X.copy()
            for pos in range(64):
                Xin_ = net_input(cur, np.full(nn, pos))
                h = np.maximum(Xin_ @ P[0] + P[1], 0)
                lg = (h @ P[2] + P[3]) / max(temp, 1e-6)
                mx = lg.max(1, keepdims=True)
                pr = np.exp(lg - mx); pr /= pr.sum(1, keepdims=True)
                X[:, pos] = [r.choice(C, p=pr[j]) for j in range(nn)]
        return X

    def to_img(l): return pal[l].reshape(8, 8, 3)

    SC = 8
    def grid(imgs, cols=16):
        rows_ = (len(imgs) + cols - 1) // cols
        cv = np.full((rows_ * 8, cols * 8, 3), 40, dtype=np.uint8)
        for i, im in enumerate(imgs):
            r_, c_ = divmod(i, cols)
            cv[r_*8:(r_+1)*8, c_*8:(c_+1)*8] = im
        big = np.kron(cv, np.ones((SC, SC, 1), dtype=np.uint8))
        for r_ in range(rows_ + 1):
            big[r_*8*SC:r_*8*SC+2, :] = 200
        for c_ in range(cols + 1):
            big[:, c_*8*SC:c_*8*SC+2] = 200
        return big

    for temp, nm in ((0.7, 'T07'), (1.0, 'T10'), (1.5, 'T15')):
        G = gen(32, seed=123, temp=temp, gsteps=3)
        Image.fromarray(grid([to_img(v) for v in G])).save(
            os.path.join(RES, f'distill_{tag}_{nm}.png'))
        uc = np.mean([len(np.unique(v)) for v in G])
        print(f"  {nm}: 颜色数 {uc:.2f} -> distill_{tag}_{nm}.png", flush=True)
    # 真实对照
    r2 = np.random.default_rng(7)
    Image.fromarray(grid([to_img(v) for v in lab[r2.choice(len(lab), 32, replace=False)]])).save(
        os.path.join(RES, f'distill_{tag}_real.png'))
    print(f"已保存 distill_{tag}_*.png")


if __name__ == '__main__':
    main()

#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""
卷积式 + 掩码预测 + 对称分解 + 未知标志（v3 修正）
====================================================
v2 的 bug: "未知"与"白色(0号)"编码冲突
  白色 one-hot = [+1,-1,...,-1]   未知 = [-1,-1,...,-1]
  两者共享 15 个 -1 通道 -> 模型把"未知"学成了白色
  证据: 真实中轴线白色 7.32%，生成图 25.81%（3.4 倍）

v3 修正: 给每个邻域像素加一条独立的"未知标志"通道
  输入 = 9x16(颜色) + 9(未知标志) + 32(位置) = 185

输出: conv3_<档位>_T<温度>.png  conv3_<档位>_real.png  conv3_<档位>_net.json
"""
import os, json, sys, time, colorsys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); RES = os.path.join(ROOT, "results")
C = 16
LIDX = np.array([r * 8 + c for r in range(8) for c in range(4)], dtype=int)
RIDX = np.array([r * 8 + (7 - c) for r in range(8) for c in range(4)], dtype=int)
H_HID = int(os.environ.get('H_HID', '128'))
EPOCHS = int(os.environ.get('EPOCHS', '40'))
BS = 256
LR = 1e-3
NIN = 9 * C + 9 + 32           # 185
POSOH = np.eye(32, dtype=np.float32)


def rgb_to_hsl_arr(pal):
    out = []
    for c in pal:
        h, l, s = colorsys.rgb_to_hls(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
        out.append([h, s, l])
    return np.array(out, dtype=np.float64)


def hood(X, known):
    """
    X:(B,64) 颜色; known:(B,64) bool
    返回 每个左半位置的 3x3 邻域 (B,32,9) 颜色 与 (B,32,9) 已知标志
    未知位置的"颜色"填 0（会被标志通道标记为未知，故不影响）
    """
    B = len(X)
    # 颜色图: 未知处置 0
    Xc = np.where(known, X, 0).reshape(B, 8, 8)
    Kc = known.reshape(B, 8, 8).astype(np.float32)
    Pc = np.zeros((B, 10, 10), dtype=np.int64); Pc[:, 1:9, 1:9] = Xc
    Pk = np.zeros((B, 10, 10), dtype=np.float32); Pk[:, 1:9, 1:9] = Kc
    nc, nk = [], []
    for pos in LIDX:
        r, c = divmod(pos, 8)
        nc.append(Pc[:, r:r+3, c:c+3].reshape(B, 9))
        nk.append(Pk[:, r:r+3, c:c+3].reshape(B, 9))
    return np.stack(nc, 1), np.stack(nk, 1)       # (B,32,9), (B,32,9)


def encode(nc, nk):
    """(B,32,9)x2 -> (B,32,185)"""
    B, L, K = nc.shape
    oh = np.zeros((B, L, K, C), dtype=np.float32)
    oh[np.arange(B)[:, None, None], np.arange(L)[None, :, None],
       np.arange(K)[None, None, :], nc] = 1.0
    oh = oh.reshape(B, L, K * C) * 2 - 1
    return np.concatenate([oh, nk, np.tile(POSOH[None], (B, 1, 1))], 2)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'd15'
    dclip = 3 if tag == 'd3' else 15
    print("=" * 96)
    print(f"卷积式 v3（含未知标志 + 对称分解）   {tag}")
    print("=" * 96)
    lab = np.load(os.path.join(DATA, 'faces8.npy')).astype(np.int64)
    pal = np.array(json.load(open(os.path.join(DATA, 'palette_avatar16.json')))['palette_ordered'],
                   dtype=np.uint8)
    HSL = rgb_to_hsl_arr(pal)
    N = min(len(lab), 30000)
    rng = np.random.default_rng(0)
    sub = lab[rng.choice(len(lab), N, replace=False)]
    print(f"数据 {lab.shape} -> 用 {N}   调色板 {C} 色   输入维 {NIN}")

    W1 = rng.normal(0, np.sqrt(2.0 / NIN), (NIN, H_HID)).astype(np.float32)
    b1 = np.zeros(H_HID, dtype=np.float32)
    W2 = rng.normal(0, np.sqrt(2.0 / H_HID), (H_HID, C)).astype(np.float32)
    b2 = np.zeros(C, dtype=np.float32)
    W3 = rng.normal(0, np.sqrt(2.0 / H_HID), (H_HID, 3)).astype(np.float32)
    b3 = np.zeros(3, dtype=np.float32)
    P = [W1, b1, W2, b2, W3, b3]
    M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
    print(f"网络 {NIN} -> {H_HID} -> (16色 | 3HSL)   参数 {sum(p.size for p in P)}")

    t = 0; t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        idx = rng.permutation(N)
        tl, tc, tn = 0.0, 0, 0
        for s in range(0, N - BS + 1, BS):
            Xb = sub[idx[s:s + BS]]
            B_ = len(Xb)
            # 随机挖洞（只挖左半）
            n_unk = rng.integers(6, 25, B_)
            unk = np.zeros((B_, 32), dtype=bool)
            for i in range(B_):
                unk[i, rng.choice(32, n_unk[i], replace=False)] = True
            known = np.ones((B_, 64), dtype=bool)
            known[:, LIDX] = ~unk
            nc, nk = hood(Xb, known)
            Xin = encode(nc, nk)                        # (B,32,185)
            flat = Xin.reshape(B_ * 32, NIN)
            h = np.maximum(flat @ W1 + b1, 0)
            lg = (h @ W2 + b2).reshape(B_, 32, C)
            hsl = (h @ W3 + b3).reshape(B_, 32, 3)
            tgt = Xb[:, LIDX]
            m = unk.reshape(-1)
            if m.sum() == 0:
                continue
            lgf = lg.reshape(-1, C)[m]
            tgf = tgt.reshape(-1)[m]
            mx = lgf.max(1, keepdims=True)
            e = np.exp(lgf - mx); sm = e / e.sum(1, keepdims=True)
            tl += float((-np.log(sm[np.arange(len(tgf)), tgf] + 1e-12)).mean()) * len(tgf)
            tc += int((lgf.argmax(1) == tgf).sum()); tn += len(tgf)
            dl = np.zeros((B_ * 32, C), dtype=np.float32)
            im = np.flatnonzero(m)
            dl[im] = sm.copy(); dl[im, tgf] -= 1; dl[im] /= len(tgf)
            hslf = hsl.reshape(-1, 3)
            dhsl = np.zeros_like(hslf)
            dhsl[im] = 2 * (hslf[im] - HSL[tgt.reshape(-1)][im]) * 0.02 / len(tgf)
            gW2 = h.T @ dl; gb2 = dl.sum(0)
            gW3 = h.T @ dhsl; gb3 = dhsl.sum(0)
            dh = (dl @ W2.T + dhsl @ W3.T) * (h > 0)
            gW1 = flat.T @ dh; gb1 = dh.sum(0)
            t += 1
            for k, g in enumerate((gW1, gb1, gW2, gb2, gW3, gb3)):
                M[k] = 0.9 * M[k] + 0.1 * g
                V[k] = 0.999 * V[k] + 0.001 * g ** 2
                P[k] -= LR * (M[k] / (1 - 0.9 ** t)) / (np.sqrt(V[k] / (1 - 0.999 ** t)) + 1e-8)
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d}  CE {tl/max(tn,1):.4f}  准确率 {tc/max(tn,1):.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    json.dump({'W1': W1.tolist(), 'b1': b1.tolist(), 'W2': W2.tolist(), 'b2': b2.tolist(),
               'W3': W3.tolist(), 'b3': b3.tolist(), 'palette': pal.tolist(),
               'mode': 'conv3_unknownflag', 'hid': H_HID},
              open(os.path.join(RES, f'conv3_{tag}_net.json'), 'w'))
    print(f"已保存 conv3_{tag}_net.json")

    # ---------- 生成 ----------
    def gen(nn, seed=0, temp=1.0, max_rounds=80):
        r = np.random.default_rng(seed)
        X = np.zeros((nn, 64), dtype=np.int64)
        known = np.zeros((nn, 64), dtype=bool)
        for rd in range(max_rounds):
            nc, nk = hood(X, known)
            Xin = encode(nc, nk)
            flat = Xin.reshape(nn * 32, NIN)
            h = np.maximum(flat @ W1 + b1, 0)
            lg = (h @ W2 + b2).reshape(nn, 32, C) / max(temp, 1e-6)
            mx = lg.max(2, keepdims=True)
            pr = np.exp(lg - mx); pr /= pr.sum(2, keepdims=True)
            # 统计未知数
            left_unknown = ~known[:, LIDX]
            if not left_unknown.any():
                break
            # 每轮每样本填一个未知位置
            for i in range(nn):
                cand = np.flatnonzero(left_unknown[i])
                if len(cand) == 0:
                    continue
                k = r.choice(cand)
                X[i, LIDX[k]] = r.choice(C, p=pr[i, k])
                known[i, LIDX[k]] = True
        # 右半: 用 clamp(s + d)，d 由网络对右半位置独立预测给出
        nc2, nk2 = hood(X, known)
        Xin2 = encode(nc2, nk2)
        h2 = np.maximum(Xin2.reshape(nn * 32, NIN) @ W1 + b1, 0)
        lg2 = (h2 @ W2 + b2).reshape(nn, 32, C)
        d_pred = np.clip(lg2.argmax(2) - X[:, LIDX], -dclip, dclip)
        X[:, RIDX] = np.clip(X[:, LIDX] + d_pred, 0, C - 1)
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

    MID = [r * 8 + c for r in range(8) for c in (3, 4)]
    NIMG = 32
    for temp, nm in ((0.6, 'T06'), (0.8, 'T08'), (1.0, 'T10')):
        G = gen(NIMG, seed=77, temp=temp)
        uc = np.mean([len(np.unique(v)) for v in G])
        sym = float((G[:, LIDX] == G[:, RIDX]).mean())
        wmid = float((G[:, MID] == 0).mean())
        Image.fromarray(grid([to_img(v) for v in G])).save(
            os.path.join(RES, f'conv3_{tag}_{nm}.png'))
        print(f"  {nm}: 颜色数 {uc:.2f}  对称率 {sym:.2f}  中轴线白色 {wmid:.2%} "
              f"(真实 7.32%)  -> conv3_{tag}_{nm}.png", flush=True)
    r2 = np.random.default_rng(7)
    Image.fromarray(grid([to_img(v) for v in lab[r2.choice(len(lab), NIMG, replace=False)]])).save(
        os.path.join(RES, f'conv3_{tag}_real.png'))
    print(f"  真实对照 -> conv3_{tag}_real.png")


if __name__ == '__main__':
    main()

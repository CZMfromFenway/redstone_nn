#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
对称 8x8 头像上的「离散 token」diffusion —— 纯 MLP 去噪器，输出 32x16 logits。

为什么还要做这一版：四期/二期最硬的一条纪律是「颜色必须当离散类别」。
连续 RGB 的回归已经被证死（E6：MAE 1.349 但视觉糊、脏）。而**连续 RGB 的 diffusion
在输出端仍然是回归**——它预测的是均值，只是用多步把它慢慢拉回来。既然现役的解码器
本来就是「32 位置 x 16 类 argmax + 查表」，那么把 diffusion 直接做在这个**类别**上，
既符合项目自己的纪律，又不增加任何解码硬件。

离散化方案：uniform D3PM（不动点 = 均匀分布）
  q(x_t | x_0) = ᾱ_t · δ(x_t = x_0) + (1-ᾱ_t)/K
  q(x_t | x_{t-1}) = ᾱ_{t|t-1} · δ + (1-ᾱ_{t|t-1})/K,   ᾱ_{t|t-1} = ᾱ_t/ᾱ_{t-1}
训练：给定 x_t 的 token，预测 x_0 的 32x16 logits（交叉熵）。
采样：**确定性**（对每个位置取后验 argmax）——
  x_{t-1} = argmax_j [ log q(x_t | j) + log prior(j) ],
  prior(j) = ᾱ_{t-1} if j == x̂_0 else (1-ᾱ_{t-1})/K
  这一步不需要新的随机数（只需初始的 x_T），对红石友好。

体积口径（一次前向的乘加数）：
  输入 = token embedding 32 x de（查表，红石上是 ROM，不算乘法）+ 时间条件 2*nf
  主体 = (32*de + 2nf) -> h -> h -> 32*K
  K=16 时最后一层 h x 512，与现役颜色头的最后一层同形，所以体积与现役同量级。

用法：
  python train_token_diffusion.py --tag tok_h96  --hid 96  --layers 2 --steps 40000
  python train_token_diffusion.py --tag tok_h192 --hid 192 --layers 2 --steps 40000
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
REPO = os.path.dirname(P4)
CACHE = os.path.join(REPO, "02_phase2_avatar", "results", "colour", "cache.npz")
OUTDIR = os.path.join(P4, "results", "diffusion")

ROW = np.array([p % 8 for p in range(32)])
COL = np.array([p // 8 for p in range(32)])
P, K = 32, 16


def mirror_to64(col32):
    img = np.zeros((64, 3), dtype=np.float64)
    img[ROW * 8 + COL] = col32
    img[ROW * 8 + (7 - COL)] = col32
    return img


def hex64(col32):
    img = mirror_to64(np.asarray(col32, dtype=np.float64).reshape(32, 3))
    return ["#%02x%02x%02x" % tuple(int(round(v * 17)) for v in np.clip(img[k], 0, 15))
            for k in range(64)]


def cosine_ab(T, s=0.008):
    t = np.arange(T + 1, dtype=np.float64) / T
    f = np.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    ab = np.clip(f / f[0], 1e-4, 0.9999)
    ab[0] = 1.0
    return ab.astype(np.float32)


def fourier(t_norm, nf=8):
    freqs = np.exp(-math.log(10000.0) * np.arange(nf) / nf)
    ang = (t_norm[:, None] * freqs[None, :]) * math.pi
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


def load_tokens(palette):
    z = np.load(CACHE)
    rgb = z["rgb96"].astype(np.float64).reshape(-1, P, 3)
    d = ((rgb[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(-1)   # (N,P,K)
    return d.argmin(-1).astype(np.int64)


# ------------------------------------------------------------------ 采样（确定性）
def sample_tokens(logit_fn, ab, xT, steps, kind="uniform"):
    """确定性采样。

    kind="uniform"：uniform D3PM 的后验 argmax（每步允许改任意位置）
    kind="mask"   ：吸收态（mask）扩散，每步只把「最有把握」的几个位置揭开
    """
    T = len(ab) - 1
    ts = [int(round(T - 1 - (T - 1) * i / steps)) for i in range(steps + 1)]

    if kind == "mask":
        x = np.full(P, K, dtype=np.int64)          # 全 mask
        left = P
        for i in range(steps):
            t = ts[i]
            lg = logit_fn(x, t / T)                 # (P,K)
            p = np.exp(lg - lg.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            conf = p.max(1)
            conf[x != K] = -1.0                     # 已揭开的不能再揭
            n = max(1, int(round(left / max(1, steps - i))))
            pick = np.argsort(-conf)[:min(n, left)]
            pick = pick[conf[pick] >= 0]
            x[pick] = lg[pick].argmax(1)
            left -= len(pick)
            if left <= 0:
                break
        if (x == K).any():                          # 兜底
            lg = logit_fn(x, 0.0)
            m = x == K
            x[m] = lg[m].argmax(1)
        return x

    x = xT.copy()
    for i in range(steps):
        t, tprev = ts[i], ts[i + 1]
        lg = logit_fn(x, t / T)                                  # (P,K)
        x0 = np.argmax(lg, axis=-1)                              # (P,)
        a_prev = ab[tprev]
        a_cond = ab[t] / max(ab[tprev], 1e-8)                    # ᾱ_{t|t-1}
        prior = np.full((P, K), (1 - a_prev) / K, dtype=np.float64)
        prior[np.arange(P), x0] = a_prev
        lq = np.full((P, K), (1 - a_cond) / K, dtype=np.float64)
        lq[np.arange(P), x] += a_cond
        post = np.log(lq + 1e-12) + np.log(prior + 1e-12)
        x = post.argmax(-1)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--kind", default="uniform", choices=["uniform", "mask"], help="uniform D3PM / 吸收态(mask)扩散")
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--de", type=int, default=8, help="token embedding 维数（查表）")
    ap.add_argument("--nfreq", type=int, default=8)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--log", type=int, default=10000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--no-ema", dest="use_ema", action="store_false", default=True)
    ap.add_argument("--nsample", type=int, default=200)
    ap.add_argument("--ssteps", default="4,10")
    ap.add_argument("--palette", default="results/fit/deploy_palette.json")
    ap.add_argument("--pal-arm", default="K16_w128_k16")
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)

    pal_all = json.load(open(os.path.join(P4, a.palette), encoding="utf-8"))
    palette = np.array(pal_all[a.pal_arm]["palette"], dtype=np.float64)
    toks = load_tokens(palette)
    print("token 数据：%s  K=%d  位置=%d" % (toks.shape, K, P), flush=True)

    N = len(toks)
    nv = max(1024, N // 50)
    tr = torch.from_numpy(toks[:-nv])
    va = torch.from_numpy(toks[-nv:])
    abnp = cosine_ab(a.T)                       # numpy 版：采样器用
    ab = torch.from_numpy(abnp)                 # torch 版：训练用

    din = P * a.de + 2 * a.nfreq
    dout = P * K
    emb = nn.Embedding(K + (1 if a.kind == "mask" else 0), a.de)
    body = []
    prev = din
    for i in range(a.layers):
        body.append(nn.Linear(prev, a.hid))
        body.append(nn.ReLU())
        prev = a.hid
    head = nn.Linear(prev, dout)
    net = nn.Sequential(*body, head)
    opt = torch.optim.Adam(list(net.parameters()) + list(emb.parameters()), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200.0) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))

    def forward(xt, tn):
        e = emb(xt).reshape(xt.shape[0], -1)                 # (B, P*de)
        f = torch.from_numpy(fourier(tn.numpy(), a.nfreq))
        return net(torch.cat([e, f], 1)).reshape(-1, P, K)

    def make_batch(bs, src):
        i = torch.randint(0, src.shape[0], (bs,))
        x0 = src[i]
        t = torch.randint(0, a.T, (bs,))
        abt = ab[t].unsqueeze(1)
        keep = torch.rand(bs, P) < abt
        if a.kind == "mask":
            xt = torch.where(keep, x0, torch.full_like(x0, K))   # 吸收态 = K
        else:
            rnd = torch.randint(0, K, (bs, P))
            xt = torch.where(keep, x0, rnd)
        return xt, x0, t.float() / a.T, (~keep)

    t0 = time.time()
    ema = None
    dm = min(a.ema, 1.0 - 1.0 / (a.steps + 1))
    for s in range(1, a.steps + 1):
        xt, x0, tn, msk = make_batch(a.bs, tr)
        lg = forward(xt, tn)
        if a.kind == "mask" and msk.any():
            # 只对被 mask 的位置算损失（标准做法；否则大半 token 是"抄答案"）
            lg2 = lg.reshape(-1, K)[msk.reshape(-1)]
            tgt = x0.reshape(-1)[msk.reshape(-1)]
            loss = nn.functional.cross_entropy(lg2, tgt)
        else:
            loss = nn.functional.cross_entropy(lg.reshape(-1, K), x0.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        params = list(net.parameters()) + list(emb.parameters())
        with torch.no_grad():
            if ema is None:
                ema = [p.detach().clone() for p in params]
            else:
                for e, p in zip(ema, params):
                    e.mul_(dm).add_(p, alpha=1 - dm)
        if s % a.log == 0 or s == 1:
            with torch.no_grad():
                xt, x0, tn, msk = make_batch(4096, va)
                lg = forward(xt, tn)
                if a.kind == "mask" and msk.any():
                    sel = msk.reshape(-1)
                    vl = nn.functional.cross_entropy(lg.reshape(-1, K)[sel], x0.reshape(-1)[sel]).item()
                    acc = (lg.reshape(-1, K)[sel].argmax(-1) == x0.reshape(-1)[sel]).float().mean().item()
                else:
                    vl = nn.functional.cross_entropy(lg.reshape(-1, K), x0.reshape(-1)).item()
                    acc = (lg.argmax(-1) == x0).float().mean().item()
            print("  step %6d  train %.4f  val %.4f  val_acc %.3f  %.0fs"
                  % (s, loss.item(), vl, acc, time.time() - t0), flush=True)

    # 冻结权重，转成 numpy 便于采样与导出
    if a.use_ema:
        ps = list(net.parameters()) + list(emb.parameters())
        for p, e in zip(ps, ema):
            p.data.copy_(e)
    Ew = emb.weight.detach().numpy().copy()                  # (K, de)
    mats = []
    for m in net:
        if isinstance(m, nn.Linear):
            mats.append((m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()))

    def logit_fn(xt_tokens, t_norm):
        e = Ew[xt_tokens].reshape(1, -1)
        f = fourier(np.array([t_norm], dtype=np.float32), a.nfreq)
        h = np.concatenate([e, f], axis=1)
        for i, (W, b) in enumerate(mats):
            h = h @ W + b
            if i < len(mats) - 1:
                h = np.maximum(h, 0.0)
        return h.reshape(P, K)

    mac = P * a.de * a.hid + 2 * a.nfreq * a.hid
    for i in range(a.layers - 1):
        mac += a.hid * a.hid
    mac += a.hid * P * K
    nparam = int(Ew.size + sum(W.size + b.size for W, b in mats))
    print("模型：hid=%d 层=%d de=%d · 参数 %d · **一次前向乘加 %d**"
          % (a.hid, a.layers, a.de, nparam, mac), flush=True)

    rng = np.random.default_rng(7)
    for steps in [int(x) for x in str(a.ssteps).split(",") if x.strip()]:
        rows = []
        t1 = time.time()
        for i in range(a.nsample):
            xT = rng.integers(0, K, P)
            x = sample_tokens(logit_fn, abnp, xT, steps, a.kind)
            col = palette[x]
            rows.append({"seed": int(i), "ms": 0.0, "raw": hex64(col), "snapped": hex64(col),
                         "sym": 1.0, "ncol": int(len(set(hex64(col)))),
                         "idx": [int(v) for v in x]})
        ms = (time.time() - t1) / max(1, a.nsample) * 1000
        for r in rows:
            r["ms"] = round(ms, 2)
        out = os.path.join(OUTDIR, "%s_s%d.json" % (a.tag, steps))
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"source": "train_token_diffusion.py", "tag": a.tag, "kind": ("mask" if a.kind == "mask" else "uniform-D3PM"),
                       "hid": a.hid, "layers": a.layers, "de": a.de, "T": a.T,
                       "train_steps": a.steps, "params": nparam, "mac_per_pass": mac,
                       "steps": steps, "ms_mean": round(ms, 2), "cfg_w": 0, "qb": None,
                       "alpha": 0, "k": K, "sym_tau": None, "palette_size": K,
                       "pal_arm": a.pal_arm, "samples": rows}, f, ensure_ascii=False)
        print("采样 %2d 步：%d 张，%.1f ms/张 -> %s"
              % (steps, a.nsample, ms, os.path.relpath(out, P4)), flush=True)

    wj = os.path.join(OUTDIR, "%s.json" % a.tag)
    with open(wj, "w", encoding="utf-8") as f:
        json.dump({"tag": a.tag, "kind": ("mask" if a.kind == "mask" else "uniform-D3PM"), "hid": a.hid, "layers": a.layers,
                   "de": a.de, "T": a.T, "mac_per_pass": mac, "params": nparam,
                   "emb": np.round(Ew, 6).tolist(),
                   "W": [[np.round(W, 6).tolist(), np.round(b, 6).tolist()] for W, b in mats],
                   "palette": palette.tolist(), "ab": [float(v) for v in abnp[::5]]}, f,
                  ensure_ascii=False)
    print("权重：%s" % os.path.relpath(wj, P4), flush=True)


if __name__ == "__main__":
    main()

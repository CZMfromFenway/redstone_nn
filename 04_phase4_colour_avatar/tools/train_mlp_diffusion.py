#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
对称 8x8 头像上的「纯 MLP 去噪器」diffusion —— 训练 + DDIM 采样 + 导出。

为什么是 MLP：红石的**体积**成本几乎等于「一次前向要做的乘加数」，
而对卷积来说一次前向的乘加数远大于参数量（3x3 核在 8x8 上展开成 484 个非零，
一行横跨 19 列，见 kb/20_数学/卷积_跨度与多核堆叠.md）。所以卷积在红石里天然贵。
而重复采样的 T 步**不改体积**，只改运行时间——这正是 diffusion 与红石兼容的地方。

本脚本刻意不用任何归一化层（GroupNorm/BatchNorm/LayerNorm）：
- 红石里「可变除数」的除法器极贵，而归一化每步都要算一次方差再除；
- 现役网络（4→64→32、32→128→128→32x16）也没有归一化，这是它便宜的原因。

数据：02_phase2_avatar/results/colour/cache.npz 的 rgb96 = 32 个自由位置（行内镜像后）× 3 通道，
      每通道 0..15 整数（4 bit），321,219 张。

用法：
  python train_mlp_diffusion.py --tag mlp96_h256 --hid 256 --layers 2 --steps 40000
  python train_mlp_diffusion.py --tag mlp96_h64  --hid 64  --layers 1 --steps 20000
产物：
  results/diffusion/<tag>.pt            权重（numpy 也能读，见 --export-json）
  results/diffusion/<tag>_samples.json  与 jim_diffusion_run.js 同 schema，可直接喂 jim_diffusion_gallery.py
  results/diffusion/<tag>.json          网页可用的纯 JSON（权重 + 元信息）
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


# ------------------------------------------------------------------ 数据
def load_data(n=-1, seed=0):
    z = np.load(CACHE)
    rgb = z["rgb96"].astype(np.float32)          # (N,96) uint8 0..15
    if n > 0:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(rgb))[:n]
        rgb = rgb[idx]
    # 0..15 -> [-1,1]
    return rgb / 7.5 - 1.0


# ------------------------------------------------------------------ 调度
def cosine_ab(T, s=0.008):
    """余弦 alpha_bar（Nichol & Dhariwal），与 Jim 的页面同族。"""
    t = np.arange(T + 1, dtype=np.float64) / T
    f = np.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    ab = f / f[0]
    ab = np.clip(ab, 1e-5, 0.9999)
    ab[0] = 1.0
    return ab.astype(np.float32)


# ------------------------------------------------------------------ 模型（numpy 手写，torch 只用来求梯度）
def init_mlp(din, hid, layers, dout, rng):
    """layers=1 即 单隐层；layers=2 即 双隐层。返回 [(W,b), ...]。"""
    dims = [din] + [hid] * layers + [dout]
    ps = []
    for i in range(len(dims) - 1):
        a, b = dims[i], dims[i + 1]
        scale = math.sqrt(2.0 / a) if i < len(dims) - 2 else math.sqrt(1.0 / a)
        ps.append((rng.standard_normal((a, b)).astype(np.float32) * scale,
                   np.zeros(b, dtype=np.float32)))
    return ps


def n_params(ps):
    return int(sum(W.size + b.size for W, b in ps))


def n_mac(ps):
    """一次前向的乘加数 = 体积成本的代理（对 MLP 恰好等于参数个数）。"""
    return int(sum(W.shape[0] * W.shape[1] for W, _ in ps))


# ------------------------------------------------------------------ 时间编码
def fourier(t_norm, nf=8):
    """t_norm ∈ [0,1] 的 (B,) -> (B,2nf)。红石上等价于一张 T x 2nf 的表（零乘法）。"""
    freqs = np.exp(-math.log(10000.0) * np.arange(nf) / nf)
    ang = (t_norm[:, None] * freqs[None, :]) * math.pi
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


# ------------------------------------------------------------------ torch 训练
def train(a, X):
    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)

    N, D = X.shape
    nv = max(512, N // 50)
    Xtr = torch.from_numpy(X[:-nv])
    Xva = torch.from_numpy(X[-nv:])
    T = a.T
    ab = torch.from_numpy(cosine_ab(T))
    abnp = ab.numpy()

    nf = a.nfreq
    din = D + 2 * nf

    rng = np.random.default_rng(a.seed)
    ps = init_mlp(din, a.hid, a.layers, D, rng)

    import torch.nn as nn
    mods = []
    for i, (W, b) in enumerate(ps):
        li = nn.Linear(W.shape[0], W.shape[1], bias=True)
        with torch.no_grad():
            li.weight.copy_(torch.from_numpy(W.T.copy()))
            li.bias.copy_(torch.from_numpy(b))
        mods.append(li)
    net = nn.Sequential(*[m for i, li in enumerate(mods)
                          for m in ((li, nn.ReLU()) if i < len(mods) - 1 else (li,))])
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200.0) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))

    def batch(bs, src):
        i = torch.randint(0, src.shape[0], (bs,))
        x0 = src[i]
        t = torch.randint(0, T, (bs,))
        abt = ab[t].unsqueeze(1)
        eps = torch.randn_like(x0)
        xt = abt.sqrt() * x0 + (1 - abt).sqrt() * eps
        tn = (t.float() / T).numpy()
        f = torch.from_numpy(fourier(tn, nf))
        inp = torch.cat([xt, f], dim=1)
        if a.pred == "v":
            tgt = abt.sqrt() * eps - (1 - abt).sqrt() * x0
        else:
            tgt = x0
        return inp, tgt

    t0 = time.time()
    ema = [p.detach().clone() for p in net.parameters()]
    dm = min(a.ema, 1.0 - 1.0 / (a.steps + 1))
    for s in range(1, a.steps + 1):
        inp, tgt = batch(a.bs, Xtr)
        pred = net(inp)
        loss = ((pred - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for e, p in zip(ema, net.parameters()):
                e.mul_(dm).add_(p, alpha=1 - dm)
        if s % a.log == 0 or s == 1:
            with torch.no_grad():
                inp, tgt = batch(2048, Xva)
                vl = ((net(inp) - tgt) ** 2).mean().item()
            print("  step %6d  train %.5f  val %.5f  lr %.2e  %.0fs"
                  % (s, loss.item(), vl, sched.get_last_lr()[0], time.time() - t0), flush=True)

    sd = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
    out = []
    for i in range(len(mods)):
        W = sd["%d.weight" % (2 * i)].T.copy()      # torch: (out,in) -> numpy (in,out)
        b = sd["%d.bias" % (2 * i)]
        out.append((W, b))
    # 用 EMA 权重替换（采样只用 EMA；训练曲线的 val 是原始权重，会略差）
    emaW, emab = ema[0::2], ema[1::2]
    out_ema = [(W.detach().numpy().T.copy(), b.detach().numpy().copy())
               for W, b in zip(emaW, emab)]
    return out, out_ema


# ------------------------------------------------------------------ 采样
def ddim(ps, ab, x_init, steps, pred="v", nf=8):
    """确定性 DDIM（eta=0），返回每步的 x0 列表。"""
    T = len(ab) - 1
    ts = [int(round(T - 1 - (T - 1) * i / steps)) for i in range(steps + 1)]
    x = x_init.copy().reshape(1, -1)
    hist = []
    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        inp = np.concatenate([x, fourier(np.array([t / T], dtype=np.float32), nf)], axis=1)
        h = inp
        for k, (W, b) in enumerate(ps):
            h = h @ W + b
            if k < len(ps) - 1:
                h = np.maximum(h, 0.0)
        out = h[0]
        a_t, a_n = ab[t], ab[tn]
        if pred == "v":
            x0 = math.sqrt(a_t) * x - math.sqrt(1 - a_t) * out
            eps = math.sqrt(a_t) * out + math.sqrt(1 - a_t) * x
        else:
            x0 = out
            eps = (x - math.sqrt(a_t) * x0) / max(1e-6, math.sqrt(1 - a_t))
        x0 = np.clip(x0, -1, 1)
        hist.append(x0.copy())
        x = math.sqrt(a_n) * x0 + math.sqrt(1 - a_n) * eps
    return hist


# ------------------------------------------------------------------ 解码
def mirror_to64(col32):
    """(32,3) 的左半页颜色 -> (64,3) 的完整 8x8（行内镜像）。"""
    img = np.zeros((64, 3), dtype=np.float32)
    img[ROW * 8 + COL] = col32
    img[ROW * 8 + (7 - COL)] = col32
    return img


def decode(x0, palette):
    """[-1,1] 的 96 值 -> 32 个调色板索引（最近色）-> 64 个 hex；另返回吸附前的连续色 hex。"""
    lev = np.clip((x0 + 1) * 7.5, 0, 15).reshape(32, 3)
    d = ((lev[:, None, :] - palette[None, :, :]) ** 2).sum(-1)      # (32,16)
    idx = d.argmin(1)
    raw = mirror_to64(lev)
    snap = mirror_to64(palette[idx])
    hx = lambda arr: ["#%02x%02x%02x" % tuple(int(round(v * 17)) for v in np.clip(arr[k], 0, 15))
                      for k in range(64)]
    return idx, hx(raw), hx(snap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2, help="1=单隐层 2=双隐层")
    ap.add_argument("--nfreq", type=int, default=8)
    ap.add_argument("--pred", default="v", choices=["v", "x0"])
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--log", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--no-ema", dest="sample_ema", action="store_false", default=True)
    ap.add_argument("--n", type=int, default=-1, help="只用前 n 张（调试用）")
    ap.add_argument("--nsample", type=int, default=200)
    ap.add_argument("--ssteps", default="4,8,10,20")
    ap.add_argument("--palette", default="results/fit/deploy_palette.json")
    ap.add_argument("--pal-arm", default="K16_w128_k16")
    a = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    print("数据：%s" % os.path.relpath(CACHE, REPO), flush=True)
    X = load_data(a.n, a.seed)
    print("  形状 %s  范围 %.2f..%.2f" % (X.shape, X.min(), X.max()), flush=True)

    ps, ps_ema = train(a, X)
    ps = ps_ema if a.sample_ema else ps
    ab = cosine_ab(a.T)
    D = X.shape[1]
    mac = n_mac(ps)
    par = n_params(ps)
    print("模型：%d 层 · hid=%d · 参数 %d · **一次前向乘加 %d**" % (a.layers, a.hid, par, mac), flush=True)

    # 调色板
    pal_all = json.load(open(os.path.join(P4, a.palette), encoding="utf-8"))
    palette = np.array(pal_all[a.pal_arm]["palette"], dtype=np.float32)
    print("调色板：%s / %s（%d 槽）" % (a.palette, a.pal_arm, len(palette)), flush=True)

    rng = np.random.default_rng(12345)
    summary = {
        "source": "train_mlp_diffusion.py",
        "tag": a.tag, "hid": a.hid, "layers": a.layers, "pred": a.pred,
        "T": a.T, "train_steps": a.steps, "bs": a.bs,
        "params": par, "mac_per_pass": mac,
        "space": "rgb96 (32 位置 x RGB, 行内镜像)",
        "palette_source": a.palette, "pal_arm": a.pal_arm, "palette_size": len(palette),
        "cfg_w": 0, "qb": None, "alpha": 0, "k": len(palette), "sym_tau": None,
        "samples": [],
    }

    for steps in [int(s) for s in str(a.ssteps).split(",") if s.strip()]:
        rows = []
        t0 = time.time()
        for i in range(a.nsample):
            x_init = rng.standard_normal(D).astype(np.float32)
            hist = ddim(ps, ab, x_init, steps, a.pred, a.nfreq)
            x0 = hist[-1]
            idx, raw_hexes, hexes = decode(x0, palette)
            rows.append({
                "seed": int(i),
                "ms": 0.0,
                "raw": raw_hexes,
                "snapped": hexes,
                "sym": 1.0,
                "ncol": int(len(set(hexes))),
                "idx": [int(v) for v in idx],
            })
        ms = (time.time() - t0) / max(1, a.nsample) * 1000
        for r in rows:
            r["ms"] = round(ms, 2)
        out = os.path.join(OUTDIR, "%s_s%d.json" % (a.tag, steps))
        summary2 = dict(summary)
        summary2["steps"] = steps
        summary2["ms_mean"] = round(ms, 2)
        summary2["samples"] = rows
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary2, f, ensure_ascii=False)
        print("采样 %2d 步：%d 张，%.1f ms/张 -> %s" %
              (steps, a.nsample, ms, os.path.relpath(out, P4)), flush=True)

    # 权重导出（网页/后续工具可直接读）
    wj = os.path.join(OUTDIR, "%s.json" % a.tag)
    with open(wj, "w", encoding="utf-8") as f:
        json.dump({**{k: v for k, v in summary.items() if k != "samples"},
                   "layers_io": [[int(W.shape[0]), int(W.shape[1])] for W, _ in ps],
                   "W": [[np.round(W, 6).tolist(), np.round(b, 6).tolist()] for W, b in ps],
                   "palette": palette.tolist(),
                   "ab": [float(v) for v in ab[::10]]}, f, ensure_ascii=False)
    print("权重：%s" % os.path.relpath(wj, P4), flush=True)


if __name__ == "__main__":
    main()

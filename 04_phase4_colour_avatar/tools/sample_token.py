#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
只做采样、不重训：读 `train_token_diffusion.py --kind mask` 存下的权重，试不同的「揭盖」策略。

为什么单独写：mask（吸收态）扩散的**采样器**是决定成败的部分，换策略不该重训 14 分钟。

策略：
  conf   每步揭开「最自信」的若干个位置，token 取 argmax（确定性）
  rand   同上，但 token 从预测分布里**采样**（每步要 32 个随机数）
  slow   前 1/3 步每步只揭 1 个位置（argmax），后 2/3 按 conf 批量揭开

用法：
  python sample_token.py --weights results/diffusion/tokM_h192.json --scheme rand --steps 10 --n 200
"""
import argparse
import json
import math
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
OUTDIR = os.path.join(P4, "results", "diffusion")

ROW = np.array([p % 8 for p in range(32)])
COL = np.array([p // 8 for p in range(32)])
P, K = 32, 16


def fourier(t_norm, nf=8):
    freqs = np.exp(-math.log(10000.0) * np.arange(nf) / nf)
    ang = (t_norm[:, None] * freqs[None, :]) * math.pi
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


def hex64(col32):
    img = np.zeros((64, 3), dtype=np.float64)
    c = np.asarray(col32, dtype=np.float64).reshape(32, 3)
    img[ROW * 8 + COL] = c
    img[ROW * 8 + (7 - COL)] = c
    return ["#%02x%02x%02x" % tuple(int(round(v * 17)) for v in np.clip(img[k], 0, 15))
            for k in range(64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--scheme", default="conf", choices=["conf", "rand", "slow"])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    w = json.load(open(os.path.join(P4, a.weights), encoding="utf-8"))
    emb = np.array(w["emb"], dtype=np.float64)             # (K+1, de)
    mats = [(np.array(W, dtype=np.float64), np.array(b, dtype=np.float64)) for W, b in w["W"]]
    palette = np.array(w["palette"], dtype=np.float64)
    nf = (mats[0][0].shape[0] - P * emb.shape[1]) // 2
    T = len(w["ab"]) * 5 - 1                                # 导出时按 ::5 抽样过

    def logits(x, t_norm):
        e = emb[x].reshape(1, -1)
        f = fourier(np.array([t_norm], dtype=np.float32), nf)
        h = np.concatenate([e, f], axis=1)
        for i, (W, b) in enumerate(mats):
            h = h @ W + b
            if i < len(mats) - 1:
                h = np.maximum(h, 0.0)
        return h.reshape(P, K)

    def probs(lg):
        p = np.exp(lg - lg.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    rng = np.random.default_rng(a.seed)
    rows = []
    for i in range(a.n):
        x = np.full(P, K, dtype=np.int64)
        left = P
        for s in range(a.steps):
            t_norm = 1.0 - (s + 0.5) / a.steps
            t_norm = max(0.0, min(1.0, t_norm)) * (T / (T + 1.0))
            lg = logits(x, t_norm)
            p = probs(lg)
            conf = p.max(1)
            conf[x != K] = -1.0
            if a.scheme == "slow":
                n = 1 if s < max(1, a.steps // 3) else max(1, int(round(left / max(1, a.steps - s))))
            else:
                n = max(1, int(round(left / max(1, a.steps - s))))
            pick = np.argsort(-conf)[:min(n, left)]
            pick = pick[conf[pick] >= 0]
            if a.scheme == "rand":
                tok = np.array([rng.choice(K, p=p[j]) for j in pick])
            else:
                tok = lg[pick].argmax(1)
            x[pick] = tok
            left -= len(pick)
            if left <= 0:
                break
        if (x == K).any():
            lg = logits(x, 0.0)
            m = x == K
            x[m] = lg[m].argmax(1)
        col = palette[x]
        hx = hex64(col)
        rows.append({"seed": i, "ms": 0.0, "raw": hx, "snapped": hx,
                     "sym": 1.0, "ncol": int(len(set(hx))), "idx": [int(v) for v in x]})

    tag = a.tag or "%s_%s" % (os.path.basename(a.weights).replace(".json", ""), a.scheme)
    out = os.path.join(OUTDIR, "%s_s%d.json" % (tag, a.steps))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"source": "sample_token.py", "tag": tag, "scheme": a.scheme,
                   "hid": w.get("hid"), "layers": w.get("layers"), "de": w.get("de"),
                   "T": w.get("T"), "steps": a.steps, "ms_mean": 0.0, "cfg_w": 0, "qb": None,
                   "alpha": 0, "k": K, "sym_tau": None, "palette_size": K,
                   "mac_per_pass": w.get("mac_per_pass"), "pal_arm": "K16_w128_k16",
                   "samples": rows}, f, ensure_ascii=False)
    print("写了 %s（%d 张，scheme=%s）" % (os.path.relpath(out, P4), len(rows), a.scheme))


if __name__ == "__main__":
    main()

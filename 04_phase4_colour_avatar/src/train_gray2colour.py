#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
只给灰度，直接预测颜色：模型能学到「好看的颜色」吗？

输入 32 个灰度级（同一张脸，0..15 /15）→ 输出 96 个 RGB 值。不给任何颜色 PCA。
对照：
    C  两段式（灰度 → 线性预测颜色 PCA → 上色网络）—— 信息上等价，但多一层瓶颈
    E6 单阶段（只有颜色 PCA，8 维随机数）
数据：cache.npz（rgb96、cpc）+ L0_full_idx（同一行的灰度）

产物：results/colourise/gray2colour.json、g2c.png
用法：python train_gray2colour.py [--steps 12000] [--tag g2c]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
REPO = os.path.dirname(P4)
P2 = os.path.join(REPO, "02_phase2_avatar")
OUT = os.path.join(P4, "results", "colourise")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(P4, "src"))

ROW = np.array([p % 8 for p in range(32)])
COL = np.array([p // 8 for p in range(32)])
LUM = np.array([0.299, 0.587, 0.114])


def font(sz):
    for fn in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def mirror32(v):
    img = np.zeros((len(v), 8, 8, 3))
    img[:, ROW, COL] = v
    img[:, ROW, 7 - COL] = v
    return img


def stats(p):
    """p: (N,32,3) 0..15 -> 观感统计"""
    flat = p.reshape(len(p), 32, 3)
    sat = np.where(flat.max(-1) > 1e-6, (flat.max(-1) - flat.min(-1)) / np.maximum(flat.max(-1), 1e-6), 0)
    ncol = np.mean([len({tuple(np.rint(c).astype(int)) for c in flat[i]}) for i in range(len(flat))])
    img = mirror32(p)
    dh = np.linalg.norm(img[:, :, 1:] - img[:, :, :-1], axis=-1)
    dv = np.linalg.norm(img[:, 1:] - img[:, :-1], axis=-1)
    return {"sat": round(float(sat.mean()), 3), "ncol": round(float(ncol), 2),
            "edge_p90": round(float(np.percentile(np.concatenate([dh.ravel(), dv.ravel()]), 90)), 2),
            "lum": round(float((flat @ LUM).mean()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="g2c")
    A = ap.parse_args()
    torch.manual_seed(A.seed)

    import train_palette as tp
    rgb96, cpc, tr, va = tp.load_cache(8, "ext32")
    full = np.load(os.path.join(P2, "data_v2_canonical", "L0_full_idx.npy")).astype(np.float64)
    gray = full[:, ROW, COL] / 15.0
    Y = rgb96 / 15.0
    net = nn.Sequential(nn.Linear(32, A.hid), nn.ReLU(), nn.Linear(A.hid, A.hid), nn.ReLU(),
                        nn.Linear(A.hid, 96))
    opt = torch.optim.Adam(net.parameters(), lr=A.lr)
    mults = 32 * A.hid + A.hid * A.hid + A.hid * 96
    print(f"灰度 32 → {A.hid} → {A.hid} → 96；乘法 {mults:,}；步数 {A.steps}")
    rng = np.random.default_rng(A.seed)
    t0 = time.time()
    for step in range(1, A.steps + 1):
        bi = tr[rng.integers(0, len(tr), A.batch)]
        x = torch.tensor(gray[bi], dtype=torch.float32)
        y = torch.tensor(Y[bi], dtype=torch.float32)
        loss = F.mse_loss(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 4000 == 0 or step == A.steps:
            with torch.no_grad():
                pv = np.clip(net(torch.tensor(gray[va[:1000]], dtype=torch.float32)).numpy() * 15, 0, 15)
            mae = float(np.abs(pv.reshape(-1, 32, 3) - rgb96[va[:1000]].reshape(-1, 32, 3)).mean())
            print(f"  step {step:6d}  loss {float(loss.item()):.4f}  留出 MAE {mae:.2f} 级  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    rows = va[:1000]
    real = rgb96[rows].reshape(-1, 32, 3)
    with torch.no_grad():
        pG = np.clip(net(torch.tensor(gray[rows], dtype=torch.float32)).numpy() * 15, 0, 15).reshape(-1, 32, 3)

    # C 参照：灰度 → 线性预测颜色 PCA → 上色网络（复用 train_colourise 的模型）
    cj = os.path.join(OUT, "model_c8.json")
    pC = None
    if os.path.exists(cj):
        m = json.load(open(cj, encoding="utf-8"))
        Fm = np.hstack([gray[tr], np.ones((len(tr), 1))])
        Wr = np.linalg.solve(Fm.T @ Fm + 10.0 * np.eye(Fm.shape[1]), Fm.T @ cpc[tr][:, :8])
        pred_c = np.hstack([gray[rows], np.ones((len(rows), 1))]) @ Wr
        h = np.concatenate([gray[rows], pred_c], 1)
        for i, L in enumerate(m["body"]):
            h = np.maximum(0, h @ np.array(L["W"]).T + np.array(L["b"]))
        h = h @ np.array(m["head"]["W"]).T + np.array(m["head"]["b"])
        pC = np.clip(h * 15, 0, 15).reshape(-1, 32, 3)

    def hueerr(p, q):
        def hue(x):
            mx, mn = x.max(-1), x.min(-1)
            c = mx - mn
            r, g, b = x[..., 0], x[..., 1], x[..., 2]
            h = np.zeros_like(mx)
            m1 = (c > 1e-6) & (mx == r); h[m1] = ((g[m1]-b[m1])/c[m1]) % 6
            m2 = (c > 1e-6) & (mx == g); h[m2] = (b[m2]-r[m2])/c[m2] + 2
            m3 = (c > 1e-6) & (mx == b); h[m3] = (r[m3]-g[m3])/c[m3] + 4
            return h*60, c
        h1, c1 = hue(p); h2, c2 = hue(q)
        m = (c1 > 2) & (c2 > 2)
        if not m.any():
            return None
        d = np.abs(h1[m]-h2[m]); d = np.minimum(d, 360-d)
        return round(float(d.mean()), 1)

    res = {}
    for name, p in (("真实灰度→颜色", pG), ("两段式 C（灰度→预测色PCA→上色）", pC)):
        if p is None:
            continue
        res[name] = {"rgb_mae": round(float(np.abs(p - real).mean()), 3),
                     "hue_err_deg": hueerr(p, real), **stats(p)}
        print(f"  {name:28s} MAE {res[name]['rgb_mae']:5.2f} 级  色相 {res[name]['hue_err_deg']}°  "
              f"饱和 {res[name]['sat']:.2f}  色数 {res[name]['ncol']:.2f}  边缘 {res[name]['edge_p90']:.2f}")
    res["真实数据（参照）"] = stats(real)
    print(f"  {'真实数据（参照）':28s} {'':16s}        饱和 {res['真实数据（参照）']['sat']:.2f}  "
          f"色数 {res['真实数据（参照）']['ncol']:.2f}  边缘 {res['真实数据（参照）']['edge_p90']:.2f}")
    Ls = [m for m in net if isinstance(m, nn.Linear)]
    mdl = {"type": "gray2colour", "body": [], "head": {}, "meta": {"mults": int(mults),
           "steps": A.steps, "tag": A.tag, "inputs": "32 灰度级/15"}}
    for L in Ls[:-1]:
        mdl["body"].append({"W": L.weight.detach().numpy().round(7).tolist(),
                            "b": L.bias.detach().numpy().round(7).tolist()})
    mdl["head"] = {"W": Ls[-1].weight.detach().numpy().round(7).tolist(),
                   "b": Ls[-1].bias.detach().numpy().round(7).tolist()}
    json.dump(mdl, open(os.path.join(OUT, f"model_{A.tag}.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump({"args": vars(A), "mults": int(mults), "results": res},
              open(os.path.join(OUT, "gray2colour.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # 图
    TILE, PAD, LW, COLS = 52, 4, 300, 16
    W = LW + COLS * (TILE + PAD) + 20
    rows_fig = [("真实", real), ("只给灰度（本模型）", pG)]
    if pC is not None:
        rows_fig.append(("两段式 C", pC))
    H = 92 + len(rows_fig) * (TILE + 30)
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((14, 12), "只给灰度预测颜色：它学到的是「合理的配色」还是「好看的颜色」", font=font(19),
            fill=(234, 238, 244))
    dr.text((14, 40), "输入只有 32 个灰度级；同一张脸在真实世界里可以有很多种配色，模型只能给"
                      "「平均配色」。", font=font(11), fill=(148, 160, 176))
    dr.text((14, 58), f"MAE {res['真实灰度→颜色']['rgb_mae']:.2f} 级 · 饱和 {res['真实灰度→颜色']['sat']:.2f} "
                      f"（真实 {res['真实数据（参照）']['sat']:.2f}）· 色数 {res['真实灰度→颜色']['ncol']:.2f} "
                      f"（真实 {res['真实数据（参照）']['ncol']:.2f}）", font=font(11), fill=(130, 142, 158))
    y = 84
    for label, arr in rows_fig:
        dr.text((14, y + 8), label, font=font(12), fill=(112, 170, 254))
        img = mirror32(arr[:COLS])
        for i in range(COLS):
            v = np.clip(np.rint(img[i] * 17), 0, 255).astype(np.uint8)
            t = np.repeat(np.repeat(v, TILE // 8, 0), TILE // 8, 1)
            cv.paste(Image.fromarray(t), (LW + i * (TILE + PAD), y))
        y += TILE + 30
    p = os.path.join(OUT, "g2c.png")
    cv.save(p)
    print(f"wrote {os.path.relpath(p, REPO)}")
    print("G2C_DONE")


if __name__ == "__main__":
    main()

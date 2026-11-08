#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
上色训练（两阶段路线的第二半）：给「同一张头像的灰度图 + 颜色 PCA」，还原它的颜色。

三种输入口径，一次跑完对比：

  A 重建（真值灰度 + 真值颜色 PCA）         —— 这是上色模块的能力上限
  B 生成（生成器灰度 + 采样颜色 PCA）        —— 真实两阶段部署时会遇到的情况
  C 生成 + 条件先验（生成器灰度 + 由灰度预测的颜色 PCA）—— 修 B 的"颜色与脸不配套"
  参照 E6（只有颜色 PCA、不给灰度，二期已部署的 8→128→128→96）

数据：`cache.npz`（rgb96、cpc）与 `L0_full_idx.npy`（同一行的灰度 8×8，0..15），
     同一行的灰度与颜色来自**同一张脸**（这是与二期 E3 的关键差别：二期喂的是阶段 1 的预测灰度）。

产物：results/colourise/{model.json,summary.json,cmp.png}
用法：python train_colourise.py [--k 8] [--steps 12000] [--tag c8]
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
    v = np.asarray(v)
    img = np.zeros((len(v), 8, 8, v.shape[-1]))
    img[:, ROW, COL] = v
    img[:, ROW, 7 - COL] = v
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="c8")
    A = ap.parse_args()
    torch.manual_seed(A.seed)

    import train_palette as tp
    rgb96, cpc, tr, va = tp.load_cache(A.k, "ext32")
    full = np.load(os.path.join(P2, "data_v2_canonical", "L0_full_idx.npy")).astype(np.float64)
    gray32 = full[:, ROW, COL] / 15.0                     # 同一张脸的灰度，32 位置
    N = len(rgb96)
    print(f"数据 {N:,} 张；灰度 32 维 + 颜色 PCA {A.k} 维 → RGB96")

    Xg = gray32
    Xc = cpc[:, :A.k]
    Y = rgb96 / 15.0
    din = 32 + A.k

    net = nn.Sequential(nn.Linear(din, A.hid), nn.ReLU(),
                        nn.Linear(A.hid, A.hid), nn.ReLU(),
                        nn.Linear(A.hid, 96))
    opt = torch.optim.Adam(net.parameters(), lr=A.lr)
    mults = din * A.hid + A.hid * A.hid + A.hid * 96
    print(f"结构 {din}→{A.hid}→{A.hid}→96  参数 {sum(p.numel() for p in net.parameters()):,}"
          f"  乘法 {mults:,}  步数 {A.steps}")
    rng = np.random.default_rng(A.seed)
    t0 = time.time()
    for step in range(1, A.steps + 1):
        bi = tr[rng.integers(0, len(tr), A.batch)]
        x = torch.tensor(np.concatenate([Xg[bi], Xc[bi]], 1), dtype=torch.float32)
        y = torch.tensor(Y[bi], dtype=torch.float32)
        pred = net(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 4000 == 0 or step == A.steps:
            with torch.no_grad():
                xv = torch.tensor(np.concatenate([Xg[va[:1000]], Xc[va[:1000]]], 1),
                                  dtype=torch.float32)
                pv = np.clip(net(xv).numpy() * 15.0, 0, 15)
            mae = float(np.abs(pv.reshape(-1, 32, 3) - rgb96[va[:1000]].reshape(-1, 32, 3)).mean())
            print(f"  step {step:6d}  loss {float(loss.item()):.4f}  "
                  f"留出 RGB MAE {mae:.3f} 级  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- 评估三种口径
    def predict(gray, col):
        with torch.no_grad():
            x = torch.tensor(np.concatenate([gray, col], 1), dtype=torch.float32)
            return np.clip(net(x).numpy() * 15.0, 0, 15).reshape(len(gray), 32, 3)

    ev = {"A_recon_true": {}, "B_gen_sampled": {}, "C_gen_prior": {},
          "D_prior_resample": {}, "E6_cpc_only": {}}
    rows = va[:1000]
    real = rgb96[rows].reshape(-1, 32, 3)
    # A
    pA = predict(Xg[rows], Xc[rows])
    # B：灰度来自 k8 生成器（用真实坐标），颜色 PCA 独立采样
    k8 = json.load(open(os.path.join(P4, "results", "train", "K16_w128_k8", "model.json"),
                        encoding="utf-8"))
    lay = [(np.array(L["W"]), np.array(L["b"]), True) for L in k8["body"]]
    lay.append((np.array(k8["head"]["W"]), np.array(k8["head"]["b"]), False))
    h = cpc[rows][:, :8]
    for W, b, relu in lay:
        h = h @ W.T + b
        if relu:
            h = np.maximum(h, 0)
    gen_idx = h.reshape(-1, 32, 16).argmax(-1)           # 生成器的灰度级
    gen_gray = gen_idx / 15.0
    sampled = np.clip(np.random.default_rng(1).standard_normal((len(rows), A.k)), -3.5, 3.5)
    pB = predict(gen_gray, sampled)
    # C：用灰度线性预测颜色 PCA（条件先验），闭式岭回归
    Fm = np.hstack([Xg[tr], np.ones((len(tr), 1))])
    Wr = np.linalg.solve(Fm.T @ Fm + 10.0 * np.eye(Fm.shape[1]), Fm.T @ Xc[tr])
    pred_c = np.hstack([gen_gray, np.ones((len(rows), 1))]) @ Wr
    pC = predict(gen_gray, pred_c)
    # D：条件先验 + 残差重采样（把被回归压扁的色度方差补回来）
    resid = Xc[tr] - np.hstack([Xg[tr], np.ones((len(tr), 1))]) @ Wr
    Cov = np.cov(resid.T) + 1e-8 * np.eye(A.k)
    Lc = np.linalg.cholesky(Cov)
    rng2 = np.random.default_rng(2)
    pred_d = pred_c + rng2.standard_normal((len(rows), A.k)) @ Lc.T
    pD = predict(gen_gray, pred_d)
    # E6 参照
    e6 = json.load(open(os.path.join(P2, "webapp", "colour_e6_model.json"), encoding="utf-8"))
    hl = cpc[rows][:, :8]
    Ls = e6["gen"]["layers"]
    for i, L in enumerate(Ls):
        hl = hl @ np.array(L["W"]).T + np.array(L["b"])
        if i < len(Ls) - 1:
            hl = np.maximum(hl, 0)
    pE = np.clip(hl * 15.0, 0, 15).reshape(-1, 32, 3)

    for name, p in (("A_recon_true", pA), ("B_gen_sampled", pB),
                    ("C_gen_prior", pC), ("D_prior_resample", pD), ("E6_cpc_only", pE)):
        mae = float(np.abs(p - real).mean())
        # 色相误差：只在高色度像素上算
        def hue(x):
            mx = x.max(-1); mn = x.min(-1)
            c = mx - mn
            r, g, b = x[..., 0], x[..., 1], x[..., 2]
            h = np.zeros_like(mx)
            m1 = (c > 1e-6) & (mx == r); h[m1] = ((g[m1] - b[m1]) / c[m1]) % 6
            m2 = (c > 1e-6) & (mx == g); h[m2] = (b[m2] - r[m2]) / c[m2] + 2
            m3 = (c > 1e-6) & (mx == b); h[m3] = (r[m3] - g[m3]) / c[m3] + 4
            return h * 60, c
        h1, c1 = hue(p); h2, c2 = hue(real)
        m = (c1 > 2) & (c2 > 2)
        dh = np.abs(h1[m] - h2[m]); dh = np.minimum(dh, 360 - dh)
        ev[name] = {"rgb_mae": round(mae, 3),
                    "hue_err_deg": round(float(dh.mean()), 2) if m.any() else None,
                    "sat_mean": round(float((p.max(-1) - p.min(-1)).mean()), 3),
                    "hue_px_frac": round(float(m.mean()), 3)}
        print(f"  {name:16s} RGB MAE {mae:5.2f} 级  色相误差 "
              f"{ev[name]['hue_err_deg']}°  饱和度均值 {ev[name]['sat_mean']:.2f}")

    # 存模型
    model = {"type": "colourise", "body": [], "head": {},
             "meta": {"k": A.k, "hid": A.hid, "in_dim": din, "mults": int(mults),
                      "steps": A.steps, "inputs": "32 灰度级/15 + 颜色 PCA(k)", "tag": A.tag,
                      "rgb_mae_A_recon": ev["A_recon_true"]["rgb_mae"],
                      "rgb_mae_B_gen": ev["B_gen_sampled"]["rgb_mae"]}}
    Ls2 = [m for m in net if isinstance(m, nn.Linear)]
    for L in Ls2[:-1]:
        model["body"].append({"W": L.weight.detach().numpy().round(7).tolist(),
                              "b": L.bias.detach().numpy().round(7).tolist()})
    model["head"] = {"W": Ls2[-1].weight.detach().numpy().round(7).tolist(),
                     "b": Ls2[-1].bias.detach().numpy().round(7).tolist()}
    json.dump(model, open(os.path.join(OUT, f"model_{A.tag}.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump({"args": vars(A), "mults": int(mults), "evals": ev,
               "note": "A=真值灰度+真值颜色PCA；B=生成器灰度+采样颜色PCA；"
                       "C=生成器灰度+灰度预测的颜色PCA；E6=只有颜色PCA"},
              open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # 图：四行（真实 / A / B / C）
    TILE, PAD, LW, COLS = 52, 4, 300, 16
    W = LW + COLS * (TILE + PAD) + 20
    H = 96 + 5 * (TILE + 30)
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((14, 12), "上色训练：给同一张脸的灰度图 + 颜色 PCA，还原它的颜色", font=font(19),
            fill=(234, 238, 244))
    dr.text((14, 40), "A 用真值灰度与真值颜色 PCA（能力上限）；B 用生成器灰度 + 采样颜色 PCA"
                      "（真实部署会遇到）；C 用生成器灰度 + 由灰度预测的颜色 PCA。", font=font(11),
            fill=(148, 160, 176))
    dr.text((14, 58), f"RGB MAE：A {ev['A_recon_true']['rgb_mae']:.2f} · "
                      f"B {ev['B_gen_sampled']['rgb_mae']:.2f} · "
                      f"C {ev['C_gen_prior']['rgb_mae']:.2f} · "
                      f"D {ev['D_prior_resample']['rgb_mae']:.2f} · "
                      f"E6（无灰度）{ev['E6_cpc_only']['rgb_mae']:.2f} 级", font=font(11),
            fill=(130, 142, 158))
    y = 84
    for label, arr in (("真实", real), ("A 真值灰度+真值色", pA),
                       ("B 生成灰度+采样色", pB), ("C 生成灰度+预测色", pC),
                       ("D 预测色+残差重采样", pD)):
        dr.text((14, y + 8), label, font=font(12), fill=(112, 170, 254))
        img = mirror32(arr[:COLS])
        for i in range(COLS):
            v = np.clip(np.rint(img[i] * 17), 0, 255).astype(np.uint8)
            t = np.repeat(np.repeat(v, TILE // 8, 0), TILE // 8, 1)
            cv.paste(Image.fromarray(t), (LW + i * (TILE + PAD), y))
        y += TILE + 30
    p = os.path.join(OUT, "cmp.png")
    cv.save(p)
    print(f"wrote {os.path.relpath(p, REPO)}")
    print("COLOURISE_DONE")


if __name__ == "__main__":
    main()

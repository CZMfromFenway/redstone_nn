#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
两阶段·第二版：给灰度，直接预测「16 色吸附后的索引」——目标是最贴近原彩色头像吸附后的结果。

与上一版的区别（这是作者 2026-11-02 指定的做法）：
    上一版：灰度 → 连续 RGB，损失是逐通道 MSE（等于让模型自己发明颜色）
    这一版：灰度 → 32×16 logits → argmax → 吸附后的调色板；训练目标 = **原彩色像素最近的
           吸附调色板索引**。也就是"尽量复现原图的 16 色版本"，而不是"贴一层色相/饱和度"。

灰度的来源也有讲究：**用二期最终那台 PCA4 灰度模型（4→64→32，2,304 乘法）现场生成的灰度**
（min-max 拉伸后量化到 16 级），这样训练与部署的输入分布一致。

另做一臂对照：同样的生成灰度 → 连续 RGB（上一版结构，但输入换成生成灰度），
以及 32 个灰度级喂满的情况。

产物：results/twostage/{gray_pca4.json, pal16.json, rgb96.json, summary.json}
用法：python train_two_stage.py [--steps 12000]
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
REPO = os.path.dirname(P4)
P2 = os.path.join(REPO, "02_phase2_avatar")
OUT = os.path.join(P4, "results", "twostage")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(P4, "src"))

ROW = np.array([p % 8 for p in range(32)])
COL = np.array([p // 8 for p in range(32)])


def load_layers(j, key):
    Ls = j[key]["layers"]
    out = []
    for i, L in enumerate(Ls):
        out.append((np.array(L["W"], dtype=np.float64), np.array(L["b"], dtype=np.float64),
                    i < len(Ls) - 1))
    return out


def fwd_np(layers, x):
    h = np.asarray(x, dtype=np.float64)
    for W, b, relu in layers:
        h = h @ W.T + b
        if relu:
            h = np.maximum(h, 0)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    A = ap.parse_args()
    torch.manual_seed(A.seed)

    import train_palette as tp
    rgb96, cpc, tr, va = tp.load_cache(8, "ext32")
    N = len(rgb96)
    POS = rgb96.reshape(N, 32, 3)

    # ---- 1. 用二期 PCA4 灰度模型现场生成灰度
    gp = os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json")
    gj = json.load(open(gp, encoding="utf-8"))
    g_lay = load_layers(gj, "gen")
    full = np.load(os.path.join(P2, "data_v2_canonical", "L0_full_idx.npy")).astype(np.float64)
    X = full.reshape(len(full), -1)
    mu = X.mean(0, keepdims=True)
    sub = np.random.default_rng(0).choice(len(X), 60000, replace=False)
    _, _, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
    comps = Vt[:22]
    sd = ((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6
    ZZ = ((X - mu) @ comps.T) / sd[0]
    print("生成灰度：PCA4 4→64→32（二期终版，2,304 乘法）")
    raw = np.clip(fwd_np(g_lay, ZZ[:, :4]), 0, 15)
    lo = raw.min(1, keepdims=True); hi = raw.max(1, keepdims=True)
    gen_idx = np.clip(np.rint((raw - lo) / np.maximum(hi - lo, 1e-6) * 15), 0, 15)
    gen_gray = (gen_idx / 15.0).astype(np.float32)
    # 真值灰度（同一行的亮度量化），只作对照
    true_gray = (full[:, ROW, COL] / 15.0).astype(np.float32)
    print(f"  生成灰度的平均极差 {float((gen_idx.max(1)-gen_idx.min(1)).mean()):.2f} 级；"
          f"与真值灰度的平均差 {float(np.abs(gen_gray-true_gray).mean())*15:.2f} 级")

    # ---- 2. 16 色调色板（吸附到 61 个 HIGH，撞色接受）
    dep = json.load(open(os.path.join(P4, "results", "fit", "deploy_palette.json"),
                         encoding="utf-8"))["K16_w128_k8"]
    pal = np.array(dep["palette"], dtype=np.float64)          # (16,3) 0..15
    names = dep["names"]
    print(f"调色板 16 槽 / {dep['distinct_colours']} 个不同颜色；吸附误差 "
          f"{dep['snap_err_mean']} 级")

    # ---- 3. 目标：原彩色像素最近的调色板索引
    d = ((POS[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(-1)
    Yidx = d.argmin(-1)                                        # (N,32) 0..15
    floor = pal[Yidx]                                          # 该调色板下的最好结果
    print(f"地板（真值颜色→调色板）：RGB MAE {float(np.abs(floor-POS).mean()):.2f} 级")

    # 保存灰度模型给页面用（归一化成一个统一格式）
    json.dump({"layers": [{"W": W.tolist(), "b": b.tolist()} for W, b, _ in g_lay],
               "meta": {"name": "phase2_pca4_gray", "k": 4, "post": "stretch_round16",
                        "mults": 2304}},
              open(os.path.join(OUT, "gray_pca4.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    results = {}

    # ---- 4a. 灰度 → 16 色索引（作者要的那一版）
    def train_pal_head(gray_src, tag):
        net = nn.Sequential(nn.Linear(32, A.hid), nn.ReLU(), nn.Linear(A.hid, A.hid), nn.ReLU(),
                            nn.Linear(A.hid, 32 * 16))
        opt = torch.optim.Adam(net.parameters(), lr=A.lr)
        mults = 32 * A.hid + A.hid * A.hid + A.hid * 32 * 16
        print(f"\n[{tag}] 32 → {A.hid} → {A.hid} → 32×16；乘法 {mults:,}；步数 {A.steps}")
        rng = np.random.default_rng(A.seed)
        Yt = torch.tensor(Yidx, dtype=torch.long)
        G = torch.tensor(gray_src, dtype=torch.float32)
        t0 = time.time()
        for step in range(1, A.steps + 1):
            bi = torch.tensor(rng.integers(0, len(tr), A.batch))
            lg = net(G[bi]).reshape(-1, 32, 16)
            loss = F.cross_entropy(lg.reshape(-1, 16), Yt[bi].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 4000 == 0 or step == A.steps:
                with torch.no_grad():
                    lgv = net(G[va[:1000]]).reshape(-1, 32, 16)
                    pred = lgv.argmax(-1).numpy()
                acc = float((pred == Yidx[va[:1000]]).mean())
                mae = float(np.abs(pal[pred] - POS[va[:1000]]).mean())
                print(f"  step {step:6d}  CE {float(loss.item()):.3f}  "
                      f"索引命中 {acc*100:5.1f}%  RGB MAE {mae:.2f} 级  ({time.time()-t0:.0f}s)",
                      flush=True)
        return net, {"mults": int(mults), "ce": round(float(loss.item()), 4),
                     "idx_acc": round(acc, 4), "rgb_mae": round(mae, 3)}

    net_pal, sc_pal = train_pal_head(gen_gray, "两阶段·16 色（生成灰度）")
    results["two_stage_pal16"] = sc_pal
    Ls = [m for m in net_pal if isinstance(m, nn.Linear)]
    mdl = {"type": "palette", "body": [], "head": {}, "palette": np.round(pal, 5).tolist(),
           "meta": {"k": 32, "input": "gray32 (from PCA4 generator)", "K": 16,
                    "hid": A.hid, "mults": sc_pal["mults"], "bits_per_pixel": 4,
                    "note": "两阶段：灰度→16 色索引（目标＝原彩色头像吸附后的索引）"}}
    for L in Ls[:-1]:
        mdl["body"].append({"W": L.weight.detach().numpy().round(7).tolist(),
                            "b": L.bias.detach().numpy().round(7).tolist()})
    mdl["head"] = {"W": Ls[-1].weight.detach().numpy().round(7).tolist(),
                   "b": Ls[-1].bias.detach().numpy().round(7).tolist()}
    json.dump(mdl, open(os.path.join(OUT, "pal16.json"), "w", encoding="utf-8"), ensure_ascii=False)

    # 对照：同一结构，但输入用真值灰度
    net_pal_t, sc_pal_t = train_pal_head(true_gray, "两阶段·16 色（真值灰度，对照）")
    results["two_stage_pal16_truegray"] = sc_pal_t

    # ---- 4b. 灰度 → 连续 RGB（上一版结构，输入换成生成灰度）
    net_rgb = nn.Sequential(nn.Linear(32, A.hid), nn.ReLU(), nn.Linear(A.hid, A.hid), nn.ReLU(),
                            nn.Linear(A.hid, 96))
    opt = torch.optim.Adam(net_rgb.parameters(), lr=A.lr)
    mults_rgb = 32 * A.hid + A.hid * A.hid + A.hid * 96
    print(f"\n[两阶段·连续 RGB（生成灰度）] 32 → {A.hid} → {A.hid} → 96；乘法 {mults_rgb:,}")
    Yn = torch.tensor(rgb96 / 15.0, dtype=torch.float32)
    G = torch.tensor(gen_gray, dtype=torch.float32)
    rng = np.random.default_rng(A.seed)
    t0 = time.time()
    for step in range(1, A.steps + 1):
        bi = torch.tensor(rng.integers(0, len(tr), A.batch))
        loss = F.mse_loss(net_rgb(G[bi]), Yn[bi])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 4000 == 0 or step == A.steps:
            with torch.no_grad():
                pv = np.clip(net_rgb(G[va[:1000]]).numpy() * 15, 0, 15).reshape(-1, 32, 3)
            mae = float(np.abs(pv - POS[va[:1000]]).mean())
            print(f"  step {step:6d}  loss {float(loss.item()):.4f}  RGB MAE {mae:.2f} 级  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    results["two_stage_rgb96"] = {"mults": int(mults_rgb), "rgb_mae": round(mae, 3)}
    Ls2 = [m for m in net_rgb if isinstance(m, nn.Linear)]
    mdl2 = {"type": "rgb96", "body": [], "head": {},
            "meta": {"k": 32, "input": "gray32", "hid": A.hid, "mults": int(mults_rgb),
                     "output_mode": "rgb96", "note": "两阶段：灰度→连续 RGB"}}
    for L in Ls2[:-1]:
        mdl2["body"].append({"W": L.weight.detach().numpy().round(7).tolist(),
                             "b": L.bias.detach().numpy().round(7).tolist()})
    mdl2["head"] = {"W": Ls2[-1].weight.detach().numpy().round(7).tolist(),
                    "b": Ls2[-1].bias.detach().numpy().round(7).tolist()}
    json.dump(mdl2, open(os.path.join(OUT, "rgb96.json"), "w", encoding="utf-8"), ensure_ascii=False)

    summary = {"palette": {"names": names, "distinct": dep["distinct_colours"],
                           "snap_err_mean": dep["snap_err_mean"]},
               "floor_rgb_mae": round(float(np.abs(floor - POS).mean()), 3),
               "results": results,
               "cost": {"gray_pca4": 2304,
                        "pal16_head": results["two_stage_pal16"]["mults"],
                        "rgb96_head": results["two_stage_rgb96"]["mults"],
                        "total_pal16": 2304 + results["two_stage_pal16"]["mults"],
                        "total_rgb96": 2304 + results["two_stage_rgb96"]["mults"],
                        "single_stage_pal16": 82944}}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\n== 汇总 ==")
    print(f"  地板（真值颜色→调色板）RGB MAE {summary['floor_rgb_mae']:.2f} 级")
    for k, v in results.items():
        print(f"  {k:26s} {json.dumps(v, ensure_ascii=False)}")
    print(f"  成本：灰度 2,304 + 16 色头 {results['two_stage_pal16']['mults']:,} = "
          f"{summary['cost']['total_pal16']:,}（单阶段 16 色是 82,944）")
    print("TWOSTAGE_DONE")


if __name__ == "__main__":
    main()

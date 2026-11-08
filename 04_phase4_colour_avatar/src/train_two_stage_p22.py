#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
两阶段·第二版（续）：灰度来源换成更强的 PCA22 灰度，再训「灰度 → 16 色吸附索引」。

对比两条灰度来源（都不训练，直接取二期已导出的模型）：
    pca4   二期终版 4→64→32（2,304 乘法，acc 0.3651）—— 第四轮已测
    pca22  22→128→128→32（23,296 乘法，acc 0.7392）—— 本条新增

颜色头同一结构：32 → 128 → 128 → 32×16（86,016 乘法），
训练目标 = 原彩色头像吸附到 16 色（61 HIGH）后的索引。

产物：results/twostage/{gray_pca22.json, pal16_p22.json, summary_p22.json}
用法：python train_two_stage_p22.py [--steps 12000]
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


def load_layers(path, key="gen"):
    j = json.load(open(path, encoding="utf-8"))
    Ls = j[key]["layers"]
    lay = []
    for i, L in enumerate(Ls):
        lay.append((np.array(L["W"], dtype=np.float64), np.array(L["b"], dtype=np.float64),
                    i < len(Ls) - 1))
    return j, lay


def fwd(lay, x):
    h = np.asarray(x, dtype=np.float64)
    for W, b, relu in lay:
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

    # 22 维灰度坐标（与二期 dim_sweep 同一套基）
    full = np.load(os.path.join(P2, "data_v2_canonical", "L0_full_idx.npy")).astype(np.float64)
    X = full.reshape(len(full), -1)
    mu = X.mean(0, keepdims=True)
    sub = np.random.default_rng(0).choice(len(X), 60000, replace=False)
    _, _, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
    comps = Vt[:22]
    sd = ((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6
    ZZ = ((X - mu) @ comps.T) / sd[0]
    true_gray = (full[:, ROW, COL] / 15.0).astype(np.float32)

    gp = os.path.join(P2, "webapp", "pca22_w128_128_model.json")
    gj, gl = load_layers(gp)
    print(f"PCA22 灰度：{gj['meta'].get('label', '22->128->128->32')}，"
          f"乘法 {gj['meta'].get('multiplies', '?')}")
    raw = np.clip(fwd(gl, ZZ[:, :22]), 0, 15)
    lo = raw.min(1, keepdims=True); hi = raw.max(1, keepdims=True)
    g_idx = np.clip(np.rint((raw - lo) / np.maximum(hi - lo, 1e-6) * 15), 0, 15)
    g22 = (g_idx / 15.0).astype(np.float32)

    # 与 PCA4 灰度、真值灰度对比
    gp4 = os.path.join(P2, "webapp", "pca22_dim4w64L1_model.json")
    _, g4l = load_layers(gp4)
    raw4 = np.clip(fwd(g4l, ZZ[:, :4]), 0, 15)
    lo4 = raw4.min(1, keepdims=True); hi4 = raw4.max(1, keepdims=True)
    g4 = (np.clip(np.rint((raw4 - lo4) / np.maximum(hi4 - lo4, 1e-6) * 15), 0, 15) / 15.0).astype(np.float32)
    print(f"  灰度与真值的平均绝对差：PCA4 {float(np.abs(g4-true_gray).mean())*15:.2f} 级，"
          f"PCA22 {float(np.abs(g22-true_gray).mean())*15:.2f} 级")

    dep = json.load(open(os.path.join(P4, "results", "fit", "deploy_palette.json"),
                         encoding="utf-8"))["K16_w128_k8"]
    pal = np.array(dep["palette"], dtype=np.float64)
    d = ((POS[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(-1)
    Yidx = d.argmin(-1)
    floor = pal[Yidx]

    json.dump({"layers": [{"W": W.tolist(), "b": b.tolist()} for W, b, _ in gl],
               "meta": {"name": "phase2_pca22_gray", "k": 22, "post": "stretch_round16",
                        "mults": int(gj["meta"].get("multiplies", 23296))}},
              open(os.path.join(OUT, "gray_pca22.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    net = nn.Sequential(nn.Linear(32, A.hid), nn.ReLU(), nn.Linear(A.hid, A.hid), nn.ReLU(),
                        nn.Linear(A.hid, 32 * 16))
    opt = torch.optim.Adam(net.parameters(), lr=A.lr)
    mults = 32 * A.hid + A.hid * A.hid + A.hid * 32 * 16
    print(f"[两阶段·16 色（PCA22 灰度）] 32 → {A.hid} → {A.hid} → 32×16；乘法 {mults:,}")
    rng = np.random.default_rng(A.seed)
    Yt = torch.tensor(Yidx, dtype=torch.long)
    G = torch.tensor(g22, dtype=torch.float32)
    t0 = time.time()
    for step in range(1, A.steps + 1):
        bi = torch.tensor(rng.integers(0, len(tr), A.batch))
        lg = net(G[bi]).reshape(-1, 32, 16)
        loss = F.cross_entropy(lg.reshape(-1, 16), Yt[bi].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 4000 == 0 or step == A.steps:
            with torch.no_grad():
                pred = net(G[va[:1000]]).reshape(-1, 32, 16).argmax(-1).numpy()
            acc = float((pred == Yidx[va[:1000]]).mean())
            mae = float(np.abs(pal[pred] - POS[va[:1000]]).mean())
            print(f"  step {step:6d}  CE {float(loss.item()):.3f}  索引命中 {acc*100:5.1f}%  "
                  f"RGB MAE {mae:.2f} 级  ({time.time()-t0:.0f}s)", flush=True)

    Ls = [m for m in net if isinstance(m, nn.Linear)]
    mdl = {"type": "palette", "body": [], "head": {}, "palette": np.round(pal, 5).tolist(),
           "meta": {"k": 32, "input": "gray32 (from PCA22 generator)", "K": 16,
                    "hid": A.hid, "mults": int(mults), "bits_per_pixel": 4,
                    "note": "两阶段：PCA22 灰度→16 色索引（目标＝原彩图吸附后的索引）"}}
    for L in Ls[:-1]:
        mdl["body"].append({"W": L.weight.detach().numpy().round(7).tolist(),
                            "b": L.bias.detach().numpy().round(7).tolist()})
    mdl["head"] = {"W": Ls[-1].weight.detach().numpy().round(7).tolist(),
                   "b": Ls[-1].bias.detach().numpy().round(7).tolist()}
    json.dump(mdl, open(os.path.join(OUT, "pal16_p22.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    summary = {"gray": {"pca22_vs_true_lv": round(float(np.abs(g22 - true_gray).mean()) * 15, 2),
                        "pca4_vs_true_lv": round(float(np.abs(g4 - true_gray).mean()) * 15, 2),
                        "pca22_mults": int(gj["meta"].get("multiplies", 23296)),
                        "pca4_mults": 2304},
               "head": {"mults": int(mults), "idx_acc": round(acc, 4),
                        "rgb_mae": round(mae, 3), "ce": round(float(loss.item()), 4)},
               "floor_rgb_mae": round(float(np.abs(floor - POS).mean()), 3),
               "total_pca22": int(gj["meta"].get("multiplies", 23296)) + int(mults),
               "total_pca4": 2304 + int(mults),
               "single_stage_pal16": 82944}
    json.dump(summary, open(os.path.join(OUT, "summary_p22.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\n地板 {summary['floor_rgb_mae']:.2f} 级；索引命中 {acc*100:.1f}%，"
          f"RGB MAE {mae:.2f} 级；总成本 {summary['total_pca22']:,}（PCA4 版 "
          f"{summary['total_pca4']:,}，单阶段 {summary['single_stage_pal16']:,}）")
    print("TWOSTAGE_P22_DONE")


if __name__ == "__main__":
    main()

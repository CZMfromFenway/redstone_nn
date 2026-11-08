#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
四期：PCA → 16 色 的训练入口（朴素方案 E1）。

这是二期 `colour_palette.py` 的四期版：唯一的结构性改动是**输入维数可调**（k=8/16/24/32），
并把「调色板从哪来」拆成三个选项，因为二期已经证明「事前约束 vs 事后吸附」不是小事：

    --palette free      现场 k-means（默认，可叠 --sat 1.2）
    --palette-json F    读一个固定的 16 色调色板（例如吸附到 244 个地图色之后的）
    --palette medoid    在候选色集合（默认 244 个地图色）上做 k-medoids，调色板本身就是物理色

数据（只读）：
    data_v2_canonical/L0_split.npz          train / val 行号
    results/colour/cache.npz                rgb96 (N,96,0..15)、cpc (N,24,标准化)

产物：
    results/train/<tag>/model.json          结构与二期一致：{type, body, head, palette, meta}
    results/train/<tag>/summary.json        训练曲线 + 验证集指标 + 调色板 + 成本

用法：
    python train_palette.py --k 8  --K 16 --hid 128 --steps 12000 --tag k8_K16_w128
    python train_palette.py --k 16 --K 16 --hid 128 --steps 12000 --tag k16_K16_w128
    python train_palette.py --k 16 --K 16 --hid 128 --steps 300   --tag smoke   # 冒烟

注意：本脚本只写 `04_phase4_colour_avatar/results/`，不碰二期目录。
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


def extended_cpc(k, rgb192, cached, ncomp=32):
    """缓存里只有 24 维系数；k>24 时把基扩到 32 维。

    只在 npz 里存**基**（均值 + 分量 + 每维尺度，约 50 KB），系数用时现算，
    这样不必把 321k x 32 的系数矩阵（38 MB）塞进仓库。
    前 min(24, ncomp) 列按符号对齐到二期缓存系数上：SVD 的符号是任意的，
    不对齐会让"同一个旋钮"在不同臂里反号。
    """
    cache_path = os.path.join(P4, "results", "prep", "cpc32_basis.npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        if z["comps"].shape[0] >= k:
            comps, mu, sd = z["comps"][:k], z["mu"].reshape(1, -1), z["sd"].ravel()[:k]
            return ((rgb192 - mu) @ comps.T / sd).astype(np.float64)
    mu = rgb192.mean(0, keepdims=True)
    sub = np.random.default_rng(0).choice(len(rgb192), 40000, replace=False)
    _, _, Vt = np.linalg.svd(rgb192[sub] - mu, full_matrices=False)
    comps = Vt[:ncomp]
    coef = (rgb192 - mu) @ comps.T
    sd = coef.std(0, keepdims=True) + 1e-9
    cpc = coef / sd
    m = min(cached.shape[1], ncomp)
    for j in range(m):
        if np.corrcoef(cpc[:, j], cached[:, j])[0, 1] < 0:
            comps[j] *= -1.0
            cpc[:, j] *= -1.0
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, comps=comps, mu=mu.ravel(), sd=sd.ravel())
    return cpc[:, :k]



def load_cache(k=None, basis="ext32"):
    z = np.load(os.path.join(P2, "results", "colour", "cache.npz"))
    rgb96 = z["rgb96"].astype(np.float64)          # (N,96) 0..15
    cpc = z["cpc"].astype(np.float64)              # (N,24) 标准化后的颜色 PCA 系数
    if k is not None and basis == "ext32":
        rgb192 = z["rgb192"].astype(np.float64)
        print(f"  basis=ext32：32 维重算基的前 {k} 列（前 {cpc.shape[1]} 列符号对齐二期缓存）")
        cpc = extended_cpc(k, rgb192, cpc)
    sp = np.load(os.path.join(P2, "data_v2_canonical", "L0_split.npz"))
    return rgb96, cpc, sp["train"], sp["val"]


def load_map_candidates():
    j = json.load(open(os.path.join(P2, "data", "mc_map_colors.json"), encoding="utf-8"))
    mods = j["_modifiers"]
    base = np.array([b["rgb"] for b in j["bases"]], dtype=np.float64)
    cols, tags = [], []
    for m in ("HIGH", "NORMAL", "LOW", "LOWEST"):
        cols.append(np.clip(base * mods[m], 0, 255) / 17.0)
        tags += [(b["name"], m) for b in j["bases"]]
    return np.concatenate(cols), tags


def kmeans(X, k, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(k - 1):
        i = rng.choice(len(X), p=d2 / d2.sum())
        C.append(X[i])
        d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(1))
    C = np.array(C)
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        new = C.copy()
        for j in range(k):
            m = lab == j
            if m.any():
                new[j] = X[m].mean(0)
        if np.allclose(new, C, atol=1e-6):
            break
        C = new
    return C, lab


def kmedoids(cands, X, k, iters=40, seed=0):
    """medoid 只能取自 cands：调色板天生是物理可达色。"""
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), min(60000, len(X)), replace=False)]
    sel = rng.choice(len(cands), k, replace=False)
    for _ in range(iters):
        d = ((sub[:, None, :] - cands[sel][None]) ** 2).sum(-1)
        lab = d.argmin(1)
        new = sel.copy()
        for j in range(k):
            m = lab == j
            if not m.any():
                continue
            mu = sub[m].mean(0)
            new[j] = int(np.argmin(((cands - mu) ** 2).sum(-1)))
        if np.array_equal(np.sort(new), np.sort(sel)):
            break
        sel = new
    return cands[sel], sel


def make_palette(args, rgb96):
    K = args.K
    if args.palette == "free":
        rng = np.random.default_rng(3)
        sel = rng.choice(len(rgb96), min(40000, len(rgb96)), replace=False)
        P = rgb96[sel].reshape(-1, 3)
        C, _ = kmeans(P, K, seed=1)
        if args.sat != 1.0:
            g = C.mean(1, keepdims=True)
            C = np.clip(g + args.sat * (C - g), 0, 15)
        return C, {"source": "kmeans", "sat": args.sat}
    if args.palette == "medoid":
        cands, tags = load_map_candidates()
        rng = np.random.default_rng(3)
        sel = rng.choice(len(rgb96), min(40000, len(rgb96)), replace=False)
        P = rgb96[sel].reshape(-1, 3)
        C, idx = kmedoids(cands, P, K, seed=1)
        return C, {"source": "kmedoids", "candidates": len(cands),
                   "names": [tags[i] for i in idx]}
    pal = json.load(open(args.palette_json, encoding="utf-8"))
    C = np.array(pal["palette"] if isinstance(pal, dict) and "palette" in pal else pal,
                 dtype=np.float64)
    if C.ndim == 2 and C.shape[0] != K and C.shape[1] == K:
        C = C.T
    return C[:K], {"source": "json", "file": os.path.basename(args.palette_json)}


def build_model(k, hid, K, layers, seed=0):
    torch.manual_seed(seed)
    mods = [nn.Linear(k, hid), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hid, hid), nn.ReLU()]
    mods += [nn.Linear(hid, 32 * K)]
    return nn.Sequential(*mods)


def metrics(img, pal_idx):
    """img: (B,8,8,3) 0..15；返回与人无关的三个量。"""
    a = np.asarray(img, dtype=np.float64)
    flat = a.reshape(len(a), 64, 3)
    mx, mn = flat.max(-1), flat.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0).mean()
    ncol = np.mean([len({tuple(np.rint(c).astype(int)) for c in flat[i]}) for i in range(len(a))])
    dh = np.linalg.norm(a[:, :, 1:] - a[:, :, :-1], axis=-1)
    dv = np.linalg.norm(a[:, 1:] - a[:, :-1], axis=-1)
    ed = np.concatenate([dh.reshape(len(a), -1), dv.reshape(len(a), -1)], 1)
    return {"sat": round(float(sat), 4), "ncol": round(float(ncol), 3),
            "edge_p90": round(float(np.percentile(ed, 90)), 3),
            "lum": round(float((flat @ np.array([0.299, 0.587, 0.114])).mean()), 3)}


def mirror(idx32):
    """(B,32) 索引或 (B,32,3) 颜色 -> (B,8,8) / (B,8,8,3)，行内镜像。"""
    v = np.asarray(idx32)
    rows = np.array([p % 8 for p in range(32)])
    cols = np.array([p // 8 for p in range(32)])
    if v.ndim == 2:
        img = np.zeros((len(v), 8, 8))
        img[:, rows, cols] = v
        img[:, rows, 7 - cols] = v
        return img
    img = np.zeros((len(v), 8, 8, v.shape[-1]))
    img[:, rows, cols] = v
    img[:, rows, 7 - cols] = v
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="输入 PCA 维数")
    ap.add_argument("--K", type=int, default=16, help="调色板颜色数")
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2, help="隐层数（128 是红石上限）")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--palette", default="free", choices=["free", "medoid", "json"])
    ap.add_argument("--palette-json", default="")
    ap.add_argument("--sat", type=float, default=1.2, help="free 调色板的饱和度倍数，1.0=不放大")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--keep", default="", help="keep_index.npz 路径：只用其中 keep=True 的样本训练")
    ap.add_argument("--basis", default="ext32", choices=["ext32", "cache"],
                    help="ext32=统一用 32 维重算基的前 k 列（跨臂可比）；cache=用二期的 24 维缓存")
    ap.add_argument("--tag", required=True)
    A = ap.parse_args()

    if A.threads:
        torch.set_num_threads(A.threads)
    torch.manual_seed(A.seed)

    out = os.path.join(P4, "results", "train", A.tag)
    os.makedirs(out, exist_ok=True)

    rgb96, cpc, tr, va = load_cache(A.k, A.basis)
    if A.keep:
        keep = np.load(A.keep)["keep"]
        n0 = len(tr)
        tr = tr[keep[tr]]
        print(f"  清洗：训练集 {n0:,} → {len(tr):,}（剔除 {n0-len(tr):,}）")
        va = va[keep[va]]
        print(f"        验证集 → {len(va):,}（同样只留干净样本，便于比较）")
    K = A.K
    pal, pal_meta = make_palette(A, rgb96)
    K = len(pal)
    POS = rgb96.reshape(len(rgb96), 32, 3)
    Y = ((POS[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(-1).argmin(-1)   # (N,32)
    Xall = cpc[:, :A.k]
    print(f"tag={A.tag}  k={A.k}  K={K}  hid={A.hid}  layers={A.layers}  steps={A.steps}")
    print(f"  palette={pal_meta['source']}  train={len(tr):,}  val={len(va):,}")
    print(f"  理论上限（调色板量化误差）见 summary.json 的 pal_quant_err")

    Xt = torch.tensor(Xall[tr], dtype=torch.float32)
    Yt = torch.tensor(Y[tr], dtype=torch.long)
    net = build_model(A.k, A.hid, K, A.layers, A.seed)
    opt = torch.optim.Adam(net.parameters(), lr=A.lr)
    nparam = sum(p.numel() for p in net.parameters())
    mults = A.k * A.hid + (A.layers - 1) * A.hid * A.hid + A.hid * 32 * K
    print(f"  参数 {nparam:,}  乘法 {mults:,}  输入随机源 {A.k} 条")

    rng = np.random.default_rng(A.seed)
    t0 = time.time()
    hist = []
    for step in range(1, A.steps + 1):
        bi = tr[rng.integers(0, len(tr), A.batch)]
        lg = net(torch.tensor(Xall[bi], dtype=torch.float32)).view(-1, 32, K)
        loss = F.cross_entropy(lg.reshape(-1, K), torch.tensor(Y[bi], dtype=torch.long).reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % A.eval_every == 0 or step == A.steps:
            with torch.no_grad():
                bv = va[:2000]
                lg = net(torch.tensor(Xall[bv], dtype=torch.float32)).view(-1, 32, K)
                pred = lg.argmax(-1).numpy()
                acc = float((pred == Y[bv]).mean())
                img = mirror(pal[pred])
                eqh = (img[:, :, :-1] == img[:, :, 1:])
                eqv = (img[:, :-1, :] == img[:, 1:, :])
                if eqh.ndim == 4:                 # 彩色：三个通道全等才算同色
                    eqh = eqh.all(-1)
                    eqv = eqv.all(-1)
                h = int(eqh.sum())
                v2 = int(eqv.sum())
                agree = (h + v2) / (len(img) * 8 * 7 * 2)
                m = metrics(img, pred)
            rec = {"step": step, "train_ce": round(float(loss.item()), 4),
                   "val_pos_acc": round(acc, 4), "val_agree": round(agree, 4),
                   "sec": round(time.time() - t0, 1), **m}
            hist.append(rec)
            print(f"  step {step:6d}  ce {loss:6.4f}  acc {acc:.4f}  同色率 {agree*100:5.1f}%  "
                  f"色数 {m['ncol']:.2f}  边缘 {m['edge_p90']:.2f}  ({rec['sec']:.0f}s)", flush=True)

    # 导出（与二期 webapp 的 palette 模型同构）
    body = []
    Ls = [m for m in net if isinstance(m, nn.Linear)]
    for L in Ls[:-1]:
        body.append({"W": L.weight.detach().numpy().round(7).tolist(),
                     "b": L.bias.detach().numpy().round(7).tolist()})
    head = {"W": Ls[-1].weight.detach().numpy().round(7).tolist(),
            "b": Ls[-1].bias.detach().numpy().round(7).tolist()}
    model = {"type": "palette", "K": K, "body": body, "head": head,
             "palette": np.round(pal, 5).tolist(),
             "meta": {"k": A.k, "K": K, "hid": A.hid, "body_layers": len(body),
                      "mults": int(mults), "bits_per_pixel": int(np.ceil(np.log2(K))),
                      "steps": A.steps, "batch": A.batch, "lr": A.lr, "seed": A.seed,
                      "palette_source": pal_meta,
                      "usage": "forward(k inputs) -> (32,K) logits -> argmax -> palette -> 行内镜像",
                      "note": f"phase4 朴素 PCA→{K} 色：k={A.k}, {A.hid}x{A.layers}"}}
    json.dump(model, open(os.path.join(out, "model.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    pe = ((POS[va[:2000]][:, :, None, :] - pal[None, None, :, :]) ** 2).sum(-1)
    pe = np.sqrt(pe.min(-1))
    summary = {"tag": A.tag, "args": vars(A), "history": hist,
               "final": hist[-1] if hist else None, "multiplies": int(mults),
               "params": int(nparam), "palette": np.round(pal, 5).tolist(),
               "palette_meta": pal_meta,
               "pal_quant_err": {"mean": round(float(pe.mean()), 3),
                                 "p95": round(float(np.percentile(pe, 95)), 3)},
               "baseline_note": "二期 k8snap：round11 57.3%(n=103)、round12 61.1%(n=108)"}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote {os.path.relpath(out, REPO)}/model.json + summary.json")
    print("PHASE4_TRAIN_DONE")


if __name__ == "__main__":
    main()

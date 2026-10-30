#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Phase-2 progress sheet, redrawn as the user asked.

WHAT CHANGED FROM THE FIRST VERSION
1. "由随机的参数生成（逐模型随机）"
   The first sheet fed every model the SAME 32 condition vectors, so a row measured "how
   well does this model render these particular inputs". The user wants instead each model
   sampled from its OWN random draw, so a row is a sample of that model's output
   DISTRIBUTION. Rows are then compared as textures, not face by face.

2. "每个模型的32个头像聚在一起"
   Each model is now a compact 8x4 BLOCK of 32 faces rather than a long 32-wide strip, so
   one model reads as one unit and four models sit side by side.

3. "最新模型的纯化过的随机也要用上"
   The last block is the w128 model sampled through the reward model fitted on the user's
   269 labels, keeping the top 15% of a 600-candidate pool. That is what the preference
   work actually produces.

Every block also carries its measured neighbour agreement and accuracy so the visual
impression has numbers beside it.
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import importlib.util as _ilu
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_lspec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_lspec); _lspec.loader.exec_module(layout)
NLEV = 16
LEVELS = np.linspace(0, 255, 16)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "PROGRESS")
os.makedirs(OUT, exist_ok=True)
SEED = 20261018
PER_MODEL = 32


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


# ---------------------------------------------------------------- shared data
rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.clip(np.ceil(gray / 17.0 - 0.5), 0, 15).astype(np.int64)
kn = (gyi[:, layout.POS_ROW, layout.POS_COL] ==
      gyi[:, layout.POS_ROW, layout.MIRROR_COL]).mean(1) >= 0.95
Gg = gray[kn]
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy"))
va = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"]
X = Gg.reshape(len(Gg), -1)
mu = X.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(len(X), min(60000, len(X)), replace=False)
U_, S_, Vt = np.linalg.svd(X[sub] - mu, full_matrices=False)
comps = Vt[:22]
sd = (((X - mu) @ comps.T).std(0, keepdims=True) + 1e-6)
bv = va[:PER_MODEL]
HALF_OH = np.zeros((PER_MODEL, 32 * NLEV), dtype=np.float32)
for k in range(PER_MODEL):
    for p in range(32):
        HALF_OH[k, p * NLEV + int(full[bv[k]][layout.POS_ROW[p], layout.POS_COL[p]])] = 1.0
RTRUE = full[bv][:, layout.POS_ROW, layout.MIRROR_COL]

# the reward model, for the purified block
REW = None
rp = os.path.join(ROOT, "webapp", "preference_reward.json")
if os.path.exists(rp):
    REW = json.load(open(rp, encoding="utf-8"))


def load_net(path):
    ck = torch.load(path, weights_only=False)
    st = ck.get("G") or ck.get("S")
    norm = {(k.replace("net.", "", 1) if k.startswith("net.") else k): v
            for k, v in st.items()}
    idx = sorted({int(k.split(".")[0]) for k in norm if k.endswith(".weight")})
    seq, dims = [], []
    for j in idx:
        W = norm[f"{j}.weight"]
        seq.append(nn.Linear(int(W.shape[1]), int(W.shape[0])))
        dims.append((int(W.shape[1]), int(W.shape[0])))
        if j != idx[-1]:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq)
    m.load_state_dict(norm)
    m.eval()
    return m, dims[0][0], dims[-1][1], sum(a * b for a, b in dims), ck.get("args", {})


def condition_for(in_dim, kind, rng):
    """Build 32 inputs from THIS model's own random draw."""
    z = np.clip(rng.standard_normal((PER_MODEL, 22)), -3.5, 3.5)
    noise = np.clip(rng.standard_normal((PER_MODEL, 64)), -3.5, 3.5)
    if kind == "uncond":
        return noise.astype(np.float32)
    if kind == "half":
        return np.concatenate([HALF_OH, noise], 1).astype(np.float32), z
    if in_dim == 22:
        return z.astype(np.float32), z
    if in_dim - 22 == 64:
        return np.concatenate([z, noise], 1).astype(np.float32), z
    if in_dim == 68:
        return np.concatenate([z[:, :4], noise], 1).astype(np.float32), z
    if in_dim == 72:
        return np.concatenate([z[:, :8], noise], 1).astype(np.float32), z
    if in_dim == 80 and kind == "code":
        oh = np.zeros((PER_MODEL, 16), dtype=np.float32)
        for k in range(PER_MODEL):
            oh[k, k % 16] = 1.0
        return np.concatenate([oh, noise], 1).astype(np.float32), z
    if in_dim == 80:
        return np.concatenate([z[:, :16], noise], 1).astype(np.float32), z
    raise ValueError(f"in_dim={in_dim} kind={kind}")


def agree_of(imgs):
    out = []
    for g in imgs:
        pairs = same = 0
        for r_ in range(8):
            for c in range(8):
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    rr, cc = r_ + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8:
                        pairs += 1
                        if g[r_, cc] == g[rr, cc]:
                            same += 1
        out.append(same / pairs)
    return float(np.mean(out))


def render(net, cond, out_dim):
    with torch.no_grad():
        raw = net(torch.tensor(cond, dtype=torch.float32))
    if out_dim == 512:
        dec = raw.view(-1, 32, NLEV).argmax(-1).numpy()
    else:
        dec = raw.clamp(0, 15).round().numpy().astype(int)
    return layout.image_from_left_half(dec), dec


ACC = {"G": 0.4779, "H": 0.6429, "I": 0.6683, "N": 0.7005, "O": 0.7348, "P": 0.7719,
       "Q": 0.7138, "R": 0.7392, "S": 0.7057, "T": 0.6791}

MILESTONES = [
    ("sweep/sweep_uncond/final.pt",        "A", "无条件 GAN", "uncond", "失败"),
    ("sweep/sweep_half/final.pt",          "B", "半边真实条件", "half", "作弊"),
    ("sweep/sweep_code3/final.pt",         "C", "k-means 编码", "code", ""),
    ("sweep/sweep_pca4/final.pt",          "D", "PCA-4", "pca", ""),
    ("sweep/sweep_pca8/final.pt",          "E", "PCA-8", "pca", ""),
    ("sweep/sweep_pca16/final.pt",         "F", "PCA-16", "pca", ""),
    ("sweep/sweep_pca22/final.pt",         "G", "PCA-22 有噪声", "pca", ""),
    ("sweep/pca22_nonoise/final.pt",       "H", "PCA-22 无噪声", "pca", "去噪声"),
    ("sweep/pca22_long/final.pt",          "I", "PCA-22 10k", "pca", ""),
    ("coherence/long_lam2_mode/final.pt",  "J", "+TV λ=2", "pca", "加平滑"),
    ("coherence/n_const_lam5/final.pt",    "K", "TV λ=5", "pca", ""),
    ("coherence/n_ramp_2_10/final.pt",     "L", "TV 斜坡 2→10", "pca", ""),
    ("coherence/n_ramp_long_15k/final.pt", "M", "TV 斜坡 15k", "pca", ""),
    ("coherence/final_lam5_20k/final.pt",  "N", "最佳 512 输出", "pca", "巅峰"),
    ("distill/distill_plain/final.pt",     "O", "蒸馏 32 输出", "pca", "压缩"),
    ("shrink/s32_tv5/final.pt",            "P", "32输出 TV5", "pca", ""),
    ("shrink/s32_tv30/final.pt",           "Q", "32输出 TV30", "pca", ""),
    ("shrink5/ladder_128_128.pt",          "R", "22→128→128→32", "pca", "收缩"),
    ("shrink5/ladder_64_64.pt",            "S", "22→64→64→32", "pca", ""),
    ("shrink5/ladder_32.pt",               "T", "22→32→32", "pca", "最省"),
]

blocks = []
for i, (rel, tag, detail, kind, era) in enumerate(MILESTONES):
    path = os.path.join(RES, rel.replace("/", os.sep))
    if not os.path.exists(path):
        continue
    rng = np.random.default_rng(SEED + i * 977)      # PER-MODEL random, per the request
    net, in_dim, out_dim, mult, args = load_net(path)
    got = condition_for(in_dim, kind, rng)
    cond, z = (got if isinstance(got, tuple) else (got, None))
    imgs, dec = render(net, cond, out_dim)
    blocks.append({"tag": tag, "detail": detail, "era": era, "imgs": imgs,
                   "agree": agree_of(imgs), "in_dim": in_dim, "out_dim": out_dim,
                   "mult": mult, "acc": ACC.get(tag), "purified": False})
    print(f"  {tag} {detail:22s} in={in_dim:3d} out={out_dim:3d} mult={mult:7,d} "
          f"agree={blocks[-1]['agree']*100:5.1f}%", flush=True)

# ---- the purified block: w128 sampled through the reward model ----------------------
if REW and REW.get("model") == "w128":
    net, in_dim, out_dim, mult, args = load_net(
        os.path.join(RES, "shrink5", "ladder_128_128.pt"))
    w = np.array(REW["w"]); b = float(REW["b"])
    zm = np.array(REW["zmean"]); zsd = np.array(REW["zsd"])
    rng = np.random.default_rng(SEED + 99991)
    picked = []
    while len(picked) < PER_MODEL:
        pool = np.clip(rng.standard_normal((600, 22)), -3.5, 3.5)
        s = ((pool - zm) / zsd) @ w + b
        k = max(1, int(600 * 0.15))
        top = np.argsort(-s)[:k]
        picked.append(pool[top[rng.integers(0, k)]])
    cond = np.array(picked, dtype=np.float32)
    imgs, dec = render(net, cond, out_dim)
    # Per-pixel accuracy is MEANINGLESS here: these inputs come from the guided
    # distribution, not from the 32 held-out real faces in RTRUE. Report the reward score
    # instead, which is the quantity the sampler actually maximises.
    mscore = float((((cond - zm) / zsd) @ w + b).mean())
    blocks.append({"tag": "U", "detail": "★ 纯化后采样", "era": "打标", "imgs": imgs,
                   "agree": agree_of(imgs), "in_dim": in_dim, "out_dim": out_dim,
                   "mult": mult, "acc": None, "reward": mscore, "purified": True})
    print(f"  U 纯化后采样 (w128 + 奖励模型)  agree={blocks[-1]['agree']*100:5.1f}%  "
          f"mean reward={mscore:.3f}")

# ---------------------------------------------------------------- draw
TILE = 22
BW, BH = 8, 4                      # each model is an 8x4 block of 32 faces
GAP = 14
LABH = 30
PER_ROW = 4
rows = (len(blocks) + PER_ROW - 1) // PER_ROW
CW = BW * TILE + GAP
CH = BH * TILE + LABH + GAP
W = PER_ROW * CW + GAP + 26
H = 74 + rows * CH + 30
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_m, f_s, f_x = font(19), font(12), font(10), font(9)

dr.text((GAP, 10), "头像生成二期工程 · 每个模型各自随机 32 张（看整体趋势，不比单张）",
        font=f_t, fill=(234, 238, 244))
dr.text((GAP, 36), "每块 = 一个模型的 32 张，8×4 排列；A 无条件、B 用真实半边，已注明；"
        "最后一块★是用你的标签纯化后的采样",
        font=f_s, fill=(150, 162, 178))
dr.text((GAP, 52), "相邻同色率参考：真实数据 77.6%   ·   acc = 逐像素准确率（该次训练自报的 1000 张留出值）",
        font=f_s, fill=(150, 162, 178))

for bi, blk in enumerate(blocks):
    r_, c_ = divmod(bi, PER_ROW)
    x0 = GAP + c_ * CW
    y0 = 74 + r_ * CH
    for k in range(PER_MODEL):
        rr, cc = divmod(k, BW)
        g = np.clip(blk["imgs"][k] / 15.0 * 255.0, 0, 255)
        t = np.kron(g, np.ones((TILE // 8, TILE // 8)))
        cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                 (x0 + cc * TILE, y0 + rr * TILE))
    # border: highlight the purified block
    box = [x0 - 1, y0 - 1, x0 + BW * TILE, y0 + BH * TILE]
    dr.rectangle(box, outline=(240, 200, 110) if blk["purified"] else (52, 58, 70), width=2)
    dr.text((x0, y0 + BH * TILE + 2), f"{blk['tag']}  {blk['detail']}", font=f_m,
            fill=(240, 200, 110) if blk["purified"] else (206, 214, 226))
    sub = f"in {blk['in_dim']} · {blk['mult']:,}乘 · 同色 {blk['agree']*100:.0f}%"
    if blk.get("reward") is not None:
        sub += f" · 奖励分 {blk['reward']:.2f}"
    elif blk["acc"] is not None:
        sub += f" · acc {blk['acc']:.3f}"
    if blk["era"]:
        sub += f" 【{blk['era']}】"
    dr.text((x0, y0 + BH * TILE + 16), sub, font=f_x, fill=(132, 144, 160))

dr.text((GAP, H - 26), "读法：A 是无条件（灰块）、B 是抄真实半边（看着好但无意义）。"
        "D 学会用 22 维坐标后，",
        font=f_x, fill=(140, 152, 168))
dr.text((GAP, H - 14), "各块观感接近，真正的进步在数字：acc 0.478→0.772、"
        "同色率 71%→80%、乘法 202,240→1,728。",
        font=f_x, fill=(140, 152, 168))

p = os.path.join(OUT, "PHASE2_PROGRESS_V2.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

json.dump([{k: v for k, v in b.items() if k != "imgs"} for b in blocks],
          open(os.path.join(OUT, "progress_v2.json"), "w"), indent=2, ensure_ascii=False)
print("PROGRESS_V2_DONE")

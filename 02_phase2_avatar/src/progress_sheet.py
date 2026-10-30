#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
The phase-2 progress sheet, with the quantitative half of the story beside the pictures.

The pictures alone are honest but misleading: from row D (PCA-4) onwards the faces look
similar, because the big VISUAL jump happened when the condition went from "nothing /
a learned code" to "actual PCA coordinates". What kept improving afterwards was per-pixel
accuracy, smoothness, and cost -- none of which is obvious at 8x8.

So each row also carries:
  * neighbour agreement, measured here on these 32 faces
  * per-pixel accuracy, taken from each run's own recorded evaluation on 1000 held-out
    faces where available (a dash where it was never recorded)
  * multiplies per image, computed from the weight shapes
  * a marker for the two failure routes and for the cost optimisations

Models are fed the same 32 condition vectors, so a row reflects the model, not the draw.
"""
import json, os, sys
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


def font(sz):
    for c in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", c)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


rng = np.random.default_rng(20261017)
NOISE = np.clip(rng.standard_normal((32, 64)), -3.5, 3.5)

rgb = np.load(os.path.join(ROOT, "data", "faces8_rgb_v2.npy"))
LUM = np.array([0.299, 0.587, 0.114])
gray = (rgb.astype(np.float64) * LUM).sum(-1)
gyi = np.abs(gray[:, :, :, None] - LEVELS[None, None, None, :]).argmin(-1)
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
bv = va[:32]
ZCOND = ((X[bv] - mu[0]) @ comps.T) / sd[0]
HALF_OH = np.zeros((32, 32 * NLEV), dtype=np.float32)
for k in range(32):
    for p in range(32):
        HALF_OH[k, p * NLEV + int(full[bv[k]][layout.POS_ROW[p], layout.POS_COL[p]])] = 1.0
RTRUE = full[bv][:, layout.POS_ROW, layout.MIRROR_COL]


def load_net(path):
    ck = torch.load(path, weights_only=False)
    st = ck.get("G") or ck.get("S")
    norm = {(k.replace("net.", "", 1) if k.startswith("net.") else k): v
            for k, v in st.items()}
    idx = sorted({int(k.split(".")[0]) for k in norm if k.endswith(".weight")})
    seq = []
    dims = []
    for j in idx:
        W = norm[f"{j}.weight"]
        seq.append(nn.Linear(int(W.shape[1]), int(W.shape[0])))
        dims.append((int(W.shape[1]), int(W.shape[0])))
        if j != idx[-1]:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq)
    m.load_state_dict(norm)
    m.eval()
    mult = sum(a * b for a, b in dims)
    return m, dims[0][0], dims[-1][1], mult


def condition_for(in_dim, kind):
    if kind == "uncond":
        return NOISE.astype(np.float32)
    if kind == "half":
        return np.concatenate([HALF_OH, NOISE], 1).astype(np.float32)
    if in_dim == 22:
        return ZCOND.astype(np.float32)
    if in_dim - 22 == 64:
        return np.concatenate([ZCOND, NOISE], 1).astype(np.float32)
    if in_dim == 68:
        return np.concatenate([ZCOND[:, :4], NOISE], 1).astype(np.float32)
    if in_dim == 72:
        return np.concatenate([ZCOND[:, :8], NOISE], 1).astype(np.float32)
    if in_dim == 80 and kind == "code":
        oh = np.zeros((32, 16), dtype=np.float32)
        for k in range(32):
            oh[k, k % 16] = 1.0
        return np.concatenate([oh, NOISE], 1).astype(np.float32)
    if in_dim == 80:
        return np.concatenate([ZCOND[:, :16], NOISE], 1).astype(np.float32)
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


# accuracy measured by each run on 1000 held-out faces (see the run's summary.json)
ACC = {
    "G": 0.4779, "H": 0.6429, "I": 0.6683, "N": 0.7005, "O": 0.7348, "P": 0.7719,
    "Q": 0.7138, "R": 0.7392, "S": 0.7057, "T": 0.6791,
}

MILESTONES = [
    ("sweep/sweep_uncond/final.pt",        "A", "无条件 GAN（只有噪声）",   "uncond", "失败"),
    ("sweep/sweep_half/final.pt",          "B", "半边真实条件",             "half",   "作弊"),
    ("sweep/sweep_code3/final.pt",         "C", "学到的 k-means 编码",       "code",   ""),
    ("sweep/sweep_pca4/final.pt",          "D", "PCA-4 + 噪声",              "pca",    ""),
    ("sweep/sweep_pca8/final.pt",          "E", "PCA-8 + 噪声",              "pca",    ""),
    ("sweep/sweep_pca16/final.pt",         "F", "PCA-16 + 噪声",             "pca",    ""),
    ("sweep/sweep_pca22/final.pt",         "G", "PCA-22 + 噪声",             "pca",    ""),
    ("sweep/pca22_nonoise/final.pt",       "H", "PCA-22 无噪声",             "pca",    "去噪声"),
    ("sweep/pca22_long/final.pt",          "I", "PCA-22 + 噪声 10k 步",      "pca",    ""),
    ("coherence/long_lam2_mode/final.pt",  "J", "+TV λ=2",                  "pca",    "加平滑"),
    ("coherence/n_const_lam5/final.pt",    "K", "TV λ=5 恒定",               "pca",    ""),
    ("coherence/n_ramp_2_10/final.pt",     "L", "TV 斜坡 2→10",              "pca",    ""),
    ("coherence/n_ramp_long_15k/final.pt", "M", "TV 斜坡 15k 步",            "pca",    ""),
    ("coherence/final_lam5_20k/final.pt",  "N", "最佳 512 输出 λ=5 20k",     "pca",    "巅峰"),
    ("distill/distill_plain/final.pt",     "O", "蒸馏 32 输出",              "pca",    "压缩"),
    ("shrink/s32_tv5/final.pt",            "P", "32 输出 hard+TV5",          "pca",    ""),
    ("shrink/s32_tv30/final.pt",           "Q", "32 输出 hard+TV30",         "pca",    ""),
    ("shrink5/ladder_128_128.pt",          "R", "22→128→128→32",             "pca",    "收缩"),
    ("shrink5/ladder_64_64.pt",            "S", "22→64→64→32",               "pca",    ""),
    ("shrink5/ladder_32.pt",               "T", "22→32→32",                  "pca",    "最省"),
]

rows = []
for rel, tag, detail, kind, era in MILESTONES:
    path = os.path.join(RES, rel.replace("/", os.sep))
    if not os.path.exists(path):
        print(f"  skip {tag}")
        continue
    net, in_dim, out_dim, mult = load_net(path)
    cond = condition_for(in_dim, kind)
    with torch.no_grad():
        raw = net(torch.tensor(cond, dtype=torch.float32))
    if out_dim == 512:
        dec = raw.view(-1, 32, NLEV).argmax(-1).numpy()
    else:
        dec = raw.clamp(0, 15).round().numpy().astype(int)
    imgs = layout.image_from_left_half(dec)
    acc_here = float((dec == RTRUE).mean())
    rows.append({"tag": tag, "detail": detail, "era": era, "imgs": imgs,
                 "agree": agree_of(imgs), "in_dim": in_dim, "out_dim": out_dim,
                 "mult": mult, "acc32": acc_here, "acc1000": ACC.get(tag)})
    print(f"  {tag} {detail:24s} in={in_dim:3d} out={out_dim:3d} "
          f"mult={mult:7,d} agree={rows[-1]['agree']*100:5.1f}% "
          f"acc(32)={acc_here:.3f} acc(1000)={ACC.get(tag, float('nan')):.3f}", flush=True)

# ---------------------------------------------------------------- draw
COLS, TILE, PAD = 32, 30, 4
LEFT, RIGHT, TOP, LAB = 300, 250, 92, 16
W = LEFT + COLS * (TILE + PAD) + PAD + RIGHT
H = TOP + len(rows) * (TILE + PAD + LAB) + 46
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_h, f_r, f_s = font(20), font(13), font(12), font(11)

dr.text((12, 10), "头像生成二期工程 · 每代模型同一批 32 个条件", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), "A 无条件（只有噪声）· B 用真实半边（靠抄，高分但无意义）· D 起注入 22 维 PCA 坐标",
        font=f_s, fill=(152, 164, 180))
dr.text((12, 54), "acc = 逐像素准确率（该次训练自己记录的 1000 张留出评估）· mult = 每次生成所需乘法",
        font=f_s, fill=(152, 164, 180))
dr.text((12, 70), "相邻同色率参考：真实数据 77.6%", font=f_s, fill=(152, 164, 180))

X0 = LEFT
XM = X0 + COLS * (TILE + PAD) + PAD + 10
dr.text((XM, TOP - 18), "相邻同色", font=f_s, fill=(130, 142, 158))
dr.text((XM + 58, TOP - 18), "acc", font=f_s, fill=(130, 142, 158))
dr.text((XM + 104, TOP - 18), "乘法", font=f_s, fill=(130, 142, 158))

y = TOP
for i, r in enumerate(rows):
    if i and i % 5 == 0:
        dr.line([8, y - 3, W - 8, y - 3], fill=(46, 52, 62))
    dr.text((12, y + 1), r["tag"], font=f_h, fill=(110, 168, 254))
    dr.text((34, y + 1), r["detail"], font=f_r, fill=(206, 214, 226))
    sub_txt = f"in {r['in_dim']} → out {r['out_dim']}"
    if r["era"]:
        sub_txt += f"  【{r['era']}】"
    dr.text((34, y + 16), sub_txt, font=f_s, fill=(126, 138, 154))
    for col in range(COLS):
        g = np.clip(r["imgs"][col] / 15.0 * 255.0, 0, 255)
        t = np.kron(g, np.ones((TILE // 8, TILE // 8)))
        cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                 (X0 + col * (TILE + PAD), y))
    ag = r["agree"]
    c_ag = (110, 200, 130) if ag > 0.75 else (224, 192, 90) if ag > 0.65 else (224, 122, 95)
    dr.text((XM, y + 6), f"{ag*100:.0f}%", font=f_r, fill=c_ag)
    a = r["acc1000"]
    if a is not None:
        c_a = (110, 200, 130) if a > 0.72 else (224, 192, 90) if a > 0.65 else (224, 122, 95)
        dr.text((XM + 58, y + 6), f"{a:.3f}", font=f_r, fill=c_a)
    else:
        dr.text((XM + 58, y + 6), "—", font=f_r, fill=(110, 120, 134))
    m = r["mult"]
    dr.text((XM + 104, y + 6), f"{m:,}", font=f_r,
            fill=(110, 200, 130) if m < 30000 else (206, 214, 226))
    y += TILE + PAD + LAB

dr.text((12, H - 32), "图片的诚实结论：视觉上的大跳跃发生在 C→D（学会用 22 维坐标定位）。之后各行看起来相近，",
        font=f_s, fill=(150, 162, 178))
dr.text((12, H - 18), "真正的进步换成了三个数字：准确率 0.478→0.772、平滑度 71%→80%、乘法 202,240→1,728。",
        font=f_s, fill=(150, 162, 178))

p = os.path.join(OUT, "PHASE2_PROGRESS.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")
json.dump([{k: v for k, v in r.items() if k != "imgs"} for r in rows],
          open(os.path.join(OUT, "progress.json"), "w"), indent=2, ensure_ascii=False)
print("PROGRESS_DONE")

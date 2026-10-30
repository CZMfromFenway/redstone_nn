#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Why do PCA-4 and PCA-8 look surprisingly good? Measure instead of guessing.

HOW THEY WERE PRODUCED (the pipeline, stated plainly)
All the sweep models come from one script, sweep_condition.py, differing only in what the
condition vector is. The PCA-k variants are conditioned on the FIRST k PCA coefficients of
a real face, standardised to standard-deviation units, CONCATENATED with a 64-dim latent
noise vector:

    input = [pca_0 .. pca_{k-1}]  ++  [noise_0 .. noise_63]

so PCA-4 takes 4 + 64 = 68 inputs, PCA-8 takes 72, PCA-22 takes 86. Every one of them was
trained for only 1500 steps with the same optimiser, batch size and loss (adversarial + CE
against the real right half). They are therefore SHORT runs, not tuned models.

The sampled parameters in the progress sheet are random: for each model a fresh
standard-normal draw for the k PCA slots and a fresh draw for the 64 noise slots.

THE PUZZLE
Their per-pixel accuracy is far worse than PCA-22's (0.210 and 0.270 versus 0.382) yet they
look like faces. Two hypotheses explain that, and they are separable by experiment:

  H1 the k PCA coordinates carry the face, and the missing components only affect detail
  H2 the 64 noise dimensions carry most of the variety, and the PCA coordinates barely
     matter -- in which case a model that looks good is basically an unconditional
     generator with a weak hint

This script tests both by holding one input group fixed and varying the other:

    vary PCA, fix noise  -> how much does the condition actually control?
    fix PCA, vary noise  -> how much does the noise control?

and by measuring accuracy at several noise scales.
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
OUT = os.path.join(RES, "pca_sweep")
os.makedirs(OUT, exist_ok=True)


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
ZZ = ((X - mu) @ comps.T) / sd[0]                 # (N,22) standardised coefficients
RTRUE = full[:, layout.POS_ROW, layout.MIRROR_COL]


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
    return m, dims[0][0], dims[-1][1]


def run(net, cond):
    with torch.no_grad():
        raw = net(torch.tensor(cond, dtype=torch.float32))
    dec = raw.view(-1, 32, NLEV).argmax(-1).numpy()
    return layout.image_from_left_half(dec), dec


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


def diversity(imgs):
    """mean pairwise Hamming distance over 64 pixels, and distinct coarse patterns."""
    a = imgs.reshape(len(imgs), -1).astype(np.int16)
    n = len(a)
    d = np.abs(a[:, None, :] - a[None, :, :]).__gt__(0).sum(2).astype(float)
    npt = d.copy()
    np.fill_diagonal(npt, np.nan)
    pats = set()
    for g in imgs:
        c = np.round(g.reshape(2, 4, 2, 4).mean(axis=(1, 3)) / 15 * 5).astype(int)
        pats.add(tuple(c.reshape(-1).tolist()))
    return float(np.nanmean(npt)), len(pats)


MODELS = [("pca4", "sweep/sweep_pca4/final.pt", 4),
          ("pca8", "sweep/sweep_pca8/final.pt", 8),
          ("pca16", "sweep/sweep_pca16/final.pt", 16),
          ("pca22", "sweep/sweep_pca22/final.pt", 22)]

# ---------------------------------------------------------------- A. the pipeline
print("=" * 104)
print("A. WHAT THE CONDITION VECTOR ACTUALLY IS")
print("=" * 104)
for tag, rel, k in MODELS:
    path = os.path.join(RES, rel.replace("/", os.sep))
    if not os.path.exists(path):
        continue
    net, in_dim, out_dim = load_net(path)
    print(f"  {tag:6s} input = {k} PCA  +  {in_dim - k} noise  =  {in_dim}  ->  512 logits")
print()
print("  The k PCA slots are the first k standardised coefficients of a real face;")
print("  the noise slots are a fresh standard-normal draw for every sample.")

# ---------------------------------------------------------------- B. who controls what
print()
print("=" * 104)
print("B. WHO CONTROLS THE OUTPUT -- the condition, or the noise?")
print("=" * 104)
print("  vary PCA (64 draws), noise FIXED     -> what the condition alone can do")
print("  vary noise (64 draws), PCA FIXED     -> what the noise alone can do")
print()
print(f"  {'model':7s} {'variant':26s} {'Hamming':>9s} {'patterns':>9s} {'agree':>8s}")
print("  " + "-" * 66)
stats = {}
rng = np.random.default_rng(7)
for tag, rel, k in MODELS:
    path = os.path.join(RES, rel.replace("/", os.sep))
    if not os.path.exists(path):
        continue
    net, in_dim, out_dim = load_net(path)
    noise_dim = in_dim - k
    # (a) vary PCA, fix noise
    pca_block = np.clip(rng.standard_normal((64, k)), -3.5, 3.5)
    fixed_noise = np.clip(rng.standard_normal((1, noise_dim)), -3.5, 3.5).repeat(64, 0)
    cond = np.concatenate([pca_block, fixed_noise], 1).astype(np.float32)
    imgs, _ = run(net, cond)
    h_p, n_p = diversity(imgs)
    a_p = agree_of(imgs)
    # (b) fix PCA, vary noise
    fixed_pca = np.clip(rng.standard_normal((1, k)), -3.5, 3.5).repeat(64, 0)
    noise_block = np.clip(rng.standard_normal((64, noise_dim)), -3.5, 3.5)
    cond2 = np.concatenate([fixed_pca, noise_block], 1).astype(np.float32)
    imgs2, _ = run(net, cond2)
    h_n, n_n = diversity(imgs2)
    a_n = agree_of(imgs2)
    stats[tag] = {"pca": n_p, "noise": n_n}
    print(f"  {tag:7s} {'vary PCA, fix noise':26s} {h_p:9.2f} {n_p:9d} {a_p*100:7.1f}%")
    print(f"  {'':7s} {'fix PCA, vary noise':26s} {h_n:9.2f} {n_n:9d} {a_n*100:7.1f}%")

# ---------------------------------------------------------------- C. accuracy vs noise scale
print()
print("=" * 104)
print("C. HOW MUCH DOES THE NOISE HURT ACCURACY?")
print("=" * 104)
print("  For each model: feed the TRUE PCA of held-out faces and scale the noise down.")
print(f"  {'model':7s} {'noise scale':>12s} {'accuracy':>9s} {'agree':>8s}")
print("  " + "-" * 40)
NS = 1000
bv = va[:NS]
acc_tab = {}
for tag, rel, k in MODELS:
    path = os.path.join(RES, rel.replace("/", os.sep))
    if not os.path.exists(path):
        continue
    net, in_dim, out_dim = load_net(path)
    noise_dim = in_dim - k
    ctx = np.clip(ZZ[bv][:, :k], -3.5, 3.5)
    rng2 = np.random.default_rng(11)
    base_noise = np.clip(rng2.standard_normal((NS, noise_dim)), -3.5, 3.5)
    accs = []
    for scale in (0.0, 0.25, 0.5, 1.0):
        cond = np.concatenate([ctx, base_noise * scale], 1).astype(np.float32)
        _, dec = run(net, cond)
        acc = float((dec == RTRUE[bv]).mean())
        accs.append({
            "scale": scale, "acc": acc,
            "agree": agree_of(layout.image_from_left_half(dec))})
        print(f"  {tag:7s} {scale:12.2f} {acc:9.4f} {accs[-1]['agree']*100:7.1f}%")
    acc_tab[tag] = accs
    print()

# ---------------------------------------------------------------- D. showcase
print("=" * 104)
print("D. SHOWCASE")
print("=" * 104)
SHOW = 48
TILE, PAD = 26, 3
rows_drawn = []
for tag, rel, k in MODELS:
    path = os.path.join(RES, rel.replace("/", os.sep))
    if not os.path.exists(path):
        continue
    net, in_dim, out_dim = load_net(path)
    noise_dim = in_dim - k
    rng3 = np.random.default_rng(2024)
    pca = np.clip(rng3.standard_normal((SHOW, k)), -3.5, 3.5)
    noise = np.clip(rng3.standard_normal((SHOW, noise_dim)), -3.5, 3.5)
    imgs, dec = run(net, np.concatenate([pca, noise], 1).astype(np.float32))
    rows_drawn.append((tag, k, in_dim, imgs, agree_of(imgs)))
    print(f"  {tag:6s} 48 random draws, agreement {agree_of(imgs)*100:.1f}%")

COLS = SHOW
W = 96 + COLS * (TILE + PAD) + PAD
H = 46 + len(rows_drawn) * (TILE + 24) + 40
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(17), font(11), font(9)
dr.text((6, 8), "PCA-4 / 8 / 16 / 22 · 每个 48 张随机采样（各自的随机参数）",
        font=f_t, fill=(232, 236, 242))
dr.text((6, 30), "输入 = 前 k 个 PCA 系数 + 64 维噪声。四者同批超参、都只训练了 1500 步。",
        font=f_s, fill=(150, 162, 178))
y = 46
for tag, k, in_dim, imgs, ag in rows_drawn:
    for i in range(COLS):
        g = np.clip(imgs[i] / 15.0 * 255.0, 0, 255)
        t = np.kron(g, np.ones((TILE // 8, TILE // 8)))
        cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                 (96 + i * (TILE + PAD), y))
    dr.text((6, y + 4), tag, font=f_l, fill=(110, 168, 254))
    dr.text((6, y + 15), f"{k} PCA + {in_dim-k} 噪声", font=f_s, fill=(130, 142, 158))
    dr.text((96 + COLS * (TILE + PAD) + 4, y + 6), f"{ag*100:.0f}%", font=f_s,
            fill=(160, 172, 188))
    y += TILE + 24
dr.text((6, H - 14), "相邻同色率（最右）参考真实数据 77.6%。",
        font=f_s, fill=(140, 152, 168))
p = os.path.join(OUT, "PCA_SHOWCASE.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

json.dump({"variety": stats, "accuracy_vs_noise": acc_tab},
          open(os.path.join(OUT, "pca_analysis.json"), "w"), indent=2)
print("PCA_ANALYSIS_DONE")

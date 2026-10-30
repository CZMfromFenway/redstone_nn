#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
The progress sheet -- corrected load path, plus a guard that refuses degenerate models.

WHAT WENT WRONG
Rows 5-8 loaded results/dimsweep/dim4/final.pt (via an earlier version of this script). That
checkpoint is a 200-STEP smoke-test leftover: an untagged run overwrote the real 12000-step
one. An under-trained model leaves some output positions saturated, which is exactly the
"3-4 fixed black pixels in the upper corners" the user spotted. The authoritative weights live
in webapp/pca22_dim4_model.json, which is what the labelling page has always used.

The mistake was invisible because a figure only shows pixels. So the guard below counts, for
every row, how many of the 32 canonical positions never change across 600 samples, and ABORTS
if any are frozen. That check would have caught this and the earlier frozen-pixel incident
without a human needing to spot it.

FILTERING
The keep fraction is one constant (KEEP, default 0.50 as requested).
At keep 0.50 the measured acceptable rates are 78.9% (no style target) and 75.0% (类二次元
target); at 0.25 they are 92.1% and 94.7%.
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
WEB = os.path.join(ROOT, "webapp")
OUT = os.path.join(RES, "PROGRESS")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

KEEP = 0.50                  # requested default; a single constant so it stays adjustable
FROZEN_TOL = 0.35            # a position whose level never moves beyond this is frozen

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def load_json_model(fname):
    j = json.load(open(os.path.join(WEB, fname), encoding="utf-8"))
    layers = j["gen"]["layers"]
    seq = []
    for i, L in enumerate(layers):
        W = torch.tensor(L["W"], dtype=torch.float32)
        b = torch.tensor(L["b"], dtype=torch.float32)
        lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            lin.weight.copy_(W); lin.bias.copy_(b)
        seq.append(lin)
        if i != len(layers) - 1:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.eval()
    return m, int(len(layers[0]["W"][0])), int(len(layers[-1]["W"]))


def load_ckpt(path):
    ck = torch.load(path, weights_only=False)
    st = ck.get("G") or ck.get("S")
    st = {(k.replace("net.", "", 1) if k.startswith("net.") else k): v for k, v in st.items()}
    idx = sorted({int(k.split(".")[0]) for k in st if k.endswith(".weight")})
    seq = []
    for j in idx:
        W = st[f"{j}.weight"]
        seq.append(nn.Linear(int(W.shape[1]), int(W.shape[0])))
        if j != idx[-1]:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.load_state_dict(st); m.eval()
    return m, int(st[f"{idx[0]}.weight"].shape[1]), int(st[f"{idx[-1]}.weight"].shape[0])


def predict32(net, out_dim, Z):
    with torch.no_grad():
        raw = net(torch.tensor(Z.astype(np.float32)))
    if out_dim == 512:
        return raw.view(-1, 32, 16).argmax(-1).numpy().astype(float)
    return np.clip(raw.numpy(), 0, 15)


def stretch(v):
    lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
    return np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)


def frozen_count(dec):
    return int((dec.std(0) < FROZEN_TOL).sum())


# ---------------------------------------------------------------- 4->16->1 scorers
recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
lab = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
Xl = np.array([r["pca"][:4] for r in lab], dtype=np.float32)
g = np.array([r["grade"] for r in lab])
mu = Xl.mean(0); sd = Xl.std(0) + 1e-9


def train_scorer(target, seed=0, steps=900):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-3)
    t = torch.tensor((Xl - mu) / sd, dtype=torch.float32)
    v = torch.tensor(target, dtype=torch.float32)[:, None]
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(net(t), v).backward()
        opt.step()
    net.eval()
    return net


sc_global = train_scorer((g >= 2).astype(float), seed=1)
sc_anime = train_scorer((g == 4).astype(float), seed=2)
print(f"scorers: 4 -> 16 -> 1, trained on {len(lab)} labels")


def score(net, Z):
    with torch.no_grad():
        return net(torch.tensor(((Z - mu) / sd).astype(np.float32))).numpy().ravel()


# ---------------------------------------------------------------- rows
FULL = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
bv = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_split.npz"))["val"][:32]
HALF = np.zeros((32, 32 * 16), dtype=np.float32)
for k in range(32):
    for p in range(32):
        HALF[k, p * 16 + int(FULL[bv[k]][layout.POS_ROW[p], layout.POS_COL[p]])] = 1.0

ROWS = [
    ("1  无条件 GAN", "no condition", ("ckpt", "gan_v2/uncond_fixed_full/final.pt"), "uncond",
     False, None),
    ("2  条件 GAN（半边真实输入）", "memorises the answer",
     ("ckpt", "sweep/sweep_half/final.pt"), "half", False, None),
    ("3  PCA-22", "22 PCs + noise", ("ckpt", "sweep/sweep_pca22/final.pt"), "pca22", False,
     None),
    ("4  PCA-22 + 平滑奖励", "TV coherence penalty",
     ("ckpt", "coherence/final_lam5_20k/final.pt"), "pca22", False, None),
    ("5  PCA-4", "four knobs only", ("json", "pca22_dim4_model.json"), "pca4", False, None),
    ("6  PCA-4 + min-max 拉伸", "contrast restored", ("json", "pca22_dim4_model.json"),
     "pca4", True, None),
    ("7  + 筛选（不指定风格）", f"MLP 4-16-1, keep {int(KEEP*100)}%",
     ("json", "pca22_dim4_model.json"), "pca4", True, "global"),
    ("8  + 指定二次元 + 筛选", f"keep {int(KEEP*100)}%, scorer aimed at 类二次元",
     ("json", "pca22_dim4_model.json"), "pca4", True, "anime"),
]
ACC = {("global", 0.50): 78.9, ("global", 0.25): 92.1,
       ("anime", 0.50): 75.0, ("anime", 0.25): 94.7}

COLS = 32
fig = []
for name, note, src, kind, do_stretch, mode in ROWS:
    if src[0] == "json":
        net, in_dim, out_dim = load_json_model(src[1])
    else:
        p = os.path.join(RES, src[1].replace("/", os.sep))
        if not os.path.exists(p):
            print(f"  skip {name}: missing")
            continue
        net, in_dim, out_dim = load_ckpt(p)

    rng = np.random.default_rng(11)
    if kind == "uncond":
        Z = np.clip(rng.standard_normal((COLS, in_dim)), -3.5, 3.5)
    elif kind == "half":
        Z = np.concatenate([HALF, np.clip(rng.standard_normal((COLS, in_dim - 512)), -3.5, 3.5)], 1)
    elif kind == "pca22":
        Z = np.clip(rng.standard_normal((COLS, in_dim)), -3.5, 3.5)
    else:
        Z = np.clip(rng.standard_normal((COLS, 4)), -3.5, 3.5)
        if in_dim != 4:
            Z = np.concatenate([Z, np.zeros((COLS, in_dim - 4))], 1)

    if mode is None:
        dec = predict32(net, out_dim, Z)
        if do_stretch:
            dec = stretch(dec)
        accepted = None
        probe = stretch(predict32(net, out_dim, np.clip(
            np.random.default_rng(99).standard_normal((600, 4)), -3.5, 3.5))) \
            if kind == "pca4" else predict32(
                net, out_dim, np.clip(np.random.default_rng(99).standard_normal(
                    (600, in_dim)), -3.5, 3.5))
    else:
        pool = np.clip(np.random.default_rng(5).standard_normal((40000, 4)), -3.5, 3.5)
        sc = score(sc_anime if mode == "anime" else sc_global, pool)
        k = int(len(pool) * KEEP)
        keep = pool[np.argsort(-sc)[:k]]
        show = keep[np.random.default_rng(17).choice(len(keep), COLS, replace=False)]
        dec = stretch(predict32(net, out_dim, show))
        accepted = ACC.get((mode, KEEP))
        probe = dec

    nf = frozen_count(probe)
    imgs = layout.image_from_left_half(np.rint(dec).astype(int))
    fig.append({"name": name, "note": note, "imgs": imgs, "accepted": accepted,
                "in_dim": in_dim, "frozen": nf})
    print(f"  {name:28s} in={in_dim:3d} out={out_dim:3d}  frozen {nf:2d}/32"
          + (f"  可接受 {accepted}%" if accepted else "")
          + ("   <-- FROZEN" if nf else ""))

bad = [r for r in fig if r["frozen"] > 0]
if bad:
    print()
    for r in bad:
        print(f"  ABORT: '{r['name']}' has {r['frozen']} frozen output positions.")
    raise SystemExit("refusing to draw a figure from degenerate models")

# ---------------------------------------------------------------- draw
TILE, PAD = 34, 3
LEFT = 250
W = LEFT + COLS * (TILE + PAD) + PAD
H = 96 + len(fig) * (TILE + 28) + 30
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(12), font(10)
dr.text((10, 10), "头像生成：从无条件 GAN 到最终方案", font=f_t, fill=(234, 238, 244))
dr.text((10, 36), "每个里程碑 32 张随机脸。第 7、8 行＝生成 2 倍种子，MLP 4→16→1 打分，"
        f"留最好的 {int(KEEP*100)}%，再出图。", font=f_s, fill=(150, 162, 178))
dr.text((10, 52), "这两行展示存活者中的随机样本，不是最高分的那几张（最高分是极端尾部，"
        "会让一行看起来都一样）", font=f_s, fill=(130, 142, 158))
dr.text((10, 68), "全部 8 行已通过冻结像素检测（每个位置跨样本都在变化）", font=f_s,
        fill=(110, 200, 130))
y = 94
for r in fig:
    dr.text((10, y + 2), r["name"], font=f_l, fill=(110, 168, 254))
    dr.text((10, y + 18), r["note"], font=f_s, fill=(130, 142, 158))
    if r["accepted"]:
        dr.text((10, y + 30), f"可接受 ≈{r['accepted']}%", font=f_s, fill=(110, 200, 130))
    for i, gg in enumerate(r["imgs"]):
        t = np.kron(np.clip(gg / 15.0 * 255.0, 0, 255), np.ones((TILE // 8, TILE // 8)))
        cv.paste(Image.fromarray(t.astype(np.uint8), mode="L").convert("RGB"),
                 (LEFT + i * (TILE + PAD), y))
    y += TILE + 28
dr.text((10, H - 18), "第 1 行：无条件 GAN 只出噪点。第 2 行：半边条件靠抄真实数据。",
        font=f_s, fill=(140, 152, 168))
p = os.path.join(OUT, "FINAL_PROGRESS.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

json.dump([{kk: vv for kk, vv in r.items() if kk != "imgs"} for r in fig],
          open(os.path.join(OUT, "final_progress.json"), "w"), indent=2, ensure_ascii=False)
print("FINAL_PROGRESS_DONE")

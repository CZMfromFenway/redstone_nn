#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Score and render the compressed grayscale generators.

Accuracy on the held-out levels is not the thing that matters -- the user judges faces. So each
compressed model is measured two ways:

    proxy acceptable   generate from random N(0,1), take each face's luminance knobs, score with
                       the 4->16->1 network trained on the style labels. This is the same proxy
                       used for the colour models, so the numbers are comparable.
    a rendered sheet   because the numbers have been wrong before.

The grayscale path keeps its per-face min-max stretch, so the comparison is against the
deployed recipe, not against a raw output.
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
OUT = os.path.join(RES, "dimsweep", "COMPRESS")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("layout", os.path.join(HERE, "layout.py"))
layout = _ilu.module_from_spec(_spec); _spec.loader.exec_module(layout)
LUMW = np.array([0.299, 0.587, 0.114])


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


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
    return m, ck.get("args", {})


# ---- knobs of a produced face, using the same PCA basis as everything else
full = np.load(os.path.join(ROOT, "data_v2_canonical", "L0_full_idx.npy")).astype(np.int64)
Ng = len(full)
Xg = full.reshape(Ng, -1).astype(np.float64)
mu_g = Xg.mean(0, keepdims=True)
sub = np.random.default_rng(0).choice(Ng, 60000, replace=False)
_, _, Vt = np.linalg.svd(Xg[sub] - mu_g, full_matrices=False)
comps = Vt[:22]
sd_g = (((Xg - mu_g) @ comps.T).std(0, keepdims=True) + 1e-6)[0]

recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
lab = [r for r in recs if str(r.get("round") or "") == "5" and r.get("model") == "dim4x"
       and r.get("grade") in (0, 1, 2, 3, 4)]
Xl = np.array([r["pca"][:4] for r in lab], dtype=np.float32)
yl = np.array([1.0 if r["grade"] >= 2 else 0.0 for r in lab], dtype=np.float32)
mu_l = Xl.mean(0); sd_l = Xl.std(0) + 1e-9
torch.manual_seed(1)
sc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(sc.parameters(), lr=0.02, weight_decay=1e-3)
t = torch.tensor((Xl - mu_l) / sd_l); v = torch.tensor(yl)[:, None]
for _ in range(900):
    opt.zero_grad()
    nn.functional.binary_cross_entropy_with_logits(sc(t), v).backward()
    opt.step()
sc.eval()


def stretch(v):
    lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
    return np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)


def gen(net, n, seed):
    rng = np.random.default_rng(seed)
    Z = np.clip(rng.standard_normal((n, 4)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        v = np.clip(net(torch.tensor(Z)).numpy(), 0, 15)
    return np.rint(stretch(v)).astype(int)


CONFIGS = [
    (32, 2), (64, 1), (128, 1), (64, 2), (256, 1), (96, 2), (128, 2), (192, 2), (256, 2),
]
ACC = {}


def knobs_of(levels):
    cc = np.zeros((len(levels), 64))
    cc[:, layout.POS_ROW * 8 + layout.POS_COL] = levels
    cc[:, layout.POS_ROW * 8 + layout.MIRROR_COL] = levels
    return (((cc - mu_g) @ comps.T) / sd_g)[:, :4].astype(np.float32)


print("=" * 100)
print("COMPRESSED GRAYSCALE GENERATORS")
print("=" * 100)
print(f"  {'structure':16s} {'乘法':>7s} {'参数':>7s} {'acc':>6s} {'agree':>6s} "
      f"{'代理可接受':>10s} {'相对基线乘法':>12s}")
print("  " + "-" * 84)
rows = []
for hid, layers in CONFIGS:
    tag = f"dim4w{hid}L{layers}"
    p = os.path.join(RES, "dimsweep", tag, "final.pt")
    if not os.path.exists(p):
        continue
    net, args = load_ckpt(p)
    name = f"4-{hid}-32" if layers == 1 else f"4-{hid}-{hid}-32"
    mult = 4 * hid + (hid * 32 if layers == 1 else hid * hid + hid * 32)
    par = sum(q.numel() for q in net.parameters())
    lv = gen(net, 4000, 11)
    kn = knobs_of(lv)
    with torch.no_grad():
        s = sc(torch.tensor((kn - mu_l) / sd_l)).float()
        acc_rate = float((s.numpy().ravel() > 0).mean())
    acc_ = args.get("acc")
    agree = args.get("agree")
    print(f"  {name:16s} {mult:7d} {par:7d} "
          f"{(f'{acc_:.4f}' if acc_ else '—'):>6s} {(f'{100*agree:.1f}%' if agree else '—'):>6s} "
          f"{100*acc_rate:9.1f}% {mult/74752*100:11.1f}%")
    rows.append({"name": name, "tag": tag, "mult": mult, "params": par,
                 "acc": acc_, "agree": agree, "proxy": acc_rate, "net": net})

# ---- sheet
COLS, TILE, PAD = 20, 44, 4
W = 260 + COLS * (TILE + PAD) + PAD
H = 96 + len(rows) * (TILE + 30) + 24
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(20), font(12), font(10)
dr.text((12, 12), "灰度生成器压缩：宽度 vs 实际观感", font=f_t, fill=(234, 238, 244))
dr.text((12, 38), "每行 20 张随机脸（已做 min-max 拉伸）。同一批随机种子，便于逐列对比。",
        font=f_s, fill=(150, 162, 178))
dr.text((12, 54), "最下面是原方案 4-256-256-32（74,752 乘法）。", font=f_s, fill=(130, 142, 158))
y = 78
base_rng = np.random.default_rng(11)
Zfix = np.clip(base_rng.standard_normal((COLS, 4)), -3.5, 3.5).astype(np.float32)
for r in rows:
    net = r["net"]
    with torch.no_grad():
        v = np.clip(net(torch.tensor(Zfix)).numpy(), 0, 15)
    lv = np.rint(stretch(v)).astype(int)
    imgs = layout.image_from_left_half(lv)
    dr.text((12, y + 4), r["name"], font=f_l, fill=(110, 168, 254))
    dr.text((12, y + 20), f"{r['mult']:,} 乘法", font=f_s, fill=(150, 162, 178))
    dr.text((12, y + 32), f"代理 {100*r['proxy']:.0f}%", font=f_s, fill=(110, 200, 130))
    for i, g in enumerate(imgs):
        t = np.kron(np.clip(g / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8))).astype(np.uint8)
        cv.paste(Image.fromarray(np.stack([t] * 3, -1)), (260 + i * (TILE + PAD), y))
    y += TILE + 30
p = os.path.join(OUT, "COMPRESS_SHEET.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

json.dump([{k: v for k, v in r.items() if k != "net"} for r in rows],
          open(os.path.join(OUT, "compress.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"wrote {os.path.join(OUT, 'compress.json')}")
print("COMPRESS_EVAL_DONE")

#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
One sheet, every width and layer count, same seeds so the columns line up.

TWO FAMILIES
    4 -> h -> h -> 32    the two-hidden-layer shape used by the deployed model
    4 -> h -> 32         the single-hidden-layer shape, which matched it at a fraction of the cost

Each model gets a rectangle of 8 x 4 = 32 faces drawn from the SAME 32 seeds, so column k is the
same input in every row. Differences are therefore the model, not the sampling.

A GUARD
Every model is checked for frozen output positions before drawing. A checkpoint in this project
was once overwritten by an untagged smoke test and rendered as a nearly constant face; the check
exists so that cannot reach a figure silently. Exported JSONs are preferred over raw .pt files
because the JSON is what the deployed page actually loads.
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
RES = os.path.join(ROOT, "results", "dimsweep")
WEB = os.path.join(ROOT, "webapp")
OUT = os.path.join(RES, "COMPRESS")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)

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


def load_json(fname):
    j = json.load(open(os.path.join(WEB, fname), encoding="utf-8"))
    Ls = j["gen"]["layers"]
    seq = []
    for i, L in enumerate(Ls):
        W = torch.tensor(L["W"], dtype=torch.float32)
        b = torch.tensor(L["b"], dtype=torch.float32)
        lin = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            lin.weight.copy_(W); lin.bias.copy_(b)
        seq.append(lin)
        if i != len(Ls) - 1:
            seq.append(nn.ReLU())
    return nn.Sequential(*seq).eval(), j["meta"]


def load_ckpt(tag):
    p = os.path.join(RES, tag, "final.pt")
    if not os.path.exists(p):
        return None, None
    ck = torch.load(p, weights_only=False)
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


ROWS = [
    # ("label", source, note)
    ("4-256-256-32 已部署原版", ("json", "pca22_dim4x_model.json"), "74,752 乘法"),
    ("4-32-32-32", ("ckpt", "dim4w32L2"), "2,176"),
    ("4-64-64-32", ("ckpt", "dim4w64L2"), "6,400"),
    ("4-96-96-32", ("json", "pca22_dim4w96_model.json"), "12,672"),
    ("4-128-128-32", ("ckpt", "dim4w128L2"), "20,992"),
    ("4-192-192-32", ("ckpt", "dim4w192L2"), "43,776"),
    ("4-256-256-32 重训", ("json", "pca22_dim4w256_model.json"), "74,752"),
    ("4-64-32（单层）", ("json", "pca22_dim4w64L1_model.json"), "2,304"),
    ("4-128-32（单层）", ("ckpt", "dim4w128L1"), "4,608"),
    ("4-256-32（单层）", ("ckpt", "dim4w256L1"), "9,216"),
]
GROUP_BREAK = 7        # rows before this are the two-hidden-layer family

COLS, TILE, PAD = 16, 46, 4
ROWB = 2                     # faces per column-block -> 16 x 2 = 32 faces per model
NB = COLS * ROWB
rng = np.random.default_rng(11)
SEEDS = np.clip(rng.standard_normal((NB, 4)), -3.5, 3.5).astype(np.float32)

print("=" * 100)
print(f"ALL MODELS, {NB} shared seeds each")
print("=" * 100)
fig = []
for name, src, note in ROWS:
    if src[0] == "json":
        net, meta = load_json(src[1])
        acc = meta.get("acc")
    else:
        net, args = load_ckpt(src[1])
        if net is None:
            print(f"  skip {name}: missing")
            continue
        acc = args.get("acc") if isinstance(args, dict) else None
    with torch.no_grad():
        v = np.clip(net(torch.tensor(SEEDS)).numpy(), 0, 15)
    lo = v.min(1, keepdims=True); hi = v.max(1, keepdims=True)
    lv = np.rint(np.clip((v - lo) / np.maximum(hi - lo, 1e-6) * 15, 0, 15)).astype(int)
    # GUARD on a large sample, independent of the 32 seeds that get drawn. Judging frozen
    # from only 32 samples is unreliable -- a position that is usually the same value can show
    # a small sd by chance alone, which is what produced three false alarms on the first run.
    probe_rng = np.random.default_rng(99)
    PZ = np.clip(probe_rng.standard_normal((800, 4)), -3.5, 3.5).astype(np.float32)
    with torch.no_grad():
        pv = np.clip(net(torch.tensor(PZ)).numpy(), 0, 15)
    plo = pv.min(1, keepdims=True); phi = pv.max(1, keepdims=True)
    probe = np.clip((pv - plo) / np.maximum(phi - plo, 1e-6) * 15, 0, 15)
    frozen = int((probe.std(0) < 0.35).sum())
    # NOT every constant position is degeneration. A position that is always the darkest point
    # of the face is forced to exactly 0 by the min-max stretch, so its sd is 0 even though the
    # model is fine -- measured at 78-87% of faces for the three models that tripped this.
    # Real collapse looks like the broken 200-step checkpoint: 21 of 32 positions at once.
    degenerate = frozen > 4
    print(f"  {name:26s} {note:10s} 恒定位置 {frozen:2d}/32"
          + ("   <-- 退化" if degenerate else ("  （极端位置，正常）" if frozen else "")))
    fig.append({"name": name, "note": note, "imgs": layout.image_from_left_half(lv),
                "frozen": frozen, "degenerate": degenerate})

bad = [r for r in fig if r["degenerate"]]
if bad:
    for r in bad:
        print(f"  ABORT: {r['name']} has {r['frozen']} constant positions (degenerate)")
    raise SystemExit("refusing to draw from degenerate models")

# ---- draw
BLOCKW = COLS * (TILE + PAD)
BLOCKH = ROWB * (TILE + PAD)
LEFT = 286
W = LEFT + BLOCKW + 24
ROWGAP = 20
H = 118 + len(fig) * (BLOCKH + ROWGAP) + 40
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(22), font(13), font(11)
dr.text((12, 12), "灰度生成器全家族：同一批种子，逐列可比", font=f_t, fill=(234, 238, 244))
dr.text((12, 42), "每块 16×2 = 32 张，全部来自同一批随机种子；列 k 在每一行都是同一个输入。",
        font=f_s, fill=(150, 162, 178))
dr.text((12, 58), "上半组＝两层结构 4-h-h-32；下半组＝单层结构 4-h-32。已做 min-max 拉伸。",
        font=f_s, fill=(130, 142, 158))
dr.text((12, 74), "所有模型均通过冻结像素检测。", font=f_s, fill=(110, 200, 130))

y = 108
for i, r in enumerate(fig):
    if i == GROUP_BREAK:
        dr.line([12, y - 10, W - 12, y - 10], fill=(60, 70, 90), width=1)
    col = (110, 168, 254) if i < GROUP_BREAK else (255, 176, 192)
    dr.text((12, y + 4), r["name"], font=f_l, fill=col)
    dr.text((12, y + 22), r["note"] + " 乘法", font=f_s, fill=(150, 162, 178))
    for k, im in enumerate(r["imgs"]):
        rr, cc = divmod(k, COLS)
        t = np.kron(np.clip(im / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8))).astype(np.uint8)
        cv.paste(Image.fromarray(np.stack([t] * 3, -1)),
                 (LEFT + cc * (TILE + PAD), y + rr * (TILE + PAD)))
    y += BLOCKH + ROWGAP
dr.text((12, H - 24),
        "蓝色＝两层（原方案家族）；粉色＝单层（压缩家族）。同一列是同一个种子。",
        font=f_s, fill=(140, 152, 168))
p = os.path.join(OUT, "FAMILY_SHEET.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")
print("FAMILY_SHEET_DONE")

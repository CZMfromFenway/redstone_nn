#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Summarise the expansion sweep: does more training help, and where does the adversarial term fail.

Three things are produced:
    a table of final metrics per run
    training curves, because a metric that improves then degrades is a different story from one
    that plateaus, and only the curve shows which happened
    sample grids, because the metrics have already been shown to be gameable once in this project

The diagnostic that matters most for the adversarial question is D accuracy. If the discriminator
separates real from generated with near certainty, the gradient reaching the generator is close to
saturated and the term is doing little more than a constant push; if it stays near 0.5, the two
are matched and the term is doing real work.
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
OUT = os.path.join(ROOT, "results", "colour", "expand")
WEB = os.path.join(ROOT, "webapp")
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


TAGS = ["base12k", "base48k", "adv48k", "big48k", "bigadv48k"]
runs = {}
for t in TAGS:
    p = os.path.join(OUT, f"{t}.json")
    if os.path.exists(p):
        runs[t] = json.load(open(p, encoding="utf-8"))

print("=" * 108)
print("EXPANSION SWEEP -- final metrics")
print("=" * 108)
print(f"  {'tag':11s} {'steps':>6s} {'hid':>4s} {'adv':>4s} {'行方差':>7s} {'行跳变':>7s} "
      f"{'眼睛':>6s} {'边缘':>6s} {'颜色数':>6s} {'不重复率':>8s} {'D准确率':>7s}")
print("  " + "-" * 92)
for t, r in runs.items():
    f = r["final"]
    dlast = r["log"][-1]["d_acc"] if r["log"] and r["log"][-1]["d_acc"] == r["log"][-1]["d_acc"] else None
    print(f"  {t:11s} {r['steps']:6d} {r['hid']:4d} {r['adv']:4.1f} "
          f"{f['row_spread']:7.2f} {f['row_jump']:7.2f} {f['eye']:6.2f} {f['edge']:6.2f} "
          f"{f['ncol']:6.2f} {f['distinct']:8.3f} "
          + (f"{dlast:7.3f}" if dlast is not None else f"{'—':>7s}"))

print()
print("  不重复率＝2000 张样本里不同脸的比例，用来发现模式坍缩。真实数据分布下应接近 1.0。")

# ---- curves
def curve(tag, key, color, label, xs, ys, W, H, pad, canvas, draw, vmin=None, vmax=None):
    pass


CW, CH = 620, 300
PANELS = [("row_spread", "行亮度方差"), ("eye", "眼睛与皮肤对比"),
          ("distinct", "不重复率（模式坍缩检测）"), ("d_acc", "判别器 D 准确率")]
SW = CW * 2 + 60
SH = 92 + len(PANELS) * (CH + 46) + 30
cv = Image.new("RGB", (SW, SH), (18, 20, 24))
dr = ImageDraw.Draw(cv)
f_t, f_l, f_s = font(21), font(12), font(10)
dr.text((12, 12), "扩大训练与对抗：训练过程曲线", font=f_t, fill=(234, 238, 244))
dr.text((12, 40), "红线＝带对抗项；蓝线＝纯交叉熵。D 准确率接近 1 表示判别器碾压生成器。",
        font=f_s, fill=(150, 162, 178))
dr.text((12, 56), "不重复率跌破 0.8 就是模式坍缩的证据。", font=f_s, fill=(130, 142, 158))

COLORS = {"base12k": (110, 168, 254), "base48k": (110, 210, 254),
          "big48k": (150, 140, 254),
          "adv48k": (255, 130, 130), "bigadv48k": (255, 190, 120)}

y = 84
for pi, (key, title) in enumerate(PANELS):
    col = pi % 2
    row = pi // 2
    x0 = 20 + col * (CW + 20)
    y0 = 84 + row * (CH + 46)
    dr.rectangle([x0, y0, x0 + CW, y0 + CH], outline=(40, 46, 58))
    dr.text((x0 + 8, y0 + 5), title, font=f_l, fill=(200, 210, 226))
    vals = []
    for t, r in runs.items():
        pts = [(lg["step"], lg[key]) for lg in r["log"] if lg.get(key) == lg.get(key)]
        if pts:
            vals += [v for _, v in pts]
    if not vals:
        dr.text((x0 + 8, y0 + 40), "无数据", font=f_s, fill=(120, 130, 145))
        continue
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-6:
        hi = lo + 1
    maxstep = max(r["steps"] for r in runs.values())
    for t, r in runs.items():
        pts = [(lg["step"], lg[key]) for lg in r["log"] if lg.get(key) == lg.get(key)]
        if not pts:
            continue
        xy = []
        for s, v in pts:
            px = x0 + 40 + (s / maxstep) * (CW - 60)
            py = y0 + CH - 26 - ((v - lo) / (hi - lo)) * (CH - 56)
            xy.append((px, py))
        if len(xy) > 1:
            dr.line(xy, fill=COLORS.get(t, (200, 200, 200)), width=2)
        for px, py in xy:
            dr.ellipse([px - 2, py - 2, px + 2, py + 2],
                       fill=COLORS.get(t, (200, 200, 200)))
    dr.text((x0 + 8, y0 + CH - 18), f"{lo:.2f}", font=f_s, fill=(130, 142, 158))
    dr.text((x0 + CW - 52, y0 + CH - 18), f"{hi:.2f}", font=f_s, fill=(130, 142, 158))

# legend
ly = SH - 24
lx = 14
for t in runs:
    dr.rectangle([lx, ly - 6, lx + 10, ly + 4], fill=COLORS.get(t, (200, 200, 200)))
    dr.text((lx + 15, ly - 6), t, font=f_s, fill=(180, 190, 205))
    lx += 100
p = os.path.join(OUT, "EXPAND_CURVES.png")
cv.save(p)
print(f"\nwrote {p}  {cv.size}")

# ---- sample grids per run
M = 16
Z = np.clip(np.random.default_rng(77).standard_normal((M, 8)), -3.5, 3.5).astype(np.float32)
rows_img = []
for t, r in runs.items():
    ck = torch.load(os.path.join(OUT, f"{t}.pt"), weights_only=False)
    st = ck["G"]; pal = np.array(ck["palette"]); Kx = len(pal)
    idxs = sorted({int(k.split(".")[0]) for k in st if k.endswith(".weight")})
    seq = []
    for j in idxs:
        seq.append(nn.Linear(int(st[f"{j}.weight"].shape[1]), int(st[f"{j}.weight"].shape[0])))
        if j != idxs[-1]:
            seq.append(nn.ReLU())
    m = nn.Sequential(*seq); m.load_state_dict(st); m.eval()
    with torch.no_grad():
        ii = m(torch.tensor(Z)).view(-1, 32, Kx).argmax(-1).numpy()
    faces = pal[ii].reshape(len(Z), 96)
    a = np.clip(faces, 0, 15).reshape(len(Z), 32, 3)
    img = np.zeros((len(a), 8, 8, 3))
    img[:, layout.POS_ROW, layout.POS_COL] = a
    img[:, layout.POS_ROW, layout.MIRROR_COL] = a
    rows_img.append((t, img, r["final"]))

TILE, PAD = 46, 4
W2 = 300 + M * (TILE + PAD)
H2 = 96 + len(rows_img) * (TILE + 22)
cv2 = Image.new("RGB", (W2, H2), (18, 20, 24))
dr2 = ImageDraw.Draw(cv2)
dr2.text((12, 12), "同一批输入下各配置的样子", font=f_t, fill=(234, 238, 244))
dr2.text((12, 38), "逐列可比。指标可以骗人，图不行。", font=f_s, fill=(150, 162, 178))
yy = 68
for t, img, f in rows_img:
    dr2.text((12, yy + 5), t, font=f_l, fill=COLORS.get(t, (200, 200, 200)))
    dr2.text((12, yy + 21), f"眼 {f['eye']:.1f} 跳 {f['row_jump']:.1f}", font=f_s,
             fill=(150, 162, 178))
    for i in range(M):
        tt = np.kron(np.clip(img[i] / 15.0 * 255, 0, 255).astype(np.uint8),
                     np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        cv2.paste(Image.fromarray(tt), (300 + i * (TILE + PAD), yy))
    yy += TILE + 22
p2 = os.path.join(OUT, "EXPAND_SAMPLES.png")
cv2.save(p2)
print(f"wrote {p2}  {cv2.size}")
print("EXPAND_SUMMARY_DONE")

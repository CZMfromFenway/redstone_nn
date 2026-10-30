#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Stage-2 figure, assembled only from artefacts produced on 2026-10-16 evening.

WHY IT IS BUILT HERE
The document previously cited `results/pca_sweep/PCA48_SHOWCASE.png`, which was rendered on
10-17 15:15 -- a day after the condition sweep it was supposed to illustrate. A figure that
post-dates its stage misrepresents when the evidence existed.

The condition sweep wrote one `COMPARE_real_top_generated_bottom.png` per arm at 19:48-20:04.
Each is 1024x268: the real reference in the top band, the generated faces below. The real band is
identical across arms, so it is shown once and the eight generated bands are stacked beneath it.
"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "sweep", "STAGE2_CONDITIONS.png")

ARMS = [
    ("uncond", "无条件", "仅噪声"),
    ("half", "半脸", "已知左半 + 噪声"),
    ("code3", "编码 3 维", "3 维编码 + 噪声"),
    ("code5", "编码 5 维", "5 维编码 + 噪声"),
    ("pca4", "PCA-4", "前 4 个主成分 + 噪声"),
    ("pca8", "PCA-8", "前 8 个主成分 + 噪声"),
    ("pca16", "PCA-16", "前 16 个主成分 + 噪声"),
    ("pca22", "PCA-22", "前 22 个主成分 + 噪声"),
]


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def load_bands(arm):
    p = os.path.join(RES, "sweep", f"sweep_{arm}", "COMPARE_real_top_generated_bottom.png")
    if not os.path.exists(p):
        return None, None
    im = Image.open(p).convert("RGB")
    w, h = im.size
    half = h // 2
    return im.crop((0, 0, w, half)), im.crop((0, half, w, h))


real = None
rows = []
for arm, name, desc in ARMS:
    top, bot = load_bands(arm)
    if top is None:
        print(f"  missing sweep_{arm}")
        continue
    if real is None:
        real = top
    rows.append((name, desc, bot))
    print(f"  sweep_{arm}: band {bot.size}")

BW, BH = rows[0][2].size
LEFT = 200
f_t, f_l, f_s = font(19), font(13), font(10)
W = LEFT + BW + 16
H = 66 + (1 + len(rows)) * (BH + 8) + 30
cv = Image.new("RGB", (W, H), (18, 20, 24))
dr = ImageDraw.Draw(cv)
dr.text((12, 12), "二期的八种条件：同上为真实数据，下为各条件下的生成结果", font=f_t,
        fill=(234, 238, 244))
dr.text((12, 38), "全部为 2026-10-16 19:48–20:04 的同一轮扫描，每臂 1,500 步、batch 256、对抗损失加交叉熵。",
        font=f_s, fill=(150, 162, 178))

y = 62
dr.text((12, y + BH // 2 - 8), "真实数据", font=f_l, fill=(110, 200, 130))
cv.paste(real, (LEFT, y))
y += BH + 8
for name, desc, band in rows:
    col = (255, 170, 110) if name in ("无", "半脸") else (200, 210, 226)
    dr.text((12, y + BH // 2 - 14), name, font=f_l, fill=col)
    dr.text((12, y + BH // 2 + 2), desc, font=f_s, fill=(130, 142, 158))
    cv.paste(band, (LEFT, y))
    y += BH + 8

cv.save(OUT)
print(f"\nwrote {OUT}  {cv.size}")
print("STAGE2_FIG_DONE")

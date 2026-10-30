#!/usr/bin/env -u python
# -*- coding: utf-8 -*-
r"""
Number the colour faces the user called acceptable, so they can point at the best of the best.

The labels store the post-stretch 96 RGB values, so each face is redrawn exactly as it was
shown during labelling. Numbering follows the order in labels.jsonl (chronological), which is
the order the faces appeared.

Three sheets:
    COLOUR_GOOD_INDEXED.png    the 67 acceptable faces, numbered, large
    COLOUR_MID_INDEXED.png     the 中 faces, so the "blurry mass" can be compared directly
    COLOUR_INDEX.csv           index -> grade, timestamp, so a reply of "3,7,12" is unambiguous
"""
import csv
import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "colour")
LABELS = os.path.join(ROOT, "results", "preference", "labels.jsonl")

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


STYLE = {4: "类二次元", 3: "类游戏", 2: "类三次元", 1: "中", 0: "差"}
recs = [json.loads(l) for l in open(LABELS, encoding="utf-8") if l.strip()]
r6 = [r for r in recs if str(r.get("round") or "") == "6" and len(r.get("predicted_levels") or []) == 96]
print(f"colour labels with stored pixels: {len(r6)}")


def face(r):
    v = np.clip(np.array(r["predicted_levels"], dtype=np.float32), 0, 15).reshape(32, 3)
    img = np.zeros((8, 8, 3), dtype=np.float32)
    img[layout.POS_ROW, layout.POS_COL] = v
    img[layout.POS_ROW, layout.MIRROR_COL] = v
    return img


def sheet(rows, title, sub, path, cols=12, px=76):
    TILE, PAD, LAB = px, 6, 16
    n = len(rows)
    r_ = (n + cols - 1) // cols
    W = 20 + cols * (TILE + PAD) + PAD
    H = 92 + r_ * (TILE + PAD + LAB) + 26
    cv = Image.new("RGB", (W, H), (18, 20, 24))
    dr = ImageDraw.Draw(cv)
    f_t, f_l, f_s = font(20), font(13), font(11)
    dr.text((12, 12), title, font=f_t, fill=(234, 238, 244))
    dr.text((12, 38), sub, font=f_s, fill=(150, 162, 178))
    dr.text((12, 54), "编号就是回复时用的编号（按标注顺序）。", font=f_s, fill=(130, 142, 158))
    y = 80
    for i, r in enumerate(rows):
        c = i % cols
        if c == 0 and i:
            y += TILE + PAD + LAB
        im = face(r)
        t = np.kron(np.clip(im / 15.0 * 255, 0, 255).astype(np.uint8),
                    np.ones((TILE // 8, TILE // 8, 1))).astype(np.uint8)
        x = 20 + c * (TILE + PAD)
        cv.paste(Image.fromarray(t), (x, y))
        dr.text((x + 2, y + TILE + 2), f"#{r['_idx']}", font=f_l, fill=(110, 168, 254))
    cv.save(path)
    print(f"wrote {path}  {cv.size}  ({n} faces)")


for i, r in enumerate(r6):
    r["_idx"] = i + 1

good = [r for r in r6 if r["grade"] >= 2]
mid = [r for r in r6 if r["grade"] == 1]
bad = [r for r in r6 if r["grade"] == 0]

sheet(good,
      f"你评为「可接受」的彩色脸（{len(good)} 张）",
      "类二次元 / 类游戏 / 类三次元 都算。请告诉我哪些是优中之优（用 # 编号）。",
      os.path.join(OUT, "COLOUR_GOOD_INDEXED.png"))
sheet(mid,
      f"你评为「中」的彩色脸（{len(mid)} 张）",
      "按你的说法，这些大多是色彩糊成一团。可以用来对照什么是「清晰」。",
      os.path.join(OUT, "COLOUR_MID_INDEXED.png"))

with open(os.path.join(OUT, "COLOUR_INDEX.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["index", "grade", "label", "timestamp", "colour_pcs"])
    for r in r6:
        w.writerow([r["_idx"], r["grade"], STYLE[r["grade"]], r.get("ts", ""),
                    " ".join(f"{v:.3f}" for v in r["pca"][:8])])
print(f"wrote {os.path.join(OUT, 'COLOUR_INDEX.csv')}")
print(f"  good {len(good)}  mid {len(mid)}  bad {len(bad)}")
print("COLOUR_INDEX_DONE")

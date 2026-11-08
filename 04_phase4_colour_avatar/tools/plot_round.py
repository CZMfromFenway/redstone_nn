#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
把一轮盲测的等级分布画成堆叠条：一眼看出各臂是"泥沙俱下"还是"不烂也不惊艳"。

用法：python plot_round.py --round 1
产物：results/labels/round<r>_dist.png
"""
import argparse
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
P4 = os.path.dirname(HERE)
REPO = os.path.dirname(P4)
LDIR = os.path.join(P4, "results", "labels")

COLS = {0: (150, 40, 40), 1: (196, 106, 60), 2: (176, 168, 96),
        3: (108, 176, 108), 4: (74, 134, 232)}


def font(sz, bold=False):
    for fn in ("msyhbd.ttc" if bold else "msyh.ttc", "simhei.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="1")
    A = ap.parse_args()
    rows = [json.loads(l) for l in
            open(os.path.join(LDIR, f"colour_round{A.round}.jsonl"), encoding="utf-8")
            if l.strip()]
    arms = {}
    for r in rows:
        arms.setdefault(r["model"], []).append(r["grade"])
    order = sorted(arms, key=lambda m: -float(np.mean(arms[m])))

    LW, BARX, BARW, BH, GAP = 200, 210, 760, 34, 26
    W = BARX + BARW + 220
    H = 120 + len(order) * (BH + GAP)
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((16, 14), f"盲测第 {A.round} 轮：等级分布（{len(rows)} 条）", font=font(21),
            fill=(234, 238, 244))
    dr.text((16, 44), "每条 = 一个臂；长度 = 该臂的 n；颜色 = 等级 0..4。右端数字是"
                      "「好脸率 ≥3」与「可接受率 ≥2」。", font=font(12), fill=(148, 160, 176))
    # 图例
    x0 = 16
    for g in range(5):
        dr.rectangle([x0, 74, x0 + 16, 90], fill=COLS[g])
        dr.text((x0 + 22, 74), str(g), font=font(11), fill=(200, 210, 226))
        x0 += 46

    y = 104
    for m in order:
        g = np.array(arms[m])
        n = len(g)
        dr.text((16, y + 8), m, font=font(14, True), fill=(112, 170, 254))
        x = BARX
        for k in range(5):
            w = BARW * float((g == k).mean())
            if w <= 0:
                continue
            dr.rectangle([x, y, x + w - 1, y + BH], fill=COLS[k])
            if w > 22:
                dr.text((x + 5, y + 9), str(int((g == k).sum())), font=font(11),
                        fill=(18, 20, 24))
            x += w
        dr.text((BARX + BARW + 12, y + 4), f"好 {float((g>=3).mean())*100:.0f}%",
                font=font(11), fill=(140, 200, 150))
        dr.text((BARX + BARW + 12, y + 18), f"可接受 {float((g>=2).mean())*100:.0f}%",
                font=font(11), fill=(150, 162, 178))
        y += BH + GAP

    out = os.path.join(LDIR, f"round{A.round}_dist.png")
    cv.save(out)
    print(f"wrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()

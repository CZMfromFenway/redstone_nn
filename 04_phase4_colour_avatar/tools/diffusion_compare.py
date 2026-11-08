#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
把若干臂的样本拼成一张「逐行一臂」的对照图 —— 回答"对称头像 diffusion 到底长什么样"。

  python diffusion_compare.py --out results/diffusion/compare_arms.png ^
      --row real:results/diffusion/real_n200.json ^
      --row cur:results/diffusion/cur_n200.json ^
      --row "mlp-x0:results/diffusion/mlp96_h256x_s10.json"

每行取前 --per 张（同一行内的样本互不相同；不同行之间**不是同一批种子**，
所以只能看"分布像不像"，不能逐列比）。
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


def font(sz, bold=False):
    for fn in (("msyhbd.ttc" if bold else "msyh.ttc"), "simhei.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def hex2rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def feats(rows):
    """与 jim_diffusion_gallery.stats 逐字同口径：颜色按 0..255 算，边缘再换算成「级」。"""
    import math
    ncol, sat, edge, lum, eye = [], [], [], [], []
    for r in rows:
        hs = r["snapped"]
        rgb = [hex2rgb(h) for h in hs]                       # 0..255
        ncol.append(len(set(hs)))
        s, l = [], []
        for c0, c1, c2 in rgb:
            mx, mn = max(c0, c1, c2), min(c0, c1, c2)
            s.append(0.0 if mx <= 1e-6 else (mx - mn) / mx)
            l.append((0.299 * c0 + 0.587 * c1 + 0.114 * c2) / 255.0)
        sat.append(sum(s) / 64)
        lum.append(sum(l) / 64)
        g = []
        for rr in range(8):
            for cc in range(8):
                aa = rgb[rr * 8 + cc]
                for dr, dc in ((0, 1), (1, 0)):
                    r2, c2 = rr + dr, cc + dc
                    if r2 > 7 or c2 > 7:
                        continue
                    g.append(math.dist(aa, rgb[r2 * 8 + c2]))
        g.sort()
        edge.append(g[int(0.9 * (len(g) - 1))])
        vals = sorted(l[r2 * 8 + c2] for r2 in range(2, 7) for c2 in range(1, 7))
        base = vals[len(vals) // 2]
        best = 0
        for r2 in range(2, 5):
            run = 0
            for c2 in range(1, 7):
                if l[r2 * 8 + c2] < base - 0.16:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
        eye.append(best)

    def med(v):
        v = sorted(v)
        return v[len(v) // 2]

    return {"n": len(rows), "ncol": round(float(np.mean(ncol)), 2),
            "sat": round(float(np.mean(sat)), 3), "edge": round(float(np.mean(edge)), 2),
            "lum": round(float(np.mean(lum)), 2), "eye": med(eye)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", required=True, help="标签:json路径")
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--cell", type=int, default=34)
    ap.add_argument("--out", default="results/diffusion/compare_arms.png")
    a = ap.parse_args()

    items = []
    for spec in a.row:
        label, path = spec.split(":", 1)
        p = path if os.path.isabs(path) else os.path.join(P4, path)
        j = json.load(open(p, encoding="utf-8"))
        rows = j["samples"][:a.per]
        extra = ""
        if j.get("mac_per_pass"):
            extra = " · 体积 %s" % ("{:,}".format(int(j["mac_per_pass"])))
        if j.get("steps"):
            extra += " · %s 步" % j["steps"]
        items.append((label + extra, rows, feats(j["samples"])))

    cell = a.cell
    face = 8 * cell
    gap = 5
    labw = 250
    pad = 30
    W = labw + a.per * (face + gap) + gap + 210
    H = pad + len(items) * (face + 16 + gap)
    cv = Image.new("RGB", (W, H), (250, 248, 242))
    dr = ImageDraw.Draw(cv)
    f = font(12)
    fb = font(13, True)
    dr.text((gap, 7), "对称 8x8 头像：真实数据 / 现役管线 / 各种 diffusion（每行 %d 张，行内互不相同）" % a.per,
            fill=(40, 40, 40), font=fb)

    for ri, (label, rows, st) in enumerate(items):
        y = pad + ri * (face + 16 + gap)
        dr.text((gap + 2, y + 4), label, fill=(30, 30, 30), font=f)
        dr.text((gap + 2, y + 22), "色数 %.1f 饱和 %.2f" % (st["ncol"], st["sat"]),
                fill=(110, 110, 110), font=f)
        dr.text((gap + 2, y + 38), "边缘 %.1f 级 眼带 %d" % (st["edge"] / 17.0, st["eye"]),
                fill=(110, 110, 110), font=f)
        for ci, r in enumerate(rows):
            x0 = labw + ci * (face + gap)
            for k, h in enumerate(r["snapped"][:64]):
                rr, cc = k // 8, k % 8
                dr.rectangle([x0 + cc * cell, y + rr * cell,
                              x0 + (cc + 1) * cell - 1, y + (rr + 1) * cell - 1],
                             fill=hex2rgb(h))
        dr.line([(0, y + face + 14), (W, y + face + 14)], fill=(225, 220, 210))

    out = os.path.join(P4, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv.save(out)
    print("图：%s" % os.path.relpath(out, P4))
    for label, _, st in items:
        print("  %-52s ncol %.2f  sat %.3f  edge(0-255) %.1f  lum %.2f  eye %d"
              % (label, st["ncol"], st["sat"], st["edge"], st["lum"], st["eye"]))


if __name__ == "__main__":
    main()

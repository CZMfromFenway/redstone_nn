#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
「为什么会出现几乎一样的图」——把每个样本池内部的重复结构摊开：

  1. 精确重复（同一张图出现两次）
  2. 近似重复：逐格比较，统计"只差 k 格"的样本对有多少
  3. 出图：每个池列出最像的若干对（左 A / 右 B，标出差异格数）

判据：距离 = 64 格里颜色不同的格数（0 = 完全相同）。

用法：python near_dup_report.py [--per 6]
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
DIFF = os.path.join(P4, "results", "diffusion")
OUT = os.path.join(P4, "results", "labels")

POOLS = [
    ("real", "真实数据（200 张）", "real_n200.json"),
    ("cur_a", "现役 cur_a（200 张）", "cur_n200.json"),
    ("cur_b", "现役 cur_b（200 张）", "cur_b_n200.json"),
    ("tokM_rand", "token mask 扩散（200 张）", "tokM_h192_rand_s10.json"),
    ("mlp_x0", "MLP 连续扩散（200 张）", "mlp96_h256x_s10.json"),
]


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


def draw_face(dr, hexes, x0, y0, cell):
    for k, h in enumerate(hexes[:64]):
        r, c = k // 8, k % 8
        dr.rectangle([x0 + c * cell, y0 + r * cell,
                      x0 + (c + 1) * cell - 1, y0 + (r + 1) * cell - 1], fill=hex2rgb(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=5, help="每个池画几对最像的")
    ap.add_argument("--maxdist", type=int, default=16, help="只统计差 <= 这个格数的对")
    a = ap.parse_args()

    cell = 30
    face = 8 * cell
    gap = 8
    labw = 210
    per = a.per
    W = labw + per * (2 * face + 3 * gap) + gap
    H = 30 + len(POOLS) * (face + 18 + gap)
    cv = Image.new("RGB", (W, H), (250, 248, 242))
    dr = ImageDraw.Draw(cv)
    f, fb = ImageFont.load_default(), ImageFont.load_default()
    f, fb = font(11), font(13, True)
    dr.text((gap, 7), "每个样本池内部最像的 %d 对（左 A / 右 B；差异格数越小越像；0 = 完全相同）" % per,
            fill=(40, 40, 40), font=fb)

    print("%-10s %5s %6s %8s %8s %8s %9s" % ("池", "张数", "精确重复", "差<4", "差<8", "差<16", "最小差"))
    for ri, (key, label, fn) in enumerate(POOLS):
        j = json.load(open(os.path.join(DIFF, fn), encoding="utf-8"))
        S = [tuple(s["snapped"]) for s in j["samples"]]
        n = len(S)
        arr = np.array([list(s) for s in S])
        exact = n - len(set(S))
        # 全量两两比较（200 张 = 19900 对，够快）
        best = []
        cnt = {4: 0, 8: 0, 16: 0}
        for i in range(n):
            d = (arr[i + 1:] != arr[i]).sum(1)
            for k in (4, 8, 16):
                cnt[k] += int((d < k).sum())
            for off in np.argsort(d)[:3]:
                if d[off] < a.maxdist:
                    best.append((int(d[off]), i, i + 1 + int(off)))
        best.sort()
        seen, pairs = set(), []
        for d, i, k in best:
            if i in seen or k in seen:
                continue
            seen.update((i, k))
            pairs.append((d, i, k))
            if len(pairs) >= per:
                break
        print("%-10s %5d %6d %8d %8d %8d %9d"
              % (key, n, exact, cnt[4], cnt[8], cnt[16], pairs[0][0] if pairs else -1))

        y = 30 + ri * (face + 18 + gap)
        dr.text((gap + 2, y + 4), label, fill=(30, 30, 30), font=f)
        dr.text((gap + 2, y + 22), "差<8: %d 对" % cnt[8], fill=(110, 110, 110), font=f)
        dr.text((gap + 2, y + 38), "精确重复 %d" % exact, fill=(110, 110, 110), font=f)
        for ci, (d, i, k) in enumerate(pairs):
            x0 = labw + ci * (2 * face + 3 * gap)
            draw_face(dr, S[i], x0, y, cell)
            draw_face(dr, S[k], x0 + face + gap, y, cell)
            col = (200, 40, 40) if d == 0 else (90, 90, 90)
            dr.text((x0, y + face + 1), "差 %d 格 · #%d / #%d" % (d, i, k), fill=col, font=f)
        dr.line([(0, y + face + 14), (W, y + face + 14)], fill=(225, 220, 210))

    out = os.path.join(OUT, "near_duplicates.png")
    cv.save(out)
    print("\n图：%s" % os.path.relpath(out, P4))


if __name__ == "__main__":
    main()

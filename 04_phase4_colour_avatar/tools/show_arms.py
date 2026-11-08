#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
把 results/train/<tag>/model.json 渲染成一张对照图：每个臂一行，同一批随机输入。

用法：python show_arms.py K16_w128_k8 K16_w128_k16 K16_w128_k24 K16_w128_k32
产物：04_phase4_colour_avatar/results/prep/arms_<n>.png
"""
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
FIG = os.path.join(P4, "results", "prep")
os.makedirs(FIG, exist_ok=True)


def font(sz):
    for fn in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


class Net:
    def __init__(self, path):
        j = json.load(open(path, encoding="utf-8"))
        self.pal = np.array(j["palette"], dtype=np.float64)
        self.layers = []
        for L in j["body"]:
            self.layers.append((np.array(L["W"], dtype=np.float64),
                                np.array(L["b"], dtype=np.float64), True))
        self.layers.append((np.array(j["head"]["W"], dtype=np.float64),
                            np.array(j["head"]["b"], dtype=np.float64), False))
        self.meta = j["meta"]
        self.K = len(self.pal)

    def __call__(self, x):
        h = x
        for W, b, relu in self.layers:
            h = h @ W.T + b
            if relu:
                h = np.maximum(h, 0)
        idx = h.reshape(len(x), 32, self.K).argmax(-1)
        rows = np.array([p % 8 for p in range(32)])
        cols = np.array([p // 8 for p in range(32)])
        rgb = self.pal[idx]
        img = np.zeros((len(x), 8, 8, 3))
        img[:, rows, cols] = rgb
        img[:, rows, 7 - cols] = rgb
        return img


def main():
    tags = sys.argv[1:]
    if not tags:
        print("usage: python show_arms.py TAG [TAG ...]")
        return
    rng = np.random.default_rng(7)
    seed = np.clip(rng.standard_normal((20, 32)), -3.5, 3.5)
    rows = []
    for t in tags:
        p = os.path.join(P4, "results", "train", t, "model.json")
        net = Net(p)
        k = net.meta["k"]
        img = net(seed[:, :k])
        rows.append((t, net.meta, img))
        print(f"  {t}: k={k} mults={net.meta['mults']:,} K={net.K}")

    COLS, TILE, PAD = 20, 48, 4
    LW = 320
    W = LW + COLS * (TILE + PAD) + 20
    H = 100 + len(rows) * (TILE + 34) + 26
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((14, 12), "四期 E1：PCA 维数 × 16 色（同一批 20 个随机输入）", font=font(20),
            fill=(234, 238, 244))
    dr.text((14, 42), "每行一个臂；列是同一个输入，所以逐列可比。行末是位置准确率 / 同色率 / 平均色数。",
            font=font(12), fill=(148, 160, 176))
    dr.text((14, 60), "准确率只在同一调色板内可比（medoid 臂的调色板更粗，准确率会虚高）。",
            font=font(12), fill=(148, 160, 176))
    y = 88
    for t, meta, img in rows:
        dr.text((14, y + 4), t, font=font(13), fill=(112, 170, 254))
        dr.text((14, y + 22), f"k={meta['k']} · {meta['mults']:,} 乘法 · K={meta['K']}",
                font=font(10), fill=(150, 162, 178))
        for i in range(COLS):
            v = np.clip(np.rint(img[i] * 17), 0, 255).astype(np.uint8)
            tile = np.repeat(np.repeat(v, TILE // 8, 0), TILE // 8, 1)
            cv.paste(Image.fromarray(tile), (LW + i * (TILE + PAD), y))
        y += TILE + 34
    out = os.path.join(FIG, f"arms_{len(rows)}.png")
    cv.save(out)
    print(f"wrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()

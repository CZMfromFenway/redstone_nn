#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
任意一轮盲测的描框图（同时支持**现场生成**的臂与**预计算池**的臂）。

  python round_gallery.py --round 14 --man label_manifest_diffusion14.json --per 14

现场生成的臂：标签里带 `cols`（96 个 0..15 的颜色值）→ 直接还原成 8x8（行内镜像）。
预计算池的臂：标签里带 `sample`（池内下标）→ 从清单指到的样本 JSON 里取 `snapped`。

产物：results/labels/round<N>_graded.png（行=臂、按等级降序、框色=等级）
      results/labels/round<N>_badtail.png（本轮全部 0/1 分）
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
LDIR = os.path.join(P4, "results", "labels")
ROW = np.array([p % 8 for p in range(32)])
COL = np.array([p // 8 for p in range(32)])
GRADE_COL = {4: (0, 214, 214), 3: (64, 196, 96), 2: (232, 200, 64),
             1: (240, 148, 48), 0: (228, 64, 64)}


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


def grid_from_cols(cols):
    """96 个 0..15 的颜色值（32 位置 x RGB）-> 8x8 的 RGB 0..255。"""
    a = np.asarray(cols, dtype=np.float64).reshape(32, 3)
    img = np.zeros((8, 8, 3))
    img[ROW, COL] = a
    img[ROW, 7 - COL] = a
    return np.round(np.clip(img, 0, 15) * 17).astype(int)


def grid_from_hex(hs):
    g = np.array([hex2rgb(h) for h in hs[:64]], dtype=int).reshape(8, 8, 3)
    return g


def jitter(g, by):
    """同色块的相邻格做一点亮度扰动，让 8x8 放大后边界看得见（只影响显示）。"""
    if by <= 0:
        return g
    out = g.astype(float).copy()
    for k in range(1, 8):
        out[:, k] += by * (1 if k % 2 else -1)
    out[:, 1:] -= by * 0.6
    return np.clip(out, 0, 255).astype(int)


def draw_face(dr, g, x0, y0, cell, grade=None):
    for r in range(8):
        for c in range(8):
            col = tuple(int(v) for v in g[r, c])
            dr.rectangle([x0 + c * cell, y0 + r * cell,
                          x0 + (c + 1) * cell - 1, y0 + (r + 1) * cell - 1], fill=col)
    if grade is not None:
        dr.rectangle([x0, y0, x0 + 8 * cell, y0 + 8 * cell],
                     outline=GRADE_COL[grade], width=max(2, cell // 9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--man", default="")
    ap.add_argument("--per", type=int, default=14)
    ap.add_argument("--cell", type=int, default=30)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(LDIR, "colour_round%s.jsonl" % a.round),
                                        encoding="utf-8") if l.strip()]
    man_path = os.path.join(HERE, a.man) if a.man else None
    pools = {}
    if man_path and os.path.exists(man_path):
        man = json.load(open(man_path, encoding="utf-8"))
        for arm in man["arms"]:
            if arm.get("precomputed"):
                p = os.path.normpath(os.path.join(HERE, man_path and arm["file"]))
                pools[arm["name"]] = [s["snapped"] for s in
                                      json.load(open(p, encoding="utf-8"))["samples"]]

    arms = []
    for r in rows:
        if r["model"] not in arms:
            arms.append(r["model"])
    # 按 ≥3 率降序排列，读起来舒服
    def rate(m):
        g = [x["grade"] for x in rows if x["model"] == m]
        return sum(1 for v in g if v >= 3) / max(1, len(g))
    arms.sort(key=lambda m: -rate(m))

    def image_of(r):
        if r.get("cols"):
            return grid_from_cols(r["cols"])
        if r.get("sample") is not None and r["model"] in pools:
            return grid_from_hex(pools[r["model"]][r["sample"]])
        return None

    per = a.per
    cell = a.cell
    face = 8 * cell
    gap = 5
    labw = 232
    W = labw + per * (face + gap) + gap
    H = 30 + len(arms) * (face + 30 + gap)
    cv = Image.new("RGB", (W, H), (250, 248, 242))
    dr = ImageDraw.Draw(cv)
    f, fb = font(12), font(14, True)
    dr.text((gap, 7), "第 %s 轮盲测（每臂 %d 张，按等级降序；框色 4=青 3=绿 2=黄 1=橙 0=红）"
            % (a.round, per), fill=(40, 40, 40), font=fb)

    print("%-12s %4s %6s %8s %8s %8s   %s" % ("臂", "n", "均值", "≥2", "≥3", "=4", "分布 0/1/2/3/4"))
    for ri, m in enumerate(arms):
        items = [r for r in rows if r["model"] == m]
        items.sort(key=lambda r: (-r["grade"], r.get("sample") if r.get("sample") is not None else 0))
        g = [r["grade"] for r in items]
        hist = [sum(1 for x in g if x == k) for k in range(5)]
        print("%-12s %4d %6.2f %7.0f%% %7.0f%% %8d   %s"
              % (m, len(g), sum(g) / len(g),
                 100 * sum(1 for x in g if x >= 2) / len(g),
                 100 * sum(1 for x in g if x >= 3) / len(g), hist[4], hist))
        y = 30 + ri * (face + 30 + gap)
        live = any(r.get("cols") for r in items)
        dr.text((gap + 2, y + 6), m, fill=(30, 30, 30), font=fb)
        dr.text((gap + 2, y + 26), "n=%d 均值 %.2f  %s" % (len(g), sum(g) / len(g),
                                                          "现场生成" if live else "预计算池"),
                fill=(110, 110, 110), font=f)
        dr.text((gap + 2, y + 42), "≥3 %.0f%%   ≥2 %.0f%%"
                % (100 * sum(1 for x in g if x >= 3) / len(g),
                   100 * sum(1 for x in g if x >= 2) / len(g)),
                fill=(110, 110, 110), font=f)
        for ci, r in enumerate(items[:per]):
            gi = image_of(r)
            if gi is None:
                continue
            x0 = labw + ci * (face + gap)
            draw_face(dr, gi, x0, y, cell, r["grade"])
            dr.text((x0 + 2, y + face + 1), str(r["grade"]), fill=(90, 90, 90), font=f)
        dr.line([(0, y + face + 26), (W, y + face + 26)], fill=(225, 220, 210))

    out1 = os.path.join(LDIR, "round%s_graded.png" % a.round)
    cv.save(out1)
    print("\n图：%s" % os.path.relpath(out1, P4))

    # ---- 0/1 分那一批
    bad = [r for r in rows if r["grade"] <= 1]
    if bad:
        print("被判 <2 的：", dict(Counter(r["model"] for r in bad)), " 合计", len(bad))
        ncol = max(1, min(12, len(bad)))
        cell2 = 44
        face2 = 8 * cell2
        gap2 = 6
        nrow = (len(bad) + ncol - 1) // ncol
        W2 = gap2 + ncol * (face2 + gap2)
        H2 = 34 + nrow * (face2 + 18 + gap2)
        cv2 = Image.new("RGB", (W2, H2), (250, 248, 242))
        dr2 = ImageDraw.Draw(cv2)
        dr2.text((gap2, 8), "第 %s 轮被判 0/1 的全部 %d 张（左上角是分数）" % (a.round, len(bad)),
                 fill=(40, 40, 40), font=fb)
        bad.sort(key=lambda r: (r["model"], r["grade"]))
        for i, r in enumerate(bad):
            gi = image_of(r)
            if gi is None:
                continue
            rr, cc = i // ncol, i % ncol
            x0 = gap2 + cc * (face2 + gap2)
            y0 = 34 + rr * (face2 + 18 + gap2)
            draw_face(dr2, gi, x0, y0, cell2, r["grade"])
            dr2.text((x0 + 3, y0 + 3), str(r["grade"]), fill=(0, 0, 0), font=fb)
            dr2.text((x0 + 2, y0 + face2 + 2), r["model"], fill=(90, 90, 90), font=f)
        out2 = os.path.join(LDIR, "round%s_badtail.png" % a.round)
        cv2.save(out2)
        print("图：%s" % os.path.relpath(out2, P4))


if __name__ == "__main__":
    main()

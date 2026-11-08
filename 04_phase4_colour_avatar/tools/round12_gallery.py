#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
第十二轮盲测的出图与补充分统计（4 个**预计算**臂：real / cur_pal / mlp_x0 / tokM_rand）。

  python round12_gallery.py            # 两张图 + 一段统计
  python round12_gallery.py --per 14

产物：
  results/labels/round12_graded.png    行=臂、格=脸、框色=等级（按等级降序排）
  results/labels/round12_badtail.png   只看"被判 <2"的那些脸，用来复核那个反直觉的数
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
LDIR = os.path.join(P4, "results", "labels")
DIFF = os.path.join(P4, "results", "diffusion")

ARMS = [
    ("real", "真实数据（人判基线）", "real_n200.json"),
    ("cur_pal", "现役：灰度+16色头+换调色板+4选1", "cur_n200.json"),
    ("mlp_x0", "MLP 连续 x0 扩散 10 步", "mlp96_h256x_s10.json"),
    ("tokM_rand", "MLP token(mask) 扩散 + 采样揭盖 10 步", "tokM_h192_rand_s10.json"),
]

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


def load():
    rows = [json.loads(l) for l in open(os.path.join(LDIR, "colour_round12.jsonl"),
                                        encoding="utf-8") if l.strip()]
    pools = {}
    for key, _, fn in ARMS:
        j = json.load(open(os.path.join(DIFF, fn), encoding="utf-8"))
        pools[key] = [s["snapped"] for s in j["samples"]]
    return rows, pools


def draw_face(dr, hexes, x0, y0, cell, grade=None, idx=None):
    for k, h in enumerate(hexes[:64]):
        r, c = k // 8, k % 8
        dr.rectangle([x0 + c * cell, y0 + r * cell,
                      x0 + (c + 1) * cell - 1, y0 + (r + 1) * cell - 1], fill=hex2rgb(h))
    if grade is not None:
        dr.rectangle([x0, y0, x0 + 8 * cell, y0 + 8 * cell],
                     outline=GRADE_COL[grade], width=max(2, cell // 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=14)
    a = ap.parse_args()
    rows, pools = load()

    # ---- 图 1：行=臂，按等级降序
    per = a.per
    cell = 30
    face = 8 * cell
    gap = 5
    labw = 240
    pad = 30
    W = labw + per * (face + gap) + gap
    H = pad + len(ARMS) * (face + 16 + gap)
    cv = Image.new("RGB", (W, H), (250, 248, 242))
    dr = ImageDraw.Draw(cv)
    f, fb = font(12), font(13, True)
    dr.text((gap, 7), "第十二轮盲测（每臂 50 张，按等级降序；框色 4=青 3=绿 2=黄 1=橙 0=红）",
            fill=(40, 40, 40), font=fb)

    print("%-10s %4s %6s %8s %8s %8s   %s" % ("臂", "n", "均值", "≥2", "≥3", "=4", "分布 0/1/2/3/4"))
    for ri, (key, label, _) in enumerate(ARMS):
        items = [r for r in rows if r["model"] == key]
        items.sort(key=lambda r: (-r["grade"], r.get("sample") or 0))
        g = [r["grade"] for r in items]
        hist = [sum(1 for x in g if x == k) for k in range(5)]
        print("%-10s %4d %6.2f %7.0f%% %7.0f%% %8d   %s"
              % (key, len(g), sum(g) / len(g),
                 100 * sum(1 for x in g if x >= 2) / len(g),
                 100 * sum(1 for x in g if x >= 3) / len(g), hist[4], hist))
        y = pad + ri * (face + 16 + gap)
        dr.text((gap + 2, y + 6), label, fill=(30, 30, 30), font=f)
        dr.text((gap + 2, y + 26), "n=%d 均值 %.2f" % (len(g), sum(g) / len(g)),
                fill=(110, 110, 110), font=f)
        dr.text((gap + 2, y + 44), "≥3 %.0f%%  ≥2 %.0f%%"
                % (100 * sum(1 for x in g if x >= 3) / len(g),
                   100 * sum(1 for x in g if x >= 2) / len(g)),
                fill=(110, 110, 110), font=f)
        for ci, r in enumerate(items[:per]):
            x0 = labw + ci * (face + gap)
            draw_face(dr, pools[key][r["sample"]], x0, y, cell, r["grade"])
            dr.text((x0 + 2, y + face + 1), str(r["grade"]), fill=(90, 90, 90), font=f)
        dr.line([(0, y + face + 14), (W, y + face + 14)], fill=(225, 220, 210))

    out1 = os.path.join(LDIR, "round12_graded.png")
    cv.save(out1)
    print("\n图：%s" % os.path.relpath(out1, P4))

    # ---- 图 2：只看被判 <2 的（复核反直觉的那个数）
    bad = [r for r in rows if r["grade"] <= 1]
    byarm = Counter(r["model"] for r in bad)
    print("被判 <2 的：", dict(byarm), " 合计", len(bad))
    ncol = max(1, min(12, len(bad)))
    cell = 46
    face = 8 * cell
    gap = 6
    nrow = (len(bad) + ncol - 1) // ncol
    W2 = gap + ncol * (face + gap) + 210
    H2 = 34 + nrow * (face + 18 + gap)
    cv2 = Image.new("RGB", (W2, H2), (250, 248, 242))
    dr2 = ImageDraw.Draw(cv2)
    dr2.text((gap, 8), "本轮被判 0/1 的全部 %d 张（左上角是分数；用来复核「真实数据 58%%」这个反直觉的数）"
             % len(bad), fill=(40, 40, 40), font=fb)
    bad.sort(key=lambda r: (r["model"], r["grade"]))
    for i, r in enumerate(bad):
        rr, cc = i // ncol, i % ncol
        x0 = gap + cc * (face + gap)
        y0 = 34 + rr * (face + 18 + gap)
        draw_face(dr2, pools[r["model"]][r["sample"]], x0, y0, cell, r["grade"])
        dr2.text((x0 + 2, y0 + 2), str(r["grade"]), fill=(0, 0, 0), font=fb)
        dr2.text((x0 + 2, y0 + face + 2), r["model"], fill=(90, 90, 90), font=f)
    out2 = os.path.join(LDIR, "round12_badtail.png")
    cv2.save(out2)
    print("图：%s" % os.path.relpath(out2, P4))

    # ---- 同一张脸被评两次的一致性（标签噪声）
    pair = defaultdict(list)
    for r in rows:
        pair[(r["model"], r.get("sample"))].append(r["grade"])
    rep = {k: v for k, v in pair.items() if len(v) > 1}
    if rep:
        same = sum(1 for v in rep.values() if len(set(v)) == 1)
        print("重复评到的样本 %d 个，其中两次完全同分 %d 个（%.0f%%），最大分差 %d"
              % (len(rep), same, 100 * same / len(rep),
                 max(abs(v[0] - v[-1]) for v in rep.values())))


if __name__ == "__main__":
    main()

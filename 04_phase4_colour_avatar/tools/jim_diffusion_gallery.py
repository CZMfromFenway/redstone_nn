#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
把 jim_diffusion_run.js 的产物画成对照图。

  python jim_diffusion_gallery.py --in results/diffusion/smoke.json --out results/diffusion/smoke.png
  python jim_diffusion_gallery.py --in ... --per 16 --cell 48

每个样本两格：左=吸附前的连续色，右=吸附到 234 色 + 对称吸附后的成品。
两格都按行内镜像还原成 8x8（JSON 里存的是 64 个 hex，已经是完整 8x8）。
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)


def font(sz):
    for fn in ("msyh.ttc", "simhei.ttf"):
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
                      x0 + (c + 1) * cell - 1, y0 + (r + 1) * cell - 1],
                     fill=hex2rgb(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--per", type=int, default=12, help="每行几张样本")
    ap.add_argument("--cell", type=int, default=40, help="每像素放大倍数")
    a = ap.parse_args()

    j = json.load(open(os.path.join(P4, a.inp) if not os.path.isabs(a.inp) else a.inp,
                       encoding="utf-8"))
    rows = j["samples"]
    per = max(1, min(a.per, len(rows)))
    nrow = (len(rows) + per - 1) // per
    cell = a.cell
    face = 8 * cell
    gap = 6
    pad = 28
    W = per * (2 * face + 3 * gap) + gap + 200
    H = pad + nrow * (face + gap + 18) + gap
    cv = Image.new("RGB", (W, H), (250, 248, 242))
    dr = ImageDraw.Draw(cv)
    f = font(13)
    f2 = font(12)

    dr.text((gap, 6), "%s   参数 steps=%s w=%s qb=%s alpha=%s k=%s tau=%s   平均 %.0f ms/张"
            % (os.path.basename(a.inp), j["steps"], j["cfg_w"], j["qb"], j["alpha"],
               j["k"], j["sym_tau"], j["ms_mean"]), fill=(40, 40, 40), font=f)

    for i, s in enumerate(rows):
        r, c = i // per, i % per
        x0 = gap + c * (2 * face + 3 * gap)
        y0 = pad + r * (face + gap + 18)
        draw_face(dr, s["raw"], x0, y0, cell)
        dr.rectangle([x0 - 1, y0 - 1, x0 + face, y0 + face], outline=(190, 185, 175))
        x1 = x0 + face + gap
        draw_face(dr, s["snapped"], x1, y0, cell)
        dr.rectangle([x1 - 1, y0 - 1, x1 + face, y0 + face], outline=(190, 185, 175))
        dr.text((x0, y0 + face + 2), "seed %d  sym %.2f  %d 色" % (s["seed"], s["sym"], s["ncol"]),
                fill=(90, 90, 90), font=f2)

    # 右侧图例：这两列是什么
    lx = gap + per * (2 * face + 3 * gap) + 4
    dr.text((lx, pad), "左：吸附前", fill=(60, 60, 60), font=f)
    dr.text((lx, pad + 18), "（连续 RGB）", fill=(120, 120, 120), font=f2)
    dr.text((lx, pad + 44), "右：吸附后", fill=(60, 60, 60), font=f)
    dr.text((lx, pad + 62), "（234 色 +", fill=(120, 120, 120), font=f2)
    dr.text((lx, pad + 78), "  对称吸附）", fill=(120, 120, 120), font=f2)

    out = a.out or os.path.join(P4, "results", "diffusion",
                                os.path.basename(a.inp).replace(".json", ".png"))
    out = os.path.join(P4, out) if not os.path.isabs(out) else out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv.save(out)
    print("图：%s（%d 张样本，%d 行）" % (os.path.relpath(out, P4), len(rows), nrow))

    # ---- 客观代理指标（口径与 tools/grade_gallery.py 的 feats() 一致，只作筛臂用，不作结论）
    st = stats([s["snapped"] for s in rows])
    print("代理指标（n=%d，口径同 grade_gallery.feats）：色数 %.2f  饱和 %.3f  边缘p90 %.2f  亮度 %.2f  眼带中位 %s px"
          % (len(rows), st["ncol"], st["sat"], st["edge"], st["lum"], st["eye"]))
    js = out.replace(".png", "_stats.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump({"n": len(rows), **st}, f, ensure_ascii=False, indent=1)
    print("代理指标 JSON：%s" % os.path.relpath(js, P4))


def stats(faces):
    """faces = [[64 个 hex], ...]；返回与仓库既有口径可比的四个量 + 眼带连续度。"""
    import math
    ncol, sat, lum, edges, eyes = [], [], [], [], []
    for hs in faces:
        rgb = [hex2rgb(h) for h in hs]
        ncol.append(len(set(hs)))
        s_list, l_list = [], []
        for r, g, b in rgb:
            mx, mn = max(r, g, b), min(r, g, b)
            s_list.append(0.0 if mx <= 1e-6 else (mx - mn) / mx)
            l_list.append((0.299 * r + 0.587 * g + 0.114 * b) / 255.0)
        sat.append(sum(s_list) / 64)
        lum.append(sum(l_list) / 64)
        # 边缘：4 邻域 RGB 梯度，取 p90
        g = []
        for r in range(8):
            for c in range(8):
                a = rgb[r * 8 + c]
                for dr, dc in ((0, 1), (1, 0)):
                    rr, cc = r + dr, c + dc
                    if rr > 7 or cc > 7:
                        continue
                    b2 = rgb[rr * 8 + cc]
                    g.append(math.dist(a, b2))
        g.sort()
        edges.append(g[int(0.9 * (len(g) - 1))])
        # 眼带连续度：第 2~4 行、第 1~6 列，最长连续「亮度 < 带内中位 − 0.16」
        vals = sorted(l_list[r * 8 + c] for r in range(2, 7) for c in range(1, 7))
        base = vals[len(vals) // 2]
        best = 0
        for r in range(2, 5):
            run = 0
            for c in range(1, 7):
                if l_list[r * 8 + c] < base - 0.16:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
        eyes.append(best)

    def med(v):
        v = sorted(v)
        return v[len(v) // 2]

    return {
        "ncol": round(sum(ncol) / len(ncol), 2),
        "sat": round(sum(sat) / len(sat), 3),
        "edge": round(sum(edges) / len(edges), 2),
        "lum": round(sum(lum) / len(lum), 3),
        "eye": med(eyes),
        "eye_hist": {k: eyes.count(k) for k in sorted(set(eyes))},
    }



if __name__ == "__main__":
    main()

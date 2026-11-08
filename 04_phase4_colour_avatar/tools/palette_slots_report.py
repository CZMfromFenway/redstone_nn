#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
把「调色板配置」摊开来看：

  A. 61 个 HIGH 地图色的全景（色块 + 明度 + 饱和 + 色相 + 名称）
  B. 原 16 槽 × 每槽的替换候选（|ΔL|≤1 的"保亮度"档；另附 |ΔL|≤4.5 的红发档）
     每槽同时给出它在 4096 点地图里的「使用频率」——用来判断放宽哪些槽位代价最小

产物：
    results/palette_shift/high_colours.png
    results/palette_shift/slot_candidates.png
    results/palette_shift/slot_table.json
用法：python palette_slots_report.py
"""
import colorsys
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
P2 = os.path.join(REPO, "02_phase2_avatar")
OUT = os.path.join(P4, "results", "palette_shift")
os.makedirs(OUT, exist_ok=True)


def font(sz, bold=False):
    for fn in (("msyhbd.ttc" if bold else "msyh.ttc"), "simhei.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def hsv(c):
    r, g, b = np.asarray(c, float) / 15.0
    mx, mn = max(r, g, b), min(r, g, b)
    L = (0.299 * r + 0.587 * g + 0.114 * b) * 15
    s = 0.0 if mx <= 1e-9 else (mx - mn) / mx
    h = 0.0 if mx - mn <= 1e-9 else colorsys.rgb_to_hsv(r, g, b)[0] * 360
    return L, s, h


def is_skin(L, s, h):
    return (8 <= h <= 55) and s > 0.15 and 3.5 <= L <= 13.5


def is_extreme(L):
    return L < 2.5 or L > 13.0


def is_red(h, s):
    return s >= 0.40 and (h >= 330 or h <= 25)


def swatch(dr, x, y, w, h, c):
    v = np.clip(np.rint(np.asarray(c, float) * 17), 0, 255).astype(int)
    dr.rectangle([x, y, x + w - 1, y + h - 1], fill=tuple(v), outline=(60, 68, 82))


def main():
    dp = json.load(open(os.path.join(P4, "results", "fit", "deploy_palette.json"),
                        encoding="utf-8"))["K16_w128_k8"]
    pal = np.array(dp["palette"], float)
    names = dp["names"]
    mc = json.load(open(os.path.join(P2, "data", "mc_map_colors.json"), encoding="utf-8"))
    base = np.array([b["rgb"] for b in mc["bases"]], float)
    bn = [b["name"] for b in mc["bases"]]
    mods = mc["_modifiers"]

    # 全部 244 个候选
    cand = []
    for m in ("HIGH", "NORMAL", "LOW", "LOWEST"):
        cc = np.clip(base * mods[m], 0, 255) / 17.0
        for i in range(len(base)):
            L, s, h = hsv(cc[i])
            cand.append({"rgb": np.round(cc[i], 3).tolist(), "name": bn[i], "mod": m,
                         "L": round(L, 2), "s": round(s, 3), "h": round(h, 1)})

    # HIGH 全景
    high = [dict(c) for c in cand if c["mod"] == "HIGH"]
    high.sort(key=lambda c: c["L"])

    # 每槽使用频率（取 4096 点地图的 argmax 结果）
    mp = json.load(open(os.path.join(P4, "results", "pca4map", "score.json"),
                        encoding="utf-8"))
    freq = None
    fpath = os.path.join(P4, "results", "pca4map", "slot_usage.json")
    if os.path.exists(fpath):
        raw_freq = json.load(open(fpath, encoding="utf-8"))
        freq = raw_freq.get("deploy", {}).get("s", raw_freq)

    # ---- A. HIGH 全景
    COLS = 16
    CW, CH, GAP = 54, 54, 8
    rows = (len(high) + COLS - 1) // COLS
    W = 20 + COLS * (CW + GAP)
    H = 92 + rows * (CH + 46)
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((16, 12), f"地图色 HIGH 档全景（{len(high)} 个）：颜色多样性池", font=font(19),
            fill=(234, 238, 244))
    dr.text((16, 40), "按明度 L 排序；每格上方标 L，下方是名称与 饱和/色相。这些就是「换色」时"
                      "可用的全部基色。", font=font(11), fill=(148, 160, 176))
    y = 70
    for i, c in enumerate(high):
        r_, c_ = divmod(i, COLS)
        x = 20 + c_ * (CW + GAP)
        yy = y + r_ * (CH + 46)
        swatch(dr, x, yy, CW, CH, c["rgb"])
        dr.text((x, yy - 13), f"L{c['L']:.0f}", font=font(9), fill=(190, 200, 214))
        nm = c["name"].replace("TERRACOTTA_", "T_")
        dr.text((x, yy + CH + 1), nm[:11], font=font(8), fill=(200, 210, 226))
        dr.text((x, yy + CH + 13), f"{c['s']:.2f}/{c['h']:.0f}°", font=font(8),
                fill=(140, 152, 168))
    p = os.path.join(OUT, "high_colours.png")
    cv.save(p)
    print(f"wrote {os.path.relpath(p, REPO)}")

    # ---- B. 槽位 × 候选
    PER = 8
    SW, GAP2, LW = 46, 5, 300
    nrow = 16
    W2 = LW + 2 * (PER * (SW + GAP2) + 24) + 20
    H2 = 108 + nrow * (SW + 30)
    cv = Image.new("RGB", (W2, H2), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((14, 12), "原调色板每个槽位的替换候选（左半：|ΔL|≤1 保亮度档；右半：|ΔL|≤4.5 红发档）",
            font=font(18), fill=(234, 238, 244))
    dr.text((14, 38), "每行左端是原槽位（含 L / 饱和 / 色相 / 使用频率）；候选按 |ΔL| 从小到大。"
                      "肤色与极端明度的槽位不参与替换。", font=font(10), fill=(148, 160, 176))
    dr.text((14, 54), "「用 x%」= 部署分布(N(0,1)截断 + 4选1，2000 张)下该槽位的像素占比 —— 这才是换色时真正"
                      "动到多少面积；占比越低的槽位，换掉它越看不出、也越白换。", font=font(10), fill=(130, 142, 158))
    x_left = LW
    x_mid = LW + PER * (SW + GAP2) + 26
    dr.text((x_left, 74), "保亮度替换 |ΔL|≤1", font=font(10), fill=(112, 170, 254))
    dr.text((x_mid, 74), "红发档 |ΔL|≤4.5（仅红系）", font=font(10), fill=(228, 140, 140))

    table = []
    y = 92
    for i, c in enumerate(pal):
        L0, s0, h0 = hsv(c)
        kind = ("extreme" if is_extreme(L0) else
                "skin" if is_skin(L0, s0, h0) else "backbone")
        use = None
        if freq:
            use = freq.get(str(i))
        dr.text((14, y + 4), f"#{i} {names[i].replace('TERRACOTTA_', 'T_')[:22]}", font=font(10),
                fill=(226, 232, 242))
        dr.text((14, y + 18), f"L{L0:.1f} s{s0:.2f} h{h0:.0f}°", font=font(9),
                fill=(150, 162, 178))
        if use is not None:
            dr.text((14, y + 30), f"用 {use*100:.1f}%", font=font(9), fill=(150, 190, 160))
        dr.text((14, y + 42), kind, font=font(8),
                fill={"extreme": (200, 120, 120), "skin": (210, 180, 120)}.get(kind, (120, 190, 140)))
        swatch(dr, 250, y, SW, SW, c)

        def same(o):        # 排掉"候选就是它自己"
            return float(np.linalg.norm(np.asarray(o["rgb"]) - np.asarray(c))) < 0.6
        tight = [o for o in cand
                 if abs(o["L"] - L0) <= 1.0 and not is_skin(o["L"], o["s"], o["h"])
                 and not is_extreme(o["L"]) and not same(o)]
        tight.sort(key=lambda o: (abs(o["L"] - L0), -o["s"]))
        reds = [o for o in cand if abs(o["L"] - L0) <= 4.5 and is_red(o["h"], o["s"])
                and not same(o)]
        reds.sort(key=lambda o: abs(o["L"] - L0))
        if kind != "backbone":
            tight, reds = [], []
            dr.text((x_left + 4, y + 16), "（该槽位不参与替换）", font=font(9),
                    fill=(140, 150, 165))
        for j, o in enumerate(tight[:PER]):
            x = x_left + j * (SW + GAP2)
            swatch(dr, x, y, SW, SW, o["rgb"])
            dr.text((x, y + SW + 1), f"{o['L'] - L0:+.1f}", font=font(8), fill=(190, 200, 214))
            dr.text((x, y + SW + 12), o["name"].replace("TERRACOTTA_", "T_")[:9], font=font(7),
                    fill=(150, 162, 178))
            dr.text((x, y + SW + 22), f"/{o['mod'][:3].lower()}", font=font(7),
                    fill=(130, 142, 158))
        for j, o in enumerate(reds[:PER]):
            x = x_mid + j * (SW + GAP2)
            swatch(dr, x, y, SW, SW, o["rgb"])
            dr.text((x, y + SW + 1), f"{o['L'] - L0:+.1f}", font=font(8), fill=(220, 160, 160))
            dr.text((x, y + SW + 12), o["name"].replace("TERRACOTTA_", "T_")[:9], font=font(7),
                    fill=(150, 162, 178))
            dr.text((x, y + SW + 22), f"/{o['mod'][:3].lower()}", font=font(7),
                    fill=(130, 142, 158))
        table.append({"slot": i, "name": names[i], "L": round(L0, 2), "sat": round(s0, 3),
                      "hue": round(h0, 1), "kind": kind, "usage": use,
                      "tight": tight[:PER], "red": reds[:6]})
        y += SW + 30
    p = os.path.join(OUT, "slot_candidates.png")
    cv.save(p)
    print(f"wrote {os.path.relpath(p, REPO)}")
    json.dump({"high": high, "slots": table,
               "n_candidates_total": len(cand)},
              open(os.path.join(OUT, "slot_table.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("wrote results/palette_shift/slot_table.json")

    # 控制台也打一份精简表
    print("\n槽位 → 保亮度候选（前 5）")
    for row in table:
        if row["kind"] != "backbone":
            print(f"  #{row['slot']:2d} {row['name']:24s} [{row['kind']}] 不参与")
            continue
        tops = ", ".join(f"{o['name'][:10]}/{o['mod'][:3]}({o['L'] - row['L']:+.1f})"
                         for o in row["tight"][:5])
        print(f"  #{row['slot']:2d} {row['name']:24s} L{row['L']:5.1f} 用"
              f"{(row['usage'] or 0)*100:4.1f}%  → {tops}")


if __name__ == "__main__":
    main()

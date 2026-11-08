#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
「保亮度换色相」的调色板扰动（作者 2026-11-02 提出的方案）。

做法：把 16 色调色板里的槽位分三类——
    肤色系（橙/棕那一段的色相）
    极暗 / 极亮（明度过小或过大）
    中流砥柱（剩下的鲜艳浅色、灰色、蓝绿紫）
然后**只动第三类**：把每个槽位按概率 p 换成「明度相近（|ΔL| ≤ tol）、色相或饱和度明显不同」的地图色。
因为明度保持，灰度结构（= 模型的形状）不变，理论上"好脸不会变丑"，但整批的颜色更花。

产物：
    results/palette_shift/palettes.json      每个变体的逐槽映射与 |ΔL|
    results/palette_shift/shift.png          原调色板 vs 各变体（含明度标注）
    results/twostage/pal16_s{30,60,100}.json 带新调色板的模型（权重不变）
    results/palette_shift/diversity.json     多样性指标（色数、色相熵）
用法：python palette_shift.py
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
LEVELS = ["0", "0", "0", "0"]          # 占位，避免误用


def font(sz):
    for fn in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        p = os.path.join("C:\\Windows\\Fonts", fn)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                pass
    return ImageFont.load_default()


def hsv(c):
    r, g, b = (np.asarray(c, float) / 15.0)
    mx, mn = max(r, g, b), min(r, g, b)
    L = 0.299 * r + 0.587 * g + 0.114 * b
    s = 0.0 if mx <= 1e-9 else (mx - mn) / mx
    h = 0.0 if mx - mn <= 1e-9 else colorsys.rgb_to_hsv(r, g, b)[0] * 360
    return L * 15, s, h


def hue_dist(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def is_skin(L, s, h):
    return (8 <= h <= 55) and s > 0.15 and 3.5 <= L <= 13.5


def is_extreme(L):
    return L < 2.5 or L > 13.0


def classify(c):
    L, s, h = hsv(c)
    if is_extreme(L):
        return "extreme", L, s, h
    if is_skin(L, s, h):
        return "skin", L, s, h
    return "backbone", L, s, h


def main():
    dp = json.load(open(os.path.join(P4, "results", "fit", "deploy_palette.json"),
                        encoding="utf-8"))["K16_w128_k8"]
    pal = np.array(dp["palette"], float)
    names = dp["names"]
    mc = json.load(open(os.path.join(P2, "data", "mc_map_colors.json"), encoding="utf-8"))
    base = np.array([b["rgb"] for b in mc["bases"]], float)
    bnames = [b["name"] for b in mc["bases"]]
    cands = []
    for m in ("HIGH", "NORMAL", "LOW", "LOWEST"):
        cc = np.clip(base * mc["_modifiers"][m], 0, 255) / 17.0
        for i in range(len(base)):
            cands.append((cc[i], bnames[i], m))
    print(f"候选 {len(cands)} 个地图色（61 基色 × 4 修饰）")

    # ---- 分类
    cls = []
    for i, c in enumerate(pal):
        kind, L, s, h = classify(c)
        cls.append({"i": i, "name": names[i], "rgb": np.round(c, 3).tolist(),
                    "L": round(L, 2), "sat": round(s, 3), "hue": round(h, 1), "kind": kind})
    print("\n槽位分类：")
    for k in cls:
        print(f"  #{k['i']:2d} {k['name']:22s} L={k['L']:5.2f} sat={k['sat']:.2f} "
              f"hue={k['hue']:5.0f} → {k['kind']}")

    # ---- 每个 backbone 槽位的候选替换（保明度、换色相/饱和度、不要肤色）
    tol = 1.0
    repl = {}
    for k in cls:
        if k["kind"] != "backbone":
            continue
        L0, s0, h0 = k["L"], k["sat"], k["hue"]
        opts = []
        for c, nm, mod in cands:
            L, s, h = hsv(c)
            if abs(L - L0) > tol:
                continue
            if is_skin(L, s, h) or is_extreme(L):
                continue
            dh = hue_dist(h, h0)
            ds = abs(s - s0)
            # 目的：换色相/饱和度、但不要洗成灰。优先「色相明显不同且还有颜色」，
            # 或「饱和度差得很多且色相也动了」。纯灰（sat=0）只在原色本身也接近灰时才收。
            vivid = s >= 0.30
            if not ((dh >= 45 and vivid) or (dh >= 20 and ds >= 0.30 and s >= 0.15)
                    or (s0 < 0.10 and s >= 0.25)):
                continue
            opts.append({"rgb": np.round(c, 3).tolist(), "name": nm, "mod": mod,
                         "L": round(L, 2), "sat": round(s, 3), "hue": round(h, 1),
                         "dL": round(L - L0, 2), "dhue": round(dh, 0), "dsat": round(s - s0, 2)})
        opts.sort(key=lambda o: (abs(o["dL"]), -o["sat"], -o["dhue"]))
        repl[k["i"]] = opts
        print(f"\n  #{k['i']:2d} {k['name']:22s} 有 {len(opts)} 个候选（|ΔL|≤{tol}）")
        for o in opts[:4]:
            print(f"       {o['name']:22s}{o['mod']:7s} L={o['L']:5.2f} "
                  f"sat={o['sat']:.2f} hue={o['hue']:5.0f} ΔL={o['dL']:+.2f} Δhue={o['dhue']:.0f}")

    # ---- 生成变体（概率替换，同调色板内不重复）
    rng = np.random.default_rng(2026)
    variants = {}
    for p in (0.3, 0.6, 1.0):
        newpal = pal.copy()
        used = {tuple(np.round(c, 2)) for i, c in enumerate(pal) if cls[i]["kind"] != "backbone"}
        mapping = []
        for k in cls:
            i = k["i"]
            if k["kind"] != "backbone":
                mapping.append({"slot": i, "from": k["name"], "to": None, "reason": k["kind"]})
                continue
            if rng.random() > p or not repl[i]:
                mapping.append({"slot": i, "from": k["name"], "to": None, "reason": "未抽中"})
                continue
            pool = [o for o in repl[i] if tuple(np.round(o["rgb"], 2)) not in used]
            if not pool:
                mapping.append({"slot": i, "from": k["name"], "to": None, "reason": "无可用候选"})
                continue
            o = pool[int(rng.integers(0, min(len(pool), 12)))]
            newpal[i] = o["rgb"]
            used.add(tuple(np.round(o["rgb"], 2)))
            mapping.append({"slot": i, "from": k["name"], "to": f"{o['name']}/{o['mod']}",
                            "L": o["L"], "dL": o["dL"], "dhue": o["dhue"], "reason": "换"})
        variants[f"p{int(p*100)}"] = {"palette": np.round(newpal, 3).tolist(),
                                      "p": p, "mapping": mapping}
        n_swapped = sum(1 for m in mapping if m["to"])
        dl = [abs(m["dL"]) for m in mapping if m["to"]]
        print(f"\n变体 p={p}: 换了 {n_swapped} / {len(cls)} 个槽位；"
              f"平均 |ΔL| = {np.mean(dl):.2f} 级" if dl else f"\n变体 p={p}: 没有换")
        # 写模型文件（权重不变，只换调色板）
        src = os.path.join(P4, "results", "twostage", "pal16.json")
        mdl = json.load(open(src, encoding="utf-8"))
        mdl["palette"] = [list(map(float, c)) for c in newpal]
        mdl["meta"] = dict(mdl.get("meta", {}), palette_scheme=f"shift_p{int(p*100)}",
                           note="16 色调色板按概率保亮度换色相（作者方案）")
        tag = f"s{int(p*100)}"
        json.dump(mdl, open(os.path.join(P4, "results", "twostage", f"pal16_{tag}.json"),
                            "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  → results/twostage/pal16_{tag}.json")

    json.dump({"tol": tol, "slots": cls, "candidates": {str(k): v[:40] for k, v in repl.items()},
               "variants": variants},
              open(os.path.join(OUT, "palettes.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ---- 图：原色板 vs 各变体
    TILE = 68
    rows = [("原调色板（16 槽 / 15 色）", pal, None)]
    for tag, v in variants.items():
        rows.append((f"变体 {tag}（换了 {sum(1 for m in v['mapping'] if m['to'])} 个槽）",
                     np.array(v["palette"]), v["mapping"]))
    W = 150 + 16 * (TILE + 8) + 20
    H = 92 + len(rows) * (TILE + 54)
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((14, 12), "保亮度换色相：调色板扰动（只动中流砥柱，肤色与极端明度不动）",
            font=font(19), fill=(234, 238, 244))
    dr.text((14, 40), "每格上方是槽位号与明度 L；换过的槽位标出了 ΔL 与 Δhue。"
                      "权重完全不变，只换调色板 → 形状不变、颜色变。", font=font(11),
            fill=(148, 160, 176))
    y = 74
    for label, p_, mapping in rows:
        dr.text((14, y + 18), label, font=font(11), fill=(112, 170, 254))
        for i, c in enumerate(p_):
            x = 150 + i * (TILE + 8)
            v = np.clip(np.rint(np.asarray(c) * 17), 0, 255).astype(int)
            dr.rectangle([x, y, x + TILE - 1, y + TILE - 1], fill=tuple(v))
            L, s, h = hsv(c)
            dr.text((x + 2, y + TILE + 2), f"#{i} L={L:.1f}", font=font(8), fill=(190, 200, 214))
            if mapping and mapping[i]["to"]:
                dr.text((x + 2, y + TILE + 14), f"ΔL{mapping[i]['dL']:+.1f}", font=font(8),
                        fill=(120, 200, 140))
                dr.text((x + 2, y + TILE + 26), f"Δh{mapping[i]['dhue']:.0f}", font=font(8),
                        fill=(120, 200, 140))
        y += TILE + 54
    p = os.path.join(OUT, "shift.png")
    cv.save(p)
    print(f"\nwrote {os.path.relpath(p, REPO)}")
    print("PALETTE_SHIFT_DONE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
为什么没有红头发 + 「允许重染色前后」的优质头像对照。

两件事：
  1. 造一个允许红发的变体 `red`：只对**亮槽位**（L ≥ 7）找候选，候选必须是红系
     （色相 ≥330° 或 ≤25°、饱和 ≥0.4），并且把 |ΔL| 放宽到 4.5 级
     ——因为地图色里 7 级以上几乎没有红色（见 shift 报告）。
  2. 把第六轮里 **pal_base 评为 ≥3 的同一批好脸**，在同一套旋钮下分别用
     base / s30 / s60 / s100 / red 五种调色板重绘，做成对照图。

产物：results/palette_shift/ba_good.png、red_stats.json、../twostage/pal16_red.json
用法：python palette_before_after.py
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
LDIR = os.path.join(P4, "results", "labels")
ROW = np.array([q % 8 for q in range(32)])
COL = np.array([q // 8 for q in range(32)])


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
    r, g, b = np.asarray(c, float) / 15.0
    mx, mn = max(r, g, b), min(r, g, b)
    L = (0.299 * r + 0.587 * g + 0.114 * b) * 15
    s = 0.0 if mx <= 1e-9 else (mx - mn) / mx
    h = 0.0 if mx - mn <= 1e-9 else colorsys.rgb_to_hsv(r, g, b)[0] * 360
    return L, s, h


def is_red(h, s):
    return s >= 0.40 and (h >= 330 or h <= 25)


def is_skin(L, s, h):
    return (8 <= h <= 55) and s > 0.15 and 3.5 <= L <= 13.5


def main():
    dp = json.load(open(os.path.join(P4, "results", "fit", "deploy_palette.json"),
                        encoding="utf-8"))["K16_w128_k8"]
    pal = np.array(dp["palette"], float)
    names = dp["names"]
    mc = json.load(open(os.path.join(P2, "data", "mc_map_colors.json"), encoding="utf-8"))
    base = np.array([b["rgb"] for b in mc["bases"]], float)
    bn = [b["name"] for b in mc["bases"]]
    cand = []
    for m in ("HIGH", "NORMAL", "LOW", "LOWEST"):
        cc = np.clip(base * mc["_modifiers"][m], 0, 255) / 17.0
        for i in range(len(base)):
            L, s, h = hsv(cc[i])
            cand.append({"rgb": cc[i], "name": bn[i], "mod": m, "L": L, "s": s, "h": h})

    # ---- 红发变体：只动亮槽位（L>=7），只换成红系，|ΔL| 放宽到 4.5
    rng = np.random.default_rng(11)
    newpal = pal.copy()
    mapping = []
    used = set()
    for i, c in enumerate(pal):
        L0, s0, h0 = hsv(c)
        if L0 < 7:
            mapping.append({"slot": i, "from": names[i], "to": None, "reason": "不够亮"})
            continue
        if is_skin(L0, s0, h0):
            mapping.append({"slot": i, "from": names[i], "to": None, "reason": "肤色不动"})
            continue
        if is_red(h0, s0):
            mapping.append({"slot": i, "from": names[i], "to": None, "reason": "本来就是红系"})
            continue
        opts = [o for o in cand if abs(o["L"] - L0) <= 4.5 and is_red(o["h"], o["s"])]
        opts.sort(key=lambda o: abs(o["L"] - L0))
        opts = [o for o in opts if tuple(np.round(o["rgb"], 2)) not in used][:6]
        if not opts:
            mapping.append({"slot": i, "from": names[i], "to": None, "reason": "无红系候选"})
            continue
        o = opts[int(rng.integers(0, len(opts)))]
        newpal[i] = o["rgb"]
        used.add(tuple(np.round(o["rgb"], 2)))
        mapping.append({"slot": i, "from": names[i], "to": f"{o['name']}/{o['mod']}",
                        "L": round(o["L"], 2), "dL": round(o["L"] - L0, 2)})
    n_sw = sum(1 for m in mapping if m["to"])
    print(f"红发变体：换了 {n_sw} 个槽位（只动 L≥7 的）")
    for m in mapping:
        if m["to"]:
            print(f"  #{m['slot']:2d} {m['from']:22s} → {m['to']:24s} L={m['L']:.1f} ΔL={m['dL']:+.1f}")
    mdl = json.load(open(os.path.join(P4, "results", "twostage", "pal16.json"), encoding="utf-8"))
    mdl["palette"] = [[float(v) for v in c] for c in np.round(newpal, 3)]
    mdl["meta"] = dict(mdl.get("meta", {}), palette_scheme="red_allowed",
                       note="允许红发：亮槽位只换红系，|ΔL|≤4.5")
    json.dump(mdl, open(os.path.join(P4, "results", "twostage", "pal16_red.json"), "w",
                        encoding="utf-8"), ensure_ascii=False)

    # ---- 模型/前向
    g = json.load(open(os.path.join(P4, "results", "twostage", "gray_pca4.json"),
                       encoding="utf-8"))
    GL = [(np.array(L["W"]), np.array(L["b"])) for L in g["layers"]]
    sc = json.load(open(os.path.join(LDIR, "scorers_round5.json"),
                        encoding="utf-8"))["families"]["two_stage"]
    sw = np.array(sc["weights"], float)

    def score(z):
        z = list(z)
        f = [1.0] + z + [v * v for v in z] + [z[i] * z[j] for i in range(4) for j in range(i + 1, 4)]
        return float(np.dot(sw, f))

    def fwd(layers, x):
        h = np.asarray(x, float)
        for i, (W, b) in enumerate(layers):
            h = h @ W.T + b
            if i < len(layers) - 1:
                h = np.maximum(h, 0)
        return h

    def palettes(tag):
        p = json.load(open(os.path.join(P4, "results", "twostage", f"pal16{tag}.json"),
                           encoding="utf-8"))
        pal_ = np.array(p["palette"], float)
        lay = [(np.array(L["W"]), np.array(L["b"])) for L in p["body"]]
        lay.append((np.array(p["head"]["W"]), np.array(p["head"]["b"])))
        return pal_, lay

    def idx_of(lay, Z, K):
        raw = fwd(GL, Z)
        lo = raw.min(1, keepdims=True); hi = raw.max(1, keepdims=True)
        lv = np.clip(np.rint((raw - lo) / np.maximum(hi - lo, 1e-6) * 15), 0, 15)
        lg = fwd(lay, lv / 15.0).reshape(len(Z), 32, K)
        return lg.argmax(-1)

    # ---- 取第六轮 pal_base 的优质头像（grade >= 3）
    rows = [json.loads(l) for l in open(os.path.join(LDIR, "colour_round6.jsonl"),
                                        encoding="utf-8") if l.strip()]
    good = [r for r in rows if r["model"] == "pal_base" and r["grade"] >= 3]
    good.sort(key=lambda r: -r["grade"])
    pick = good[:16]
    Z = np.array([(r.get("z") or r["pca"])[:4] for r in pick], dtype=float)
    print(f"\n第六轮 pal_base 里 ≥3 的好脸 {len(good)} 张，取前 {len(pick)} 张做对照")

    arms = [(("pal_base", ""), "原调色板"), (("pal_s30", "_s30"), "s30 换 3 槽"),
            (("pal_s60", "_s60"), "s60 换 4 槽"), (("pal_s100", "_s100"), "s100 换 8 槽"),
            (("pal_red", "_red"), "允许红发（仅亮槽位，|ΔL|≤4.5）")]
    TILE, PAD, LW = 52, 6, 210
    W = LW + 16 * (TILE + PAD) + 20
    H = 104 + len(arms) * (TILE + 34)
    cv = Image.new("RGB", (W, H), (16, 18, 22))
    dr = ImageDraw.Draw(cv)
    dr.text((14, 12), "允许随机重染色前后：同一批「好脸」在不同调色板下的样子", font=font(19),
            fill=(234, 238, 244))
    dr.text((14, 40), "脸与旋钮完全相同（它们是第六轮 pal_base 里评为 ≥3 的同一批），"
                      "只换调色板。最后一行是放宽明度约束、允许红发的变体。", font=font(11),
            fill=(148, 160, 176))
    stats = {}
    y = 78
    for (arm_name, tag), label in arms:
        pal_, lay = palettes(tag)
        K = len(pal_)
        idx = idx_of(lay, Z, K)
        rgb = pal_[idx]
        # 红系像素占比
        reds = 0
        for i in range(len(rgb)):
            for c in rgb[i]:
                L, s, h = hsv(c)
                if is_red(h, s):
                    reds += 1
        faces_with_red = sum(
            1 for i in range(len(rgb))
            if any(is_red(hsv(c)[2], hsv(c)[1]) for c in rgb[i]))
        ent = None
        mx = rgb.max(-1); mn = rgb.min(-1); ch = mx - mn
        hh = np.zeros_like(mx)
        r_, g_, b_ = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        m1 = (ch > 1e-6) & (mx == r_); hh[m1] = ((g_[m1] - b_[m1]) / ch[m1]) % 6
        m2 = (ch > 1e-6) & (mx == g_); hh[m2] = (b_[m2] - r_[m2]) / ch[m2] + 2
        m3 = (ch > 1e-6) & (mx == b_); hh[m3] = (r_[m3] - g_[m3]) / ch[m3] + 4
        bins = (hh[ch > 1.0] / 6 * 12).astype(int)
        q = np.bincount(bins, minlength=12) / max(1, len(bins))
        ent = float(-(q[q > 0] * np.log(q[q > 0])).sum())
        stats[arm_name] = {"red_pixels": int(reds),
                      "faces_with_red": int(faces_with_red),
                      "hue_entropy": round(ent, 3),
                      "palette": np.round(pal_, 3).tolist()}
        dr.text((14, y + 16), label, font=font(12), fill=(112, 170, 254))
        dr.text((14, y + 32), f"红像素 {reds} · 有红的脸 {faces_with_red}/16", font=font(9),
                fill=(200, 170, 120))
        dr.text((14, y + 44), f"色相熵 {ent:.2f}", font=font(9), fill=(150, 162, 178))
        for i in range(len(rgb)):
            img = np.zeros((8, 8, 3))
            img[ROW, COL] = rgb[i]
            img[ROW, 7 - COL] = rgb[i]
            v = np.clip(np.rint(img * 17), 0, 255).astype(np.uint8)
            t = np.repeat(np.repeat(v, TILE // 8, 0), TILE // 8, 1)
            cv.paste(Image.fromarray(t), (LW + i * (TILE + PAD), y))
        y += TILE + 34
    p = os.path.join(OUT, "ba_good.png")
    cv.save(p)
    json.dump({"n_good_in_round6": len(good), "n_shown": len(pick), "arms": stats,
               "red_mapping": mapping},
              open(os.path.join(OUT, "red_stats.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote {os.path.relpath(p, REPO)}")
    for k, v in stats.items():
        print(f"  {k:8s} 红像素 {v['red_pixels']:4d}  有红的脸 {v['faces_with_red']:2d}/16  "
              f"色相熵 {v['hue_entropy']:.2f}")


if __name__ == "__main__":
    main()

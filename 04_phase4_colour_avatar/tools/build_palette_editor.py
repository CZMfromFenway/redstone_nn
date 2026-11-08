#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
把调色板替换做成一个可编辑的单文件网页（拖拽 / 导出）。

嵌进去的：
    16 个槽位（原色、明度、饱和、色相、部署用量、分类）
    61 个 HIGH 地图色（候选池，可按明度/色相筛选）
    5 个预设方案（原色板 / s30 / s60 / s100 / 红发档）
    24 张"样本脸"的调色板索引（固定形状，换色板后即时重绘）

产物：04_phase4_colour_avatar/webapp/palette_editor.html
用法：python build_palette_editor.py
"""
import colorsys
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
P4 = os.path.dirname(HERE)
REPO = os.path.dirname(P4)
WEB = os.path.join(P4, "webapp")
OUT = os.path.join(P4, "results", "palette_shift")


def hsv(c):
    r, g, b = np.asarray(c, float) / 15.0
    mx, mn = max(r, g, b), min(r, g, b)
    L = (0.299 * r + 0.587 * g + 0.114 * b) * 15
    s = 0.0 if mx <= 1e-9 else (mx - mn) / mx
    h = 0.0 if mx - mn <= 1e-9 else colorsys.rgb_to_hsv(r, g, b)[0] * 360
    return L, s, h


def main():
    tab = json.load(open(os.path.join(OUT, "slot_table.json"), encoding="utf-8"))
    dp = json.load(open(os.path.join(P4, "results", "fit", "deploy_palette.json"),
                        encoding="utf-8"))["K16_w128_k8"]
    slots = []
    for r in tab["slots"]:
        i = r["slot"]
        slots.append({"i": i, "name": r["name"],
                      "rgb": [round(float(v), 3) for v in dp["palette"][i]],
                      "L": r["L"], "sat": r["sat"], "hue": r["hue"],
                      "use": r["usage"], "kind": r["kind"]})
    high = [{"name": h["name"], "rgb": h["rgb"], "L": h["L"], "s": h["s"], "h": h["h"]}
            for h in tab["high"]]

    base = [s["rgb"] for s in slots]

    # ---- 预设：与基准色板逐槽对比得出
    def preset_from(path, label):
        p = json.load(open(path, encoding="utf-8"))
        pal = p["palette"]
        swaps = []
        for i, (a, b) in enumerate(zip(base, pal)):
            if np.linalg.norm(np.asarray(a) - np.asarray(b)) > 0.6:
                L, s, h = hsv(b)
                nm = None
                for hh in high:
                    if np.linalg.norm(np.asarray(hh["rgb"]) - np.asarray(b)) < 0.4:
                        nm = hh["name"] + "/HIGH"
                        break
                swaps.append({"slot": i, "rgb": [round(float(v), 3) for v in b],
                              "name": nm or "地图色", "L": round(L, 2),
                              "sat": round(s, 3), "hue": round(h, 1),
                              "dL": round(L - slots[i]["L"], 2),
                              "dhue": round(min(abs(h - slots[i]["hue"]) % 360,
                                                360 - abs(h - slots[i]["hue"]) % 360), 0)})
        return {"label": label, "swaps": swaps}

    meta = json.load(open(os.path.join(OUT, "palettes.json"), encoding="utf-8"))
    presets = {"原色板": {"label": "原色板（不做替换）", "swaps": []}}
    for tag, fn in (("s30", "pal16_s30.json"), ("s60", "pal16_s60.json"),
                    ("s100", "pal16_s100.json"), ("红发档", "pal16_red.json")):
        presets[tag] = preset_from(os.path.join(P4, "results", "twostage", fn),
                                   tag)

    # ---- 样本脸：第六轮 pal_base 评为 >=3 的前 24 张，取索引
    g = json.load(open(os.path.join(P4, "results", "twostage", "gray_pca4.json"),
                       encoding="utf-8"))
    GL = [(np.array(L["W"]), np.array(L["b"])) for L in g["layers"]]
    p16 = json.load(open(os.path.join(P4, "results", "twostage", "pal16.json"),
                         encoding="utf-8"))
    LAY = ([(np.array(L["W"]), np.array(L["b"])) for L in p16["body"]]
           + [(np.array(p16["head"]["W"]), np.array(p16["head"]["b"]))])

    def fwd(layers, x):
        h = np.asarray(x, float)
        for i, (W, b) in enumerate(layers):
            h = h @ W.T + b
            if i < len(layers) - 1:
                h = np.maximum(h, 0)
        return h

    rows = [json.loads(l) for l in
            open(os.path.join(P4, "results", "labels", "colour_round6.jsonl"),
                 encoding="utf-8") if l.strip()]
    good = [r for r in rows if r["model"] == "pal_base" and r["grade"] >= 3]
    good.sort(key=lambda r: -r["grade"])
    seen, pick = set(), []
    for r in good:
        z = tuple(np.round(np.array((r.get("z") or r["pca"])[:4], float), 2))
        if z in seen:
            continue
        seen.add(z)
        pick.append((z, r["grade"]))
        if len(pick) == 24:
            break
    Z = np.array([z for z, _ in pick], float)
    raw = fwd(GL, Z)
    lo = raw.min(1, keepdims=True); hi = raw.max(1, keepdims=True)
    lv = np.clip(np.rint((raw - lo) / np.maximum(hi - lo, 1e-6) * 15), 0, 15)
    idx = fwd(LAY, lv / 15.0).reshape(len(Z), 32, len(base)).argmax(-1)
    faces = [{"idx": idx[k].tolist(), "grade": int(pick[k][1])} for k in range(len(Z))]

    # 每槽的“保亮度候选”前 4 个，供网页的「自动填满」
    prefill = {}
    for r in tab["slots"]:
        if r["kind"] != "backbone":
            continue
        lst = []
        for o in r["tight"][:4]:
            lst.append({"rgb": [round(float(v), 3) for v in o["rgb"]], "name": o["name"],
                        "mod": o["mod"], "L": o["L"], "sat": o["s"], "hue": o["h"],
                        "weight": 1})
        if lst:
            prefill[str(r["slot"])] = lst

    data = {"slots": slots, "high": high, "presets": presets, "faces": faces,
            "prefill": prefill,
            "cost": {"pixels_black_white": round(
                (slots[2]["use"] or 0) + (slots[15]["use"] or 0), 3)}}

    tpl = open(os.path.join(HERE, "palette_editor_template.html"), encoding="utf-8").read()
    out = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False,
                                                     separators=(",", ":")))
    os.makedirs(WEB, exist_ok=True)
    p = os.path.join(WEB, "palette_editor.html")
    open(p, "w", encoding="utf-8").write(out)
    print(f"wrote {os.path.relpath(p, REPO)}  ({os.path.getsize(p)/1024:.1f} KB)")
    print(f"  槽位 {len(slots)}；HIGH 候选 {len(high)}；样本脸 {len(faces)}；"
          f"预设 {list(presets)}")
    for k, v in presets.items():
        print(f"    {k:6s} 换 {len(v['swaps'])} 槽")


if __name__ == "__main__":
    main()
